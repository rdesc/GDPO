"""
Value model for PPO training.

Two modes are supported:

1. **Lightweight** (`ValueModelWrapper`): A small linear head that takes hidden
   states extracted from the *policy* backbone.  No extra backbone is loaded —
   only the head(s) are trainable.

2. **Separate backbone** (`ValueModel`): Loads its own full transformer backbone
   (frozen by default), with trainable value head(s) on top.  Optionally applies
   LoRA adapters to the backbone for richer value predictions.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoConfig

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class ValueHead(nn.Module):
    """Linear projection from hidden_size to scalar per-token value."""

    def __init__(self, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H)
        Returns:
            values: (B, T)
        """
        return self.linear(self.dropout(hidden_states)).squeeze(-1)


class ValueModelWrapper(nn.Module):
    """
    Manages a reward value head and optional cost value heads for PPO-Lagrangian.

    Parameters:
        hidden_size: Hidden dimension of the policy backbone.
        cost_head_names: List of constraint names for cost value heads. Empty for plain PPO.
        lr: Learning rate for the value head optimizer.
        dropout_prob: Dropout probability in the value heads.
    """

    def __init__(
        self,
        hidden_size: int,
        cost_head_names: list[str] | None = None,
        lr: float = 1e-4,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.reward_head = ValueHead(hidden_size, dropout_prob)
        self.cost_head_names = cost_head_names or []
        self.cost_heads = nn.ModuleDict(
            {name: ValueHead(hidden_size, dropout_prob) for name in self.cost_head_names}
        )
        self.lr = lr
        self.optimizer = None  # created after .to(device) or accelerator.prepare

    def create_optimizer(self):
        """Create the AdamW optimizer over all trainable parameters."""
        self.optimizer = AdamW(self.parameters(), lr=self.lr, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor):
        """
        Compute reward and cost values from hidden states.

        Args:
            hidden_states: (B, T, H) completion-only hidden states

        Returns:
            reward_values: (B, T)
            cost_values: dict[str, (B, T)] or empty dict
        """
        reward_values = self.reward_head(hidden_states)
        cost_values = {name: head(hidden_states) for name, head in self.cost_heads.items()}
        return reward_values, cost_values

    @staticmethod
    def compute_value_loss(predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        MSE value loss, normalized per sequence then averaged.

        Args:
            predicted: (B, T) predicted values
            targets: (B, T) target returns
            mask: (B, T) completion mask
        Returns:
            scalar loss
        """
        sq_error = (predicted - targets) ** 2
        per_seq_loss = (sq_error * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return per_seq_loss.mean()


class ValueModel(nn.Module):
    """
    Separate-backbone value model for PPO.

    Loads a full transformer backbone (frozen by default) and attaches trainable
    value head(s).  Optionally applies LoRA adapters to the backbone so that the
    value representation can diverge from the pretrained weights.

    Unlike ``ValueModelWrapper`` which receives pre-computed hidden states, this
    module takes raw ``input_ids`` / ``attention_mask`` and runs its own forward
    pass through the backbone.

    Parameters:
        model_id: HuggingFace model name or path (e.g. ``"Qwen/Qwen2.5-1.5B"``).
        model_init_kwargs: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``.
        lr: Learning rate for the optimizer (heads + optional LoRA params).
        dropout_prob: Dropout probability in value heads.
    """

    def __init__(
        self,
        model_id: str,
        model_init_kwargs: dict | None = None,
        lr: float = 1e-4,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        model_init_kwargs = model_init_kwargs or {}

        # Expose config so that DeepSpeed / accelerate can inspect the model.
        self.config = AutoConfig.from_pretrained(model_id)
        hidden_size = self.config.hidden_size

        # Load backbone and freeze all parameters
        full_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        # Keep only the transformer backbone (drop the LM head)
        # Works for Qwen2, LLaMA, Mistral, etc. where .model is the backbone
        self.backbone = full_model.model
        del full_model  # free the LM head memory

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable reward value head
        self.reward_head = ValueHead(hidden_size, dropout_prob)
        self.cost_head_names: list[str] = []
        self.cost_heads = nn.ModuleDict()

        self.lr = lr
        self.optimizer = None  # created after .to(device) or accelerator.prepare
        self._lora_applied = False

    def add_cost_heads(self, cost_names: list[str]):
        """Add separate value heads for each cost / constraint signal."""
        self.cost_head_names = list(cost_names)
        for name in cost_names:
            self.cost_heads[name] = ValueHead(self.config.hidden_size)

    def apply_lora(self, rank: int = 64, alpha: int = 128):
        """Apply LoRA adapters to backbone linear layers (requires peft)."""
        if not _PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA on the value backbone. "
                "Install it with: pip install peft"
            )
        if self._lora_applied:
            return

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self._lora_applied = True

    def create_optimizer(self):
        """Create the AdamW optimizer over all *trainable* parameters."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.lr, eps=1e-5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
        Full forward pass through backbone + heads.

        Args:
            input_ids: (B, T) token ids (prompt + completion concatenated)
            attention_mask: (B, T) attention mask

        Returns:
            reward_values: (B, T) per-token reward values
            cost_values: dict[str, (B, T)] per-token cost values (empty dict if no cost heads)
        """
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        reward_values = self.reward_head(hidden_states)
        cost_values = {name: head(hidden_states) for name, head in self.cost_heads.items()}
        return reward_values, cost_values

    def get_values_for_completion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
        batch_size: int | None = None,
        **kwargs,
    ):
        """
        Returns values only for completion tokens.

        Runs the backbone in sub-batches to manage memory, then slices out
        the completion portion (everything after ``prompt_length``).

        Args:
            input_ids: (B, T_full) full prompt+completion token ids
            attention_mask: (B, T_full) full attention mask
            prompt_length: number of prompt tokens to skip
            batch_size: optional sub-batch size for chunked forward

        Returns:
            reward_values: (B, T_completion) per-token reward values
            cost_values: dict[str, (B, T_completion)]
        """
        batch_size = batch_size or input_ids.size(0)
        all_reward = []
        all_cost: dict[str, list[torch.Tensor]] = {name: [] for name in self.cost_head_names}

        for i in range(0, input_ids.size(0), batch_size):
            ids_b = input_ids[i:i + batch_size]
            mask_b = attention_mask[i:i + batch_size]
            rv, cv = self.forward(ids_b, mask_b)
            # Slice to completion tokens only
            all_reward.append(rv[:, prompt_length:])
            for name in self.cost_head_names:
                all_cost[name].append(cv[name][:, prompt_length:])

        reward_values = torch.cat(all_reward, dim=0)
        cost_values = {name: torch.cat(chunks, dim=0) for name, chunks in all_cost.items()}
        return reward_values, cost_values

    # Reuse the same static loss method as ValueModelWrapper
    compute_value_loss = staticmethod(ValueModelWrapper.compute_value_loss)
