"""
Value model for PPO training.

A lightweight value head (linear layer) that operates on hidden states from the
policy model's backbone. This avoids duplicating the full model — only the small
linear head(s) are trainable.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW


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
