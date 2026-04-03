"""
PPO Trainer built on the GRPO generation/reward infrastructure.

Extends GRPOTrainer with:
  - A learned value head on top of the policy backbone for per-token value estimation
  - GAE (Generalized Advantage Estimation) replacing group-relative normalization
  - A value function loss (MSE on returns)
  - An optional value warmup phase before policy training begins
  - PPO-Lagrangian: separate cost value heads per constraint with Lagrangian multiplier updates
"""

import warnings
from typing import Optional, Union

import torch
import torch.nn.functional as F
from accelerate.utils import gather, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)

from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc, nanmin, nanmax
from .value_model import ValueModelWrapper, ValueModel
from ..models import prepare_deepspeed

try:
    from peft import PeftConfig
except ImportError:
    PeftConfig = None

# Prefix used to flatten cost_returns into the return dict so that
# shuffle_tensor_dict / split_tensor_dict (which expect flat tensor dicts)
# can handle them without crashing on nested dicts.
_COST_RETURNS_PREFIX = "cost_returns/"


class GRPOPPOTrainer(GRPOTrainer):
    """
    PPO trainer that reuses GRPO's generation and reward pipeline but replaces
    group-relative advantage estimation with GAE over a learned value function.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Let GRPOTrainer handle all the standard setup
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # PPO-specific config (read from args with defaults for backward compat)
        self.gae_gamma = getattr(args, "gae_gamma", 1.0)
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)
        self.value_loss_coef = getattr(args, "value_loss_coef", 0.5)
        self.value_warmup_steps = getattr(args, "value_warmup_steps", 0)
        self.separate_cost_values = getattr(args, "separate_cost_values", True)

        self.value_model_type = getattr(args, "value_model_type", "lightweight")
        cost_head_names = (
            self.constraint_names
            if self.use_constraints and self.separate_cost_values
            else []
        )

        if self.value_model_type == "separate":
            # Separate backbone value model
            value_model_id = getattr(args, "value_model_name_or_path", None)
            if value_model_id is None:
                # Fall back to the policy model name
                value_model_id = args.model_name_or_path if hasattr(args, "model_name_or_path") else model if isinstance(model, str) else model.config._name_or_path
            model_init_kwargs = {}
            if hasattr(args, "torch_dtype") and args.torch_dtype is not None:
                model_init_kwargs["torch_dtype"] = args.torch_dtype
            self.value_model = ValueModel(
                model_id=value_model_id,
                model_init_kwargs=model_init_kwargs,
                lr=getattr(args, "value_model_lr", 1e-4),
                dropout_prob=getattr(args, "value_head_dropout", 0.0),
            )
            if cost_head_names:
                self.value_model.add_cost_heads(cost_head_names)
            if getattr(args, "value_model_use_lora", False):
                self.value_model.apply_lora(
                    rank=getattr(args, "value_model_lora_rank", 64),
                    alpha=getattr(args, "value_model_lora_alpha", 128),
                )
        else:
            # Lightweight value head on policy hidden states (original behaviour)
            hidden_size = self.model.config.hidden_size
            self.value_model = ValueModelWrapper(
                hidden_size=hidden_size,
                cost_head_names=cost_head_names,
                lr=getattr(args, "value_model_lr", 1e-4),
                dropout_prob=getattr(args, "value_head_dropout", 0.0),
            )

        # Create optimizer BEFORE DDP wrapping (DDP hides custom methods).
        # Store a direct reference since DDP wrapping makes .optimizer inaccessible.
        self.value_model.create_optimizer()
        self.value_optimizer = self.value_model.optimizer

        # Prepare value model for distributed training
        if self.is_deepspeed_enabled and self.value_model_type == "separate":
            # Under DeepSpeed ZeRO-3, we need deepspeed.initialize() with the
            # optimizer so the backbone gets properly sharded while remaining
            # trainable (heads + optional LoRA).  prepare_deepspeed() is for
            # inference-only models (no optimizer, calls model.eval()).
            self.value_model, self.value_optimizer = self._prepare_value_model_deepspeed(
                self.value_model, self.value_optimizer
            )
        else:
            # DDP or lightweight heads — accelerator.prepare handles wrapping
            self.value_model = self.accelerator.prepare(self.value_model)

    def _prepare_value_model_deepspeed(self, value_model, optimizer):
        """
        Initialize the separate-backbone value model with DeepSpeed.

        Unlike ``prepare_deepspeed`` (which is inference-only and calls
        ``model.eval()``), this passes the optimizer so that ZeRO can
        shard both the frozen backbone *and* the trainable heads/LoRA.
        """
        import deepspeed
        from copy import deepcopy

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        stage = config_kwargs["zero_optimization"]["stage"]

        hidden_size = getattr(value_model.config, "hidden_size", None)
        if hidden_size is not None and stage == 3:
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

        # Remove scheduler config — the value model uses a fixed LR, and
        # the Trainer's scheduler config references the *policy* optimizer's
        # total steps which would be wrong here.
        config_kwargs.pop("scheduler", None)

        # Remove optimizer config so DeepSpeed uses the optimizer we pass in
        # rather than trying to construct one from config.
        config_kwargs.pop("optimizer", None)

        engine, ds_optimizer, *_ = deepspeed.initialize(
            model=value_model,
            optimizer=optimizer,
            config=config_kwargs,
        )
        # engine.train() is the default — keep it so heads + LoRA get gradients
        return engine, ds_optimizer

    # ── GAE helpers ──

    def _compute_gae(self, rewards, values, mask):
        """
        Generalized Advantage Estimation.

        Args:
            rewards: (B,) per-sequence scalar rewards
            values: (B, T) per-token value predictions (detached)
            mask: (B, T) completion mask

        Returns:
            advantages: (B, T) per-token advantages
            returns: (B, T) per-token returns (for value loss target)
        """
        B, T = values.shape
        gamma, lam = self.gae_gamma, self.gae_lambda

        # Place the trajectory reward at the last valid token
        token_rewards = torch.zeros_like(values)
        seq_lengths = mask.sum(dim=1).long()
        for i in range(B):
            if seq_lengths[i] > 0:
                token_rewards[i, seq_lengths[i] - 1] = rewards[i]

        # Backward pass
        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(B, device=values.device)
        for t in reversed(range(T)):
            next_val = values[:, t + 1] * mask[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device)
            delta = token_rewards[:, t] + gamma * next_val - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            last_gae = last_gae * mask[:, t]
            advantages[:, t] = last_gae

        returns = (advantages + values) * mask
        advantages = advantages * mask
        return advantages, returns

    def _normalize_advantages(self, advantages, mask):
        """Normalize per-token advantages globally across all processes."""
        local_count = mask.sum()
        local_sum = (advantages * mask).sum()
        local_sq = ((advantages ** 2) * mask).sum()

        counts = self.accelerator.gather(local_count.unsqueeze(0))
        sums = self.accelerator.gather(local_sum.unsqueeze(0))
        sq_sums = self.accelerator.gather(local_sq.unsqueeze(0))

        total = counts.sum()
        mean = sums.sum() / total.clamp(min=1)
        std = ((sq_sums.sum() / total.clamp(min=1)) - mean ** 2).clamp(min=0).sqrt() + 1e-8
        return ((advantages - mean) / std) * mask

    def _get_completion_hidden_states(self, model, input_ids, attention_mask, prompt_length, batch_size=None):
        """Extract hidden states for completion tokens from the policy backbone.

        Always runs under torch.no_grad() — call site is responsible for this.
        """
        batch_size = batch_size or input_ids.size(0)
        all_hidden = []
        for i in range(0, input_ids.size(0), batch_size):
            ids_b = input_ids[i:i + batch_size]
            mask_b = attention_mask[i:i + batch_size]
            unwrapped = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped):
                backbone = unwrapped.base_model.model.model
            else:
                backbone = unwrapped.model
            hidden = backbone(input_ids=ids_b, attention_mask=mask_b).last_hidden_state
            # Keep only completion token positions
            hidden = hidden[:, prompt_length:, :]
            all_hidden.append(hidden)
        return torch.cat(all_hidden, dim=0)

    # ── Override: generation and scoring with GAE advantages ──

    def _generate_and_score_completions(self, inputs):
        """
        Extends the parent to replace group-relative advantages with GAE.

        Uses the parent's generation and reward pipeline up to the point where
        rewards_per_func is gathered, then computes GAE advantages instead.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ── Reuse parent's generation + reward computation ──
        # We need to duplicate the parent method because the advantage computation
        # is tightly interleaved with the return dict construction.
        # We call the parent's code up through reward gathering, then diverge.

        from ..data_utils import is_conversational, maybe_apply_chat_template
        from ..extras.profiling import profiling_context

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer_prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # Generate completions (reuse parent's generation logic)
        prompt_completion_ids, completion_ids, prompt_ids = self._generate_completions(
            prompt_ids, prompt_mask, prompts_text, device
        )

        # Mask after first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Compute rewards (same as parent)
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    from ..data_utils import apply_chat_template
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = Trainer_prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [r if r is not None else torch.nan for r in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_kw = {key: value[nan_idx] for key, value in reward_kwargs.items()}
            row_kw["prompt"] = prompts[nan_idx]
            row_kw["completion"] = completions[nan_idx]
            warnings.warn(f"All reward functions returned None for: {row_kw}")

        rewards_per_func = gather(rewards_per_func)

        # ── PPO advantage computation (replaces GRPO group normalization) ──

        rewards_per_func_filter = torch.nan_to_num(rewards_per_func)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        # Constraint handling (Lagrangian multiplier updates)
        constraint_values = None
        cost_weights = None
        reward_weight = 1.0
        if self.use_constraints:
            main_rewards = rewards_per_func_filter[:, 0]
            constraint_rewards = rewards_per_func_filter[:, 1:]
            constraint_values = 1.0 - constraint_rewards
            self.constraints_list.append(constraint_values.detach().cpu())

            avg_cv = constraint_values.mean(0)
            avg_cs = constraint_rewards.mean(0)
            for k, name in enumerate(self.constraint_names):
                self._metrics[mode][f"constraints/{name}"].append(avg_cv[k].item())
                self._metrics[mode][f"constraint_satisfaction/{name}"].append(avg_cs[k].item())

            multipliers = F.softmax(self.multiplier_params, dim=0)[1:]

            # Skip multiplier updates during value warmup
            in_warmup = self.value_warmup_steps > 0 and self.state.global_step <= self.value_warmup_steps
            if self.state.global_step % self.update_every_k_policy_steps == 0 and len(self.constraints_list) > 0 and not in_warmup:
                train_avg = torch.cat(self.constraints_list, dim=0).to(device=device).mean(0)
                thresholds = []
                for k_i in range(len(self.constraint_names)):
                    ct = self.constraint_thresholds[k_i]
                    et = 0.5 + (ct - 0.5) * min(1.0, max(0.0, self.state.global_step / self.constraint_warmup_steps))
                    thresholds.append(et)
                thresholds = torch.stack(thresholds)
                mult_loss = torch.sum(self.multiplier_signs * multipliers * (train_avg - thresholds))
                self.multipliers_optim.zero_grad()
                mult_loss.backward()
                self.multipliers_optim.step()
                self.constraints_list = []

            cost_weights = self.multiplier_signs * multipliers.detach()
            reward_weight = 1.0 - torch.sum(torch.abs(cost_weights), dim=0)
            self._metrics[mode]["multipliers/reward_weight"].append(
                self.accelerator.gather_for_metrics(reward_weight).mean().item()
            )
            for k, name in enumerate(self.constraint_names):
                self._metrics[mode][f"raw_multipliers_values/{name}"].append(
                    self.accelerator.gather_for_metrics(self.multiplier_params[k + 1]).mean().item()
                )
                self._metrics[mode][f"multipliers/{name}"].append(
                    self.accelerator.gather_for_metrics(torch.abs(cost_weights[k])).mean().item()
                )

        # Compute per-sequence rewards for GAE
        if self.use_constraints:
            rewards = main_rewards
        else:
            rewards = (rewards_per_func_filter * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        local_rewards = rewards[process_slice]

        # Compute value predictions (no grad — only for GAE targets)
        with torch.no_grad():
            if self.value_model_type == "separate":
                vm = self.value_model.module if hasattr(self.value_model, "module") else self.value_model
                reward_values, cost_values_dict = vm.get_values_for_completion(
                    prompt_completion_ids, attention_mask, prompt_ids.size(1), batch_size
                )
            else:
                completion_hidden = self._get_completion_hidden_states(
                    self.model, prompt_completion_ids, attention_mask, prompt_ids.size(1), batch_size
                )
                reward_values, cost_values_dict = self.value_model(completion_hidden)

        # GAE for main reward
        advantages, returns = self._compute_gae(local_rewards, reward_values.detach(), completion_mask)
        advantages = self._normalize_advantages(advantages, completion_mask)

        # PPO-Lagrangian: separate GAE per constraint, blend with multipliers
        cost_returns_dict = {}
        if self.use_constraints and self.separate_cost_values and constraint_values is not None:
            advantages = reward_weight * advantages
            for k, name in enumerate(self.constraint_names):
                local_cost = constraint_values[process_slice, k]
                cost_val = cost_values_dict[name].detach()
                cost_adv, cost_ret = self._compute_gae(local_cost, cost_val, completion_mask)
                cost_adv = self._normalize_advantages(cost_adv, completion_mask)
                cost_returns_dict[name] = cost_ret
                advantages = advantages + cost_weights[k] * cost_adv
        elif self.use_constraints and not self.separate_cost_values and constraint_values is not None:
            # Scalarize then single GAE (simpler, like original CGRPO)
            scalar_rewards = local_rewards * reward_weight
            for k, name in enumerate(self.constraint_names):
                k_cost = constraint_values[process_slice, k]
                scalar_rewards = scalar_rewards + cost_weights[k] * k_cost
            advantages, returns = self._compute_gae(scalar_rewards, reward_values.detach(), completion_mask)
            advantages = self._normalize_advantages(advantages, completion_mask)

        # Log value metrics
        mean_val = (reward_values.detach() * completion_mask).sum() / completion_mask.sum().clamp(min=1)
        self._metrics[mode]["value/mean"].append(self.accelerator.gather_for_metrics(mean_val).mean().item())
        mean_ret = (returns * completion_mask).sum() / completion_mask.sum().clamp(min=1)
        self._metrics[mode]["value/mean_return"].append(self.accelerator.gather_for_metrics(mean_ret).mean().item())

        # ── Logging (same structure as parent) ──
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_lengths = agg_completion_lengths[agg_eos]
        self._metrics[mode]["completions/clipped_ratio"].append(1 - len(term_lengths) / len(completion_lengths))
        if len(term_lengths) == 0:
            term_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_lengths.float().max().item())

        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(rewards_per_func[:, i].std().item())

        # Reward stats for logging
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        if self.num_generations > 1:
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
            self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
            self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        else:
            # No groups — report batch-level std
            self._metrics[mode]["reward_std"].append(rewards.std().item())
            self._metrics[mode]["frac_reward_zero_std"].append(0.0)

        # Textual logs — for PPO advantages are (B,T), log per-sequence mean
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        per_seq_adv = (advantages * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        # Gather per-seq advantages for all processes for logging
        all_per_seq_adv = self.accelerator.gather(per_seq_adv)
        self._textual_logs["advantages"].extend(all_per_seq_adv.tolist())

        # Build return dict — flatten cost_returns into top-level keys so that
        # shuffle_tensor_dict / split_tensor_dict can handle them.
        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,           # (B, T) per-token
            "old_per_token_logps": old_per_token_logps,
            "returns": returns,                 # (B, T) for value loss
        }
        for name, cost_ret in cost_returns_dict.items():
            result[_COST_RETURNS_PREFIX + name] = cost_ret
        return result

    def _generate_completions(self, prompt_ids, prompt_mask, prompts_text, device):
        """
        Run the generation step (vLLM or HF generate). Returns prompt_completion_ids,
        completion_ids, and possibly updated prompt_ids.

        Factored out to avoid duplicating the generation logic from the parent.
        """
        from ..extras.profiling import profiling_context
        from .utils import pad

        if self.use_vllm:
            from accelerate.utils import broadcast_object_list, gather_object
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    ordered_set = all_prompts_text[::self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set, n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature, top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                n = prompt_ids.size(0)
                process_slice = slice(self.accelerator.process_index * n, (self.accelerator.process_index + 1) * n)
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                from vllm import SamplingParams
                if self.guided_decoding_regex:
                    from vllm.sampling_params import GuidedDecodingParams
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1, repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature, top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )
                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sub in gathered for p in sub]
                else:
                    all_prompts_text = prompts_text
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                if self.vllm_tensor_parallel_size > 1:
                    local_rank = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank * orig_size, (local_rank + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            from ..models import unwrap_model_for_generation
            from contextlib import nullcontext
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        return prompt_completion_ids, completion_ids, prompt_ids

    # ── Override: loss computation with value model ──

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # KL divergence
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # PPO clipped surrogate loss — advantages are already (B, T)
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            ratio = torch.clamp(ratio, max=self.args.delta)

        # advantages is (B, T) for PPO — no unsqueeze needed
        per_token_loss1 = ratio * advantages
        per_token_loss2 = clipped_ratio * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            policy_loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            policy_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            policy_loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # ── Value loss ──
        if self.value_model_type == "separate":
            # Separate backbone: forward through the value model engine/wrapper.
            # Under DeepSpeed ZeRO-3 the forward must go through the engine so
            # that sharded parameters are properly gathered.
            vm = self.value_model.module if hasattr(self.value_model, "module") else self.value_model
            reward_values, cost_values_dict = vm.get_values_for_completion(
                input_ids, attention_mask, prompt_ids.size(1)
            )
        else:
            # Lightweight: extract hidden states from policy backbone (detached)
            # so value loss gradients do NOT flow back into the policy.
            with torch.no_grad():
                completion_hidden = self._get_completion_hidden_states(
                    model, input_ids, attention_mask, prompt_ids.size(1)
                )
            completion_hidden = completion_hidden.detach()
            reward_values, cost_values_dict = self.value_model(completion_hidden)

        returns = inputs["returns"]
        vf_loss = ValueModelWrapper.compute_value_loss(reward_values, returns, completion_mask)

        # Cost value losses — unflatten the cost_returns from top-level keys
        for key, cost_ret in inputs.items():
            if key.startswith(_COST_RETURNS_PREFIX):
                name = key[len(_COST_RETURNS_PREFIX):]
                cost_vf = ValueModelWrapper.compute_value_loss(cost_values_dict[name], cost_ret, completion_mask)
                vf_loss = vf_loss + cost_vf

        # Step value optimizer (separate from policy optimizer).
        # Only during training — during eval, no gradient graph exists.
        if self.model.training:
            scaled_vf_loss = self.value_loss_coef * vf_loss
            if self.is_deepspeed_enabled and self.value_model_type == "separate":
                # DeepSpeed engine manages backward + optimizer step
                self.value_model.backward(scaled_vf_loss)
                self.value_model.step()
            else:
                self.value_optimizer.zero_grad()
                scaled_vf_loss.backward()
                self.value_optimizer.step()

        # ── Logging ──
        mode = "train" if self.model.training else "eval"

        self._metrics[mode]["value/loss"].append(self.accelerator.gather_for_metrics(vf_loss.detach()).mean().item())

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Clip ratio logging — advantages are (B, T), no unsqueeze
        is_low_clipped = (ratio < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (ratio > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low).item())
        gathered_high = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high).item())
        gathered_clip = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip.nanmean().item())

        in_warmup = self.value_warmup_steps > 0 and self.state.global_step <= self.value_warmup_steps
        self._metrics[mode]["value/warmup_active"].append(1.0 if in_warmup else 0.0)

        # During value warmup, zero policy loss so only value model trains
        if in_warmup:
            return policy_loss * 0.0
        return policy_loss


def Trainer_prepare_inputs(trainer, inputs):
    """Call the grandparent Trainer._prepare_inputs (skipping GRPOTrainer's override)."""
    from transformers import Trainer
    return Trainer._prepare_inputs(trainer, inputs)
