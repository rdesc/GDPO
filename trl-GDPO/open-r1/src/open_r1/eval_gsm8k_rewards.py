import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from open_r1.configs import GRPOConfig
from open_r1.gsm8k import build_reward_funcs, get_gsm8k_questions
from trl import GRPOTrainer


logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the RL-finetuned checkpoint or saved model."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path. For your GSM8K runs this is typically the base model tokenizer."},
    )
    split: str = field(default="test", metadata={"help": "GSM8K split to evaluate."})
    output_dir: str = field(default="gsm8k_eval_results", metadata={"help": "Directory for evaluation outputs."})
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional override for per-device eval batch size."},
    )
    max_prompt_length: Optional[int] = field(default=None, metadata={"help": "Optional prompt length override."})
    max_completion_length: Optional[int] = field(
        default=None, metadata={"help": "Optional completion length override."}
    )
    num_generations: Optional[int] = field(default=None, metadata={"help": "Optional num_generations override."})
    use_vllm: Optional[bool] = field(default=None, metadata={"help": "Optional use_vllm override."})
    vllm_mode: Optional[str] = field(default=None, metadata={"help": "Optional vllm_mode override."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional attention implementation override for model loading."},
    )


def resolve_tokenizer_path(model_path: str, tokenizer_path: Optional[str]) -> str:
    if tokenizer_path is not None:
        return tokenizer_path

    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    if any(os.path.exists(os.path.join(model_path, filename)) for filename in tokenizer_files):
        return model_path

    parent = os.path.dirname(model_path.rstrip("/"))
    if parent and any(os.path.exists(os.path.join(parent, filename)) for filename in tokenizer_files):
        return parent

    return model_path


def load_training_args(model_path: str):
    training_args_path = os.path.join(model_path, "training_args.bin")
    if not os.path.exists(training_args_path):
        raise FileNotFoundError(
            f"Could not find training args at {training_args_path}. Pass a checkpoint or save directory that "
            "contains training_args.bin."
        )
    return torch.load(training_args_path, map_location="cpu", weights_only=False)


def build_eval_training_args(saved_training_args, eval_args: EvalArguments):
    training_args_dict = saved_training_args.to_dict()
    training_args_dict.pop("generation_batch_size", None)
    training_args = GRPOConfig(**training_args_dict)

    training_args.output_dir = eval_args.output_dir
    training_args.report_to = []
    training_args.run_name = None
    training_args.do_eval = True
    training_args.eval_strategy = "steps"
    training_args.save_strategy = "no"
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 1
    training_args.log_completions = False
    training_args.gradient_checkpointing = False
    training_args.gradient_checkpointing_kwargs = None
    training_args.remove_unused_columns = False
    training_args.beta = 0.0
    training_args.sync_ref_model = False

    if eval_args.per_device_eval_batch_size is not None:
        training_args.per_device_eval_batch_size = eval_args.per_device_eval_batch_size
    if eval_args.max_prompt_length is not None:
        training_args.max_prompt_length = eval_args.max_prompt_length
    if eval_args.max_completion_length is not None:
        training_args.max_completion_length = eval_args.max_completion_length
    if eval_args.num_generations is not None:
        training_args.num_generations = eval_args.num_generations
    if eval_args.use_vllm is not None:
        training_args.use_vllm = eval_args.use_vllm
    if eval_args.vllm_mode is not None:
        training_args.vllm_mode = eval_args.vllm_mode

    return training_args


def main():
    parser = HfArgumentParser(EvalArguments)
    (eval_args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    saved_training_args = load_training_args(eval_args.model_name_or_path)
    training_args = build_eval_training_args(saved_training_args, eval_args)

    tokenizer_path = resolve_tokenizer_path(eval_args.model_name_or_path, eval_args.tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": "auto",
        "use_cache": True,
    }
    if eval_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = eval_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_name_or_path, **model_kwargs)

    eval_dataset = get_gsm8k_questions(eval_args.split)
    reward_funcs = build_reward_funcs(training_args)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[],
    )

    logger.info(
        "Starting trainer-faithful GSM8K eval on split=%s with num_examples=%s, num_generations=%s, use_vllm=%s",
        eval_args.split,
        len(eval_dataset),
        training_args.num_generations,
        training_args.use_vllm,
    )

    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    metrics["split"] = eval_args.split
    metrics["model_name_or_path"] = eval_args.model_name_or_path
    metrics["tokenizer_name_or_path"] = tokenizer_path

    os.makedirs(eval_args.output_dir, exist_ok=True)
    metrics_path = os.path.join(eval_args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
