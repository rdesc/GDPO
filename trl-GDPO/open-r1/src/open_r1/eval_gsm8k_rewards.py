import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from open_r1.gsm8k import (
    correctness_reward_func,
    format_reward_func,
    get_gsm8k_questions,
    int_reward_func,
    length_constraint_reward_func,
    length_reward_func,
)


logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the RL-finetuned checkpoint or saved model."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer path. Defaults to the model path, then the parent directory for checkpoints."},
    )
    split: str = field(default="test", metadata={"help": "GSM8K split to evaluate. Usually 'test'."})
    output_dir: str = field(default="gsm8k_eval_results", metadata={"help": "Directory for metrics and predictions."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Per-device evaluation batch size."})
    max_prompt_length: int = field(default=512, metadata={"help": "Max prompt length in tokens."})
    max_completion_length: int = field(default=1024, metadata={"help": "Max generated completion length in tokens."})
    temperature: float = field(default=0.0, metadata={"help": "Sampling temperature. Use 0.0 for greedy decoding."})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling parameter when temperature > 0."})
    do_sample: bool = field(default=False, metadata={"help": "Enable sampling instead of greedy decoding."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Torch dtype: bfloat16, float16, float32, auto."})
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation passed to from_pretrained."}
    )
    max_length_threshold: Optional[int] = field(
        default=None,
        metadata={"help": "If set, also compute the binary length constraint reward with this token threshold."},
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


def get_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def format_prompt(tokenizer, prompt_messages):
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    rendered = []
    for message in prompt_messages:
        rendered.append(f"{message['role']}: {message['content']}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def collate_examples(batch):
    return {
        "question": [example["question"] for example in batch],
        "prompt": [example["prompt"] for example in batch],
        "answer": [example["answer"] for example in batch],
    }


def summarize(values):
    count = len(values)
    mean = sum(values) / count if count else 0.0
    return {"mean": mean, "count": count}


def main():
    parser = HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_path = resolve_tokenizer_path(args.model_name_or_path, args.tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": get_dtype(args.torch_dtype),
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.to(accelerator.device)
    model.eval()

    eval_dataset: Dataset = get_gsm8k_questions(args.split)
    shard_indices = list(range(accelerator.process_index, len(eval_dataset), accelerator.num_processes))
    eval_dataset = eval_dataset.select(shard_indices)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )

    local_records = []
    reward_names = [
        "length_reward_func",
        "int_reward_func",
        "format_reward_func",
        "correctness_reward_func",
    ]
    if args.max_length_threshold is not None:
        reward_names.append("length_constraint_reward_func")

    for batch in dataloader:
        prompt_texts = [format_prompt(tokenizer, prompt) for prompt in batch["prompt"]]
        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        tokenized = {key: value.to(accelerator.device) for key, value in tokenized.items()}

        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=args.max_completion_length,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated[:, tokenized["input_ids"].shape[1] :]
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        completion_ids_list = []
        for row in completion_ids:
            row_ids = row.tolist()
            if tokenizer.eos_token_id in row_ids:
                row_ids = row_ids[: row_ids.index(tokenizer.eos_token_id) + 1]
            completion_ids_list.append(row_ids)

        completions = [[{"content": text}] for text in completion_texts]

        reward_values = {
            "length_reward_func": length_reward_func(
                completions=completions,
                answer=batch["answer"],
                completion_ids=completion_ids_list,
            ),
            "int_reward_func": int_reward_func(completions=completions),
            "format_reward_func": format_reward_func(completions=completions),
            "correctness_reward_func": correctness_reward_func(
                prompts=batch["prompt"],
                completions=completions,
                answer=batch["answer"],
                verbose_reward_logging=False,
            ),
        }
        if args.max_length_threshold is not None:
            reward_values["length_constraint_reward_func"] = length_constraint_reward_func(
                completions=completions,
                answer=batch["answer"],
                completion_ids=completion_ids_list,
                max_length_threshold=args.max_length_threshold,
            )

        for idx in range(len(batch["answer"])):
            record = {
                "question": batch["question"][idx],
                "answer": batch["answer"][idx],
                "completion": completion_texts[idx],
                "completion_token_length": len(completion_ids_list[idx]),
            }
            for reward_name in reward_names:
                record[reward_name] = reward_values[reward_name][idx]
            local_records.append(record)

    gathered_records = gather_object(local_records)

    if accelerator.is_main_process:
        records = []
        for chunk in gathered_records:
            records.extend(chunk)

        metrics = {
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_name_or_path": tokenizer_path,
            "split": args.split,
            "num_examples": len(records),
            "per_reward": {},
        }
        for reward_name in reward_names:
            metrics["per_reward"][reward_name] = summarize([record[reward_name] for record in records])
        metrics["mean_completion_token_length"] = summarize(
            [record["completion_token_length"] for record in records]
        )["mean"]

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(predictions_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        logger.info("Saved metrics to %s", metrics_path)
        logger.info("Saved predictions to %s", predictions_path)


if __name__ == "__main__":
    main()
