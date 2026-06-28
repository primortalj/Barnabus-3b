# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--data_path", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="methuselah-3b-lora")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def load_local_jsonl(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def format_example(example):
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    parts = []
    if instruction:
        parts.append(f"<|user|>\n{instruction}")
    if context:
        parts.append(f"\n<|context|>\n{context}")
    parts.append(f"\n<|assistant|>\n{response}")
    return "\n".join(parts)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_local_jsonl(args.data_path)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        seed=args.seed,
        fp16=True,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=format_example,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_lora] saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
