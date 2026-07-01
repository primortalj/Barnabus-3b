"""LoRA/QLoRA fine-tuning pipeline using TRL SFTTrainer."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def format_example(example: dict) -> str:
    instruction = example.get('instruction', '')
    context = example.get('context', '')
    response = example.get('response', '')
    parts = []
    if instruction:
        parts.append(f'Instruction: {instruction}')
    if context:
        parts.append(f'Context: {context}')
    if response:
        parts.append(f'Response: {response}')
    text = '</s></s>'.join(parts) + '</s>'
    return text


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='microsoft/Phi-3.5-mini-instruct')
    parser.add_argument('--data_path', default='data/train.jsonl')
    parser.add_argument('--output_dir', default='barnabus-3b-lora')
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_4bit', action='store_true', default=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    dataset = dataset.map(lambda x: {'text': format_example(x)})
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        quantization_config=bnb_cfg,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora,
        dataset_text_field='text',
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=max(1, 8 // args.batch_size),
            num_train_epochs=args.num_epochs,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy='epoch',
            fp16=True,
            optim='paged_adamw_8bit',
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
