# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import json
import re

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--adapter_path", default="methuselah-3b-lora")
    parser.add_argument("--test_path", default="data/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def load_local_jsonl(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def citation_score(text):
    pattern = re.compile(r"(?:^|[\s()\[\],;])(?:\d\s*)?(?:[A-Z][a-z]+\.?|1\s*[A-Z][a-z]+\.?|2\s*[A-Z][a-z]+\.?)\s*\d+:\d+", re.MULTILINE)
    return 1.0 if pattern.search(text) else 0.0


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)

    dataset = None
    if Path(args.test_path).exists():
        dataset = load_local_jsonl(args.test_path)

    correct = 0
    total = 0
    cited = 0

    for example in (dataset or []):
        instruction = example.get("instruction", "")
        expected = example.get("response", "")
        if not instruction:
            continue

        inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        prediction = tokenizer.decode(out[0], skip_special_tokens=True)

        total += 1
        if expected and expected.strip():
            correct += 1 if expected.strip().lower() in prediction.lower() else 0
        cited += citation_score(prediction)

    if total:
        print(f"accuracy={correct/total:.3f}")
        print(f"citation_recall={cited/total:.3f}")
    else:
        print("no test examples found")


if __name__ == "__main__":
    main()
