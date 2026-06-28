# -*- coding: utf-8 -*-
from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--adapter_path", default="methuselah-3b-lora")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)

    while True:
        question = input("\nQuestion> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        print(answer)


if __name__ == "__main__":
    main()
