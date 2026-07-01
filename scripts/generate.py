"""Inference wrapper for fine-tuned Barnabus 3B models."""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--adapter', default='barnabus-3b-lora')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.adapter, device_map='auto')
    model.eval()
    print('Barnabus 3B ready. Ctrl+C to stop.')
    try:
        while True:
            prompt = input('Q> ')
            if not prompt.strip():
                continue
            inputs = tokenizer(prompt, return_tensors='pt')
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            print(tokenizer.decode(out[0], skip_special_tokens=True))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
