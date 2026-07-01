"""Convert a trained Barnabus 3B model to GGUF for Ollama."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Path to trained model or adapter')
    parser.add_argument('--base_model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--output_dir', default='barnabus-3b-gguf')
    parser.add_argument('--quant', default='Q4_K_M')
    parser.add_argument('--merge_first', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge_first:
        merged = out / 'merged'
        print(f'Merging LoRA adapter into base model: {merged}')
        cmd = [
            sys.executable, '-m', 'llama_cpp.tools.llama_merge',
            '--model', args.base_model,
            '--adapter', args.model_dir,
            '--outdir', str(merged),
        ]
        subprocess.run(cmd, check=True)
        convert_from = merged
    else:
        convert_from = Path(args.model_dir)
    gguf = out / f'barnabus-3b.{args.quant}.gguf'
    print(f'Converting to GGUF: {gguf}')
    cmd = [
        sys.executable, '-m', 'llama_cpp.tools.llama_convert',
        '--model', str(convert_from),
        '--outfile', str(gguf),
        '--outtype', args.quant,
    ]
    subprocess.run(cmd, check=True)
    print(f'Done: {gguf}')


if __name__ == '__main__':
    main()
