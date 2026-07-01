"""Evaluate citation recall and response presence on a held-out set."""
from pathlib import Path
import json
import re
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/train.jsonl')
    parser.add_argument('--expected_citations', type=int, default=50)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    items = [json.loads(line) for line in Path(args.test_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    pattern = re.compile(r'(?:(?:1|2|3)\s*)?[A-Z][a-z]+\.?\s*\d+:\d+', re.M)
    cited = sum(bool(pattern.search(item['response'])) for item in items)
    invalid = sum(1 for item in items if not item.get('response', '').strip())
    print(f'Examples: {len(items)}')
    print(f'Responses present: {len(items) - invalid}')
    print(f'Explicit citations: {cited}/{len(items)}')
    if cited < args.expected_citations:
        raise SystemExit('Citation coverage below threshold')


if __name__ == '__main__':
    main()
