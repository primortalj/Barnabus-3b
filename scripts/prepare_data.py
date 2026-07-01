"""Prepare and normalize raw corpus into JSONL training format."""
from pathlib import Path
import json
import re

RAW_DIR = Path('data')
OUT = Path('data/train.jsonl')


def normalize(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    out = []
    if isinstance(data, dict):
        for key in ('pairs', 'examples', 'data', 'train', 'qa'):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise SystemExit('Unsupported JSON shape')
    for obj in data:
        if not isinstance(obj, dict):
            continue
        instruction = str(obj.get('instruction') or obj.get('question') or '').strip()
        response = str(obj.get('response') or obj.get('answer') or '').strip()
        context = str(obj.get('context', '')).strip()
        if instruction and response:
            out.append({'instruction': instruction, 'context': context, 'response': response})
    return out


def main():
    files = sorted(RAW_DIR.glob('*.json')) + sorted(RAW_DIR.glob('*.jsonl'))
    if not files:
        raise SystemExit(f'No source files found in {RAW_DIR}')
    merged = []
    seen = set()
    for path in files:
        for item in normalize(path):
            key = (item['instruction'].lower(), item['response'].lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    pattern = re.compile(r'(?:(?:1|2|3)\s*)?[A-Z][a-z]+\.?\s*\d+:\d+', re.M)
    cited = sum(1 for item in merged if pattern.search(item['response']))
    with OUT.open('w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Wrote {len(merged)} pairs to {OUT}')
    print(f'Explicit citations: {cited}')


if __name__ == '__main__':
    main()
