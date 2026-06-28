# -*- coding: utf-8 -*-
from pathlib import Path
import json
import re

RAW_PATHS = [
    Path("data/raw/bible_esv.jsonl"),
    Path("data/raw/theology_notes.jsonl"),
]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def normalize(obj):
    instruction = (obj.get("instruction") or obj.get("question") or "").strip()
    response = (obj.get("response") or obj.get("answer") or "").strip()
    context = (obj.get("context") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": context, "response": response}


def dedup(examples):
    seen = set()
    out = []
    for ex in examples:
        key = (ex["instruction"].lower(), ex["response"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def main():
    examples = []
    for path in RAW_PATHS:
        examples.extend(normalize(obj) for obj in load_jsonl(path))
    examples = dedup(examples)
    out = Path("data/train.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} -> {out}")


if __name__ == "__main__":
    main()
