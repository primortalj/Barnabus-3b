#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${1:-barnabus-3b-lora}"
BASE_MODEL="${2:-meta-llama/Llama-3.2-3B-Instruct}"
OUT="barnabus-3b-gguf"
mkdir -p "$OUT"
python scripts/convert_to_gguf.py \
  --model_dir "$MODEL_DIR" \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUT" \
  --quant Q4_K_M \
  --merge_first
echo "GGUF ready in $OUT"
