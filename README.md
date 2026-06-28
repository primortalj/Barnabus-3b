# Barnabus 3B

A fine-tuned 3B-parameter model for biblical ethics from a nondenominational Protestant, Sola Scriptura perspective.

- Base model: `microsoft/Phi-3.5-mini-instruct` (open-weight)
- Method: LoRA / QLoRA-ready
- Training framework: Hugging Face Transformers + TRL `SFTTrainer`
- License: MIT

## Source policy

Sources ONLY:
- ESV Bible text
- Evangelical Protestant theology (Westminster Shorter Catechism, modern evangelical ethics writers)

Excluded:
- Catholic
- Orthodox
- Mainline Protestant

## Stance guardrails

- Condemn homosexual acts as sin, never affirm them as acceptable
- Never command or celebrate lying; require truth spoken in love
- Cite passages explicitly; avoid vague "the Bible says" without reference
- Balance moral clarity with personal dignity

## Training tracks

1. Scripture Q&A
2. Applied ethics
3. Pastoral counsel

## Commands

```bash
uv pip install transformers datasets peft trl torch accelerate bitsandbytes

python scripts/train_lora.py \
  --model_name microsoft/Phi-3.5-mini-instruct \
  --data_path data/train.jsonl \
  --output_dir methuselah-3b-lora \
  --num_epochs 4 \
  --batch_size 8 \
  --lora_r 8 \
  --lora_alpha 16 \
  --seed 42
```
