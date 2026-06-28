# Barnabus 3B

A free, open biblical-ethics small language model (SLM) created by **Kroice of Koinonia AI (KOKAI)**.
At ~3 billion parameters, Barnabus 3B is an SLM, not an LLM.

## What it is

- A fine-tuned small language model for biblical ethics
- Trained on evangelical Protestant, Sola Scriptura sources
- Weighs in at ~1.1B parameters (TinyLlama base + LoRA adapter)
- Designed to run offline on consumer hardware

## What it believes

- The Bible is the final authority for faith and practice
- Homosexual acts are sin; the persons involved are image-bearers to be loved
- Truth must never be celebrated as optional or flexible
- Answers should cite book, chapter, and verse whenever possible
- Moral clarity and personal dignity belong together

## Sources

- ESV Bible text
- Evangelical Protestant theology
- Westminster Shorter Catechism
- Modern evangelical ethics writers

Excluded by design: Catholic, Orthodox, and mainline Protestant sources.

## Use it

### Ollama

1. Download the released GGUF
2. Import with the included `Modelfile`
3. Run: `ollama run barnabus`

### Training

```bash
uv pip install transformers datasets peft trl torch accelerate bitsandbytes

python scripts/train_lora.py \
  --model_name TinyLlama/TinyLlama-1.1B-chat-v1.0 \
  --data_path data/train.jsonl \
  --output_dir barnabus-3b-lora \
  --num_epochs 2 \
  --batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --seed 42
```

## License

Barnabus 3B is built on Meta Llama, so it is governed by the Meta Llama Community License.
The adapter fine-tune is provided under the same Meta Llama Community License terms.

Kroice of Koinonia AI · KOKAI · 2026
