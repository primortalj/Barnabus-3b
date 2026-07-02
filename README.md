# Barnabus 3B

A free, open biblical-ethics small language model (SLM) created by **Kroice of Koinonia AI (KOKAI)**.
At ~3 billion parameters, Barnabus 3B is an SLM, not an LLM.

## What it is

- A fine-tuned small language model for biblical ethics
- Trained on evangelical Protestant, Sola Scriptura sources
- Weighs in at ~3B parameters (Llama 3.1 Instruct GGUF base)
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

```bash
ollama pull kokai/barnabus-3b
ollama run kokai/barnabus-3b
```

## Training

Local GGUF inference is the supported path at this time. If LoRA fine-tuning is added later, it must use the local GGUF/Llama.cpp pipeline under `D:\hermes\barnabus-3b`.

## License

Barnabus 3B is built on Meta Llama, so it is governed by the Meta Llama Community License.
The adapter fine-tune is provided under the same Meta Llama Community License terms.

Kroice of Koinonia AI · KOKAI · 2026
