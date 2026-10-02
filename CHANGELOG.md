# Changelog

All notable development milestones for the JEV-Laya Speculative Decoder are recorded here.

## Unreleased

### Added
- Added `laya_selector.py` with a `LayaSelector` class for selecting among Gemma candidate drafts.
- Added a clean Gemma-to-Laya interface: user prompt, assistant response-so-far, and the generated draft dictionaries are passed to `LayaSelector.select()`.
- Laya returns a draft name and the selector maps that choice back to the original Gemma draft so its exact token IDs can be preserved.
- Added Laya decision metadata including candidate probabilities, confidence, answer confidence, inference latency, token usage, and Gemma-Laya agreement.
- Added local editable Laya source workflow; `laya-source/` is excluded from Git.
- Added `requirements.txt` and documented Gemma GGUF/Hugging Face setup in the README.

### Current decoder
- Gemma 4 E4B Q4_K_M runs locally through `llama-cpp-python`.
- Direct next-token logits are exposed to Python.
- Draft length is currently `k = 2` with branching factor `b = 2`, producing four candidate drafts: `11`, `12`, `21`, and `22`.
- Candidate joint probabilities and timing metrics are recorded.
- Selected token IDs are appended to the generation context for iterative decoding.
- CSV benchmark/output logging is available.

### In progress
- Connect `gemma.py` to `LayaSelector.select()` in the live iterative generation loop.
- Add Laya probabilities, selection latency, confidence, and Gemma-Laya agreement to the decoder CSV.
- Improve early `<end_of_turn>` detection so terminal branches do not trigger unnecessary second-level inference.
- Replace repeated full-context branch evaluation with KV-cache reuse and/or batched branch evaluation.
- Benchmark normal Gemma decoding against the Laya-guided decoder.

## Initial prototype

### Added
- Local Gemma inference baseline through Ollama.
- End-to-end latency, TTFT, prompt-processing time, generation time, and token-throughput measurements.
- Direct Gemma GGUF inference through `llama-cpp-python`.
- Raw logit extraction and top-token probability calculation.
- Initial `k = 2`, `b = 2` branching draft generator.
