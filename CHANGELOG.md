# Changelog

Chronological development log for the JEV-Laya Speculative Decoder.

## TODO / Next Steps

- Generalize the Gemma draft tree generator so `K` and branching factor `b` dynamically produce `b^k` candidate drafts.
- Connect `gemma.py` to `LayaSelector.select()` in the live iterative generation loop.
- Add Laya candidate probabilities, selection latency, confidence, and Gemma-Laya agreement to the decoder CSV.
- Implement early `<end_of_turn>` detection in `gemma.py` so terminal branches are emitted as short terminal leaves and are not expanded further.
- Replace repeated full-context branch evaluation with KV-cache reuse and/or batched branch evaluation.
- Benchmark normal Gemma decoding against the Laya-guided decoder.
- Evaluate output quality alongside latency/throughput so selection improvements are not measured only by speed.

---

## 2026-10-03 — Early Terminal Leaf Support in LayaSelector

### Changed
- Relaxed the selector contract from exactly `b^k` full-length drafts to at most `b^k` valid leaf drafts.
- Full non-terminal drafts must still contain exactly `k` tokens.
- Drafts may contain fewer than `k` tokens only when explicitly marked `is_terminal=True`.
- Draft names are validated against the actual number of tokens in each leaf, allowing terminal paths such as `12` in a `k = 3` tree.
- Added `terminal_draft_count` to Laya selection metadata.

### Why
- A branch that reaches EOS or `<end_of_turn>` should become a leaf immediately rather than performing meaningless inference to force it to depth `k`.
- Laya can now choose among the actual valid leaves produced by a pruned speculative tree.

## 2026-10-03 — Variable Draft Support in LayaSelector

### Added
- Generalized `LayaSelector` to accept configurable draft length `k` and branching factor `b`.
- Expected draft count is now calculated as `n = b^k`.
- Added validation that Gemma supplies exactly `b^k` candidate drafts.
- Added validation that each candidate contains exactly `k` token IDs.
- Added validation that draft names represent valid branch paths for the configured branching factor.
- Added `draft_length`, `branching_factor`, `draft_count`, and `expected_draft_count` to Laya metadata.

### Current limitation
- `LayaSelector` can handle variable `k` and `b`, but `gemma.py` still generates only `k = 2`, `b = 2`.

## 2026-10-03 — Laya Selector Module

### Added
- Added `laya_selector.py` containing the `LayaSelector` class.
- Added a clean Gemma-to-Laya interface using the original user prompt, assistant response-so-far, and Gemma candidate draft dictionaries.
- Laya selects a draft name, which is mapped back to the original Gemma draft so its exact token IDs are preserved.
- Added Laya candidate probabilities, confidence, answer confidence, inference latency, token usage, and Gemma-Laya agreement metadata.
- Added a local editable checkout of the Laya source for development; `laya-source/` is excluded from Git.

### Tested
- Confirmed local Laya inference using four candidate continuations.
- Confirmed that Laya returns a selected draft, per-candidate probabilities, confidence information, and usage statistics.

## 2026-10-03 — Repository and Reproducible Setup

### Added
- Added `requirements.txt`.
- Expanded `.gitignore` to exclude model weights, generated CSV outputs, local environments, and the local Laya source checkout.
- Documented Gemma GGUF installation and Hugging Face model download steps in the README.
- Documented the current speculative draft architecture and known limitations.

## 2026-10-02 — Iterative Gemma Draft Decoder

### Added
- Extended the initial token-tree experiment into an iterative decoder.
- Added full user-prompt input using Gemma chat formatting.
- Selected draft token IDs are appended directly to the model context before the next iteration.
- Added iterative generation until a termination condition or development token limit is reached.
- Added empty-draft and stop-token termination handling.
- Added CSV logging for per-iteration candidate drafts, selections, and timing.
- Added model, generation, selector, end-to-end, and effective throughput measurements.

### Known limitation
- Draft selection still uses Gemma joint probability inside `gemma.py`; the new Laya selector has not yet been connected to the live loop.

## 2026-10-02 — Branching Draft Generation

### Added
- Moved experimental inference from the Ollama API to direct `llama-cpp-python` inference to expose raw logits.
- Added direct next-token logit access and softmax probability calculation.
- Implemented the initial `k = 2`, `b = 2` token tree.
- Generated four two-token candidate drafts: `11`, `12`, `21`, and `22`.
- Added per-token probabilities, joint draft probabilities, and joint log probabilities.
- Added timing for first-level inference, second-level branch inference, and total draft generation.
- Added CSV output for draft inspection.

### Findings
- Looking ahead across a two-token path can change the ranking relative to selecting only the most probable first token.
- Fixed top-2 branching can omit a desirable continuation entirely; the selector cannot choose a continuation that Gemma did not include in the candidate tree.
- Increasing draft length and increasing branching factor solve different problems: larger `k` provides longer lookahead, while larger `b` increases candidate coverage.

## 2026-10-02 — Direct Gemma GGUF Setup

### Added
- Identified the Ollama Gemma 4 E4B configuration as Q4_K_M.
- Confirmed the Ollama model blob used a GGUF container but was incompatible with upstream llama.cpp tensor expectations.
- Switched the experimental decoder to an upstream-compatible Gemma 4 E4B Q4_K_M GGUF.
- Successfully loaded the model through `llama-cpp-python`.

## 2026-10-02 — Ollama Baseline

### Added
- Established local Gemma 4 E4B inference through Ollama.
- Added streamed output.
- Added end-to-end latency measurement from prompt submission to final token received by Python.
- Added time-to-first-token (TTFT), model-loading, prompt-processing, generation-time, and token-throughput measurements.
- Added repeated fixed-prompt benchmarking and CSV output.
- Established the Ollama implementation as the normal-generation baseline for later comparison.
