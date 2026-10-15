# Changelog

Chronological development log for the JEV-Laya Speculative Decoder.

## TODO / Next Steps

1. **KV-cache and speculative-tree optimization (highest priority).** Preserve accepted-prefix KV state across decoding iterations; cache and reuse intermediate tree-node states rather than restoring the root and re-evaluating each path; investigate batched branch evaluation. Existing root-state reuse is only a partial optimization.
2. **Controlled baseline benchmarks.** Compare Gemma-only greedy decoding, Gemma-tree maximum-joint-probability selection, and Laya-guided decoding on identical prompts, including latency, throughput, accuracy, and verbosity.
3. **Laya selection reliability.** Investigate incorrect numerical choices, candidate ordering/presentation sensitivity, effects of draft length `k`, and the checkpoint temperature-calibration warning. Do not treat uncalibrated confidence as correctness.
4. **Completion-controller validation.** Compare the 80% terminal-draft completion override and format-based stopping against the unmodified selector; measure premature stops and prevention of post-answer rambling.
5. **Long-context KV-cache management.** Design context rollover, trimming, or summarization to handle long sessions within `n_ctx` while retaining useful information.
6. **Regression tests and scaling.** Test `k=2`, `k=3`, early terminal leaves, EOS/EOT, output-format stopping, and additional GGUF versions; benchmark growth with `k` and branching factor `b`.

---

## 2026-10-16 — Gemma 4 Termination, Completion Control, and Persistent Inference

### Gemma decoder changes
- Updated prompt framing to match the GGUF's embedded Gemma 4 canonical chat template, using `<|turn>user`, `<turn|>`, and `<|turn>model`.
- Diagnosed special end-of-turn token ID 106 (`<turn|>`) and EOS token ID 1. The older `<start_end_turn/>` spelling was not the active single-token end-of-turn marker for this model.
- Added EOS/EOT detection during speculative tree construction and token commitment, plus configurable root-level stop-probability overrides.
- Preserved meaningful terminal candidates (e.g. `22` followed by EOT), while excluding control-only or malformed protocol candidates from Laya selection.
- Added format-aware completion detection and a high-probability terminal-draft override to prevent unnecessary continuation after a sufficiently complete short answer. This is a format heuristic, not a correctness verifier.
- Retained arbitrary-`k` binary draft trees and saved-root KV-state reuse. **Persistent accepted-prefix and intermediate tree-node KV-cache reuse remain unimplemented.**
- Added an optional Gemma-only greedy baseline and detailed per-candidate Gemma/Laya probability logging.
- Reworked the command-line driver into a persistent session: load Gemma and Laya once, reuse them across prompts, and reset inference context per prompt.
- Moved results to automatically created `results/YYYY-MM-DD/` subfolders with timestamped CSV and optional baseline filenames.

### Laya selector changes
- Changed `LayaSelector.select()` to return a structured dictionary containing `selected_draft`, `selected_name`, `candidate_evaluations`, `metadata`, and `raw_result` rather than only `(selected_draft, metadata)`.
- Preserved original Gemma token IDs for the selected draft and per-candidate Gemma/Laya probability comparisons.
- Updated the choice-question presentation to label complete terminal drafts versus continuations and emphasize correctness and brevity. **This instruction/presentation change is experimental and has not been demonstrated to improve accuracy.**
- Continued supporting variable `k`, pruned terminal leaves, and candidate-count validation.

### Observations and limitations
- In one arithmetic test, Laya selected the correct `twenty-two` continuation despite Gemma preferring `18`, but subsequently continued into contradictory explanation until a completion controller was introduced.
- Other tests showed incorrect numerical selections and strong sensitivity to draft length and candidate wording. No general accuracy improvement has yet been established.
- The Laya checkpoint warns that some temperature-calibration values are invalid; reported confidence may be uncalibrated.
- The current implementation has passed local user testing for selected prompts; broader regression tests remain outstanding.


## 2026-10-15 — Termination Controller Review and Rollback

### Changed
- Reverted the unreviewed control-token filtering changes to `gemma.py`, restoring the prior implementation.
- Retained arbitrary-`k` draft generation, Laya selection, saved root KV-state reuse, and CSV performance logging.
- Deferred the proposed termination controller until it can be reviewed and tested locally.

### Issues identified
- The observed `<start_end_turn/>` marker tokenizes into seven tokens, which the current single-token stop logic cannot reliably recognize across iterations.
- Laya can select incomplete control-token sequences, causing malformed trailing output.
- `gemma.py` currently passes only full-length drafts to Laya, despite `LayaSelector` supporting shorter terminal leaves.
- The manually constructed Gemma chat template has not yet been verified against the GGUF metadata.

### Next steps
- Implement sequence-aware termination detection and separate control-token handling from semantic selection.
- Verify clean termination at `k=2` and `k=3` before approving a new implementation.

## 2026-10-03 — Single-source Runtime Draft Configuration

### Changed
- Removed the default `k = 2` from `LayaSelector`; draft length must now be supplied explicitly by the caller.
- `gemma.py` is the single source of truth for runtime draft length and passes the selected `k` directly to Laya.
- Moved Gemma model loading out of module initialization and into `load_gemma()`.
- Runtime draft length and the user prompt are now collected before either Gemma or Laya is loaded.

### Why
- Prevents Gemma and Laya from silently operating with different draft lengths.
- Avoids loading large models before the run configuration and prompt have been supplied.

## 2026-10-03 — Arbitrary-k Gemma Tree and Live Laya Integration

### Added
- Replaced the hard-coded two-level Gemma tree with a generic binary tree controlled by runtime draft length `k`.
- Draft paths are now generated dynamically; for example, `k = 3` produces `111`, `112`, `121`, `122`, `211`, `212`, `221`, and `222`.
- Connected `gemma.py` directly to `LayaSelector.select()` so Laya now chooses the continuation committed to Gemma's context.
- Removed the fixed 100-token output ceiling; generation now relies on termination conditions and remaining context capacity.
- Added dynamic per-candidate CSV rows, Laya probabilities/confidence, Gemma-Laya agreement, average log probability, and per-level Gemma timing.
- Added saved root-state reuse so the persistent context is evaluated once per tree generation and speculative paths are evaluated from that state.
- Added a development guard limiting the tree to 64 leaf drafts.

### Tested
- Successfully ran `k = 3` with eight candidate drafts.
- In a Russia-capital test, Gemma strongly preferred `111: "The capital of"` while Laya selected `121: "The official capital"`; subsequent selections still produced the correct semantic answer, demonstrating that Laya can materially alter the decoding trajectory.
- The observed `k = 3` run generated 23 tokens over 8 iterations with 55.386 s Gemma draft time, 7.480 s Laya decision time, 62.921 s end-to-end time, and 0.366 tokens/s effective throughput.

### Issue identified
- The GGUF's observed turn-ending text is `<start_end_turn/>`, tokenized as a seven-token sequence rather than one special token. The current single-token stop logic therefore does not recognize it reliably.
- This allowed Laya to select competing control-token fragments after the semantic answer had completed, producing malformed trailing control text.
- Gemma tree generation remains the dominant runtime cost; increasing `k` increases candidate coverage/lookahead but expands inference work exponentially.

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
