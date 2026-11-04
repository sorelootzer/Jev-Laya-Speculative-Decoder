# Repository Guide for Code-Review Agents

## Purpose and scope

This repository is an ESP3201A research prototype investigating **Gemma-guided candidate generation with JEV selection**. Gemma 4 E4B generates alternative next-token continuations; a JEV choice service evaluates eligible drafts, and the selected draft's original token IDs are committed to the Gemma context. The project studies changes in answer quality and decoding cost under this selection architecture.

This document is a navigation aid for independent review. Treat source code, reproducible commands, and saved experiment artifacts as primary evidence. Do not treat this document as a substitute for inspecting them.

## Where to start

1. Read `README.md` for the repository's original setup and background; consult `CHANGELOG.md` for subsequent architectural evolution. The README may describe an earlier development stage.
2. Inspect `gemma_jev.py` for candidate-tree construction, Gemma 4 prompt framing, EOS/EOT handling, draft commitment, and baseline generation.
3. Inspect `jev_selector.py` for JEV API requests, candidate presentation, selection mapping, and response metadata.
4. Inspect `nlp_utils.py` for shared NLP dataset adapters, deterministic sampling, label mapping, and result persistence.
5. Inspect `requirements.txt` and the `results/` directory for environment requirements and recorded experiments.
6. Inspect other decoder or benchmark variants *if present in the checkout*; their availability and status must be established from actual files, not inferred from historical notes.

## Architectural reading map

- **Draft generator:** Gemma GGUF loaded using `llama-cpp-python`, with token logits used to construct a binary speculative tree. Branching factor `b` and draft length `k` determine the candidate count, subject to terminal pruning and selector limits.
- **Selector:** JEV receives a question derived from the user prompt, current response, and eligible candidate drafts. Its choice is mapped to the existing Gemma draft, retaining token IDs rather than retokenizing selected text.
- **Commit and continuation:** Accepted token IDs are appended to the generation context. EOS and end-of-turn handling determine when a response finishes.
- **Measurement:** Per-candidate probabilities, selection metadata, response times, and benchmark outcomes may be recorded as CSV or JSON, depending on the decoder variant.

Read the exact implementation being evaluated to determine its default `k`, output cap, cache policy, and whether JEV can be bypassed. Do not assume all experimental variants share settings.

## Setup and reproducibility

Install dependencies using:

```powershell
python -m pip install -r requirements.txt
```

The Gemma GGUF is not stored in Git. The documented E4B file is `models/Gemma4/gemma-4-E4B-it-Q4_K_M.gguf`; the exact path is configurable in some variants. JEV API access is separately required and should be provided through a secure prompt or environment configuration. Never print, persist, or request disclosure of API keys in review artifacts.

When reproducing a benchmark, record: Git commit SHA, source script, GGUF filename and quantization, JEV model identifier/version where available, dataset and split, seed, question IDs, `k`, `b`, context size, maximum output tokens, CPU/GPU configuration, and timing methodology. Compare paired runs on identical questions and prompts.

## Experimental evidence map

- **Multiple-choice NLP:** HellaSwag, SST-2, and other benchmark adapters are represented by `nlp_utils.py`. Additional scripts may be maintained outside the tracked repository.
- **Paired selection experiments:** Historical local WinoGrande and Story Cloze runs compared Gemma-only and Gemma–JEV on 20 questions each. Consult the matching CSV and JSON artifacts before quoting results.
- **Long-form generation:** Local SAMSum experiments compared full-text summaries using ROUGE metrics and logged JEV participation, override counts, token counts, and latency.
- **KV optimization:** Development progressed from repeated prompt evaluation to Python-level saved-state reuse and a refined traversal. Experimental native sequence-copy and multi-sequence batching were also investigated. Inspect the exact available variant before attributing a result to repository code.

## Reading performance results

Use **accuracy** for labeled choices and **ROUGE-1/2/L** for reference-based summarization. ROUGE measures text overlap, not automatically factual correctness. Report end-to-end latency, generated tokens, and throughput together; different output lengths affect timing comparisons. For JEV-assisted claims, confirm a positive selector-call count and distinguish a draft override from a final-answer change.

The local SAMSum example discussed during development recorded approximately 267.76 s for the initial combined decoder, 143.61 s with Python KV reuse, and 83.88 s with refined traversal. These are single-example local measurements, not a repository-wide speed benchmark. Use the corresponding saved results for precise attribution.

## Review procedure

Read code and experiment artifacts directly. Follow the actual control flow through candidate generation, selector invocation, token commitment, and termination. Distinguish source-level behavior from benchmark observations, and historical experiments from the current checked-out implementation. Verify reported figures from their CSV/JSON sources and identify the matching model/configuration. Do not modify source, results, or Git history during an assessment unless the repository owner authorizes it.

## Development status

The active research direction is **Gemma–JEV**. Earlier Laya integration is retained in historical files and changelog entries for provenance. The next evaluation work emphasizes matched three-arm comparisons (greedy, tree-only, JEV-guided), completion behavior, E4B/E12B comparisons, and broader long-form datasets.
