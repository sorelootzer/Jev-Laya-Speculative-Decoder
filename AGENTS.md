# AGENTS.md — ESP3201A Final Repository Guide

This file helps independent academic reviewers navigate the submitted code and evidence. Begin with `README.md` for the current overview and `CHANGELOG.md` for the chronological development record. Inspect the actual source and result artifacts when evaluating a claim. Review should be read-only unless the repository owner explicitly requests changes.

## Two experimental systems

**Gemma–JEV semantic candidate selection:** `gemma_jev.py`, `gemma_jev_kv.py`, `gemma_jev_kv_optimized.py` and `jev_selector.py`. Gemma E4B proposes candidate token continuations; remote JEV selects among them, and the decoder commits the original Gemma token IDs. This is semantic candidate reranking rather than target-model distribution-preserving verification. `laya_selector.py` records earlier work.

**E4B → 12B draft–target verification:** `benchmark_common.py`, `small_only.py`, `large_only.py` and `small_large.py`. The E4B draft proposes tokens, and the larger target verifies them. `test_decoding.py`, `diagnose_mismatch.py`, `compare_outputs.py` and `presentation_demo.py` support this separate system.

Do not conflate JEV **draft overrides** with target-model **draft-token acceptance**.

## Suggested inspection order

1. `README.md` — architecture, setup, exact benchmark commands and results overview.
2. `gemma_jev_kv_optimized.py` — refined JEV-guided generation, tree traversal, cache-state operations and EOS/EOT handling.
3. `jev_selector.py` — remote selector authentication, candidate construction and response validation.
4. `NLP Test/test_gemma_jev_nlp_extended.py` — paired WinoGrande, Story Cloze, RTE and BLiMP evaluation.
5. `NLP Test/test_gemma_jev_summarization_kv_compare.py` — SAMSum/CNN-DailyMail evaluation, ROUGE and latency.
6. `benchmark_common.py` and `small_large.py` — separate draft–target decoding and measurement.
7. `test_decoding.py`, `CPU_SUMMARY_SMOKE.md` and `DRAFT_TARGET_STATUS.md` — verification and CPU experiment context.
8. `CHANGELOG.md` — historical decisions and milestones.

## Evidence locations

| Experiment | Source |
| --- | --- |
| Paired choice benchmarks | `results/2026-11-04/paired_*_summary.json` |
| SAMSum 20-attempt JEV run | `results/2026-11-04/paired_samsum_optimized_gemma-4-E4B-it-Q4_K_M_k3_20261104_193107_298853_summary.json` |
| KV optimization comparison | `results/2026-11-04/paired_samsum_kv_gemma-4-E4B-it-Q4_K_M_k3_20261104_181839_536995_summary.json` and `paired_samsum_optimized_gemma-4-E4B-it-Q4_K_M_k3_20261104_190355_820531_summary.json` |
| Successful CNN/DailyMail JEV run | `groupmate_cnn_verify_20CNN_DailyMail_Laptop_20Questions_K3/` (CSV and JSON) |
| Earlier unsuccessful CNN run | `groupmate_cnn_verify20/` |
| CPU E4B/12B draft–target smoke | `summary_cpu_smoke/`, `summary_smoke_seed42.json` |
| Additional CPU and RAM data | `cpu_results/`, `ram_table_fill/`, `results_cpu/` |
| Demo outputs | `presentation_results/`, `presentation/final/` |
| Other experiments | `GSM8K Test/`, `NLP Test/`, `results/` |

## Representative reproduction commands

Run from repository root with models and dependencies installed:

```powershell
python -m pip install -r requirements.txt
python ".\NLP Test\test_gemma_jev_nlp_extended.py" --dataset winogrande --count 20 --seed 42 --k 2 --max-tokens 16
python ".\NLP Test\test_gemma_jev_summarization_kv_compare.py" --dataset samsum --decoder optimized --count 20 --seed 42 --k 3 --max-tokens 256
python ".\NLP Test\test_gemma_jev_summarization_kv_compare.py" --dataset cnn_dailymail --decoder optimized --count 20 --seed 42 --k 3 --max-tokens 256
python test_decoding.py
```

`requirements_cpu.txt` records a pinned CPU environment, including `llama-cpp-python==0.3.36`. GGUF weights are not committed. JEV runs require private API credentials. Never request, expose or store API keys in review artifacts. `presentation_demo.py` may require machine-specific `MODEL_DIR` configuration.

## Reading the results

- WinoGrande: 15/20 Gemma and 19/20 Gemma–JEV; 20 selector calls.
- Story Cloze: 20/20 for both; 20 selector calls.
- SAMSum: 19 completed of 20; JEV mean ROUGE-1/2/L 0.3698/0.1495/0.2846 versus Gemma 0.3275/0.1062/0.2433. Mean response times 82.15 s versus 18.26 s.
- CNN/DailyMail: 20 completed; JEV mean ROUGE-1/2/L 0.3676/0.1153/0.2195 versus Gemma 0.3679/0.1332/0.2287. Mean response times 111.50 s versus 19.54 s. Five inputs truncated.
- Single-example Python KV optimization: combined-decoder times 267.76 → 143.61 → 83.88 s; state loads reduced from 125 to 63 between the first and refined KV variants.
- CPU smoke: four prompts (two SAMSum, two CNN/DailyMail), Ryzen 7 8845HS, 32 GB DDR5-5600, Windows 11, eight CPU threads. Mean E4B 13.57 s; 12B 36.57 s; K2 49.30 s; K3 51.28 s; K4 51.88 s.

ROUGE measures reference overlap, not factuality by itself. Record model filename/quantization, commit, hardware, dataset/split, seed, output/context budget and selector version when interpreting or reproducing runs. The `jev-latest` identifier may change over time. Keep the distinct experimental protocols separate.
