# ESP3201A — Gemma decoding experiments

Research repository for **Gemma–JEV semantic candidate selection** and a separate **Gemma E4B → 12B draft–target speculative decoding** experiment. The original Gemma–Laya prototype and standalone NLP/GSM8K tests remain in the repository as development history.

## Project overview

Two complementary systems were developed and evaluated:

1. **Gemma–JEV guided decoding:** a locally loaded Gemma 4 E4B GGUF proposes alternative token continuations. A remote JEV choice service ranks eligible draft strings; the chosen continuation is committed using its *original Gemma token IDs*. This is **semantic candidate reranking**, not distribution-preserving target verification.
2. **E4B → 12B draft–target decoding:** a smaller local Gemma model proposes a token block, and a larger Gemma target checks it through the benchmark backend. This pipeline is independent of JEV and reports token acceptance, verification calls, timings and memory measurements.

The central research questions are whether semantic selection changes answer quality and what inference costs are incurred on constrained hardware, alongside whether local draft–target verification improves target-model throughput.

## System architecture

### A. Gemma–JEV selection

```text
User prompt / accepted prefix
            |
      Gemma 4 E4B logits
            |
    Candidate tree (b=2, k=3 by default)
            |
    Candidate strings + prompt context
            |
       JEV choice API (remote)
            |
      Selected candidate ID
            |
   Commit original Gemma token IDs
            |
   Continue until EOS / EOT / budget
```

- `gemma_jev.py`: combined candidate-generation/selection implementation.
- `jev_selector.py`: JEV request construction, authentication, response validation and candidate mapping.
- `gemma_jev_kv.py`: Python-level KV-state reuse.
- `gemma_jev_kv_optimized.py`: refined traversal reducing redundant `load_state()` operations.
- `laya_selector.py`: historical selector integration.

The combined decoder has format-aware completion and termination controls; consult the selected implementation and benchmark flags for the exact policy. `k` is draft depth, `b` is branching factor; these do not represent a separate target-model verification step.

### B. E4B → 12B draft–target decoding

```text
Prompt → E4B draft model → proposed token block
                               |
                         12B target model
                               |
                    Verify / accept / reject
                               |
                         Commit tokens
```

- `benchmark_common.py`: shared decoding loop, GGUF adapter and instrumentation.
- `small_only.py`, `large_only.py`, `small_large.py`: baseline and draft–target launchers.
- `compare_outputs.py`, `diagnose_mismatch.py`, `test_decoding.py`: comparison, diagnostics and model-free checks.
- `presentation_demo.py`: interactive CPU demonstration with model paths configured locally.
- `CPU_SUMMARY_SMOKE.md`, `DRAFT_TARGET_STATUS.md`: experiment protocol and status.

These two systems must not be combined when interpreting selector overrides versus draft-token acceptance.

## Setup

Python 3.10+ is expected for the CPU benchmark tooling. On Windows PowerShell:

```powershell
git clone https://github.com/sorelootzer/Jev-Laya-Speculative-Decoder.git
cd Jev-Laya-Speculative-Decoder
python -m pip install -r requirements.txt
```

For the recorded CPU environment, `requirements_cpu.txt` contains a version-pinned package snapshot (including `llama-cpp-python==0.3.36`). A compatible prebuilt `llama-cpp-python` wheel may be necessary on Windows.

The Gemma GGUF weights are **not committed**. For the E4B JEV benchmark, the default model path is `models/Gemma4/gemma-4-E4B-it-Q4_K_M.gguf`. One documented source/download command is:

```powershell
python -m pip install -U huggingface_hub
hf download SoAIHQ/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir models/Gemma4
```

The E4B → 12B pipeline additionally needs a compatible 12B GGUF. Check `presentation_demo.py` for its `MODEL_DIR` and filenames; local paths may require editing. JEV-assisted runs require a JEV API key entered at the secure prompt; do not commit credentials. Hugging Face dataset access and availability may vary.

## Reproducing key experiments

Run from the repository root. These commands require the corresponding local model files, dependencies and, for JEV runs, API access.

**Paired NLP selection:**

```powershell
python ".\NLP Test\test_gemma_jev_nlp_extended.py" --dataset winogrande --count 20 --seed 42 --k 2 --max-tokens 16
```

Available extended choice tasks: `winogrande`, `rte`, `story_cloze`, `blimp`. Dataset-specific flags are documented by `--help`.

**Paired long-form summarization:**

```powershell
python ".\NLP Test\test_gemma_jev_summarization_kv_compare.py" --dataset samsum --decoder optimized --count 20 --seed 42 --k 3 --max-tokens 256
python ".\NLP Test\test_gemma_jev_summarization_kv_compare.py" --dataset cnn_dailymail --decoder optimized --count 20 --seed 42 --k 3 --max-tokens 256
```

These tests compare Gemma greedy and Gemma–JEV using reference ROUGE-1/2/L scores, token counts, timings and selector participation. The CNN/DailyMail loader uses `abisee/cnn_dailymail` configuration `3.0.0`; SAMSum uses `knkarthick/samsum`. Some long inputs are truncated to fit the configured input budget.

**CPU draft–target and demo:**

```powershell
python small_only.py --help
python large_only.py --help
python small_large.py --help
python presentation_demo.py --max-tokens 32
```

For exact CPU smoke-test conditions, saved prompts and environment metadata, consult `summary_smoke_seed42.json`, `summary_cpu_smoke/`, `CPU_SUMMARY_SMOKE.md`, and `presentation/final/HOW TO RUN PRESENTATION LIVE TEST.txt`. The interactive demo expects model paths and a local Python environment to be configured.

## Results snapshot

These are **separate pilot experiments**; comparisons are paired within each reported test, not pooled across hardware or protocols.

| Experiment | Gemma baseline | Assisted / optimized | Recorded observation |
| --- | --- | --- | --- |
| WinoGrande (20 choices) | 15/20 (75%) | Gemma–JEV 19/20 (95%) | 20 JEV calls, 6 overrides |
| Story Cloze (20 choices) | 20/20 | Gemma–JEV 20/20 | 20 JEV calls, 3 draft overrides |
| SAMSum (19 completed of 20) | ROUGE-1 0.3275; ROUGE-2 0.1062; ROUGE-L 0.2433 | Gemma–JEV 0.3698; 0.1495; 0.2846 | 302 JEV calls; mean 18.26 s vs 82.15 s |
| CNN/DailyMail (20/20) | ROUGE-1 0.3679; ROUGE-2 0.1332; ROUGE-L 0.2287 | Gemma–JEV 0.3676; 0.1153; 0.2195 | 607 JEV calls; mean 19.54 s vs 111.50 s; 5 inputs truncated |
| Single-example SAMSum KV optimization | Initial combined decoder: 267.76 s | Python KV: 143.61 s; refined KV: 83.88 s | Refined state loads 63 vs 125 in first KV variant |
| CPU summarization smoke (4 prompts: 2 SAMSum, 2 CNN/DailyMail) | E4B 13.57 s; 12B 36.57 s mean | E4B → 12B: K2 49.30 s; K3 51.28 s; K4 51.88 s | Draft acceptance 59.7%, 50.6%, 43.0% respectively |

The SAMSum and CNN/DailyMail figures refer to **Gemma–JEV semantic selection**, while the CPU smoke figures refer to **E4B → 12B target verification**. ROUGE measures overlap with references, not factuality by itself. The one-example KV timing comparison measures engineering optimization, not an end-to-end advantage over Gemma-only generation.

## Repository evidence map

| Path | Purpose |
| --- | --- |
| `results/2026-11-04/` | Paired NLP, SAMSum and KV experiment summaries |
| `groupmate_cnn_verify_20CNN_DailyMail_Laptop_20Questions_K3/` | Successful 20-example CNN/DailyMail CSV and summary JSON |
| `groupmate_cnn_verify20/` | Earlier unsuccessful CNN/DailyMail run, retained as history |
| `summary_cpu_smoke/` | E4B, 12B and K2/K3/K4 CPU smoke details and manifests |
| `cpu_results/`, `ram_table_fill/` | Additional draft–target runs and RAM measurements |
| `presentation_results/` | Saved interactive demo outputs |
| `GSM8K Test/`, `NLP Test/` | Standalone and paired benchmark scripts |
| `presentation/final/` | Presentation-related execution guidance |
| `CHANGELOG.md` | Chronological development record |
| `AGENTS.md` | Source-navigation guidance for independent reviewers |

## Evaluation and reproducibility notes

Record the Git commit, GGUF file and quantization, dataset/split, question IDs, seed, draft depth, context and output budgets, hardware, backend versions, and selector model/version when available. Saved results sometimes use the rolling identifier `jev-latest`. CPU smoke runs used Ryzen 7 8845HS, 32 GB DDR5-5600, Windows 11, eight CPU threads and `llama-cpp-python 0.3.36`. Results from other machines should be identified separately.

The project demonstrates an implemented semantic selection pipeline, Python KV-cache engineering, and a separate draft–target benchmark. It does not claim that remote JEV selection is local inference or that these CPU experiments establish a general distribution-preserving acceleration.
