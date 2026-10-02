# JEV-Laya Speculative Decoder

ESP3201A project exploring a JEV-style speculative decoding architecture using Gemma 4 as the draft generator and Laya as the planned lightweight decision/verifier model.

## Current status

The current prototype:

- runs Gemma 4 E4B locally through `llama-cpp-python`;
- reads raw next-token logits;
- uses draft length `k = 2` and branching factor `b = 2`;
- constructs four two-token candidate drafts: `11`, `12`, `21`, and `22`;
- calculates per-token and joint draft probabilities;
- repeatedly appends the selected draft token IDs to the model context;
- records generation/selection timing and iteration results to CSV;
- currently uses Gemma joint probability as a temporary selector.

Laya integration, branch batching/KV-cache optimization, and final benchmarking are still in progress.

## Model

The experimental decoder uses:

- Gemma 4 E4B instruction-tuned
- Q4_K_M GGUF
- Source: `SoAIHQ/gemma-4-E4B-it-GGUF`
- File: `gemma-4-E4B-it-Q4_K_M.gguf`

GGUF is not stored in this repo, go download it yourself you bum

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/sorelootzer/Jev-Laya-Speculative-Decoder.git
cd Jev-Laya-Speculative-Decoder
```

### 2. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

On Windows, `llama-cpp-python` may try to compile from source if a compatible wheel is unavailable. For a CPU-only installation, the prebuilt wheel index can be tried directly:

```powershell
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 3. Install the Hugging Face CLI

```powershell
python -m pip install -U huggingface_hub
```

### 4. Download Gemma 4 E4B Q4_K_M

From the repository root:

```powershell
hf download SoAIHQ/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir models/Gemma4
```

The expected project-relative location is:

```text
models/
└── Gemma4/
    └── gemma-4-E4B-it-Q4_K_M.gguf
```

The `models/` directory and all `*.gguf` files are ignored by Git.

If the decoder currently contains an absolute `MODEL_PATH`, update it to point to the downloaded file. A portable form is:

```python
from pathlib import Path

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "Gemma4"
    / "gemma-4-E4B-it-Q4_K_M.gguf"
)
```

When passing it to `Llama`, use `str(MODEL_PATH)` if required by the installed binding.

## Draft architecture

For the current configuration:

```text
k = 2
b = 2
n = b^k = 4
```

The decoder explores:

```text
                    context
                       |
                  Gemma logits
                   /       \
                 T1         T2
                /  \       /  \
              T11  T12   T21  T22
               |    |     |    |
              11   12    21   22
                    |
                 selector
                    |
             selected 2 tokens
                    |
              append to context
                    |
                  repeat
```

At present the selector chooses the candidate with the highest Gemma joint log probability. The intended next stage is to replace only this selector with Laya while retaining the same draft-generation loop.

## Output and benchmarking

Generated CSV files are ignored by Git because they are experimental outputs. Current measurements include draft-generation latency, selector latency, selected draft, joint probability, end-to-end latency, and effective token throughput.

## Current limitations

- Laya is not yet integrated.
- Branches are currently evaluated sequentially.
- Full-context re-evaluation is still used instead of an optimized shared KV-cache/batched branch implementation.
- End-of-turn handling is still being refined.
- The current prototype is intended for research/testing rather than production use.
