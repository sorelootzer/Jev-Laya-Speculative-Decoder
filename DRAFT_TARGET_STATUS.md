# E4B → 12B benchmark status

Main experiment: E4B-only, 12B-only, and E4B → 12B with K = 1, 2, 4, 8.
Use seed 42 and identical saved prompts across methods and machines.

Working:
- CPU laptop presentation runner and CUDA PC execution.
- Baselines and greedy draft/block-verification implementation.
- Timing, acceptance/rejection, verification-call and RAM logging.
- Model-free decoding tests and small real-model smoke tests.

Limitations:
- Normal-batch long outputs can diverge from 12B-only output.
- Batch-size-one matched the tested refrigerator output, but removes
  multi-token verification batching and is only a correctness control.
- No general correctness-preserving speedup has been established.
- Larger-scale report experiments have not been completed.
- GPU memory is monitored separately using nvidia-smi.
- Model loading time is not total process startup time.
- Presentation model paths currently need editing for another machine.
- A sampled GSM8K reference discrepancy is flagged; preserve original labels.

JEV-assisted generation is a separate experiment from this draft–target pipeline.