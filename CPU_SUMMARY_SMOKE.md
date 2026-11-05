\# CPU summarization smoke test



Hardware: Ryzen 7 8845HS, 32 GB DDR5-5600, Windows 11.

Backend: llama-cpp-python 0.3.36; CPU only; 8 threads.



Data: 2 SAMSum and 2 CNN/DailyMail validation examples.

Seed: 42. Exact prompts, references and selection metadata are

in summary\_smoke\_seed42.json. Sources were length-filtered,

not truncated.



Methods: E4B-only, 12B-only, E4B-to-12B with K=2,3,4.

Settings: context 2048, batch 256, output cap 512,

one excluded warm-up and one measured repetition.



All 20 measured responses completed without truncation or errors.

An additional K=3 warm-up was interrupted; exclude that attempt.



Mean response seconds:

\- E4B: 13.57

\- 12B: 36.57

\- K=2: 49.30

\- K=3: 51.28

\- K=4: 51.88



Weighted draft acceptance:

\- K=2: 59.7%

\- K=3: 50.6%

\- K=4: 43.0%



Exact output matches against 12B:

\- K=2: 1/4

\- K=3: 2/4

\- K=4: 2/4



These are preliminary measurements, not evidence of

correctness-preserving speedup. Different output lengths,

unresolved token divergence and high system memory usage

limit interpretation. ROUGE has not yet been calculated.



Load time excludes generation. Timing fields measure Python/backend

wall time. RAM metrics are sampled; no GPU is used.

