\# Final presentation



Supporting evidence remains in its existing repository locations:



\- cpu\_results/run\_8340\_31604/: E4B, 12B and K=1 NLP collection.

\- summary\_cpu\_smoke/: four-prompt CPU summarization comparison.

\- ram\_table\_fill/: supplementary runs completing the RAM table.

\- presentation\_results/: saved live-demo reports.

\- groupmate\_cnn\_verify\_20CNN\_DailyMail\_Laptop\_20Questions\_K3/:

&#x20; successful 20-example CNN/DailyMail JEV verification.

\- results/: historical and teammate experiment artifacts.



Core code: benchmark\_common.py, small\_only.py, large\_only.py,

small\_large.py, compare\_outputs.py and presentation\_demo.py.



Seed: 42. CPU-only local measurements use eight threads.



Limitations:

\- Normal-batch draft–target output divergence remains unresolved.

\- RAM figures are sampled process RSS, not minimum RAM requirements.

\- Interrupted/failed attempts are excluded from successful-run averages.

\- JEV overrides are different from draft-token acceptance.

\- Different experiment protocols must not be pooled.

