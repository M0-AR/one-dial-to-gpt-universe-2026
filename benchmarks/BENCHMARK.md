# BENCHMARK — executed 2026-10-06, `run_all --fast` 10/10 PASS

| exp | params | before → after | verdict |
|---|---|---|---|
| 01 single dial (4) | 4 | loss 0.6692→0.0509, acc 1.00 | rules→examples holds |
| 02 valley | – | lr .02 slow, .95 oscillate, 1.05 diverge | 2/L limit reproduced |
| 03 spirals 26 | 26 | lin 0.4933→1.0000 gentle; hard 0.6625 vs 501-param 0.87 | capacity threshold (hidden) |
| 04 backprop | – | max rel err 2.68e-09 | analytic==numeric |
| 05 CNN edges | – | edge 2.3731, acc 0.9639 digits-1797 | edges first |
| 06 vectors | 8-dim | queen rank1 cos0.727 | analogy=closeness |
| 07 attention | – | it→animal 0.1882 | routing works |
| 08 tiny GPT fast | 210,432 | 3.55→2.3446 ppl10.43 | engine generates |
| 08 tiny GPT full | 818,176 | 3.2722→1.8301 ppl6.23 | brackets MiniGPT |
| 09 scaling | 30k→210k | 2.6672→2.4653→2.4399 slope −0.0482 | smooth, loss≠intel |
| 10 live market | ridge(3) | MSE 0.00015878 vs 0.00017036 persist, dir 0.5263, n=64, BTC 85980→86082 live | no edge (falsification) |

Full JSON: `benchmarks/results/*.json`. Live provenance: `data/live_snapshot.json` (UTC-stamped, Yahoo 3mo n=65 + Coinbase).
Reproduce: `docker compose up --build lab`.
