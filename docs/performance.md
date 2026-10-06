# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 122 vessels, 155,285 AIS fixes, 911 SAR detections,
697 stored silences and dark targets, 23 meetings. Reproduce with `make demo`, then `make load-test`.

## Reads: suspicious events, a vessel timeline and the reserve's activity, closed loop, 10 s per level

Hardware: arm64, 12 cores, Darwin (client and server on the same machine).

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 463 | 46.2 req/s | 21.5 ms | 35.1 ms | 48.0 ms | 0.00% |
| 8 | 1204 | 120.1 req/s | 59.8 ms | 93.9 ms | 182.9 ms | 0.00% |
| 32 | 1003 | 98.4 req/s | 239.0 ms | 308.2 ms | 332.0 ms | 0.00% |

Single calls, warm: the timeline 9–10 ms, the reserve's activity 16–18 ms, suspicious events 18–20 ms. The first call to suspicious
events after a restart or after new fixes arrive takes 2.9 s: it reads every fix back from Postgres to run the distance-only rule
beside the detectors, and rebuilds the simulation truth to label the findings. Both are cached until the clock or the fix count
changes. In a real centre neither would be on the read path: the comparison rule exists only for the evaluation, and there is no truth.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Simulate a fortnight (122 vessels at 10-minute steps, AIS reception, SAR passes) | 1.4 s |
| Train the gap and risk models on three past fortnights | 6.6 s, cached |
| Every detector over a fortnight (coverage, silences, fusion, attribution, meetings) | 0.44 s |
| Demo scenario end to end in-process, region load and 134,000 fixes written included | about 28 s |
| Evaluation: three held-out fortnights, training included | 12–13 s |

Not measured: the blueprint's global scale. Detectors here hold a whole region's fortnight in memory (122 vessels); a real ocean
basin has tens of thousands of vessels and would need the work split by region and time window, with AIS ingest as a stream.
