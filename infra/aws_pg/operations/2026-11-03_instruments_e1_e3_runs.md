# INSTRUMENTS E1 and E3 runs, 2026-11-03 (operations record; science in `studies/instruments/`)

| stage | instance | type | launched (UTC) | prefix | budget | reaper | tag |
|---|---|---|---|---|---|---|---|
| E3 capture (T_15, 353 poles, 17,650 captures) | `i-0fc3d37e65324181e` | g6e.4xlarge (2xlarge and xlarge had no capacity) | 2026-11-03T22:00:35Z | `results/instruments/e3/1` | 90 min | `pg-reap-…` at 23:40:34Z | `instruments-freeze-1b` |
| E1 replacement values (T_15, 56 pairs, 150 cases) | `i-01621d9517d023799` | g6e.4xlarge (same) | 2026-11-03T22:03:42Z | `results/instruments/e1/1` | 150 min | at 2026-11-04T00:43:41Z | `instruments-freeze-1a` |

Launched from `../interp_research-reuse` (fast-forwarded to main `f676b7a67`; its `.pgvenv`), `--no-weights` (the checkpoint is fetched in-job). Items uploaded first: `e3_items.json` (sha256 `5914e374ab73558d…`) to `results/instruments/e3/inputs/`, `e1_items.json` (`21b132a08479ca6e…`) to `results/instruments/inputs/`. The E1 steps file names the corpus under `results/mdl/neurons/w1/q4b/1/out/corpus/corpus` and E3's under `results/neurons/w1/q4b/1/out/corpus/corpus`; both objects exist (81,702,089 bytes each, one a copy of the other; 20 and 97 earlier steps files use one or the other). Watches running in the background (`pg watch <prefix> --instance <id>`); `metrics.json` polled, never `job.log`. Reapers to be deleted after each confirmed termination; `pg audit` at the end.

2026-11-03 22:12Z: E3 STATUS exit=0 after 518 s; instance shutting down at 22:15Z; orphan reaper `pg-reap-i-0fc3d37e65324181e` deleted; `pg audit`: 1 running (E1), 1 reaper, clean. E3 billed time about 0.15 h on a g6e.4xlarge.

| E2 manifest 1 (rank, collateral, pilot, gen base/E/D) | `i-084dca2412e652d53` | g6e.4xlarge | 2026-11-03T22:44:21Z | `results/instruments/e2/1` | 280 min | at 2026-11-04T03:34:20Z | `instruments-freeze-2` |
| E1b (gemma-3-4b-pt, Qwen3-4B-Base; weights from the S3 cache) | `i-08d56bca76a5218ab` | g6e.4xlarge | 2026-11-03T23:24:02Z | `results/instruments/e1b/1` | 120 min | at 2026-11-04T01:34:01Z | `instruments-freeze-1c` |

2026-11-03 22:42Z: E1 STATUS exit=0 after 2,287 s; orphan reaper `pg-reap-i-01621d9517d023799` deleted. Two g6e.4xlarge instances at once (E2 m1 and E1b) use the full 32-vCPU G-family quota; m2 and m3 wait for E1b to end or for the pilot's gates, whichever is later.

| E2 manifest 2 (gen A, P16, P64) | `i-0352233af0767ae0e` | g6e.4xlarge (smaller types: no capacity in any AZ) | 2026-11-03T23:58Z (after four quota-limited attempts) | `results/instruments/e2/1_m2` | 210 min | at 2026-11-04T03:38:48Z | `instruments-freeze-2` |

2026-11-03 23:50Z: E1b STATUS exit=0 after 1,555 s; orphan reaper `pg-reap-i-08d56bca76a5218ab` deleted. 2026-11-04 00:10Z: manifest 3 (gen P256, R_norm, R_KL) refused eight times on `VcpuLimitExceeded` (two g6e.4xlarge = the 32-vCPU G-family quota; g6e.2xlarge and xlarge have no capacity in any AZ); a background loop retries every three minutes until manifest 1 ends (about 01:45Z). The three manifests' expected billed time together is about 7 h; the quota, not the budget, sets the wall time.

| E2 manifest 3 (gen P256, R_norm, R_KL) | `i-042d8bc52801b0e80` | g6e.2xlarge | 2026-11-04T03:09Z | `results/instruments/e2/1_m3` | 210 min | at 2026-11-04T06:48:18Z | `instruments-freeze-2` |
| E4 (exploratory, Ward clusters; 277 cases) | `i-0071595cb32042ef6` | g6e.2xlarge | 2026-11-04T03:09Z | `results/instruments/e4/1` | 60 min | at 2026-11-04T04:19:16Z | `instruments-freeze-4x` |

2026-11-04 03:00Z: manifest 1 STATUS exit=0 after 15,340 s (about 4.3 h billed); 03:07Z: manifest 2 STATUS exit=0 (about 3.2 h billed); both orphan reapers deleted. 03:09Z: manifest 3 and E4 launched on g6e.2xlarge instances (capacity available at that hour); the manifest-3 retry loop was stopped before the direct launch to avoid a double launch. Audit: 2 running, 2 reapers, clean.

| E5 (T1 input hold, T2 marked-token write, T4 construction on Olmo, T5 member writes on three models) | `i-0762252be3d0877cc` | g6e.xlarge (2xlarge and 4xlarge had no capacity) | 2026-11-04T18:33:24Z | `results/instruments/e5/1` | 120 min | at 2026-11-04T20:43:23Z | `instruments-freeze-5` |

2026-11-04 18:33Z: E5 launched on the user's instruction (12:00 CST) after the pre-registration (E5 section, amendment 43); weights for gemma-3-4b-pt and Qwen3-4B-Base synced from the S3 cache for T5; Olmo fetched in-job. Audit before launch: clean.

| E6b (OLMo-2-1124-7B: harvest, items, E1b job) | `i-06c31116e736487e1` | g6e.xlarge (2xlarge and 4xlarge had no capacity) | 2026-11-04T20:12Z | `results/instruments/e6b/1` | 360 min | at 2026-11-05T02:22:55Z | `instruments-freeze-6` |
| E6c (the marked-token instrument on the 150 cases) | `i-0b29941915342d7af` | g6e.4xlarge (after one refused attempt at 20:16Z: `InsufficientInstanceCapacity` on every type) | 2026-11-04T20:18Z | `results/instruments/e6c/1` | 60 min | at 2026-11-04T21:28:41Z | `instruments-freeze-6` |

2026-11-04 20:16Z: E5's reaper deleted earlier; audit clean before the E6 launches. E6b landed on a g6e.xlarge (4 vCPU): if the harvest's bookkeeping is CPU-bound the run may exceed its 360-minute budget; a throughput check at about ten minutes decides whether to keep it or relaunch on a larger type (or on vast.ai per the user's note) when capacity returns.

2026-11-05 01:11Z: E6b `i-06c31116e736487e1` STATUS exit 0 at 01:11:20Z (launched 20:12:55Z: 4.97 h billed on a g6e.xlarge; the harvest 4.6 h at 3,263 tokens per second, the E1b job 18 min); instance shutting down at 01:11:50Z; reaper `pg-reap-i-06c31116e736487e1` deleted; `pg audit` 0 running, 0 reapers. E6c `i-0b29941915342d7af`: STATUS at 20:31:28Z (0.2 h). E5: STATUS at 19:07:16Z (0.57 h). Outputs pulled to `studies/instruments/results/e6/b/1/` (harvest stats 510 MB and the pair bank's npz stay local and in S3; the small artifacts are committed).
