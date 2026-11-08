# REUSE SPEC4 (birth) on vast.ai, 2026-11-02: operations record

Lane: `infra/vast` (controller, watchdog under a session Monitor, Lambda reaper backstop), program `mendeleev`, S3 base `s3://pg-lineage-769063704696/results/mdl/`, box IAM user reads and writes `results/mdl/*` only. Science: `studies/reuse/SPEC4_birth.md`, journal `studies/reuse/JOURNAL.md`.

| step | time (UTC) | record |
|---|---|---|
| freeze 4 | 2026-11-02 16:0x | tag `reuse-freeze-4` at `38586ebda`; preflight tier 1 stamp `38586ebda` (26 of 26 policy scenarios) |
| inputs on S3 | 16:09 | `results/mdl/reuse_birth/inputs/{birth_items.json, manifest_birth_31706fd1fdbe.json, primary_t15/foreign.json}`; steps `results/mdl/reuse_birth/1/steps.json` |
| reaper | 16:09 | `bash deploy_reaper.sh`: code updated, 5-minute schedule live |
| offers | 16:09:35 | `vastai search offers` with the manifest's requirements, 7 offers, 6 pass; `offers_birth.py` chose 56735573 (RTX PRO 5000, 49 GB, $0.756/h, down 4,442 Mb/s, reliability2 0.9814, dlperf 165, estimate 39.1 min, $0.507), no other offer within $0.10; raw offers and `offers.json` under `studies/reuse/results/birth/1/` |
| rent | 16:09:55 | `controller.py rent manifests/reuse_birth.json 56735573` → instance 58846849, label `mendeleev_reuse_birth`, machine 98453, US; manifest and code (`git archive HEAD infra configs scripts` at `f2610b892`, infra tree equal to the stamp's) banked under `results/mdl/mendeleev_reuse_birth/` before the create |
| watch | 16:11 | `python3 -u watchdog.py` under a session Monitor; manifest watch bounds max_billed_h 3, load_backstop_min 30, progress_stall_min 20, progress_kill_min 45 |

Budget: vast credit $30.56 at launch; cap for this run $3.00; `budget.json` ceiling $36 unchanged. Estimate 0.8 h billed, about $0.5.

Incidents and the close-out are appended below as they happen.

**17:45Z, watch bound raised.** Five of ten loads done at 17:41Z after 90 minutes; the host's checkpoint downloads took 368 to 929 s each (T_5 929 s, T_6 552, T_7 700, T_8 368; T_15 341), so the projected end was 18:40 to 19:30Z against a 3-hour kill at 19:11Z. `watch.max_billed_h` raised from 3 to 3.9 in the banked manifest (`results/mdl/mendeleev_reuse_birth/manifest.json`, original kept in the session scratchpad) and in the repo's manifest; 3.9 h × $0.756/h = $2.95, inside SPEC4 §9's $3.00 cap. No other field changed; the reaper's margin follows the banked value.

**Close-out, 19:35Z.** `_rental_complete.json` at 19:17:57Z, cell time 11,203 s, 2,750 artifacts verified by sha256 under `cells/birth/`; the instance was destroyed by the lane after the marker; `controller.py ledger` reconciles instance 58846849 at 3.166 GPU hours, $2.397 realized ($2.322 GPU, $0.071 storage, $0.004 other), host verdict clean, unbilled loading 0.02 h; vast credit $27.00 after (from $30.56; the difference includes rounding on the account). Per-load timing: downloads 179 to 929 s (T_15 341, T_5 929, T_6 552, T_7 700, T_8 368, T_9 335, T_10 312, T_11 287, T_12 274, T_13 179), units 4 to 1,073 s (sum 76.6 minutes). Peak memory 20.5 to 21.0 GB of 50.8. Cost against the estimate: $0.51 and 39 min estimated, $2.40 and 3.17 h realized; both terms of the price rule missed by about five times (downloads 3.0 to 15.5 min against 0.5 to 3 estimated; compute 76.6 min against 16 on a card whose dlperf did not translate into this job's speed). The REUSE study's 5.0 GPU-hour cap is exceeded by 0.85 h by this rental. Lesson for the lane: the download term of the price rule should use a measured rate (the net probe runs after the rent; a pre-rent probe or a rate floor from the spend record would have excluded this host), and the watch bound should be set from the measured first-load download, not the advertised rate.
