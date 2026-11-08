# REALIZEDNULL session 2 — vast.ai rental, 2026-11-02 (UTC)

Infrastructure record for the Qwen2.5-14B rental of `studies/realizednull/SPEC.md` §9.6 (session S4c). Science
goes in `studies/realizednull/JOURNAL.md`; this file holds the offer, the deviation, cost and lifecycle.

## Deviation from the offer-filter rule (dated 2026-11-02, coordinator's decision)

- Rule set aside: `infra/vast/README.md` "Offer filter" and `controller.py BASE_QUERY`
  (`inet_down_cost<=0.0001 inet_up_cost<=0.0001 storage_cost<=0.30`), which SPEC §3.8 and §9.6 step 6 cite.
- Why: no RTX A6000 offer passed the base filter in any plan on 2026-11-02 (07:47Z, 07:56Z, 08:25Z;
  `studies/realizednull/S4_staging.md`, journal entry "S4c"). GB-3 ran on an RTX A6000 and G14-PRE compares this
  cell's NLL with GB-3's at rel 1e-3, so the coordinator chose the A6000 offer that meets the cell's 56 GiB host
  memory assert, and accepted its bandwidth charge.
- Offer taken: **58617054**, 1x RTX A6000, $0.4922/h `dph_total` at rent (the brief quoted $0.519/h), host RAM
  94.3 GiB, disk 70 GB, dlperf 57.2, inet 103 Mb/s down and 116 Mb/s up, **$0.02/GB down and $0.02/GB up**,
  storage $0.40/GB/month, reliability 0.989, machine m149573 (Arizona, US), Xeon Gold 6246R (16 effective cores).
- Expected bandwidth charge before renting (not measured): the image is 27,589,755,013 bytes in ECR (about
  $0.55 at $0.02/GB if pulled whole), the inputs are about 0.4 GB, and the full cell's `banks.npz` (about 4 GB)
  is uploaded twice (live sync and cell end, about $0.16).

## Lifecycle

- 08:25:40Z watchdog started (pid 66471).
- 08:25:48Z `python3 infra/vast/controller.py rent infra/vast/manifests/realizednull_q14.json 58617054` →
  instance **58792723**, created 08:25:53Z. Spend-record row appended to
  `infra/vast/operations/probe_session_20261027/spend.json`.
- 08:25:53Z → about 09:04:00Z `loading` (image pull, about 38 min at the host's 103 Mb/s; 69 layers, 27.59 GB).
  The watchdog warned `no pull progress` up to 14 polls (kill rule 25) while the status line stayed on one layer
  from about 08:27Z to 08:42:59Z.
- 09:03:51Z lifecycle start on the box; `NVIDIA RTX A6000, 49140 MiB, 595.99.02`.
- 09:04:19Z net probe **2.0 MB/s** (50 MB ranged GET from `results/mdl/_netprobe/probe.bin`) < 10 MB/s → exit 9,
  `_rental_complete.json {"reason": "net_probe", "exit_code": 9, "mb_per_s": 2.0}`; the lifecycle trap destroyed the
  box at about 09:04:20Z. The advertised 103 Mb/s down is 12.9 MB/s; the probe measured 2.0 MB/s.
- Watchdog: host verdict `slow_net` for m149573 (now excluded from plans); attempt files moved to
  `results/mdl/mendeleev_realizednull_q14/attempts/1/`; auto-respawn.
- 09:04:45Z watchdog (pid 66471) stopped by the operator, too late: its respawn subprocess had created instance
  **58796428** at 09:04:48Z on offer 58337839, a **CMP 170HX** ($0.3996/h, m152047, Kenya), the base filter's
  cheapest offer. 09:05:04Z destroyed while loading (`controller.py kill 58796428`); spend-record row marked
  `killed_operator`. A respawn onto the filter's first offer cannot honour a GPU-type choice: the base filter
  admits no A6000 today, and `expected_gpu` is empty in this manifest.
- 09:05:4xZ watchdog restarted (pid 1972), same log.
- 09:05:52Z rental 2: `controller.py rent … 58796199` → instance **58796532**, created 09:05:58Z. RTX A6000,
  $0.5528/h, m51579 (Virginia, US), 189 GiB host RAM (offer), dlperf 54.5, 2,276/2,234 Mb/s, **$0.0039/GB each
  way** (outside the base filter: the same dated deviation), storage $0.20/GB/month, reliability 0.9992,
  Xeon E5-2699 v4 (11 effective cores).
- About 09:11:50Z running (loading about 6 min at 2.3 Gb/s); net probe 43.0 MB/s; driver 580.173.02 (the same as
  GB-3's rental); smoke cell started 09:11:58Z.
- Some time after 09:12Z the watchdog (pid 1972) exited on `subprocess.TimeoutExpired` from
  `vastai show instances --raw` (60 s timeout, `watchdog.py:74/78`): a CLI timeout raises instead of being a skipped
  tick. Restarted 09:25:16Z under a restart-on-exit shell loop (pid 38737). Each restart resets the watchdog's
  `run_t0`, so `max_billed_h` was also enforced by an absolute deadline check (13:12:16Z).
- 10:26Z decision (science side, `studies/realizednull/JOURNAL.md` S4c): the smoke's timing projected the full cell
  above `max_billed_h` (SPEC §6.6); the smoke was given until 11:12:16Z to bank.
- 11:12:42Z the cutoff check destroyed instance 58796532 (smoke still in `construct_all`); `vastai show instances`
  `[]` at 11:12:57Z; watchdog `instance gone`; watchdog loop, deadline check and poller stopped. No job.log shipped
  (external destroy); the streamer's last `job_live.log` and `live_out/q14_smoke/` remain on S3.

## Billing (vast invoices via `python3 infra/vast/controller.py ledger`, 2026-11-02T11:15:42Z)

| instance | role | GPU | storage | download (bandwidth) | total |
|---|---|---|---|---|---|
| 58792723 | rental 1, m149573 | 0.010 h, $0.005 | 0.642 h, $0.025 | none invoiced | $0.030 |
| 58796428 | auto-respawn, CMP 170HX | — | — | — | no invoice line |
| 58796532 | rental 2, m51579 | 2.045 h, $1.091 (rate $0.5333/h) | 2.139 h, $0.042 | 0.483 GB, **$0.002** | $1.135 |

- Session total **$1.165**; vast credit 30.5577 → 29.3938.
- **Measured bandwidth charge: $0.002** (rental 2 only). Neither image pull (27.59 GB each) was invoiced as
  download, including rental 1's at $0.02/GB; the 0.483 GB matches the cell's inputs (0.39 GB), the scipy wheel
  (35 MB) and the net probe (50 MB). The deviation's expected $0.55 pull charge did not occur.
- Spend record: `infra/vast/operations/probe_session_20261027/spend.json` rows for all three instances (statuses
  `killed_infra`/`slow_net`, `killed_operator`, `killed_operator`/`clean`); realized total over the record $33.29
  against the watchdog's $36 program ceiling. Raw invoices: `spend_invoices_raw.json` beside it.

## Follow-ups (infrastructure, not fixed here)

1. `watchdog.py` crashes on a `vastai` CLI timeout (`TimeoutExpired` is not caught by the returncode check).
2. The watchdog's auto-respawn takes the base filter's first offer. With `expected_gpu` empty, it put a GB-3
   reproduction rental on a CMP 170HX. A manifest that needs one GPU type should set `expected_gpu` or
   `watch.auto_respawn: false`.
3. The 25-poll pull-stall rule (about 27 min of an unchanged status line) is close to one layer's download time on
   a 100 Mb/s host: rental 1 reached 14 polls during a healthy pull.
4. The 10 MB/s net probe rejected a host advertising 103 Mb/s (12.9 MB/s): it measured 2.0 MB/s from S3.
5. The vast API key is read from `~/.config/vastai/vast_api_key` (`infra/vast/controller.py:148`,
   `infra/vast/deploy_reaper.sh:20`). `infra/vast/README.md` does not name the path.
6. The preflight stamp `infra/vast/.preflight_pass` reads `a77ea3db9`. Its `infra` tree (`1d4fc7e1…`) no longer
   equals `HEAD:infra` (`7d9e8559…` at `906d19b22`; the S4c and S4d commits change `infra/`). Any rent needs a new
   `bash infra/vast/launch_preflight.sh` at the new HEAD, with the untracked `infra/pg/realizednull_provenance.json`
   moved aside while it runs.

---

# Session S5b (session-2 relaunch under SPEC §0.4 U5), 2026-11-02 (UTC)

## Deviation from the offer-filter rule (dated 2026-11-02, S5b)

- Rule set aside: `controller.py BASE_QUERY` terms `verified=true` and the bandwidth terms
  (`inet_down_cost<=0.0001 inet_up_cost<=0.0001`). The bandwidth charge is accepted as before (coordinator's brief).
  `verified=true` is set aside by this executor: no verified RTX A6000 offer met both the cell's 56 GiB host memory
  assert and `reliability>0.98` (SPEC §3.8), at 19:52:35Z or at 19:54:11Z. Verified offers then: 57940697
  (reliability 0.793) and 51222176 / 51222007 / 57626454 (47.2 GiB host RAM).
- `controller.py plan` at 19:52Z: `0 offers (8 wedged hosts excluded)`.
- Offer taken: **57085859**, 1x RTX A6000, unverified, reliability 0.986, Xeon w9-3475X (36 effective cores), host
  RAM 62.7 GiB in the offer (instance `cpu_ram` 128,430 MB), dlperf 54.5, inet 655 Mb/s down and 828 Mb/s up,
  **$0.0026/GB down and $0.0039/GB up**, storage $0.267/GB/month, disk 70 GB, machine m154819 (Kentucky, US),
  $0.4552/h in the search, `dph_total` $0.4793/h on the instance.

## Lifecycle

- 19:51:44Z: no `mendeleev_*` instance; the lock's pid 5883 (REUSE) dead; no watchdog process.
- 19:52:01Z: S4c's `job_live.log` moved to `results/mdl/mendeleev_realizednull_q14/attempts/2/job_live.log`.
- About 19:52:10Z: watchdog started under a shell restart loop (`WATCHDOG up pid=68206 poll=60s ceiling=$42.0`).
- 19:54:17Z `python3 infra/vast/controller.py rent infra/vast/manifests/realizednull_q14.json 57085859` → instance
  **58869906**, created 19:54:22Z; spend-record row appended (`infra/vast/operations/probe_session_20261027/spend.json`,
  row 30). Driver 615.71.10.
- Absolute deadline loop: destroys 58869906 if alive at 2026-11-03T02:54:22Z (creation + 7 h).
- 20:08:29Z lifecycle start on the box; `NVIDIA RTX A6000, 49140 MiB, 615.71.10`; net probe 29.2 MB/s; first `running`
  read 20:08:52Z (loading 14.5 min; the record's `load_min` 14.7, unbilled loading 0.231 h).
- Watchdog: `BUDGET-SOFT realized+open=$33.29 rate=$0.48/h ceiling=$42.0` on the first poll (warn only, as
  expected); `output static … min (warn-only: no progress_kill_min declared)` from about 21:07Z while construction
  ran (the boundary `live_out/progress.json` changes only between cells); no kill, no respawn, no restart of the
  watchdog process in the whole rental.
- 20:43:40Z `cells/q14_smoke/_done.json`; 20:43:42Z `q14_full` start; 23:51:46Z `cells/q14_full/_done.json`;
  23:51:47Z `_rental_complete.json`; the lifecycle trap destroyed the box. 23:52Z `vastai show instances` `[]`;
  watchdog `EVENT 58869906: instance gone (complete marker present -> ledger closed)`. Watchdog loop, deadline loop
  and pollers stopped at 23:52Z.

## Billing (vast invoices via `python3 infra/vast/controller.py ledger`, 2026-11-02T23:54:43Z)

| instance | GPU | storage | other (bandwidth) | total |
|---|---|---|---|---|
| 58869906 (m154819) | 3.714 h, $1.684 | 3.945 h, $0.102 | $0.043 | **$1.829** |

- Vast credit 26.9970 (19:51Z) → 25.1682 (23:54Z), Δ $1.829.
- Spend record: 31 rows, realized total **$37.51** against the $42.0 program ceiling. Raw invoices refreshed in
  `spend_invoices_raw.json`. Note 2026-11-03: the `reuse` study's row (`reuse_birth`, instance 58846849, $2.397,
  created 2026-11-02T16:09:54Z) landed in the same record on `main`. The merge of `science/realizednull` keeps both
  rows in start-time order; this sentence read "30 rows, $35.12" before it.
- Session 2's vast GPU time: S4c 2.055 h + S5b 3.714 h = 5.77 h, against §12.1's ceiling of 7.2 vast GPU-h.

## Follow-ups (infrastructure, not fixed here)

7. The watchdog's static-output warning reads the boundary `live_out/progress.json`, which changes only between
   cells, so it warns every minute through a long single cell.
8. Committing this record and the spend record changes `HEAD:infra`, so `infra/vast/.preflight_pass` (`5ab61f08e`)
   no longer covers HEAD. Any further rental needs `bash infra/vast/launch_preflight.sh` first.
