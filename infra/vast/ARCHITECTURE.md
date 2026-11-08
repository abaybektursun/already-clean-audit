# SHACKLETON vast.ai ops architecture (v2, 2026-08-18 — post-D19 hardening)

## Design principle

**Detection is progress- and liveness-based, never time-based — and every destroy
decision is made in ONE pure, table-tested module: `policy.py`.** Universal detectors
are size-independent (a dead streamer means the same thing on a 21-second anchor
measurement and a 20-hour training run). Anything size-dependent (runtime ceilings,
load backstops) is DECLARED per-rental in its manifest `watch` block — `rent` refuses
a manifest without `max_billed_h` + `load_backstop_min`, so bounds always exist and
are always the author's, never a global constant. Early kills are allowed only when a
rental is already certain to miss its own declared bound (the doom projection uses a
lower bound on remaining work). Silence never kills a computing job: output stalls
warn by default and kill only on author opt-in (`progress_kill_min`). Silence is also
made impossible operationally: every layer emits heartbeats, so "no news" always
means "verified healthy," never "nobody looked" (D14's lesson).

## The four layers

| layer | lives | detects | acts |
|---|---|---|---|
| 1. instance traps | inside the rental (`lifecycle.sh`) | its own exit, any code | uploads timings+log, self-destroys (piped confirm + label fallback) |
| 2. watchdog daemon | operator side (`python3 -u watchdog.py`, persistent — `-u` is REQUIRED: buffered stdout silently starves layer 3) | full signal table via `policy.decide()` | destroys, classifies, auto-respawns (infra only, bounded), stamps host verdicts |
| 3. session monitor | Claude session (Monitor over the daemon's stdout) | every daemon line | wakes the operator; anomalies acted on within one poll |
| 4. Lambda reaper | AWS (`reaper_lambda.py`, EventBridge every 5 min) | `policy.backstop_decide()`: markers, declared bounds ×1.5/×1.2 margins, 30-min-stale liveness, orphans | destroys via vast API — the laptop-independent money guarantee (during `loading` nothing runs on the instance, so layers 1–3 can all be absent) |

Layer 2 is the precise authority; layer 1 is the fast path; layer 3 is the human/agent
loop; layer 4 bounds the bleed when the first three are asleep. Deploy layer 4 once
with `bash deploy_reaper.sh`; smoke it with `bash deploy_reaper.sh invoke`.

## Signal table (layer 2, polled every 60s, decisions in policy.py)

| phase | signal | healthy | anomaly → response |
|---|---|---|---|
| loading | `status_msg` fingerprint | changes between polls | identical 5 polls → WARN; 25 → destroy (grouped 40 GB layers freeze it, D15) |
| loading | layer pull rate vs declared backstop | projection inside bound | certain miss (lower-bound projection) → destroy EARLY — saves the storage-hours a doomed load would bill |
| loading | lifetime vs `load_backstop_min` | under | exceeded → destroy |
| running | S3 `job_live.log` age (streamer re-uploads every 60 s regardless of content — staleness == dead machinery, never "quiet job") | < 4 min | > 4 min → WARN; > 8 min → destroy |
| running | output progress (live-log SIZE growth, `live_out/progress.json` from cell boundaries) | advancing | static > 30 min → WARN (default); kill ONLY if author declared `progress_kill_min` |
| running | billed time vs `max_billed_h` | under | > 90 % → WARN; exceeded → destroy, classified `ceiling` (no respawn — raising it is the author's call) |
| any | `job.log` without `_rental_complete.json` | absent | lifecycle died → destroy; traceback ⇒ `science` (NO respawn — a bug replays), else `infra` (respawn) |
| any | `_rental_complete.json` | terminal | destroy-if-alive + DONE + ledger reconcile |
| group | 2+ siblings (rent `--race N`), one running | — | loading siblings destroyed as `race_lost` (storage cents) |
| fleet | ledger realized + open estimate vs `budget.json` ceiling | under soft | soft (80 % projected) → loud WARN, work continues; hard (ceiling) → destroy all, respawns disabled |

## Respawn (closing the detect→act loop)

Infra-classified kills auto-respawn via `controller.py respawn <rental>`: the BANKED
manifest+code are reused (no re-ship), offers exclude every machine_id this rental
already ran on plus all ledger-wedged hosts, bounded by `MAX_RESPAWNS=3` counted
durably in the spend ledger. Science failures, ceiling kills, and budget kills never
respawn. Work is never lost: per-cell `_done.json` resume + the 60 s `out/` streamer
mean a respawned rental redoes ≤ 1 cell and ≤ 60 s of streamed rows.

## Manifest `watch` block (mandatory)

```json
"watch": {"max_billed_h": 6, "load_backstop_min": 90,
          "progress_stall_min": 60, "progress_kill_min": 240}
```
First two are required by `rent` and `preflight` (they parameterize layers 2 and 4).
The progress fields are optional author opt-ins for jobs whose output cadence is
known; without them output stalls only warn.

## Provenance of every rental

Rent-time: manifest banked to `results/shackleton_<rental>/manifest.json` BEFORE the
instance exists (what runs is what's banked, D11 — and layer 4 reaps labeled
instances with no banked manifest); code tar banked alongside; ECR token minted
fresh; offer + machine_id into `shackleton_spend.json`. Run-time: `timings.tsv`,
`job_live.log` (60 s), `live_out/` (60 s artifact sync + cell-boundary
`progress.json`), per-cell `_done.json` with sha256s. Terminal: `job.log`,
`_rental_complete.json`, invoice-category reconcile (`controller.py ledger`) with
`unbilled_loading_hours` cross-check, host verdict (`clean`/`wedged`/`died`/
`race_lost`) — wedged hosts are excluded from all future plans.

## Failure history that shaped this (defect ledger D1–D19)

D8 driver mismatch (fail-fast worked), D9 interactive destroy prompt (both auto-kill
layers no-oped), D11 manifests baked in images, D14 change-only watcher silent
through a 14-hour loading wedge (→ heartbeats + progress detection + layer 4),
D15 aufs layer cap (→ grouped COPYs; why LOAD_KILL is generous), D17 offline-ENV
import freeze (→ lifecycle env contract + preflight check). Each response above maps
to one of these.
