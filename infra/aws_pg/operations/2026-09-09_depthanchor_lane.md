# DEPTHANCHOR lane — instances, capacity, cost

Infrastructure only. The science is in `studies/depthanchor/`.

## Instance shape

`g6e.2xlarge` (one L40S, 48 GB) is the smallest that fits the job: `MolmoAct2-Think-LIBERO` at fp32 peaks
at **24.04 GB** of device memory (`preflight/12` `stages.model.cuda_max_memory_gb`), and the figures/refs
path adds Depth-Anything-V2 and the VQ-VAE. A 24 GB card (L4, A10G) does not fit fp32, and fp32 is the
card's own recommendation.

## Capacity

`2026-09-09 05:38Z` — `--type g6e.2xlarge,g6e.4xlarge` refused in every AZ
(`InsufficientInstanceCapacity` on both). The same command with
`g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.12xlarge,g6e.16xlarge` launched a `g6e.2xlarge` two minutes
later. Pass the wider list by default; the launcher walks it in order and the extra names cost nothing
when the small one is available.

## Runs

| prefix | type | outcome | wall |
|---|---|---|---|
| `preflight/10` | g6e.2xlarge | killed at 3 min — two orientation bugs found in review before it could reach them | — |
| `preflight/11` | g6e.2xlarge | exit 1 in stage `g0`: CUDA illegal memory access on the first graph-enabled call after the latency stage's graph/eager alternation (SPEC A17) | 11 min |
| `preflight/12` | g6e.2xlarge | exit 0, `ready_for_gates` true, `ready_for_loop` false (1/5) | 21 min |
| `preflight/13` | g6e.2xlarge | exit 1 on the alignment gate's horizontal clause (A19); its sweep settled the render transform at 5/5 | 33 min |
| `preflight/14` | g6e.2xlarge | exit 0, green end to end | 19 min |
| `gates/1` | g6e.2xlarge | exit 0, 2,668 calls at 23.2 s per frame | 50 min |
| `contrast/1` | g6e.2xlarge | exit 0, 4,030 calls | 79 min |
| `loop/spatial/1` | g6e.2xlarge | exit 0, 70 rollouts | 51 min |
| `loop/object/1` | g6e.2xlarge | exit 0, 70 rollouts | 58 min |
| `loop/goal/1` | g6e.2xlarge | exit 1 at rollout 54 (short depth grid) | 42 min |
| `loop/goal/2` | g6e.2xlarge | exit 0, 70 rollouts | 50 min |
| `loop/10/1` | g6e.2xlarge | exit 1 at rollout 54 (no aimed cells at that state) | 76 min |
| `loop/10/2` | g6e.2xlarge | exit 0, 70 rollouts at 93 s each | 110 min |
| `ungated/1` | g6e.2xlarge | exit 0, 4,440 calls (the exploratory cell) | 74 min |
| `figures/1` | g5.xlarge | exit 1 — matplotlib refused a −1.1e-16 error bar | 1 min |
| `figures/2` | g5.xlarge | exit 0, F1–F9 and the gallery | 3 min |

## Two things worth remembering for the next lane

**The figures job does not need an L40S.** It loads only the VQ-VAE decoder and Depth-Anything-V2 — the
seeded depth-repo cache holds just `vae-final.pt` and `depth_anything_v2_vitb.pth` — so `g5.xlarge` runs it,
and g5 had capacity when every g6e size in every AZ did not.

**Orphan reaper schedules accumulate.** `pg audit` lists a `pg-reap-<id>` `at` job for each terminated
instance. They fire, find nothing and expire; no action was taken and none appears to be needed, but a lane
of a dozen jobs leaves a dozen of them.

Per-call cost on this instance, eager: an injected call **0.87–0.90 s**, a generating call **4.7 s**
(`preflight/11` and `/12` `stages.latency`). Projected session cost at those rates: gates 0.64 h,
contrast 0.96 h. The launcher takes 1.6× the projection as `--budget-min`.
