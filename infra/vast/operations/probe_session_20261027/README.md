# Manual probe session operations — 2026-10-27 UTC

Research record: `studies/neuronshow/SESSION2.md`. Infrastructure events stay here. Existing program ceiling remains $36; no budget increase. Starting reconciled ledger total was $31.348. Final invoice reconciliation is appended at shutdown.

## Rentals and incidents

- 57939792, RTX 6000 Ada, $0.6622/hour advertised: startup tests ran; historical clean NLL passed but ablation-delta calibration failed. Banked and destroyed. No exploratory scientific requests.
- 57941003, L40S, $0.8682/hour plus $0.04/GB download: network preflight measured about 7.8 MB/s, below its 10 MB/s threshold. Destroyed before model download. L40S was considered for comparability, not established price/performance superiority.
- Two watchdogs in separate worktrees auto-respawned that failure into 57941270 and 57941273, both wrong for the manifest's expected L40S. Both supervisors stopped and duplicate rentals destroyed before exploratory work. No automatic retries enabled for the final rental.
- Explicit offers 55576648 and 57629805 became unavailable before checkout; no rental was created by those attempts.
- 57941783, RTX PRO 5000, host 98453, US, $0.7555556/hour including selected disk, very low download charges: final station. Max billed time 1.25 hours; idle timeout 30 minutes. Loaded Q4B/G4/O7 together, then replaced that process with Qwen3.5, reusing the same rental and disk cache. Qwen3.5 first load 57.0 seconds; cached reload 4.5 seconds. This is a measured cache benefit, not a cross-GPU benchmark.

## Changes implemented and verified

- A per-user/program `flock` prevents two updated watchdogs on the same control machine, across worktrees. It does not coordinate different control machines or old code that ignores the lock.
- `auto_respawn: false` is honored by the watchdog. Retry offer selection also filters by the manifest's expected GPU.
- Optional `estimated_billed_h` and `estimated_download_gb` rank overhead-bound offers by advertised time + download cost. Actual loading, inference speed, output traffic and later price changes remain unmeasured in this estimate.
- Environment values are quoted for Vast's parser. Previously `EXPECTED_GPU=RTX PRO 5000` arrived as only RTX, weakening the GPU check. Unsupported quote/newline characters now fail before submission. An actual installed-Vast-parser round trip passed with synthetic values.
- `probe.py status` prints an allowlist of instance fields. Raw Vast instance objects include secrets and must not be printed. One raw status call in this session exposed rental credentials in tool output; the scoped AWS, Vast and Hugging Face credentials require rotation. No secret values are retained in this document.
- Failed CLI commands suppress argument lists containing credentials. Frozen per-job data can be included through tracked `code_paths`.
- Cached SSH connection and URL avoid two Vast lookups per request. Each probe banks a timestamped prediction, exact request, response and transport timing locally before the next question.
- Fixed-prefix inspect reports actual realized injection direction/norm, token IDs, candidate probabilities and clean/pushed top tokens. These realized checks are specific to inspect; generation/NLL do not record per-request realized deltas.

Validation: 4 focused control-plane regressions pass; 26 policy scenarios plus invariants pass; tier-1 preflight passes. Real GPU startup passed 25 tiny-model/instrument tests. Historical composite calibration remains FAILED; see calibration_audit.md and the explicitly recorded D1 deviation. No threshold was widened.

## Commands for the final station

Run from the probe-session worktree. Credentials come from existing configured profiles; never paste them into commands or logs.

```sh
python3 infra/vast/probe.py --rental neuronshow_manual_20261027_s4 status
python3 infra/vast/probe.py --rental neuronshow_manual_20261027_s4 send path/to/request.json --timeout 600
python3 infra/vast/probe.py --rental neuronshow_manual_20261027_s4 stop
```

The stop command banks the station and triggers lifecycle self-destruction. Verify the credential-free status returns an empty list. Final S3 prefix: `s3://pg-lineage-769063704696/results/mdl/mendeleev_neuronshow_manual_20261027_s4/`.

Code provenance: initial three-model exploratory server 334f5845; initial Qwen3.5 wrapper e66f4872; corrected NumPy-median wrapper b00d3938. Earlier Q4 variants use the lower-median convention and remain explicitly preserved. Server requests are answered sequentially. No API judge or large sample bank was used.

## Final verified closeout

All rentals destroyed; credential-free API snapshot `instances_after.json` records zero live instances. Watchdog 42186 stopped after that verification. The station bank `_done.json` attests 160 artifacts and 2031.8 seconds of station time. All 75 local response hashes exactly match the final S3 bank manifest; `bank_verification.json` records each match.

Invoices: Ada 57939792 $0.148; L40S 57941003 $0.065; duplicate 57941270 $0.001; duplicate 57941273 $0.001; final PRO 5000 57941783 $0.455. **Total continuation cost $0.670**, including failures/cleanup. Program ledger realized total $32.018 against unchanged $36 ceiling. Source: invoices_final.json, cost_summary.json. No live rental remains to accrue charges.

75 requests: 74 successful, one rejected empty-push request retained and retried explicitly. 64 generated samples, including original and corrected Q4 variants. Sum of successful server-reported execution times: 102.387 seconds. Median three-model fixed-prefix inspect 0.0805 seconds; median three-model generation request 2.3745 seconds (mostly two 60-token samples, some 80-token requests). These exclude startup, transport, banking and agent analysis. No measured claim that this is the fastest GPU.
