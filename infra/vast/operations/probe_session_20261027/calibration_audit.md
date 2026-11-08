# Independent calibration audit

Reader: gpt-6-astra, task calibration_audit, 2026-10-27. Read-only; no compute or edits by reader.

The frozen Ada calibration failure is real: delta NLL difference 0.0067403 > 0.002. Loader (bf16/SDPA/eval), g_known, known_read, pole ablation and next_token paths match the original. Exact plan hashes and all 72 window lengths match. Q4B has no BOS; exclude_first=1; pole L31N6653+, 100 contexts, window 2048. No active Q4B code change found. Original L40S truncated-context diagnostic already had a separate non-gating delta discrepancy 0.0033563.

Remaining untested software difference: wrapper invokes g_known directly; original job first ran RMS/placement/identity/generation checks. Reader saw no persistent mutation explaining a difference; proposed same-GPU pre/post warm-up comparison, not performed. Hardware/backend explanation remains unproven.

Reader: this failure does not logically invalidate within-run causal contrasts with exact identity, repeatability, working nonzero controls and identical inputs. It does block claiming historical calibration passed. A dated provisional exploratory deviation is required. Parent recorded D1 before any new exploratory response, preserving all failed flags and original thresholds.
