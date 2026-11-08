# Pad tokens explain the gap (Fable, 2026-11-05 02:45 CST, from the local copies of the a2 run's outputs)

Inputs: `batch_60001/loss_check_sdpa.json` (`per_row_mean_ce`, 512 rows, HF revision `stage1-step60000`, sdpa)
and `batch_60001/tokens.npy` (pad id 100277 counted per row).

| quantity | value |
|---|---|
| mean over all 512 rows | 2.2882 |
| public training-log value at step 60,001 | 2.1240 |
| rows containing pad tokens | 9 (3,795 / 4,185 / 8,192 / 6,717 / 1,801 / 5,634 / 5,985 / 2,821 / 2,738 pads) |
| their row-mean losses | 11.19 / 12.39 / 21.28 / 12.96 / 3.26 / 15.41 / 16.41 / 3.68 / 4.04 |
| mean over the 503 rows without pad tokens | **2.1291** |
| difference from the log, in per-step sd (0.0248) | 0.21 |
| standard error of a 512-row mean (row sd 1.538) | 0.068 |

Reading. Every padded row's loss rises with its pad count (the all-pad row reaches 21.3 nats per token, far above
ln(100,278) = 11.5, so the model assigns pad near-zero probability). The training run therefore did not count pad
positions in its logged loss, contrary to the recipe's reading that nothing was masked. With padded rows removed,
the reconstructed batch's loss matches the public log to 0.005 nats, inside a quarter of the per-step spread. This
is an approximation (whole rows dropped, not pad positions); the exact token-weighted number needs per-token
losses, which `loss_check.py` does not save. No token id at or above 100,278 appears, so the vocabulary difference
(100,352 training classes against 100,278 on HF) cannot affect these inputs.

Verdict from this analysis: the reconstructed training order is confirmed at step 60,001 within noise, pending the
neighbour control on `batch_60000` with the same exclusion. The agent's `GPU_CHECK.md` is the record of the runs.
