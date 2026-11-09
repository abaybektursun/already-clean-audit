# Files fetched by the head for the walkthrough's pictures

Each file here was copied once from S3 by the head (Fable), hashed, and committed. Data scripts read these files, never
S3. The laptop rule allows this because each fetch is small and the file is banked data, not weights.

| file | S3 object | bytes | sha256 | fetched (CST) | recorded in |
|---|---|---|---|---|---|
| `n2v_losses_t05.npz` | `s3://pg-lineage-769063704696/results/next_n2v/score-02/out/score/n2v/losses/t05.npz` | 464,987 | `7a913c9b4705e55f1d50e39c471c639b3a69d4caf562acc94205907485c56ed3` | 2026-11-08 16:40 | `studies/next/n2v/READ_n2v.md` §3 (same sha256) |
| `n13_ranking_quantiles.json` (derived) | `s3://pg-lineage-769063704696/results/next_n13/s3gate-01/out/r1_rank/ranking.npz` | 11,273,184 (source); 20,070 (derived) | `324229425ba6e56d2bf0803dfb406a51a70e5ce139193b66eb2c4a5ff6fb6729` (source) | 2026-11-08 17:15 | `studies/next/n13/STATE.md` line 283 ("ranking.npz stays on S3"); `n13/s3/s3gate-01/r1_rank/` |

`n2v_losses_t05.npz` holds the 32-bit per-token losses for one held-out birth, side `L03N9245+` born at snapshot 6 (step 12,000), read at snapshot 5, the one before the birth,
over its five unit windows (8205, 8209, 8211, 8215, 8482). Keys are `<side>:c4|<window>|<kind>|<index>` with kind in
`clean`, `zero`, `fd`, `jvp`, `probe`; each value is 2,048 float32 losses. The meaning of the index (0 to 3) is defined
in `infra/pg/n2v.py` and `studies/next/07_PICKS/N2V_SPEC.md` §2; the data script states it in `data.js`. This file
serves picture M8 (one window token by token).

`n13_ranking_quantiles.json` is derived, not a copy: the source `ranking.npz` (11.3 MB) was fetched once into an empty
scratch directory and reduced with numpy to a 399-point sorted curve of the per-unit change size s_n = sqrt(s2) over all
352,256 MLP units, the energy and smallest s of the four most-changed sets, and 32 per-layer rows; its total energy equals
`m11/data.js` → `whole_mlp_change_energy` to the last digit (11.182138138866943). The source file itself is not committed.
It serves picture M11 (the sorted curve).
