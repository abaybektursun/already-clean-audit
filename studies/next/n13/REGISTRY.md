# N13 registry

Every number and figure of N13's S5 → the script that generated it, its inputs by sha256, and CONFIRMATORY (the rules of §6.3) or
EXPLORATORY (§6.3 last paragraph). Written 2026-11-08 by the twelfth N13 executor. Code: `studies/next/n13/scripts/` and `infra/pg/`
equal `n13-freeze-2a` (`e69f2b2b7`). Environment: the laptop's n13 venv (Python 3.12; numpy 2.5.4; matplotlib 3.11.2). Banks tag:
`n13-s4-complete` (`2d81963d9`).

## Inputs

| input | sha256 (first 12) | where |
|---|---|---|
| S5 tree manifest (165 files, 411,082,754 bytes) | `56c87b3c06c4` (full: `56c87b3c06c47dff17302f352d0436063add2f5398c57a4abb58f781bed790de`) | `s3://pg-lineage-769063704696/results/next_n13/s5-01/manifest_sha256.json`; copy `studies/next/n13/s5/manifest_sha256.json` |
| S5 tree | per file in the manifest | `results/next_n13/s5-01/banks/`; built by `s4/tools/assemble.py` (`c21e1eb206f7`) and `s4/tools/s5_extras.py` (`7d3b3c0ce1fb`) from the steps in `s5/assemble_steps.txt`; sources and `generations.jsonl` hashes in `s5/s5_sources.json` (`4fd4d6badaeb`) |
| `thresholds.json` (freeze 2) | `60bf6d4eea9c` | `studies/next/n13/thresholds.json`; tree `_inputs/thresholds.json` |
| R1 `controls.json` | `f98c30e4c3d2` | `studies/next/n13/s3/s3gate-03/r1_rank/controls.json`; tree `_inputs/controls.json` |
| R0 known metrics | `ee359fc6c5f3` | `studies/next/n13/r0/s2setup-07/r0_known/metrics.json`; tree `_inputs/r0_known/metrics.json` |
| setup `grader.json`, `template.json` | `151384d2b8eb`, `ad003c9c9c1d` | `studies/next/n13/s3/s3gate-03/setup/`; tree `_inputs/setup/` |
| `sets.json` (R1; not read by the verdict) | `49a1065c5705` | tree `_inputs/sets.json` |

## Outputs

| output | generator (sha256, first 12) | inputs | sha256 (first 12) | class |
|---|---|---|---|---|
| `studies/next/n13/verdict_n13.json`: gates and their failing inputs, three numbers, every arm's accuracy, Wilson interval and label, the 14 comparisons (difference, McNemar p, BH p, bootstrap interval), the GSM8K readout, the hypothesis verdicts | `scripts/verdict_n13.py` (`f4ca90174df2`), run once 2026-11-08T10:51:28Z, exit 0 | the S5 tree and the inputs above | `cae55c04ebf8` | CONFIRMATORY |
| `verdict_n13.json` fields `arms.*.cap_hit_share`, `answer_line_share`, `mean_tokens` | same run | same | same | EXPLORATORY (format rates, mean length) |
| `studies/next/n13/verdict_n13.md` (table view of the verdict) | same run | same | `9da504de8c82` | CONFIRMATORY |
| `studies/next/n13/figures/fig_n13.png`, `fig_n13.pdf` | `scripts/figure_n13.py` (`fcb785a2d3c6`) | `verdict_n13.json` (`cae55c04ebf8`) for every plotted value; the m_k labels only from the rndT arm records `s4/s4armsB1-02/rndT_k0.1`, `rndT_k0.3`, `rndT_k3` and `s4/s4armsB2-01/rndT_k1` `arm_record.json` (`0098b92902d4`, `eb2c38d9bd77`, `adf588332a2e`, `54bcaa4a18bc`) | `3afbbf33749e`, `aa81fbaab467` | CONFIRMATORY (picture of the verdict) |
| `studies/next/n13/results/SAMPLES_COVERAGE.md` (42 present, 0 pending) | `scripts/samples_coverage.py` (`bb57a434bf90`) | the S5 tree | `007a7a67506f` | record |
| §11 of `N13_SPEC.md` (prediction grid and arm table) | `studies/next/n13/s5/grid11.py` | `verdict_n13.json` only | — | CONFIRMATORY; the row "unchanged share of MLP entries" is not in the verdict file and is not filled |
| `studies/next/n13/CLAIMS.md` | by hand from `verdict_n13.json` and the arm records | — | — | per line |
| `studies/next/n13/s4/gates_table.txt` and `s4/<launch>/<arm>/arm_record.json` (gates, hashes against R1 stage 6, A31, energies, cap hits, lengths, wall time; no accuracy) | `s4/tools/table.py`, `s4/tools/arm_record.py` | the S4 banks | `ec70254f7e49` (table) | record |

Figure correction, 2026-11-08: the independent read (`READ_n13.md`, check (j) item 6) found that `fig_n13.png` carried no m_k labels
although §10.3 asks for one at each rndT point, and that the direct labels "rndT", "top", "rndE" overlapped at k = 3 %. `figure_n13.py`
now writes "rndT / m = <count>" under each k, read from `build.n_neurons_in_set` of the four rndT arm records (new required argument
`--arm-records studies/next/n13/s4`; the verdict does not carry m_k), and spaces the k = 3 % labels (top +10 pt, rndE 0, rndT -10 pt).
Plotted values, axes, lines, title and caption are unchanged: the new PNG differs from the old one (`6bf9a4ff67ad`, script
`a7092c877735`) only in those label pixels. Command: `PYTHONPATH=. python -m studies.next.n13.scripts.figure_n13 --verdict
studies/next/n13/verdict_n13.json --arm-records studies/next/n13/s4 --out-dir studies/next/n13/figures --title "Top 0.1-3 % MLP neurons
recover little of the MATH-500 gain; DARE on as many entries recovers most" --footer "Before allenai/Olmo-3-1025-7B a81bae42, after
allenai/Olmo-3.1-7B-RL-Zero-Math 364820b0. Attempts s3gate-02, -03, -04, s4armsA1-01, A2-01, B1-02, B2-01, B2-02; banks tag
n13-s4-complete (2d81963d9); figure_n13.py."` (with the old script, this title and footer reproduce the old PNG byte for byte). The
PDF embeds its creation date, so its sha256 changes on every render.

## Not yet

- The independent recount: `python -m studies.next.n13.scripts.recount_n13 --banks <S5 tree> --verdict verdict_n13.json` (`c680fd2ce5d8`)
  must print `RECOUNT_OK`; the reader writes `studies/next/n13/READ_n13.md`. The reader downloads the tree from
  `results/next_n13/s5-01/banks/`, checks it against the manifest, and opens the verdict (`results/next_n13/s5-01/verdict_n13.json`)
  only after its own recount is committed.
- The R1 weight statistics of §6.3's exploratory list (layer histogram of S_k, change energy by family, unchanged / one-step /
  above-floor counts) are in R1's outputs (`s3gate-03`), not in the verdict file; not registered here.
