# D1 registry: figure and number → generating script, inputs, DISCOVERY or CONFIRMATORY (SPEC_D1 §10)

Written from `bank/w5/verdict/metrics.json` (sha256 a95bfa57bfe65246…, unblind 2026-10-24T04:06:05.084615Z). Explain worktree HEAD 07339ac9; `neurons_verdict.py` last changed 07339ac9, `neurons_figures.py` last changed 5a313c46. Every figure's footer prints the metrics sha256 it was drawn from.

## Figures (`infra/pg/neurons_figures.py --unblind bank/w5`; svg, pdf/ via svglib+reportlab, png/ via sips from the PDFs, full figure)

| figure | content | numbers from | inputs (sha256, first 16) | kind |
|---|---|---|---|---|
| `F1_audit_classes.svg` (pdf/, png/) | Q1 class fractions per model with Wilson intervals, essay reference line labelled | `q1.<cell>.fractions` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY (audit sets, 600 poles per model) |
| `F2_strong_rates.svg` (pdf/, png/) | strong rates for positives, near misses, distant and shuffled texts per model | `q1.<cell>.strong_rates` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY |
| `F3_grid_population.svg` (pdf/, png/) | double-negative share, dead share, density over the OLMo grid | `q2.grid_population` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY (S, 600 poles) |
| `F4_births.svg` (pdf/, png/) | cluster births over the grid, main and non-main, with the §6.4 null drawn | `q2.births, q2.birth_order` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY |
| `F5_parent_test.svg` (pdf/, png/) | parent-test accuracy on late-born clusters, control, chance, random and hard decoys, p_u with interval | `q2.p_u, q2.hard, judge_gate` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY (S); the gate's a_ctrl is DISCOVERY (S_gate) |
| `F6_gallery.svg` (pdf/, png/) | three poles' branching trees over training | `q2.gallery` | metrics.json a95bfa57bfe65246 (the figures read metrics.json only) | CONFIRMATORY (illustrative) |

## Numbers (`infra/pg/neurons_verdict.py unblind`, `verdict/grid.md`, RESULTS.md)

| number | metrics field | inputs (sha256, first 16) | kind |
|---|---|---|---|
| θ_s = 0.3153 | `theta` | discovery scores 33dc1b539cb0b6f5 | DISCOVERY (Q4B discovery set, frozen 2026-10-20) |
| judge gate a_ctrl = 0.910 (39 poles) | `judge_gate` | judgments 86397dcfab6ca30b, clusters 54116c7356eb01cc | DISCOVERY (S_gate, 40 poles) |
| Q1 G4: ONE 0.563, UNK 0.365, kappa 0.167 | `q1.G4` | explanations 7903648ad08e2588, texts 1974105f6ed72ca0, scores fa08966ba968fa5d | CONFIRMATORY (audit set) |
| Q1 O7: ONE 0.518, UNK 0.415, kappa 0.070 | `q1.O7` | explanations 64b89bec9efcf480, texts 80917a1a6617c63e, scores 1ed18219c54bae48 | CONFIRMATORY (audit set) |
| Q1 Q4B: ONE 0.597, UNK 0.320, kappa 0.180 | `q1.Q4B` | explanations 229af6f9f5c09131, texts c3745295078a058f, scores 86ff7e047c8a0015 | CONFIRMATORY (audit set) |
| p_u = 0.265 [0.211, 0.321] | `q2.p_u` | judgments 800fb2f774254ad5, clusters afa666e2b0bf2650, 16 grid dirs | CONFIRMATORY (S) |
| p_u sensitivity (T_7/T_15 left out) = 0.275 | `q2.sensitivity_F2_excl_T7_T15` | judgments 800fb2f774254ad5, clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY, labelled sensitivity |
| b = 0.476 [0.436, 0.517], null mean 0.414, 97.5th 0.449, p = 0.0005, drift share 0.524 | `q2.birth_order` | clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY |
| v = 0.035 [0.026, 0.044] | `q2.vestigial` | clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY |
| H_MIG = 17 poles | `q2.H_MIG` | judgments 800fb2f774254ad5, clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY |
| hard decoy accuracy 0.697, on flagged 0.356, control 0.798; flag rate 0.101 (125 flagged) | `q2.hard`, `q2.flags` | judgments 800fb2f774254ad5, clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY, not measurable |
| r(c) at birth medians 0.291 / 0.295 | `q2.r_at_birth` | clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY, exploratory |
| B4 m-within-1 0.678, unmatched share 0.047; status counts {"never_born": 0, "earliest": 919, "formed_before_step_1000": 0, "readings_disagree": 244, "unmatched_in_sample_1": 53, "late_primary": 1238} | `q2.B4_m_within_1`, `q2.B4_unmatched_share`, `q2.status_counts` | clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY |
| center birth curve: median first t T_3; by class {"ONE": "T_4", "TWO-REL": "T_6", "UNK": "T_1", "UNREL": "T_6"}; share non-decreasing 0.010; poles 599, without texts 1 | `q2.center_birth_curve` | per-step probe scores (16 sha256 in `inputs.scores_sha256`), clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, 16 grid dirs (`q2.inputs.grid`) | CONFIRMATORY data, exploratory reading |
| G2 = 0.000 (0 of 40) | `G2` | bank/w4/o7_t0_g2/explain/explanations.jsonl 26fae1b3fbec1ad8 | DISCOVERY (S_gate at T_0) |
| Holm: b 0.0010, v 1.0000 | `holm` | `q2.birth_order.p`, `q2.vestigial.p` (metrics.json a95bfa57bfe65246) | CONFIRMATORY |

Kinds: DISCOVERY = set or tuned on the discovery sets or S_gate before any confirmation number was read; CONFIRMATORY = read once at W5 on the audit sets and S. Reproduce with `runbook/w5_run.sh unblind`, `figures`, `results` after moving `bank/w5` aside (drafts repo; `CENTER=` for the center probe; `all` stops at collect, which refuses on the collected S; handoff/S6.md, Resume steps), then `runbook/results_hand_edits.py` and this script.
