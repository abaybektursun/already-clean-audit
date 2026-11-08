"""Write bank/w5/REGISTRY.md (SPEC_D1 §10): every figure and number -> generating script, inputs (sha256), DISCOVERY or CONFIRMATORY.
usage: python3 -I write_registry.py <explain worktree root>   (reads bank/w5/verdict/metrics.json and git HEAD there; for F4, the three
chance-level files of this repo that runbook/f4_births.py reads)"""
import hashlib, json, pathlib, subprocess, sys
E = pathlib.Path(sys.argv[1]); m = json.load(open(E / "bank/w5/verdict/metrics.json"))
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
git = lambda *a: subprocess.run(["git", "-C", str(E), *a], capture_output=True, text=True).stdout.strip()
head, verc, figc = git("rev-parse", "--short", "HEAD"), git("log", "-1", "--format=%h", "--", "infra/pg/neurons_verdict.py"), git("log", "-1", "--format=%h", "--", "infra/pg/neurons_figures.py")
ms = sha16(E / "bank/w5/verdict/metrics.json"); X = E / "bank/w4/o7_s_v2/explain"
jh, ch, ph = sha16(X / "judgments.jsonl"), sha16(X / "clusters.jsonl"), sha16(X / "positions.npz")
L = ["# D1 registry: figure and number → generating script, inputs, DISCOVERY or CONFIRMATORY (SPEC_D1 §10)", "",
     f"Written from `bank/w5/verdict/metrics.json` (sha256 {sha16(E / 'bank/w5/verdict/metrics.json')}…, unblind {m['created_utc']}). Explain worktree HEAD {head}; "
     f"`neurons_verdict.py` last changed {verc}, `neurons_figures.py` last changed {figc}. Every figure's footer prints the metrics sha256 it was drawn from.", "",
     "## Figures (`infra/pg/neurons_figures.py --unblind bank/w5`; svg, pdf/ via svglib+reportlab, png/ via sips from the PDFs, full figure)", "", "| figure | content | numbers from | inputs (sha256, first 16) | kind |", "|---|---|---|---|---|"]
only = f"metrics.json {ms} (the figures read metrics.json only)"
R = pathlib.Path(__file__).resolve().parents[3]     # this repo: the chance-level files F4 reads (runbook/f4_births.py, 2026-11-05 retag)
f4_in = f"metrics.json {ms}; " + "; ".join(f"`{p}` {sha16(R / p)}" for p in (
    "studies/next/10_recount2/out/05_chance.json", "studies/next/10_recount2/out/04_b.json", "studies/next/06b_verify_births/out/s03_histogram_tv.json"))
blo, bhi = json.load(open(R / "studies/next/10_recount2/out/04_b.json"))["b_ci95_bootstrap_poles"]
r1t = json.load(open(R / "studies/next/10_recount2/out/05_chance.json"))["summary"]["R1t"]       # the retag's chance level for b
f4_c = ("cluster births over the grid, main and non-main, with the chance level of random same-size groups drawn on each bar and b stated against "
        "the random-group reference beside the §6.4 permutation band. Redrawn 2026-11-05 by `runbook/f4_births.py` (the panels of the old "
        "`neurons_figures.py` f4, which since the housekeeping of 2026-11-05 calls that script's draw(); its helpers imported); b's interval is the "
        f"recount's bootstrap [{blo:.3f}, {bhi:.3f}] (`04_b.json`; redrawn again 2026-11-05, `studies/next/10_RETAG.md` Follow-ups item 4); pdf/ by headless "
        "Chrome print, png/ by sips from the PDF at width 1840; the earlier F4 drew the §6.4 band only and read \"main oldest\"")
for f, c, n, i, k in [("F1_audit_classes", "Q1 class fractions per model with Wilson intervals, essay reference line labelled", "`q1.<cell>.fractions`", only, "CONFIRMATORY (audit sets, 600 poles per model)"),
                      ("F2_strong_rates", "strong rates for positives, near misses, distant and shuffled texts per model", "`q1.<cell>.strong_rates`", only, "CONFIRMATORY"),
                      ("F3_grid_population", "double-negative share, dead share, density over the OLMo grid", "`q2.grid_population`", only, "CONFIRMATORY (S, 600 poles)"),
                      ("F4_births", f4_c, "`q2.births, q2.birth_order`; `summary.R1t`; `b_ci95_bootstrap_poles`; `main_random_mean, non_main_random_mean`", f4_in,
                       "CONFIRMATORY (births, b); the chance level EXPLORATORY (computed after unblinding, 2026-11-05)"),
                      ("F5_parent_test", "parent-test accuracy on late-born clusters, control, chance, random and hard decoys, p_u with interval", "`q2.p_u, q2.hard, judge_gate`", only, "CONFIRMATORY (S); the gate's a_ctrl is DISCOVERY (S_gate)"),
                      ("F6_gallery", "three poles' branching trees over training", "`q2.gallery`", only, "CONFIRMATORY (illustrative)")]:
    L.append(f"| `{f}.svg` (pdf/, png/) | {c} | {n} | {i} | {k} |")
L += ["", "## Numbers (`infra/pg/neurons_verdict.py unblind`, `verdict/grid.md`, RESULTS.md)", "", "| number | metrics field | inputs (sha256, first 16) | kind |", "|---|---|---|---|"]
th, jg, q2, qi = m["theta"], m["judge_gate"], m["q2"], m["q2"]["inputs"]
L.append(f"| θ_s = {th['theta_s']:.4f} | `theta` | discovery scores {th['inputs']['scores']['sha256'][:16]} | DISCOVERY (Q4B discovery set, frozen 2026-10-20) |")
L.append(f"| judge gate a_ctrl = {jg['a_ctrl']:.3f} ({jg['n_poles']} poles) | `judge_gate` | judgments {jg['inputs']['judgments']['sha256'][:16]}, clusters {jg['inputs']['clusters']['sha256'][:16]} | DISCOVERY (S_gate, 40 poles) |")
for cell, q in sorted(m["q1"].items()):
    i = q["inputs"]; L.append(f"| Q1 {cell}: ONE {q['fractions']['ONE']['p']:.3f}, UNK {q['fractions']['UNK']['p']:.3f}, kappa {q['kappa_label']:.3f} | `q1.{cell}` | explanations {i['explanations']['sha256'][:16]}, texts {i['texts']['sha256'][:16]}, scores {i['scores']['sha256'][:16]} | CONFIRMATORY (audit set) |")
L.append(f"| p_u = {q2['p_u']['point']:.3f} [{q2['p_u']['ci95'][0]:.3f}, {q2['p_u']['ci95'][1]:.3f}] | `q2.p_u` | judgments {qi['judgments']['sha256'][:16]}, clusters {qi['clusters']['sha256'][:16]}, 16 grid dirs | CONFIRMATORY (S) |")
grid = f"clusters {ch}, positions {ph}, 16 grid dirs (`q2.inputs.grid`)"
L.append(f"| p_u sensitivity (T_7/T_15 left out) = {q2['sensitivity_F2_excl_T7_T15']['p_u']['point']:.3f} | `q2.sensitivity_F2_excl_T7_T15` | judgments {jh}, {grid} | CONFIRMATORY, labelled sensitivity |")
L.append(f"| b = {q2['birth_order']['b']:.3f} [{q2['birth_order']['ci95'][0]:.3f}, {q2['birth_order']['ci95'][1]:.3f}], null mean {q2['birth_order']['null_mean']:.3f}, 97.5th {q2['birth_order']['null_97_5']:.3f}, p = {q2['birth_order']['p']:.4f}, drift share {q2['birth_order']['drift_share']:.3f} | `q2.birth_order` | {grid} | CONFIRMATORY; the \"main oldest\" reading RETAGGED 2026-11-05, not supported: below random same-size groups {r1t['b_ref_mean']:.3f} [{r1t['b_ref_ci95_over_replicates'][0]:.3f}, {r1t['b_ref_ci95_over_replicates'][1]:.3f}] (`RESULTS.md`, Retag) |")
L.append(f"| v = {q2['vestigial']['v']:.3f} [{q2['vestigial']['ci95'][0]:.3f}, {q2['vestigial']['ci95'][1]:.3f}] | `q2.vestigial` | {grid} | CONFIRMATORY |")
L.append(f"| H_MIG = {q2['H_MIG']['count']} poles | `q2.H_MIG` | judgments {jh}, {grid} | CONFIRMATORY |")
L.append(f"| hard decoy accuracy {q2['hard']['accuracy']:.3f}, on flagged {q2['hard']['accuracy_on_flagged']:.3f}, control {q2['hard']['ctrl_accuracy']:.3f}; flag rate {q2['flags']['share']:.3f} ({q2['flags']['n_flagged']} flagged) | `q2.hard`, `q2.flags` | judgments {jh}, {grid} | CONFIRMATORY, not measurable |")
L.append(f"| r(c) at birth medians {q2['r_at_birth']['median_late_primary']:.3f} / {q2['r_at_birth']['median_all_born']:.3f} | `q2.r_at_birth` | {grid} | CONFIRMATORY, exploratory |")
L.append(f"| B4 m-within-1 {q2['B4_m_within_1']:.3f}, unmatched share {q2['B4_unmatched_share']:.3f}; status counts {json.dumps(q2['status_counts'])} | `q2.B4_m_within_1`, `q2.B4_unmatched_share`, `q2.status_counts` | {grid} | CONFIRMATORY |")
cb = q2["center_birth_curve"]; L.append(f"| center birth curve: median first t {cb['blind_max']['median_first_t']}; by class {json.dumps({k: v['blind_max']['median_first_t'] for k, v in cb['by_class'].items()})}; share non-decreasing {cb['blind_max']['share_nondecreasing']:.3f}; poles {cb['blind_max']['n_poles']}, without texts {cb['n_poles_without_texts']} | `q2.center_birth_curve` | per-step probe scores (16 sha256 in `inputs.scores_sha256`), {grid} | CONFIRMATORY data, exploratory reading |")
g2 = m["G2"]; L.append(f"| G2 = {g2['share']:.3f} ({g2['k_confidence_gt_0.5']} of {g2['n_poles']}) | `G2` | {g2['input']} {sha16(E / g2['input'])} | DISCOVERY (S_gate at T_0) |")
L.append(f"| Holm: b {m['holm']['b']:.4f}, v {m['holm']['v']:.4f} | `holm` | `q2.birth_order.p`, `q2.vestigial.p` (metrics.json {ms}) | CONFIRMATORY |")
L += ["", "Kinds: DISCOVERY = set or tuned on the discovery sets or S_gate before any confirmation number was read; CONFIRMATORY = read once at W5 on the audit sets and S. Reproduce with `runbook/w5_run.sh unblind`, `figures`, `results` after moving `bank/w5` aside (drafts repo; `CENTER=` for the center probe; `all` stops at collect, which refuses on the collected S; handoff/S6.md, Resume steps), then `runbook/results_hand_edits.py` and this script."]
(E / "bank/w5/REGISTRY.md").write_text("\n".join(L) + "\n"); print("REGISTRY.md:", len(L), "lines")
