"""Weight-free offline gate of M2 (studies/next/07_PICKS/M2_OFFLINE_GATE.md). No weight download, no GPU, no AWS
call. CPU torch 2.12.0 and transformers 5.9.0 (the versions of the banked grid harvest).

    python -m infra.aws_pg.jobs.m2_dryrun --bank <BANK/neurons> --hf <HF text reads> --births studies/next/m2/births.json \
        --repeat <the T_14 L40S repeat exemplar_a.npz, m2.REPEAT_T14> --scan studies/next/m2/offline_gate/bank_scan.json \
        --a5 studies/next/m2/offline_gate/a5_gate_sides.json --out <scratch dir>

--hf: a directory with tok/ (the OLMo 3 tokenizer files), config_stage1-step476838.json and the HF reads of the 11
branches the plan loads (config_stage1-stepN.json, index_stage1-stepN.json).
Sections, one CHECK line per check:
  REAL     the laptop banks: sha256 of every input; the dating rule reproduces every banked date; J1's failing input
           moves a date; the pre-check count; the split; the pseudo-birth twins; births.json is reproducible and
           registered; the example selection; the 11 branch configs pass G-CFG and fail on its failing input;
           amendment A1, its revision and A2: the tolerances recomputed from the T_14 device pair, the pass side
           across devices (A2: the cross-device G-EX gates |a_bank| >= 0.1 only), and every failing input of the
           known-number gates failing under the statistic in force. A4: the bank scan's facts (members, shapes,
           edges, every pole with no token of its sign, q999 on the others, no scored unit's own pole degenerate).
           A5: both sides of every gate at every point from the banked files (a5_gate_sides.py): G0 with u x 1.1 as its
           failing input, G-REV, G-CFG, K0, G-EX, G-TOPK, I-ID, the construction control under A (freeze half).
  SCORE    the pure functions on small arrays with known answers (percentile, ranks with ties, matched poles,
           overlap, the pole sign, Wilson, the paired interval).
  JOB      a tiny random Olmo 3 (4 layers, hidden 64, MLP width 96, the real tokenizer) at four planted checkpoints;
           its banks written by neurons_harvest.run itself (topk/, stats/, poles/exemplar_a.npz); then
           m2_score.main from its entry point in a child process; every gate passes and every failing input fails.
           A2/A3: one more run with T_14's bank marked as another device, so the job's cross-device G-EX path runs
           (A3: reported, not gated) while G-TOPK stays a gate. A4: one more run with a competitor of a birth planted
           at T_13 as a pole with no token of its sign: the job completes, B = 0, ranked last, out of the matched set.
           A5: every G0 gate's failing input is u x 1.1 and fails; the activation swap is reported at every point.
  FAIL     each instance gate on a planted failing input, in a child process: the job must exit non-zero naming it.
  VERDICT  freeze and verdict on the tiny job's records; then on synthetic records, one case per branch.
  PROJECT  GPU-hours of the launch from the real window plan.
Ends with M2_DRYRUN_OK when every check holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = []
WINDOW, BATCH = 128, 4
TINY_T = [12, 13, 14, 15]


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:58s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------- REAL
def real(bank: pathlib.Path, hf: pathlib.Path, births_path: pathlib.Path, repeat: pathlib.Path, scan: pathlib.Path,
         a5: pathlib.Path):
    from infra.pg import m1, m2
    from infra.aws_pg.jobs import m2_score as J
    B = m2.load_bank(bank)
    check("REAL bank arrays.npz and births.jsonl sha256 registered", True, "load_bank asserted both")
    ex_ok = all(m2.sha256_file(bank / "w3" / "grid" / m2.POINTS[t][1] / "poles" / "exemplar_a.npz") == m2.EXEMPLAR_SHA[t]
                for t in range(16))
    cx_ok = all(m2.sha256_file(bank / "w3" / "grid" / m2.POINTS[t][1] / "samples" / "contexts.jsonl") == m2.CONTEXTS_SHA[t]
                for t in range(16))
    check("REAL 16 exemplar_a.npz and 16 contexts.jsonl sha256 registered", ex_ok and cx_ok)
    # the arrays in arrays.npz are the 16 exemplar files themselves
    same = all(np.array_equal(np.load(bank / "w3" / "grid" / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"]
                              .astype(np.float64), B["a"][t]) for t in range(16))
    check("REAL arrays.npz a[t] equals exemplar_a.npz of point t (16 of 16)", same)
    n_ok = n_tot = 0
    for r in B["births"]:
        if r["sample"] != 0:
            continue
        i = B["row"][r["pole"]]
        k = int(m2.births_matrix(B["a_plus"][:, i, :], B["q999"][:, i], np.array([r["ranks"]]))[0])
        n_tot += 1
        n_ok += int(k == r["birth_pct"])
    check("REAL the dating rule reproduces every banked date (sample 0)", n_ok == n_tot == 2454, f"{n_ok} of {n_tot}")
    bj = json.loads(births_path.read_text())
    moved = 0
    for b in bj["births"]:
        ap = B["a_plus"][:, b["row"], :].copy()
        ap[b["k"], b["ranks"]] = 0.0                      # J1's failing input: the median at the birth point lowered
        k2 = int(m2.births_matrix(ap, B["q999"][:, b["row"]], np.array([b["ranks"]]))[0])
        moved += int(k2 > b["k"])
    check("REAL failing input: zeroing the birth-point exemplars moves every date later", moved == len(bj["births"]),
          f"{moved} of {len(bj['births'])}")
    check("REAL pre-check: at least 100 passing births", bj["n_births"] >= 100, bj["n_births"])
    check("REAL the count equals 10_RECOUNT2.md §8 (153 later than chance, R1t)", bj["n_births"] == 153)
    check("REAL split: every birth's half is its neuron parity",
          all(b["half"] == ("freeze" if b["neuron"] % 2 == 0 else "test") for b in bj["births"]), bj["halves"])
    check("REAL both halves at or above N_MIN", min(bj["halves"].values()) >= m2.N_MIN, bj["halves"])
    pb_ok = 0
    for b in bj["births"]:
        p = b["pb"]
        if p is None:
            continue
        k = int(m2.births_matrix(B["a_plus"][:, p["row"], :], B["q999"][:, p["row"]], np.array([p["ranks"]]))[0])
        pb_ok += int(k == b["k"] and p["layer"] == b["layer"] and p["neuron"] != b["neuron"] and len(p["ranks"]) ==
                     b["size"] and p["tok"] == B["tok15"][p["row"], p["ranks"]].tolist())
    n_pb = sum(b["pb"] is not None for b in bj["births"])
    check("REAL every pseudo-birth twin: same layer, other neuron, same size, dated k", pb_ok == n_pb, f"{pb_ok} of {n_pb}")
    check("REAL every exemplar position is the carrier's T_15 top-50 position",
          all(b["tok"] == B["tok15"][b["row"], b["ranks"]].tolist() for b in bj["births"]))
    fresh = m2.build_births(bank)
    check("REAL births.json reproduces from the banks (byte-identical JSON)",
          json.dumps(fresh, indent=1) == births_path.read_text())
    check("REAL births.json sha256 is registered in m2_score.BIRTHS_SHA", m2.sha256_file(births_path) == J.BIRTHS_SHA,
          m2.sha256_file(births_path)[:16])
    ex = bj["examples"]
    half = {b["id"]: b["half"] for b in bj["births"]}
    check("REAL examples: 10 freeze-half and 10 test-half ids",
          len(ex["freeze"]) == 10 and len(ex["test"]) == 10 and all(half[i] == "freeze" for i in ex["freeze"])
          and all(half[i] == "test" for i in ex["test"]))
    check("REAL examples: 5 with A_own >= 1 and 5 with A_own = 0",
          sorted(v >= 1 for v in ex["freeze_a_own"].values()) == [False] * 5 + [True] * 5)
    cfg_ok = cfg_fail = 0
    for t in range(5, 16):
        c = json.loads((hf / f"config_{m2.POINTS[t][2]}.json").read_text())
        g = m1.config_gate(c, m1.REF_CONFIG)
        cfg_ok += int(g["pass"] and g["diff_keys"] == [])
        cfg_fail += int(not m1.config_gate({**c, "model_type": "olmo2-retrofit"}, m1.REF_CONFIG)["pass"])
        idx = json.loads((hf / f"index_{m2.POINTS[t][2]}.json").read_text())
        assert sorted(set(idx["weight_map"].values())) == [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    check("REAL G-CFG passes on the 11 branch configs T_5..T_15 (diff empty)", cfg_ok == 11, cfg_ok)
    check("REAL G-CFG failing input (model_type renamed) fails on all 11", cfg_fail == 11, cfg_fail)
    check("REAL the 11 branch indexes name exactly 3 bf16 shards", True, "asserted")
    # K0 and G-EX on the real banks, offline: the bank against itself passes; the registered failing input fails
    grid = bank / "w3" / "grid"
    ex = {t: np.load(grid / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"] for t in range(16)}
    us = sorted((u for u in m2.units(bj["births"]) if u["kind"] == "real"), key=lambda u: u["unit"])
    k0 = []
    for u in us:
        if u["pole"] not in {x["pole"] for x in k0}:
            k0.append(u)
        if len(k0) == J.K0_POLES:
            break
    rows = np.array([u["row"] for u in k0 for _ in u["ranks"]])
    cols = np.array([r for u in k0 for r in u["ranks"]])
    # (new, bank) in the job's order: new = the forward (here the bank of the point itself), bank = the compared bank
    gi = m2.values_gate(ex[15][rows, cols], ex[15][rows, cols], m2.TOL["same"], 10, len(set(rows.tolist())))
    gf = m2.values_gate(ex[15][rows, cols], ex[14][rows, cols], m2.TOL["same"], 10, len(set(rows.tolist())))
    check("REAL K0 offline: the T_15 bank against itself passes", gi["pass"], f"{gi['n']} entries, {gi['n_poles']} poles")
    check("REAL K0 offline: T_14's values at the same entries fail", not gf["pass"],
          f"median {gf['median']:.4f}, p99 {gf['p99']:.4f} against {m2.TOL['same']['p99']:.4f}")
    pl = m2.plan(bj["births"])
    um = {u["unit"]: u for u in pl["units"]}
    worst = []
    for t, p in pl["points"].items():
        ent = sorted({(um[uid]["row"], r) for uid, _ in p["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in ent]), np.array([x[1] for x in ent])
        nb = t + 1 if t < 15 else t - 1
        f = m2.values_gate(ex[t][rr, cc], ex[nb][rr, cc], m2.tol_ex(t, m2.L40S))
        worst.append((t, len(ent), round(f["median"], 3), round(f["p99"], 3), f["pass"]))
    check("REAL G-EX offline: the neighbour point's values fail at all 11 points", not any(w[4] for w in worst),
          [(w[0], w[1], w[2], w[3]) for w in worst])
    # Amendment A1 (2026-11-05): the tolerance recomputes from the banked device pair at T_14; the pass side across
    # devices (the L40S repeat against the Blackwell bank) passes on all entries and on the plan's entries at T_14;
    # T_13's values in T_14's place fail.
    check("REAL A1 the T_14 L40S repeat has its registered sha256", m2.sha256_file(repeat) == m2.REPEAT_T14["sha256"],
          m2.sha256_file(repeat)[:16])
    rep = np.load(repeat)["a"]
    cp = m2.control_p99(ex[14], rep)
    # The revision of A1 (head, 2026-11-05 20:50 CST): across devices twice the p99 over |a_bank| < 0.5; same device as A1.
    check("REAL A1 control p99 recomputes to the registered tolerance", m2.CTRL_P99 == max(cp["all"]["p99"],
          cp["below_1"]["p99"]) and m2.CTRL_P99_X == cp["below_0.5"]["p99"] and
          m2.TOL["same"]["p99"] == m2.CTRL_P99 and m2.TOL["cross"]["p99"] == 2 * m2.CTRL_P99_X,
          f"all {cp['all']['n']}: p99 {cp['all']['p99']:.6f}; |a_bank| < 1, {cp['below_1']['n']}: p99 "
          f"{cp['below_1']['p99']:.6f}; < 0.5, {cp['below_0.5']['n']}: p99 {cp['below_0.5']['p99']:.6f}; < 0.25, "
          f"{cp['below_0.25']['n']}: p99 {cp['below_0.25']['p99']:.6f} (reported); same {m2.TOL['same']['p99']:.6f}, "
          f"cross {m2.TOL['cross']['p99']:.6f}")
    # Amendment A2 (head, 2026-11-06 03:45Z): the cross-device G-EX reads |a_bank| >= 0.1 only; its tolerance is twice
    # the pair's p99 of d on that subset. G-TOPK keeps "cross"; K0 and I-ID keep "same".
    ge = cp[f"ge_{m2.EX_ABS_MIN}"]
    check("REAL A2 control p99 over |a_bank| >= 0.1 recomputes to the registered tolerance",
          m2.CTRL_P99_EX == ge["p99"] and m2.TOL["ex_cross"] == {"p99": 2 * m2.CTRL_P99_EX, "abs_min": 0.1} and
          m2.tol_ex(14, m2.L40S) is m2.TOL["ex_cross"] and m2.tol_ex(15, m2.L40S) is m2.TOL["same"] and
          m2.tol_for(14, m2.L40S) is m2.TOL["cross"],
          f"{ge['n']} entries: median {ge['median']:.6f}, p99 {ge['p99']:.6f}; G-EX cross {m2.TOL['ex_cross']['p99']:.6f}")
    ga = m2.values_gate(rep.ravel(), ex[14].ravel(), m2.tol_ex(14, m2.L40S))
    check("REAL A2 pass side across devices: L40S repeat against the Blackwell bank, all 32,000", ga["pass"],
          f"gated {ga['n_gated']}: median {ga['median']:.5f}, p99 {ga['p99']:.5f}, max {ga['max']:.4f}; below 0.1 "
          f"{ga['below_abs_min']['n']}: p99 {ga['below_abs_min']['p99']:.5f}")
    e14 = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][14]["roles"] for r in um[uid]["ranks"]})
    r14, c14 = np.array([x[0] for x in e14]), np.array([x[1] for x in e14])
    gp = m2.values_gate(rep[r14, c14], ex[14][r14, c14], m2.tol_ex(14, m2.L40S), 1, len(set(r14.tolist())))
    check("REAL A2 pass side across devices: the plan's entries at T_14 (score-01's 477)", gp["pass"] and gp["n"] == 477,
          f"{gp['n']} entries, {gp['n_poles']} poles, gated {gp['n_gated']}: median {gp['median']:.5f}, p99 "
          f"{gp['p99']:.5f}, max {gp['max']:.4f}; below 0.1 {gp['below_abs_min']['n']}: p99 {gp['below_abs_min']['p99']:.5f}")
    g13 = m2.values_gate(rep[r14, c14], ex[13][r14, c14], m2.tol_ex(14, m2.L40S))
    g13a = m2.values_gate(rep.ravel(), ex[13].ravel(), m2.tol_ex(14, m2.L40S))
    check("REAL A2 failing input: T_13's values in T_14's place fail (plan's 477; all 32,000)",
          not g13["pass"] and not g13a["pass"], f"p99 {g13['p99']:.4f} (gated {g13['n_gated']}); {g13a['p99']:.4f} "
          f"(gated {g13a['n_gated']})")
    # values_gate asserts "nothing compared": G-EX reads a non-empty subset at every point, on the bank's side too
    gs = {}
    for t, p in pl["points"].items():
        ent = sorted({(um[uid]["row"], r) for uid, _ in p["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in ent]), np.array([x[1] for x in ent])
        gs[t] = m2.values_gate(ex[t][rr, cc], ex[t][rr, cc], m2.tol_ex(t, m2.L40S))["n_gated"]
    check("REAL G-EX offline: the bank against itself passes at all 11 points (read subset non-empty)", min(gs.values()) > 0,
          gs)
    a4_scan(scan, bj, pl, um)
    a5_real(a5)
    return bj


def a5_real(path: pathlib.Path):
    """Amendment A5: both sides of every gate the job runs, at every point, from the banked files
    (studies/next/m2/offline_gate/a5_gate_sides.py). A gate goes to the instance only if its failing input fails and
    its pass side passes wherever the bank holds the quantity it reads."""
    from infra.pg import m2
    from infra.pg.neurons_harvest import G0_TOL
    from infra.aws_pg.jobs import m2_score as J
    r = json.loads(path.read_text())
    g0 = r["G0"]
    pts = [str(t) for t in range(5, 16)]
    check("REAL A5 the 11 harvest controls.json carry their registered sha256", sorted(g0) == sorted(pts) and all(
        len(g0[t]["controls_sha256"]) == 64 for t in pts), "asserted by a5_gate_sides.py against diag/score05_g0_fail.json")
    check("REAL A5 G0 pass side: harvest (windows 0-7) at 11 points, score-05 (the plan's windows) at T_15-T_6",
          all(g0[t]["pass_side_harvest"]["pass"] for t in pts) and
          all(g0[t]["jobs"]["score-05"]["pass"] for t in pts if t != "5"),
          {t: round(g0[t]["pass_side_harvest"]["max_ratio"], 5) for t in pts})
    check("REAL A5 G0 failing input u x 1.1 fails on every layer at all 11 points (harvest)",
          all(g0[t]["u1.1_harvest"]["fails_every_layer"] and g0[t]["u1.1_harvest"]["min_ratio"] > G0_TOL for t in pts),
          {t: round(g0[t]["u1.1_harvest"]["min_ratio"], 4) for t in pts})
    check("REAL A5 u x 1.1 within 0.1 +- 1.1 x the gate's ratio on every layer (so it fails wherever G0 passes)",
          all(g0[t]["u1.1_bound"]["n_within"] == g0[t]["u1.1_bound"]["n_layers"] == 32 for t in pts),
          f"{sum(g0[t]['u1.1_bound']['n_within'] for t in pts)} of 352 layer-points")
    check("REAL A5 the activation swap has no power at early points (passes at T_5 harvest, T_6 score-05): reported",
          g0["5"]["act_swap_harvest"]["would_pass"] and g0["6"]["jobs"]["score-05"]["act_swap_would_pass"] and
          not any(g0[t]["act_swap_harvest"]["would_pass"] for t in pts if t != "5"),
          f"T_5 {g0['5']['act_swap_harvest']['max_ratio']:.5f}, T_6 {g0['6']['jobs']['score-05']['act_swap_max_ratio']:.5f}")
    rc = r["rev_cfg"]["points"]
    check("REAL A5 G-REV registered = banked manifest = HF ref today, failing input fails; G-CFG both sides (11 points)",
          all(rc[t]["G_REV"]["pass"] and rc[t]["G_REV"]["failing_input_fails"] and rc[t]["G_CFG"]["pass"] and
              rc[t]["G_CFG"]["failing_input_fails"] and rc[t]["G_REV"]["registered"] == m2.POINTS[int(t)][3]
              for t in pts), r["rev_cfg"]["hf_refs"]["sha256"][:16])
    same = [rc[t]["manifest"] for t in ("5", "6", "15")]
    check("REAL A5 the same-device points T_5, T_6, T_15 were banked by one harvest run (device, versions, code)",
          all(m["device_name"] == m2.L40S and {"torch": m["torch"], "transformers": m["transformers"]} == J.VERSIONS and
              m["pg_git_sha"] == same[0]["pg_git_sha"] and m["out"].startswith("/workspace/out/grid_a/") for m in same),
          f"{same[0]['pg_git_sha']}, {[m['out'].split('/')[-1] for m in same]}")
    ex = r["G_EX"]
    check("REAL A5 G-EX gated at T_15, T_6, T_5: failing input fails; T_15 pass side exact in score-02 to score-05",
          [t for t in pts if ex[t]["gated"]] == ["5", "6", "15"] and all(ex[t]["failing_input"]["fails"] for t in pts) and
          all(v["p99"] == 0.0 and v["exact_share"] == 1.0 for v in ex["15"]["jobs"].values()),
          {t: f"{ex[t]['failing_input']['times_tol']:.0f}x" for t in ("15", "6", "5")})
    tk, tj = r["G_TOPK"]["offline"], r["G_TOPK"]["jobs"]
    check("REAL A5 G-TOPK: the offline count reproduces score-05 at T_7; its failing bound lies below the job's values",
          tk["7"]["n_entries"] == tj["7"]["score-05"]["n"] and
          tk["7"]["failing_lower_bound"]["p99"] <= tj["7"]["score-05"]["failing_p99"] and
          tk["7"]["failing_lower_bound"]["median"] <= tj["7"]["score-05"]["failing_median"],
          f"{tk['7']['n_entries']} entries; bound p99 {tk['7']['failing_lower_bound']['p99']:.3f} <= "
          f"{tj['7']['score-05']['failing_p99']:.3f}")
    check("REAL A5 G-TOPK at T_6 and T_5: entries exist and the failing input fails for certain (lower bound)",
          all(tk[t]["n_entries"] > 0 and tk[t]["failing_lower_bound"]["fails_for_certain"] for t in ("6", "5")),
          {t: (tk[t]["n_entries"], round(tk[t]["failing_lower_bound"]["p99"], 3)) for t in ("6", "5")})
    check("REAL A5 G-TOPK at T_15-T_7 in score-05: pass side passes, failing input fails at every point",
          all(tj[t]["score-05"]["pass"] and tj[t]["score-05"]["failing_pass"] is False for t in pts if t not in ("5", "6")))
    ii = r["I_ID"]
    check("REAL A5 I-ID both sides (score-04, score-05): bit-identical, 0 flips; failing input fails",
          all(v["pass"] and v["p99"] == 0.0 and v["carrier_top10_flips_B"] == 0 and not v["failing_pass"]
              for v in ii.values()) and len(ii) == 2)
    c = r["construction_A_freeze"]
    check("REAL A5 construction control under A at T_15, freeze half: >= 0.90; opposite pole below",
          c["pass"] and c["failing_input_fails"] and c["carrier_A_equals_size"] == c["n"] == 82 and
          c["precision_at_10"] >= m2.CTRL_MIN,
          f"precision {c['precision_at_10']:.3f} ({c['n_rank_1']} rank 1, {c['n_rank_2_10']} tied 2-10, max rank "
          f"{c['max_rank']}); opposite {c['opposite_precision_at_10']:.3f}")
    us = r["usable"]
    check("REAL A5 usable births and twins >= N_MIN per half when nothing is parked",
          all(us[h]["usable_if_nothing_parked"] >= m2.N_MIN and us[h]["twins"] >= m2.N_MIN for h in us), us)


def a4_scan(scan_path: pathlib.Path, bj: dict, pl: dict, um: dict):
    """Amendment A4: the facts of the banked layer files that the scoring path assumes, from the bank scan
    (studies/next/m2/offline_gate/bank_scan.py: small members and .npy headers of all 352 stats and 352 topk files
    at T_5-T_15, read by byte ranges from S3)."""
    from infra.pg import m2
    sc = json.loads(scan_path.read_text())
    F = sc["files"]
    want = {f"t{t:02d}_L{lyr:02d}" for t in range(5, 16) for lyr in range(m2.N_LAYERS)}
    N2 = [m2.N_NEURONS, 2]
    shapes_ok = all(
        set(("hist", "hist_edges", "q999", "max_abs", "density", "n_pole_tokens")) <= set(f["stats"]["members"]) and
        {"tok", "a"} <= set(f["topk"]["members"]) and f["stats"]["edges_equal"] and
        all(f["stats"]["shapes"][k]["shape"] == N2 for k in ("max_abs", "n_pole_tokens", "q999", "density")) and
        f["stats"]["shapes"]["hist_edges"]["shape"] == [m2.HIST_B + 1] and
        f["stats"]["headers"]["hist"]["shape"] == N2 + [m2.HIST_B] and
        f["topk"]["headers"]["tok"]["shape"] == f["topk"]["headers"]["a"]["shape"] == N2 + [m2.K_TOP]
        for f in F.values())
    check("REAL A4 scan: 352 stats and 352 topk files at T_5-T_15; members, shapes and histogram edges as the job reads",
          set(F) == want and shapes_ok, f"{len(F)} layer files, {sc['n_calls']} S3 calls")
    c = sc["control"]
    check("REAL A4 scan positive control: the 640 grid poles' max_abs and q999 equal the laptop bank at all 11 points",
          all(c[t][k]["equal"] == c[t][k]["n"] == 640 for t in c for k in ("max_abs", "q999")))
    check("REAL A4 scan failing input: against the neighbouring point's laptop values they differ",
          all(c[t]["max_abs"]["equal_to_neighbour"] == 0 and c[t]["q999"]["equal_to_neighbour"] < 64 for t in c),
          {t: (c[t]["max_abs"]["equal_to_neighbour"], c[t]["q999"]["equal_to_neighbour"]) for t in c})
    deg = [(k, d) for k, f in sorted(F.items()) for d in f["stats"]["degenerate"]]
    check("REAL A4 scan finds the two poles score-04's read found (L24N9826+ at T_10, L07N640- at T_8)",
          ("t10_L24", "L24N9826+") in {(k, d["pole"]) for k, d in deg} and
          ("t08_L07", "L07N640-") in {(k, d["pole"]) for k, d in deg})
    check("REAL A4 every degenerate pole: empty histogram, max_abs 0, q999 NaN, density 0 (consistent)",
          all(d["n_pole_tokens"] == 0 and d["max_abs"] == 0 and d["q999"] is None and d["density"] == 0 for _, d in deg)
          and all(f["stats"]["n_empty"] == f["stats"]["n_dead"] for f in F.values()),
          [f"{k[:3]} {d['pole']}" for k, d in deg])
    check("REAL A4 every other pole: q999 finite and positive, max_abs finite (ratio and ranks need no other rule)",
          all(f["stats"]["other_q999_not_finite_positive"] == 0 and f["stats"]["other_max_abs_not_finite"] == 0
              for f in F.values()))
    own, cand, scored = [], [], {}
    for t, p in pl["points"].items():
        layers = sorted({lyr for w in p["need"].values() for lyr, _, _ in w})
        dset = {d["pole"] for lyr in layers for d in F[f"t{t:02d}_L{lyr:02d}"]["stats"]["degenerate"]}
        own += [(t, uid, role) for uid, role in p["roles"] if um[uid]["pole"] in dset]
        cand.append(min(F[f"t{t:02d}_L{lyr:02d}"]["stats"]["n_other"] for lyr in layers) - 2)
        scored[t] = sorted(dset)
        assert layers == sc["points"][str(t)]["scored_layers"], (t, layers)
    check("REAL A4 no scored unit's own pole is degenerate at its point; every scored layer has >= K_MATCH candidates",
          own == [] and min(cand) >= m2.K_MATCH, f"own-pole cases {own}; fewest candidates {min(cand)}")
    check("REAL A4 degenerate poles in the scored layers (the job's expected metrics.json counts)",
          {t: v for t, v in scored.items() if v} == {10: ["L24N9826+"], 8: ["L07N640-"]},
          {f"T_{t}": len(v) for t, v in scored.items()})


# ------------------------------------------------------------------------------------------------------ SCORE
def score_checks():
    from infra.pg import m2
    edges = (10.0 ** np.linspace(-6, 4, 401)).astype(np.float32)
    hist = np.zeros((1, 2, 400), np.int64)
    hist[0, 0, 240] = 10                  # |a| in [1.0, 1.059)
    hist[0, 0, 280] = 30                  # |a| in [10, 10.59)
    hist[0, 1, 200] = 5
    v = np.array([[1.0, 0.0]])
    p = m2.percentile(v, hist, edges)
    check("SCORE percentile: v at a bin's lower edge counts the bins below", abs(p[0, 0] - 0.0) < 1e-12 and p[0, 1] == 0)
    p2 = m2.percentile(np.array([[10.0 ** 0.0125, 1.0]]), hist, edges)
    check("SCORE percentile: halfway through a bin counts half of it (log scale)", abs(p2[0, 0] - 5 / 40) < 1e-6, p2[0, 0])
    p3 = m2.percentile(np.array([[100.0, 1.0]]), hist, edges)
    check("SCORE percentile: above every token gives 1", abs(p3[0, 0] - 1.0) < 1e-12 and abs(p3[0, 1] - 1.0) < 1e-12)
    rng = np.random.default_rng(3)
    x = 10.0 ** rng.normal(-1, 0.6, 200000)
    b = np.clip(np.floor((np.log10(x) + 6) * 40), 0, 399).astype(int)
    h = np.bincount(b, minlength=400)[None, None, :].repeat(2, 1)
    q = 10.0 ** rng.uniform(-2, 0.5, 50)
    emp = np.array([(x < t).mean() for t in q])
    got = np.array([m2.percentile(np.array([[t, t]]), h, edges)[0, 0] for t in q])
    check("SCORE percentile within one bin's share of the empirical fraction", np.abs(got - emp).max() <= h.max() / x.size,
          f"max diff {np.abs(got - emp).max():.5f}")
    r = m2.ranks(np.array([[3.0, 3.0], [1.0, 0.5]]))
    check("SCORE ranks: ties count against (3, 3, 1, 0.5 -> 2, 2, 3, 4)", r.tolist() == [[2, 2], [3, 4]], r.tolist())
    ma = np.exp(np.linspace(0, 3, 400)).reshape(200, 2)
    m = m2.matched(ma, 50, 0, k=20)
    check("SCORE matched: K poles, the carrier's neuron excluded, nearest first",
          len(m) == 20 and all(n != 50 for n, _ in m) and m[0] in ([49, 1], [51, 0]), m[:2])
    tk = np.full((3, 2, 4), -1)
    tk[0, 0, :2] = [5, 7]
    tk[2, 1, 0] = 7
    sc = m2.scores(np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 0.0]]), tk,
                   {"hist": np.ones((3, 2, 400)), "hist_edges": edges, "q999": np.ones((3, 2)),
                    "max_abs": np.ones((3, 2)), "density": np.ones((3, 2)), "n_pole_tokens": np.full((3, 2), 400)},
                   np.ones(3), [5, 7])
    check("SCORE overlap counts exemplar positions, ignores -1 slots", sc["A"].tolist() == [[2, 0], [0, 0], [0, 1]],
          sc["A"].tolist())
    check("SCORE ratio uses a+ = max(a, 0) for + and max(-a, 0) for -",
          np.allclose(sc["ratio"], [[2.0, 0.0], [0.0, 1.5], [0.25, 0.0]]), sc["ratio"].tolist())
    # Amendment A4: a degenerate pole (no token of its sign: empty histogram, max_abs 0, q999 NaN, top-50 slots empty).
    # Neuron 1's - pole has a+ = 2 and 1 at the two positions (v = 1.5 > 0), so the rule, not v, sets its B to 0.
    a4 = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 0.0]])
    hd = np.ones((3, 2, 400))
    hd[1, 1] = 0
    st = {"hist": hd, "hist_edges": edges, "q999": np.where(hd.sum(-1) > 0, 1.0, np.nan),
          "max_abs": np.where(hd.sum(-1) > 0, 1.0, 0.0), "density": np.ones((3, 2)), "n_pole_tokens": hd.sum(-1)}
    tk4 = tk.copy()
    tk4[1, 1] = -1
    s4 = m2.scores(a4, tk4, st, np.ones(3), [5, 7])
    s0 = m2.scores(a4, tk4, {**st, "hist": np.ones((3, 2, 400)), "n_pole_tokens": np.full((3, 2), 400),
                             "q999": np.ones((3, 2)), "max_abs": np.ones((3, 2))}, np.ones(3), [5, 7])
    others = np.ones((3, 2), bool)
    others[1, 1] = False
    check("SCORE A4 a degenerate pole with v > 0: B = 0, ratio = 0, A = 0, ranked last under B and ratio",
          s4["B"][1, 1] == 0 and s4["ratio"][1, 1] == 0 and s4["A"][1, 1] == 0 and s0["B"][1, 1] > 0 and
          m2.ranks(s4["B"])[1, 1] == 6 and m2.ranks(s4["ratio"])[1, 1] == 6 and
          not any(np.isnan(v).any() for v in s4.values()),
          f"B {s4['B'][1, 1]} (non-degenerate {s0['B'][1, 1]:.3f}), ranks B {m2.ranks(s4['B'])[1, 1]} of 6")
    check("SCORE A4 the other poles' scores are unchanged by the rule",
          all(np.array_equal(s4[k][others], s0[k][others]) for k in ("A", "B", "ratio")))
    ma4 = np.exp(np.linspace(0, 3, 400)).reshape(200, 2)
    d_old = np.abs(np.log(ma4) - np.log(ma4[50, 0]))
    d_old[50, :] = np.inf
    old_order = [[int(i // 2), int(i % 2)] for i in np.argsort(d_old.ravel(), kind="stable")]
    ma4[49, 1] = 0.0
    m4 = m2.matched(ma4, 50, 0, k=20)
    check("SCORE A4 matched: a max_abs 0 pole is not a candidate; the others keep the freeze-1e order",
          [49, 1] in old_order[:20] and [49, 1] not in m4 and m4 == [x for x in old_order if x != [49, 1]][:20],
          f"freeze-1e nearest {old_order[:2]}, A4 nearest {m4[:2]}")
    for name, code, msg in (
            ("degenerate: an empty histogram with max_abs > 0 stops",
             "m2.degenerate({'n_pole_tokens': np.array([[0, 5]]), 'max_abs': np.array([[1.0, 1.0]])})", "disagree"),
            ("matched: a carrier with max_abs 0 stops",
             "m2.matched(np.r_[[[0.0, 1.0]], np.ones((200, 2))], 0, 0, k=20)", "the carrier has no token"),
            ("matched: fewer than K candidates stops",
             "m2.matched(np.r_[np.ones((5, 2)), np.zeros((200, 2))], 0, 0, k=20)", "candidate poles")):
        rc = subprocess.run([sys.executable, "-c", "import numpy as np; from infra.pg import m2; " + code], cwd=REPO_ROOT,
                            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True)
        check(f"SCORE A4 {name}", rc.returncode != 0 and msg in rc.stderr, last_error(rc))
    w = m2.wilson(0, 71)
    check("SCORE Wilson 0 of 71: upper bound 0.0513", abs(w[2] - 0.0513) < 5e-4, w)
    pr = m2.paired(np.array([1] * 30 + [0] * 41), np.array([0] * 71))
    check("SCORE paired interval excludes 0 for 30 of 71 against 0", pr["lo"] > 0 and abs(pr["diff"] - 30 / 71) < 1e-12)


# ------------------------------------------------------------------------------------------------------ JOB
def make_corpus(d: pathlib.Path, hf: pathlib.Path, n_windows: int) -> dict:
    import transformers
    from infra.pg.matrix_models import vocab_sha256
    tok = transformers.AutoTokenizer.from_pretrained(hf / "tok")
    rng = np.random.default_rng(1)
    n = n_windows * WINDOW
    lens = []
    while sum(lens) < n:
        lens.append(int(rng.integers(200, 700)))
    lens[-1] -= sum(lens) - n
    tokens = np.concatenate([rng.integers(1000, 1000 + 40 * (i % 7 + 1), L) for i, L in enumerate(lens)]).astype(np.uint32)
    td = d / "tok" / "olmo3"
    td.mkdir(parents=True, exist_ok=True)
    np.save(td / "tokens.npy", tokens)
    np.save(td / "doc_offsets.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    shutil.copytree(hf / "tok", td / "tokenizer")
    man = {"tokenizers": {"olmo3": {"n_tokens": int(n), "sha256_of_tokens_npy": sha(td / "tokens.npy"),
                                    "vocab_sha256": vocab_sha256(tok), "bos_id": 100257}},
           "bands": {"1": {"tok_end": {"olmo3": int(n)}}}}
    (d / "manifest.json").write_text(json.dumps(man))
    return man


def tiny_cfg(hf: pathlib.Path) -> dict:
    c = json.loads((hf / "config_stage1-step476838.json").read_text())
    c.update(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
             layer_types=c["layer_types"][:4])
    return c


def tiny_model(cfg: dict, seed: int, base=None, noise: float = 0.0):
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(seed)
    m = Olmo3ForCausalLM(Olmo3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type",
                                                                                   "transformers_version")}))
    if base is not None:
        m.load_state_dict({k: v + noise * torch.randn_like(v) * v.std().clamp_min(1e-3)
                           for k, v in base.state_dict().items()})
    return m.eval()


def make_remote(d: pathlib.Path, model, cfg: dict, hf: pathlib.Path):
    import torch
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().clone() for k, v in model.state_dict().items()}
    names = sorted(sd)
    wm = {}
    for i, part in enumerate(np.array_split(np.array(names), 3), 1):
        f = f"model-{i:05d}-of-00003.safetensors"
        save_file({k: sd[k] for k in part}, str(d / f), metadata={"format": "pt"})
        wm.update({str(k): f for k in part})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps(cfg))
    for t in (hf / "tok").iterdir():
        shutil.copy(t, d / t.name)


def fake_hub(remotes: dict, commits: dict, cache: pathlib.Path):
    from huggingface_hub.utils import filter_repo_objects

    def hub(repo, revision, allow_patterns):
        src = pathlib.Path(remotes[revision])
        names = list(filter_repo_objects([p.name for p in src.iterdir()], allow_patterns=allow_patterns))
        snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / commits[revision]
        snap.mkdir(parents=True, exist_ok=True)
        for nm in names:
            if not (snap / nm).exists():
                shutil.copy(src / nm, snap / nm)
        return str(snap)
    return hub


def child(spec_path: str):
    """Entry point of a child process: m2_score.main with the stand-in hub and the tiny world's constants."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    from infra.pg import m2
    import infra.aws_pg.jobs.m2_score as J
    m2.WINDOW, m2.N_WINDOWS = WINDOW, spec["n_windows"]
    m2.N_WINDOWED = WINDOW * spec["n_windows"]
    m2.POINTS = [tuple(p) for p in spec["points"]]
    from infra.pg import neurons_harvest as NH
    if spec.get("plant_swap_noop"):              # A5: the activation swap made the identity (no power anywhere)
        import torch.nn.functional as F
        NH.WRONG_ACT = {"silu": F.silu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh")}
    if "plant_u_scale" in spec:                  # A5: u x 1.0 in place of u x 1.1 (the failing input is the gate)
        NH.WRONG_U_SCALE = spec["plant_u_scale"]
    J.HUB = fake_hub(spec["remotes"], spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    J.main(spec["argv"])


def run_child(scratch: pathlib.Path, name: str, spec: dict, env: dict) -> subprocess.CompletedProcess:
    p = scratch / f"child_{name}.json"
    p.write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import m2_dryrun as D; "
                           "D.child(sys.argv[1])", str(p)], capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def last_error(rc) -> str:
    lines = [x for x in rc.stderr.strip().splitlines() if x.strip()]
    return lines[-1][:160] if lines else ""


def launch(world: dict, name: str, reg_over=None, inputs_dir=None, budget: float = 60.0, cross_t=None, grid_dir=None,
           spec_over=None):
    """m2_score.main in a child process on the tiny world, with an optional planted registry, inputs or grid directory.
    cross_t (A2, A3): that point's bank is marked as made on another device, so its G-EX takes the cross-device path."""
    scratch = world["scratch"]
    rp = scratch / f"registry_{name}.json"
    rp.write_text(json.dumps(dict(world["reg"], **(reg_over or {}))))
    out = scratch / f"out_{name}"
    e = dict(world["env"], HF_HUB_CACHE=str(scratch / f"hf_cache_{name}"))
    argv = ["--out", str(out), "--inputs", str(inputs_dir or world["inputs"]), "--corpus", str(world["corpus"]),
            "--grid", str(grid_dir or world["grid"]), "--budget-min", str(budget), "--registry", str(rp), "--device", "cpu"]
    spec = dict(world["spec"], argv=argv, **(spec_over or {}))
    if cross_t is not None:
        spec["points"] = [p[:4] + ["other-device"] if p[0] == cross_t else p for p in spec["points"]]
    return run_child(scratch, name, spec, e), out


def tiny_births(poles: list, pos15: np.ndarray) -> dict:
    from infra.pg import m2
    out = []
    by_layer = {}
    for i, p in enumerate(poles):
        by_layer.setdefault(m2.parse_pole(p)[0], []).append(i)
    for i, p in enumerate(poles):
        lyr, n, s = m2.parse_pole(p)
        m = 3 + i % 3
        ranks = list(range(m))
        j = [x for x in by_layer[lyr] if m2.parse_pole(poles[x])[1] != n][i % 3]
        lj, nj, sj = m2.parse_pole(poles[j])
        pb = None if i % 5 == 4 else {"pole": poles[j], "layer": lj, "neuron": nj, "sign": sj, "row": j,
                                      "ranks": list(range(10, 10 + m)), "tok": pos15[j, 10:10 + m].tolist()}
        out.append({"id": f"{p}:c0", "pole": p, "layer": lyr, "neuron": n, "sign": s, "c": 0, "name": "tiny",
                    "size": m, "ranks": ranks, "tok": pos15[i, :m].tolist(), "row": i, "k": 13 + i % 3,
                    "half": m2.half_of(n), "p_late": 0.0, "is_main": False, "pb": pb})
    fz = sorted(b["id"] for b in out if b["half"] == "freeze")
    te = sorted(b["id"] for b in out if b["half"] == "test")
    return {"n_births": len(out), "births": out, "examples": {"freeze": fz[:2], "test": te[:2]}}


def job(hf: pathlib.Path, scratch: pathlib.Path) -> pathlib.Path:
    import torch
    import transformers
    from infra.pg import m2
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    n_windows = 64
    corpus_dir = scratch / "corpus"
    make_corpus(corpus_dir, hf, n_windows)
    corpus = NC.load_tok(corpus_dir, "olmo3")
    tok = transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"])
    cfg = tiny_cfg(hf)
    models = {15: tiny_model(cfg, 15)}
    for t, nz in ((14, 0.3), (13, 0.4), (12, 0.5)):
        models[t] = tiny_model(cfg, t, models[15], nz)
    points = [[t, f"{'1' if t in (0, 1, 2, 3, 4, 5, 6, 15) else '1b'}/out/t{t:02d}", f"stage1-step{t}",
               hashlib.sha1(f"stage1-step{t}".encode()).hexdigest(), "cpu"] for t in range(16)]
    poles = [NH.pole_id(lyr, n, s) for lyr in range(4) for n in (3, 17, 40, 63, 90) for s in (0, 1)]
    grid = scratch / "grid"
    remotes, commits = {}, {}
    pos15 = None
    for t in (15, 14, 13, 12):
        d = scratch / "remote" / points[t][2]
        make_remote(d, models[t], cfg, hf)
        remotes[points[t][2]], commits[points[t][2]] = str(d), points[t][3]
        model, _ = NH.load_model(str(d), "O7", device="cpu")
        NH.run(grid / points[t][1], model=model, tok=tok, corpus=corpus, model_key="O7", repo=m2.REPO,
               revision="bank", branch=points[t][2], max_tokens=n_windows * WINDOW, window=WINDOW, batch=BATCH,
               poles=poles, positions=pos15, device="cpu", nll_tokens=1000)
        if t == 15:
            pos15 = np.load(grid / points[t][1] / "poles" / "positions.npz")["tok"]
    inputs = scratch / "inputs"
    inputs.mkdir()
    bj = tiny_births(poles, pos15)
    (inputs / "births.json").write_text(json.dumps(bj))
    for t in range(16):
        src = grid / points[max(t, 12)][1] / "poles" / "exemplar_a.npz"
        shutil.copy(src, inputs / f"exemplar_a_t{t:02d}.npz")
    reg = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "births_sha": sha(inputs / "births.json"),
           "exemplar_sha": [sha(inputs / f"exemplar_a_t{t:02d}.npz") for t in range(16)],
           "tokens_sha256": corpus["tokens_sha256"], "points": {t: p for t, p in enumerate(points)},
           "ref_config": cfg, "window": WINDOW, "batch": BATCH, "bos": 100257, "cpu_device_name": "cpu"}
    spec = {"n_windows": n_windows, "points": points, "remotes": remotes, "commits": commits}
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    world = {"scratch": scratch, "inputs": inputs, "grid": grid, "corpus": corpus_dir, "reg": reg, "spec": spec,
             "env": env, "births": bj}
    rc, out = launch(world, "main")
    check("JOB m2_score.main exits 0 and prints M2_JOB_OK", rc.returncode == 0 and "M2_JOB_OK" in rc.stdout,
          f"exit {rc.returncode}")
    if rc.returncode != 0:
        print(rc.stderr[-4000:])
        raise SystemExit(1)
    jm = json.loads((out / "m2" / "metrics.json").read_text())
    check("JOB metrics complete, no checkpoint parked", jm["complete"] is True and jm["parked"] == [])
    check("JOB every gate passed", all(g["pass"] for g in jm["gates"].values()), sorted(jm["gates"]))
    fi = [n for n, g in jm["gates"].items() if "failing_input" in g]
    check("JOB every registered failing input failed", all(not jm["gates"][n]["failing_input"]["pass"] for n in fi),
          f"{len(fi)} failing inputs")
    k0 = jm["gates"]["K0"]
    check("JOB K0: known number on >= 10 poles before any score", k0["n_poles"] >= 10 and k0["pass"],
          f"poles {k0['n_poles']}, exact share {k0['exact_share']:.3f}, median {k0['median']:.2e}")
    check("JOB K0 failing input (T_14's banked values) fails", not k0["failing_input"]["pass"],
          f"median {k0['failing_input']['median']:.3f}, p99 {k0['failing_input']['p99']:.3f}")
    tk = [g for n, g in jm["gates"].items() if n.endswith("G_TOPK")]
    check("JOB G-TOPK compared banked top-50 entries at every point", all(g["n"] > 0 for g in tk), [g["n"] for g in tk])
    iid = jm["gates"]["I_ID"]
    check("JOB I-ID passes with 0 carrier flips; its failing input fails",
          iid["pass"] and iid["carrier_top10_flips_B"] == 0 and not iid["failing_input"]["pass"],
          f"median {iid['median']:.2e}; failing median {iid['failing_input']['median']:.3f}, "
          f"p99 {iid['failing_input']['p99']:.3f}")
    from infra.pg.neurons_harvest import G0_TOL
    g0 = {n: g for n, g in jm["gates"].items() if n.endswith("_G0")}
    check("JOB A5 every G0 gate's failing input is u x 1.1 and fails on every layer, within the bound",
          len(g0) == 4 and all(g["failing_input"]["u_scale"] == 1.1 and not g["failing_input"]["pass"] and
                               g["failing_input"]["min_ratio"] > G0_TOL and
                               g["failing_input"]["min_ratio"] >= 0.1 - 1.1 * g["max_ratio"] - 1e-5 for g in g0.values()),
          {n: (round(g["max_ratio"], 5), round(g["failing_input"]["min_ratio"], 4)) for n, g in g0.items()})
    sw = {n: g for n, g in jm.get("reports", {}).items() if n.endswith("_G0_act_swap")}
    check("JOB A5 the activation swap is reported at every point, not gated",
          sorted(sw) == sorted(n + "_act_swap" for n in g0) and all("max_ratio" in g for g in sw.values()) and
          not any(n.endswith("act_swap") for n in jm["gates"]), {n: round(g["max_ratio"], 4) for n, g in sw.items()})
    keys = json.dumps(jm)
    check("JOB metrics.json carries no hit rate or rank", '"rank' not in keys and "hit" not in keys)
    recs = {h: m2.read_jsonl(out / "m2" / f"ranks_{h}.jsonl") for h in ("freeze", "test")}
    m2.WINDOW = WINDOW
    plan = m2.plan(bj["births"])
    m2.WINDOW = 2048
    want = sorted((uid, role) for p in plan["points"].values() for uid, role in p["roles"])
    got = sorted((r["unit"], r["role"]) for h in recs for r in recs[h])
    check("JOB one record per planned (unit, role)", want == got, f"{len(got)} records")
    check("JOB records split by the unit's half", all(r["half"] == h for h in recs for r in recs[h]))
    end = [r for h in recs for r in recs[h] if r["role"] == "end" and r["kind"] == "real"]
    ra = [r["rank"]["A"] for r in end]
    check("JOB construction control: every carrier in the top 10 under A at T_15",
          max(ra) <= 10, f"ranks {sorted(set(ra))} (a rank above 1 is a tie with a pole holding the same positions)")
    check("JOB construction control under B: at least 0.9 of carriers in the top 10",
          np.mean([r["rank"]["B"] <= 10 for r in end]) >= 0.9, np.mean([r["rank"]["B"] <= 10 for r in end]))
    check("JOB construction failing input: the opposite pole is not in the top 10 under A",
          np.mean([r["rank_opposite"]["A"] <= 10 for r in end]) < 0.9)
    vec = list((out / "m2" / "vectors").rglob("*.npz"))
    check("JOB activations banked per (unit, role)", len(vec) == len(got), len(vec))
    exs = {h: m2.read_jsonl(out / "m2" / "samples" / f"raw_examples_{h}.jsonl") for h in ("freeze", "test")}
    ok = all(sorted(e["unit"] for e in exs[h]) == sorted(bj["examples"][h]) for h in exs) and all(
        len(e["top10"]["B"]) == 10 and e["top10"]["B"][0]["top3"] and "[[" in e["exemplars"][0]["text"]
        for h in exs for e in exs[h])
    check("JOB raw examples: the selected ids, top-10 lists with decoded contexts", ok)
    # the recomputation from the banked activations equals the record (the independent reader's path)
    r0 = recs["freeze"][0]
    t0 = r0["t"]
    u0 = {u["unit"]: u for u in plan["units"]}[r0["unit"]]
    am = np.load(out / "m2" / "vectors" / f"t{t0:02d}" / f"{r0['unit'].replace(':', '_')}_{r0['role']}.npz")["a"]
    st = np.load(grid / points[t0][1] / "stats" / f"layer_{u0['layer']:02d}.npz")
    tkz = np.load(grid / points[t0][1] / "topk" / f"layer_{u0['layer']:02d}.npz")
    sc = m2.scores(am, tkz["tok"], {k: st[k] for k in st.files}, np.ones(96), u0["tok"])
    n, s = u0["carrier"]
    check("JOB a record recomputes from the banked activations and the grid arrays",
          int(m2.ranks(sc["B"])[n, s]) == r0["rank"]["B"] and int(m2.ranks(sc["A"])[n, s]) == r0["rank"]["A"])
    a4_job(world, out, recs, plan, points)
    # A2/A3: the job's cross-device G-EX path (T_14's bank marked as made on another device): reported, not gated
    rcx, outx = launch(world, "cross14", cross_t=14)
    jx = json.loads((outx / "m2" / "metrics.json").read_text())
    gx, rx = jx["gates"], jx.get("reports", {})
    g14 = rx.get("t14_G_EX", {})
    check("JOB A3 cross-device G-EX at T_14 is reported, not gated (A2 statistic); G-TOPK there a gate; T_13 G-EX a gate",
          rcx.returncode == 0 and "t14_G_EX" not in gx and g14.get("tol") == m2.TOL["ex_cross"] and g14["pass"] and
          0 < g14["n_gated"] <= g14["n"] and "below_abs_min" in g14 and g14["n_above_tol"] == len(g14["above"]) == 0
          and gx["t13_G_EX"]["tol"] == m2.TOL["same"] and gx["t14_G_TOPK"]["tol"] == m2.TOL["cross"]
          and not g14["failing_input"]["pass"] and jx["complete"] is True,
          f"exit {rcx.returncode}; read {g14.get('n_gated')} of {g14.get('n')}, p99 {g14.get('p99')}, above "
          f"{g14.get('n_above_tol')}; failing input p99 {g14.get('failing_input', {}).get('p99')}")
    check("JOB A3 the verdict's line: no reported point over the tolerance", m2.cross_device_gex(jx)["points_over"] == [],
          m2.cross_device_gex(jx)["line"])
    return world, out


def plant_degenerate(grid: pathlib.Path, dst: pathlib.Path, d: str, layer: int, n: int, s: int, topk: bool = True,
                     stats: bool = True):
    """A copy of the tiny world's grid with pole (layer, n, s) at the bank directory d made a pole with no token of its
    sign, as the harvest writes one (empty histogram, max_abs 0, q99/q999/q9999 NaN, density 0, top-50 slots empty).
    stats=False keeps max_abs, q999 and density (an empty histogram with max_abs > 0: the inconsistent case)."""
    shutil.copytree(grid, dst)
    st_p, tk_p = dst / d / "stats" / f"layer_{layer:02d}.npz", dst / d / "topk" / f"layer_{layer:02d}.npz"
    z = dict(np.load(st_p))
    z["hist"][n, s] = 0
    z["n_pole_tokens"][n, s] = 0
    if stats:
        z["max_abs"][n, s], z["density"][n, s] = 0, 0
        for q in ("q99", "q999", "q9999"):
            z[q][n, s] = np.nan
    np.savez_compressed(st_p, **z)
    if topk:
        z = dict(np.load(tk_p))
        z["tok"][n, s] = -1
        for k in ("a", "g", "u"):
            z[k][n, s] = np.nan
        np.savez_compressed(tk_p, **z)
    return dst


def a4_job(world: dict, out: pathlib.Path, recs: dict, plan: dict, points: list):
    """Amendment A4 in the tiny world: one competitor of a birth at T_13 (a pre point, and the identity pair's point)
    planted as a pole with no token of its sign. The job must complete; that pole gets B = 0 and ratio = 0, ranks
    last, leaves the matched set, and every other pole's scores are unchanged."""
    from infra.pg import m2
    grid_neurons = {u["carrier"][0] for u in plan["units"]}
    um = {u["unit"]: u for u in plan["units"]}
    r = next(x for h in ("freeze", "test") for x in recs[h] if x["t"] == 13 and x["role"] == "pre" and x["kind"] == "real")
    u = um[r["unit"]]
    d = points[13][1]
    st0 = np.load(world["grid"] / d / "stats" / f"layer_{u['layer']:02d}.npz")
    tk0 = np.load(world["grid"] / d / "topk" / f"layer_{u['layer']:02d}.npz")
    vec = f"{r['unit'].replace(':', '_')}_pre.npz"
    am0 = np.load(out / "m2" / "vectors" / "t13" / vec)["a"]
    sc0 = m2.scores(am0, tk0["tok"], {k: st0[k] for k in st0.files}, np.ones(96), u["tok"])
    pid = next(p for p in r["matched"] if m2.parse_pole(p)[1] not in grid_neurons
               and sc0["B"][m2.parse_pole(p)[1], m2.parse_pole(p)[2]] > 0)
    _, n, s = m2.parse_pole(pid)
    gd = plant_degenerate(world["grid"], world["scratch"] / "grid_a4", d, u["layer"], n, s)
    rc, outd = launch(world, "a4_degenerate", grid_dir=gd)
    jd = json.loads((outd / "m2" / "metrics.json").read_text())
    gates_ok = all(g["pass"] for g in jd["gates"].values()) and not any(
        g["failing_input"]["pass"] for g in jd["gates"].values() if "failing_input" in g)
    degs = {t: (p["n_degenerate_poles"], p["degenerate_poles"]) for t, p in jd["points"].items()}
    check("JOB A4 a planted degenerate competitor at T_13: the job completes, every gate passes, every failing input fails",
          rc.returncode == 0 and "M2_JOB_OK" in rc.stdout and jd["complete"] is True and gates_ok,
          f"exit {rc.returncode}; {last_error(rc) if rc.returncode else 'complete'}")
    if rc.returncode != 0:
        print(rc.stderr[-4000:])
        raise SystemExit(1)
    check("JOB A4 metrics.json lists the planted pole at T_13 and none elsewhere",
          degs == {k: ((1, [pid]) if k == "13" else (0, [])) for k in degs}, degs)
    rd = next(x for h in ("freeze", "test") for x in m2.read_jsonl(outd / "m2" / f"ranks_{h}.jsonl")
              if x["unit"] == r["unit"] and x["role"] == "pre")
    st1 = np.load(gd / d / "stats" / f"layer_{u['layer']:02d}.npz")
    tk1 = np.load(gd / d / "topk" / f"layer_{u['layer']:02d}.npz")
    am1 = np.load(outd / "m2" / "vectors" / "t13" / vec)["a"]
    sc1 = m2.scores(am1, tk1["tok"], {k: st1[k] for k in st1.files}, np.ones(96), u["tok"])
    nc, scs = u["carrier"]
    oth = np.ones((96, 2), bool)
    oth[n, s] = False
    check("JOB A4 the planted pole: B = 0 and ratio = 0 (B was > 0), ranked last of 192 under both",
          sc1["B"][n, s] == 0 and sc1["ratio"][n, s] == 0 and sc0["B"][n, s] > 0 and
          m2.ranks(sc1["B"])[n, s] == 192 and m2.ranks(sc1["ratio"])[n, s] == 192,
          f"{pid}: B {sc0['B'][n, s]:.4f} -> {sc1['B'][n, s]}, rank under B {m2.ranks(sc0['B'])[n, s]} -> "
          f"{m2.ranks(sc1['B'])[n, s]}")
    check("JOB A4 the planted pole leaves the matched set; the other 99 keep their order",
          pid in r["matched"] and pid not in rd["matched"] and
          rd["matched"][:m2.K_MATCH - 1] == [p for p in r["matched"] if p != pid],
          f"was matched place {r['matched'].index(pid) + 1}; new 100th {rd['matched'][-1]}")
    check("JOB A4 same forward; every other pole's A, B and ratio unchanged; the record recomputes",
          np.array_equal(am0, am1) and all(np.array_equal(sc1[k][oth], sc0[k][oth]) for k in ("A", "B", "ratio")) and
          rd["rank"]["B"] == int(m2.ranks(sc1["B"])[nc, scs]) and rd["rank"]["A"] == int(m2.ranks(sc1["A"])[nc, scs]) and
          not any(p == pid for p, _ in rd["top10"]["B"]),
          f"carrier {u['pole']} rank under B {r['rank']['B']} -> {rd['rank']['B']}, under A {r['rank']['A']} -> "
          f"{rd['rank']['A']}")
    check("JOB A4 the verdict's degenerate-pole line reads the job's count", m2.degenerate_counts(jd)["points"]["13"]["n"]
          == 1 and m2.degenerate_counts(jd)["points"]["15"]["n"] == 0, m2.degenerate_counts(jd)["line"])
    world["a4"] = {"layer": u["layer"], "pole": (n, s), "carrier": (nc, scs), "dir": d, "unit": r["unit"]}


def fails(world: dict):
    """Each instance gate on a planted failing input: the child must exit non-zero, naming the gate."""
    scratch, reg, inputs = world["scratch"], world["reg"], world["inputs"]
    from infra.pg import m2

    def planted_inputs(name, mutate):
        d = scratch / f"inputs_{name}"
        shutil.copytree(inputs, d)
        mutate(d)
        return d, {"births_sha": sha(d / "births.json"), "exemplar_sha": [sha(d / f"exemplar_a_t{t:02d}.npz")
                                                                         for t in range(16)]}

    cases = []
    cases.append(("G_SAME", {"versions": {**reg["versions"], "transformers": "0.0.0"}}, None))
    cases.append(("INPUTS", {"births_sha": "0" * 64}, None))
    cases.append(("G_CORPUS", {"tokens_sha256": "f" * 64}, None))
    pts = {k: list(v) for k, v in reg["points"].items()}
    pts[15][3] = "1" * 40
    cases.append(("t15_G_REV", {"points": pts}, None))
    cases.append(("t15_G_CFG", {"ref_config": {**reg["ref_config"], "intermediate_size": 97}}, None))
    d, over = planted_inputs("k0", lambda d: shutil.copy(d / "exemplar_a_t14.npz", d / "exemplar_a_t15.npz"))
    cases.append(("K0", over, d))
    d, over = planted_inputs("gex14", lambda d: shutil.copy(d / "exemplar_a_t13.npz", d / "exemplar_a_t14.npz"))
    cases.append(("t14_G_EX", over, d))

    def shift(d):
        b = json.loads((d / "births.json").read_text())
        for x in b["births"]:
            x["tok"] = [e + 1 for e in x["tok"]]
        (d / "births.json").write_text(json.dumps(b))
    d, over = planted_inputs("shift", shift)
    cases.append(("K0", over, d))
    for gate, over, d in cases:
        rc, _ = launch(world, f"fail_{gate}_{len(RESULTS)}", over, d)
        err = last_error(rc)
        check(f"FAIL {gate} on its planted input: exit non-zero, gate named", rc.returncode != 0 and gate in err, err)
    # A3: a planted excess at a cross-device point is written and does not stop the job (T_13's bank scaled by 1.2,
    # T_13 marked as made on another device)
    def scale13(d):
        z = dict(np.load(d / "exemplar_a_t13.npz"))
        z["a"] = (z["a"] * 1.2).astype(z["a"].dtype)
        np.savez(d / "exemplar_a_t13.npz", **z)
    d, over = planted_inputs("gex13x", scale13)
    rc, outf = launch(world, "a3_excess_t13", over, d, cross_t=13)
    jf = json.loads((outf / "m2" / "metrics.json").read_text())
    rf = jf.get("reports", {}).get("t13_G_EX", {})
    check("FAIL A3 a planted G-EX excess at a cross-device point (T_13 x 1.2) is written and does not stop the job",
          rc.returncode == 0 and "M2_JOB_OK" in rc.stdout and jf["complete"] is True and "t13_G_EX" not in jf["gates"]
          and rf.get("pass") is False and rf["n_above_tol"] > 0 and len(rf["above"]) == rf["n_above_tol"] and
          m2.cross_device_gex(jf)["points_over"] == [13],
          f"exit {rc.returncode}; p99 {rf.get('p99')}, {rf.get('n_above_tol')} of {rf.get('n_gated')} read entries above; "
          f"{m2.cross_device_gex(jf)['line'][:70]}")
    # A3: a planted G-TOPK excess still stops the job (T_14's banked top-50 values scaled by 1.1, T_14 cross-device)
    gd = scratch / "grid_topk14"
    shutil.copytree(world["grid"], gd)
    for f in (gd / "1b" / "out" / "t14" / "topk").glob("layer_*.npz"):
        z = dict(np.load(f))
        z["a"] = (z["a"] * 1.1).astype(z["a"].dtype)
        np.savez(f, **z)
    rc, _ = launch(world, "fail_t14_G_TOPK_cross", cross_t=14, grid_dir=gd)
    err = last_error(rc)
    check("FAIL A3 a planted G-TOPK excess (T_14 top-50 x 1.1, cross-device) stops the job, gate named",
          rc.returncode != 0 and "t14_G_TOPK" in err, err)
    # A4: fail loud for anything else unexpected. (1) A pole whose histogram is empty while max_abs > 0 stops the job.
    # (2) A carrier with no token of its sign at its pre point stops it (matched has no rule for it; the bank scan
    # shows no scored unit's own pole is degenerate).
    a4 = world["a4"]
    gd = plant_degenerate(world["grid"], scratch / "grid_a4_mixed", a4["dir"], a4["layer"], *a4["pole"], topk=False,
                          stats=False)
    rc, _ = launch(world, "fail_a4_mixed", grid_dir=gd)
    err = last_error(rc)
    check("FAIL A4 an empty histogram with max_abs > 0 stops the job", rc.returncode != 0 and "disagree" in err, err)
    gd = plant_degenerate(world["grid"], scratch / "grid_a4_carrier", a4["dir"], a4["layer"], *a4["carrier"])
    rc, _ = launch(world, "fail_a4_carrier", grid_dir=gd)
    err = last_error(rc)
    check("FAIL A4 a carrier with no token of its sign at its pre point stops the job",
          rc.returncode != 0 and "the carrier has no token" in err, err)
    # A5: (1) the activation swap planted as the identity (no power at any point, as at T_5 and T_6 of the real run):
    # the job completes, the swap is reported under the tolerance, G0's failing input u x 1.1 still fails. (2) u x 1.0
    # planted in place of u x 1.1: the failing input is the gate itself and passes, so the job stops at t15_G0.
    rc, outs = launch(world, "a5_swap_noop", spec_over={"plant_swap_noop": True})
    js = json.loads((outs / "m2" / "metrics.json").read_text())
    sw = {n: g for n, g in js.get("reports", {}).items() if n.endswith("_G0_act_swap")}
    check("FAIL A5 a swap with no power is reported and does not stop the job; u x 1.1 still fails",
          rc.returncode == 0 and "M2_JOB_OK" in rc.stdout and js["complete"] is True and len(sw) == 4 and
          all(g["under_tol"] for g in sw.values()) and
          all(g["pass"] and not g["failing_input"]["pass"] for n, g in js["gates"].items() if n.endswith("_G0")),
          f"exit {rc.returncode}; swap max {max(g['max_ratio'] for g in sw.values()):.5f}")
    rc, _ = launch(world, "a5_u_scale_1", spec_over={"plant_u_scale": 1.0})
    err = last_error(rc)
    check("FAIL A5 u x 1.0 in place of u x 1.1 stops the job at t15_G0 (its failing input passed)",
          rc.returncode != 0 and "t15_G0: its failing input passed" in err, err)
    rc, out = launch(world, "park", budget=0.01)
    jm = json.loads((out / "m2" / "metrics.json").read_text())
    check("FAIL stop rule: a 0.6 s budget parks every point after T_15", rc.returncode == 0 and jm["parked"] == [14, 13, 12]
          and jm["complete"] is True, jm["parked"])
    fz = m2.read_jsonl(out / "m2" / "ranks_freeze.jsonl")
    check("FAIL stop rule: parked births have no pre record (not usable)", m2.usable(fz) == [], len(fz))


# ---------------------------------------------------------------------------------------------------- VERDICT
def synth(root: pathlib.Path, n: dict, spec: dict, seed: int = 0, gates_fail: bool = False):
    """Planted job outputs: per half n births; per score the hit probability of each role (ranks 1 or 5,000)."""
    from infra.pg import m2
    rng = np.random.default_rng(seed)
    (root / "m2").mkdir(parents=True, exist_ok=True)
    for h in ("freeze", "test"):
        lines = []
        for i in range(n[h]):
            bid = f"L01N{2 * i + (h == 'test')}+:c0"

            def rk(p):
                return 1 if rng.random() < p else 5000
            base = {"birth": bid, "half": h, "layer": 1, "carrier": bid[:-3], "value": {}, "top10": {"A": [], "B": []}}
            pre = {**base, "unit": bid, "kind": "real", "role": "pre", "t": 12,
                   "rank": {sc: rk(spec["pre"].get(sc, 0.0)) for sc in m2.SCORES},
                   "matched_rank": {sc: [rk(spec.get("matched", 0.0)) for _ in range(m2.K_MATCH)]
                                    for sc in ("A", "B", "ratio")}}
            lines.append(pre)
            lines.append({**base, "unit": bid, "kind": "real", "role": "birth", "t": 13,
                          "rank": {sc: rk(spec["birth"]) for sc in m2.SCORES}})
            lines.append({**base, "unit": bid, "kind": "real", "role": "end", "t": 15,
                          "rank": {sc: rk(spec["end"]) for sc in m2.SCORES},
                          "rank_opposite": {"A": rk(spec["opp"]), "B": rk(spec["opp"])}})
            if i < n[h] * spec.get("pb_share", 1.0):
                lines.append({**base, "unit": bid + ":pb", "kind": "pb", "role": "pre", "t": 12,
                              "rank": {sc: rk(spec["pb"].get(sc, 0.0)) for sc in m2.SCORES}})
        (root / "m2" / f"ranks_{h}.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    gates = {"K0": {"pass": True, "failing_input": {"pass": False}}, "I_ID": {"pass": True,
             "failing_input": {"pass": gates_fail}}}
    (root / "m2" / "metrics.json").write_text(json.dumps({"complete": True, "gates": gates, "parked": [], "points": {
        str(t): {"n_degenerate_poles": 0, "degenerate_poles": []} for t in (15, 13, 12)}}))
    return root


def verdict_cases(scratch: pathlib.Path, tiny_out: pathlib.Path):
    from infra.pg import m2
    good = {"pre": {"A": 0.3, "B": 0.7, "ratio": 0.6}, "birth": 0.95, "end": 1.0, "opp": 0.0,
            "pb": {"A": 0.1, "B": 0.3}, "matched": 0.0}
    n = {"freeze": 82, "test": 71}

    def run(name, spec, nn=n, gates_fail=False, seed=0):
        root = synth(scratch / "synth" / name, nn, spec, seed, gates_fail)
        th = m2.freeze(root)
        th["_sha256"] = "planted"
        return m2.verdict(root, th), th, root

    v, th, root = run("good", good)
    L = v["labels"]
    # A3: a reported cross-device G-EX over the tolerance is written in the verdict's line and never changes the headline
    mp = root / "m2" / "metrics.json"
    jm = json.loads(mp.read_text())
    jm["reports"] = {"t13_G_EX": {"pass": False, "n": 717, "n_gated": 591, "median": 0.0047, "p99": 0.0519, "max": 0.0766,
                                  "n_above_tol": 7}, "t12_G_EX": {"pass": True, "n": 717, "n_gated": 650, "median": 0.004,
                                                                  "p99": 0.02, "max": 0.05, "n_above_tol": 2}}
    mp.write_text(json.dumps(jm))
    v2 = m2.verdict(root, th)
    check("VERDICT A3 the cross-device G-EX line names the points over the tolerance; the headline is unchanged",
          v2["headline"] == v["headline"] and v2["cross_device_gex"]["points_over"] == [13] and
          v["cross_device_gex"]["points_over"] == [], v2["cross_device_gex"]["line"])
    check("VERDICT satisfying case: H_A SUPPORTED, H_B SUPPORTED, H_AB LOSES, H_sel_B ABOVE",
          (L["H_A"]["label"], L["H_B"]["label"], L["H_AB"]["label"], L["H_sel_B"]["label"]) ==
          ("SUPPORTED", "SUPPORTED", "LOSES", "ABOVE_SELECTION"), v["headline"])
    check("VERDICT the bar is the largest of chance and the two floors", all(
        v["bars"][sc] == max(m2.CHANCE, th["floor_freeze"][sc]["q975"], v["test"][f"floor_{sc}"]["q975"])
        for sc in ("A", "B")), v["bars"])
    v, _, _ = run("a_none", {**good, "pre": {"A": 0.0, "B": 0.7}})
    check("VERDICT A never hits: H_A REFUTED", v["labels"]["H_A"]["label"] == "REFUTED", v["headline"])
    v, _, _ = run("a_eq_b", {**good, "pre": {"A": 0.5, "B": 0.5}}, seed=4)
    check("VERDICT A and B drawn alike: H_AB TIES", v["labels"]["H_AB"]["label"] == "TIES",
          (v["labels"]["H_AB"]["lo"], v["labels"]["H_AB"]["hi"]))
    v, _, _ = run("a_gt_b", {**good, "pre": {"A": 0.8, "B": 0.2}})
    check("VERDICT A above B: H_AB BEATS", v["labels"]["H_AB"]["label"] == "BEATS")
    v, _, _ = run("pb_eq", {**good, "pb": {"A": 0.3, "B": 0.7}}, seed=5)
    check("VERDICT twins drawn like the carriers: H_sel_B AT_SELECTION", v["labels"]["H_sel_B"]["label"] == "AT_SELECTION",
          (v["labels"]["H_sel_B"]["lo"], v["labels"]["H_sel_B"]["hi"]))
    v, _, _ = run("pb_hi", {**good, "pb": {"A": 0.3, "B": 1.0}})
    check("VERDICT twins above the carriers: H_sel_B BELOW_SELECTION", v["labels"]["H_sel_B"]["label"] == "BELOW_SELECTION")
    v, _, _ = run("pb_few", {**good, "pb_share": 0.5})
    check("VERDICT fewer than N_MIN twins: H_sel NOT_RUN", v["labels"]["H_sel_B"]["label"] == "NOT_RUN")
    v, _, _ = run("floor_hi", {**good, "matched": 0.6})
    check("VERDICT a floor above the carrier's value: H_A not SUPPORTED", v["labels"]["H_A"]["label"] != "SUPPORTED",
          (v["labels"]["H_A"]["label"], round(v["bars"]["A"], 3)))
    v, _, _ = run("ceiling", {**good, "pre": {"A": 0.9, "B": 0.95}, "birth": 0.2})
    check("VERDICT pre above the birth point: DOES NOT COUNT until inspected", "until inspected" in v["headline"],
          v["headline"])
    root = synth(scratch / "synth" / "ctrl", n, {**good, "end": 0.5})
    refused = subprocess.run([sys.executable, "-c", f"from infra.pg import m2; m2.freeze('{root}')"], cwd=REPO_ROOT,
                             env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True)
    check("VERDICT construction control at 0.5: freeze refuses", refused.returncode != 0 and "construction control"
          in refused.stderr, last_error(refused))
    v, th, root = run("good2", good)
    recs = m2.read_jsonl(root / "m2" / "ranks_test.jsonl")
    for r in recs:
        if r["role"] == "end":
            r["rank"]["A"] = 5000
    (root / "m2" / "ranks_test.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    v = m2.verdict(root, th)
    check("VERDICT construction control failing on the test half: DOES NOT COUNT", v["headline"].startswith(
        "DOES NOT COUNT") and "construction control A" in v["headline"], v["headline"][:90])
    v, th, root = run("opp", good)
    recs = m2.read_jsonl(root / "m2" / "ranks_test.jsonl")
    for r in recs:
        if r["role"] == "end":
            r["rank_opposite"]["B"] = 1
    (root / "m2" / "ranks_test.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    v = m2.verdict(root, th)
    check("VERDICT the construction failing input passing: DOES NOT COUNT", "failing input passed" in v["headline"])
    v, th, root = run("gate", good)
    jm = json.loads((root / "m2" / "metrics.json").read_text())
    jm["gates"]["I_ID"]["pass"] = False
    (root / "m2" / "metrics.json").write_text(json.dumps(jm))
    v = m2.verdict(root, th)
    check("VERDICT a failed gate: DOES NOT COUNT", v["headline"].startswith("DOES NOT COUNT: gates"), v["headline"][:80])
    root = synth(scratch / "synth" / "fi_pass", n, good, gates_fail=True)
    refused = subprocess.run([sys.executable, "-c", f"from infra.pg import m2; m2.freeze('{root}')"], cwd=REPO_ROOT,
                             env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True)
    check("VERDICT a failing input that passed: freeze refuses", refused.returncode != 0 and "failing input passed"
          in refused.stderr, last_error(refused))
    v, th, root = run("missing", good, nn={"freeze": 82, "test": 30})
    check("VERDICT 30 usable test births: DOES NOT COUNT (missing)", "missing" in v["headline"], v["headline"][:80])
    v, th, root = run("changed", good)
    p = root / "m2" / "ranks_freeze.jsonl"
    p.write_text(p.read_text() + "\n")
    refused = subprocess.run([sys.executable, "-c", "import json; from infra.pg import m2; "
                              f"m2.verdict('{root}', json.loads('{json.dumps(th)}'))"], cwd=REPO_ROOT,
                             env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True)
    check("VERDICT freeze records changed after freeze: verdict refuses", refused.returncode != 0 and
          "freeze records changed" in refused.stderr, last_error(refused))
    rc = m2.room_check()
    check("VERDICT room: each label branch reachable at the test half's size", rc["pass"], rc["labels"])
    # the tiny job's own records through freeze and verdict (20 + 20 births: below N_MIN, so N_MIN is lowered here)
    keep = m2.N_MIN
    m2.N_MIN = 5
    th = m2.freeze(tiny_out)
    th["_sha256"] = "tiny"
    v = m2.verdict(tiny_out, th)
    m2.N_MIN = keep
    check("VERDICT runs on the tiny job's records (N_MIN 5 for the tiny world)", "headline" in v and
          v["test"]["n_usable"] > 0, v["headline"][:100])


# ---------------------------------------------------------------------------------------------------- PROJECT
def project(bj: dict) -> dict:
    from infra.pg import m2
    p = m2.project(bj["births"])
    check("PROJECT the launch at the lowest banked rates fits 0.9 of a 150-minute budget",
          p["gpu_h_low_rate"] <= 0.9 * 150 / 60, f"{p['gpu_h_low_rate']} GPU-h low, {p['gpu_h_measured_rate']} measured")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--births", required=True)
    ap.add_argument("--repeat", required=True)
    ap.add_argument("--scan", required=True, help="A4: studies/next/m2/offline_gate/bank_scan.json")
    ap.add_argument("--a5", required=True, help="A5: studies/next/m2/offline_gate/a5_gate_sides.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scratch = pathlib.Path(a.out)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.environ.update(HF_TOKEN="dryrun-no-network", HF_HUB_CACHE=str(scratch / "hf_cache"),
                      HF_HUB_DISABLE_TELEMETRY="1")
    print("== REAL")
    bj = real(pathlib.Path(a.bank), pathlib.Path(a.hf), pathlib.Path(a.births), pathlib.Path(a.repeat),
              pathlib.Path(a.scan), pathlib.Path(a.a5))
    print("== SCORE")
    score_checks()
    print("== JOB")
    world, tiny_out = job(pathlib.Path(a.hf), scratch)
    print("== FAIL")
    fails(world)
    print("== VERDICT")
    verdict_cases(scratch, tiny_out)
    print("== PROJECT")
    p = project(bj)
    (scratch / "dryrun_summary.json").write_text(json.dumps({"project": p, "checks": RESULTS}, indent=1))
    failed = [nm for nm, ok in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed: {failed}")
    assert not failed, failed
    print("M2_DRYRUN_OK")


if __name__ == "__main__":
    main()
