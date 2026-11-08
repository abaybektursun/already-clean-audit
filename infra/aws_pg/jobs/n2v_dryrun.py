"""Weight-free offline gate of N2V (studies/next/07_PICKS/N2V_OFFLINE_GATE.md). No weight download, no GPU, no AWS call.
CPU torch 2.12.0 and transformers 5.9.0.

    python -m infra.aws_pg.jobs.n2v_dryrun --hf <HF text reads> --sides studies/next/n2v/offline_gate/gate_sides.json \
        --out <scratch dir>

Sections, one CHECK line per check:
  REAL     the registered inputs, the plan from N12's banked records, and the two sides of every gate the banks hold
           (gate_sides.json, written by studies/next/n2v/offline_gate/gate_sides.py).
  SCORE    the pure functions of infra/pg/n2v.py on small arrays with known answers.
  INSTR    the job's row functions on a tiny fp32 Olmo 3: the forward-mode prediction against reverse-mode sum(-h G),
           zero ablation and the 10 % finite difference; the probe row at the rounding floor; G-CAUSAL bitwise; a
           wrong tangent (the neighbouring neuron) as the failing input.
  JOB      N12's tiny world (n12_dryrun.job: M2's and N12's tiny jobs write N12's records), then n2v_score.main from
           its entry point in a child process: every gate passes, every failing input fails, K-A equal to N12's banked
           dL bitwise, the records and raw examples written.
  FAIL     each instance gate on a planted failing input, in a child process: the job exits non-zero naming it; a tiny
           budget parks the later points.
  VERDICT  freeze and verdict end to end on synthetic N2V banks built over N12's real banked records, one case per
           branch of the decision table, including the N12 re-reading on the real records.
  PROJECT  GPU-hours of the launch from the real plan.
Ends with N2V_DRYRUN_OK when every check holds.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from infra.aws_pg.jobs import m2_dryrun as MD
from infra.aws_pg.jobs import n12_dryrun as ND

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = []


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:64s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


# ------------------------------------------------------------------------------------------------------- REAL
def real(sides: pathlib.Path):
    from infra.pg import m2, n2v
    from infra.aws_pg.jobs import n2v_score as J
    up = REPO_ROOT / "studies/next/n12/units.json"
    nb = REPO_ROOT / "studies/next/n12/banks/score-02/n12"
    check("REAL units.json sha256 is the one n2v_score registers", m2.sha256_file(up) == J.UNITS_SHA, J.UNITS_SHA[:12])
    for h in ("freeze", "test"):
        check(f"REAL N12 n12_{h}.jsonl sha256 registered", m2.sha256_file(nb / f"n12_{h}.jsonl") == J.N12_SHA[h],
              J.N12_SHA[h][:12])
    st = json.loads((REPO_ROOT / "infra/aws_pg/steps/next_n2v_score-02.json").read_text())
    check("REAL the manifest runs the registered parts and job budget (core,fd,jvp; 700 min; score-02, freeze 1b)",
          len(st) == 1 and st[0]["module"] == "infra.aws_pg.jobs.n2v_score" and
          "--parts core,fd,jvp --budget-min 700" in st[0]["args"], st[0]["args"][-40:])
    g = json.loads(sides.read_text())
    check("REAL gate_sides.json was computed from the registered N12 records",
          g["inputs"]["n12_freeze"] == J.N12_SHA["freeze"] and g["inputs"]["n12_test"] == J.N12_SHA["test"])
    p = g["plan"]
    check("REAL plan: 153 real units, 987 unit windows, 9 pre points, 328 and 284 pairs",
          p["units"] == 153 and p["unit_windows"] == 987 and len(p["points"]) == 9 and p["pairs"] == {"freeze": 328, "test": 284},
          json.dumps(p["pairs"]))
    uj = json.loads(up.read_text())
    U = uj["units"]
    recs = m2.read_jsonl(nb / "n12_freeze.jsonl") + m2.read_jsonl(nb / "n12_test.jsonl")
    pl = n2v.plan(U, recs)
    real_ids = {u["unit"]: u["half"] for u in U if u["kind"] == "real"}
    check("REAL the keys the job reads from units.json exist in the registered file (N1 A1's lesson): examples of "
          "both halves, each a real unit of its half",
          {"units", "examples"} <= set(uj) and all(len(uj["examples"][h]) == 10 and
                                                   all(real_ids.get(b) == h for b in uj["examples"][h])
                                                   for h in ("freeze", "test")), sorted(uj))
    ok = all(it["poles"][0] == list(next(u for u in U if u["unit"] == it["unit"])["carrier"]) for t in pl for it in pl[t])
    check("REAL plan: every unit's first pole is its carrier; poles in N12's abl order", ok)
    k = g["K_A"]
    check("REAL K-A units are N12's K0 units; the rolled-pole failing input fails with room",
          k["equal_to_N12_K0_units"] and k["failing_input_fails"],
          f"min |diff| {k['failing_input_min_abs_diff']:.4g} against {k['tol']}")
    r = g["reread_inputs"]
    check("REAL the re-reading needs no gradient run: every unit of both halves carries its pull",
          r["records_without_pull"] == 0 and r["real_with_u_matched"] == 153 and r["pb_dt_with_u_S"] == 289, json.dumps(r))
    for h in ("freeze", "test"):
        b = g["halves"][h]
        v = b["G_VAL_on_N12_bf16_measurement"]
        check(f"REAL {h}: G-VAL fails on N12's bf16 measurement (N12's rho reproduces)",
              not v["pass"] and abs(v["rho"] - {"freeze": 0.0996, "test": 0.0425}[h]) < 5e-5, f"rho {v['rho']:.4f}")
        vp = b["G_VAL_planted_fp32_5pct"]
        check(f"REAL {h}: G-VAL passes on a planted fp32 measurement; the sign flip fails",
              vp["pass"] and not vp["failing_input_pass"], f"rho {vp['rho']:.5f}")
        check(f"REAL {h}: above N12's ceiling the inspection clears a 5 % deviation and refuses a copy",
              vp["above_ceiling"] and vp["inspection"]["cleared"] and not b["G_VAL_planted_copy"]["inspection"]["cleared"])
        fd = b["G_FD_planted"]
        check(f"REAL {h}: G-FD passes planted, its copy reads above its ceiling, the sign flip fails",
              fd["pass"] and not fd["above_ceiling"] and b["G_FD_planted_copy"]["above_ceiling"] and not fd["failing_input_pass"])
        rf = b["G_ROOM_failing_input_bf16"]
        check(f"REAL {h}: G-ROOM fails on its failing input (N12's bf16 measured change as the floor)",
              not rf["pass"], f"median rho {rf['median_rho']:.3f}")
        check(f"REAL {h}: G-ROOM passes at a planted floor of 1e-4 nats (window and fd)",
              b["G_ROOM_planted_1e-4"]["pass"] and b["G_ROOM_fd_planted_1e-4"]["pass"])
        check(f"REAL {h}: room holds for a floor up to about 0.017 (window) and 0.0017 nats (fd)",
              b["largest_floor_with_room_window"] > 0.01 and b["largest_floor_with_room_fd"] > 0.001,
              f"{b['largest_floor_with_room_window']:.4f}, {b['largest_floor_with_room_fd']:.5f}")
    rr = g["reread"]
    check("REAL re-reading on N12's records: planted fp32 and cleared gives the gradient labels",
          rr["planted_fp32_cleared"]["headline"].endswith("H_signal REFUTED; H_bridge UNDECIDED; H_dt SUPPORTED"),
          rr["planted_fp32_cleared"]["headline"])
    check("REAL re-reading: N12's bf16 measurement gives N12's verdict (DOES NOT COUNT); not cleared also",
          "H_signal DOES NOT COUNT" in rr["N12_bf16_measurement"]["headline"] and
          "H_signal DOES NOT COUNT" in rr["planted_fp32_not_cleared"]["headline"] and rr["n12_ceiling_restored"])


# ------------------------------------------------------------------------------------------------------- SCORE
def score_checks():
    from infra.pg import n2v
    lc = {3: np.zeros(8, np.float32), 5: np.ones(8, np.float32)}
    rows = {}
    for w in (3, 5):
        for q in range(4):
            rows[("zero", q, w)] = lc[w] + np.float32(0.5 * (q + 1))
            rows[("fd", q, w)] = lc[w] + np.float32(0.05 * (q + 1))
            rows[("jvp", q, w)] = np.full(8, 0.25 * (q + 1), np.float32)
        rows[("probe", 0, w)] = lc[w].copy()
        rows[("probe", 0, w)][7] += np.float32(1e-3)
    pv = n2v.pair_values(lc, rows, {3: [2, 4], 5: [6]})
    want = {"dL": 8.0, "dL_own": 1.5, "dL_fd": 0.8, "dL_fd_own": 0.15, "pred_jvp": 4.0, "pred_jvp_own": 0.75}
    check("SCORE pair_values: window and own sums over two windows", all(abs(pv["pairs"][0][k] - v) < 1e-6
                                                                          for k, v in want.items()), pv["pairs"][0])
    check("SCORE pair_values: the probe's window sum and own sum", abs(pv["probe"]["probe"] - 2e-3) < 1e-6
          and pv["probe"]["probe_own"] == 0.0)
    a = np.arange(10, dtype=np.float32)
    b = a.copy()
    b[6:] += 1
    check("SCORE causal_ok: equal before i0 passes; the comparison through i0 fails",
          n2v.causal_ok(a, b, 6) == (True, False) and n2v.causal_ok(a, b, 7) == (False, False))
    rng = np.random.default_rng(0)
    pr = 10 ** rng.uniform(-3, -1, 200) * rng.choice([-1, 1], 200)
    ps = [{"unit": f"u{i // 4}", "pole": f"p{i}", "rank": i % 4, "pred_bf16": float(x), "dL": float(x), "probe": 1e-5,
           "meas_bf16": 0.6} for i, x in enumerate(pr)]
    check("SCORE room: a tiny floor passes, a floor of 0.1 nats fails",
          n2v.room(ps, "window", reps=100)["pass"] and not n2v.room(ps, "window", noise={p["unit"]: 0.1 for p in ps},
                                                                     reps=100)["pass"])
    check("SCORE bf16_noise: one value per unit from the carrier's banked measurement",
          len(n2v.bf16_noise(ps)) == 50 and set(n2v.bf16_noise(ps).values()) == {0.6})
    val = {"pass": True, "above_ceiling": False, "failing_input": {"pass": False}}
    check("SCORE g_label: PASS, FAIL, no room, failing input passed, above ceiling (cleared, not), NOT_RUN",
          n2v.g_label(val, {"pass": True}) == "PASS" and n2v.g_label({**val, "pass": False}, {"pass": True}) == "FAIL"
          and n2v.g_label(val, {"pass": False}).startswith("DOES NOT COUNT: no room")
          and n2v.g_label({**val, "failing_input": {"pass": True}}, None).startswith("DOES NOT COUNT: failing")
          and n2v.g_label({**val, "above_ceiling": True}, None, {"cleared": True}) == "PASS"
          and n2v.g_label({**val, "above_ceiling": True}, None, {"cleared": False}).startswith("DOES NOT COUNT: above")
          and n2v.g_label(None, None) == "NOT_RUN")
    check("SCORE both halves: PASS only if both; a does-not-count half wins; NOT_RUN kept",
          n2v.both("PASS", "PASS") == "PASS" and n2v.both("PASS", "FAIL") == "FAIL" and
          n2v.both("FAIL", "DOES NOT COUNT: no room").startswith("DOES NOT COUNT") and n2v.both("NOT_RUN", "NOT_RUN") == "NOT_RUN")
    check("SCORE decision table: A, B, C, D",
          [n2v.decision(w, f, "NOT_RUN")["branch"] for w, f in (("PASS", "FAIL"), ("FAIL", "PASS"), ("FAIL", "FAIL"),
                                                                ("DOES NOT COUNT: no room", "PASS"), ("FAIL", "NOT_RUN"))]
          == ["A", "B", "C", "D", "C"])
    ins = n2v.inspect([dict(p, dL=p["pred_bf16"] * 1.01) for p in ps], "window")
    check("SCORE inspect: a 1 % deviation clears, a copy does not",
          ins["cleared"] and not n2v.inspect(ps, "window")["cleared"], f"median rel {ins['median_rel_dev']:.4f}")


# ------------------------------------------------------------------------------------------------------- INSTR
def instr(hf: pathlib.Path):
    import torch
    import torch.nn.functional as F
    from infra.pg import n2v
    from infra.pg import neurons_harvest as NH
    from infra.aws_pg.jobs import n2v_score as J
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = J.CUBLAS_CFG
    cfg = MD.tiny_cfg(hf)
    model = MD.tiny_model(cfg, 15).float()
    model.config._attn_implementation = "sdpa"
    for p in model.parameters():
        p.requires_grad_(False)
    tower = NH.text_tower(model)
    W = 128
    tokens = np.random.default_rng(3).integers(0, 50000, 4 * W).astype(np.int64)
    reg = {"window": W, "bos": 100257, "batch": 4, "batch_jvp": 2}
    lyr, idxs = 1, [20, 50, 51, 90]
    from torch.nn.attention import SDPBackend, sdpa_kernel
    store = {}

    def pre(module, args):
        h = args[0].detach().requires_grad_(True)
        store["h"] = h
        return (h,)
    hd = tower.layers[lyr].mlp.down_proj.register_forward_pre_hook(pre)
    x = torch.cat([torch.full((1, 1), 100257), torch.from_numpy(tokens[W:2 * W])[None]], 1)
    with sdpa_kernel([SDPBackend.MATH]), torch.enable_grad():
        lg = NH.forward(model, x, True).logits
        F.cross_entropy(lg[0, :-1].float(), x[0, 1:], reduction="sum").backward()
    hd.remove()
    allp = (-store["h"].detach()[0, idxs] * store["h"].grad[0, idxs]).sum(0)
    n = int(allp.abs().argmax())                      # the neuron with the largest first-order effect at these indices
    prev = float(allp[n])
    specs = [(1, lyr, idxs, k, n) for k in ("clean", "zero", "fd", "probe")]
    L = J.plain_rows(model, tower, tokens, specs, reg, "cpu")
    i0 = min(idxs)
    cz, cf = n2v.causal_ok(L[0], L[1], i0)
    check("INSTR G-CAUSAL on the tiny model: zero row bitwise before i0, differs through i0", cz and not cf)
    check("INSTR G-CAUSAL: fd and probe rows bitwise before i0",
          n2v.causal_ok(L[0], L[2], i0)[0] and n2v.causal_ok(L[0], L[3], i0)[0])
    d0 = float((L[1].astype(np.float64) - L[0]).sum())
    dfd = float((L[2].astype(np.float64) - L[0]).sum())
    dp = float((L[3].astype(np.float64) - L[0]).sum())
    T = J.jvp_rows(model, tower, tokens, [(1, lyr, idxs, n), (1, lyr, idxs, (n + 1) % allp.numel())], reg, "cpu")
    pj = float(T[0].astype(np.float64).sum())
    check("INSTR forward-mode prediction equals reverse-mode sum(-h G) (fp32)", abs(pj - prev) <= 1e-4 * abs(prev) + 1e-7,
          f"neuron {n}: jvp {pj:.7f} reverse {prev:.7f}; zero ablation {d0:.5f}")
    check("INSTR the neighbouring neuron's tangent differs (failing input)", abs(float(T[1].sum()) - pj) > 1e-3 * abs(pj),
          f"{float(T[1].sum()):.6f}")
    check("INSTR the 10 % finite difference agrees with 0.1 x the prediction to 5 %",
          abs(dfd - 0.1 * pj) <= 0.05 * abs(0.1 * pj), f"fd {dfd:.6f} 0.1 pred {0.1 * pj:.6f}")
    check("INSTR the probe row is at the fp32 floor, far below the zero-ablation change",
          abs(dp) < 1e-3 * abs(d0), f"probe {dp:.2e} zero {d0:.4f}")
    check("INSTR own-loss tangent: the per-position derivative is 0 before i0 (causal)",
          float(np.abs(T[0][:i0]).max()) == 0.0)
    # A1: the alignment that made score-01's G-CAUSAL fail (the arithmetic only: CPU kernels do not reproduce the
    # L40S log-softmax's alignment-dependent order)
    x4, _ = J.M.windows_tensor(tokens, [1] * 4, W, 100257, 4)
    with torch.inference_mode():
        lg = NH.forward(model, x4, True).logits
        offs = [int(lg[b, :-1].data_ptr() % 16) for b in range(4)]
        copies = [lg[b, :-1].to(torch.float32, copy=True) for b in range(4)]
        coffs = [int(c.data_ptr() % 16) for c in copies]
        ce_new = J.ce_pos(lg, x4)
        ce_view = [F.cross_entropy(lg[b, :-1], x4[b, 1:], reduction="none") for b in range(4)]
    check("INSTR A1 alignment: the vocabulary's fp32 row is 8 bytes past a multiple of 16; the row view's start "
          "alternates with the slot; each copy starts 16-byte aligned",
          lg.shape[-1] * 4 % 16 == 8 and offs[0] == offs[2] and offs[1] == offs[3] and offs[0] != offs[1] and
          all(o == 0 for o in coffs), f"vocab {lg.shape[-1]}; view {offs}; copy {coffs}")
    check("INSTR A1 ce_pos (the copy) equals the cross-entropy of the row view bitwise on the CPU",
          all(torch.equal(a, b) for a, b in zip(ce_new, ce_view)))
    sc = J.slot_control(model, tokens, 1, reg, "cpu")
    check("INSTR A1 slot_control runs and reports four slots per path (CPU: every path 0)",
          all(len(sc[k]) == 4 and max(sc[k]) == 0.0 for k in ("logits_max_abs_vs_slot0", "copy_max_abs_vs_slot0",
                                                                  "view_max_abs_vs_slot0")),
          json.dumps({k: sc[k] for k in ("view_offset_mod16_bytes", "copy_max_abs_vs_slot0")}))
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG")


# ------------------------------------------------------------------------------------------------------- JOB
def child(spec_path: str):
    """Entry point of a child process: n2v_score.main with the stand-in hub, the tiny world's constants and plants."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    from infra.pg import m2
    import infra.aws_pg.jobs.m2_score as M
    import infra.aws_pg.jobs.n2v_score as J
    m2.WINDOW, m2.N_WINDOWS = MD.WINDOW, spec["n_windows"]
    m2.N_WINDOWED = MD.WINDOW * spec["n_windows"]
    m2.POINTS = [tuple(p) for p in spec["points"]]
    plant = spec.get("plant")
    op, oj = J.plain_rows, J.jvp_rows
    calls = {"jvp": 0}

    def plain_rows(model, tower, tokens, specs, reg, device, hcap=None):
        if plant == "causal_early":
            specs = [(w, l, [min(ix) - 1] + list(ix), k, n) if k == "zero" else (w, l, ix, k, n) for w, l, ix, k, n in specs]
        out = op(model, tower, tokens, specs, reg, device, hcap)
        if plant == "fwd_second_noisy" and specs and all(s[3] == "clean" for s in specs) and len(specs) > 1:
            out = [v + np.float32(1e-3) for v in out]
        return out

    def jvp_rows(model, tower, tokens, specs, reg, device):
        calls["jvp"] += 1
        out = oj(model, tower, tokens, specs, reg, device)
        if plant == "jvp_second_noisy" and calls["jvp"] == 3:       # T_14 main, T_13 main, T_13 I-JVP
            out = [v * np.float32(1.001) for v in out]
        return out
    J.plain_rows, J.jvp_rows = plain_rows, jvp_rows
    if plant == "precision_high":
        import torch
        orig = torch.set_float32_matmul_precision
        torch.set_float32_matmul_precision = lambda p: orig("high")
    M.HUB = MD.fake_hub(spec["remotes"], spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    J.main(spec["argv"])


def run_child(scratch: pathlib.Path, name: str, spec: dict, env: dict) -> subprocess.CompletedProcess:
    p = scratch / f"n2vchild_{name}.json"
    p.write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import n2v_dryrun as D; "
                           "D.child(sys.argv[1])", str(p)], capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def launch(w: dict, name: str, reg_over=None, inputs=None, plant=None, budget: float = 60.0, parts: str = "core,fd,jvp"):
    scratch = w["scratch"]
    rp = scratch / f"n2vregistry_{name}.json"
    rp.write_text(json.dumps(dict(w["reg"], **(reg_over or {}))))
    out = scratch / f"n2vout_{name}"
    e = dict(w["env"], HF_HUB_CACHE=str(scratch / f"n2vhf_{name}"))
    argv = ["--out", str(out), "--inputs", str(inputs or w["inputs"]), "--corpus", str(w["corpus"]),
            "--budget-min", str(budget), "--registry", str(rp), "--device", "cpu", "--parts", parts]
    spec = dict(w["spec"], argv=argv, plant=plant)
    return run_child(scratch, name, spec, e), out


def job(hf: pathlib.Path, scratch: pathlib.Path):
    from infra.pg import m2
    nw, n12out = ND.job(hf, scratch)
    check("JOB N12's tiny world and N12's tiny job (n12_dryrun.job) checks all pass", all(ok for _, ok in ND.RESULTS),
          f"{sum(ok for _, ok in ND.RESULTS)} of {len(ND.RESULTS)}")
    inputs = scratch / "n2v_inputs"
    inputs.mkdir()
    shutil.copy(nw["n_inputs"] / "units.json", inputs / "units.json")
    for h in ("freeze", "test"):
        shutil.copy(n12out / "n12" / f"n12_{h}.jsonl", inputs / f"n12_{h}.jsonl")
    reg = {**{k: v for k, v in nw["nreg"].items() if k not in ("units_sha", "m2_ranks_sha", "exemplar_sha")},
           "device": "cpu", "units_sha": m2.sha256_file(inputs / "units.json"),
           "n12_sha": {h: m2.sha256_file(inputs / f"n12_{h}.jsonl") for h in ("freeze", "test")}, "batch": 4,
           "batch_jvp": 2}
    w = {"scratch": scratch, "inputs": inputs, "corpus": nw["corpus"], "reg": reg, "env": nw["env"], "spec": nw["spec"]}
    rc, out = launch(w, "main")
    check("JOB n2v_score.main exits 0 and prints N2V_JOB_OK", rc.returncode == 0 and "N2V_JOB_OK" in rc.stdout,
          f"exit {rc.returncode} {MD.last_error(rc)}")
    if rc.returncode != 0:
        print(rc.stderr[-5000:])
        raise SystemExit(1)
    jm = json.loads((out / "n2v" / "metrics.json").read_text())
    check("JOB metrics complete, nothing parked, parts core,fd,jvp", jm["complete"] and jm["parked"] == [] and
          jm["parts"] == ["core", "fd", "jvp"])
    check("JOB every gate passed", all(g["pass"] for g in jm["gates"].values()), sorted(jm["gates"]))
    fi = [k for k, g in jm["gates"].items() if "failing_input" in g]
    check("JOB every registered failing input failed", all(not jm["gates"][k]["failing_input"]["pass"] for k in fi),
          f"{len(fi)} failing inputs")
    ka = jm["gates"]["K_A"]
    check("JOB K-A: N12's bf16 ablation rows re-measured equal N12's banked dL; rolled poles fail",
          ka["pass"] and ka["max_abs_diff"] <= 1e-6 and not ka["failing_input"]["pass"],
          f"{ka['n_pairs']} pairs, {ka['n_exact']} bitwise, max {ka['max_abs_diff']:.2g}; "
          f"failing min {ka['failing_input']['min_abs_diff']:.3g}")
    gc = [g for k, g in jm["gates"].items() if k.endswith("_G_CAUSAL")]
    check("JOB A1 G-CAUSAL records per-row differences at every point: tolerance 0, largest 0, none over, the ten "
          "worst rows with batch and slot",
          len(gc) >= 3 and all(g["tol"] == 0.0 and g["max_abs_before_i0"] == 0.0 and g["n_over_tol"] == 0 and
                               g["n_over_tol_by_slot_parity"] == {"same_as_clean": 0, "other": 0} and
                               0 < len(g["worst10"]) <= 10 and
                               {"unit", "window", "kind", "q", "i0", "max_abs", "batch", "slot", "clean_batch",
                                "clean_slot"} <= set(g["worst10"][0]) for g in gc),
          f"{len(gc)} points; rows {[g['n_rows'] for g in gc]}")
    sc = jm["reports"]["slot_control"]
    check("JOB A1 slot_control reported at the first point (CPU: copy and view 0 at every slot)",
          sc["copy_max_abs_vs_slot0"] == [0.0] * 4 and sc["view_max_abs_vs_slot0"] == [0.0] * 4,
          json.dumps(sc["view_offset_mod16_bytes"]))
    check("JOB A1 I-FWD and I-JVP record their largest difference (0)",
          jm["gates"]["I_FWD"]["max_abs_diff"] == 0.0 and jm["gates"]["I_JVP"]["max_abs_diff"] == 0.0)
    names = list(jm["gates"])
    check("JOB order: K-A before the fp32 cast at T_14; G-CAUSAL at every point; I-FWD and I-JVP at T_13",
          names.index("K_A") < names.index("t14_G_FP32") and all(f"t{t:02d}_G_CAUSAL" in names for t in (12, 13, 14))
          and "I_FWD" in names and "I_JVP" in names)
    pairs = {h: m2.read_jsonl(out / "n2v" / f"n2v_{h}.jsonl") for h in ("freeze", "test")}
    n12r = {h: m2.read_jsonl(n12out / "n12" / f"n12_{h}.jsonl") for h in ("freeze", "test")}
    check("JOB one pair per N12 ablated pair, in both halves",
          all(len(pairs[h]) == sum(len(r["abl"]) for r in n12r[h] if r["kind"] == "real") for h in pairs),
          {h: len(v) for h, v in pairs.items()})
    keys = {"dL", "dL_own", "dL_fd", "dL_fd_own", "pred_jvp", "pred_jvp_own", "probe", "probe_own", "pred_bf16",
            "meas_bf16", "sum_abs_h", "n_later"}
    check("JOB every pair carries every readout", all(keys <= set(p) for h in pairs for p in pairs[h]))
    allp = pairs["freeze"] + pairs["test"]
    pj = np.array([p["pred_jvp"] for p in allp])
    dl = np.array([p["dL"] for p in allp])
    from infra.pg import n12
    check("JOB tiny world: the fp32 zero ablation tracks the fp32 forward-mode prediction",
          n12.spearman(pj, dl) > 0.7, f"rho {n12.spearman(pj, dl):.3f}")
    check("JOB tiny world: the probe is far below the zero-ablation change",
          np.median([abs(p["probe"]) for p in allp]) < 1e-2 * np.median(np.abs(dl)))
    z = np.load(out / "n2v" / "losses" / "t13.npz")
    p0 = [p for p in allp if p["t"] == 13][0]
    ks = [k for k in z.files if k.startswith(p0["unit"] + "|")]
    lc = {int(k.split("|")[1]): z[k] for k in ks if k.endswith("|clean|None")}
    re = sum(float((z[f"{p0['unit']}|{w}|zero|{p0['rank']}"].astype(np.float64) - lc[w]).sum()) for w in lc)
    check("JOB a pair's dL recounts from the banked per-row losses", abs(re - p0["dL"]) < 1e-9, f"{re:.6g}")
    exs = {h: m2.read_jsonl(out / "n2v" / "samples" / f"raw_examples_{h}.jsonl") for h in ("freeze", "test")}
    check("JOB raw examples: the selected births with decoded exemplars, pairs and the largest per-position changes",
          all(len(exs[h]) >= 1 and all(e["exemplars"] and len(e["pairs"]) == 4 and len(e["carrier_zero_top5"]) == 5
                                       for e in exs[h]) for h in exs))
    check("JOB metrics.json carries no Spearman, room or label", "rho" not in json.dumps(jm).replace("throughput", "")
          and "PASS" not in json.dumps(jm))
    # parts core and the finite difference alone (freeze 1's choice, replaced on 2026-11-07 by core,fd,jvp): still a
    # supported path of the job, kept as a check
    rc2, out2 = launch(w, "core_fd", parts="core,fd")
    jm2 = json.loads((out2 / "n2v" / "metrics.json").read_text()) if rc2.returncode == 0 else {}
    p2 = [p for h in ("freeze", "test") for p in m2.read_jsonl(out2 / "n2v" / f"n2v_{h}.jsonl")] if rc2.returncode == 0 else []
    pm = {(p["unit"], p["pole"]): p for p in allp}
    check("JOB parts core,fd (a supported path): exits 0, every gate passes, every failing input fails, no I-JVP, "
          "no forward-mode value; dL, dL_own, dL_fd equal the full run's bitwise",
          rc2.returncode == 0 and jm2["complete"] and jm2["parked"] == [] and jm2["parts"] == ["core", "fd"] and
          all(g["pass"] for g in jm2["gates"].values()) and
          all(not g["failing_input"]["pass"] for g in jm2["gates"].values() if "failing_input" in g) and
          "I_JVP" not in jm2["gates"] and "I_FWD" in jm2["gates"] and len(p2) == len(allp) and
          all("pred_jvp" not in p and "pred_jvp_own" not in p for p in p2) and
          all(all(p[k] == pm[(p["unit"], p["pole"])][k] for k in ("dL", "dL_own", "dL_fd", "dL_fd_own", "probe"))
              for p in p2),
          f"exit {rc2.returncode} {MD.last_error(rc2) if rc2.returncode else ''}; {len(p2)} pairs; "
          f"gates {len(jm2.get('gates', {}))}")
    return w, out


def fails(w: dict):
    cases = [
        ("versions", {"versions": {"torch": "0", "transformers": "0.0.0"}}, None, None, "G_SAME"),
        ("units_sha", {"units_sha": "0" * 64}, None, None, "INPUTS"),
        ("causal_early", None, None, "causal_early", "G_CAUSAL"),
        ("fwd_second_noisy", None, None, "fwd_second_noisy", "I_FWD"),
        ("jvp_second_noisy", None, None, "jvp_second_noisy", "I_JVP"),
        ("precision_high", None, None, "precision_high", "G_FP32"),
    ]
    from infra.pg import m2
    ka_in = w["scratch"] / "n2v_inputs_ka"
    shutil.copytree(w["inputs"], ka_in)
    for h in ("freeze", "test"):
        rs = m2.read_jsonl(ka_in / f"n12_{h}.jsonl")
        for r in rs:
            if r["kind"] == "real":
                r["abl"] = [[a[0], a[1], a[2] + 1e-3] for a in r["abl"]]
        (ka_in / f"n12_{h}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    cases.append(("ka_off", {"n12_sha": {h: m2.sha256_file(ka_in / f"n12_{h}.jsonl") for h in ("freeze", "test")}},
                  ka_in, None, "K_A"))
    for name, ro, inp, plant, gate in cases:
        rc, out = launch(w, name, reg_over=ro, inputs=inp, plant=plant)
        err = MD.last_error(rc)
        check(f"FAIL {name}: the job stops at {gate}", rc.returncode != 0 and gate in err, err[:110])
        if name == "causal_early":
            g = json.loads((out / "n2v" / "metrics.json").read_text())["gates"]["t14_G_CAUSAL"]
            wr = g["worst10"]
            check("FAIL causal_early: G-CAUSAL's per-row record shows the planted rows (count over tolerance, the "
                  "largest difference, the worst rows are zero rows)",
                  g["n_over_tol"] == g["n_rows"] - g["n_bitwise_before_i0"] > 0 and g["max_abs_before_i0"] > 0 and
                  wr[0]["max_abs"] == g["max_abs_before_i0"] and all(x["kind"] == "zero" for x in wr if x["max_abs"] > 0),
                  f"{g['n_over_tol']} of {g['n_rows']} over; max {g['max_abs_before_i0']:.3g}; "
                  f"worst {wr[0]['kind']} batch {wr[0]['batch']} slot {wr[0]['slot']}")
    rc, out = launch(w, "park", budget=0.05)
    jm = json.loads((out / "n2v" / "metrics.json").read_text())
    check("FAIL a tiny budget parks the later points and the job ends normally", rc.returncode == 0 and jm["parked"],
          jm.get("parked"))


# ------------------------------------------------------------------------------------------------------- VERDICT
def verdict_cases(scratch: pathlib.Path):
    from infra.pg import m2, n2v
    nb = REPO_ROOT / "studies/next/n12/banks/score-02"
    th12 = REPO_ROOT / "studies/next/n12/thresholds.json"

    def bank(name, plant, gates_pass=True):
        d = scratch / "vbanks" / name / "n2v"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"complete": True, "gates": {"G": {"pass": gates_pass}}}))
        rng = np.random.default_rng(7)
        for h in ("freeze", "test"):
            rows = []
            for r in m2.read_jsonl(nb / "n12" / f"n12_{h}.jsonl"):
                if r["kind"] != "real":
                    continue
                for i, a in enumerate(r["abl"]):
                    p = {"unit": r["unit"], "half": h, "pole": a[0], "rank": i, "pred_bf16": a[1], "meas_bf16": a[2]}
                    p.update(plant(a[1], rng))
                    rows.append(p)
            (d / f"n2v_{h}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        return d.parent

    def good(pr, g):
        return {"dL": pr * (1 + 0.05 * g.normal()) + 1e-4 * g.normal(), "dL_own": 0.9 * pr * (1 + 0.05 * g.normal()),
                "dL_fd": 0.1 * pr * (1 + 0.01 * g.normal()), "dL_fd_own": 0.09 * pr,
                "pred_jvp": pr * (1 + 0.01 * g.normal()), "pred_jvp_own": 0.9 * pr * (1 + 0.01 * g.normal()),
                "probe": 1e-4 * g.normal(), "probe_own": 1e-5 * g.normal()}

    def noise(pr, g):
        return {**good(pr, g), "dL": 0.03 * g.normal(), "dL_own": 0.03 * g.normal()}

    def run(name, plant, **kw):
        b = bank(name, plant, kw.get("gates_pass", True))
        th = n2v.freeze(b)
        (b / "th.json").write_text(json.dumps(th))
        th["_sha256"] = m2.sha256_file(b / "th.json")
        return th, n2v.verdict(b, th, nb, th12)

    th, v = run("A", good)
    bA = scratch / "vbanks" / "A"
    nt = scratch / "vbanks" / "A_freeze_only" / "n2v"
    nt.mkdir(parents=True)
    for f in ("metrics.json", "n2v_freeze.jsonl"):
        shutil.copy(bA / "n2v" / f, nt / f)
    thf = n2v.freeze(nt.parent)
    check("VERDICT A2 freeze reads the freeze half only: no test file in the bank, the same thresholds as with it",
          json.dumps({k: x for k, x in thf.items() if k != "created_from"}, sort_keys=True) ==
          json.dumps({k: x for k, x in th.items() if k not in ("created_from", "_sha256")}, sort_keys=True))
    check("VERDICT branch A: fp32 measurement tracks the banked prediction; N12 re-read with its frozen rules",
          v["decision"]["branch"] == "A" and v["n12_reread"]["labels"]["H_signal"]["label"] == "REFUTED" and
          v["n12_reread"]["ceiling_cleared_by_inspection"], v["headline"][:150])
    th, v = run("B", noise)
    check("VERDICT branch B: window fails with room, the 10 % finite difference passes",
          v["decision"]["branch"] == "B" and v["labels"]["window"] == "FAIL" and v["labels"]["fd"] == "PASS", v["headline"])
    th, v = run("C", lambda pr, g: {**noise(pr, g), "dL_fd": 0.003 * g.normal()})
    check("VERDICT branch C: window and finite difference fail: the instrument is retired",
          v["decision"]["branch"] == "C" and v["labels"]["fd"] == "FAIL", v["headline"])
    th, v = run("D", lambda pr, g: {**good(pr, g), "probe": 0.5 * g.normal(), "probe_own": 0.5 * g.normal()})
    check("VERDICT branch D: a floor of 0.5 nats leaves no room: DOES NOT COUNT",
          v["decision"]["branch"] == "D" and v["labels"]["window"].startswith("DOES NOT COUNT: no room"), v["headline"])
    th, v = run("copy", lambda pr, g: {**good(pr, g), "dL": pr})
    check("VERDICT a measurement equal to the prediction is above the ceiling and not cleared",
          v["labels"]["window"].startswith("DOES NOT COUNT: above ceiling") and v["decision"]["branch"] == "D")
    th, v = run("core", lambda pr, g: {k: x for k, x in noise(pr, g).items() if k in ("dL", "dL_own", "probe", "probe_own")})
    check("VERDICT parts core only: fd and own read NOT_RUN; a failing window gives branch C",
          v["labels"]["fd"] == "NOT_RUN" and v["labels"]["own"] == "NOT_RUN" and v["decision"]["branch"] == "C")
    jvp_keys = ("pred_jvp", "pred_jvp_own")
    _, va = run("coreA", lambda pr, g: {k: x for k, x in good(pr, g).items() if k not in jvp_keys})
    _, vb = run("coreB", lambda pr, g: {k: x for k, x in noise(pr, g).items() if k not in jvp_keys})
    check("VERDICT parts core,fd (a supported path): own and window_jvp read NOT_RUN; branches A and B reachable",
          all(v["labels"]["own"] == "NOT_RUN" and v["labels"]["window_jvp"] == "NOT_RUN" for v in (va, vb)) and
          va["decision"]["branch"] == "A" and vb["decision"]["branch"] == "B" and vb["labels"]["fd"] == "PASS",
          f"{va['headline'][:90]} | {vb['headline']}")
    b = bank("gates", good, gates_pass=False)
    try_ok = True
    r = subprocess.run([sys.executable, "-c", f"from infra.pg import n2v; n2v.freeze('{b}')"], capture_output=True,
                       text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT freeze refuses when a job gate failed", r.returncode != 0 and "freeze refused" in r.stderr, try_ok)
    b = bank("roomfail", lambda pr, g: {**good(pr, g)})
    for h in ("freeze",):
        f = b / "n2v" / f"n2v_{h}.jsonl"
        rows = m2.read_jsonl(f)
        f.write_text("".join(json.dumps(dict(x, meas_bf16=1e-6)) + "\n" for x in rows))
    r = subprocess.run([sys.executable, "-c", f"from infra.pg import n2v; n2v.freeze('{b}')"], capture_output=True,
                       text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT freeze refuses when G-ROOM's failing input passes", r.returncode != 0 and "failing input" in r.stderr)
    b = scratch / "vbanks" / "A"
    th = json.loads((b / "th.json").read_text())
    with open(b / "n2v" / "n2v_freeze.jsonl", "a") as f:
        f.write(" ")
    r = subprocess.run([sys.executable, "-c", "import json; from infra.pg import n2v; "
                        f"n2v.verdict('{b}', json.load(open('{b / 'th.json'}')), '{nb}', '{th12}')"],
                       capture_output=True, text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT the verdict refuses if the freeze records changed", r.returncode != 0 and "freeze records changed" in r.stderr)


def project():
    from infra.pg import m2, n2v
    U = json.loads((REPO_ROOT / "studies/next/n12/units.json").read_text())["units"]
    nb = REPO_ROOT / "studies/next/n12/banks/score-02/n12"
    recs = m2.read_jsonl(nb / "n12_freeze.jsonl") + m2.read_jsonl(nb / "n12_test.jsonl")
    out = {p: n2v.project(U, recs, tuple(p.split(","))) for p in ("core", "core,fd", "core,fd,jvp")}
    full = out["core,fd,jvp"]
    check("PROJECT the full launch from the real plan and measured rates", full["unit_windows"] == 987,
          {p: (o["gpu_h_center"], o["gpu_h_high"], o["budget_min_hard_kill"]) for p, o in out.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--sides", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scratch = pathlib.Path(a.out)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    print("== REAL")
    real(pathlib.Path(a.sides))
    print("== SCORE")
    score_checks()
    print("== INSTR")
    instr(pathlib.Path(a.hf))
    print("== JOB")
    w, _ = job(pathlib.Path(a.hf), scratch)
    print("== FAIL")
    fails(w)
    print("== VERDICT")
    verdict_cases(scratch)
    print("== PROJECT")
    pj = project()
    n_ok = sum(ok for _, ok in RESULTS)
    summary = {"checks": len(RESULTS), "passed": n_ok, "failed": [n for n, ok in RESULTS if not ok],
               "projection": {p: {k: o[k] for k in ("gpu_h_center", "gpu_h_high", "budget_min_hard_kill")}
                              for p, o in pj.items()}}
    (scratch / "dryrun_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if n_ok == len(RESULTS):
        print("N2V_DRYRUN_OK")
    else:
        raise SystemExit(f"{len(RESULTS) - n_ok} checks failed")


if __name__ == "__main__":
    main()
