"""Weight-free offline gate of N1 (studies/next/07_PICKS/N1_OFFLINE_GATE.md). No weight download, no GPU, no AWS call.

    python -m infra.aws_pg.jobs.n1_dryrun --hf <HF text reads> --sides studies/next/n1/offline_gate/gate_sides.json \
        --out <scratch dir>

Sections, one CHECK line per check:
  REAL     the unit table (reproducible from N12's units and M2's ranks), the onset pre-check (k reproduces for every
           unit), K-B's inputs and both sides from the banks (gate_sides.json), the room of the labels.
  SCORE    the pure functions of infra/pg/n1.py on small arrays with known answers (r, robust z, the excess and its
           windows, the floor, the labels, the hypotheses).
  JOB      a tiny world of 16 checkpoints (a tiny Olmo 3, 4 layers, MLP width 96, saved as bf16 safetensors in 3
           shards like the real ones; one neuron's input rows given a planted change in one bracket): n1_weights.main
           from its entry point in a child process; every gate passes, every failing input fails; the statistics equal
           an independent numpy recount from the safetensors; the planted neuron reads high on the input side only.
           A1 (2026-11-07): the tiny units.json carries the real file's keys only ("units", "n"); the example births
           come from the registry, as on the instance.
  FAIL     each instance gate on a planted failing input, in a child process: the job exits non-zero naming it.
           K-B is reported, not gated (the head, 2026-11-06): a bank moved by 0.1 % is reported failing at every
           banked point and the job completes.
  VERDICT  freeze and verdict on synthetic records, one case per branch.
  PROJECT  instance-hours of the launch.
Ends with N1_DRYRUN_OK when every check holds.
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

from infra.aws_pg.jobs import m2_dryrun as MD

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = []
PLANT = (1, 5, 10)          # the tiny world's planted event: layer 1, neuron 5, input rows changed in bracket 10


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:64s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


# ------------------------------------------------------------------------------------------------------- REAL
def real(sides: pathlib.Path):
    from infra.pg import m2, n1
    n12u = json.loads((REPO_ROOT / "studies/next/n12/units.json").read_text())["units"]
    rk = [r for h in ("freeze", "test") for r in m2.read_jsonl(REPO_ROOT / f"studies/next/m2/banks/score-06/m2/ranks_{h}.jsonl")]
    again = n1.units(n12u, rk)
    cur = json.loads((REPO_ROOT / "studies/next/n1/units.json").read_text())
    check("REAL n1 units.json is reproducible from N12's units and M2's ranks", again == cur, json.dumps(cur["n"]))
    check("REAL units: real 82/71, dt 73/67, pb 79/70", cur["n"] == {"real_freeze": 82, "real_test": 71, "dt_freeze": 73,
                                                                      "dt_test": 67, "pb_freeze": 79, "pb_test": 70})
    U = cur["units"]
    check("REAL every real unit has at least 98 matched neurons of its layer, none its own",
          all(len(u["matched"]) >= 98 and u["neuron"] not in u["matched"] for u in U if u["kind"] == "real"))
    on = json.loads((REPO_ROOT / "studies/next/n1/onsets/onsets.json").read_text())
    check("REAL onset pre-check: the dating rule reproduces k for all 442 units", all(r["k"] == r["k_unit"] for r in on["units"])
          and len(on["units"]) == 442)
    s = on["summary"]
    conc = {k: v["concentrated"] for k, v in s.items()}
    check("REAL onset pre-check: most births rise over several brackets (concentrated in fewer than half)",
          s["real_freeze"]["concentrated"] + s["real_test"]["concentrated"] < 0.5 * 153, conc)
    check("REAL every unit's window and reference brackets exist (k >= 6)", min(u["k"] for u in U) >= 6 and
          all(n1.reference(u["k"]) and n1.window(u["k"])[0] > n1.reference(u["k"])[-1] for u in U))
    g = json.loads(sides.read_text())
    cp = g["control_pairs_bank_path"]
    check("REAL K-B bank path reproducible: M2, N12 and N6 column norms bitwise equal (three L40S runs)",
          all(v["max_abs"] == 0.0 and v["n_bitwise"] == v["n"] for v in cp.values()), {k: v["n"] for k, v in cp.items()})
    f = g["K_B"]["failing_input_adjacent_point"]
    check("REAL K-B failing input (the neighbouring point) fails for every banked neuron, with room",
          f["n_within_tol"] == 0 and f["min_rel"] > 10 * n1.KB_TOL, f"min rel {f['min_rel']:.5f} against {n1.KB_TOL}")
    check("REAL kb_bank.json and the N6 array are the registered inputs",
          g["kb_bank"]["sha256"] == m2.sha256_file(REPO_ROOT / "studies/next/n1/kb_bank.json") and g["n6_t15"]["lfs_oid_registered"])
    check("REAL room: every label and hypothesis branch reachable at 67 pairs", g["room"]["pass"])
    from infra.aws_pg.jobs import n1_weights as J
    ex = json.loads((REPO_ROOT / "studies/next/n12/units.json").read_text())["examples"]
    byid = {u["unit"]: u for u in U}
    check("REAL A1: the job's registered example births are N12's, each a real unit of its half in N1's units.json",
          J.registry()["examples"] == n1.EXAMPLES == {h: ex[h] for h in ("freeze", "test")} and
          all(byid[b]["kind"] == "real" and byid[b]["half"] == h for h in n1.EXAMPLES for b in n1.EXAMPLES[h]),
          {h: len(n1.EXAMPLES[h]) for h in n1.EXAMPLES})


# ------------------------------------------------------------------------------------------------------- SCORE
def score_checks():
    from infra.pg import m2, n1
    norms = np.ones((3, 1, 4, 3))
    norms[..., 2] = 2.0
    diff = np.full((2, 1, 4, 3), 0.1)
    diff[1, 0, 2, 0] = 0.4
    r = n1.side_r(norms, diff)
    check("SCORE side_r: in combines gate and up; out divides by the column norm",
          abs(r["in"][0, 0, 0] - 0.1) < 1e-12 and abs(r["out"][0, 0, 0] - 0.05) < 1e-12 and
          abs(r["in"][1, 0, 2] - np.sqrt(0.17 / 2)) < 1e-12)
    rng = np.random.default_rng(0)
    rr = np.exp(rng.normal(size=(15, 2, 1000)))
    z = n1.robust_z(rr)
    check("SCORE robust_z: median 0 and MAD 1/1.4826 per bracket and layer", np.abs(np.median(z, -1)).max() < 1e-12 and
          np.abs(np.median(np.abs(z), -1) - 1 / 1.4826).max() < 1e-9)
    zz = np.zeros((15, 1, 3))
    zz[9:12, 0, 1] = 2.0
    zz[1:8, 0, 1] = 0.5
    check("SCORE excess: window mean minus reference mean (k = 12: window 10-12, reference 2-9)",
          n1.window(12) == [10, 11, 12] and n1.reference(12) == list(range(2, 10)) and
          abs(n1.excess(zz, 0, 1, 12, n1.window(12)) - (2.0 - 0.5 * 7 / 8)) < 1e-12)
    me = np.zeros((50, 10))
    me[:, 0] = 1.0
    fl = n1.floor_reps(me, reps=500)
    check("SCORE floor: replicate means of one matched neuron per unit", 0.05 < fl["mean"] < 0.15 and fl["q975"] > fl["mean"])
    check("SCORE hypotheses from the two selection labels",
          [n1.hypothesis(a, b) for a, b in ((n1.SEL[0], n1.SEL[1]), (n1.SEL[1], n1.SEL[0]), (n1.SEL[0], n1.SEL[0]),
                                            (n1.SEL[2], n1.SEL[1]), (n1.SEL[1], n1.SEL[1]))]
          == ["ROWS_FIRST", "COLUMN_FIRST", "TOGETHER", "NEITHER", "NEITHER (H_0: both at selection)"])
    check("SCORE label_sign on paired intervals", n1.label_sign(m2.paired(np.ones(60), np.zeros(60))) == n1.SEL[0] and
          n1.label_sign(m2.paired(np.zeros(60), np.ones(60))) == n1.SEL[2])


# ------------------------------------------------------------------------------------------------------- JOB
def tiny_world(hf: pathlib.Path, scratch: pathlib.Path) -> dict:
    import torch
    from infra.pg import m2
    cfg = MD.tiny_cfg(hf)
    models = {0: MD.tiny_model(cfg, 0)}
    torch.manual_seed(1)
    for t in range(1, 16):
        m = MD.tiny_model(cfg, t, models[t - 1], 0.0)
        sd = m.state_dict()
        for k, v in sd.items():
            if v.dim() == 2:
                sd[k] = v + 0.02 * torch.randn_like(v) * v.std()
        lyr, n, kp = PLANT
        if t == kp:
            for nm in ("gate_proj", "up_proj"):
                w = sd[f"model.layers.{lyr}.mlp.{nm}.weight"]
                w[n] = w[n] + 0.5 * torch.randn_like(w[n]) * w[n].norm() / w[n].numel() ** 0.5
        m.load_state_dict(sd)
        models[t] = m
    points = [[t, f"x/t{t:02d}", f"stage1-step{t}", hashlib.sha1(f"stage1-step{t}".encode()).hexdigest(), "cpu"]
              for t in range(16)]
    remotes, commits = {}, {}
    for t in range(16):
        d = scratch / "remote" / points[t][2]
        MD.make_remote(d, models[t], cfg, hf)
        remotes[points[t][2]], commits[points[t][2]] = str(d), points[t][3]
    bad = scratch / "remote" / "transposed"
    shutil.copytree(scratch / "remote" / points[3][2], bad)
    from safetensors.torch import load_file, save_file
    idx = json.loads((bad / "model.safetensors.index.json").read_text())["weight_map"]
    nm = "model.layers.0.mlp.down_proj.weight"
    sd = load_file(str(bad / idx[nm]))
    sd[nm] = sd[nm].t().contiguous()
    save_file(sd, str(bad / idx[nm]), metadata={"format": "pt"})
    N, L = 96, 4
    rng = np.random.default_rng(5)
    U = []
    spec_units = [(1, 5, 10), (0, 8, 8), (2, 11, 12), (3, 20, 14), (1, 30, 12), (2, 41, 15), (0, 52, 9), (3, 63, 11)]
    for i, (lyr, n, k) in enumerate(spec_units):
        half = m2.half_of(n)
        b = f"L{lyr:02d}N{n}+:c0"
        base = {"half": half, "birth": b, "layer": lyr, "sign": 0, "k": k, "d": 3, "size": 3}
        others = [x for x in range(N) if x != n]
        U.append({**base, "unit": b, "kind": "real", "neuron": n, "pole": f"L{lyr:02d}N{n}+",
                  "matched": [int(x) for x in rng.choice(others, 20, replace=False)]})
        for kind, tn in (("dt", others[(7 * i + 3) % len(others)]), ("pb", others[(11 * i + 5) % len(others)])):
            U.append({**base, "unit": f"{b}:{kind}", "kind": kind, "neuron": int(tn), "pole": f"L{lyr:02d}N{tn}+"})
    fz = sorted(u["birth"] for u in U if u["kind"] == "real" and u["half"] == "freeze")
    te = sorted(u["birth"] for u in U if u["kind"] == "real" and u["half"] == "test")
    inputs = scratch / "inputs"
    inputs.mkdir()
    # A1 (2026-11-07): the real units.json carries only "units" and "n"; the example births come from the registry
    (inputs / "units.json").write_text(json.dumps({"units": U, "n": {f"{k}_{h}": sum(u["kind"] == k and u["half"] == h
                                                                                     for u in U)
                                                                     for k in ("real", "dt", "pb")
                                                                     for h in ("freeze", "test")}}))
    kb, n6 = [], np.zeros((L, N), np.float32)
    for t in range(5, 16):
        sdt = models[t].state_dict()
        for lyr in range(L):
            cn = sdt[f"model.layers.{lyr}.mlp.down_proj.weight"].to(torch.bfloat16).float().norm(dim=0).numpy()
            if t == 15:
                n6[lyr] = cn
            for u in U:
                if u["layer"] == lyr and t in (u["k"] - 1, u["k"]):
                    kb.append([lyr, u["neuron"], t, float(cn[u["neuron"]])])
    kb = [list(x) for x in {tuple(x) for x in kb}]
    (inputs / "kb_bank.json").write_text(json.dumps({"values": sorted(kb)}))
    np.save(inputs / "n6_colnorms_t15.npy", n6)
    reg = {"versions": {"torch": torch.__version__.split("+")[0]}, "units_sha": m2.sha256_file(inputs / "units.json"),
           "kb_sha": m2.sha256_file(inputs / "kb_bank.json"), "n6_sha": m2.sha256_file(inputs / "n6_colnorms_t15.npy"),
           "points": {t: p for t, p in enumerate(points)}, "n_layers": L, "n_neurons": N, "hidden": 64,
           "examples": {"freeze": fz[:2], "test": te[:2]}}
    return {"scratch": scratch, "inputs": inputs, "reg": reg, "remotes": remotes, "commits": commits, "points": points,
            "env": dict(os.environ, PYTHONPATH=str(REPO_ROOT)), "U": U}


def child(spec_path: str):
    spec = json.loads(pathlib.Path(spec_path).read_text())
    from infra.pg import m2
    import infra.aws_pg.jobs.m2_score as M
    import infra.aws_pg.jobs.n1_weights as J
    m2.POINTS = [tuple(p) for p in spec["points"]]
    plant = spec.get("plant")
    if plant == "iid_second":                    # the second read of T_15 (the 17th read) differs in one tensor
        orig, calls = J.read_mlp, {"n": 0}

        def read_mlp(local, n_layers):
            calls["n"] += 1
            out = orig(local, n_layers)
            if calls["n"] == 17:
                g, u, d = out[0]
                out[0] = (g, u, d * 1.01)        # 1.0001 rounds back to the same bf16 value
            return out
        J.read_mlp = read_mlp
    remotes = dict(spec["remotes"])
    if plant == "transposed":
        remotes[spec["points"][3][2]] = spec["transposed"]
    M.HUB = MD.fake_hub(remotes, spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    J.main(spec["argv"])


def launch(w: dict, name: str, reg_over=None, inputs=None, plant=None, budget: float = 60.0):
    scratch = w["scratch"]
    rp = scratch / f"n1registry_{name}.json"
    rp.write_text(json.dumps(dict(w["reg"], **(reg_over or {}))))
    out = scratch / f"n1out_{name}"
    e = dict(w["env"], HF_HUB_CACHE=str(scratch / f"n1hf_{name}"))
    argv = ["--out", str(out), "--inputs", str(inputs or w["inputs"]), "--budget-min", str(budget), "--registry", str(rp)]
    spec = {"argv": argv, "points": w["points"], "remotes": w["remotes"], "commits": w["commits"], "plant": plant,
            "transposed": str(scratch / "remote" / "transposed")}
    p = scratch / f"n1child_{name}.json"
    p.write_text(json.dumps(spec))
    rc = subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import n1_dryrun as D; D.child(sys.argv[1])",
                         str(p)], capture_output=True, text=True, env=e, cwd=REPO_ROOT)
    return rc, out


def job(hf: pathlib.Path, scratch: pathlib.Path) -> dict:
    import torch
    from safetensors.torch import load_file
    from infra.pg import m2
    w = tiny_world(hf, scratch)
    real_keys = sorted(json.loads((REPO_ROOT / "studies/next/n1/units.json").read_text()))
    check("JOB A1: the tiny units.json has the real units.json's top-level keys (no examples key)",
          sorted(json.loads((w["inputs"] / "units.json").read_text())) == real_keys, real_keys)
    rc, out = launch(w, "main")
    check("JOB n1_weights.main exits 0 and prints N1_JOB_OK", rc.returncode == 0 and "N1_JOB_OK" in rc.stdout,
          f"exit {rc.returncode} {MD.last_error(rc)}")
    if rc.returncode != 0:
        print(rc.stderr[-5000:])
        raise SystemExit(1)
    jm = json.loads((out / "n1" / "metrics.json").read_text())
    check("JOB every gate passed", jm["complete"] and all(g["pass"] for g in jm["gates"].values()), sorted(jm["gates"])[:6])
    fi = [k for k, g in jm["gates"].items() if "failing_input" in g]
    check("JOB every registered failing input failed", all(not jm["gates"][k]["failing_input"]["pass"] for k in fi),
          f"{len(fi)} failing inputs")
    kbs = {k: g for k, g in jm.get("reports", {}).items() if k.endswith("K_B")}
    kb_pts = {x[2] for x in json.loads((w["inputs"] / "kb_bank.json").read_text())["values"]} | {15}
    check("JOB K-B reported (not a gate) at every banked point: fp64 norms within the bound of the fp32 bank; "
          "the neighbour fails",
          sorted(kbs) == sorted(f"t{t:02d}_K_B" for t in kb_pts) and all(g["max_rel"] <= 1.3e-4 and g["pass"]
                                                                         for g in kbs.values()) and
          all(not g["failing_input"]["pass"] for g in kbs.values() if "failing_input" in g) and
          not any(k.endswith("K_B") for k in jm["gates"]),
          {k: f"{g['max_rel']:.1e}" for k, g in list(kbs.items())[:3]})
    pcs = {k: g for k, g in jm["gates"].items() if k.endswith("_PC")}
    check("JOB PC at brackets 2-15: planted q99-size changes land above q95; size 0 does not",
          len(pcs) == 14 and all(g["in"] >= 0.9 and g["out"] >= 0.9 and g["in_size0"] < 0.5 for g in pcs.values()))
    check("JOB I-ID at T_15: norms bitwise on a second read; self change 0; T_14 against T_15 is not",
          jm["gates"]["I_ID"]["pass"] and not jm["gates"]["I_ID"]["failing_input"]["pass"])
    z = np.load(out / "n1" / "stats.npz")
    ok, mx = True, 0.0
    for t in (0, 9, 10, 15):
        d = pathlib.Path(w["remotes"][w["points"][t][2]])
        idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        sd = {}
        for f in sorted(set(idx.values())):
            sd.update(load_file(str(d / f)))
        dp = pathlib.Path(w["remotes"][w["points"][t - 1][2]]) if t else None
        if dp is not None:
            ip = json.loads((dp / "model.safetensors.index.json").read_text())["weight_map"]
            sp = {}
            for f in sorted(set(ip.values())):
                sp.update(load_file(str(dp / f)))
        for lyr in range(4):
            g = sd[f"model.layers.{lyr}.mlp.gate_proj.weight"].to(torch.float64).numpy()
            dn = sd[f"model.layers.{lyr}.mlp.down_proj.weight"].to(torch.float64).numpy()
            mx = max(mx, np.abs(np.linalg.norm(g, axis=1) - z["norms"][t, lyr, :, 0]).max(),
                     np.abs(np.linalg.norm(dn, axis=0) - z["norms"][t, lyr, :, 2]).max())
            if dp is not None:
                pd_ = sp[f"model.layers.{lyr}.mlp.down_proj.weight"].to(torch.float64).numpy()
                mx = max(mx, np.abs(np.linalg.norm(dn - pd_, axis=0) - z["diff"][t - 1, lyr, :, 2]).max())
    check("JOB statistics equal an independent numpy recount from the safetensors (fp64)", mx < 1e-10, f"max {mx:.1e}")
    recs = {h: m2.read_jsonl(out / "n1" / f"n1_{h}.jsonl") for h in ("freeze", "test")}
    allr = {r["unit"]: r for h in recs for r in recs[h]}
    pl = allr["L01N5+:c0"]
    others = [r["in"]["e_window"] for u, r in allr.items() if u != "L01N5+:c0"]
    check("JOB the planted neuron reads high on the input side and not on the output side",
          pl["in"]["e_window"] > 3 * max(abs(x) for x in others) and abs(pl["out"]["e_window"]) < pl["in"]["e_window"] / 5,
          f"in {pl['in']['e_window']:.2f}, out {pl['out']['e_window']:.2f}, others max {max(abs(x) for x in others):.2f}")
    check("JOB records: one per unit, with the profile, matched neurons' excess and the ceiling",
          len(allr) == len(w["U"]) and all("matched_e_window" in r for r in allr.values() if r["kind"] == "real") and
          all(len(r["in"]["r"]) == 15 for r in allr.values()))
    exs = {h: m2.read_jsonl(out / "n1" / "samples" / f"raw_examples_{h}.jsonl") for h in ("freeze", "test")}
    check("JOB raw examples: the example births with the real unit and its twins", all(
        len(exs[h]) == 2 and all("real" in e and "dt" in e and "pb" in e for e in exs[h]) for h in exs))
    keys = json.dumps(jm)
    check("JOB metrics.json carries no excess, z or label", "e_window" not in keys and "ROWS" not in keys)
    return w


def fails(w: dict):
    from infra.pg import m2
    kb_in = w["scratch"] / "inputs_kb"
    shutil.copytree(w["inputs"], kb_in)
    kb = json.loads((kb_in / "kb_bank.json").read_text())
    kb["values"] = [[a, b, t, v * 1.001] for a, b, t, v in kb["values"]]
    (kb_in / "kb_bank.json").write_text(json.dumps(kb))
    pts = [list(p) for p in w["points"]]
    pts_bad = [p if i != 2 else [p[0], p[1], p[2], "0" * 40, p[4]] for i, p in enumerate(pts)]
    cases = [("versions", {"versions": {"torch": "0"}}, None, None, "G_SAME"),
             ("units_sha", {"units_sha": "0" * 64}, None, None, "INPUTS"),
             ("revision", {"points": {i: p for i, p in enumerate(pts_bad)}}, None, None, "t02_G_REV"),
             ("transposed", None, None, "transposed", "t03_G_SHAPE"),
             ("iid_second", None, None, "iid_second", "I_ID")]
    for name, ro, inp, plant, gate in cases:
        rc, _ = launch(w, name, reg_over=ro, inputs=inp, plant=plant)
        err = MD.last_error(rc)
        check(f"FAIL {name}: the job stops at {gate}", rc.returncode != 0 and gate in err, err[:110])
    rc, out = launch(w, "kb_off", reg_over={"kb_sha": m2.sha256_file(kb_in / "kb_bank.json")}, inputs=kb_in)
    rep = json.loads((out / "n1" / "metrics.json").read_text()).get("reports", {}) if rc.returncode == 0 else {}
    kbr = {k: g for k, g in rep.items() if k.endswith("K_B")}
    n_pts = len({x[2] for x in kb["values"]} | {15})
    check("FAIL kb_off: the bank moved by 0.1 %: K-B reports a failure at every banked point; the job "
          "completes (K-B is reported, not gated)", rc.returncode == 0 and len(kbr) == n_pts and
          all(not g["pass"] for g in kbr.values()), f"exit {rc.returncode}; {len(kbr)} reports; "
          f"min max_rel {min((g['max_rel'] for g in kbr.values()), default=0):.1e}")


# ------------------------------------------------------------------------------------------------------- VERDICT
def synth(n: int, d_in: float, d_out: float, seed: int, half: str, real_level: float = 0.0, n_dt=None) -> list:
    """Synthetic records: a per-birth component shared by the real unit and its twins (sd 0.5), independent noise
    (sd 0.2), and the real unit shifted by d_in, d_out."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = f"b{half}{i}"
        common = rng.normal(scale=0.5, size=4)
        for kind, lv in (("real", (real_level + d_in, real_level + d_out)), ("dt", (real_level, real_level)),
                         ("pb", (real_level, real_level))):
            if kind == "dt" and n_dt is not None and i >= n_dt:
                continue
            r = {"unit": b if kind == "real" else f"{b}:{kind}", "kind": kind, "half": half, "birth": b, "k": 12}
            for j, (s, base) in enumerate((("in", lv[0]), ("out", lv[1]), ("gate", lv[0]), ("up", lv[0]))):
                e = base + common[j] + rng.normal(scale=0.2)
                r[s] = {"e_window": e, "e_k": e, "e_by_bracket": {str(x): e for x in (10, 11, 12, 13)}}
            if kind == "real":
                r["matched_e_window"] = {s: list(rng.normal(scale=0.5, size=99)) for s in ("in", "out")}
                r["ceiling"] = {"in": 3.0, "out": 3.0}
            out.append(r)
    return out


def verdict_cases(scratch: pathlib.Path):
    from infra.pg import m2, n1

    def run(name, fz, te, gates_pass=True):
        b = scratch / "vbanks" / name / "n1"
        b.mkdir(parents=True)
        (b / "metrics.json").write_text(json.dumps({"complete": True, "gates": {"G": {"pass": gates_pass}}}))
        for h, rs in (("freeze", fz), ("test", te)):
            (b / f"n1_{h}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        th = n1.freeze(b.parent) if gates_pass else None
        if th is None:
            return None, b.parent
        th["_sha256"] = "x"
        return n1.verdict(b.parent, th), b.parent

    fz = synth(82, 0.0, 0.0, 1, "freeze")
    for name, di, do, want in (("rows", 0.8, 0.0, "ROWS_FIRST"), ("col", 0.0, 0.8, "COLUMN_FIRST"),
                               ("both", 0.8, 0.8, "TOGETHER"), ("none", 0.0, 0.0, "NEITHER (H_0: both at selection)"),
                               ("below", -0.8, 0.0, "NEITHER")):
        v, _ = run(name, fz, synth(71, di, do, 2, "test"))
        check(f"VERDICT {name}: {want}", v["hypothesis"] == want and not v["headline"].startswith("DOES NOT COUNT"),
              v["headline"][:120])
    v, _ = run("floor", fz, synth(71, 0.8, 0.0, 3, "test", real_level=0.0))
    check("VERDICT the floor label reads the real mean against the matched floor's q975",
          v["labels"]["floor_in"]["label"] == "ABOVE_FLOOR" and v["labels"]["floor_out"]["label"] == "NOT_ABOVE_FLOOR")
    v, _ = run("ceiling", fz, synth(71, 4.0, 0.0, 4, "test"))
    check("VERDICT a side above its ceiling: DOES NOT COUNT until inspected", v["headline"].startswith("DOES NOT COUNT")
          and "ceiling" in v["headline"])
    v, _ = run("fewdt", fz, synth(71, 0.8, 0.0, 5, "test", n_dt=30))
    check("VERDICT fewer than 50 dt pairs: no selection control, DOES NOT COUNT", v["headline"].startswith("DOES NOT COUNT")
          and "dt pairs" in v["headline"])
    _, b = run("gates", fz, synth(71, 0.8, 0.0, 6, "test"), gates_pass=False)
    r = subprocess.run([sys.executable, "-c", f"from infra.pg import n1; n1.freeze('{b}')"], capture_output=True, text=True,
                       cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT freeze refuses when a job gate failed", r.returncode != 0 and "freeze refused" in r.stderr)
    b = scratch / "vbanks" / "rows"
    th = n1.freeze(b)
    with open(b / "n1" / "n1_freeze.jsonl", "a") as f:
        f.write(" ")
    r = subprocess.run([sys.executable, "-c", "import json; from infra.pg import n1; "
                        f"n1.verdict('{b}', {json.dumps({'sha256_n1_freeze': th['sha256_n1_freeze']})})"],
                       capture_output=True, text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT the verdict refuses if the freeze records changed", r.returncode != 0 and "freeze records changed" in r.stderr)


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
    print("== JOB")
    w = job(pathlib.Path(a.hf), scratch)
    print("== FAIL")
    fails(w)
    print("== VERDICT")
    verdict_cases(scratch)
    print("== PROJECT")
    from infra.pg import n1
    pj = n1.project()
    check("PROJECT instance-hours of the CPU launch", pj["gpu_h"] == 0.0, {k: pj[k] for k in ("instance_h_center",
                                                                                              "instance_h_high",
                                                                                              "budget_min_hard_kill")})
    n_ok = sum(ok for _, ok in RESULTS)
    summary = {"checks": len(RESULTS), "passed": n_ok, "failed": [n for n, ok in RESULTS if not ok], "projection": pj}
    (scratch / "dryrun_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if n_ok == len(RESULTS):
        print("N1_DRYRUN_OK")
    else:
        raise SystemExit(f"{len(RESULTS) - n_ok} checks failed")


if __name__ == "__main__":
    main()
