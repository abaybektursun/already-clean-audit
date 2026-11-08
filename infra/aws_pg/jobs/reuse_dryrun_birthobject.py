"""Weight-free dry run of SPEC8 BIRTHOBJECT (studies/reuse/SPEC8_birthobject.md §9). No model and no download: tiny NumPy tensors only.

    python -m infra.aws_pg.jobs.reuse_dryrun_birthobject --out <dir>

A synthetic bank with the layout `reuse_dating --stage dating+scale --abar` writes, at T_0..T_15: <run>/t<nn>/metrics.json (stage dating, the job's
gate records, every case), dating/<pole>__c<c>.npz (t, pole, side, tok, gP, uP, aP), and at T_15 scale/, abar.npz, high_background.json,
background_1m.npz and samples/prompts.jsonl; <run>/t<nn>/scale_stage/metrics.json (stage scale), scale/<pole>.npz, abar.npz and background.json.
24 cases on 12 poles (two clusters per pole), 8 per depth tercile, births T_4..T_13. Every a_P is sign silu(g) u of float32 g and u. Construction:
g_l(t) = c_l (3 - 2 t / 15); s_P(t) = c_P (0.5 + t / 30) with the harvest bank's T_14 at 0.8 of its T_15; O*'s contexts carry 30% noise per load.
Event cases (even index): J's contexts are 3 times higher from t_b on (1% noise), so the break of log J/O* is t_b - 1, every J context jumps at
t_b - 1 (J's concentration 1) and the largest log-rate per log-step is the pair (t_b - 1, t_b). Drift cases (odd): J ramps by exp(0.12 t) with 30% noise.
Checks: verdict_birthobject.py stops on a bank without t13/scale_stage/abar.npz (exit 4, nothing written); is green (exit 0) with those known answers
and E3's ratios equal to an independent recomputation from the construction; exits 1 under each failing input with its target failed (every rule
"not read" for the gate targets; E1 and E4 inside their nulls for dates_observed_shuffled). Prints REUSE_DRYRUN_BIRTHOBJECT_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
VERDICT = REPO / "studies/reuse/scripts/verdict_birthobject.py"
TOKENS = json.loads((REPO / "infra/pg/olmo3_tokens.json").read_text())
TS = list(range(16))
GRID = [f"t{t:02d}_step{g['step']}" for t, g in zip(TS, TOKENS["grid"])]
N_UP, W, N_E1, N_SCALE, N_J, N_O = 31, 17, 200, 50, 8, 7
LAYERS = (2, 14, 25)
MODES = {"known_shifted": "known_number", "capture_wrong_layer": "capture_identity", "scale_other_pole": "scale",
         "global_other_checkpoint": "global_scale", "dates_observed_shuffled": None}


def silu64(g):
    return g / (1 + np.exp(-g))


def gu_for(a_target: np.ndarray, sign: float, rng) -> tuple:
    """Float32 g and u whose sign silu(g) u (float64 from the float32 values) is close to a_target; returns g, u and that exact a_P."""
    g = rng.uniform(0.5, 2.0, a_target.shape).astype(np.float32)
    u = (a_target / (sign * silu64(g.astype(np.float64)))).astype(np.float32)
    return g, u, sign * silu64(g.astype(np.float64)) * u.astype(np.float64)


def dump(p: pathlib.Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def gate_records(pole: str, job: str) -> dict:
    """The job's three per-load gate records (unite_stats.gate_known / gate_complete and reuse_dating.Job.identity's fields), passing."""
    return {"known_number": {"ratio": 0.001, "wrong_ratio": 0.4, "tol": 0.01, "pass": True, "failing_input_failed": True, "per_pole": {}, "n_ctx": 16,
                             "wrong_rule": "the previous grid checkpoint's exemplar_a", "failing_input": "a_wrong: the previous grid checkpoint's exemplar_a"},
            "completeness": {"median_err_over_pole_max": 0.002, "wrong_layer": 0.3, "tol": 0.01, "pass": True, "failing_input_failed": True},
            "capture_identity": {"pole": pole, "job": job, "n_contexts": 20, "max_abs_diff": 0.0, "tol": 1e-5, "pass": True,
                                 "failing_input": "P's neuron read at layer L-1 (the hook on L-1)", "failing_input_max_abs_diff": 1.7, "failing_input_failed": True,
                                 "a_P_abs_max": 3.0}}


def build(out: pathlib.Path) -> dict:
    rng = np.random.default_rng(0)
    cases, construction = [], {}
    tok = iter(range(1000, 10 ** 6, 3))
    poles = [f"L{LAYERS[k % 3]:02d}N{100 + k}{'+' if k % 2 == 0 else '-'}" for k in range(12)]
    pole_scale = {p: 1.0 + 0.1 * k for k, p in enumerate(poles)}
    scale_tok = {p: [next(tok) for _ in range(N_SCALE)] for p in poles}
    sfac = np.array([0.5 + t / 30 for t in TS])
    bank_fac = sfac.copy()
    bank_fac[14] = 0.8 * bank_fac[15]
    for i in range(24):
        p = poles[i // 2]
        L, n = int(p[1:3]), int(p[4:-1])
        cases.append({"job": f"{p}:c{1 + i % 2}", "pole": p, "c": 1 + i % 2, "layer": L, "neuron": n, "sign": p[-1], "tercile": int(L >= 11) + int(L >= 22),
                      "birth": 4 + (3 * i) % 10, "t_before": 3 + (3 * i) % 10, "set": "third", "J_tok": [next(tok) for _ in range(N_J)],
                      "O_tok": [next(tok) for _ in range(N_O)], "scale_tok": scale_tok[p],
                      "s_P": {f"T_{t}": float(pole_scale[p] * bank_fac[t]) for t in TS}})
    for c in cases:
        c["J_ranks"], c["O_ranks"] = list(range(N_J)), list(range(N_J, N_J + N_O))
    items = {"cases": cases, "known": {"poles": [], "tok": [], "a_bank": {}, "a_wrong": {}, "wrong_rule": "synthetic"}, "grid_ts": TS,
             "sources": {"grid": GRID}, "counts": {"cases": len(cases)}}
    ip = out / "birthobject_items.json"
    dump(ip, items)
    isha = hashlib.sha256(ip.read_bytes()).hexdigest()

    cl = np.array([0.002 * (1 + l / 10) for l in range(N_UP)])
    gt = np.array([3 - 2 * t / 15 for t in TS])
    G = cl[None, :] * gt[:, None]                                         # [t, layer]
    spread = np.linspace(0.5, 1.5, W)                                     # median 1.0 (W odd)
    aJ, aO = {}, {}
    for i, c in enumerate(cases):
        sg = 1.0 if c["sign"] == "+" else -1.0
        s = pole_scale[c["pole"]] * sfac
        cJ, cO = rng.uniform(0.5, 1.5, N_J), rng.uniform(0.5, 1.5, N_O)
        if i % 2 == 0:
            step = np.where(np.array(TS) >= c["birth"], 3.0, 1.0)
            aJ[i] = s[:, None] * cJ[None] * step[:, None] * (1 + 0.01 * rng.uniform(-1, 1, (16, N_J)))
        else:
            aJ[i] = s[:, None] * cJ[None] * np.exp(0.12 * np.array(TS))[:, None] * (1 + 0.3 * rng.uniform(-1, 1, (16, N_J)))
        aO[i] = s[:, None] * cO[None] * (1 + 0.3 * rng.uniform(-1, 1, (16, N_O)))
        construction[c["job"]] = {"event": i % 2 == 0, "sign": sg}

    run = out / "run"
    exact = {"J": {}, "O": {}}
    for k, t in enumerate(TS):
        d, ds = run / f"t{t:02d}", run / f"t{t:02d}" / "scale_stage"
        m = {"stage": "dating", "t": f"T_{t}", "status": "done", "items_sha256": isha, "pg_git_sha": "dry", "device_name": "none", "width": W,
             "n_layers": N_UP + 1, "timing_s": {"wall_after_load": 100.0}, "download_and_corpus_s": 60.0, "load_s": 30.0, "cases": {},
             **gate_records(cases[0]["pole"], cases[0]["job"])}
        ms = {"stage": "scale", "t": f"T_{t}", "status": "done", "items_sha256": isha, "pg_git_sha": "dry", "device_name": "none", "width": W,
              "n_layers": N_UP + 1, "timing_s": {"wall_after_load": 50.0}, "download_and_corpus_s": 0.0, "load_s": 0.0, "cases": {},
              **gate_records(cases[0]["pole"], cases[0]["job"])}
        for i, c in enumerate(cases):
            sg = construction[c["job"]]["sign"]
            gJ, uJ, xJ = gu_for(aJ[i][k], sg, rng)
            gO, uO, xO = gu_for(aO[i][k], sg, rng)
            exact["J"][c["job"], t], exact["O"][c["job"], t] = xJ, xO
            path = d / "dating" / f"{c['pole']}__c{c['c']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, t=f"T_{t}", pole=c["pole"], side=np.array(["J"] * N_J + ["O"] * N_O), tok=np.array(c["J_tok"] + c["O_tok"], np.int64),
                                gP=np.concatenate([gJ, gO]), uP=np.concatenate([uJ, uO]), aP=np.concatenate([xJ, xO]))
            m["cases"][c["job"]] = {"set": "third", "pole": c["pole"], "layer": c["layer"], "birth": c["birth"], "n_J": N_J, "n_O": N_O, "mean_J": float(xJ.mean()),
                                    "mean_O": float(xO.mean()), "path": str(path.relative_to(d)), "seconds": 0.1}
        scale_dirs = [(ds, ms)] + ([(d, m)] if t == 15 else [])
        for p in sorted({c["pole"] for c in cases}):
            sg = 1.0 if p.endswith("+") else -1.0
            target = pole_scale[p] * sfac[k] * 1.002 * np.linspace(0.8, 1.2, N_SCALE)        # median 1.002 c_P sfac(t)
            g, u, x = gu_for(target, sg, np.random.default_rng(k * 100 + int(p[4:-1])))         # the same capture in both stages of a load
            for dd, mm in scale_dirs:
                sp = dd / "scale" / f"{p}.npz"
                sp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(sp, t=f"T_{t}", pole=p, tok=np.array(scale_tok[p], np.int64), gP=g, uP=u, aP=x)
                sb = next(c["s_P"][f"T_{t}"] for c in cases if c["pole"] == p)
                mm.setdefault("scale", {})[p] = {"median_captured": float(np.median(x)), "median_bank": sb, "rel_diff": abs(float(np.median(x)) - sb) / abs(sb),
                                                 "n": N_SCALE, "path": str(sp.relative_to(dd))}
        e1 = (np.arange(N_E1) * 5 + 1 + t).astype(np.int64)
        abar = (G[k][:, None] * spread[None, :] * np.where(np.arange(W) % 2, -1.0, 1.0)[None, :]).astype(np.float32)
        bgs = [(ds, ms)] + ([(d, m)] if t == 15 else [])
        for dd, mm in bgs:
            np.savez_compressed(dd / "abar.npz", abar=abar, e1=e1)
            dump(dd / "background.json", {"e1": e1.tolist(), "t_index": t, "n_layers": N_UP, "abar_abs_median_by_layer": np.median(np.abs(abar), 1).tolist(),
                                          "abar_abs_max_by_layer": np.abs(abar).max(1).tolist()})
            dump(dd / "high_background.json", {"t": f"T_{t}", "n_up": N_UP, "members": [[0, 0, float(abar[0, 0])]]})
            mm["background"] = {"n_positions": N_E1, "n_layers": N_UP, "finite": True, "abar_abs_median": float(np.median(np.abs(abar)))}
        if t == 15:
            np.savez_compressed(d / "background_1m.npz", abar_1m=abar, n_positions=np.int64(10))
            (d / "samples").mkdir(parents=True, exist_ok=True)
            with open(d / "samples" / "prompts.jsonl", "w") as fh:
                for c in cases:
                    for side in ("J", "O"):
                        for x in c[f"{side}_tok"][:2]:
                            fh.write(json.dumps({"job": c["job"], "side": side, "tok": x, "prompt": f"synthetic context {x}", "marked_token": " x", "n_tokens": 8}) + "\n")
        dump(d / "metrics.json", m)
        dump(ds / "metrics.json", ms)
    ref = {}
    for tt, fac in ((15, 1.001), (13, 2.0)):
        ref[tt] = out / f"spec4_t{tt}_background.json"
        dump(ref[tt], {"e1": (np.arange(N_E1) * 5 + 1 + 15).tolist() if tt == 15 else (np.arange(N_E1) * 5 + 1 + 13).tolist(), "t_index": tt, "n_layers": N_UP,
                       "abar_abs_median_by_layer": (G[15] * fac).tolist(), "abar_abs_max_by_layer": (G[15] * fac * 1.5).tolist()})
    return {"items": ip, "run": run, "ref15": ref[15], "ref13": ref[13], "cases": cases, "construction": construction, "G": G, "exact": exact, "sfac": sfac,
            "pole_scale": pole_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    b = build(out)
    vout = out / "verdict"
    cmd = [sys.executable, str(VERDICT), "--items", str(b["items"]), "--spec4-t15", str(b["ref15"]), "--spec4-t13", str(b["ref13"]), "--out", str(vout)]
    exits = {}

    # an incomplete bank stops the verdict
    inc = out / "run_incomplete"
    shutil.copytree(b["run"], inc)
    (inc / "t13" / "scale_stage" / "abar.npz").unlink()
    r = subprocess.run(cmd + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "BIRTHOBJECT_BANK_INCOMPLETE" in r.stderr and "t13/scale_stage/abar.npz" in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits["incomplete"] = r.returncode
    print("verdict on a bank without t13/scale_stage/abar.npz:", r.stderr.splitlines()[1].strip(), "exit", r.returncode)
    shutil.rmtree(inc)

    # green, with the construction's known answers
    r = subprocess.run(cmd + ["--run", str(b["run"])], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_BIRTHOBJECT_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_birthobject.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()) and all(x["verdict"] != "not read" for x in v["rules"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    assert len([k for k in v["gates"] if k.startswith("known_number")]) == 2 * 16 + 1 and len([k for k in v["gates"] if k.startswith("capture_identity")]) == 32, sorted(v["gates"])
    assert (vout / "GRID8.md").read_text().startswith("Cases 24 on 12 neurons"), "GRID8.md does not open with the count line"
    assert v["count"]["live"] == 24 and v["count"]["E1"] == 24 and v["count"]["E4"] == 24, v["count"]
    births = {c["job"]: c["birth"] for c in b["cases"]}
    event = {j for j, x in b["construction"].items() if x["event"]}
    r1 = {x["job"]: x for x in v["E1_rows"]}
    r2 = {x["job"]: x for x in v["E2_rows"]}
    r4 = {x["job"]: x for x in v["E4_rows"]}
    for j in event:
        assert r1[j]["t_break"] == births[j] - 1, (j, r1[j]["t_break"], births[j], "the break of a step at t_b is not t_b - 1")
        assert r4[j]["t_rate"] == births[j] - 1, (j, r4[j]["t_rate"], births[j], "the largest log-rate is not the pair into t_b")
        assert j in r2 and r2[j]["conc"] == 1.0 and set(r2[j]["jumps_J"]) == {births[j] - 1}, (j, r2.get(j))
    n_within = sum(abs(r1[j]["t_break"] - births[j]) <= 1 for j in r1)
    assert v["rules"]["E1"]["n_within1"] == n_within and abs(v["rules"]["E1"]["value"] - n_within / 24) < 1e-12, (v["rules"]["E1"], n_within)
    # E3 recomputed from the construction: r^g = mean a_P / g_L(t), ratio t_b over t_b - 2, median over the 24 cases
    G = b["G"]
    ex = b["exact"]
    rat = {side: [(ex[side][c["job"], c["birth"]].mean() / G[c["birth"], c["layer"]]) / (ex[side][c["job"], c["birth"] - 2].mean() / G[c["birth"] - 2, c["layer"]])
                  for c in b["cases"]] for side in ("J", "O")}
    e3 = v["rules"]["E3"]
    want = (float(np.median(rat["J"])), float(np.median(rat["O"])))     # g_l is banked in float32: agreement to 1e-6 relative
    assert np.allclose([e3["r_J_ratio"]["value"], e3["r_O_ratio"]["value"]], want, rtol=1e-6, atol=0), (e3["r_J_ratio"]["value"], e3["r_O_ratio"]["value"], want)
    gs = v["gates"]["global_scale"]["value"]
    assert abs(gs["median_rel_diff_T15"] - 1e-3 / 1.001) < 1e-6 and gs["g_min_all_loads"] > 0, gs
    sc = v["gates"]["scale"]["value"]
    assert sc["max_abs_diff_T15"] == 0.0 and sc["n_nonpositive_T15"] == 0 and sc["min_valid_share_from_T9"] == 1.0, sc
    kn = v["gates"]["known_number scale T_15"]
    assert abs(kn["value"] - 0.002) < 1e-6 and kn["failing_input"]["value"] > 0.2, kn
    samp = json.loads((vout / "samples" / "per_context_curves.json").read_text())
    assert len(samp) == 6 and all(len(s_["prompts"]) == 4 and len(s_["c_J"]) == N_J and len(s_["c_J"][0]) == 16 for s_ in samp), [(s_["job"], len(s_["prompts"])) for s_ in samp]
    print(f"known answers: break t_b - 1, largest log-rate at the pair into t_b and J's concentration 1 on all {len(event)} event cases; E1 {n_within} of 24 "
          f"within one step; E3 r_J^g {e3['r_J_ratio']['value']:.4f}, r_O^g {e3['r_O_ratio']['value']:.4f} equal to the construction's; six sample cases with prompts")

    # each failing input
    fails = {}
    for mode, prefix in MODES.items():
        r = subprocess.run(cmd + ["--run", str(b["run"]), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_birthobject.json").read_text())
        if prefix:
            tg = {k: g for k, g in f["gates"].items() if k.split()[0] == prefix}
            assert tg and not any(g["ok"] for g in tg.values()) and all(x["verdict"] == "not read" for x in f["rules"].values()), (mode, {k: g["ok"] for k, g in tg.items()})
        else:
            assert all(g["ok"] for g in f["gates"].values()) and not f["rules"]["E1"]["fires"] and not f["rules"]["E4"]["fires"], (mode, f["rules"]["E1"], f["rules"]["E4"])
            assert f["rules"]["E1"]["dates"].startswith("shuffled") and f["rules"]["E4"]["pairs"].startswith("shuffled"), mode
        fails[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_birthobject exit codes:", exits)
    (out / "metrics.json").write_text(json.dumps({"exits": exits, "failing_inputs": fails, "E1": v["rules"]["E1"]["value"], "E3": [e3["r_J_ratio"]["value"], e3["r_O_ratio"]["value"]]}, indent=1))
    print("PASS birthobject: a synthetic 16-load bank in the job's layout; the verdict stops on an incomplete bank, is green with the construction's known "
          "answers, and fails on each of its five failing inputs")
    print("REUSE_DRYRUN_BIRTHOBJECT_OK")


if __name__ == "__main__":
    main()
