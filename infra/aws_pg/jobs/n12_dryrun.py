"""Weight-free offline gate of N12 (studies/next/07_PICKS/N12_OFFLINE_GATE.md). No weight download, no GPU, no AWS
call. CPU torch 2.12.0 and transformers 5.9.0 (the banked grid harvest's versions).

    python -m infra.aws_pg.jobs.n12_dryrun --bank <BANK/neurons> --hf <HF text reads> --sides \
        studies/next/n12/offline_gate/gate_sides.json --examples studies/next/n12/samples/raw_examples_zero_gpu.json \
        --out <scratch dir>

Sections, one CHECK line per check:
  REAL     the unit table (reproducible, registered, every twin re-verified, the reader's distinct-token counts), the
           registered M2 inputs, the plan, the loss-matched candidates of every real window, the zero-GPU raw examples'
           checks, and every gate's two sides at every point from the banked files (gate_sides.json, the A5 table).
  SCORE    the pure functions of infra/pg/n12.py on small arrays with known answers.
  INSTR    the gradient instrument on a tiny fp32 Olmo 3: n12_score.grad_capture against central finite differences
           (positive control) and against the neighbouring neuron (failing input); zero ablation against the
           first-order prediction. A1: the gradient pass dispatches the math attention kernel, not the fused one.
  JOB      the tiny world of M2's dry run (4 layers, hidden 64, MLP width 96, the real tokenizer, 4 checkpoints): M2's
           job writes its banks (ranks, vectors), then n12_score.main runs from its entry point in a child process;
           every gate passes, every failing input fails, the records and the raw examples are written.
  FAIL     each instance gate on a planted failing input, in a child process: the job exits non-zero naming it.
           A1: a second gradient pass with one perturbed entry keeps Spearman >= 0.999, changes u_S only in units
           whose rows hold that entry, and fails I-GRAD.
  VERDICT  freeze and verdict on synthetic records, one case per branch; the tiny job's records summarized.
  PROJECT  GPU-hours of the launch from the real plan.
Ends with N12_DRYRUN_OK when every check holds.
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
WINDOW, BATCH, EXCL_TINY, M_LM_TINY = MD.WINDOW, MD.BATCH, 1, 4


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:64s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------- REAL
def real(bank: pathlib.Path, sides: pathlib.Path, examples: pathlib.Path, scratch: pathlib.Path):
    from infra.pg import m2, n12
    from infra.aws_pg.jobs import n12_score as J
    up = REPO_ROOT / "studies/next/n12/units.json"
    bp = REPO_ROOT / "studies/next/m2/births.json"
    bj = json.loads(bp.read_text())
    bj["_sha256"] = m2.sha256_file(bp)
    again = n12.build_units(bank, bj)
    (scratch / "units_again.json").write_text(json.dumps(again, indent=1))
    check("REAL units.json is reproducible from the laptop banks and births.json",
          sha(scratch / "units_again.json") == sha(up), sha(up)[:12])
    check("REAL units.json sha256 is the one n12_score registers", sha(up) == J.UNITS_SHA, J.UNITS_SHA)
    for h in ("freeze", "test"):
        f = REPO_ROOT / f"studies/next/m2/banks/score-06/m2/ranks_{h}.jsonl"
        check(f"REAL M2 ranks_{h}.jsonl sha256 registered", sha(f) == J.M2_RANKS_SHA[h], J.M2_RANKS_SHA[h][:12])
    uj = json.loads(up.read_text())
    U = uj["units"]
    check("REAL births.json sha256 recorded in units.json is M2's registered one",
          uj["births_sha256"] == bj["_sha256"] == "fdca794bc98a78ecf7aa360cad33ca598b633830e2c9891f8dbb7ca73aaf45b7")
    cnt = uj["n_units"]
    check("REAL units: real 82/71, pb 79/70, dt at least 50 per half",
          cnt["real_freeze"] == 82 and cnt["real_test"] == 71 and cnt["pb_freeze"] == 79 and cnt["pb_test"] == 70
          and cnt["dt_freeze"] >= n12.N_MIN and cnt["dt_test"] >= n12.N_MIN, cnt)
    B = m2.load_bank(bank)
    ctx = n12.contexts_t15(bank)
    births = {b["id"]: b for b in bj["births"]}
    ok_dt = ok_pb = ok_real = 0
    for u in U:
        b = births[u["birth"]]
        if u["kind"] == "real":
            ok_real += u["tok"] == b["tok"] and u["pole"] == b["pole"] and u["d"] == n12.distinct(ctx, b["pole"], b["ranks"])
        elif u["kind"] == "pb":
            ok_pb += u["tok"] == b["pb"]["tok"] and u["pole"] == b["pb"]["pole"]
        else:
            i = B["row"][u["pole"]]
            k = int(m2.births_matrix(B["a_plus"][:, i, :], B["q999"][:, i], np.array([u["ranks"]]))[0])
            ok_dt += (k == b["k"] and u["layer"] == b["layer"] and m2.parse_pole(u["pole"])[1] != b["neuron"] and
                      len(u["tok"]) == b["size"] and n12.distinct(ctx, u["pole"], u["ranks"]) == u["d_real"] and
                      B["tok15"][i, u["ranks"]].tolist() == u["tok"] and
                      set(u["ranks"]) <= set(m2.nonecho_ranks(B["clusters"][u["pole"]])))
    n_dt = sum(u["kind"] == "dt" for u in U)
    check("REAL every real unit is its birth (positions, pole, distinct-token count)", ok_real == 153, ok_real)
    check("REAL every pb unit is M2's twin", ok_pb == 149, ok_pb)
    check("REAL every dt twin re-dated k, same layer, other neuron, same size, same distinct count, non-echo ranks",
          ok_dt == n_dt, f"{ok_dt} of {n_dt}")
    sh_real = [u["d"] / u["size"] for u in U if u["kind"] == "real" and u["birth"] + ":pb" in {x["unit"] for x in U}]
    sh_pb = [u["d"] / u["size"] for u in U if u["kind"] == "pb"]
    check("REAL the reader's distinct-token counts reproduce: births with a twin 78 of 149 below 0.67, twins 16",
          sum(x < 0.67 for x in sh_real) == 78 and sum(x < 0.67 for x in sh_pb) == 16,
          f"{sum(x < 0.67 for x in sh_real)}, {sum(x < 0.67 for x in sh_pb)}")
    dts = [u for u in U if u["kind"] == "dt"]
    check("REAL dt twins carry repeated tokens like the births: share below 0.67 among dt",
          sum(u["d"] / u["size"] < 0.67 for u in dts) >= 0.4 * len(dts), sum(u["d"] / u["size"] < 0.67 for u in dts))
    pl = n12.plan(U)
    want = sorted((u["unit"], r) for u in U for r in u["roles"])
    got = sorted((uid, r) for p in pl["points"].values() for uid, r in p["roles"])
    check("REAL plan: every (unit, role) exactly once; points in the registered order",
          want == got and list(pl["points"]) == [t for t in n12.ORDER if t in pl["points"]], list(pl["points"]))
    check("REAL plan: gradient windows are forward windows; K0 has 10 real pre-role carriers at T_14",
          all(set(p["grad_windows"]) <= set(p["windows"]) for p in pl["points"].values()) and
          sum(u["kind"] == "real" and u["roles"]["pre"] == n12.KNOWN_T for u in U) >= n12.K0_N)
    mins = []
    for w, ex in pl["excl"].items():
        ok = np.ones(m2.WINDOW, bool)
        ok[0] = False
        for p in ex:
            ok[max(0, p - n12.EXCL):min(m2.WINDOW, p + n12.EXCL + 1)] = False
        mins.append(int(ok.sum()))
    check("REAL every window keeps at least M_LM loss-matched candidates after the exclusions",
          min(mins) >= n12.M_LM, f"min {min(mins)} of 2047 over {len(mins)} windows")
    ex = json.loads(examples.read_text())
    cp = ex["checks_pass"]
    check("REAL raw examples: M2's top-10 under B recomputed from the vectors at the pre and birth roles (10 of 10)",
          cp["top10_pre"] and cp["top10_birth"], {k: sum(v) for k, v in ex["checks"].items() if k.startswith("top10")})
    check("REAL raw examples: the rolled neuron axis does not reproduce the top-10 (failing input)", cp["top10_fail_fails"])
    check("REAL raw examples: the decomposition's identity on real arrays; 20 competitors with B > 0",
          cp["identity"] and cp["n_comp_20"])
    s = json.loads(sides.read_text())
    for name, g in s["gates"].items():
        check(f"REAL sides {name}: {g['scope']}", g["pass_side_ok"] and g["fail_side_ok"],
              g.get("summary", ""))
    return uj


# ------------------------------------------------------------------------------------------------------- SCORE
def score_checks():
    from infra.pg import n12
    Bs = np.zeros((6, 2))
    Bs[0, 0], Bs[1, 1], Bs[2, 0], Bs[3, 0], Bs[4, 1], Bs[5, 0] = 0.9, 0.95, 0.99, 0.5, 0.99, 0.0
    c = n12.competitors(Bs, 2, k=3)
    check("SCORE competitors: own neuron excluded, highest B first, stable", c == [[4, 1], [1, 1], [0, 0]], c)
    check("SCORE layer rank: ties count against the pole", n12.layer_rank(Bs, 2, 0) == 2 and n12.layer_rank(Bs, 4, 1) == 2)
    pre = np.zeros((40, 2))
    pre[:, 0] = np.linspace(0.5, 0.99, 40)
    birth = pre.copy()
    birth[5, 0] = 0.995                               # only the pole rises
    cf = n12.cf_ranks(pre, birth, 5, 0)
    check("SCORE counterfactual: only the pole rises -> r_c = r1 = 1, r_o = r0, F > 0",
          cf["r1"] == 1 and cf["r_c"] == 1 and cf["r_o"] == cf["r0"] == 35 and cf["F"] > 0, cf)
    birth = pre.copy()
    birth[6:, 0] = 0.1                                # only the others fall
    cf = n12.cf_ranks(pre, birth, 5, 0)
    check("SCORE counterfactual: only the others fall -> r_o = r1 = 1, r_c = r0, F < 0",
          cf["r1"] == 1 and cf["r_o"] == 1 and cf["r_c"] == 35 and cf["F"] < 0, cf)
    cf = n12.cf_ranks(pre, pre, 5, 0)
    check("SCORE counterfactual identity: r_c = r_o = r0, F = 0", cf["r_c"] == cf["r_o"] == cf["r0"] and cf["F"] == 0.0)
    G = np.array([[-1.0, 2.0]])
    q = np.array([[0.5, 0.25], [1.0, np.nan]])
    s = n12.pull(G, q)
    check("SCORE pull sign: + pole pulled up by G < 0; - pole by G > 0; NaN q999 gives 0",
          np.allclose(s[0], [[0.5, -0.25], [-2.0, 0.0]]), s[0].tolist())
    ell = np.array([0, 5.0, 1.0, 1.1, 0.9, 3.0, 1.05, 8.0, 1.0, 2.0])
    lm = n12.loss_matched(ell, 2, excl=[2], m=3, ex=1)
    check("SCORE loss-matched: nearest losses, the excluded neighbourhood and index 0 left out, ties by index",
          lm == [8, 6, 4], lm)
    S = np.array([[0.1, 0.2], [0.3, 0.05], [0.3, 0.0]])
    check("SCORE u_share: the share of competitors at or above the pole (ties count against)",
          n12.u_share(S, [[0, 1], [1, 0], [1, 1]], [2, 0]) == 1 / 3, n12.u_share(S, [[0, 1], [1, 0], [1, 1]], [2, 0]))
    check("SCORE spearman: known values, ties averaged, constant input undefined",
          abs(n12.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12 and
          abs(n12.spearman([1, 2, 2, 3], [1, 3, 2, 4]) - 0.9486832980505138) < 1e-9 and
          n12.spearman([1, 1, 1], [1, 2, 3]) is None)
    rng = np.random.default_rng(3)
    good = [{"abl": [[f"p{j}", x, x + 0.1 * rng.normal()] for j, x in enumerate(rng.normal(size=4))]}
            for _ in range(60)]
    v = n12.validation(good)
    check("SCORE validation passes on a faithful prediction; its sign-flipped input fails",
          v["pass"] and not v["failing_input"]["pass"] and not v["above_ceiling"], f"rho {v['rho']:.3f}")
    bad = [{"abl": [[f"p{j}", x, rng.normal()] for j, x in enumerate(rng.normal(size=4))]} for _ in range(60)]
    v = n12.validation(bad)
    check("SCORE validation fails on an unrelated prediction", not v["pass"], f"rho {v['rho']:.3f}")
    same = [{"abl": [[f"p{j}", x, x] for j, x in enumerate(rng.normal(size=4))]} for _ in range(60)]
    check("SCORE validation flags its ceiling (prediction equal to the ablation)", n12.validation(same)["above_ceiling"])
    rc = n12.room_check()
    check("SCORE room: every label branch reachable at the test size", rc["pass"], rc["labels"])
    b = n12.boot_mean(np.r_[np.zeros(50), np.ones(50)])
    check("SCORE bootstrap interval brackets the mean", b["lo"] < 0.5 < b["hi"] and b["mean"] == 0.5, b)
    fu = n12.floor_u(np.full((10, 5), 0.4))
    check("SCORE matched floor of a constant u is that constant", np.isclose(fu["q025"], 0.4) and np.isclose(fu["mean"], 0.4))
    from infra.aws_pg.jobs import n12_score as J
    env = os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
    rc = subprocess.run([sys.executable, "-c", "from infra.aws_pg.jobs import n12_score as J\nwith J.grad_mode(): pass"],
                        capture_output=True, text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = env
    import torch
    with J.grad_mode():
        on = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    off = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    check("SCORE A1 grad_mode: deterministic and bf16 math inside, off after; refuses without CUBLAS_WORKSPACE_CONFIG",
          on == (True, True) and off == (False, False) and rc.returncode != 0 and
          "CUBLAS_WORKSPACE_CONFIG" in rc.stderr, MD.last_error(rc))
    rr = [{"kind": "real", "pull": {"S": 0.5}, "comp": [{"S": 0.2}, {"S": 0.5}, {"S": 0.9}]},
          {"kind": "real", "pull": {"S": 0.5}, "comp": [{"S": 0.1}, {"S": 0.7}]}]
    r0, r1 = n12.u_resolution(rr, 0.0), n12.u_resolution(rr, 0.25)
    check("SCORE A1 u_S resolution: noise 0 counts exact ties only; a larger noise counts every gap below it",
          r0["n_within_noise"] == 1 and r1["n_within_noise"] == 2 and n12.order_ranks([0.1, 0.9, 0.1]) == [2, 1, 3],
          (r0, r1))


# ------------------------------------------------------------------------------------------------------- INSTR
def instr(hf: pathlib.Path):
    """The gradient instrument on a tiny fp32 model: autograd G against central finite differences, and ablation
    against the first-order prediction."""
    import torch
    import torch.nn.functional as F
    from infra.aws_pg.jobs import n12_score as J
    from infra.pg import neurons_harvest as NH
    cfg = MD.tiny_cfg(hf)
    model = MD.tiny_model(cfg, 15).double()
    for p in model.parameters():
        p.requires_grad_(False)
    tower = NH.text_tower(model)
    rng = np.random.default_rng(0)
    W = 64
    tokens = rng.integers(1000, 1300, 4 * W).astype(np.uint32)
    reg = {"window": W, "bos": 100257, "batch": 4}
    need = {1: [(1, 10, W + 9), (2, 30, W + 29)]}
    gc = {}
    J.grad_capture(model, tower, tokens, need, [1], {1: [10, 30]}, reg, "cpu", gc)

    def L_with(layer, idx, n, delta=None, zero=False):
        from infra.aws_pg.jobs.m2_score import windows_tensor
        x, _ = windows_tensor(tokens, [1], W, 100257, 1)

        def hook(module, args):
            h = args[0].clone()
            if zero:
                h[0, idx, n] = 0
            else:
                h[0, idx, n] += delta
            return (h,)
        hd = tower.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        with torch.no_grad():
            lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        hd.remove()
        return float(F.cross_entropy(lg[0, :-1].double(), x[0, 1:], reduction="sum"))
    errs, errs_fail, pairs = [], [], []
    for (lyr, idx, e) in need[1]:
        G = gc[(lyr, e)]["G"]
        H = gc[(lyr, e)]["H"]
        for n in np.argsort(-np.abs(G))[:4]:
            d = 1e-4
            fd = (L_with(lyr, idx, int(n), d) - L_with(lyr, idx, int(n), -d)) / (2 * d)
            errs.append(abs(fd - G[n]) / abs(fd))
            errs_fail.append(abs(fd - G[(n + 1) % G.size]) / abs(fd))
        L0 = L_with(lyr, idx, 0, 0.0)
        for n in rng.choice(G.size, 12, replace=False):
            pairs.append((-H[n] * G[n], L_with(lyr, idx, int(n), zero=True) - L0))
    from infra.pg import n12
    check("INSTR grad_capture equals central finite differences (fp64 model, 8 largest-gradient neurons)",
          max(errs) < 2e-2, f"max relative error {max(errs):.2e}")
    check("INSTR the neighbouring neuron's gradient does not (failing input)", min(errs_fail) > 0.1,
          f"min relative error {min(errs_fail):.2e}")
    rho = n12.spearman([p[0] for p in pairs], [p[1] for p in pairs])
    check("INSTR zero ablation against -h*G on 24 (neuron, position) pairs: Spearman >= VAL_RHO_MIN",
          rho >= n12.VAL_RHO_MIN, f"rho {rho:.3f}")
    ell_ok = gc[(1, W + 9)]["valid"] and len(gc[(1, W + 9)]["lm"]) == n12.M_LM and 10 not in gc[(1, W + 9)]["lm"]
    check("INSTR loss-matched positions captured beside each exemplar", ell_ok, gc[(1, W + 9)]["lm"][:5])
    # A1: which attention kernel the gradient pass dispatches (the fp32 tiny model; CPU's fused kernel is flash)
    import contextlib
    from torch.utils._python_dispatch import TorchDispatchMode

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.add(str(func))
            return func(*args, **(kwargs or {}))
    m32 = MD.tiny_model(cfg, 15)
    for p in m32.parameters():
        p.requires_grad_(False)
    seen = {}
    for name, mode in (("a1", J.grad_mode), ("default", contextlib.nullcontext)):
        og = J.grad_mode
        J.grad_mode = mode
        with Ops() as o:
            J.grad_capture(m32, NH.text_tower(m32), tokens, need, [1], {1: [10, 30]}, reg, "cpu", {})
        J.grad_mode = og
        seen[name] = sorted(x for x in o.names if "attention" in x or "softmax" in x)
    fused = [x for x in seen["a1"] if "scaled_dot_product" in x]
    check("INSTR A1: the gradient pass dispatches the math kernel; without grad_mode the fused kernel (failing input)",
          not fused and any("_safe_softmax" in x for x in seen["a1"]) and
          any("scaled_dot_product" in x for x in seen["default"]), seen)


# ------------------------------------------------------------------------------------------------------- JOB
def child(spec_path: str):
    """Entry point of a child process: n12_score.main with the stand-in hub, the tiny world's constants and plants."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    from infra.pg import m2, n12
    import infra.aws_pg.jobs.m2_score as M
    import infra.aws_pg.jobs.n12_score as J
    m2.WINDOW, m2.N_WINDOWS = WINDOW, spec["n_windows"]
    m2.N_WINDOWED = WINDOW * spec["n_windows"]
    m2.POINTS = [tuple(p) for p in spec["points"]]
    n12.EXCL, n12.M_LM = spec["excl"], spec["m_lm"]
    plant = spec.get("plant")
    if plant == "u_scale_1":
        from infra.pg import neurons_harvest as NH
        NH.WRONG_U_SCALE = 1.0
    if plant == "abl_clean_ablates":
        orig = J.ablate

        def ablate(model, tower, tokens, rows, reg, device):
            rows = [(w, lyr, list(range(1, WINDOW)), slice(None)) if n is None else (w, lyr, ix, n)
                    for w, lyr, ix, n in rows]
            return orig(model, tower, tokens, rows, reg, device)
        J.ablate = ablate
    if plant in ("grad_second_noisy", "fwd_second_scaled", "grad_second_one_entry"):
        og, of = J.grad_capture, J.forward_loss

        def grad_capture(model, tower, tokens, need_g, ws, excl, reg, device, into):
            r = og(model, tower, tokens, need_g, ws, excl, reg, device, into)
            if plant == "grad_second_noisy" and len(ws) > 1 and ws[0] > ws[-1]:
                for v in into.values():
                    v["G"] = v["G"] + np.random.default_rng(0).normal(size=v["G"].shape).astype(np.float32)
            if plant == "grad_second_one_entry" and len(ws) > 1 and ws[0] > ws[-1]:
                lyr, e, n, delta = spec["one_entry"]
                into[(lyr, e)]["G"][n] += np.float32(delta)
            return r

        def forward_loss(model, tap, acts, tokens, need, ws, reg, device, cap, loss):
            r = of(model, tap, acts, tokens, need, ws, reg, device, cap, loss)
            if plant == "fwd_second_scaled" and len(ws) > 1 and ws[0] > ws[-1]:
                for k in cap:
                    cap[k] = cap[k] * 1.1
            return r
        J.grad_capture, J.forward_loss = grad_capture, forward_loss
    M.HUB = MD.fake_hub(spec["remotes"], spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    J.main(spec["argv"])


def run_child(scratch: pathlib.Path, name: str, spec: dict, env: dict) -> subprocess.CompletedProcess:
    p = scratch / f"n12child_{name}.json"
    p.write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import n12_dryrun as D; "
                           "D.child(sys.argv[1])", str(p)], capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def tiny_units(bj: dict, poles: list, pos15: np.ndarray) -> dict:
    """N12 units of the tiny world: real and pb from M2's tiny births; dt planted as another pole's ranks 20.."""
    from infra.pg import m2
    U = []
    by_layer = {}
    for i, p in enumerate(poles):
        by_layer.setdefault(m2.parse_pole(p)[0], []).append(i)
    for b in bj["births"]:
        base = {"half": b["half"], "layer": b["layer"], "birth": b["id"], "k": b["k"], "size": b["size"],
                "d_real": b["size"]}
        U.append({**base, "unit": b["id"], "kind": "real", "pole": b["pole"], "carrier": [b["neuron"], b["sign"]],
                  "row": b["row"], "tok": b["tok"], "ranks": b["ranks"], "d": b["size"],
                  "roles": {"pre": b["k"] - 1, "birth": b["k"]}})
        if b["pb"] is not None:
            p = b["pb"]
            U.append({**base, "unit": b["id"] + ":pb", "kind": "pb", "pole": p["pole"],
                      "carrier": [p["neuron"], p["sign"]], "row": p["row"], "tok": p["tok"], "ranks": p["ranks"],
                      "d": b["size"], "roles": {"pre": b["k"] - 1, "birth": b["k"]}})
        j = [x for x in by_layer[b["layer"]] if m2.parse_pole(poles[x])[1] != b["neuron"]][-1]
        lj, nj, sj = m2.parse_pole(poles[j])
        ranks = list(range(20, 20 + b["size"]))
        U.append({**base, "unit": b["id"] + ":dt", "kind": "dt", "pole": poles[j], "carrier": [nj, sj], "row": j,
                  "tok": pos15[j, 20:20 + b["size"]].tolist(), "ranks": ranks, "d": b["size"],
                  "roles": {"pre": b["k"] - 1, "birth": b["k"], "end": 15}})
    return {"units": U, "examples": bj["examples"], "n_units": {}}


def job(hf: pathlib.Path, scratch: pathlib.Path):
    """M2's tiny world (models, banked grid, births), M2's job on it, then the N12 job."""
    import torch
    import transformers
    from infra.pg import m2
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    n_windows = 64
    corpus_dir = scratch / "corpus"
    MD.make_corpus(corpus_dir, hf, n_windows)
    corpus = NC.load_tok(corpus_dir, "olmo3")
    tok = transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"])
    cfg = MD.tiny_cfg(hf)
    models = {15: MD.tiny_model(cfg, 15)}
    for t, nz in ((14, 0.3), (13, 0.4), (12, 0.5)):
        models[t] = MD.tiny_model(cfg, t, models[15], nz)
    points = [[t, f"{'1' if t in (0, 1, 2, 3, 4, 5, 6, 15) else '1b'}/out/t{t:02d}", f"stage1-step{t}",
               hashlib.sha1(f"stage1-step{t}".encode()).hexdigest(), "cpu"] for t in range(16)]
    poles = [NH.pole_id(lyr, n, s) for lyr in range(4) for n in (3, 17, 40, 63, 90) for s in (0, 1)]
    grid = scratch / "grid"
    remotes, commits, pos15 = {}, {}, None
    for t in (15, 14, 13, 12):
        d = scratch / "remote" / points[t][2]
        MD.make_remote(d, models[t], cfg, hf)
        remotes[points[t][2]], commits[points[t][2]] = str(d), points[t][3]
        model, _ = NH.load_model(str(d), "O7", device="cpu")
        NH.run(grid / points[t][1], model=model, tok=tok, corpus=corpus, model_key="O7", repo=m2.REPO,
               revision="bank", branch=points[t][2], max_tokens=n_windows * WINDOW, window=WINDOW, batch=BATCH,
               poles=poles, positions=pos15, device="cpu", nll_tokens=1000)
        if t == 15:
            pos15 = np.load(grid / points[t][1] / "poles" / "positions.npz")["tok"]
    inputs = scratch / "inputs"
    inputs.mkdir()
    bj = MD.tiny_births(poles, pos15)
    (inputs / "births.json").write_text(json.dumps(bj))
    for t in range(16):
        shutil.copy(grid / points[max(t, 12)][1] / "poles" / "exemplar_a.npz", inputs / f"exemplar_a_t{t:02d}.npz")
    reg = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "births_sha": sha(inputs / "births.json"),
           "exemplar_sha": [sha(inputs / f"exemplar_a_t{t:02d}.npz") for t in range(16)],
           "tokens_sha256": corpus["tokens_sha256"], "points": {t: p for t, p in enumerate(points)},
           "ref_config": cfg, "window": WINDOW, "batch": BATCH, "bos": 100257, "cpu_device_name": "cpu"}
    spec = {"n_windows": n_windows, "points": points, "remotes": remotes, "commits": commits}
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    world = {"scratch": scratch, "inputs": inputs, "grid": grid, "corpus": corpus_dir, "reg": reg, "spec": spec,
             "env": env, "births": bj}
    rc, m2out = MD.launch(world, "m2")
    check("JOB M2's tiny job writes the banks N12 reads (ranks, vectors)", rc.returncode == 0 and
          (m2out / "m2" / "ranks_freeze.jsonl").exists(), f"exit {rc.returncode} {MD.last_error(rc)}")
    if rc.returncode != 0:
        print(rc.stderr[-3000:])
        raise SystemExit(1)
    n_in = scratch / "n12_inputs"
    n_in.mkdir()
    (n_in / "units.json").write_text(json.dumps(tiny_units(bj, poles, pos15)))
    for h in ("freeze", "test"):
        shutil.copy(m2out / "m2" / f"ranks_{h}.jsonl", n_in / f"m2_ranks_{h}.jsonl")
    nreg = {**{k: v for k, v in reg.items() if k != "births_sha"}, "units_sha": sha(n_in / "units.json"),
            "m2_ranks_sha": {h: sha(n_in / f"m2_ranks_{h}.jsonl") for h in ("freeze", "test")}}
    nw = {**world, "n_inputs": n_in, "nreg": nreg, "m2out": m2out, "poles": poles, "pos15": pos15}
    rc, out = launch(nw, "main")
    check("JOB n12_score.main exits 0 and prints N12_JOB_OK", rc.returncode == 0 and "N12_JOB_OK" in rc.stdout,
          f"exit {rc.returncode} {MD.last_error(rc)}")
    if rc.returncode != 0:
        print(rc.stderr[-5000:])
        raise SystemExit(1)
    jm = json.loads((out / "n12" / "metrics.json").read_text())
    check("JOB metrics complete, nothing parked", jm["complete"] is True and jm["parked"] == [])
    check("JOB every gate passed", all(g["pass"] for g in jm["gates"].values()), sorted(jm["gates"]))
    fi = [n for n, g in jm["gates"].items() if "failing_input" in g]
    check("JOB every registered failing input failed", all(not jm["gates"][n]["failing_input"]["pass"] for n in fi),
          f"{len(fi)} failing inputs")
    k0 = jm["gates"]["K0"]
    names = list(jm["gates"])
    check("JOB K0 at T_14 first: 10 carriers' B equal M2's banked pre-role B; birth-role B fails",
          names.index("K0") < names.index("t14_G_M2V") and set(names[:names.index("K0")]) ==
          {"G_SAME", "INPUTS", "G_CORPUS", "t14_G_REV", "t14_G_CFG", "t14_G0"} and k0["pass"] and
          len(k0["units"]) == 10 and not k0["failing_input"]["pass"],
          f"max |dB| {k0['max_abs_dB']:.1e}; failing min |dB| {k0['failing_input']['min_abs_dB']:.3e}")
    m2v = {n: g for n, g in jm["gates"].items() if n.endswith("G_M2V")}
    check("JOB G-M2V at every point: the forward equals M2's banked vectors (same device)",
          len(m2v) == 4 and all(g["exact_share"] == 1.0 for g in m2v.values()),
          {n: (g["n_units"], g["exact_share"]) for n, g in m2v.items()})
    ab = {n: g for n, g in jm["gates"].items() if n.endswith("G_ABL_ID")}
    check("JOB G-ABL-ID: clean rows equal the forward's loss; another window's loss fails",
          len(ab) == 3 and all(g["max_abs"] <= 1e-6 for g in ab.values()),
          {n: (g["max_abs"], round(g["failing_input"]["min_abs"], 3)) for n, g in ab.items()})
    ig = jm["gates"]["I_GRAD"]
    check("JOB I-GRAD: the reversed gradient pass reproduces G; the rolled axis fails",
          ig["pass"] and ig["u_S_flips"] == 0 and not ig["failing_input"]["pass"],
          f"rho {ig['rho']:.6f}, exact {ig['exact_share']:.3f}, failing rho {ig['failing_input']['rho']:.3f}")
    gs = jm["grad_pass_settings"]
    chg = {h: (out / "n12" / f"igrad_changed_{h}.jsonl").read_text() for h in ("freeze", "test")}
    check("JOB A1: settings recorded; I-GRAD writes |dG|, |dS| (0 here) and empty changed-unit files",
          gs["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and gs["use_deterministic_algorithms"] and
          gs["sdpa_backends"] == ["MATH"] and ig["max_abs_dG"] == 0.0 and ig["max_abs_dS"] == 0.0 and
          ig["changed_units"] == {"freeze": 0, "test": 0} and chg == {"freeze": "", "test": ""},
          f"n_units {ig['n_units']}, n {ig['n']}")
    ii = jm["gates"]["I_ID"]
    check("JOB I-ID: the reversed forward reproduces a, 0 flips; M2's birth-role vectors fail",
          ii["pass"] and ii["carrier_top10_flips_B"] == 0 and not ii["failing_input"]["pass"])
    keys = json.dumps(jm)
    check("JOB metrics.json carries no rank, hit or label", '"rank' not in keys and "hit" not in keys
          and "SUPPORTED" not in keys)
    from infra.pg import n12
    recs = {h: m2.read_jsonl(out / "n12" / f"n12_{h}.jsonl") for h in ("freeze", "test")}
    U = json.loads((n_in / "units.json").read_text())["units"]
    check("JOB one N12 record per unit", sorted(r["unit"] for h in recs for r in recs[h]) == sorted(u["unit"] for u in U),
          f"{sum(len(v) for v in recs.values())} records")
    r0 = [r for r in recs["freeze"] if r["kind"] == "real"][0]
    check("JOB a real record has 20 competitors, pull, 4 ablation pairs, matched u, the identity, the drop",
          len(r0["comp"]) == n12.K_COMP and "pull" in r0 and len(r0["abl"]) == n12.N_ABL + 1 and
          len(r0["pull"]["u_matched"]) == m2.K_MATCH and r0["cf_identity"]["F"] == 0.0 and
          r0["drop"] == -float(np.median(r0["gain"])))
    m2r = {(r["unit"], r["role"]): r for h in ("freeze", "test") for r in m2.read_jsonl(m2out / "m2" / f"ranks_{h}.jsonl")}
    same = all(r["cf"]["r0"] == m2r[(r["unit"], "pre")]["rank"]["B"] and
               r["cf"]["r1"] == m2r[(r["unit"], "birth")]["rank"]["B"]
               for h in recs for r in recs[h] if r["kind"] == "real")
    check("JOB the carrier's r0 and r1 equal M2's banked ranks under B (every real unit)", same)
    # recompute one record's pull from the banked arrays
    u = {x["unit"]: x for x in U}[r0["unit"]]
    pz = np.load(out / "n12" / "arrays" / f"t{u['roles']['pre']:02d}" / f"{u['unit'].replace(':', '_')}_pull.npz")
    st = np.load(world["grid"] / world["spec"]["points"][u["roles"]["pre"]][1] / "stats" / f"layer_{u['layer']:02d}.npz")
    P = n12.pull(pz["G"][pz["valid"]], st["q999"]).mean(0)
    check("JOB the pull recomputes from the banked gradient rows and the grid q999", np.allclose(P, pz["P"]))
    exs = {h: m2.read_jsonl(out / "n12" / "samples" / f"raw_examples_{h}.jsonl") for h in ("freeze", "test")}
    ok = all(len(exs[h]) == len(nw["births"]["examples"][h]) and all(
        "real" in e and e["real"]["comp_top3"] and "[[" in e["real"]["exemplars"][0] and e["real"]["lm_text"]
        for e in exs[h]) for h in exs)
    check("JOB raw examples: the selected births, competitors' contexts, exemplars, loss-matched texts", ok)
    return nw, out


def launch(nw: dict, name: str, reg_over=None, inputs_dir=None, m2_dir=None, exemplars=None, plant=None,
           budget: float = 60.0, one_entry=None):
    scratch = nw["scratch"]
    rp = scratch / f"n12registry_{name}.json"
    rp.write_text(json.dumps(dict(nw["nreg"], **(reg_over or {}))))
    out = scratch / f"n12out_{name}"
    e = dict(nw["env"], HF_HUB_CACHE=str(scratch / f"n12hf_{name}"))
    argv = ["--out", str(out), "--inputs", str(inputs_dir or nw["n_inputs"]), "--exemplars",
            str(exemplars or nw["inputs"]), "--m2", str(m2_dir or (nw["m2out"] / "m2")), "--corpus", str(nw["corpus"]),
            "--grid", str(nw["grid"]), "--budget-min", str(budget), "--registry", str(rp), "--device", "cpu"]
    spec = dict(nw["spec"], argv=argv, excl=EXCL_TINY, m_lm=M_LM_TINY, plant=plant, one_entry=one_entry)
    return run_child(scratch, name, spec, e), out


# ------------------------------------------------------------------------------------------------------- FAIL
def fails(nw: dict):
    scratch, reg = nw["scratch"], nw["nreg"]
    cases = [("G_SAME", {"versions": {**reg["versions"], "transformers": "0.0.0"}}, {}),
             ("INPUTS", {"units_sha": "0" * 64}, {}),
             ("G_CORPUS", {"tokens_sha256": "f" * 64}, {})]
    pts = {k: list(v) for k, v in reg["points"].items()}
    pts[14][3] = "1" * 40
    cases.append(("t14_G_REV", {"points": pts}, {}))
    cases.append(("t14_G_CFG", {"ref_config": {**reg["ref_config"], "intermediate_size": 97}}, {}))
    cases.append(("t14_G0", {}, {"plant": "u_scale_1"}))
    # K0: M2's banked pre-role B moved by 1e-3 in the ranks files
    d = scratch / "n12_inputs_k0"
    shutil.copytree(nw["n_inputs"], d)
    for h in ("freeze", "test"):
        rows = [json.loads(x) for x in (d / f"m2_ranks_{h}.jsonl").read_text().splitlines()]
        for r in rows:
            if r["role"] == "pre":
                r["value"]["B"] = r["value"]["B"] - 1e-3
        (d / f"m2_ranks_{h}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cases.append(("K0", {"m2_ranks_sha": {h: sha(d / f"m2_ranks_{h}.jsonl") for h in ("freeze", "test")}},
                  {"inputs_dir": d}))
    # G-M2V at T_15: M2's vectors there scaled by 1.1
    m2d = scratch / "m2_scaled15"
    shutil.copytree(nw["m2out"] / "m2", m2d)
    for f in (m2d / "vectors" / "t15").glob("*.npz"):
        z = dict(np.load(f))
        z["a"] = z["a"] * 1.1
        np.savez_compressed(f, **z)
    cases.append(("t15_G_M2V", {}, {"m2_dir": m2d}))
    # G-EX at T_14: the bank at T_14 replaced by T_13's
    ex = scratch / "exemplars_ex14"
    shutil.copytree(nw["inputs"], ex)
    shutil.copy(ex / "exemplar_a_t13.npz", ex / "exemplar_a_t14.npz")
    cases.append(("t14_G_EX", {"exemplar_sha": [sha(ex / f"exemplar_a_t{t:02d}.npz") for t in range(16)]},
                  {"exemplars": ex}))
    cases.append(("t14_G_ABL_ID", {}, {"plant": "abl_clean_ablates"}))
    cases.append(("I_GRAD", {}, {"plant": "grad_second_noisy"}))
    cases.append(("I_ID", {}, {"plant": "fwd_second_scaled"}))
    for gate, over, kw in cases:
        rc, _ = launch(nw, f"fail_{gate}", over, **kw)
        err = MD.last_error(rc)
        check(f"FAIL {gate} on its planted input: exit non-zero, gate named", rc.returncode != 0 and gate in err, err)
    one_entry(nw)
    rc, out = launch(nw, "park", budget=0.02)
    jm = json.loads((out / "n12" / "metrics.json").read_text())
    check("FAIL stop rule: a tiny budget parks the later points and the job ends normally",
          rc.returncode == 0 and jm["parked"] and jm["complete"] is True, jm.get("parked"))


def one_entry(nw: dict):
    """A1, the planted near-exact second pass: one gradient entry of the reversed pass moved, at the carrier neuron of
    a T_13 unit whose u_S is below 1, by 1e6 in the direction that lowers its pull (to u_S = 1). Spearman over every
    entry stays >= 0.999 (one entry moved), u_S changes only in units whose rows hold the entry, I-GRAD fails."""
    from infra.pg import m2
    main_out = nw["scratch"] / "n12out_main"
    recs = [r for h in ("freeze", "test") for r in m2.read_jsonl(main_out / "n12" / f"n12_{h}.jsonl")]
    U = {u["unit"]: u for u in json.loads((nw["n_inputs"] / "units.json").read_text())["units"]}
    r = sorted((r for r in recs if r["t_pre"] == m2.IDENTITY_T and "pull" in r and r["pull"]["u_S"] < 1.0),
               key=lambda r: r["unit"])[0]
    u = U[r["unit"]]
    pz = np.load(main_out / "n12" / "arrays" / f"t{m2.IDENTITY_T:02d}" / f"{u['unit'].replace(':', '_')}_pull.npz")
    e = [x for x, ok in zip(u["tok"], pz["valid"]) if ok][0]
    n, sgn = u["carrier"]
    delta = 1e6 if sgn == 0 else -1e6
    rc, out = launch(nw, "fail_I_GRAD_one_entry", plant="grad_second_one_entry", one_entry=[u["layer"], e, n, delta])
    ig = json.loads((out / "n12" / "metrics.json").read_text())["gates"]["I_GRAD"]
    changed = [json.loads(x) for h in ("freeze", "test")
               for x in (out / "n12" / f"igrad_changed_{h}.jsonl").read_text().splitlines()]
    touched = {x["unit"] for x in U.values() if x["roles"].get("pre") == m2.IDENTITY_T and x["layer"] == u["layer"]
               and e in x["tok"]}
    flipped = {x["unit"] for x in changed if x["u_S"][0] != x["u_S"][1]}
    check("FAIL A1 I_GRAD, one perturbed entry: rho >= 0.999, u_S changes only where the entry is read, gate fails",
          rc.returncode != 0 and "I_GRAD" in MD.last_error(rc) and ig["rho"] >= 0.999 and ig["u_S_flips"] >= 1 and
          u["unit"] in flipped and flipped <= touched and {x["unit"] for x in changed} <= touched and
          abs(ig["max_abs_dG"] - abs(delta)) < 1.0,
          f"rho {ig['rho']:.6f} over {ig['n']} entries, {ig['u_S_flips']} flips {sorted(flipped)}, "
          f"{len(touched)} units read the entry, max |dG| {ig['max_abs_dG']:.0f}")


# ------------------------------------------------------------------------------------------------------- VERDICT
def synth_rec(kind, half, i, F, u_S, rho, hit0, val_noise, rng, abl=True, dt_end=True, move=0.5, u_sd=0.05,
              drop=0.3):
    r0 = 5 if hit0 else 50
    rec = {"unit": f"b{i}" + ("" if kind == "real" else ":" + kind), "kind": kind, "half": half, "birth": f"b{i}",
           "cf": {"r0": r0, "r1": max(1, int(r0 / 10 ** move)), "r_c": 3, "r_o": 3, "F": float(F + 0.05 * rng.normal()),
                  "move": move},
           "cf_identity": {"r_c": r0, "r_o": r0, "F": 0.0}, "bar_pre": 0.999,
           "drop": float(drop + 0.05 * rng.normal())}
    u = float(np.clip(u_S + u_sd * rng.normal(), 0, 1))
    rec["pull"] = {"u_S": u, "u_P": u, "u_Pp": 0.5, "u_gain": 0.0, "u_matched": list(rng.uniform(0.4, 0.6, 100)),
                   "rho_S_gain": float(rho + 0.1 * rng.normal()), "rho_S_Bpre": 0.0, "dell": 0.01, "S": 0.0}
    rec["comp"] = [{"S": 0.1 * (j + 1)} for j in range(3)]
    if abl and kind == "real":
        x = rng.normal(size=4)
        rec["abl"] = [[f"p{j}", float(x[j]), float(x[j] + val_noise * rng.normal())] for j in range(4)]
    if kind == "dt" and dt_end:
        rec["end"] = {"rank": 1, "rank_opposite": 22016, "B": 0.9999}
    return rec


def write_banks(root: pathlib.Path, recs: dict, gates_pass=True):
    (root / "n12").mkdir(parents=True, exist_ok=True)
    g = {"K0": {"pass": gates_pass, "failing_input": {"pass": False}},
         "I_GRAD": {"pass": True, "max_abs_dS": 0.0, "failing_input": {"pass": False}}}
    (root / "n12" / "metrics.json").write_text(json.dumps({"gates": g, "complete": True, "reports": {},
                                                           "points": {}}))
    for h, rs in recs.items():
        (root / "n12" / f"n12_{h}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))


def synth(n=71, F=(1.0, 0.0), u=(0.2, 0.5), rho=0.4, hits=(0.6, 0.6, 0.8), val=0.1, n_dt=None, move=0.5,
          dt_end=True, ident_break=False, u_sd=0.05, drop=(0.5, 0.2)):
    """Records of one half: real, pb and dt units with planted means. F = (real, twins); u = (real, twins);
    hits = pre-role top-10 shares (real, dt, pb)."""
    rng = np.random.default_rng(7)
    out = {}
    for h in ("freeze", "test"):
        rs = []
        for i in range(n):
            rs.append(synth_rec("real", h, i, F[0], u[0], rho, i < hits[0] * n, val, rng, move=move, u_sd=u_sd,
                                drop=drop[0]))
            rs.append(synth_rec("pb", h, i, F[1], u[1], 0.0, i < hits[2] * n, val, rng, move=move, u_sd=u_sd,
                                drop=drop[1]))
            if n_dt is None or i < n_dt:
                rs.append(synth_rec("dt", h, i, F[1], u[1], 0.0, i < hits[1] * n, val, rng, dt_end=dt_end, move=move,
                                    u_sd=u_sd, drop=drop[1]))
        if ident_break and h == "test":
            rs[0]["cf_identity"]["F"] = 0.1
        out[h] = rs
    return out


def verdict_cases(scratch: pathlib.Path, tiny_out: pathlib.Path):
    from infra.pg import m2, n12

    def run_case(name, recs, gates_pass=True):
        root = scratch / f"vc_{name}"
        write_banks(root, recs, gates_pass)
        th = n12.freeze(root) if gates_pass else None
        if th is None:
            write_banks(root, recs, True)
            th = n12.freeze(root)
            write_banks(root, recs, False)
        return n12.verdict(root, th)
    v = run_case("base", synth())
    res = v["u_S_resolution"]
    check("VERDICT A1: the u_S resolution is reported per half (noise from I-GRAD's |dS|)",
          res["freeze"]["n_real"] == 71 and res["test"]["n_within_noise"] == 0 and "score-01" in res["score01"],
          res["test"])
    L = v["labels"]
    check("VERDICT base: OWN_RISE, H_away_sel ABOVE, H_signal SUPPORTED, FORETELLS, H_dt SUPPORTED",
          L["H_cf"]["label"] == "OWN_RISE" and L["H_away_sel_pb"]["label"] == "ABOVE_SELECTION" and
          L["H_signal"]["label"] == "SUPPORTED" and L["H_bridge"]["label"] == "FORETELLS" and
          L["H_dt"]["label"] == "SUPPORTED", v["headline"])
    v = run_case("fall", synth(F=(-1.0, 0.0), u=(0.6, 0.5), rho=-0.4, hits=(0.6, 0.6, 0.6), drop=(0.2, 0.5)))
    L = v["labels"]
    check("VERDICT others fall: OTHERS_FALL, H_away_sel BELOW, H_signal REFUTED, CONTRARY, H_dt UNDECIDED",
          L["H_cf"]["label"] == "OTHERS_FALL" and L["H_away_sel_pb"]["label"] == "BELOW_SELECTION" and
          L["H_signal"]["label"] == "REFUTED" and L["H_bridge"]["label"] == "CONTRARY" and
          L["H_dt"]["label"] == "UNDECIDED" and L["H_sel_dt"]["label"] == "AT_SELECTION", v["headline"])
    v = run_case("mixed", synth(F=(0.0, 0.0), u=(0.47, 0.47), rho=0.0, hits=(0.6, 0.6, 0.6), u_sd=0.2,
                                drop=(0.3, 0.3)))
    L = v["labels"]
    check("VERDICT mixed: MIXED, H_away_sel AT, H_signal UNDECIDED, H_bridge UNDECIDED, H_dt UNDECIDED",
          L["H_cf"]["label"] == "MIXED" and L["H_away_sel_pb"]["label"] == "AT_SELECTION" and
          L["H_signal"]["label"] == "UNDECIDED" and L["H_bridge"]["label"] == "UNDECIDED" and
          L["H_dt"]["label"] == "UNDECIDED", v["headline"])
    v = run_case("dt_below", synth(hits=(0.4, 0.8, 0.8)))
    check("VERDICT dt twins hit more than the carriers: H_sel_dt BELOW_SELECTION, H_dt REFUTED",
          v["labels"]["H_sel_dt"]["label"] == "BELOW_SELECTION" and v["labels"]["H_dt"]["label"] == "REFUTED")
    recs = synth()
    for h in recs:
        for r in recs[h]:
            if r["kind"] == "real":
                r["pull"]["u_gain"] = 0.6
    v = run_case("ceiling", recs)
    check("VERDICT the pull above its ceiling (the realized gain): ABOVE_CEILING", v["labels"]["H_signal"]["label"]
          == "ABOVE_CEILING")
    v = run_case("val_fail", synth(val=10.0))
    check("VERDICT G-VAL fails: every gradient label DOES NOT COUNT, the realized labels stand",
          all(v["labels"][k]["label"] == "DOES NOT COUNT" for k in ("H_signal", "H_bridge", "H_sel_signal_pb"))
          and v["labels"]["H_cf"]["label"] == "OWN_RISE", v["labels"]["H_signal"].get("reason"))
    v = run_case("val_ceiling", synth(val=0.0))
    check("VERDICT G-VAL above its ceiling: the gradient labels DOES NOT COUNT until inspected",
          "until inspected" in v["labels"]["H_signal"].get("reason", ""))
    v = run_case("few_dt", synth(n_dt=30))
    check("VERDICT fewer than 50 dt twins: H_dt NOT_RUN, H_away_sel_dt NOT_RUN",
          v["labels"]["H_dt"]["label"] == "NOT_RUN" and v["labels"]["H_away_sel_dt"]["label"] == "NOT_RUN")
    v = run_case("dt_ctrl", synth(dt_end=False))
    check("VERDICT the dt construction control missing: H_dt DOES NOT COUNT", v["labels"]["H_dt"]["label"]
          == "DOES NOT COUNT")
    recs = synth()
    recs["test"] = [r for r in recs["test"] if not (r["kind"] == "real" and int(r["birth"][1:]) >= 30)]
    v = run_case("missing", recs)
    check("VERDICT 30 real test units: DOES NOT COUNT (missing)", v["headline"].startswith("DOES NOT COUNT") and
          "missing" in v["headline"], v["headline"][:80])
    recs = synth()
    for r in recs["test"]:
        r["cf"]["move"] = -0.2
    v = run_case("nomove", recs)
    check("VERDICT no movement on the test half: DOES NOT COUNT", "no movement" in v["headline"], v["headline"][:80])
    v = run_case("ident", synth(ident_break=True))
    check("VERDICT the decomposition's identity broken: DOES NOT COUNT", "identity" in v["headline"])
    v = run_case("gates", synth(), gates_pass=False)
    check("VERDICT a failed gate: DOES NOT COUNT (gates)", v["headline"].startswith("DOES NOT COUNT: gates"))
    try_root = scratch / "vc_changed"
    write_banks(try_root, synth())
    th = n12.freeze(try_root)
    with open(try_root / "n12" / "n12_freeze.jsonl", "a") as f:
        f.write("\n")
    rc = subprocess.run([sys.executable, "-c", "import json,sys; from infra.pg import n12; "
                         f"n12.verdict('{try_root}', json.loads(sys.argv[1]))", json.dumps(th)],
                        capture_output=True, text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT the freeze records changed after the freeze: the verdict refuses",
          rc.returncode != 0 and "freeze records changed" in rc.stderr)
    rc = subprocess.run([sys.executable, "-c", f"from infra.pg import n12; n12.freeze('{tiny_out}')"],
                        capture_output=True, text=True, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    check("VERDICT the tiny job's freeze refuses (20 real units < 50): the usable-units gate's failing input",
          rc.returncode != 0 and "needs 50" in rc.stderr, MD.last_error(rc))
    fz = m2.read_jsonl(tiny_out / "n12" / "n12_freeze.jsonl")
    s = n12.half_summary(fz, "freeze")
    check("VERDICT the tiny job's freeze half summarizes under the frozen definitions",
          s["n"]["real"] > 0 and "validation" in s and "u_S_real" in s,
          f"F {s['F_real']['mean']:+.3f}, u_S {s['u_S_real']['mean']:.3f}, val rho {s['validation']['rho']:.3f}")


def project():
    from infra.pg import n12
    uj = json.loads((REPO_ROOT / "studies/next/n12/units.json").read_text())
    p = n12.project(uj["units"])
    check("PROJECT the launch's high case fits 0.9 of its hard kill (1.3 x the high projection)",
          p["gpu_h_high"] * 60 <= 0.9 * p["budget_min_hard_kill"],
          f"{p['gpu_h_center']}-{p['gpu_h_high']} GPU-h, kill {p['budget_min_hard_kill']} min")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--sides", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scratch = pathlib.Path(a.out)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.environ.update(HF_TOKEN="dryrun-no-network", HF_HUB_CACHE=str(scratch / "hf_cache"),
                      HF_HUB_DISABLE_TELEMETRY="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")   # A1: as the job's main()
    print("== REAL")
    real(pathlib.Path(a.bank), pathlib.Path(a.sides), pathlib.Path(a.examples), scratch)
    print("== SCORE")
    score_checks()
    print("== INSTR")
    instr(pathlib.Path(a.hf))
    print("== JOB")
    nw, out = job(pathlib.Path(a.hf), scratch)
    print("== FAIL")
    fails(nw)
    print("== VERDICT")
    verdict_cases(scratch, out)
    print("== PROJECT")
    p = project()
    (scratch / "dryrun_summary.json").write_text(json.dumps({"project": p, "checks": RESULTS}, indent=1))
    failed = [nm for nm, ok in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed: {failed}")
    assert not failed, failed
    print("N12_DRYRUN_OK")


if __name__ == "__main__":
    main()
