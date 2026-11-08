"""Weight-free offline gate of N22 (studies/next/07_PICKS/N22_OFFLINE_GATE.md). No weight download, no GPU, no AWS
write. CPU torch 2.12.0 and transformers 5.9.0 (the versions of the banked grid harvest).

    python -m infra.aws_pg.jobs.n22_dryrun --hf <dir with tok/ and config_stage1-step476838.json> \
        --sides studies/next/n22/offline_gate/gate_sides.json --out <scratch dir>

Sections, one CHECK line per check:
  REAL     the registered inputs: items.json reproduces and has the registered sha256; the pool's constraints; the
           bank-side gates of gate_sides.json (K0, G-REV, G-CFG, G-CORPUS, G-TOK: both sides at all 16 points); the
           registered constants equal the banked values; the real registry carries no dry-run key.
  SCORE    the pure functions on small arrays with known answers.
  JOB      a tiny random Olmo 3 (8 layers, hidden 64, the real tokenizer) at four planted checkpoints whose known
           numbers were written by neurons_harvest.run itself; n22_score.main from its entry point in a child process,
           in the three launch modes (L1 15:full; L2 accuracy; L3 n25 + full); every gate passes and every failing
           input fails; the records, the raw examples, one interaction recomputed from the records.
  FAIL     each instance gate on a planted input, in a child process: the job exits non-zero naming it; a planted
           failing input without power stops the job; the stop rule parks.
  VERDICT  freeze and verdict on the tiny job's banks; synthetic records reaching every branch of every label.
  PROJECT  GPU-hours per launch.
Ends with N22_DRYRUN_OK when every check holds.
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

from infra.aws_pg.jobs.m2_dryrun import fake_hub, make_corpus, make_remote, tiny_model

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = []
WINDOW, BATCH = 128, 4
TINY_POINTS = [0, 13, 14, 15]


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:66s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------- REAL
def real(hf: pathlib.Path, sides: pathlib.Path, scratch: pathlib.Path):
    import transformers
    from infra.pg import m2, n22
    from infra.aws_pg.jobs import n22_score as J
    tok = transformers.AutoTokenizer.from_pretrained(hf / "tok")
    p = REPO_ROOT / "studies/next/n22/items.json"
    check("REAL items.json sha256 equals n22_score.ITEMS_SHA", sha(p) == J.ITEMS_SHA, sha(p)[:16])
    re = scratch / "items_regen.json"
    re.write_text(json.dumps(n22.items_record(tok, n22.make_pool())))
    check("REAL items.json reproduces byte for byte from the frozen rule", sha(re) == sha(p))
    rec = json.loads(p.read_text())
    it = rec["items"]
    ex = n22.exemplar_numbers()
    nums = lambda x: {x[k] for k in ("a", "b", "c", "d", "s1", "s2", "y")}     # noqa: E731
    ok = len(it) == 1200 and sum(x["half"] == "freeze" for x in it) == 600
    ok &= all(x["s1"] == x["a"] + x["b"] and x["s2"] == x["s1"] * x["c"] and x["y"] == x["s2"] + x["d"] and
              x["y"] <= 999 and len(nums(x)) == 7 and not ({x["s1"], x["s2"], x["y"]} & ex) for x in it)
    check("REAL pool: 1,200 items, 600 per half, the arithmetic, 7 distinct numbers, no exemplar copy", ok)
    ok = all(len(set(x["v"])) == 5 and all(
        v not in nums(x) and v * x["c"] + x["d"] <= 999 and not ({v * x["c"] + x["d"], v + x["d"]} & (nums(x) | ex | {v}))
        and n22.factorizations(v) for v in x["v"]) for x in it)
    check("REAL pool: five v per item, targets single tokens, distinct from every number of the item", ok)
    ok = all((x["ic"][j]["slot1"][0] + x["ic"][j]["slot1"][1]) * x["c"] + x["d"] == x["v"][j] * x["c"] + x["d"] and
             (x["ic"][j]["slot2"][0] + x["ic"][j]["slot2"][1]) * x["ic"][j]["slot2"][2] == x["v"][j] and
             x["ic"][j]["slot1"][2:] == [x["c"], x["d"]] and x["ic"][j]["slot2"][3] == x["d"] for x in it for j in range(2))
    check("REAL counterfactuals: slot-1 sum = v (same c, d), slot-2 product = v (same d)", ok)
    ok = all(it[x["partner"]]["half"] == x["half"] and (it[x["partner"]]["c"], it[x["partner"]]["d"]) != (x["c"], x["d"])
             and all(v * it[x["partner"]]["c"] + it[x["partner"]]["d"] <= 999 for v in x["v"]) for x in it)
    check("REAL partners: same half, other (c, d), shuffled targets single tokens", ok)
    cp = rec["class_prompts"]
    ok = all((q["abcd"][0] + q["abcd"][1] if k == "1" else (q["abcd"][0] + q["abcd"][1]) * q["abcd"][2]) == int(xv)
             for k, xs in cp.items() for xv, qs in xs.items() for q in qs) and \
        all(len(qs) == n22.N_CLASS for xs in cp.values() for qs in xs.values())
    check("REAL class prompts: 16 per value, the slot's intermediate equals the value",
          ok, f"{sum(len(x) for x in cp.values())} values")
    s = json.loads(sides.read_text())
    check("REAL gate_sides.json was computed on this items.json", s["items_sha256"] == sha(p))
    for g, v in s["gates"].items():
        check(f"REAL bank-side gate {g}: pass side passes, failing input fails at every point",
              v["pass_side_ok"] and v["fail_side_ok"], v["summary"])
    k0 = s["gates"]["K0"]["points"]
    check("REAL K0: every failing input (the neighbour's bank) exceeds its tier by >= 3 x",
          min(v["gap_over_tol"] for v in k0.values()) >= 3.0, min(round(v["gap_over_tol"], 2) for v in k0.values()))
    reg = J.registry()
    check("REAL the real registry has no dry-run key and the registered revisions",
          "dry_all_answered" not in reg and all(reg["points"][t][3] == m2.POINTS[t][3] for t in range(16)))
    check("REAL the real registry: target block 30, sources 0-29, upper half 16-29, 24 lens prompts",
          n22.TARGET == 30 and n22.SOURCES == list(range(30)) and n22.UPPER == list(range(16, 30)) and
          reg["lens"]["n"] == 24)


# ------------------------------------------------------------------------------------------------------- SCORE
def score_checks():
    import torch
    from infra.pg import n22
    rng = np.random.default_rng(0)
    lp1 = {"f1": -1.0, "f2": -5.0}
    lp2 = {"f1": -6.0, "f2": -2.0}
    check("SCORE interaction: [(-1)-(-6)] + [(-2)-(-5)] = 8", n22.interaction(lp1, lp2) == 8.0)
    c1, c2 = 3.7, -1.2
    check("SCORE interaction is unchanged by a log-partition shift per edit",
          abs(n22.interaction({k: v + c1 for k, v in lp1.items()}, {k: v + c2 for k, v in lp2.items()}) - 8.0) < 1e-12)
    h = torch.randn(3, 5, 16, dtype=torch.float64)
    V = torch.randn(3, 16, 2, dtype=torch.float64)
    d = n22.lens_swap_delta(h, V)
    Vp = torch.linalg.pinv(V)
    c = Vp @ h.transpose(1, 2)
    c2_ = Vp @ (h + d).transpose(1, 2)
    perp = lambda x: x - (V @ (Vp @ x.transpose(1, 2))).transpose(1, 2)     # noqa: E731
    check("SCORE lens swap: the coordinates exchange; the orthogonal part is unchanged",
          torch.allclose(c2_, c.flip(1)) and torch.allclose(perp(h + d), perp(h)))
    check("SCORE lens swap with sigma the identity is exactly zero", n22.lens_swap_delta(h, V, sigma=False).abs().max() == 0)
    from infra.aws_pg.jobs.n22_score import dom_edit, lens_edit
    g = torch.Generator().manual_seed(0)
    t = torch.randn(2, 7, 16)
    Vf = torch.randn(2, 16, 2)
    a_ = lens_edit(t, Vf, 3) - t
    r_ = lens_edit(t, Vf, 3, rand=True, gen=g) - t
    check("SCORE random lens edit: the same norm per position, another direction",
          torch.allclose(a_.norm(dim=-1), r_.norm(dim=-1), atol=1e-5) and not torch.allclose(a_, r_))
    dv = torch.randn(2, 16)
    e1 = dom_edit(t, dv, 6) - t
    e2 = dom_edit(t, dv, 6, rand=True, gen=g) - t
    check("SCORE mean-difference patch: one position; its random floor has the same norm",
          torch.allclose(e1[:, 6], dv) and e1[:, :6].abs().max() == 0 and
          torch.allclose(e2[:, 6].norm(dim=-1), dv.norm(dim=-1), atol=1e-5))
    lg = torch.tensor([[0.0, 2.0, 2.0, 1.0]])
    check("SCORE ranks: 1 + strictly larger (a tie shares the rank)",
          n22.ranks_of(lg, torch.tensor([[1, 2, 3, 0]])).tolist() == [[1, 1, 3, 4]])
    X = torch.randn(50, 8, dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))[0]
    Y = torch.randn(50, 8, dtype=torch.float64)
    ck = (n22.linear_cka(X @ X.T, X @ X.T), n22.linear_cka(X @ X.T, (X @ Q) @ (X @ Q).T), n22.linear_cka(X @ X.T, Y @ Y.T))
    check("SCORE linear CKA: 1 on itself, 1 under rotation, lower on unrelated",
          abs(ck[0] - 1) < 1e-9 and abs(ck[1] - 1) < 1e-9 and ck[2] < 0.9, [round(x, 3) for x in ck])
    K = X @ X.T
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    L_ = Y @ Y.T
    Lc = L_ - L_.mean(0, keepdim=True) - L_.mean(1, keepdim=True) + L_.mean()
    check("SCORE the job's centred-Gram CKA equals linear_cka",
          abs(float((Kc / Kc.norm() * Lc / Lc.norm()).sum()) - ck[2]) < 1e-9)
    blk = np.kron(np.eye(3), np.ones((10, 10)))
    check("SCORE cka_block: block-diagonal > 0, constant = 0", n22.cka_block(blk) > 0.5 and n22.cka_block(np.ones((30, 30))) == 0)
    z = torch.randn(4, 200000, generator=torch.Generator().manual_seed(1))
    lap = torch.distributions.Laplace(0.0, 1.0).sample((4, 200000))
    k1, k2 = n22.excess_kurtosis(z), n22.excess_kurtosis(lap)
    check("SCORE excess kurtosis: Gaussian about 0, Laplace about 3",
          float(k1.abs().max()) < 0.05 and float((k2 - 3).abs().max()) < 0.2, [round(float(k1.mean()), 3), round(float(k2.mean()), 3)])
    b = n22.boot_mean(np.ones(40))
    check("SCORE bootstrap of a constant: interval is the point", b["lo"] == b["hi"] == 1.0)
    br = n22.band_rule(np.array([0.1, 0.5, 0.5, 0.2]), np.array([0.0, 0.1, 0.4, 0.6]))
    check("SCORE band rule: l1 the first maximum (tie: lower layer), l2, the matched layer and its band",
          br == {"l1": 1, "l2": 3, "lm": 2, "band_m": [1, 2, 3], "matched": 0.4}, br)
    hit = np.r_[np.ones(40), np.zeros(10)]
    sh = np.r_[np.ones(5), np.zeros(45)]
    check("SCORE readout PRESENT and NOT_PRESENT",
          n22.readout_present(hit, sh, 0.1)["label"] == "PRESENT" and n22.readout_present(sh, sh, 0.1)["label"] == "NOT_PRESENT")
    check("SCORE factorizations of 63: (31? no) 21x3, 9x7, 7x9", n22.factorizations(63) == [(21, 3), (9, 7), (7, 9)])
    _ = rng


# ------------------------------------------------------------------------------------------------------- JOB
def tiny_cfg(hf: pathlib.Path) -> dict:
    c = json.loads((hf / "config_stage1-step476838.json").read_text())
    c.update(hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
             layer_types=c["layer_types"][:8])
    return c


def child(spec_path: str):
    """Entry point of a child process: n22_score.main with the stand-in hub and the planted failures of the spec."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    import torch
    from infra.pg import n22
    import infra.aws_pg.jobs.n22_score as J
    J.HUB = fake_hub(spec["remotes"], spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    plant = spec.get("plant")
    if plant == "I_GRAD":                      # the repeated pass of prompt 0 is not the same
        sys.path.insert(0, J.VENDOR)
        import jlens.fitting as F
        orig, n = F.jacobian_for_prompt, {"c": 0}

        def noisy(*a, **k):
            J_, s_, v_ = orig(*a, **k)
            n["c"] += 1
            if n["c"] == 25:
                J_ = {key: x + 1e-3 * torch.randn_like(x) for key, x in J_.items()}
            return J_, s_, v_
        F.jacobian_for_prompt = noisy
    if plant == "LENS_FIDELITY":               # a lens that transports nothing: random matrices in place of J
        orig_fit = J.fit_lens

        def bad_fit(*a, **k):
            A, B, g, times = orig_fit(*a, **k)
            gen = torch.Generator().manual_seed(7)
            d = next(iter(A.values())).shape[0]
            R = {key: torch.randn(d, d, generator=gen) / d ** 0.5 for key in A}
            return R, R, g, times
        J.fit_lens = bad_fit
    if plant == "I_FWD":                       # the second pool pass differs
        orig_pf, n2 = J.Point.pool_forward, {"c": 0}

        def pf(self, order):
            out = orig_pf(self, order)
            n2["c"] += 1
            return (out[0] + 1e-2,) + out[1:] if n2["c"] == 2 else out
        J.Point.pool_forward = pf
    if plant == "I_EDIT":                      # the zero edit is not zero
        orig_d = n22.lens_swap_delta
        n22.lens_swap_delta = lambda h, V, sigma=True: orig_d(h, V, sigma) + (0 if sigma else 1e-1)
    if plant == "G_IC_ID":                     # the patch at the embedding hits the wrong positions
        n22.IC_POS = {k: [p + 1 for p in v] for k, v in n22.IC_POS.items()}
    if plant == "K0_NO_POWER":                 # a tolerance so wide that the neighbour's bank passes
        n22.K0_TOL_SAME = 10.0
    J.main(spec["argv"])


def run_child(world, name, points, reg_over=None, plant=None, budget=600.0, items=None):
    scratch = world["scratch"]
    rp = scratch / f"registry_{name}.json"
    rp.write_text(json.dumps(dict(world["reg"], **(reg_over or {}))))
    out = scratch / f"out_{name}"
    env = dict(world["env"], HF_HUB_CACHE=str(scratch / f"hf_cache_{name}"))
    argv = ["--out", str(out), "--items", str(items or world["items"]), "--corpus", str(world["corpus"]),
            "--points", points, "--budget-min", str(budget), "--registry", str(rp), "--device", "cpu"]
    spec = dict(world["spec"], argv=argv, plant=plant)
    sp = scratch / f"child_{name}.json"
    sp.write_text(json.dumps(spec))
    rc = subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import n22_dryrun as D; D.child(sys.argv[1])",
                         str(sp)], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    (scratch / f"log_{name}.txt").write_text(rc.stdout + "\n----- stderr\n" + rc.stderr)
    return rc, out


def last_error(rc) -> str:
    lines = [x for x in rc.stderr.strip().splitlines() if x.strip()]
    return lines[-1][:170] if lines else ""


def world_build(hf: pathlib.Path, scratch: pathlib.Path) -> dict:
    import torch
    import transformers
    from infra.pg import m2, n22
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    corpus_dir = scratch / "corpus"
    make_corpus(corpus_dir, hf, 64)
    corpus = NC.load_tok(corpus_dir, "olmo3")
    tok = transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"])
    cfg = tiny_cfg(hf)
    models = {15: tiny_model(cfg, 15), 0: tiny_model(cfg, 100)}
    for t, nz in ((14, 0.3), (13, 0.5)):
        models[t] = tiny_model(cfg, t, models[15], nz)
    points = [[t, f"{'1' if t in (0, 1, 2, 3, 4, 5, 6, 15) else '1b'}/out/t{t:02d}", f"stage1-step{t}",
               hashlib.sha1(f"stage1-step{t}".encode()).hexdigest(), "cpu"] for t in range(16)]
    remotes, commits, bank = {}, {}, {}
    for t in TINY_POINTS:
        d = scratch / "remote" / points[t][2]
        make_remote(d, models[t], cfg, hf)
        remotes[points[t][2]], commits[points[t][2]] = str(d), points[t][3]
        model, _ = NH.load_model(str(d), "O7", device="cpu")
        g = scratch / "grid" / points[t][1]
        NH.run(g, model=model, tok=tok, corpus=corpus, model_key="O7", repo=m2.REPO, revision="bank",
               branch=points[t][2], max_tokens=64 * WINDOW, window=WINDOW, batch=BATCH, sample_neurons=2, device="cpu",
               nll_tokens=1000)
        bank[t] = json.loads((g / "controls.json").read_text())["nll_corpus_first_120k"]["mean_nll"]
    k0 = {t: bank.get(t, bank[15] + 0.5 + 0.01 * t) for t in range(16)}
    rec = n22.items_record(tok, n22.make_pool(n=120))
    ip = scratch / "items_tiny.json"
    ip.write_text(json.dumps(rec))
    reg = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "items_sha": sha(ip),
           "tokens_sha256": corpus["tokens_sha256"], "points": {t: p for t, p in enumerate(points)}, "ref_config": cfg,
           "bos": 100257, "k0_bank": k0, "k0": {"targets": 1000, "window": WINDOW, "batch": BATCH},
           "lens": {"n": 24, "len": 128, "dim_batch": 16}, "n_swap_max": 60, "n_min": 50,
           "estimate_h": {"full": 0.05, "accuracy": 0.01, "n25": 0.02}, "cpu_device_name": "cpu", "dry_all_answered": True}
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return {"scratch": scratch, "items": ip, "corpus": corpus_dir, "reg": reg, "env": env, "rec": rec, "bank": bank,
            "spec": {"remotes": remotes, "commits": commits}}


def job(world: dict):
    from infra.pg import n22
    rc, out = run_child(world, "L1", "15:full")
    check("JOB L1 (15:full) exits 0 and prints N22_JOB_OK", rc.returncode == 0 and "N22_JOB_OK" in rc.stdout,
          f"exit {rc.returncode} {last_error(rc)}")
    if rc.returncode != 0:
        print(rc.stderr[-5000:])
        raise SystemExit(1)
    m = json.loads((out / "n22" / "metrics.json").read_text())
    check("JOB L1 metrics complete, nothing parked", m["complete"] is True and m["parked"] == [])
    check("JOB L1 every gate passed", all(g["pass"] for g in m["gates"].values()), sorted(m["gates"]))
    fi = [n for n, g in m["gates"].items() if "failing_input" in g]
    check("JOB L1 every registered failing input failed", all(not m["gates"][n]["failing_input"]["pass"] for n in fi),
          f"{len(fi)} failing inputs")
    k0 = m["gates"]["t15_K0"]
    check("JOB L1 K0 equals the bank written by neurons_harvest.run, bit for bit",
          k0["diff"] == 0.0 and k0["nll"] == world["bank"][15], f"{k0['nll']!r}")
    ig = m["gates"]["t15_I_GRAD"]
    check("JOB L1 I-GRAD: the repeated pass is bit-identical; prompt 1 differs",
          ig["bit_identical"] and ig["failing_input"]["min_rel"] > 1e-3, f"min rel of the failing input {ig['failing_input']['min_rel']:.3g}")
    for gname in ("I_FWD", "I_EDIT", "G_IC_ID_freeze", "G_IC_ID_test", "LENS_FIDELITY", "G_TOK"):
        g = m["gates"][f"t15_{gname}"]
        check(f"JOB L1 {gname} passes; its failing input fails", g["pass"] and not g["failing_input"]["pass"],
              json.dumps({k: g[k] for k in g if k in ("max_abs", "cos_lens", "cos_rotated", "n")}))
    keys = json.dumps(m)
    check("JOB L1 metrics.json carries no interaction, share or label", all(x not in keys for x in ('"I"', "ABOVE", "label")))
    d = out / "n22" / "t15"
    files = ["design.json", "accuracy.json", "readout.npz", "clean_lp_numbers.npy", "massive.json", "lens_AB.pt",
             "probes.json", "n25.json", "edits_freeze.jsonl", "edits_test.jsonl", "ic_freeze.jsonl", "ic_test.jsonl",
             "dom_sweep_freeze.jsonl", "raw_example_edits.jsonl"]
    check("JOB L1 every record file written", all((d / f).exists() for f in files), [f for f in files if not (d / f).exists()])
    de = json.loads((d / "design.json").read_text())
    check("JOB L1 bands within the source layers; tiny constants (target 6, upper 4-5)",
          set(de["bands"]["band_m"]) <= set(range(6)) and 0 <= de["bands"]["lm"] < 6, de["bands"])
    ed = n22.read_jsonl(d / "edits_test.jsonl")
    want = 60 * n22.N_V * 2 * len(n22.ARMS)
    check("JOB L1 one edit row per (item, v, arm, slot) in the test half", len(ed) == want, f"{len(ed)} of {want}")
    check("JOB L1 the edit rows split by half", {world["rec"]["items"][r["item"]]["half"] for r in ed} == {"test"})
    ic = n22.read_jsonl(d / "ic_freeze.jsonl")
    sites = sorted({r["site"] for r in ic})
    check("JOB L1 interchange control: the freeze half at every site, both kinds", sites == [-1, 0, 4] and
          {r["kind"] for r in ic} == {"IC", "RAND_IC"}, sites)
    it0 = {x["i"]: x for x in world["rec"]["items"]}
    r1 = next(r for r in ed if r["arm"] == "LENS_M" and r["slot"] == 1)
    r2 = next(r for r in ed if r["arm"] == "LENS_M" and r["slot"] == 2 and r["item"] == r1["item"] and r["vi"] == r1["vi"])
    I = n22.interaction(r1["lp"], r2["lp"])
    I2 = n22.item_interactions(ed, "LENS_M")
    one = np.mean([n22.interaction(a["lp"], b["lp"]) for a in ed for b in ed if a["arm"] == b["arm"] == "LENS_M" and
                   a["item"] == b["item"] == r1["item"] and a["vi"] == b["vi"] and a["slot"] == 1 and b["slot"] == 2])
    check("JOB L1 an item's interaction recomputes from the records", abs(I2[r1["item"]] - one) < 1e-12, f"I(first v) {I:.4f}")
    v = it0[r1["item"]]["v"][r1["vi"]]
    num = world["rec"]["number_ids"]
    cl = np.load(d / "clean_lp_numbers.npy")
    check("JOB L1 the scored tokens are the item's f1, f2 (number ids) and the clean table has 1,000 columns",
          cl.shape == (120, 1000) and r1["lp"]["f1"] != r1["lp"]["f2"] and
          v * it0[r1["item"]]["c"] + it0[r1["item"]]["d"] <= 999 and len(num) == 1000)
    ex = n22.read_jsonl(out / "n22" / "samples" / "raw_examples_t15.jsonl")
    ok = len(ex) >= 10 and all(e["prompt"].endswith(" = ") and "LENS_M_slot1" in e["after_edits"] and
                               len(e["readout"]["lm"]["top10"]) == 10 for e in ex)
    check("JOB L1 raw examples: >= 10, the prompt, top-10 readouts, every arm", ok, f"{len(ex)} examples")
    pr = json.loads((d / "probes.json").read_text())
    check("JOB L1 probes: linear and MLP, real and permuted, at the probe layers",
          all(f"s1_L{pr['layers'][0]:02d}_{a}_{b}" in pr["acc"] for a in ("real", "permuted") for b in ("linear", "mlp")))
    n25 = json.loads((d / "n25.json").read_text())
    check("JOB L1 N25 statistics for the four lens variants", sorted(n25) == ["A", "B", "full", "proj"] and
          all(np.isfinite(n25[k]["cka_block"]) for k in n25))
    world["out_L1"] = out
    rc2, out2 = run_child(world, "L2", "14:accuracy,13:accuracy,0:accuracy")
    m2_ = json.loads((out2 / "n22" / "metrics.json").read_text()) if (out2 / "n22" / "metrics.json").exists() else {}
    check("JOB L2 (accuracy at three points) exits 0; K0 and the accuracy counts at each",
          rc2.returncode == 0 and all(f"t{t:02d}_K0" in m2_["gates"] and "answered" in m2_["points"][str(t)]
                                      for t in (14, 13, 0)), f"exit {rc2.returncode} {last_error(rc2)}")
    pl = n22.plan_lens({t: m2_["points"].get(str(t), {"answered": {"freeze": 0, "test": 0}})["answered"]
                        if str(t) in m2_.get("points", {}) else {"freeze": 0, "test": 0} for t in range(15)})
    check("JOB L2 plan-lens from the scan (T_13, T_14 answered in the tiny world)", pl["n22_points"] == [13, 14] and
          pl["order"][0] == 0, pl)
    rc3, out3 = run_child(world, "L3", "0:n25,14:full,13:full")
    m3 = json.loads((out3 / "n22" / "metrics.json").read_text()) if (out3 / "n22" / "metrics.json").exists() else {}
    check("JOB L3 (n25, full, full) exits 0, complete", rc3.returncode == 0 and m3.get("complete") is True,
          f"exit {rc3.returncode} {last_error(rc3)}")
    check("JOB L3 the n25 point has the lens, the readout and N25 but no edits",
          (out3 / "n22/t00/n25.json").exists() and not (out3 / "n22/t00/edits_test.jsonl").exists() and
          "t00_LENS_FIDELITY" in m3.get("reports", {}))
    world["out_L3"] = out3
    return out


# ------------------------------------------------------------------------------------------------------- FAIL
def fails(world: dict):
    cases = [("G_SAME", {"versions": {"torch": "0", "transformers": "0.0.0"}}, None, "G_SAME failed"),
             ("INPUTS", {"items_sha": "f" * 64}, None, "INPUTS failed"),
             ("G_CORPUS", {"tokens_sha256": "e" * 64}, None, "G_CORPUS failed"),
             ("G_REV", {"points": {**world["reg"]["points"], 15: world["reg"]["points"][15][:3] + ["1" * 40, "cpu"]}}, None,
              "t15_G_REV failed"),
             ("G_CFG", {"ref_config": {**world["reg"]["ref_config"], "hidden_size": 65}}, None, "t15_G_CFG failed"),
             ("K0", {"k0_bank": {**world["reg"]["k0_bank"], 15: world["reg"]["k0_bank"][15] + 1e-3}}, None, "t15_K0 failed"),
             ("K0 without power", None, "K0_NO_POWER", "t15_K0: its failing input passed"),
             ("I_GRAD", None, "I_GRAD", "t15_I_GRAD failed"),
             ("LENS_FIDELITY", None, "LENS_FIDELITY", "t15_LENS_FIDELITY failed"),
             ("I_FWD", None, "I_FWD", "t15_I_FWD failed"),
             ("I_EDIT", None, "I_EDIT", "t15_I_EDIT failed"),
             ("G_IC_ID", None, "G_IC_ID", "t15_G_IC_ID_freeze failed")]
    for name, over, plant, want in cases:
        rc, _ = run_child(world, f"fail_{name.replace(' ', '_')}", "15:full", reg_over=over, plant=plant)
        check(f"FAIL {name}: the job exits non-zero naming it", rc.returncode != 0 and want in rc.stderr, last_error(rc))
    rec = json.loads(world["items"].read_text())
    rec["items"][0]["ids"] = rec["items"][0]["ids"][:1] + [rec["items"][0]["ids"][2]] + rec["items"][0]["ids"][2:]
    bad = world["scratch"] / "items_bad.json"
    bad.write_text(json.dumps(rec))
    rc, _ = run_child(world, "fail_G_TOK", "15:full", reg_over={"items_sha": sha(bad)}, items=bad)
    check("FAIL G_TOK (a registered prompt that the tokenizer does not produce)", rc.returncode != 0 and
          "t15_G_TOK failed" in rc.stderr, last_error(rc))
    rc, out = run_child(world, "park", "0:n25,14:full,13:full", reg_over={"estimate_h": {"full": 5.0, "accuracy": 1.0,
                                                                                       "n25": 1.0}}, budget=30.0)
    m = json.loads((out / "n22" / "metrics.json").read_text())
    check("FAIL stop rule: the projection parks the remaining points; the job ends normally",
          rc.returncode == 0 and m["parked"] == [14, 13] and m["complete"] is False, m.get("projection"))


# ------------------------------------------------------------------------------------------------------- VERDICT
def synth_rows(n_items, I_real, I_rand, I_shuf, sd=1.0, seed=0, ans=2.0, pc=1.0):
    """Edit rows whose per-item interactions are planted: LENS_M ~ I_real, RAND_M ~ I_rand, its shuffle ~ I_shuf."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for vi in range(5):
            for arm, base, sh in (("LENS_M", I_real, I_shuf), ("RAND_M", I_rand, 0.0), ("LENS_U", I_real, I_shuf),
                                  ("RAND_U", I_rand, 0.0), ("DOM_M", I_real, I_shuf), ("RAND_DOM", I_rand, 0.0),
                                  ("ANS_M", 0.0, 0.0), ("ANS_U", ans, 0.0)):
                x = base + sd * rng.standard_normal()
                xs = sh + sd * rng.standard_normal()
                f1 = x / 2 + (pc if arm == "ANS_U" else 0.0)
                rows.append({"item": i, "vi": vi, "arm": arm, "slot": 1,
                             "lp": {"f1": f1, "f2": 0.0, "f1x": xs / 2, "f2x": 0.0}})
                rows.append({"item": i, "vi": vi, "arm": arm, "slot": 2,
                             "lp": {"f1": 0.0, "f2": x / 2 + (pc if arm == "ANS_U" else 0.0), "f1x": 0.0, "f2x": xs / 2}})
    return rows


def verdict_cases(world: dict):
    from infra.pg import n22
    out = world["out_L1"]
    thr = n22.freeze(out, world["scratch"] / "thr_L1.json")
    t = thr["points"]["15"]
    check("VERDICT freeze on the tiny L1 banks: delta_read, bands, matched, DoM guard, PC, IC, go/no-go, test sha256",
          all(k in t for k in ("delta_read", "bands", "matched", "dom_guard_answer_readable", "pc_ans", "ic", "go_nogo",
                               "test_sha256", "DISCOVERY_freeze_interactions")), sorted(t))
    v = n22.point_verdict(out / "n22" / "t15", t, world["rec"])
    check("VERDICT point verdict on the tiny banks: three arms labelled, readout, reports",
          all(a in v["slot"] for a in ("LENS_M", "LENS_U", "DOM_M")) and "readout" in v and "reports" in v,
          {a: v["slot"][a]["label"] for a in v["slot"]})
    f = out / "n22" / "t15" / "edits_test.jsonl"
    keep = f.read_bytes()
    f.write_bytes(keep + b"\n")
    rc = subprocess.run([sys.executable, "-c", "import json,sys; from infra.pg import n22; "
                         "n22.point_verdict(sys.argv[1], json.load(open(sys.argv[2]))['points']['15'], json.load(open(sys.argv[3])))",
                         str(out / "n22" / "t15"), str(world["scratch"] / "thr_L1.json"), str(world["items"])],
                        capture_output=True, text=True, cwd=REPO_ROOT, env=world["env"])
    f.write_bytes(keep)
    check("VERDICT refuses a test file changed after freeze", rc.returncode != 0 and "changed after freeze" in rc.stderr)
    ceil = {"hi": 100.0}
    cases = {"ABOVE_FLOORS": (2.0, 0.0, 0.0), "AT_FLOOR": (0.05, 0.0, 0.0), "BELOW_FLOOR": (-2.0, 0.0, 0.0)}
    for want, (a, b, c) in cases.items():
        lab = n22.slot_label(synth_rows(60, a, b, c), "LENS_M", "RAND_M", ceil)["label"]
        check(f"VERDICT slot label {want} on planted interactions (60 items)", lab == want, lab)
    lab = n22.slot_label(synth_rows(60, 2.0, 0.0, 2.0), "LENS_M", "RAND_M", ceil)["label"]
    check("VERDICT slot label: above the random floor but at the shuffle -> AT_FLOOR", lab == "AT_FLOOR", lab)
    lab = n22.slot_label(synth_rows(60, 50.0, 0.0, 0.0), "LENS_M", "RAND_M", {"hi": 1.0})["label"]
    check("VERDICT slot label above the ceiling -> DOES NOT COUNT", lab.startswith("DOES NOT COUNT: above"), lab)
    lab = n22.slot_label(synth_rows(30, 2.0, 0.0, 0.0), "LENS_M", "RAND_M", ceil)["label"]
    check("VERDICT slot label with 30 items -> DOES NOT COUNT", lab.startswith("DOES NOT COUNT: fewer"), lab)
    # room: at N_MIN = 50 items and a per-(item, v) SD of 1 nat, the smallest effect that reaches ABOVE
    room = {}
    for eff in (0.25, 0.5, 1.0):
        room[eff] = sum(n22.slot_label(synth_rows(50, eff, 0.0, 0.0, seed=s), "LENS_M", "RAND_M", ceil)["label"] ==
                        "ABOVE_FLOORS" for s in range(20)) / 20
    check("VERDICT room at 50 items (SD 1 nat per edit): an interaction of 1 nat reaches ABOVE in >= 0.9 of draws",
          room[1.0] >= 0.9, room)
    world["room"] = room
    # interchange control and go/no-go
    icr = []
    for i in range(60):
        for vi in range(2):
            for kind, x, xs in (("IC", 3.0, 0.0), ("RAND_IC", 0.0, 0.0)):
                e = np.random.default_rng(i * 7 + vi).standard_normal()
                for s, k in ((1, "f1"), (2, "f2")):
                    icr.append({"item": i, "vi": vi, "slot": s, "site": -1, "kind": kind,
                                "lp": {"f1": (x + e) / 2 if s == 1 else 0.0, "f2": (x + e) / 2 if s == 2 else 0.0,
                                       "f1x": xs if s == 1 else 0.0, "f2x": xs if s == 2 else 0.0}})
    g = n22.ic_go(icr)
    check("VERDICT interchange control fires on a planted effect", g["fires"], g["minus_rand"])
    g0 = n22.ic_go([dict(r, lp={k: 0.0 for k in r["lp"]}) for r in icr])
    check("VERDICT interchange control does not fire on no effect", not g0["fires"])
    check("VERDICT go/no-go: GO at 100 answered with the control; NO-GO at 99; NO-GO without the control",
          n22.go_nogo(100, g)["go"] and not n22.go_nogo(99, g)["go"] and not n22.go_nogo(500, g0)["go"])
    # order
    mk = lambda r, s: {"readout_both": "PRESENT" if r else "NOT_PRESENT",            # noqa: E731
                       "slot": {"LENS_M": {"label": "ABOVE_FLOORS" if s else "AT_FLOOR"}}}
    cases = {"SUPPORTED": {11: mk(1, 0), 13: mk(1, 1), 15: mk(1, 1)}, "CENSORED": {11: mk(1, 1), 15: mk(1, 1)},
             "REFUTED_SAME": {11: mk(0, 0), 13: mk(1, 1), 15: mk(1, 1)},
             "REFUTED_CONTRARY": {11: mk(0, 1), 13: mk(1, 1), 15: mk(1, 1)}, "NOT_RUN": {11: mk(1, 0), 15: mk(1, 0)}}
    for want, pp in cases.items():
        lab = n22.order_label({str(k): v for k, v in pp.items()}, "LENS_M")["label"]
        check(f"VERDICT H_order {want}", lab == want, lab)
    check("VERDICT H_order NOT_RUN without a dating arm", n22.order_label({"15": mk(1, 1)}, None)["label"] == "NOT_RUN")
    check("VERDICT the dating arm: LENS_M first, then LENS_U, then DOM_M",
          n22.dating_arm({"slot": {"LENS_M": {"label": "AT_FLOOR"}, "LENS_U": {"label": "ABOVE_FLOORS"}}}) == "LENS_U")
    # N25
    st = lambda f, a, b, p: {"cka_block": {"full": f, "A": a, "B": b, "proj": p}}    # noqa: E731
    lab = n22.n25_labels({0: st(0.1, 0.1, 0.11, 0.1), 8: st(0.5, 0.5, 0.51, 0.5), 15: st(0.6, 0.6, 0.61, 0.9)})
    check("VERDICT N25: LEARNED, FORMS_BETWEEN (T_0, T_8), SINK_DEPENDENT at T_15, SURVIVES at T_8",
          lab["arch"]["cka_block"]["label"] == "LEARNED" and lab["form"]["cka_block"]["first_interval"] == [0, 8] and
          lab["sink"]["cka_block"]["15"]["label"] == "SINK_DEPENDENT" and lab["sink"]["cka_block"]["8"]["label"] == "SURVIVES")
    lab = n22.n25_labels({0: st(0.5, 0.4, 0.6, 0.5), 15: st(0.55, 0.45, 0.65, 0.55)})
    check("VERDICT N25: ARCHITECTURAL within split-half noise; NO_CHANGE", lab["arch"]["cka_block"]["label"] ==
          "ARCHITECTURAL" and lab["form"]["cka_block"]["label"] == "NO_CHANGE")
    lab = n22.n25_labels({0: st(0.1, 0.05, 0.15, 0.1), 4: st(0.18, 0.13, 0.23, 0.18), 8: st(0.26, 0.21, 0.31, 0.26),
                          15: st(0.34, 0.29, 0.39, 0.34)})
    check("VERDICT N25: SMOOTH when no interval exceeds the noise but the total does", lab["form"]["cka_block"]["label"] == "SMOOTH")
    check("VERDICT massive formation: the first point at 10 x the step-0 ratio",
          n22.massive_formed({0: 3.0, 4: 20.0, 8: 40.0, 15: 50.0})["first"] == 8)
    # plan-lens
    acc = {t: {"freeze": 60 if t >= 9 else 10, "test": 60 if t >= 9 else 10} for t in range(15)}
    p = n22.plan_lens(acc)
    check("VERDICT plan-lens with six candidates: first, last, two at the thirds; T_0 and the point before",
          p["n22_points"] == [9, 11, 12, 14] and p["n25_points"] == [0, 8] and p["order"] == [0, 14, 12, 11, 9, 8], p)
    p = n22.plan_lens({t: {"freeze": 60 if t >= 13 else 0, "test": 60 if t >= 13 else 0} for t in range(15)})
    check("VERDICT plan-lens with two candidates", p["n22_points"] == [13, 14] and p["order"] == [0, 14, 13, 12], p)
    p = n22.plan_lens({t: {"freeze": 0, "test": 0} for t in range(15)})
    check("VERDICT plan-lens with none: T_0 only", p["n22_points"] == [] and p["order"] == [0], p)


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
    hf = pathlib.Path(a.hf)
    real(hf, pathlib.Path(a.sides), scratch)
    score_checks()
    world = world_build(hf, scratch)
    job(world)
    fails(world)
    verdict_cases(world)
    from infra.pg import n22
    pr = n22.project()
    check("PROJECT L1, L2, L3 GPU-hours and hard kills at 1.3 x the high projection",
          all(pr[f"{L}_kill_min"] == int(np.ceil(1.3 * pr["high"][f"{L}_h"] * 60)) for L in ("L1", "L2", "L3")),
          {k: pr["center"][k] for k in ("L1_h", "L2_h", "L3_h", "total_h")} | {"high": pr["high"]["total_h"]})
    n_fail = sum(not ok for _, ok in RESULTS)
    summary = {"n_checks": len(RESULTS), "n_failed": n_fail, "failed": [n for n, ok in RESULTS if not ok],
               "room": world.get("room"), "projection": pr}
    (scratch / "dryrun_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(RESULTS)} checks, {n_fail} failed")
    if n_fail == 0:
        print("N22_DRYRUN_OK")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
