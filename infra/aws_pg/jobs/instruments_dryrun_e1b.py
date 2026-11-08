"""Weight-free dry run of INSTRUMENTS E1b (studies/instruments/SPEC_instruments.md §3.6, §10, §12 step 12). No download and no network: tiny random
Gemma 3 and Qwen 3 (9 layers, MLP width 128, built from their transformers 5.20.2 configs, every RMSNorm gain jittered so a gain the adapter
dropped would show) and a tiny Olmo 3, a byte-level tokenizer and a fake corpus, as the E1 dry run builds them.

    python -m infra.aws_pg.jobs.instruments_dryrun_e1b --out <dir>

Part 1, pure logic and the real inputs (read only): instruments_values, instruments_linear and instruments_adapter selfchecks; the committed
inputs/e1b_items.json checked (per model 56 pairs 18/20/18, 12 contexts per side from separate documents, no document or marked token shared across
sides, 30 singles in 15 same-layer partner pairs, 3 known poles x 16, 200 background positions off window position 0) and the run's size: readouts
per item (the job's arm plan, the same count the tiny run checks), tokens by reuse_ablate.n_forwarded with each model's BOS, timed at SPEC6's
measured L40S rate on Olmo-3-7B (an upper bound for 4B models) plus the harvest's load time.
Part 2, the adapter's code-level known number: instruments_adapter.olmo_equality on a tiny Olmo 3 (capture, Rows, direct_A, c, completeness, ā,
the background capture equal to the REUSED path exactly); and on the tiny Gemma 3 and Qwen 3 the completeness of the adapter's decomposition, with
its failing inputs (the wrong layer's columns; the adapter told there is no pre-MLP norm).
Part 3, the job end to end (jobs.instruments_e1b.Job.run) on each tiny bf16 model: 6 pairs at layers 2, 2, 4, 4, 7, 7 (terciles 0, 0, 1, 1, 2, 2),
8 singles (layers 3, 5, 6, 8, two per layer), 3 known poles, 200 background positions. Checked on the bank: every item and arm banked, the readout count
equal to the arm plan, the identity arm and the resample known number exactly 0, the capture's mean equal to ā, the draws equal to an independent
capture X[index, l, k], A = a ⊙ c, the same-layer planted arms leaving a_P exactly unchanged, 54 sample lines, the known gates passed. Then
verdict_e1b.py on the two tiny banks (exit 0, 1 or 4 on random weights; 3 or 5 would be a broken reader).
Part 4, a constructed bank: part 3's banks with a_P before set to 1 + 0.05 i and every arm's a_P after set to a planted fall, ā given a heavy tail,
the metrics records rebuilt by the job's own functions, a fake harvest bank for the regime split. verdict_e1b.py: exit 0 with the planted answers
(N, G per value, D1, D1z, D1m and their readings: g4 holds, q4b refuted); exit 5 on a bank without one pair; exit 3 with a known number off; exit 1
when a single's partner set falls as much as its own (a failing input that does not fail; g4 cancelled, q4b read); exit 4 with g4's own fall 0.1
(g4 cancelled, q4b read); "no term to remove" when q4b's G(mean) is 0.02; and under each --failing-input its target gate fails on both models.
Prints one line per gate and INSTRUMENTS_DRYRUN_E1B_OK.
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
import torch

from infra.aws_pg.jobs import instruments_e1 as JE
from infra.aws_pg.jobs import instruments_e1b as JB
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import instruments_adapter as AD
from infra.pg import instruments_linear as IL
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
ITEMS = REPO / "studies/instruments/inputs/e1b_items.json"
VERDICT = REPO / "studies/instruments/scripts/verdict_e1b.py"
SPEC6_METRICS = REPO / "studies/reuse/results/generic/1/generic/t15/metrics.json"
HV = pathlib.Path("/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1")
N_LAYERS, N_WINDOWS, W_TINY = 9, 60, 128
PAIR_LAYERS = (2, 2, 4, 4, 7, 7)
SINGLE_LAYERS = (3, 5, 6, 8)
KNOWN_POLES = ("L01N5+", "L04N11+", "L07N2+")
TINY = {"g4": "gemma3", "q4b": "qwen3"}
BUDGET_MIN = 120
FAILING = {"known_wrong": "known_activation", "completeness_wrong_layer": "completeness", "own_other": "own_fall", "random_is_own": "random_matched",
           "highbg_gauss": "high_background", "identity_one": "identity", "resample_2abar": "resample_known", "planted_same_layer": "planted",
           "samples_missing": "samples"}                    # verdict_e1b.FAILING


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def jitter(model, seed: int):
    """Every RMSNorm weight moved by N(0, 0.3), so gamma != 1 (Gemma's zero-initialised weights give gain 1 exactly)."""
    g = torch.Generator().manual_seed(seed)
    for n, m in model.named_modules():
        if type(m).__name__.endswith("RMSNorm") and hasattr(m, "weight"):
            with torch.no_grad():
                m.weight.add_((0.3 * torch.randn(m.weight.shape, generator=g)).to(m.weight.dtype))
    return model


# ------------------------------------------------------------------------------------------ the arm plan, counted
def count_reads(members_J: bool, members_O: bool, inv: bool) -> dict:
    """Readouts the job asks for on one pair (before included): side J and side O."""
    nv = len(JB.VALUES)
    J = 1 + 1 + len(JB.CASE_SETS) * nv + (nv if members_J else 0) + (2 if inv else 0) + 2 + 2 + 2 + 1
    O = 1 + 1 + len(JB.CASE_SETS) * nv + (nv if members_O else 0)
    return {"J": J, "O": O}


SINGLE_READS = 1 + 1 + len(JB.SINGLE_SETS)


# ------------------------------------------------------------------------------------------ part 1: the real inputs and the size
def real_inputs() -> dict:
    it = json.loads(ITEMS.read_text())
    m6 = json.loads(SPEC6_METRICS.read_text())
    rate = m6["units_tokens_per_second"]
    out = {}
    for mk in ("g4", "q4b"):
        x = it["models"][mk]
        P, S, K = x["planted"], x["singles"], x["known"]
        assert len(P) == 56 and [sum(p["tercile"] == t for p in P) for t in (0, 1, 2)] == [18, 20, 18], mk
        for p in P:
            assert p["layer"] >= 1 and p["tercile"] == 3 * p["layer"] // x["n_layers"] and p["lam"] > 0, p["unit"]
            for s in ("J", "O"):
                assert len(p[f"{s}_tok"]) == 12 and len(set(p[f"{s}_doc"])) == 12, (p["unit"], s)
            assert not set(p["J_doc"]) & set(p["O_doc"]) and not set(p["J_token"]) & set(p["O_token"]), p["unit"]
        neurons = [(p["layer"], p["a"]) for p in P] + [(p["layer"], p["b"]) for p in P]
        assert len(set(neurons)) == 112, mk
        byp = {s["pole"]: s for s in S}
        assert len(S) == 30 and all(byp[s["partner"]]["partner"] == s["pole"] and byp[s["partner"]]["layer"] == s["layer"] for s in S), mk
        assert all(len(s["J_tok"]) == 12 and len(set(s["J_doc"])) == 12 and s["layer"] >= 1 for s in S), mk
        assert not {(s["layer"], s["neuron"]) for s in S} & set(neurons), mk
        assert len(K["poles"]) == 3 and all(len(t) == 16 for t in K["tok"]), mk
        bg = np.asarray(x["background"]["positions"])
        assert len(set(bg.tolist())) == 200 and (bg % x["window"] != 0).all() and bg.max() < x["n_tokens_windowed"], mk
        bos, hw = x["bos_id"], x["window"]
        nf = lambda t, length=RA.LENGTH: RA.n_forwarded(np.asarray(t, np.int64), length, bos, hw)
        c = count_reads(True, True, True)
        reads = len(P) * (c["J"] + c["O"]) + len(S) * SINGLE_READS
        tok = sum(c["J"] * nf(p["J_tok"]) + c["O"] * nf(p["O_tok"]) for p in P) + sum(SINGLE_READS * nf(s["J_tok"]) for s in S)
        cap = sum(nf(p["J_tok"]) + nf(p["O_tok"]) for p in P) + sum(nf(s["J_tok"]) for s in S) + 2 * nf(bg) + sum(nf(t, None) for t in K["tok"])
        fw = len(P) * (c["J"] * 12 + c["O"] * 12) + len(S) * SINGLE_READS * 12
        load = json.loads((HV / mk / "1" / "out" / mk / "manifest.json").read_text())["load_s"]
        units_min = (tok + cap) / rate / 60
        out[mk] = {"readouts_upper": reads, "forwards_upper": fw, "readout_tokens": int(tok), "capture_tokens": int(cap), "units_min": units_min,
                   "load_s": load, "wall_min": units_min + load / 60, "layers": x["n_layers"], "lam_range": x["counts"]["lambda_range"],
                   "gpu_h": (units_min + load / 60) / 60}
    out["rate"] = rate
    return out


# ------------------------------------------------------------------------------------------ part 2: the adapter's known number
def part_adapter(out: pathlib.Path) -> dict:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, 20)
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=4, hw=WINDOW, device="cpu")
    toks = np.array([w * WINDOW + 30 + 7 * w for w in range(12)])
    olmo = jitter(tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False), 1)
    units = [UR.pole_unit("L03N5+"), UR.pole_unit("L01N9-"), UR.planted_unit({"unit": "PL04", "layer": 4, "a": 3, "b": 9, "lam": 1.7})]
    eq = AD.olmo_equality(olmo, toks, units, UR.background_positions(15, 20 * WINDOW, WINDOW)["e1"][:40], **kw)
    res = {"olmo": eq}
    for arch in ("gemma3", "qwen3"):
        tk = tiny_tokenizer(bos=arch == "gemma3")
        tt = fake_corpus(tk, 20)
        k2 = {**kw, "tokens": tt, "bos": tk.bos_token_id}
        model = jitter(tiny_model(arch, seed=0, n_layers=N_LAYERS).requires_grad_(False), 1)
        ad = AD.Adapter(model, N_LAYERS)
        rows = AD.Rows(model, ad)
        r = {}
        for L, n in ((3, 5), (5, 17)):
            cap = AD.capture(ad, model, toks, L, {L}, **k2)
            ok = AD.completeness(ad, rows, cap, L, n)
            wl = AD.completeness(ad, rows, cap, L, n, shift=1)
            bad = AD.Adapter(model, N_LAYERS)
            bad.spec = {**bad.spec, "pre_mlp_norm": None}
            cb = AD.capture(bad, model, toks, L, {L}, **k2)
            nb = AD.completeness(bad, AD.Rows(model, bad), cb, L, n)
            r[f"L{L}N{n}"] = {"err": float(np.median(np.r_[ok["err_g"], ok["err_u"]])), "wrong_layer": float(np.median(np.r_[wl["err_g"], wl["err_u"]])),
                              "no_pre_norm": float(np.median(np.r_[nb["err_g"], nb["err_u"]]))}
            assert r[f"L{L}N{n}"]["err"] <= 1e-2 < min(r[f"L{L}N{n}"]["wrong_layer"], r[f"L{L}N{n}"]["no_pre_norm"]), (arch, r)
        res[arch] = {"adapter": ad.describe(), "completeness": r}
    return res


# ------------------------------------------------------------------------------------------ part 3: the tiny banks
def top_tok(ref, L, n, sign, k=50):
    a = np.asarray(ref[L])[:, :, n].astype(np.float64) * sign
    a[:, 0] = -np.inf
    return np.argsort(-a.ravel())[:k].astype(np.int64)


def tiny_items(mk: str, arch: str, ref, model, tok) -> dict:
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    pairs, used = [], set()
    for i, L in enumerate(PAIR_LAYERS):
        r = np.asarray(ref[L])[:, 1:, :].reshape(-1, W_TINY).astype(np.float64)
        for x in range(3 + 7 * i, W_TINY):
            y = (x * 37 + 11) % W_TINY
            if x == y or (L, x) in used or (L, y) in used:
                continue
            sJ, sO = r[:, x] - np.abs(r[:, y]), r[:, y] - np.abs(r[:, x])
            J, O = np.argsort(-sJ)[:12], np.argsort(-sO)[:12]
            if (sJ[J] > 0).all() and (sO[O] > 0).all():
                used |= {(L, x), (L, y)}
                pairs.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "tercile": 3 * L // N_LAYERS, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J].tolist(),
                              "O_tok": tok_rest[O].tolist()})
                break
    assert len(pairs) == len(PAIR_LAYERS), len(pairs)
    singles = []
    for L in SINGLE_LAYERS:
        two = [n for n in range(W_TINY) if (L, n) not in used][2 * L:2 * L + 2]
        got = []
        for j, n in enumerate(two):
            s = "+" if j == 0 else "-"
            t = top_tok(ref, L, n, 1.0 if s == "+" else -1.0, 12)
            got.append({"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": 3 * L // N_LAYERS, "J_tok": t.tolist()})
        got[0]["partner"], got[1]["partner"] = got[1]["pole"], got[0]["pole"]
        singles += got
    ktok = [top_tok(ref, NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0, 16).tolist() for p in KNOWN_POLES]
    val = lambda p, t: np.asarray(ref[NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].astype(float).tolist()
    ka = [val(p, t) for p, t in zip(KNOWN_POLES, ktok)]
    ad = AD.Adapter(model, N_LAYERS)
    return {"model_key": mk, "repo": f"tiny/{arch}", "revision": "tiny", "arch": arch, "nh_key": "tiny", "tokname": "tiny", "n_layers": N_LAYERS, "width": W_TINY,
            "hidden": ad.hidden, "bos_id": tok.bos_token_id, "window": WINDOW, "n_tokens_windowed": N_WINDOWS * WINDOW, "tokens_sha256": "tiny",
            "act_fn_measured": ad.act, "planted": pairs, "singles": singles,
            "known": {"poles": list(KNOWN_POLES), "tok": ktok, "a_bank": ka, "a_wrong": [ka[(i + 1) % 3] for i in range(3)], "n_ctx": 16,
                      "wrong_rule": "the next known pole's bank a (cyclic)"},
            "background": {"positions": UR.background_positions(-1000, N_WINDOWS * WINDOW, WINDOW)["e1"].tolist(), "seed": 0}}


def check_tiny(d: pathlib.Path, mk: str, items: dict, model, tok, tokens) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    g = m["gates"]
    assert m["status"] == "done" and g["capture"]["pass"] and g["capture"]["failing_input_failed"], g["capture"]
    assert g["identity"]["value"] == 0.0 and g["identity"]["failing_input_failed"], g["identity"]
    assert g["resample_known"]["value"] == 0.0 and g["resample_known"]["failing_input_failed"], g["resample_known"]
    assert g["coefficient"]["pass"] and g["coefficient"]["failing_input_failed"] and g["exclusion"]["pass"], (g["coefficient"], g["exclusion"])
    assert g["known_activation"]["pass"] and g["completeness"]["pass"] and g["samples"]["pass"], (g["known_activation"], g["completeness"], g["samples"])
    assert g["planted"]["failing_input"]["max_abs_delta"] == 0.0 or not g["planted"]["n_sides"], g["planted"]
    lines = (d / "samples" / f"e1b_{mk}.jsonl").read_text().splitlines()
    assert len(lines) == JB.N_SAMPLE_LINES and all(json.loads(x)["contexts"][0]["marked_token"] for x in lines)
    vz = np.load(d / "values.npz")
    ad = AD.Adapter(model, N_LAYERS)
    X = AD.capture_background(ad, model, np.asarray(items["background"]["positions"]), N_LAYERS - 1, tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id,
                              batch=8, hw=WINDOW, device="cpu").numpy()
    idx = vz["draw_idx"].astype(np.int64)
    L_, W_ = X.shape[1:]
    want = np.stack([X[idx[r], np.arange(L_)[:, None], np.arange(W_)[None]] for r in range(len(IV.SEEDS))])
    assert np.array_equal(vz["draws"], want) and np.allclose(vz["median"], np.median(X, 0), atol=1e-6) and np.allclose(vz["abar"], X.mean(0), rtol=1e-5, atol=1e-5)
    asked, n_mem = 0, 0
    for p in items["planted"]:
        z = np.load(d / "e1b" / mk / "pairs" / f"{p['unit']}.npz")
        names = {s: [str(x) for x in z[f"{s}_arms"]] for s in ("J", "O")}
        has = lambda x: len(z[f"set_{x}_layer"]) > 0
        c = count_reads(has("J_members"), has("O_members"), has("J_own_inv"))
        assert len(names["J"]) == c["J"] - 1 and len(names["O"]) == c["O"] - 1, (p["unit"], len(names["J"]), c)
        for n in ("planted_same_layer_dose1", "planted_same_layer_dose10"):
            assert (z["J_a_after"][names["J"].index(n)] == z["J_a_before"]).all(), (p["unit"], n, "a same-layer write moved a_P")
        assert (z["J_a_after"][names["J"].index("resconst_abar")] == z["J_a_after"][names["J"].index("own_u__mean")]).all()
        assert z["J_terms"].shape == (12, 2) and np.allclose(z["J_terms"].sum(1), z["J_a_before"], rtol=1e-4, atol=1e-5)
        asked += c["J"] + c["O"]
        n_mem += has("J_members")
    for s in items["singles"]:
        z = np.load(d / "e1b" / mk / "singles" / f"{s['pole']}.npz")
        assert [str(x) for x in z["J_arms"]] == ["identity", "own_u__mean", "random_u__mean", "other_own_u__mean"], s["pole"]
        asked += SINGLE_READS
    assert asked == m["readouts"] + m["readouts_cached"], (asked, m["readouts"], m["readouts_cached"])
    return {"m": m, "asked": asked, "n_mem": n_mem}


def part_tiny(out: pathlib.Path) -> tuple:
    built, models = {}, {}
    for mk, arch in TINY.items():
        tok = tiny_tokenizer(bos=arch == "gemma3")
        tokens = fake_corpus(tok, N_WINDOWS + 2)
        offsets = np.arange(0, tokens.size + 37, 37)
        offsets[-1] = tokens.size
        model = jitter(tiny_model(arch, seed=0, n_layers=N_LAYERS).requires_grad_(False), 1)
        ref = reference_a(model, tok, tokens, list(range(N_WINDOWS)))
        models[mk] = tiny_items(mk, arch, ref, model, tok)
        built[mk] = (model, tok, tokens, {"tokens": tokens, "doc_offsets": offsets})
    ip = out / "e1b_items_tiny.json"
    NH.dump(ip, {"models": models})
    run = out / "tiny" / "run"
    res = {}
    for mk, (model, tok, tokens, corpus) in built.items():
        d = run / mk
        a = JB.parser().parse_args(["--out", str(d), "--model-key", mk, "--items", str(ip), "--corpus", "tiny", "--harvest-window", str(WINDOW), "--batch", "8",
                                    "--full-batch", "4"])
        a.items_sha256 = sha(ip)
        JB.Job(d, a, models[mk], model, tok, corpus, "cpu").run()
        res[mk] = check_tiny(d, mk, models[mk], model, tok, tokens)
    return ip, run, {"models": models}, res


# ------------------------------------------------------------------------------------------ part 4: a constructed bank with planted falls
DR = (-0.03, 0.0, 0.01, 0.06)
PLAN = {"g4": {"mean": (0.20, 0.06), "zero": (0.15, 0.07), "median": (0.12, 0.08), "res": (0.08, 0.07)},
        "q4b": {"mean": (0.15, 0.10), "zero": (0.16, 0.11), "median": (0.14, 0.10), "res": (0.145, 0.10)}}
MEMBERS = {"mean": 0.10, "zero": 0.08, "median": 0.06, "res0": 0.03, "res1": 0.04, "res2": 0.05, "res3": 0.04}
EXTRA = {"identity": 0.0, "identity_fail": 0.01, "resconst_2abar": 0.3, "planted_dose1": 0.02, "planted": 0.2, "planted_same_layer_dose1": 0.0,
         "planted_same_layer_dose10": 0.0, "own_inv__mean": 0.5, "cross_inv__mean": 0.45}
SINGLE = {"identity": 0.0, "own_u__mean": 0.5, "random_u__mean": 0.05, "other_own_u__mean": 0.08}


def planted_fall(plan: dict, i: int, side: str, arm: str) -> float:
    if arm == "resconst_abar":
        arm = "own_u__mean"
    if arm in EXTRA:
        return EXTRA[arm]
    x, v = arm.split("__")
    if x == "members":
        return MEMBERS[v]
    a, e = x.rsplit("_", 1)
    if a == "own":
        return 0.5
    if a == "random":
        return 0.05
    off = 0.004 * i + (0.01 if side == "O" else 0.0)
    k = 0 if e == "u" else 1
    I = plan["res"][k] + DR[int(v[3])] + off if v.startswith("res") else plan[v][k] + off
    return 0.5 * I


def fake_harvest(root: pathlib.Path):
    """stats/layer_<l>.npz for the tiny models: even neurons dense (|a| spread over 1e-1..1e1), odd neurons sparse (99% near 1e-4, a thin tail)."""
    edges = NH.hist_edges().astype(np.float32)
    for mk in TINY:
        d = root / mk / "1" / "out" / mk / "stats"
        d.mkdir(parents=True, exist_ok=True)
        for l in range(N_LAYERS):
            h = np.zeros((W_TINY, 2, 400), np.int32)
            h[0::2, 0, 200:280] = 100
            h[1::2, 0, 80:100] = 4950
            h[1::2, 0, 200:280] = 12
            np.savez_compressed(d / f"layer_{l:02d}.npz", hist=h, hist_edges=edges, density=h.sum(-1).astype(np.int64), q999=np.ones((W_TINY, 2), np.float32),
                                n_pole_tokens=h.sum(-1).astype(np.int64))


def synth(dst: pathlib.Path, src: pathlib.Path, items: dict, plans=PLAN, single=None, metrics_edit=None) -> pathlib.Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for mk in TINY:
        d = dst / mk
        m = json.loads((d / "metrics.json").read_text())
        single = single or {}
        for i, p in enumerate(items["models"][mk]["planted"]):
            f = d / "e1b" / mk / "pairs" / f"{p['unit']}.npz"
            z = dict(np.load(f))
            for s in ("J", "O"):
                b = (1.0 + 0.05 * np.arange(len(z[f"{s}_tok"]))).astype(np.float32)
                z[f"{s}_a_before"] = b
                z[f"{s}_a_after"] = np.stack([(b * (1 - planted_fall(plans[mk], i, s, str(arm)))).astype(np.float32) for arm in z[f"{s}_arms"]])
            np.savez_compressed(f, **z)
            m["pairs"][p["unit"]]["sides"] = JE.record_of(z, ("J", "O"))
        for s in items["models"][mk]["singles"]:
            f = d / "e1b" / mk / "singles" / f"{s['pole']}.npz"
            z = dict(np.load(f))
            b = (1.0 + 0.05 * np.arange(len(z["J_tok"]))).astype(np.float32)
            z["J_a_before"] = b
            sf = single.get(mk, SINGLE)
            z["J_a_after"] = np.stack([(b * (1 - sf[str(arm)])).astype(np.float32) for arm in z["J_arms"]])
            np.savez_compressed(f, **z)
            m["singles"][s["pole"]]["falls"] = {str(n): {k: RA.job_fall(b, z["J_a_after"][i])[k] for k in ("mean", "median", "n_pos")} for i, n in enumerate(z["J_arms"])}
        vz = dict(np.load(d / "values.npz"))
        rng = np.random.default_rng(1)
        ab = rng.normal(0, 0.01, vz["abar"].shape).astype(np.float32)
        ab[:, :2] = 1.0
        vz["abar"] = ab
        np.savez_compressed(d / "values.npz", **vz)
        m["gates"]["high_background"] = JB.hb_gate(ab[:m["background"]["n_up"]])
        if metrics_edit:
            metrics_edit(mk, m)
        (d / "metrics.json").write_text(json.dumps(m))
    return dst


def expected(items: dict, mk: str, plan: dict) -> dict:
    ps = items["models"][mk]["planted"]
    off = np.mean([min(0.004 * i for i, p in enumerate(ps) if p["tercile"] == t) for t in (0, 1, 2)])
    G = {v: plan[v][0] - plan[v][1] for v in ("mean", "zero", "median")}
    G["res_mean"] = plan["res"][0] - plan["res"][1]
    return {"N_u": plan["mean"][0] + off, "N_e": plan["mean"][1] + off, "N_res_u": plan["res"][0] + np.mean(DR) + off, "G": G,
            "R5": G["mean"] - G["res_mean"], "R5z": G["mean"] - G["zero"], "R5m": G["mean"] - G["median"], "line": 0.5 * G["mean"]}


def verdict(run: pathlib.Path, ip: pathlib.Path, vout: pathlib.Path, hroot: pathlib.Path, *extra) -> tuple:
    cmd = [sys.executable, str(VERDICT), "--run", str(run), "--items", str(ip), "--out", str(vout), "--harvest-root", str(hroot), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 5 or f"VERDICT_E1B exit {r.returncode}" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-3000:], "the verdict crashed")
    return r.returncode, r.stdout, r.stderr


def part_synth(out: pathlib.Path, ip: pathlib.Path, run: pathlib.Path, items: dict) -> dict:
    hroot = out / "fake_harvest"
    sd = synth(out / "synth" / "run", run, items)
    vout = out / "synth" / "verdict"
    code, so, se = verdict(sd, ip, vout, hroot)
    gl = [x for x in so.splitlines() if x.startswith(("GATE", "RULE", "REPORTED", "VERDICT"))]
    assert code == 0, (code, so[-3000:], se[-3000:])
    v = json.loads((vout / "verdict_e1b.json").read_text())
    close = lambda a, b: abs(a - b) < 1e-6
    for mk in TINY:
        ex, r = expected(items, mk, PLAN[mk]), v["models"][mk]
        A1, G, R = r["reported"]["A1"], r["reported"]["G"], r["rules"]
        assert close(A1["mean__u"]["value"], ex["N_u"]) and close(A1["mean__e"]["value"], ex["N_e"]) and close(A1["res_mean__u"]["value"], ex["N_res_u"]), (mk, A1["mean__u"], ex)
        for k, x in ex["G"].items():
            assert close(G[k]["value"], x), (mk, k, G[k]["value"], x)
        for k in ("R5", "R5z", "R5m"):
            assert close(R[k]["value"], ex[k]) and close(R[k]["line"], ex["line"]), (mk, k, R[k]["value"], ex[k])
        assert r["gates"]["planted"]["n_sides"] > 0 and r["gates"]["planted"]["value"] == 1.0, (mk, r["gates"]["planted"]["n_sides"])
        assert close(r["gates"]["own_fall"]["value"], 0.5) and close(r["gates"]["random_matched"]["value"], 0.05), r["gates"]["own_fall"]
    want = {"g4": ["H_value holds, H_set refuted", "H_value refuted", "H_value holds, H_set refuted"], "q4b": ["H_value refuted"] * 3}
    got = {mk: [v["models"][mk]["rules"][k]["reading"] for k in ("R5", "R5z", "R5m")] for mk in TINY}
    assert got == want, got
    exits = {"green": code}
    inc = synth(out / "synth_incomplete" / "run", run, items)
    (inc / "q4b" / "e1b" / "q4b" / "pairs" / f"{items['models']['q4b']['planted'][-1]['unit']}.npz").unlink()
    c5, _, e5 = verdict(inc, ip, out / "synth_incomplete" / "verdict", hroot)
    assert c5 == 5 and "E1B_BANK_INCOMPLETE" in e5 and not (out / "synth_incomplete" / "verdict").exists(), (c5, e5[-2000:])
    exits["incomplete"] = c5

    def known_off(mk, m):
        if mk == "g4":
            m["gates"]["known_activation"]["ratio"] = 0.5
    ko = synth(out / "synth_known_off" / "run", run, items, metrics_edit=known_off)
    c3, so3, _ = verdict(ko, ip, out / "synth_known_off" / "verdict", hroot)
    assert c3 == 3, (c3, so3[-2000:])
    exits["known_off"] = c3
    nf = synth(out / "synth_nofail" / "run", run, items, single={"g4": {**SINGLE, "other_own_u__mean": 0.5}})
    c1, so1, _ = verdict(nf, ip, out / "synth_nofail" / "verdict", hroot)
    v1 = json.loads((out / "synth_nofail" / "verdict" / "verdict_e1b.json").read_text())
    assert c1 == 1 and "a failing input did not fail" in so1 and v1["models"]["g4"]["cancelled"] and not v1["models"]["q4b"]["cancelled"], (c1, so1[-2000:])
    assert v1["models"]["q4b"]["rules"]["R5"]["reading"] == "H_value refuted", v1["models"]["q4b"]["rules"]["R5"]
    exits["failing_input_does_not_fail"] = c1
    cm = synth(out / "synth_g4_cancelled" / "run", run, items, single={"g4": {**SINGLE, "own_u__mean": 0.1}})
    c4, so4, _ = verdict(cm, ip, out / "synth_g4_cancelled" / "verdict", hroot)
    v4 = json.loads((out / "synth_g4_cancelled" / "verdict" / "verdict_e1b.json").read_text())
    assert c4 == 4 and v4["models"]["g4"]["rules"]["R5"]["reading"].startswith("cancelled") and v4["models"]["q4b"]["rules"]["R5"]["reading"] == "H_value refuted", (c4, so4[-2000:])
    assert not v4["models"]["g4"]["gates"]["own_fall"]["pass"], v4["models"]["g4"]["gates"]["own_fall"]
    exits["g4_own_fall_0.1"] = c4
    nt = synth(out / "synth_no_term" / "run", run, items, plans={**PLAN, "q4b": {**PLAN["q4b"], "mean": (0.12, 0.10)}})
    c0, so0, _ = verdict(nt, ip, out / "synth_no_term" / "verdict", hroot)
    v0 = json.loads((out / "synth_no_term" / "verdict" / "verdict_e1b.json").read_text())
    assert c0 == 0 and v0["models"]["q4b"]["rules"]["R5"]["reading"].startswith("no term to remove"), (c0, v0["models"]["q4b"]["rules"]["R5"])
    exits["q4b_no_term"] = c0
    fails = {}
    for fi, target in sorted(FAILING.items()):
        cf, sof, sef = verdict(sd, ip, vout, hroot, "--failing-input", fi)
        f = json.loads((vout / f"failing_input_{fi}" / "verdict_e1b.json").read_text())
        want_code = 3 if target in ("known_activation", "completeness") else 4
        for mk in TINY:
            r = f["models"][mk]
            assert not r["gates"][target]["pass"] and all(x["reading"].startswith("cancelled") for x in r["rules"].values()), (fi, mk)
            assert all(g["pass"] for k, g in r["gates"].items() if k != target), (fi, mk, {k: g["pass"] for k, g in r["gates"].items()})
        assert cf == want_code, (fi, cf, sof[-2000:], sef[-2000:])
        fails[fi] = [x for x in sof.splitlines() if x.startswith("FAILING_INPUT")][0]
        exits[fi] = cf
    return {"lines": gl, "exits": exits, "fails": fails, "v": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print("SELFCHECK instruments_values:", json.dumps(IV.selfcheck(), ensure_ascii=False))
    print("SELFCHECK instruments_linear:", json.dumps(IL.selfcheck(), ensure_ascii=False))
    print("SELFCHECK instruments_adapter:", json.dumps(AD.selfcheck(), ensure_ascii=False))
    e = real_inputs()
    for mk in ("g4", "q4b"):
        x = e[mk]
        print(f"REAL INPUTS {mk}: 56 pairs (18/20/18) and 30 singles checked; at most {x['readouts_upper']} readouts = {x['forwards_upper']} forwards of <= 256 tokens, "
              f"{x['readout_tokens'] / 1e6:.1f} M tokens + captures {x['capture_tokens'] / 1e6:.2f} M; at SPEC6's L40S rate {e['rate']:.0f} tokens/s (Olmo-3-7B): units "
              f"{x['units_min']:.0f} min + load {x['load_s']:.0f} s = {x['wall_min']:.0f} min, {x['gpu_h']:.2f} GPU-h (§3.7: 0.5 GPU-h per model)")
    ad = part_adapter(out)
    print(f"ADAPTER olmo3: {', '.join(ad['olmo']['compared'])} equal to the REUSED path exactly on {ad['olmo']['units']}")
    for arch in ("gemma3", "qwen3"):
        c = ad[arch]["completeness"]
        print(f"ADAPTER {arch} ({ad[arch]['adapter']['act']}; pre {ad[arch]['adapter']['pre_mlp_norm']}, post {ad[arch]['adapter']['post_mlp_norm']}): completeness "
              + "; ".join(f"{k} {v['err']:.1e} (wrong layer {v['wrong_layer']:.2f}, no pre-norm {v['no_pre_norm']:.2f})" for k, v in c.items()))
    ip, run, items, t = part_tiny(out)
    for mk, r in t.items():
        g = r["m"]["gates"]
        print(f"TINY JOB {mk} ({TINY[mk]}): 6 pairs, 8 singles; readouts {r['m']['readouts']} + {r['m']['readouts_cached']} cached = the arm plan {r['asked']}; identity "
              f"{g['identity']['value']}, resample known {g['resample_known']['value']}, capture {g['capture']['max_abs']:.1e}, A = a ⊙ c to {g['coefficient']['value']:.1e}, "
              f"known ratio {g['known_activation']['ratio']:.1e} (wrong {g['known_activation']['wrong_ratio']:.2f}), completeness {g['completeness']['median_err_over_pole_max']:.1e}; "
              f"draws = an independent capture's X[index]; same-layer arms leave a_P unchanged; {JB.N_SAMPLE_LINES} sample lines; {r['n_mem']} pairs with a member")
    fake_harvest(out / "fake_harvest")
    code, so, se = verdict(run, ip, out / "tiny" / "verdict", out / "fake_harvest")
    assert code in (0, 1, 4) and "E1B_BANK_INCOMPLETE" not in se, (code, so[-3000:], se[-3000:])
    for x in so.splitlines():
        if x.startswith(("GATE", "VERDICT")):
            print("  tiny", x)
    s = part_synth(out, ip, run, items)
    print("CONSTRUCTED BANK (planted falls), verdict lines:")
    for x in s["lines"]:
        print("  ", x)
    print("planted answers reproduced on both models: N(mean, u), N(mean, e), N(resample, u), G for four values, D1, D1z, D1m and the line 0.5 G(mean) to 1e-6; "
          "readings g4 [holds, refuted, holds], q4b [refuted x 3]")
    for x in s["fails"].values():
        print("  ", x)
    print("EXITS:", json.dumps(s["exits"]))
    (out / "metrics.json").write_text(json.dumps({"real_inputs": e, "adapter": ad, "tiny_exit": code, "synth_exits": s["exits"], "failing_inputs": s["fails"]},
                                                 indent=1, default=str))
    print("INSTRUMENTS_DRYRUN_E1B_OK")


if __name__ == "__main__":
    main()
