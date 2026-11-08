"""Weight-free dry run of INSTRUMENTS E4 (studies/instruments/SPEC_instruments.md §6, amendment 40; EXPLORATORY). No download, no network: a tiny
random OLMo 3 (6 layers, MLP width 128, from its transformers config), a byte-level tokenizer and a fake corpus, as the E1 dry run builds them.

    <harvest venv python> -m infra.aws_pg.jobs.instruments_dryrun_e4 --out <dir>

Part 1, the real inputs (read only): studies/instruments/inputs/e4_items.json through the job's check_items; its B_t equal to E1's banked T_15
high_background.json; every known case's D'_bank of 50 members outside B_t; the size of the run: captures and readouts by the job's arm plan
(other__zero on every case but the first; the planted arms counted on every case at layer >= 8, an upper bound: B_t's first members sit at layer 7),
tokens by reuse_ablate.n_forwarded, timed at E1's measured L40S rate (results/e1/1/t15/metrics.json units_tokens_per_second) plus E1's fixed costs.
Part 2, the job end to end (jobs.instruments_e4.Job.run) on the tiny bf16 model: 8 cases at layers 2 to 5, each side 6 to 12 contexts from
distinct documents of the pole's top 50, B_t the tiny model's own (as the job computes it), 3 known cases whose D'_bank comes from a capture of the
pole's 50 contexts at another batch size, the parent's activation block. Checked on the bank: status done; the B_t, known-number, identity,
exclusion and samples gates; the identity arm's a_P equal to a_P before bit for bit; the planted gate's same-layer writes leaving a_P unchanged;
every arm present (arm_names, equal to verdict_e4's); readouts counted; I and s recomputed from the npz; D'(A) and the proxy set P(A) rebuilt from
a second capture at another batch with the proxy written out in numpy (overlap >= 0.9); a_P before against reference_a (median relative
difference under 0.02). Then the job fed its failing inputs: B_t with one member swapped, the known bank sets shifted by one case, a wrong token id;
each stops the job with the gate's message.
Part 3, verdict_e4.py: on the tiny bank (exit 0, 1 or 4: the tiny model's falls decide room); on fabricated banks of 120 cases on 100 neurons with
planted answers: "fit" (I_zero = 0.08 + 1.45 s_true + noise, s_proxy = s_true plus up to 0.16) -> R1 holds, R2 generalizes, exit 0; "none" (I
independent of s) -> R1 not separated; "negative" (I = 0.6 - 1.45 s) -> R1 refuted; every --failing-input on "fit" -> its gate fails (exit 3 for
the known numbers, 4 otherwise); one npz removed -> exit 5; other__zero equal to own (a failing input that does not fail) -> exit 1. figures_e4.py
draws the "fit" verdict (stamped DRY RUN). Prints one line per check and INSTRUMENTS_DRYRUN_E4_OK.
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

from infra.aws_pg.jobs import instruments_e4 as JE
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
ITEMS = REPO / "studies/instruments/inputs/e4_items.json"
E1_BT = REPO / "studies/instruments/results/e1/1/t15/high_background.json"
E1_METRICS = REPO / "studies/instruments/results/e1/1/t15/metrics.json"
VERDICT = REPO / "studies/instruments/scripts/verdict_e4.py"
FIGURES = REPO / "studies/instruments/scripts/figures_e4.py"
N_LAYERS, N_WINDOWS, BATCH = 6, 60, 8
CASES = [(2, 9), (3, 5), (3, 40), (4, 11), (4, 70), (5, 2), (5, 33), (2, 61)]
GPU_PRICE = 2.2
sys.path.insert(0, str(REPO / "studies/instruments/scripts"))
import verdict_e4 as V  # noqa: E402


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run_verdict(run, ip, out, *extra) -> tuple[int, str, dict | None]:
    r = subprocess.run([sys.executable, str(VERDICT), "--run", str(run), "--items", str(ip), "--out", str(out), *extra], capture_output=True, text=True)
    fi = [x for x in extra if x != "--failing-input"]
    p = pathlib.Path(out) / (f"failing_input_{fi[0]}" if fi else "") / "verdict_e4.json"
    return r.returncode, r.stdout + r.stderr, json.loads(p.read_text()) if p.exists() and r.returncode != 5 else None


# ------------------------------------------------------------------------------------------ part 1: the real inputs
def real_inputs() -> dict:
    it = json.loads(ITEMS.read_text())
    JE.check_items(it, "T_15")
    hb = json.loads(E1_BT.read_text())
    assert [[int(l), int(k)] for l, k, _ in it["B_t"]["members"]] == [[int(l), int(k)] for l, k, _ in hb["members"]], "the items' B_t is not E1's"
    bt = {(int(l), int(k)) for l, k, _ in hb["members"]}
    for k in it["known_cases"]:
        for x in ("D_bank", "D_bank_reversed"):
            assert len({tuple(q) for q in k[x]}) == 50 and not any(tuple(q) in bt for q in k[x]), (k["pole"], x)
    first7 = min(l for l, _ in bt)
    bos, hw = 1, UR.HARVEST_WINDOW
    nf = lambda t: RA.n_forwarded(np.asarray(t, np.int64), RA.LENGTH, bos, hw)
    cap_tok = read_tok = readouts = 0
    for i, c in enumerate(it["cases"]):
        n_arms = len(JE.arm_names(i > 0, c["layer"] > first7))
        readouts += 1 + n_arms
        read_tok += (1 + n_arms) * nf(c["A"]["tok"])
        cap_tok += nf(c["A"]["tok"]) + nf(c["B"]["tok"])
    m1 = json.loads(E1_METRICS.read_text())
    rate = m1["units_tokens_per_second"]
    fixed_tok = 200 * RA.LENGTH + sum(RA.n_forwarded(np.asarray(x)[:16], None, bos, hw) for x in it["known"]["tok"])
    units_s = (cap_tok + read_tok) / rate
    fixed_s = m1["download_and_corpus_s"] + m1["load_s"] + m1["timing_s"]["known"] + m1["timing_s"]["background"] + fixed_tok / rate
    return {"cases": len(it["cases"]), "neurons": it["counts"]["neurons"], "by_band": it["counts"]["by_band"], "known": len(it["known_cases"]),
            "captures": it["counts"]["captures"], "readouts": readouts, "forwards_A": sum((1 + len(JE.arm_names(i > 0, c["layer"] > first7))) * len(c["A"]["tok"])
                                                                                          for i, c in enumerate(it["cases"])),
            "capture_tokens": int(cap_tok), "readout_tokens": int(read_tok), "rate_tok_s": rate, "units_min": units_s / 60, "fixed_min": fixed_s / 60,
            "wall_min": (units_s + fixed_s) / 60, "gpu_h": (units_s + fixed_s) / 3600, "dollars": (units_s + fixed_s) / 3600 * GPU_PRICE,
            "known_reversed_median": float(np.median([k["overlap_bank_vs_reversed"] for k in it["known_cases"]]))}


# ------------------------------------------------------------------------------------------ part 2: the tiny bank
def top_tok(ref, L, n, offsets, k=50):
    """The k contexts with the largest a, at most one per document (window position 0 is not a context)."""
    a = np.asarray(ref[L])[:, :, n].astype(np.float64)
    a[:, 0] = -np.inf
    out, seen = [], set()
    for t in np.argsort(-a.ravel(), kind="stable"):
        d = int(np.searchsorted(offsets, t, side="right") - 1)
        if d not in seen:
            seen.add(d)
            out.append(int(t))
        if len(out) == k:
            return np.asarray(out, np.int64)
    raise AssertionError(f"fewer than {k} documents")


def side(ranks, docs, cap=12) -> list:
    seen, out = set(), []
    for r in ranks:
        if docs[r] not in seen:
            seen.add(docs[r])
            out.append(int(r))
    return out[:cap]


def tiny_items(ref, ref2, model, tok, tokens, offsets) -> dict:
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=BATCH, hw=WINDOW, device="cpu")
    pos = UR.background_positions(15, N_WINDOWS * WINDOW // WINDOW * WINDOW, WINDOW)["e1"]
    hb = RB.high_background(RA.background_mean_a(model, pos, N_LAYERS - 1, **kw), N_LAYERS - 1)
    mask = hb["mask"]
    rows = UR.Rows(model)
    cases, known = [], []
    for i, (L, n) in enumerate(CASES):
        tk = top_tok(ref, L, n, offsets)
        docs = [int(np.searchsorted(offsets, t, side="right") - 1) for t in tk]
        A, B = side(range(0, 50, 2), docs), side(range(1, 50, 2), docs, 10)
        assert len(A) >= 6 and len(B) >= 6, (L, n, len(A), len(B))
        rec = lambda rs, w: {"w": w, "size": len(rs), "n_docs": len(rs), "ranks": rs, "doc": [docs[r] for r in rs], "tok": [int(tk[r]) for r in rs],
                             "token_id": [int(tokens[tk[r]]) for r in rs], "token_string": [tok.decode([int(tokens[tk[r]])]) for r in rs]}
        cases.append({"pole": f"L{L:02d}N{n}+", "layer": L, "neuron": n, "sign": "+", "band": "low" if L <= 3 else "deep", "sources": ["dry"], "cluster_bank": "dry",
                      "k_W": 3, "n_qualifying": 2, "A": rec(A, 0), "B": rec(B, 1)})
        if i < 3:
            unit = UR.pole_unit(cases[-1]["pole"])
            cap = UR.capture(model, tk, L, {L}, **{**kw, "batch": 7})
            Afull = RA.direct_A(rows, cap, unit, 128)
            top = lambda rs: [[int(l), int(k)] for l, k in RB.top_m_excluded(Afull[rs].sum(0), 50, 128, mask)]
            D, Dr = top(A), top([49 - r for r in A])
            known.append({"pole": cases[-1]["pole"], "D_bank": D, "D_bank_reversed": Dr, "bank": "dry", "bank_sha256": "dry",
                          "overlap_bank_vs_reversed": len({tuple(x) for x in D} & {tuple(x) for x in Dr}) / 50})
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(ref, NH.parse_pole(p)[0], NH.parse_pole(p)[1], offsets).tolist() for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].tolist()
    s4 = [[l, k] for l, k, _ in hb["members"][:-1]] + [[0, int(np.flatnonzero(~mask[0].numpy())[0])]]
    return {"t": "T_15", "grid": "t15_step1413814", "revision": "dry", "m": 50, "cap": 12, "min_side": 6, "corpus": {"tokens_sha256": "dry"}, "sources": {"grid": []},
            "counts": {"cases": len(cases), "known_cases": len(known), "neurons": len(cases)},
            "B_t": {"path": "dry", "sha256": "dry", "cut": hb["cut"], "count": hb["count"], "n_up": N_LAYERS - 1, "members": hb["members"]},
            "B_t_spec4": {"path": "dry", "sha256": "dry", "members": s4},
            "known": {"poles": kp, "tok": ktok, "wrong_rule": "another model", "a_bank": {"T_15": [val(ref, p, t) for p, t in zip(kp, ktok)]},
                      "a_wrong": {"T_15": [val(ref2, p, t) for p, t in zip(kp, ktok)]}},
            "known_cases": known, "sample_cases": [cases[1]["pole"], cases[5]["pole"]], "cases": cases}


def job_args(d, ip):
    a = JE.parser().parse_args(["--out", str(d), "--t", "T_15", "--items", str(ip), "--harvest-window", str(WINDOW), "--scan-tokens", str(N_WINDOWS * WINDOW),
                                "--batch", str(BATCH), "--full-batch", "4"])
    a.items_sha256 = sha(ip)
    return a


def proxy_numpy(cap, L, n, rows, mask) -> list:
    """s4_sameweights' proxy written out in numpy (its coef(), its product), top 50 outside B_t."""
    g = cap["g"][L][:, n].double().numpy()
    u = cap["u"][L][:, n].double().numpy()
    s = 1 / (1 + np.exp(-g))
    cg, cu = ((s + g * s * (1 - s)) * u).mean(), (g * s).mean()
    G, U = (x.double().numpy() for x in rows.inputs(L, n))
    sc = []
    for l in range(L):
        d = rows.column(l).double().numpy()
        sc.append(cap["a"][l].abs().double().numpy().mean(0) * cap["s"][l].double().numpy().mean() * np.abs(cg * (d.T @ G) + cu * (d.T @ U)))
    sc = np.concatenate(sc)
    sc[mask[:L].reshape(-1).numpy()] = -np.inf
    i = np.argsort(-sc, kind="stable")[:50]
    return [(int(j // 128), int(j % 128)) for j in i]


def check_tiny(d: pathlib.Path, items: dict, model, tok, tokens, ref) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    g = m["gates"]
    assert m["status"] == "done", m["status"]
    assert g["B_t"]["pass"] and g["B_t"]["failing_input_failed"] and g["known_number"]["pass"], (g["B_t"], g["known_number"])
    assert g["identity"]["value"] == 0.0 and g["identity"]["failing_input_failed"] and g["exclusion"]["pass"] and g["samples"]["pass"], (g["identity"], g["exclusion"], g["samples"])
    assert g["planted"]["n_cases"] == 0 or (g["planted"]["failing_input"]["max_abs_delta"] == 0.0 and g["planted"]["failing_input_failed"]), g["planted"]
    for ho in (False, True):
        for hm in (False, True):
            assert JE.arm_names(ho, hm) == V.arm_names(ho, hm)
    rows, mask = UR.Rows(model), torch.as_tensor(RB.mask_of_members(items["B_t"]["members"], N_LAYERS - 1, 128))
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=5, hw=WINDOW, device="cpu")
    worst = {"own_overlap_min": 1.0, "proxy_overlap_min": 1.0, "aP_vs_reference_median": 0.0}
    n_read = 0
    for c in items["cases"]:
        z = np.load(d / "e4" / f"{c['pole']}.npz")
        names = [str(x) for x in z["arms"]]
        assert names == JE.arm_names("set_other_layer" in z.files, "set_planted_layer" in z.files), names
        n_read += 1 + len(names)
        b, A = z["a_before"], z["a_after"]
        assert np.array_equal(A[names.index("identity")], b), (c["pole"], "the identity arm moved a_P")
        if "planted_same_layer_dose1" in names:
            for x in ("planted_same_layer_dose1", "planted_same_layer_dose10"):
                assert np.array_equal(A[names.index(x)], b), (c["pole"], x)
        rec = m["cases"][c["pole"]]
        sets = {k[4:-6]: [(int(l), int(n)) for l, n in zip(z[k], z[k.replace("_layer", "_neuron")])] for k in z.files if k.startswith("set_") and k.endswith("_layer")}
        assert float(z["s_true"]) == JE.overlap(sets["own"], sets["cross"]) == rec["s_true"]
        assert float(z["s_proxy"]) == JE.overlap(sets["proxy_A"], sets["proxy_B"]) == rec["s_proxy"]
        jf = lambda x: RA.job_fall(b, A[names.index(x)])["mean"]
        for v in JE.VALUES:
            want = jf(f"cross__{v}") / jf(f"own__{v}")
            assert (np.isnan(want) and np.isnan(rec["I"][v])) or abs(want - rec["I"][v]) < 1e-12, (c["pole"], v)
        L, n = c["layer"], c["neuron"]
        unit = UR.pole_unit(c["pole"])
        cap = UR.capture(model, np.asarray(c["A"]["tok"]), L, {L}, **kw)
        D2 = RB.top_m_excluded(RA.direct_A(rows, cap, unit, 128).sum(0), 50, 128, mask)
        worst["own_overlap_min"] = min(worst["own_overlap_min"], JE.overlap(D2, sets["own"]))
        worst["proxy_overlap_min"] = min(worst["proxy_overlap_min"], JE.overlap(proxy_numpy(cap, L, n, rows, mask), sets["proxy_A"]))
        t = np.asarray(c["A"]["tok"])
        refa = np.asarray(ref[L])[t // WINDOW, t % WINDOW, n].astype(np.float64)
        worst["aP_vs_reference_median"] = max(worst["aP_vs_reference_median"], float(np.median(np.abs(b - refa) / np.abs(refa))))
    assert n_read == m["readouts"], (n_read, m["readouts"])
    assert worst["own_overlap_min"] >= 0.9 and worst["proxy_overlap_min"] >= 0.9 and worst["aP_vs_reference_median"] < 0.02, worst
    smp = [json.loads(x) for x in (d / "samples" / "e4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert sorted(s["pole"] for s in smp) == sorted(items["sample_cases"]) and all(s["A"]["contexts"][0]["marked_token"] for s in smp)
    return {"gates": {k: {q: g[k][q] for q in ("value", "pass", "failing_input_failed") if q in g[k]} for k in g}, "checks": worst, "readouts": m["readouts"],
            "known_number": {q: g["known_number"][q] for q in ("median_overlap", "failing_input_median_overlap")}, "planted_cases": g["planted"]["n_cases"],
            "other_cases": g["own_room"]["failing_input"]["n_cases"], "samples": len(smp)}


def stops(out, items, model, tok, corpus, key, text: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ip = out / "items.json"
    ip.write_text(json.dumps(items))
    raised = None
    try:
        JE.Job(out / "run", job_args(out / "run", ip), items, model, tok, corpus, "cpu").run()
    except AssertionError as e:
        raised = str(e)
    m = json.loads((out / "run" / "metrics.json").read_text())
    assert raised is not None and text in raised, (text, raised)
    assert key is None or m["gates"][key]["pass"] is False, (key, m["gates"].get(key))
    return {"gate": key or "marked token", "message": raised[:140]}


def part_tiny(out: pathlib.Path) -> tuple:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    model = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=N_LAYERS).requires_grad_(False)
    ref, ref2 = (reference_a(x, tok, tokens, list(range(N_WINDOWS))) for x in (model, other))
    items = tiny_items(ref, ref2, model, tok, tokens, offsets)
    ip = out / "e4_items.json"
    NH.dump(ip, items)
    d = out / "tiny" / "t15"
    JE.Job(d, job_args(d, ip), items, model, tok, corpus, "cpu").run()
    rep = check_tiny(d, items, model, tok, tokens, ref)
    fed = []
    w = json.loads(json.dumps(items))
    w["B_t"]["members"] = w["B_t"]["members"][:-1] + [[0, int(np.flatnonzero(~RB.mask_of_members(items["B_t"]["members"], N_LAYERS - 1, 128)[0])[0]), 0.0]]
    fed.append(stops(out / "fi_bt", w, model, tok, corpus, "B_t", "differs from E1's banked B_t"))
    w = json.loads(json.dumps(items))
    Ds = [k["D_bank"] for k in w["known_cases"]]
    for k, D in zip(w["known_cases"], Ds[1:] + Ds[:1]):
        k["D_bank"] = D
    fed.append(stops(out / "fi_known", w, model, tok, corpus, "known_number", "known number failed"))
    w = json.loads(json.dumps(items))
    w["cases"][4]["A"]["token_id"][2] = (w["cases"][4]["A"]["token_id"][2] + 1) % len(tok)
    fed.append(stops(out / "fi_token", w, model, tok, corpus, None, "the marked token differs"))
    return ip, d, items, rep, fed


# ------------------------------------------------------------------------------------------ part 3: fabricated banks with planted answers
def fabricate(dst: pathlib.Path, plant: str, n_cases: int = 120, other_equal_own: bool = False) -> pathlib.Path:
    """A bank in the job's layout whose falls and sets carry planted answers; items, metrics, high_background.json and samples beside it."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "run" / "e4").mkdir(parents=True)
    (dst / "run" / "samples").mkdir()
    rng = np.random.default_rng(7)
    bt = [[5, j, 1.0] for j in range(10)]
    cases, known, recs, lines = [], [], {}, []
    for i in range(n_cases):
        nrn = i if i < 100 else i - 100                                 # 20 neurons carry both poles
        pole = f"L{8 + i % 20:02d}N{nrn}{'+' if i < 100 else '-'}"
        k = int(rng.integers(0, 26))
        e = int(rng.integers(10, 19))
        s_true = k / 50
        I0 = {"fit": 0.08 + 1.45 * s_true + rng.normal(0, 0.03), "none": 0.3 + rng.normal(0, 0.1), "negative": 0.6 - 1.45 * s_true + rng.normal(0, 0.03)}[plant]
        own = [(0, j) for j in range(50)]
        cross = own[:k] + [(1, j) for j in range(50 - k)]
        sets = {"own": own, "cross": cross, "random": [(2, j) for j in range(50)], "own_u": own[:47] + [(5, 0), (5, 1), (5, 2)], "cross_u": cross,
                "proxy_A": own[:32] + [(3, j) for j in range(18)], "proxy_B": cross[:32] + [(3, j) for j in range(e, e + 18)]}
        sets["proxy_abar_A"], sets["proxy_abar_B"] = sets["proxy_A"], sets["proxy_B"]
        if i > 0:
            sets["other"] = [(6, j) for j in range(50)]
        if i % 2 == 0:
            sets["planted"] = [(5, 0)]
        names = V.arm_names(i > 0, i % 2 == 0)
        N = 10
        b = 1.0 + 0.05 * np.arange(N)
        fall = {"identity": 0.0, "identity_fail": 0.001, "own__zero": 0.5, "own__mean": 0.5, "cross__zero": 0.5 * I0, "cross__mean": 0.5 * (I0 + 0.01),
                "random__zero": 0.02, "random__mean": 0.03, "other__zero": 0.5 if other_equal_own else 0.05, "planted_dose1": 0.01, "planted_dose10": 0.05,
                "planted_same_layer_dose1": 0.0, "planted_same_layer_dose10": 0.0}
        A = np.stack([(b * (1 - fall[x])).astype(np.float32) for x in names])
        A[names.index("identity")] = b.astype(np.float32)
        b32 = b.astype(np.float32)
        s = {"s_true": V.overlap(sets["own"], sets["cross"]), "s_proxy": V.overlap(sets["proxy_A"], sets["proxy_B"]),
             "s_proxy_abar": V.overlap(sets["proxy_abar_A"], sets["proxy_abar_B"]), "proxy_overlap_A": V.overlap(sets["proxy_A"], sets["own"]),
             "proxy_overlap_B": V.overlap(sets["proxy_B"], sets["cross"])}
        z = {"pole": pole, "a_before": b32, "arms": np.array(names), "a_after": A, **{x: np.float64(v) for x, v in s.items()}}
        for x, S in sets.items():
            z[f"set_{x}_layer"] = np.array([l for l, _ in S], np.int16)
            z[f"set_{x}_neuron"] = np.array([q for _, q in S], np.int32)
        np.savez_compressed(dst / "run" / "e4" / f"{pole}.npz", **z)
        recs[pole] = {"falls": {x: {"mean": V.job_fall(b32, A[j])["mean"]} for j, x in enumerate(names)}, **s}
        docA = list(range(100 * i, 100 * i + 10))
        cases.append({"pole": pole, "layer": 8 + i % 20, "band": "deep", "A": {"doc": docA}, "B": {"doc": docA[:1] if i % 3 == 0 else [10 ** 6 + i]}})
        if i < 20:
            known.append({"pole": pole, "D_bank": [list(x) for x in own], "D_bank_reversed": [[7, j] for j in range(50)]})
        if i < 3:
            lines.append({"pole": pole, "A": {"contexts": [{"prompt": "dry", "marked_token": " x", "a_before": float(b[0]), "a_after": {x: float(A[j, 0]) for j, x in enumerate(names)}}]},
                          "falls": {x: recs[pole]["falls"][x]["mean"] for x in names}})
    items = {"cases": cases, "known_cases": known, "sample_cases": [c["pole"] for c in cases[:3]], "B_t": {"n_up": 31, "members": bt},
             "B_t_spec4": {"members": [[5, j] for j in range(2, 12)]}}
    ip = dst / "e4_items.json"
    ip.write_text(json.dumps(items))
    m = {"stage": "instruments_e4", "t": "T_15", "status": "done", "items_sha256": sha(ip), "cases": recs, "gates": {"known_number": {"median_overlap": 1.0}},
         "known_number": {"ratio": 0.0, "wrong_ratio": 0.2, "tol": 0.01, "pass": True, "failing_input_failed": True, "failing_input": "a_wrong (dry)"},
         "completeness": {"median_err_over_pole_max": 0.001, "wrong_layer": 0.3, "tol": 0.01, "pass": True, "failing_input_failed": True}}
    (dst / "run" / "metrics.json").write_text(json.dumps(m))
    (dst / "run" / "high_background.json").write_text(json.dumps({"members": bt}))
    (dst / "run" / "samples" / "e4.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return ip


def part_verdict(out: pathlib.Path, tiny_ip, tiny_run) -> dict:
    rep = {}
    code, txt, v = run_verdict(tiny_run, tiny_ip, out / "verdict_tiny")
    assert code in (0, 1, 4) and v is not None, (code, txt[-2000:])
    rep["tiny"] = {"exit": code, "reason": v["exit_reason"], "lines": [x for x in txt.splitlines() if " GATE " in x or " RULE " in x]}
    for plant, want_r1, want_r2 in (("fit", "the forecast holds", "the program's fit generalizes"), ("none", "not separated", None), ("negative", "refuted", None)):
        ip = fabricate(out / f"fab_{plant}", plant)
        code, txt, v = run_verdict(out / f"fab_{plant}" / "run", ip, out / f"fab_{plant}" / "verdict")
        assert code == 0 and v["rules"]["R1"]["reading"] == f"EXPLORATORY: {want_r1}", (plant, code, txt[-1500:])
        assert want_r2 is None or v["rules"]["R2"]["reading"] == f"EXPLORATORY: {want_r2}", (plant, v["rules"]["R2"])
        rep[plant] = [x for x in txt.splitlines() if " RULE " in x]
    fip, frun = out / "fab_fit" / "e4_items.json", out / "fab_fit" / "run"
    rep["failing_inputs"] = {}
    for fi, gt in V.FAILING.items():
        code, txt, v = run_verdict(frun, fip, out / "fab_fit" / "verdict", "--failing-input", fi)
        want = 3 if gt in V.KNOWN_GATES else 4
        assert code == want and not v["gates"][gt]["pass"], (fi, code, txt[-800:])
        rep["failing_inputs"][fi] = [x for x in txt.splitlines() if x.startswith("EXPLORATORY FAILING_INPUT")][0] + f" (exit {code})"
    shutil.copytree(out / "fab_fit", out / "fab_incomplete")
    next((out / "fab_incomplete" / "run" / "e4").glob("*.npz")).unlink()
    code, txt, _ = run_verdict(out / "fab_incomplete" / "run", out / "fab_incomplete" / "e4_items.json", out / "fab_incomplete" / "verdict")
    assert code == 5, (code, txt[-500:])
    rep["incomplete"] = [x for x in txt.splitlines() if "INCOMPLETE" in x][0] + f" (exit {code})"
    ip = fabricate(out / "fab_broken", "fit", other_equal_own=True)
    code, txt, _ = run_verdict(out / "fab_broken" / "run", ip, out / "fab_broken" / "verdict")
    assert code == 1, (code, txt[-800:])
    rep["broken"] = [x for x in txt.splitlines() if x.startswith("EXPLORATORY a failing input did not fail")][0] + f" (exit {code})"
    fig = subprocess.run(["uv", "run", "--no-project", "--with", "matplotlib", "--with", "numpy", "python", str(FIGURES), "--verdict",
                          str(out / "fab_fit" / "verdict" / "verdict_e4.json"), "--out", str(out / "figures")], capture_output=True, text=True)
    assert fig.returncode == 0 and (out / "figures" / "fig_e4.png").exists() and (out / "figures" / "fig_e4.svg").exists(), fig.stdout + fig.stderr
    rep["figure"] = fig.stdout.strip().splitlines()[-1]
    shutil.copy(out / "fab_fit" / "verdict" / "GRID_e4.md", out / "GRID_e4_constructed.md")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    real = real_inputs()
    print("real inputs:", json.dumps(real))
    ip, d, items, rep, fed = part_tiny(out)
    print("tiny run:", json.dumps(rep))
    for f in fed:
        print("job failing input:", json.dumps(f))
    vr = part_verdict(out, ip, d)
    print("verdict on the tiny bank: exit", vr["tiny"]["exit"], "-", vr["tiny"]["reason"])
    for x in vr["tiny"]["lines"]:
        print("  tiny:", x)
    for plant in ("fit", "none", "negative"):
        for x in vr[plant]:
            print(f"  fab_{plant}:", x)
    for fi, x in vr["failing_inputs"].items():
        print("  verdict failing input:", x)
    print("  incomplete bank:", vr["incomplete"])
    print("  broken gate:", vr["broken"])
    print("  figure:", vr["figure"])
    (out / "dryrun_report.json").write_text(json.dumps({"real": real, "tiny": rep, "job_failing_inputs": fed, "verdict": vr}, indent=1))
    print("INSTRUMENTS_DRYRUN_E4_OK", out)


if __name__ == "__main__":
    main()
