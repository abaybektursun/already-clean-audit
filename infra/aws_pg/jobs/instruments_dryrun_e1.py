"""Weight-free dry run of INSTRUMENTS E1 (studies/instruments/SPEC_instruments.md §3, §10, §12 step 3). No download and no network: a tiny random
OLMo 3 (6 layers, MLP width 128, built from its transformers config), a byte-level tokenizer and a fake corpus, as the REUSE dry runs build them.

    python -m infra.aws_pg.jobs.instruments_dryrun_e1 --out <dir>

Part 1, pure logic and the real inputs (read only): instruments_values.selfcheck, instruments_linear.selfcheck; studies/instruments/inputs/e1_items.json
(56 pairs, 150 cases, the known block, the regime split) and the run's size: readouts per item by count_reads (the job's arm plan, checked equal to the
tiny run's own count in part 2), tokens by reuse_ablate.n_forwarded, timed at SPEC6's measured L40S rate (results/generic/1/generic/t15/metrics.json).
Part 2, the job end to end (jobs.instruments_e1.Job.run) on the tiny bf16 model: 6 pairs at layers 3 to 5 (terciles 0, 1, 2 by layer here), 6 cases
at layers 2 to 5, a known block, SPEC4's B_t replaced by the tiny model's own. Checked on the bank: every item and arm banked, the identity arm and
the resample known number exactly 0, the capture's mean equal to ā, A = a ⊙ c, the draws equal to an independent capture X[index, l, k], the B_t
agreement 1, 48 sample lines, the job's null_level equal to verdict2.planted_ref, count_reads equal to the readouts made, and the planted gate's
failing input (the two same-layer writes) leaving a_P exactly unchanged on every pair. Then verdict_e1.py on
that bank with the known targets recounted here from the npz: the known gate passes (exit 0 or 4; 3 or 5 would be a broken reader).
Part 3, a constructed bank: part 2's bank with a_P before set to 1 + 0.05 i and every arm's a_P after set to a planted fall, the metrics records
rebuilt by the job's record_of. verdict_e1.py: exit 0 with the planted answers (N, G, D1, D1z, D1m, D2, D3, the members ratio, the draw median);
exit 5 on a bank without one case; exit 3 with shifted known targets; exit 1 when the same-layer arms are given the dose arms' values (a failing
input that does not fail); exit 4 when the dose-10 response is below the dose-1 response (the planted gate's order condition); exit 4 when the
room is 0.02; and under each --failing-input its target gate fails (exit 3 for known_inverted, 4 otherwise).
Prints one line per gate and INSTRUMENTS_DRYRUN_E1_OK.
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
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import instruments_linear as IL
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
ITEMS = REPO / "studies/instruments/inputs/e1_items.json"
VERDICT = REPO / "studies/instruments/scripts/verdict_e1.py"
SPEC6_METRICS = REPO / "studies/reuse/results/generic/1/generic/t15/metrics.json"
SPEC6_MEMBERS = REPO / "studies/reuse/results/generic/1/generic/t15/samples/members_alone.json"
HB_T15 = REPO / "studies/reuse/results/birth/1/t15/high_background.json"
N_LAYERS, N_WINDOWS = 6, 60
PAIR_LAYERS = (3, 3, 4, 4, 5, 5)
CASE_POLES = [(2, 9, "+"), (3, 5, "+"), (3, 40, "-"), (4, 11, "+"), (5, 2, "+"), (5, 70, "+")]
BUDGET_MIN, FLAG_MIN = 150, 120
GPU_PRICE = 2.2
FAILING = {"known_inverted": "known_number", "identity_one": "identity", "resample_2abar": "resample_known", "planted_same_layer": "planted",
           "samples_missing": "samples"}                    # verdict_e1.FAILING: each failing input and the gate it must fail


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ the arm plan, counted
def count_reads(layer_has_B: bool, members_J: bool, members_O: bool, inv: bool, pair: bool) -> dict:
    """Readouts the job asks for on one item (before included), and those its cache answers when the excluded sets equal the unexcluded ones
    (B_t holds no member below the pole) and when the dose-1 member is own's first member is ignored (rare)."""
    nv = len(JE.VALUES)
    if not pair:
        n = 2 + len(JE.CASE_SETS) * nv
        return {"asked": n, "cached": 0 if layer_has_B else 3 * nv}
    J = 2 + len(JE.CASE_SETS) * nv + (nv if members_J else 0) + (2 if inv else 0) + 2 + 4 + 1
    O = 2 + len(JE.CASE_SETS) * nv + (nv if members_O else 0)
    return {"asked": J + O, "cached": 0 if layer_has_B else 2 * 3 * nv}


def real_inputs() -> dict:
    it = json.loads(ITEMS.read_text())
    P, C = it["planted"], it["primary"]
    assert len(P) == 56 and len(C) == 150 and not it["fresh"] and len(it["regime"]["members"]) == 3412, (len(P), len(C))
    hb = json.loads(HB_T15.read_text())
    up = np.cumsum([0] + hb["per_layer"])                  # B_t members below layer L: up[L]
    s6 = json.loads(SPEC6_MEMBERS.read_text())
    bos, hw = 1, UR.HARVEST_WINDOW
    nf = lambda t: RA.n_forwarded(np.asarray(t, np.int64), RA.LENGTH, bos, hw)
    asked = cached = tok = 0
    cap_tok = 0
    for p in P:
        hasB = up[p["layer"]] > 0
        c = count_reads(hasB, bool(s6[p["unit"]]["members"]), hasB, hasB, True)
        nJ = 2 + len(JE.CASE_SETS) * 7 + (7 if s6[p["unit"]]["members"] else 0) + (2 if hasB else 0) + 7
        nO = c["asked"] - nJ
        frac = 1 - c["cached"] / c["asked"]
        asked, cached = asked + c["asked"], cached + c["cached"]
        tok += frac * (nJ * nf(p["J_tok"]) + nO * nf(p["O_tok"]))
        cap_tok += nf(p["J_tok"]) + nf(p["O_tok"])
    for j in C:
        c = count_reads(up[j["layer"]] > 0, False, False, False, False)
        asked, cached = asked + c["asked"], cached + c["cached"]
        tok += (c["asked"] - c["cached"]) * nf(j["J_tok"])
        cap_tok += nf(j["J_tok"]) + nf(j["O_tok"])
    fixed_tok = 2 * 200 * RA.LENGTH + sum(RA.n_forwarded(np.asarray(x)[:16], None, bos, hw) for x in it["known"]["tok"])
    m6 = json.loads(SPEC6_METRICS.read_text())
    rate = m6["units_tokens_per_second"]
    units_s = (tok + cap_tok) / rate
    fixed_s = m6["download_and_corpus_s"] + m6["load_s"] + m6["timing_s"]["known"] + m6["timing_s"]["background"] + fixed_tok / rate
    ctx = {"pairs": sum(len(p["J_tok"]) + len(p["O_tok"]) for p in P), "cases_J": sum(len(j["J_tok"]) for j in C)}
    reg = it["regime"]["members"]
    return {"asked": asked, "cached": cached, "readouts": asked - cached, "readout_tokens": int(tok), "capture_tokens": int(cap_tok), "rate": rate,
            "units_min": units_s / 60, "wall_min": (units_s + fixed_s) / 60, "ctx": ctx, "n_sparse": sum(m["regime"] == "sparse" for m in reg),
            "spec6_dense": sum(m["regime"] == "dense" for m in reg if m["seen_spec6_cross_side1"]), "spec6_n": sum(m["seen_spec6_cross_side1"] for m in reg),
            "spec6_pairs_with_members": sum(bool(v["members"]) for v in s6.values())}


# ------------------------------------------------------------------------------------------ part 2: the tiny bank
def top_tok(ref, L, n, sign, k=50):
    a = np.asarray(ref[L])[:, :, n].astype(np.float64) * sign
    a[:, 0] = -np.inf
    return np.argsort(-a.ravel())[:k].astype(np.int64)


def tiny_items(ref, ref2, model, tok, tokens) -> dict:
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    pairs, used = [], set()
    for i, L in enumerate(PAIR_LAYERS):
        r = np.asarray(ref[L])[:, 1:, :].reshape(-1, np.asarray(ref[L]).shape[-1]).astype(np.float64)
        for x in range(3 + 7 * i, 128):
            y = (x * 37 + 11) % 128
            if x == y or (L, x) in used or (L, y) in used:
                continue
            sJ, sO = r[:, x] - np.abs(r[:, y]), r[:, y] - np.abs(r[:, x])
            J, O = np.argsort(-sJ)[:12], np.argsort(-sO)[:12]
            if (sJ[J] > 0).all() and (sO[O] > 0).all():
                used |= {(L, x), (L, y)}
                pairs.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "tercile": L - 3, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J].tolist(),
                              "O_tok": tok_rest[O].tolist(), "spec4": {"I_excluded": 0.0, "I_shuffled": 0.0, "s": 0.0}})
                break
    assert len(pairs) == len(PAIR_LAYERS), len(pairs)
    cases = []
    for L, n, s in CASE_POLES:
        t = top_tok(ref, L, n, 1.0 if s == "+" else -1.0)
        cases.append({"job": f"L{L:02d}N{n}{s}:c1", "pole": f"L{L:02d}N{n}{s}", "c": 1, "layer": L, "neuron": n, "sign": s, "tercile": 0,
                      "J_tok": t[:8].tolist(), "O_tok": t[8:20].tolist(), "spec4": {"own": 0.0, "cross": 0.0, "random": 0.0, "I": 0.0, "cancelled": False}})
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(ref, NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0).tolist() for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].tolist()
    known = {"poles": kp, "tok": ktok, "wrong_rule": "another model", "a_bank": {"T_15": [val(ref, p, t) for p, t in zip(kp, ktok)]},
             "a_wrong": {"T_15": [val(ref2, p, t) for p, t in zip(kp, ktok)]}}
    pos = UR.background_positions(15, N_WINDOWS * WINDOW // WINDOW * WINDOW, WINDOW)["e1"]
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    abar = RA.background_mean_a(model, pos, N_LAYERS - 1, **kw)
    hb = RB.high_background(abar, N_LAYERS - 1)
    regime = [{"layer": l, "neuron": k, "abar": a, "density_share": 0.01 if i % 2 else 0.9, "regime": "sparse" if i % 2 else "dense"}
              for i, (l, k, a) in enumerate(hb["members"])]
    return {"planted": pairs, "primary": cases, "fresh": [], "known": known, "spec4_B_t": [[l, k] for l, k, _ in hb["members"]],
            "regime": {"line": 0.05, "members": regime, "rule": "dry run: labels alternate"}, "known_numbers": {"targets": JE.KNOWN, "tol": JE.KNOWN_TOL},
            "sources": {"grid": []}}


def recount(d: pathlib.Path, items: dict) -> dict:
    """The known numbers from the npz, independently of the verdict: side J job falls, I = cross / own, lower median per tercile, mean over terciles."""
    def jf(b, a):
        pos = b > 0
        return float(((b[pos] - a[pos]) / b[pos]).mean())

    def I_of(z, e):
        names = [str(x) for x in z["J_arms"]]
        b, A = z["J_a_before"].astype(np.float64), z["J_a_after"].astype(np.float64)
        return jf(b, A[names.index(f"cross_{e}__mean")]) / jf(b, A[names.index(f"own_{e}__mean")])
    out = {}
    for e in ("u", "e"):
        by = {}
        for p in items["planted"]:
            by.setdefault(p["tercile"], []).append(I_of(np.load(d / "e1" / "pairs" / f"{p['unit']}.npz"), e))
        out[e] = float(np.mean([np.sort(v)[int(np.ceil(len(v) / 2)) - 1] for v in by.values()]))
    I = []
    for j in items["primary"]:
        z = np.load(d / "e1" / "cases" / f"{j['pole']}__c{j['c']}.npz")
        names = [str(x) for x in z["J_arms"]]
        b, A = z["J_a_before"].astype(np.float64), z["J_a_after"].astype(np.float64)
        own, rnd = jf(b, A[names.index("own_e__mean")]), jf(b, A[names.index("random_e__mean")])
        if own >= 2 * abs(rnd) and (b > 0).sum() >= 3:
            I.append(jf(b, A[names.index("cross_e__mean")]) / own)
    out["I"] = float(np.median(I))
    return out


def check_tiny(d: pathlib.Path, items: dict, model, tok, tokens) -> dict:
    sys.path.insert(0, str(REPO / "studies/reuse/scripts"))
    import verdict2 as V2
    m = json.loads((d / "metrics.json").read_text())
    g = m["gates"]
    assert m["status"] == "done" and g["capture"]["pass"] and g["capture"]["failing_input_failed"], g["capture"]
    assert g["identity"]["value"] == 0.0 and g["identity"]["failing_input_failed"], g["identity"]
    assert g["resample_known"]["value"] == 0.0 and g["resample_known"]["failing_input_failed"], g["resample_known"]
    assert g["coefficient"]["pass"] and g["coefficient"]["failing_input_failed"] and g["exclusion"]["pass"], (g["coefficient"], g["exclusion"])
    assert g["samples"]["pass"] and m["B_t_vs_spec4"]["jaccard"] == 1.0, (g["samples"], m["B_t_vs_spec4"])
    assert g["planted"]["failing_input"]["max_abs_delta"] == 0.0 and g["planted"]["failing_input_failed"], g["planted"]
    lines = (d / "samples" / "e1.jsonl").read_text().splitlines()
    assert len(lines) == 48 and all(json.loads(x)["contexts"][0]["marked_token"] for x in lines)
    # the draws against an independent capture
    vz = np.load(d / "values.npz")
    pos = UR.background_positions(15, N_WINDOWS * WINDOW, WINDOW)["e1"]
    cap = UR.capture(model, pos, N_LAYERS - 2, set(), tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    X = torch.stack([cap["a"][l] for l in range(N_LAYERS - 1)], 1).numpy()
    idx = vz["draw_idx"].astype(np.int64)
    L_, W_ = X.shape[1:]
    want = np.stack([X[idx[r], np.arange(L_)[:, None], np.arange(W_)[None]] for r in range(len(IV.SEEDS))])
    assert np.array_equal(vz["draws"], want) and np.allclose(vz["median"], np.median(X, 0), atol=1e-6) and np.allclose(vz["abar"], X.mean(0), rtol=1e-5, atol=1e-5)
    # every item, every arm; count_reads against the readouts made
    mk = np.zeros((N_LAYERS - 1, 128), bool)
    for l, k in items["spec4_B_t"]:
        mk[l, k] = True
    asked = 0
    for p in items["planted"]:
        z = np.load(d / "e1" / "pairs" / f"{p['unit']}.npz")
        names = {s: [str(x) for x in z[f"{s}_arms"]] for s in ("J", "O")}
        has = lambda x: len(z[f"set_{x}_layer"]) > 0
        nJ = 1 + 6 * 7 + (7 if has("J_members") else 0) + (2 if has("J_own_inv") else 0) + 7
        nO = 1 + 6 * 7 + (7 if has("O_members") else 0)
        assert len(names["J"]) == nJ and len(names["O"]) == nO and "identity" in names["O"], (p["unit"], len(names["J"]), nJ, len(names["O"]), nO)
        assert (z["J_a_after"][names["J"].index("resconst_abar")] == z["J_a_after"][names["J"].index("own_u__mean")]).all()
        for n in ("planted_same_layer_dose1", "planted_same_layer_dose10"):
            assert (z["J_a_after"][names["J"].index(n)] == z["J_a_before"]).all(), (p["unit"], n, "a same-layer write moved a_P")
        asked += count_reads(bool(mk[:p["layer"]].any()), has("J_members"), has("O_members"), has("J_own_inv"), True)["asked"]
    for j in items["primary"]:
        z = np.load(d / "e1" / "cases" / f"{j['pole']}__c{j['c']}.npz")
        assert len(z["J_arms"]) == 1 + 6 * 7, len(z["J_arms"])
        asked += count_reads(bool(mk[:j["layer"]].any()), False, False, False, False)["asked"]
    assert asked == m["readouts"] + m["readouts_cached"], (asked, m["readouts"], m["readouts_cached"])
    # the job's null against verdict2.planted_ref
    I = {p["unit"]: m["pairs"][p["unit"]]["sides"]["J"]["I"]["mean__u"] for p in items["planted"]}
    ter = {p["unit"]: p["tercile"] for p in items["planted"]}
    ref = V2.planted_ref({u: {"tercile": ter[u], "I": {"t15": I[u]}} for u in I}, {"t15": 1}, JE.W_TER, "I", [])
    assert abs(JE.null_level(I, ter)["value"] - ref["value"]) < 1e-12, (JE.null_level(I, ter), ref["value"])
    rc = recount(d, items)
    kn = g["known_number"]
    assert abs(kn["pairs"]["N_mean_unexcluded"]["value"] - rc["u"]) < 1e-9 and abs(kn["pairs"]["N_mean_excluded"]["value"] - rc["e"]) < 1e-9, (kn, rc)
    assert abs(kn["cases"]["I_excluded_mean"] - rc["I"]) < 1e-9, (kn["cases"], rc)
    return {"m": m, "recount": rc, "readouts": m["readouts"], "cached": m["readouts_cached"], "asked": asked}


def part_tiny(out: pathlib.Path) -> tuple:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    model = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=N_LAYERS).requires_grad_(False)
    ref, ref2 = (reference_a(x, tok, tokens, list(range(N_WINDOWS))) for x in (model, other))
    items = tiny_items(ref, ref2, model, tok, tokens)
    ip = out / "e1_items.json"
    NH.dump(ip, items)
    d = out / "tiny" / "t15"
    a = JE.parser().parse_args(["--out", str(d), "--stage", "e1", "--t", "T_15", "--items", str(ip), "--harvest-window", str(WINDOW),
                                "--scan-tokens", str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4"])
    a.items_sha256 = sha(ip)
    JE.Job(d, a, items, model, tok, corpus, "cpu").run()
    return ip, d, items, check_tiny(d, items, model, tok, tokens)


# ------------------------------------------------------------------------------------------ part 3: a constructed bank with planted falls
DR = (-0.03, 0.0, 0.01, 0.06)                         # per-draw offsets of the planted resample cross falls: mean 0.01, median 0.005
PLAN = {"pair": {"mean": (0.20, 0.06), "zero": (0.15, 0.07), "median": (0.12, 0.08), "res": (0.08, 0.07)},
        "case": {"mean": (0.36, 0.30), "zero": (0.33, 0.29), "median": (0.34, 0.31), "res": (0.31, 0.30)}}
MEMBERS = {"mean": 0.10, "zero": 0.08, "median": 0.06, "res0": 0.03, "res1": 0.04, "res2": 0.05, "res3": 0.04}
EXTRA = {"identity": 0.0, "identity_fail": 0.01, "resconst_2abar": 0.3, "planted_dose1": 0.02, "planted": 0.2, "planted_same_layer_dose1": 0.0,
         "planted_same_layer_dose10": 0.0, "own_inv__mean": 0.5, "cross_inv__mean": 0.45}


def planted_fall(kind: str, i: int, side: str, arm: str, plan: dict, extra: dict) -> float:
    if arm == "resconst_abar":
        arm = "own_u__mean"
    if arm in extra:
        return extra[arm]
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
    I = plan[kind]["res"][k] + DR[int(v[3])] + off if v.startswith("res") else plan[kind][v][k] + off
    return 0.5 * I


def synth(dst: pathlib.Path, src: pathlib.Path, items: dict, plan=PLAN, extra=EXTRA) -> pathlib.Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    m = json.loads((dst / "metrics.json").read_text())
    for kind, rows, sides in (("pair", items["planted"], ("J", "O")), ("case", items["primary"], ("J",))):
        for i, r in enumerate(rows):
            f = dst / "e1" / (kind + "s") / (f"{r['unit']}.npz" if kind == "pair" else f"{r['pole']}__c{r['c']}.npz")
            z = dict(np.load(f))
            for s in sides:
                n = len(z[f"{s}_tok"])
                b = (1.0 + 0.05 * np.arange(n)).astype(np.float32)
                z[f"{s}_a_before"] = b
                z[f"{s}_a_after"] = np.stack([(b * (1 - planted_fall(kind, i, s, str(arm), plan, extra))).astype(np.float32) for arm in z[f"{s}_arms"]])
            np.savez_compressed(f, **z)
            rec = JE.record_of(z, sides)
            if kind == "pair":
                m["pairs"][r["unit"]]["sides"] = rec
            else:
                m["cases"][r["job"]]["sides"] = rec
                m["cases"][r["job"]]["cancel"] = JE.case_cancel(rec["J"])
    (dst / "metrics.json").write_text(json.dumps(m))
    return dst


def expected(items: dict) -> dict:
    """The planted answers, worked from PLAN: lower median per tercile (the first pair of each tercile here), mean over terciles."""
    off = np.mean([min(0.004 * i for i, p in enumerate(items["planted"]) if p["tercile"] == t) for t in (0, 1, 2)])
    P, C = PLAN["pair"], PLAN["case"]
    N = {"mean__u": P["mean"][0] + off, "mean__e": P["mean"][1] + off, "res_mean__u": P["res"][0] + np.mean(DR) + off}
    G = {"mean": P["mean"][0] - P["mean"][1], "zero": P["zero"][0] - P["zero"][1], "median": P["median"][0] - P["median"][1], "res_mean": P["res"][0] - P["res"][1]}
    ci = np.median([0.004 * i for i in range(len(items["primary"]))])
    return {"N": N, "G": G, "R1": G["mean"] - G["res_mean"], "R1z": G["mean"] - G["zero"], "R1m": G["mean"] - G["median"],
            "R2": C["res"][0] + np.mean(DR) - C["mean"][1], "R3": C["res"][1] + np.mean(DR) - C["mean"][1],
            "members_ratio": np.mean([MEMBERS[d] for d in IV.DRAWS]) / MEMBERS["mean"], "targets": (N["mean__u"], N["mean__e"], C["mean"][1] + ci)}


def r4_recount(d: pathlib.Path, items: dict) -> dict:
    """The pooled first-order fit from the npz through instruments_linear (the verdict computes it with its own code)."""
    xs, ys = [], []
    for p in items["planted"]:
        z = np.load(d / "e1" / "pairs" / f"{p['unit']}.npz")
        U = {(int(l), int(k)): i for i, (l, k) in enumerate(zip(z["U_layer"], z["U_neuron"]))}
        for s in ("J", "O"):
            names = [str(x) for x in z[f"{s}_arms"]]
            b, A = z[f"{s}_a_before"].astype(np.float64), z[f"{s}_a_after"].astype(np.float64)
            for arm, x in (("own_u", f"{s}_own_u"), ("members", f"{s}_members")):
                if not len(z[f"set_{x}_layer"]):
                    continue
                cols = [U[(int(l), int(k))] for l, k in zip(z[f"set_{x}_layer"], z[f"set_{x}_neuron"])]
                a, c = z[f"{s}_U_a"][:, cols], z[f"{s}_U_c"][:, cols]
                y = lambda v: RA.job_fall(b, A[names.index(f"{arm}__{v}")])["mean"]
                for v, t in (("mean", z[f"set_{x}_abar"]), ("zero", np.zeros(len(cols))), ("median", z[f"set_{x}_median"])):
                    xs.append(IL.side_fall(IL.pred_fall(a, c, t), b))
                    ys.append(y(v))
                xs.append(float(np.mean([IL.side_fall(q, b) for q in IL.pred_fall(a, c, z[f"set_{x}_draws"])])))
                ys.append(float(np.mean([y(dd) for dd in IV.DRAWS])))
    s, _ = np.polyfit(xs, ys, 1)
    return {"slope": float(s), "r": float(np.corrcoef(xs, ys)[0, 1]), "n": len(xs)}


def verdict(run: pathlib.Path, ip: pathlib.Path, vout: pathlib.Path, targets, *extra) -> tuple:
    cmd = [sys.executable, str(VERDICT), "--run", str(run), "--items", str(ip), "--out", str(vout), "--known-targets", ",".join(f"{x:.12g}" for x in targets), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def part_synth(out: pathlib.Path, ip: pathlib.Path, d: pathlib.Path, items: dict) -> dict:
    sd = synth(out / "synth" / "t15", d, items)
    ex = expected(items)
    vout = out / "synth" / "verdict"
    code, so, se = verdict(sd, ip, vout, ex["targets"])
    gl = [x for x in so.splitlines() if x.startswith(("GATE", "RULE", "REPORTED", "VERDICT"))]
    assert code == 0, (code, so[-3000:], se[-3000:])
    v = json.loads((vout / "verdict_e1.json").read_text())
    rp, R = v["reported"], v["rules"]
    close = lambda a, b: abs(a - b) < 1e-6                # the bank stores a_P in float32
    for k, x in ex["N"].items():
        assert close(rp["A1"][k]["value"], x), (k, rp["A1"][k]["value"], x)
    for k, x in ex["G"].items():
        assert close(rp["G"][k]["value"], x), (k, rp["G"][k]["value"], x)
    for k in ("R1", "R1z", "R1m", "R2"):
        assert close(R[k]["value"], ex[k]), (k, R[k]["value"], ex[k])
    assert [R[k]["reading"] for k in ("R1", "R1z", "R1m", "R2")] == ["H_value holds, H_set refuted"] * 3 + ["the two routes agree"], [R[k]["reading"] for k in R]
    assert close(rp["R3"]["res_mean"]["value"], ex["R3"]) and close(rp["members_ratio"]["value"], ex["members_ratio"]), (rp["R3"]["res_mean"], rp["members_ratio"])
    assert close(rp["D1_res_median"]["value"], ex["R1"]), rp["D1_res_median"]
    pg = v["gates"]["planted"]
    n_mem = sum(len(np.load(sd / "e1" / "pairs" / f"{p['unit']}.npz")["set_J_members_layer"]) > 0 for p in items["planted"])
    assert pg["value"] == 1.0 and pg["n_sides"] == n_mem > 0 and pg["n_sides_excluded"] == len(items["planted"]) - n_mem, pg
    assert pg["failing_input"]["value"] == 0.0 and pg["failing_input_max_abs_delta"] == 0.0 and pg["failing_input_failed"], pg
    r4 = r4_recount(sd, items)
    assert close(rp["R4"]["pooled"]["slope"], r4["slope"]) and close(rp["R4"]["pooled"]["r"], r4["r"]) and rp["R4"]["pooled"]["n"] == r4["n"], (rp["R4"]["pooled"], r4)
    exits = {"green": code}
    # an incomplete bank
    inc = synth(out / "synth_incomplete" / "t15", d, items)
    (inc / "e1" / "cases" / f"{items['primary'][-1]['pole']}__c1.npz").unlink()
    c5, _, e5 = verdict(inc, ip, out / "synth_incomplete" / "verdict", ex["targets"])
    assert c5 == 5 and "E1_BANK_INCOMPLETE" in e5 and not (out / "synth_incomplete" / "verdict").exists(), (c5, e5[-2000:])
    exits["incomplete"] = c5
    # the known number off
    t3 = (ex["targets"][0] + 0.05,) + tuple(ex["targets"][1:])
    c3, so3, _ = verdict(sd, ip, out / "synth" / "verdict_known_off", t3)
    assert c3 == 3, (c3, so3[-2000:])
    exits["known_off"] = c3
    # a failing input that does not fail: the same-layer arms given the dose arms' falls
    nf = synth(out / "synth_nofail" / "t15", d, items, extra={**EXTRA, "planted_same_layer_dose1": EXTRA["planted_dose1"], "planted_same_layer_dose10": EXTRA["planted"]})
    c1, so1, _ = verdict(nf, ip, out / "synth_nofail" / "verdict", ex["targets"])
    assert c1 == 1 and "a failing input did not fail: ['planted']" in so1, (c1, so1[-2000:])
    exits["failing_input_does_not_fail"] = c1
    # the planted gate's order condition: dose 10 below dose 1
    po = synth(out / "synth_planted_order" / "t15", d, items, extra={**EXTRA, "planted": 0.01})
    c4o, so4o, _ = verdict(po, ip, out / "synth_planted_order" / "verdict", ex["targets"])
    vo = json.loads((out / "synth_planted_order" / "verdict" / "verdict_e1.json").read_text())
    assert c4o == 4 and vo["gates"]["planted"]["value"] == 0.0 and [k for k, g in vo["gates"].items() if not g["pass"]] == ["planted"], (c4o, so4o[-2000:])
    exits["planted_dose10_below_dose1"] = c4o
    # no room: G(mean) 0.02
    pr = {**PLAN, "pair": {**PLAN["pair"], "mean": (0.20, 0.18)}}
    rr = synth(out / "synth_room" / "t15", d, items, plan=pr)
    c4, so4, _ = verdict(rr, ip, out / "synth_room" / "verdict", (ex["targets"][0], ex["targets"][0] - 0.02, ex["targets"][2]))
    vr = json.loads((out / "synth_room" / "verdict" / "verdict_e1.json").read_text())
    assert c4 == 4 and not vr["gates"]["room"]["pass"] and vr["rules"]["R1"]["reading"].startswith("cancelled") and vr["rules"]["R2"]["reading"] == "the two routes agree", (c4, so4[-2000:])
    exits["room_0.02"] = c4
    fails = {}
    for fi, target in sorted(FAILING.items()):
        cf, sof, sef = verdict(sd, ip, vout, ex["targets"], "--failing-input", fi)
        f = json.loads((vout / f"failing_input_{fi}" / "verdict_e1.json").read_text())
        want = 3 if target == "known_number" else 4
        assert cf == want and not f["gates"][target]["pass"] and all(r["reading"].startswith("cancelled") for r in f["rules"].values()), (fi, cf, sof[-2000:], sef[-2000:])
        assert all(g["pass"] for k, g in f["gates"].items() if k != target), (fi, {k: g["pass"] for k, g in f["gates"].items()})
        fails[fi] = [x for x in sof.splitlines() if x.startswith("FAILING_INPUT")][0]
        exits[fi] = cf
    return {"lines": gl, "exits": exits, "fails": fails, "v": v, "r4": r4}


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
    e = real_inputs()
    print(f"REAL INPUTS: 56 pairs ({e['ctx']['pairs']} contexts), 150 cases ({e['ctx']['cases_J']} J contexts); B_t sparse {e['n_sparse']} of 3412; SPEC6's "
          f"side-1 cross members {e['spec6_n']}, dense {e['spec6_dense']}, on {e['spec6_pairs_with_members']} pairs")
    print(f"SIZE: readouts asked {e['asked']}, at most {e['cached']} answered by the cache (excluded = unexcluded below layer 7) -> {e['readouts']} readouts, "
          f"{e['readout_tokens'] / 1e6:.1f} M tokens + captures {e['capture_tokens'] / 1e6:.2f} M; at SPEC6's L40S rate {e['rate']:.0f} tokens/s: units "
          f"{e['units_min']:.0f} min, job wall {e['wall_min']:.0f} min (budget {BUDGET_MIN}, flag {FLAG_MIN}); about ${e['wall_min'] / 60 * GPU_PRICE:.2f} at ${GPU_PRICE}/h")
    ip, d, items, t = part_tiny(out)
    mt = t["m"]
    print(f"TINY JOB: 6 pairs, 6 cases on a 6-layer bf16 OLMo 3; readouts {t['readouts']} + {t['cached']} from the cache = count_reads {t['asked']}; identity "
          f"{mt['gates']['identity']['value']}, resample known {mt['gates']['resample_known']['value']}, capture {mt['gates']['capture']['max_abs']:.1e}, "
          f"A = a ⊙ c to {mt['gates']['coefficient']['value']:.1e}, draws = an independent capture's X[index], B_t agreement {mt['B_t_vs_spec4']['jaccard']}, "
          f"48 sample lines; null_level = verdict2.planted_ref; known numbers = the npz recount")
    rc = t["recount"]
    code, so, se = verdict(d, ip, out / "tiny" / "verdict", (rc["u"], rc["e"], rc["I"]))
    assert code in (0, 4) and "E1_BANK_INCOMPLETE" not in se, (code, so[-3000:], se[-3000:])
    for x in so.splitlines():
        if x.startswith(("GATE", "VERDICT")):
            print("  tiny", x)
    s = part_synth(out, ip, d, items)
    print("CONSTRUCTED BANK (planted falls), verdict lines:")
    for x in s["lines"]:
        print("  ", x)
    print(f"planted answers reproduced: N, G, D1, D1z, D1m, D2, D3, the members ratio and the draw-median D1 to 1e-6 (float32 bank); R4 pooled slope "
          f"{s['r4']['slope']:.4f} r {s['r4']['r']:.4f} over {s['r4']['n']} points equals instruments_linear's recount")
    for x in s["fails"].values():
        print("  ", x)
    print("EXITS:", json.dumps(s["exits"]))
    (out / "metrics.json").write_text(json.dumps({"real_inputs": e, "tiny_exit": code, "synth_exits": s["exits"], "failing_inputs": s["fails"]}, indent=1, default=str))
    print("INSTRUMENTS_DRYRUN_E1_OK")


if __name__ == "__main__":
    main()
