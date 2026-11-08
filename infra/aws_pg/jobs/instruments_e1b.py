"""INSTRUMENTS E1b job, one model per call (studies/instruments/SPEC_instruments.md §3.6, §3.2-§3.3; amendments 6, 7, 20-22). E1's readouts
replicated on google/gemma-3-4b-pt (g4) and Qwen/Qwen3-4B-Base (q4b) through infra/pg/instruments_adapter.py. A subclass of jobs.instruments_e1.Job:
the readout and its cache (read), the excluded-set check, the state, the bank format, the identity / resample-known / planted rows and the planted
gate (gate_rows, gates_pairs) are E1's code, unchanged; capture, attribution, c and completeness go through the adapter; reuse_ablate (readout,
replaced, ablation_dict, top_m, random_matched, job_fall), reuse_birth (high_background, top_m_excluded, random_matched_excluded), reuse_generic
(members_alone, single_largest, cut) and instruments_values are imported unchanged.

    python -m infra.aws_pg.jobs.instruments_e1b --model-key g4 --items <s3 e1b_items.json> --corpus <s3 corpus> --out <dir>

Weights: the harvest's mechanism for these two models (jobs.neurons_harvest without --revision): hf_online_setup(), then the single snapshot of the
repo in the local HF cache (neurons_corpus.local_snapshot), which `pg run --weights models--google--gemma-3-4b-pt,models--Qwen--Qwen3-4B-Base`
syncs from the S3 weight cache; its sha is asserted equal to the bank's revision and matrix_models.KNOWN's.
Order: known (the adapter on real weights: the bank's a at 16 top contexts of 3 poles reproduced at the full window prefix, failing input another
pole's bank a; completeness of the adapter's decomposition of g and u, failing input the wrong layer's columns; both asserted, as
jobs.reuse_ablate.Job.known asserts them); background (ā = adapter background_mean_a over the items' 200 positions, a second capture X [200, n_up, W]
whose mean is asserted equal to ā (the capture gate), the values mean, zero, median, res0..res3 (instruments_values), B_t = the global top 1% of |ā|
over the n_up = n_layers - 1 upstream layers, the per-layer 1% reported, and the high-background gate); the 30 singles (two per layer: own_u =
top 50 by direct attribution, random_u = random_matched to own_u, other_own_u = the partner's own_u (the failing input), mean value, plus identity);
the 56 pairs (E1's arms: side J identity, own/cross/random under u and e, members, every set under the 7 values, own_inv and cross_inv, resconst_abar
and resconst_2abar, planted_dose1 and planted (10 ā), planted_same_layer_dose1 and _dose10, identity_fail; side O identity and every set under the
7 values); then the reported null N(mean, u), N(mean, e) and the summary. metrics.json after every item.
Bank: e1b/<model>/pairs/<unit>.npz and e1b/<model>/singles/<pole>.npz in E1's format (per side s_tok, s_a_before, s_arms, s_a_after, s_U_a and s_U_c
= a_k(x) and c_k(x) at the marked token for every member of every set in play, U_layer/U_neuron; per set set_<x>_layer, _neuron, _abar, _median,
_draws, _in_B); pairs add s_terms [N, 2] (the two planted terms before any write; the side purity). values.npz, high_background.json (global B_t),
high_background_per_layer.json, samples/e1b_<model>.jsonl (the first 3 pairs, side J, values mean, zero, median, resample x u, e x own, cross: 48
lines; the first 3 singles x own_u, random_u: 6 lines), samples/known_number_e1b_<model>.json.
Gates in metrics.json (the verdict decides; the job asserts the instrument checks): known_activation, completeness, capture, high_background (the
within-layer top-1% cut >= 10 x the layer's median |ā| in at least half the n_up layers; failing input ā replaced by Gaussian draws with each layer's
standard deviation, seed 0), own_fall (median over the 30 singles of the own_u fall >= 0.3; failing input other_own_u in place of own_u),
random_matched (median random_u fall < 0.5 x the median own_u fall; failing input random_u replaced by own_u; other_own_u in place of own_u
reported), identity, resample_known, planted
(amendment 20), coefficient (A = a ⊙ c; failing input c from the wrong layer's columns), exclusion, samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_e1 as JE
from infra.pg import instruments_adapter as AD
from infra.pg import instruments_linear as IL
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import reuse_generic as RG
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US

M = RB.M
CAP = 12
VALUES = IV.VALUES
DRAWS = IV.DRAWS
CONS = JE.CONS
CASE_SETS = JE.CASE_SETS
PAIR_SETS = JE.PAIR_SETS
SINGLE_SETS = ("own_u", "random_u", "other_own_u")
OWN_LINE, RANDOM_FRAC = 0.3, 0.5
HB_RATIO, HB_LAYER_SHARE, HB_FAIL_SEED = 10.0, 0.5, 0
N_SAMPLE_PAIRS, N_SAMPLE_SINGLES = 3, 3
SAMPLE_VALUES = JE.SAMPLE_VALUES
N_SAMPLE_LINES = N_SAMPLE_PAIRS * len(SAMPLE_VALUES) * len(CONS) * 2 + N_SAMPLE_SINGLES * 2
REPOS = {"g4": "google/gemma-3-4b-pt", "q4b": "Qwen/Qwen3-4B-Base"}


# ------------------------------------------------------------------------------------------ pure functions (the dry run and the verdict check against them)
def layer_cut_ratio(abar: np.ndarray) -> np.ndarray:
    """Per layer: the smallest |ā| of the layer's top round(0.01 W) (reuse_generic.cut's per-layer rule) over the layer's median |ā|."""
    x = np.abs(np.asarray(abar, np.float64))
    k = int(round(RB.FRAC * x.shape[1]))
    cut = -np.sort(-x, 1)[:, k - 1]
    return cut / np.median(x, 1)


def hb_gate(abar: np.ndarray) -> dict:
    """The high-background gate on ā [n_up, W], and its failing input: Gaussian draws with each layer's standard deviation (seed 0)."""
    ab = np.asarray(abar, np.float64)
    r = layer_cut_ratio(ab)
    fake = np.random.default_rng(HB_FAIL_SEED).normal(0, 1, ab.shape) * ab.std(1)[:, None]
    rf = layer_cut_ratio(fake)
    share, share_f = float(np.mean(r >= HB_RATIO)), float(np.mean(rf >= HB_RATIO))
    return {"value": share, "line": HB_LAYER_SHARE, "ratio_line": HB_RATIO, "pass": bool(share >= HB_LAYER_SHARE), "ratio_by_layer": r.tolist(),
            "failing_input": {"input": "ā replaced by Gaussian draws with each layer's standard deviation (seed 0)", "value": share_f,
                              "ratio_median": float(np.median(rf))},
            "failing_input_failed": not share_f >= HB_LAYER_SHARE, "identity": None,
            "rule": "share of the n_up upstream layers whose top-1% cut (the smallest |ā| of the layer's top round(0.01 W)) is >= 10 x the layer's median |ā|; passes at >= 0.5"}


def single_gates(falls: dict) -> dict:
    """§3.6's instrument gates over the singles' job falls {pole: {arm: fall}}. own: failing input 'the sets of a different pole' (the partner's own_u in
    place of own_u). random-matched: failing input random_u replaced by own_u; the partner's set in place of own_u is reported (it cannot make this
    gate fail: on the dry run's tiny bank the random fall -0.05 stayed under half the partner set's 0.19)."""
    med = lambda arm: float(np.nanmedian([f[arm] for f in falls.values()]))
    own, rnd, oth = med("own_u"), med("random_u"), med("other_own_u")
    return {"own_fall": {"value": own, "line": OWN_LINE, "pass": bool(own >= OWN_LINE), "n": len(falls),
                         "failing_input": {"input": "the partner's own_u set (the sets of a different pole) in place of own_u", "value": oth},
                         "failing_input_failed": not oth >= OWN_LINE, "rule": "median over the singles of the own_u fall (mean value) >= 0.3"},
            "random_matched": {"value": rnd, "line": RANDOM_FRAC * own, "pass": bool(rnd < RANDOM_FRAC * own), "own_median": own,
                               "failing_input": {"input": "random_u replaced by own_u (a matching that returns the set it matches)", "value": own, "line": RANDOM_FRAC * own},
                               "failing_input_failed": not own < RANDOM_FRAC * own,
                               "partner_in_place_of_own": {"value": rnd, "line": RANDOM_FRAC * oth, "pass": bool(rnd < RANDOM_FRAC * oth),
                                                           "note": "reported: §3.6's 'sets of a different pole' cannot make this gate fail (the dry run's tiny bank)"},
                               "rule": "median random_u fall < 0.5 x the median own_u fall"}}


class Job(JE.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        self.mk = a.model_key
        self.ad = AD.Adapter(model, n_layers=items["n_layers"])
        d = self.ad.describe()
        assert (d["arch"], d["width"], d["hidden"], d["act"]) == (items["arch"], items["width"], items["hidden"], items["act_fn_measured"]), (d, items["arch"])
        self.rows = AD.Rows(model, self.ad)
        self.tower, self.layers, self.W = self.ad.tower, self.ad.layers, self.ad.width
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        assert self.bos == items["bos_id"], (self.bos, items["bos_id"])
        self.hw = a.harvest_window
        assert self.hw == items["window"], (self.hw, items["window"])
        self.start = RA.window_start(self.bos)
        self.dec = NH.decoder(tok)
        self.spec = {"cap": CAP}
        self.bank_dir = self.out / "e1b" / self.mk
        for x in ("pairs", "singles"):
            (self.bank_dir / x).mkdir(parents=True, exist_ok=True)
        (self.out / "samples").mkdir(parents=True, exist_ok=True)
        self.samples_path = self.out / "samples" / f"e1b_{self.mk}.jsonl"
        self.m = {"stage": "e1b", "model_key": self.mk, "repo": items["repo"], "revision": items["revision"], "items_sha256": a.items_sha256, "adapter": d,
                  "harvest_window": self.hw, "window": RA.LENGTH, "bos": self.bos, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"),
                  "dtype": str(next(model.parameters()).dtype), "status": "running", "m": M, "cap": CAP, "values": list(VALUES), "seeds": list(IV.SEEDS),
                  "job_fall_rule": "mean over contexts with a_P > 0", "replacement": "the value written into down_proj's input at every window position; BOS left",
                  "adapter_known_number": "the code-level equality on an Olmo 3 (instruments_adapter.olmo_equality, in the dry run); here the activation "
                                          "known number and completeness on these weights",
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "readouts": 0, "readouts_cached": 0,
                  "n_to_read": {"singles": len(items["singles"]), "pairs": len(items["planted"])}, "n_done": {"singles": 0, "pairs": 0},
                  "gates": {}, "singles": {}, "pairs": {}}
        self.state = {}
        self.chk = {"id": [], "id_fail": [], "res": [], "res_fail": [], "planted": [], "c": [], "c_fail": []}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0}
        self.n_samples = 0
        self.write()

    # -------------------------------------------------------------------------------------- the adapter's capture and attribution
    def capture(self, toks, unit):
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)
        return AD.capture(self.ad, self.model, np.asarray(toks), unit["layer"], {unit["layer"]}, **self.kw(RA.LENGTH))

    def side(self, toks, unit):
        cap = self.capture(toks, unit)
        return cap, AD.direct_A(self.ad, self.rows, cap, unit, self.W)

    def planted_sides(self, p: dict):
        unit = UR.planted_unit(p)
        Jt, Ot = np.asarray(p["J_tok"])[:CAP], np.asarray(p["O_tok"])[:CAP]
        pool = np.concatenate([Jt, Ot])
        cap, A = self.side(pool, unit)
        return unit, Jt, Ot, pool, cap, A

    # -------------------------------------------------------------------------------------- the known number on these weights
    def known(self):
        t0 = time.time()
        k = self.items["known"]
        out = {}
        for i, p in enumerate(k["poles"]):
            L, n, _ = NH.parse_pole(p)
            toks = np.asarray(k["tok"][i])
            cap = AD.capture(self.ad, self.model, toks, L, {L}, **self.kw(None))
            a = cap["a"][L][:, n].cpu().numpy()
            b, w = np.asarray(k["a_bank"][i], float), np.asarray(k["a_wrong"][i], float)
            c, cw = AD.completeness(self.ad, self.rows, cap, L, n), AD.completeness(self.ad, self.rows, cap, L, n, shift=1)
            out[p] = {"known_ratio": US.known_ratio(a, b), "wrong_ratio": US.known_ratio(a, w), "err_g_median": float(np.median(c["err_g"])),
                      "err_u_median": float(np.median(c["err_u"])), "wrong_layer_err_g_median": float(np.median(cw["err_g"])),
                      "wrong_layer_err_u_median": float(np.median(cw["err_u"])), "r_median": float(cap["r"][L].median()),
                      "samples": [{"tok": int(toks[j]), **NH.context(self.tokens, self.offsets, int(toks[j]), self.dec), "a_ours": float(a[j]), "a_bank": float(b[j])}
                                  for j in range(2)]}
            self.m["tokens_forwarded"] += RA.n_forwarded(toks, None, self.bos, self.hw)
            self.m["captures"] += len(toks)
        ratio = float(np.median([v["known_ratio"] for v in out.values()]))
        wr = float(np.median([v["wrong_ratio"] for v in out.values()]))
        errs = [v[x] for v in out.values() for x in ("err_g_median", "err_u_median")]
        werrs = [v[x] for v in out.values() for x in ("wrong_layer_err_g_median", "wrong_layer_err_u_median")]
        g = self.m["gates"]
        g["known_activation"] = {**US.gate_known(ratio, wr), "failing_input": {"input": k["wrong_rule"], "value": wr}, "n_ctx": k["n_ctx"],
                                 "per_pole": {p: {x: v[x] for x in v if x != "samples"} for p, v in out.items()},
                                 "rule": "median over 3 poles of median |a_ours - a_bank| / max |a_bank| at 16 top contexts, full window prefix <= 1e-2"}
        g["completeness"] = {**US.gate_complete(float(np.median(errs)), float(np.median(werrs))),
                             "failing_input": {"input": "the wrong layer's columns (Rows.loads shift 1)", "value": float(np.median(werrs))},
                             "rule": "median |r_L G'.(embedding + attention + MLP terms) - g| / max |g| (and u) <= 1e-2"}
        self.m["timing_s"]["known"] = round(time.time() - t0, 1)
        self.write()
        NH.dump(self.out / "samples" / f"known_number_e1b_{self.mk}.json", out)
        assert g["known_activation"]["pass"] and g["known_activation"]["failing_input_failed"], f"activation known number: {g['known_activation']}"
        assert g["completeness"]["pass"] and g["completeness"]["failing_input_failed"], f"completeness: {g['completeness']}"

    # -------------------------------------------------------------------------------------- ā, B_t, the capture and the values
    def background(self):
        t0 = time.time()
        n_up = len(self.layers) - 1
        pos = np.asarray(self.items["background"]["positions"], np.int64)
        kw = self.kw(RA.LENGTH)
        self.abar = AD.background_mean_a(self.ad, self.model, pos, n_up, **kw)
        X = AD.capture_background(self.ad, self.model, pos, n_up, **kw)
        self.m["captures"] += 2 * len(pos)
        self.m["tokens_forwarded"] += 2 * RA.n_forwarded(pos, RA.LENGTH, self.bos, self.hw)
        chk = IV.mean_check(X, self.abar)
        self.tab, self.draw_idx = IV.tables(X, self.abar)
        P = X.shape[0]
        del X
        self.tab["const_abar"] = IV.constant_tables(self.abar, P, self.draw_idx)[0]
        self.tab["const_2abar"] = IV.constant_tables(2 * self.abar, P, self.draw_idx)[0]
        self.tab["dose"] = JE.DOSE * self.abar.float()
        hb = RB.high_background(self.abar, n_up)
        self.mask = hb["mask"]
        RB.write_high_background(self.out / "high_background.json", hb, self.mk)
        pl = RG.cut(self.abar, n_up, RB.FRAC, per_layer=True)
        mk_pl = pl["mask"].cpu().numpy()
        NH.dump(self.out / "high_background_per_layer.json", {"cut_by_layer": pl["cut"], "count": pl["count"], "per_layer": pl["per_layer"],
                                                               "members": [[int(l), int(k), float(self.abar[l, k])] for l, k in np.argwhere(mk_pl)],
                                                               "rule": "reuse_generic.cut(abar, n_up, 0.01, per_layer=True): the round(0.01 W) largest |ā| in each layer"})
        ab = self.abar.cpu().numpy()
        self.m["background"] = {"n_positions": int(P), "n_up": n_up, "finite": bool(np.isfinite(ab).all()), "abar_abs_median": float(np.median(np.abs(ab)))}
        self.m["high_background"] = {k: hb[k] for k in ("cut", "count", "n_entries", "n_up", "frac", "per_layer")}
        self.m["high_background"]["cut_over_median_abs_abar"] = hb["cut"] / float(np.median(np.abs(ab)))
        self.m["high_background"]["per_layer_rule"] = {"count": pl["count"], "overlap_with_global": int((mk_pl & self.mask.cpu().numpy()).sum())}
        self.m["gates"]["capture"] = {**chk, "identity": {"value": 0.0, "reason": "the gate compares one capture's mean with ā from another"}}
        self.m["gates"]["high_background"] = hb_gate(ab)
        np.savez_compressed(self.out / "values.npz", abar=ab, median=self.tab["median"].cpu().numpy(),
                            draw_idx=self.draw_idx.astype(np.uint8 if P <= 256 else np.int64), draws=torch.stack([self.tab[d] for d in DRAWS]).cpu().numpy(),
                            seeds=np.array(IV.SEEDS), positions=pos)
        self.m["timing_s"]["values"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"
        assert chk["pass"] and chk["failing_input_failed"], f"the capture's mean is not ā, or its failing input passed: {chk}"

    # -------------------------------------------------------------------------------------- readouts
    def before(self, st: dict, side: str):
        r = RA.readout(self.model, st["tok"][side], st["unit"], None, **self.kw(RA.LENGTH))
        st["before"][side] = r["a"]
        act = self.layers[st["unit"]["layer"]].mlp.act_fn
        ws = np.array([t["w"] * t["sign"] for t in st["unit"]["terms"]], np.float32)
        st.setdefault("terms", {})[side] = ws * act(torch.from_numpy(r["g"])).numpy() * r["u"]
        self.m["readouts"] += 1
        self.m["forwards"] += len(st["tok"][side])
        self.m["tokens_forwarded"] += RA.n_forwarded(st["tok"][side], RA.LENGTH, self.bos, self.hw)

    def coef(self, cap, unit, A):
        c, crec = AD.coefficients_c(self.ad, self.rows, cap, unit, self.W, A)
        self.chk["c"].append(crec["value"])
        self.chk["c_fail"].append(crec["failing_input"])
        return c, crec

    # -------------------------------------------------------------------------------------- singles
    def single_pair(self, s1: dict, s2: dict, sample: bool):
        assert s1["partner"] == s2["pole"] and s2["partner"] == s1["pole"] and s1["layer"] == s2["layer"], (s1["pole"], s2["pole"])
        pre = {}
        for s in (s1, s2):
            unit = UR.pole_unit(s["pole"])
            assert unit["layer"] == s["layer"], s["pole"]
            toks = np.asarray(s["J_tok"])[:CAP]
            cap, A = self.side(toks, unit)
            c, crec = self.coef(cap, unit, A)
            pre[s["pole"]] = {"unit": unit, "toks": toks, "cap": cap, "A": A, "c": c, "crec": crec, "own": RA.top_m(A.sum(0), M, self.W)}
        for s in (s1, s2):
            t0 = time.time()
            q = pre[s["pole"]]
            sets = {"J_own_u": q["own"], "J_random_u": RA.random_matched(q["own"], [], self.abar), "J_other_own_u": pre[s["partner"]]["own"]}
            st = self.new_state(q["unit"], {"J": q["toks"]}, q["cap"], q["A"], q["c"], sets, ("J",))
            self.before(st, "J")
            self.read(st, "J", "identity", [], "mean")
            for x in SINGLE_SETS:
                self.read(st, "J", f"{x}__mean", sets[f"J_{x}"], "mean")
            z = self.bank(st, self.bank_dir / "singles" / f"{s['pole']}.npz", {"unit": s["pole"], "layer": s["layer"], "kind": "single", "partner": s["partner"]})
            names = [str(x) for x in z["J_arms"]]
            falls = {n: RA.job_fall(z["J_a_before"], z["J_a_after"][i]) for i, n in enumerate(names)}
            self.m["singles"][s["pole"]] = {"layer": s["layer"], "sign": s["sign"], "tercile": s["tercile"], "partner": s["partner"],
                                            "npz": f"e1b/{self.mk}/singles/{s['pole']}.npz", "falls": {n: {k: f[k] for k in ("mean", "median", "n_pos")} for n, f in falls.items()},
                                            "c_check": q["crec"], "set_sizes": {x: len(v) for x, v in sets.items()},
                                            "n_in_B": {x: int(RB.in_B(v, self.mask).sum()) for x, v in sets.items()}, "seconds": round(time.time() - t0, 1)}
            self.gate_rows(s["pole"], z, ("J",))
            if sample and self.m["n_done"]["singles"] < N_SAMPLE_SINGLES:
                self.single_samples(s, st)
            self.m["n_done"]["singles"] += 1
            self.write()

    def gates_singles(self):
        falls = {p: {x: r["falls"][f"{x}__mean"]["mean"] for x in SINGLE_SETS} for p, r in self.m["singles"].items()}
        g = single_gates(falls)
        idv = max(r["falls"]["identity"]["mean"] if np.isfinite(r["falls"]["identity"]["mean"]) else 0.0 for r in self.m["singles"].values())
        for k in g:
            g[k]["identity"] = {"value": idv, "reason": "the identity arm's fall on the singles"}
        self.m["gates"].update(g)
        self.write()

    # -------------------------------------------------------------------------------------- pairs
    def pair(self, p: dict, sample: bool):
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        c, crec = self.coef(cap, unit, A)
        nJ, W = len(Jt), self.W
        sJ, sO = A[:nJ].sum(0), A[nJ:].sum(0)
        Ju, Ou = RA.top_m(sJ, M, W), RA.top_m(sO, M, W)
        Je, Oe = self.rank(sJ), self.rank(sO)
        sets = {"J_own_u": Ju, "J_cross_u": Ou, "J_random_u": RA.random_matched(Ou, Ju, self.abar),
                "J_own_e": Je, "J_cross_e": Oe, "J_random_e": RB.random_matched_excluded(Oe, Je, self.abar, self.mask),
                "J_members": RG.members_alone(Ou, self.mask, self.abar),
                "O_own_u": Ou, "O_cross_u": Ju, "O_random_u": RA.random_matched(Ju, Ou, self.abar),
                "O_own_e": Oe, "O_cross_e": Je, "O_random_e": RB.random_matched_excluded(Je, Oe, self.abar, self.mask),
                "O_members": RG.members_alone(Ju, self.mask, self.abar),
                "J_single": RG.single_largest(Ou, self.abar), "J_one": Ju[:1],
                "J_own_inv": JE.members_only(sJ, M, W, self.mask), "J_cross_inv": JE.members_only(sO, M, W, self.mask)}
        self.excluded([sets[x] for x in ("J_own_e", "J_cross_e", "J_random_e", "O_random_e")])
        assert not sets["J_members"] or sets["J_single"] == sets["J_members"][:1], "the planted member is not the cross set's largest-|ā| B_t member"
        st = self.new_state(unit, {"J": Jt, "O": Ot}, cap, A, c, sets, ("J", "O"))
        st["c_rec"] = crec
        st["same"] = JE.same_layer_neuron(unit, W)
        S = sets
        self.before(st, "J")
        self.read(st, "J", "identity", [], "mean")
        for x in CASE_SETS:
            for v in VALUES:
                self.read(st, "J", f"{x}__{v}", S[f"J_{x}"], v)
        if S["J_members"]:
            for v in VALUES:
                self.read(st, "J", f"members__{v}", S["J_members"], v)
        if S["J_own_inv"]:
            self.read(st, "J", "own_inv__mean", S["J_own_inv"], "mean")
            self.read(st, "J", "cross_inv__mean", S["J_cross_inv"], "mean")
        self.read(st, "J", "resconst_abar", S["J_own_u"], "const_abar")
        self.read(st, "J", "resconst_2abar", S["J_own_u"], "const_2abar")
        self.read(st, "J", "planted_dose1", S["J_single"], "mean")
        self.read(st, "J", "planted", S["J_single"], "dose")
        self.read(st, "J", "planted_same_layer_dose1", S["J_single"], "mean", same=st["same"])
        self.read(st, "J", "planted_same_layer_dose10", S["J_single"], "dose", same=st["same"])
        self.read(st, "J", "identity_fail", S["J_one"], "mean")
        self.before(st, "O")
        self.read(st, "O", "identity", [], "mean")
        for x in PAIR_SETS:
            if S[f"O_{x}"]:
                for v in VALUES:
                    self.read(st, "O", f"{x}__{v}", S[f"O_{x}"], v)
        z = self.finish_pair(p, st)
        self.gate_rows(p["unit"], z, ("J", "O"))
        if sample:
            self.samples(p, st)

    def finish_pair(self, p: dict, st: dict) -> dict:
        t0 = time.time()
        meta = {"unit": p["unit"], "layer": p["layer"], "tercile": p["tercile"], "kind": "pair", "same_layer_neuron": st["same"],
                **{f"{s}_terms": st["terms"][s].astype(np.float32) for s in ("J", "O")}}
        z = self.bank(st, self.bank_dir / "pairs" / f"{p['unit']}.npz", meta)
        S = st["sets"]
        tJ, tO = st["terms"]["J"], st["terms"]["O"]
        self.m["pairs"][p["unit"]] = {"layer": p["layer"], "tercile": p["tercile"], "a": p["a"], "b": p["b"], "lam": p["lam"], "npz": f"e1b/{self.mk}/pairs/{p['unit']}.npz",
                                      "sides": JE.record_of(z, ("J", "O")), "c_check": st["c_rec"], "set_sizes": {x: len(v) for x, v in S.items()},
                                      "n_in_B": {x: int(RB.in_B(v, self.mask).sum()) if v else 0 for x, v in S.items()},
                                      "single": [[int(l), int(k), float(self.abar[l, k])] for l, k in S["J_single"]], "same_layer_neuron": [p["layer"], st["same"]],
                                      "purity": {"J": float(np.mean(tJ[:, 0] > tJ[:, 1])), "O": float(np.mean(tO[:, 1] > tO[:, 0]))},
                                      "seconds": round(time.time() - t0, 1)}
        return z

    # -------------------------------------------------------------------------------------- samples
    def samples(self, p: dict, st: dict):
        """E1's 16 lines per sampled pair (side J; 4 values x 2 constructions x own, cross)."""
        pr = self.prompts(st["tok"]["J"])
        b = st["before"]["J"]
        tab = {v: self.tab[v].cpu().numpy() for v in ("mean", "median") + DRAWS}
        with open(self.samples_path, "a") as fh:
            for v in SAMPLE_VALUES:
                for e in CONS:
                    for x in ("own", "cross"):
                        S = st["sets"][f"J_{x}_{e}"]
                        vals = DRAWS if v == "resample" else (v,)
                        aft = [st["after"]["J"][f"{x}_{e}__{w}"] for w in vals]
                        wv = lambda l, k: [0.0 if w == "zero" else float(tab[w][l, k]) for w in vals]
                        ctx = [{"tok": int(t), **pr[i], "a_before": float(b[i]), "a_after": [float(a[i]) for a in aft],
                                "fall": [float((b[i] - a[i]) / b[i]) for a in aft], "terms_before": st["terms"]["J"][i].tolist()} for i, t in enumerate(st["tok"]["J"])]
                        fl = [RA.job_fall(b, a)["mean"] for a in aft]
                        fh.write(json.dumps({"model": self.mk, "kind": "pair", "pair": p["unit"], "layer": p["layer"], "side": "J", "value": v, "construction": e,
                                             "arm": x, "draws": list(vals), "members": [[int(l), int(k), float(tab["mean"][l, k]), wv(l, k)] for l, k in S],
                                             "job_fall": fl, "job_fall_mean_over_draws": float(np.mean(fl)), "job_fall_median_over_draws": float(np.median(fl)),
                                             "contexts": ctx}, ensure_ascii=False) + "\n")
                        self.n_samples += 1

    def single_samples(self, s: dict, st: dict):
        pr = self.prompts(st["tok"]["J"])
        b = st["before"]["J"]
        ab = self.abar.cpu().numpy()
        with open(self.samples_path, "a") as fh:
            for x in ("own_u", "random_u"):
                aft = st["after"]["J"][f"{x}__mean"]
                S = st["sets"][f"J_{x}"]
                ctx = [{"tok": int(t), **pr[i], "a_before": float(b[i]), "a_after": float(aft[i]), "fall": float((b[i] - aft[i]) / b[i])} for i, t in enumerate(st["tok"]["J"])]
                fh.write(json.dumps({"model": self.mk, "kind": "single", "pole": s["pole"], "layer": s["layer"], "partner": s["partner"], "value": "mean",
                                     "arm": x, "members": [[int(l), int(k), float(ab[l, k]), float(ab[l, k])] for l, k in S],
                                     "job_fall": RA.job_fall(b, aft)["mean"], "contexts": ctx}, ensure_ascii=False) + "\n")
                self.n_samples += 1

    # -------------------------------------------------------------------------------------- the reported null and the summary
    def null_report(self):
        P = self.m["pairs"]
        ter = {u: v["tercile"] for u, v in P.items()}
        I = {e: {u: v["sides"]["J"]["I"][f"mean__{e}"] for u, v in P.items()} for e in CONS}
        Nu, Ne = JE.null_level(I["u"], ter), JE.null_level(I["e"], ter)
        self.m["null_mean"] = {"N_mean_unexcluded": Nu, "N_mean_excluded": Ne, "G_mean": Nu["value"] - Ne["value"],
                               "rule": "SPEC4's null on side J (cross D(O) / own D(J)), terciles 1/3 each; reported, the verdict reads R5"}

    def summarize(self):
        c, g = self.chk, self.m["gates"]
        g["identity"] = {"value": max(c["id"]), "tol": JE.ID_TOL, "pass": max(c["id"]) <= JE.ID_TOL, "n_sides": len(c["id"]),
                         "failing_input": {"input": "own_u's first member written at ā in place of the empty set", "value": max(c["id_fail"])},
                         "failing_input_failed": max(c["id_fail"]) > JE.ID_TOL, "identity": {"value": max(c["id"]), "reason": "read on the identity arm itself"},
                         "rule": "max over sides and contexts of |a_P after - a_P before| / max |a_P before| on the identity arm"}
        g["coefficient"] = {"value": max(c["c"]), "tol": IL.C_TOL, "pass": max(c["c"]) <= IL.C_TOL,
                            "failing_input": {"input": "c from the wrong layer's columns (Rows.loads shift 1)", "value": float(np.nanmin(c["c_fail"]))},
                            "failing_input_failed": float(np.nanmin(c["c_fail"])) > IL.C_TOL}
        g["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0}
        g["samples"] = {"lines": self.n_samples, "want": N_SAMPLE_LINES, "pass": self.n_samples == N_SAMPLE_LINES, "path": f"samples/e1b_{self.mk}.jsonl"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.write()
        assert g["identity"]["pass"] and g["identity"]["failing_input_failed"], f"identity: {g['identity']}"
        assert g["resample_known"]["pass"] and g["resample_known"]["failing_input_failed"], f"resample known: {g['resample_known']}"
        assert g["coefficient"]["failing_input_failed"], f"the coefficient check's failing input passed: {g['coefficient']}"

    def run(self):
        self.known()
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        S = {s["pole"]: s for s in self.items["singles"]}
        done = set()
        for s in self.items["singles"]:
            if s["pole"] not in done:
                self.single_pair(s, S[s["partner"]], sample=True)
                done |= {s["pole"], s["partner"]}
        self.gates_singles()
        self.m["timing_s"]["singles"] = round(time.time() - t0, 1)
        for i, p in enumerate(self.items["planted"]):
            self.pair(p, sample=i < N_SAMPLE_PAIRS)
            self.m["n_done"]["pairs"] += 1
            self.write()
        self.gates_pairs()
        self.null_report()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.summarize()
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True, choices=sorted(REPOS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    from infra.aws_pg.jobs.neurons_harvest import fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    from infra.pg.matrix_models import KNOWN, vocab_sha256
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, "instruments_e1b_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())["models"][a.model_key]
    repo = REPOS[a.model_key]
    assert items["repo"] == repo and KNOWN[repo]["n_layers"] == items["n_layers"], (items["repo"], repo)
    t0 = time.time()
    local, rev = NC.local_snapshot(repo)
    assert rev == items["revision"] == KNOWN[repo]["sha"], f"{repo}: cached snapshot {rev}, bank {items['revision']}, KNOWN {KNOWN[repo]['sha']}"
    corpus = NC.load_tok(fetch_corpus(a.corpus, items["tokname"]), items["tokname"])
    assert corpus["tokens_sha256"] == items["tokens_sha256"], "the corpus is not the harvest bank's"
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, items["nh_key"])
    model.requires_grad_(False)
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "the model's vocabulary differs from the corpus tokenizer's"
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"snapshot": local, "corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "device_name": torch.cuda.get_device_name(0),
                  "transformers": __import__("transformers").__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
    job.m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
    job.write()
    print("INSTRUMENTS_E1B_OK", json.dumps({"model_key": a.model_key, "repo": repo}), flush=True)


if __name__ == "__main__":
    main()
