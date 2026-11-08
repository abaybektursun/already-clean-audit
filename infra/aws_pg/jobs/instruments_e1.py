"""INSTRUMENTS E1 job, one O7 load at T_15 (studies/instruments/SPEC_instruments.md §3.2-§3.3, §3.7; amendment 7). A subclass of
jobs.reuse_birth.Job: the model load, the corpus, the known number and completeness (jobs.reuse_ablate.Job.known), ā on UNITE's e1 positions and B_t
(jobs.reuse_birth.Job.background), the captures, the planted sides and the 12-context caps are SPEC4's code. reuse_ablate, reuse_birth,
reuse_generic and unite_route are imported unchanged; infra/pg/instruments_values.py adds the median and resample values, instruments_linear.py c.
The planted gate is amendment 20's (SPEC "After freeze 0", item 20), which replaced §3.3's 100 ā gate before freeze 1a.

    python -m infra.aws_pg.jobs.instruments_e1 --stage e1 --t T_15 --revision stage1-step1413814 --items <s3 e1_items.json> --corpus <s3> --out <dir>

Values (instruments_values): mean ā, zero, median, res0..res3 (one seeded table per draw), from one capture X [200, 31, W] of the background
positions whose mean is asserted equal to ā to 1e-5 (values.npz banks ā, the median, the draw indices and tables). m 50, direct ranking. Sets:
  pairs (56, both sides; side J = SPEC4's side 1): own D(side), cross D(other side), random (reuse_ablate.random_matched to the cross set outside
      own and cross) under u = unexcluded and e = B_t excluded (reuse_birth.top_m_excluded, random_matched_excluded); members = the B_t members of
      the unexcluded cross set (reuse_generic.members_alone; read only when it is not empty); every set under the 7 values. Side J also reads:
      own_inv and cross_inv (ranked among B_t members only, min(50, members upstream); the known number's failing input, mean value; read when
      B_t holds a member below the pole), resconst_abar and resconst_2abar (own_u under the resample code with X replaced by the constant ā and 2ā;
      the resample known number and its failing input), planted_dose1 and planted (the one member of the unexcluded cross set with the largest
      |ā|, reuse_generic.single_largest, written alone at ā and at 10 ā; on a side whose cross set holds a B_t member this is that set's
      largest-|ā| B_t member, asserted), planted_same_layer_dose1 and _dose10 (the same two values written into neuron (L, (a + 1) mod W) of the
      pole's own layer, skipping the pole's terms: the planted gate's failing input, whose activation does not enter a_P), identity_fail (own_u's
      first member at ā: the identity gate's failing input).
  cases (150, read on J): own, cross, random under u and e, every set under the 7 values.
  Every side: a_P before and the identity arm (hooks below the pole, nothing written). Readouts of an identical (side, set, value) are read once.
Order: pass 1 reads every pair's side J under mean (the six u and e sets), so the known number and the room gate are written to metrics.json before
anything else; pass 2 reads the rest of each pair; then the cases. metrics.json is rewritten after every item; npz banks after every item.
Bank: e1/pairs/<unit>.npz and e1/cases/<job>.npz with, per side s: s_tok, s_a_before, s_arms (names "<set>__<value>", "identity", extras),
s_a_after [arms, N], s_U_a and s_U_c [N, U] (a_k and c_k at the marked token for every member of every set in play, U_layer/U_neuron); per set x:
set_<x>_layer, _neuron, _abar, _median, _draws [4, n], _in_B. values.npz: abar, median, draw_idx, draws. samples/e1.jsonl: the first 3 pairs,
side J, values mean, zero, median, resample x constructions u, e x arms own, cross: contexts with the marked token, members (layer, neuron, ā,
written value), a_P before and after, falls.
Gates in metrics.json as they become readable (the verdict decides; the job asserts only instrument checks): known_number (N(mean, u), N(mean, e)
against 0.163 and 0.065 after pass 1, I_e(mean) against 0.294 after the cases, tolerance 0.02; failing input N with the inverted mask), room
(G(mean) >= 0.05), identity (<= 1e-6 relative on every side; failing input identity_fail; asserted), resample_known (<= 1e-5 relative; failing
input 2ā; asserted), planted (amendment 20: on the side-J sides whose cross set holds a B_t member, the share with |Δa_P(dose 10)| >
|Δa_P(dose 1)| > 10 x the identity tolerance is at least 0.9, |Δa_P| the mean over the side's contexts of |a_P after - a_P before| / max |a_P
before|; sides without such a member are counted and left out; failing input the same-layer writes, Δ = 0 exactly), capture (asserted),
coefficient (A = a ⊙ c to 1e-5 at every item, asserted in instruments_linear), exclusion (asserted in reuse_birth), samples.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_birth as JB
from infra.pg import instruments_linear as IL
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import reuse_generic as RG
from infra.pg import unite_route as UR

M = RB.M
CAP = JB.CAP
VALUES = IV.VALUES
DRAWS = IV.DRAWS
KNOWN = {"N_mean_unexcluded": 0.163, "N_mean_excluded": 0.065, "I_excluded_mean": 0.294}
KNOWN_TOL = 0.02
ROOM_LINE = 0.05
ID_TOL = 1e-6
RES_TOL = 1e-5
DOSE = 10.0
PLANTED_SHARE = 0.9
W_TER = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
ARMS = ("own", "cross", "random")
CONS = ("u", "e")
PAIR_SETS = tuple(f"{x}_{e}" for e in CONS for x in ARMS) + ("members",)
CASE_SETS = tuple(f"{x}_{e}" for e in CONS for x in ARMS)
SUMMARY_VALUES = ("mean", "zero", "median", "res_mean", "res_median")
N_SAMPLE_PAIRS = 3
SAMPLE_VALUES = ("mean", "zero", "median", "resample")


# ------------------------------------------------------------------------------------------ pure functions (the dry run and the verdict check against them)
def null_level(I: dict, ter: dict, w_ter: dict = W_TER) -> dict:
    """SPEC4's null at one load (verdict2.planted_ref's point with one load): sum over terciles of w_t x the inverted-CDF median of the finite I in
    the tercile, the weights renormalised over the terciles that hold a finite I."""
    meds = {}
    for t in sorted(w_ter):
        v = np.sort([x for u, x in I.items() if ter[u] == t and np.isfinite(x)])
        if v.size:
            meds[t] = float(v[int(np.ceil(0.5 * v.size)) - 1])
    w = sum(w_ter[t] for t in meds)
    return {"value": float(sum(w_ter[t] * m for t, m in meds.items()) / w) if meds else float("nan"), "by_tercile": {str(t): m for t, m in meds.items()},
            "n": int(sum(np.isfinite(x) for x in I.values()))}


def members_only(score: torch.Tensor, m: int, W: int, mask: torch.Tensor) -> list:
    """The inverted mask: the min(m, members upstream) B_t members with the largest score; empty when B_t holds none below the pole."""
    L = score.numel() // W
    mk = mask[:L].reshape(-1).to(score.device)
    n = int(mk.sum())
    if n == 0:
        return []
    s = score.clone()
    s[~mk] = -float("inf")
    D = RA.top_m(s, min(m, n), W)
    assert RB.in_B(D, mask).all(), "an inverted set holds a neuron outside B_t"
    return D


def side_summary(before, arms, after) -> dict:
    """Per arm the job fall (mean over contexts with a_P > 0, median, n_pos); per set the resample falls per draw, their mean and median."""
    names = list(arms)
    falls = {n: RA.job_fall(before, after[i]) for i, n in enumerate(names)}
    res = {}
    for n in names:
        if n.endswith("__res0"):
            x = n[:-len("__res0")]
            per = [falls[f"{x}__{d}"]["mean"] for d in DRAWS]
            res[x] = {"per_draw": per, "mean": float(np.mean(per)), "median": float(np.median(per))}
    return {"falls": {n: {k: f[k] for k in ("mean", "median", "n_pos")} for n, f in falls.items()}, "res": res}


def fall_of(s: dict, x: str, v: str) -> float:
    """The fall of set x under summary value v (res_mean, res_median: over the four draws)."""
    if v in ("res_mean", "res_median"):
        return s["res"][x]["mean" if v == "res_mean" else "median"] if x in s["res"] else float("nan")
    return s["falls"][f"{x}__{v}"]["mean"] if f"{x}__{v}" in s["falls"] else float("nan")


def ratios(s: dict) -> dict:
    """I = cross / own per (summary value, construction)."""
    return {f"{v}__{e}": JR.ratio(fall_of(s, f"cross_{e}", v), fall_of(s, f"own_{e}", v)) for v in SUMMARY_VALUES for e in CONS}


def record_of(z: dict, sides) -> dict:
    """The metrics record of one banked item from its npz arrays."""
    out = {}
    for s in sides:
        sm = side_summary(z[f"{s}_a_before"], [str(x) for x in z[f"{s}_arms"]], z[f"{s}_a_after"])
        out[s] = {**sm, "I": ratios(sm), "n": int(len(z[f"{s}_tok"]))}
    return out


def case_cancel(sm: dict) -> dict:
    """SPEC4's cancel rule per (summary value, construction) on a case's side summary: own < 2 |random| or under 3 contexts with a_P > 0."""
    return {f"{v}__{e}": RA.cancel(fall_of(sm, f"own_{e}", v), fall_of(sm, f"random_{e}", v), sm["falls"][f"own_{e}__mean"]["n_pos"])
            for v in SUMMARY_VALUES for e in CONS}


def same_layer_neuron(unit: dict, W: int) -> int:
    """The planted gate's failing-input neuron in the pole's own layer: (the first term's index + 1) mod W, stepping past the pole's terms."""
    terms = {t["n"] for t in unit["terms"]}
    k = (unit["terms"][0]["n"] + 1) % W
    while k in terms:
        k = (k + 1) % W
    return k


def planted_response(b, aft) -> float:
    """|Δa_P| on one side: the mean over contexts of |a_P after - a_P before|, in units of the side's max |a_P before| (the identity gate's units)."""
    return float(np.abs(aft - b).mean() / np.abs(b).max())


def planted_side(d1: float, d10: float) -> bool:
    """Amendment 20 on one side: |Δa_P(dose 10)| > |Δa_P(dose 1)| > 10 x the identity tolerance."""
    return bool(d10 > d1 > 10 * ID_TOL)


@contextlib.contextmanager
def same_layer(layers, L: int, k: int, val: torch.Tensor, start: int):
    """The planted gate's failing input: val written into neuron k of layer L's down_proj input at positions >= start (a pre-hook), and a forward
    hook asserting down_proj received it and counting the calls; a_P is read from layer L's gate and up projections, before this write."""
    rec = {"calls": 0}

    def pre(m, args):
        a = args[0].clone()
        a[:, start:, k] = val.to(a.dtype)
        return (a, *args[1:])

    def post(m, args, out):
        assert bool((args[0][:, start:, k] == val.to(args[0].dtype)).all()), "down_proj did not receive the same-layer write"
        rec["calls"] += 1
    hs = [layers[L].mlp.down_proj.register_forward_pre_hook(pre), layers[L].mlp.down_proj.register_forward_hook(post)]
    yield rec
    for h in hs:
        h.remove()


class Job(JB.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "e1/pairs", "e1/cases", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert a.stage == "e1" and a.t == "T_15", (a.stage, a.t)
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.start = RA.window_start(self.bos)
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "status": "running", "m": M, "cap": CAP, "values": list(VALUES), "seeds": list(IV.SEEDS),
                  "job_fall_rule": "mean over contexts with a_P > 0", "replacement": "the value written into down_proj's input at every window position; BOS left",
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "readouts": 0, "readouts_cached": 0,
                  "n_to_read": {"pairs_pass1": len(items["planted"]), "pairs_pass2": len(items["planted"]), "cases": len(items["primary"])},
                  "n_done": {"pairs_pass1": 0, "pairs_pass2": 0, "cases": 0}, "gates": {}, "pairs": {}, "cases": {}}
        self.state = {}
        self.chk = {"id": [], "id_fail": [], "res": [], "res_fail": [], "planted": [], "c": [], "c_fail": []}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0}
        self.n_samples = 0
        self.write()

    # -------------------------------------------------------------------------------------- ā, B_t, the capture and the values
    def background(self):
        super().background()
        t0 = time.time()
        n_up = len(self.layers) - 1
        pos = UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)["e1"]
        X = IV.capture_background(self.model, pos, n_up, **self.kw(RA.LENGTH))
        self.m["captures"] += len(pos)
        self.m["tokens_forwarded"] += RA.n_forwarded(pos, RA.LENGTH, self.bos, self.hw)
        chk = IV.mean_check(X, self.abar)
        self.tab, self.draw_idx = IV.tables(X, self.abar)
        P = X.shape[0]
        del X
        self.tab["const_abar"] = IV.constant_tables(self.abar, P, self.draw_idx)[0]
        self.tab["const_2abar"] = IV.constant_tables(2 * self.abar, P, self.draw_idx)[0]
        self.tab["dose"] = DOSE * self.abar.float()
        ours = {(int(l), int(k)) for l, k in torch.nonzero(self.mask).tolist()}
        theirs = {(int(l), int(k)) for l, k in self.items["spec4_B_t"]}
        self.m["gates"]["capture"] = {**chk, "identity": {"value": 0.0, "reason": "the gate compares one capture's mean with ā"}}
        self.m["B_t_vs_spec4"] = {"jaccard": len(ours & theirs) / max(len(ours | theirs), 1), "n_ours": len(ours), "n_spec4": len(theirs),
                                  "n_ours_not_spec4": len(ours - theirs)}
        np.savez_compressed(self.out / "values.npz", abar=self.abar.cpu().numpy(), median=self.tab["median"].cpu().numpy(),
                            draw_idx=self.draw_idx.astype(np.uint8 if P <= 256 else np.int64), draws=torch.stack([self.tab[d] for d in DRAWS]).cpu().numpy(),
                            seeds=np.array(IV.SEEDS), positions=np.asarray(pos))
        self.m["timing_s"]["values"] = round(time.time() - t0, 1)
        self.write()
        assert chk["pass"] and chk["failing_input_failed"], f"the capture's mean is not ā, or its failing input passed: {chk}"

    # -------------------------------------------------------------------------------------- readouts
    def read(self, st: dict, side: str, arm: str, S: list, value: str, same: int | None = None):
        """One readout of `arm` on `side` (cached on side, set and value); a_P after into st["after"][side][arm]. same: the value of S's one member
        is written into neuron `same` of the pole's own layer instead (the planted gate's failing input)."""
        key = (side, tuple(sorted(S)), value, same, arm == "identity")
        if key in st["cache"]:
            st["after"][side][arm] = st["cache"][key]
            self.m["readouts_cached"] += 1
            return
        unit, toks = st["unit"], st["tok"][side]
        L = unit["layer"]
        kw = self.kw(RA.LENGTH)
        if arm == "identity":
            a = RA.readout(self.model, toks, unit, IV.written([], "mean", self.tab, L, self.device, identity=True), **kw)["a"]
        elif same is not None:
            (l, k), = S
            with same_layer(self.layers, L, same, self.tab[value][l, k].float(), self.start) as rec:
                a = RA.readout(self.model, toks, unit, None, **kw)["a"]
            assert rec["calls"] == -(-len(toks) // kw["batch"]), f"{rec['calls']} same-layer writes for {len(toks)} contexts"
        else:
            a = RA.readout(self.model, toks, unit, IV.written(S, value, self.tab, L, self.device), **kw)["a"]
        st["cache"][key] = st["after"][side][arm] = a
        self.m["readouts"] += 1
        self.m["forwards"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)

    def before(self, st: dict, side: str):
        st["before"][side] = RA.readout(self.model, st["tok"][side], st["unit"], None, **self.kw(RA.LENGTH))["a"]
        self.m["readouts"] += 1
        self.m["forwards"] += len(st["tok"][side])
        self.m["tokens_forwarded"] += RA.n_forwarded(st["tok"][side], RA.LENGTH, self.bos, self.hw)

    def excluded(self, sets: list):
        for S in sets:
            n = int(RB.in_B(S, self.mask).sum()) if S else 0
            self.excl["n_sets"] += 1
            self.excl["n_members"] += len(S)
            self.excl["n_in_B"] += n
            assert n == 0, "an excluded set holds a B_t neuron"

    def new_state(self, unit, tok: dict, cap, A, c, sets: dict, sides) -> dict:
        L = unit["layer"]
        union = sorted(set().union(*[set(S) for S in sets.values()]))
        flat = torch.tensor([l * self.W + k for l, k in union], device=A.device, dtype=torch.long)
        a_all = torch.cat([cap["a"][l] for l in range(L)], 1)
        Ua, Uc = a_all[:, flat].cpu().numpy(), c[:, flat].cpu().numpy()
        rows = {"J": slice(0, len(tok["J"])), "O": slice(len(tok["J"]), len(tok["J"]) + len(tok.get("O", [])))}
        return {"unit": unit, "tok": tok, "sets": sets, "sides": sides, "union": union, "U": {s: (Ua[rows[s]], Uc[rows[s]]) for s in sides},
                "before": {}, "after": {s: {} for s in sides}, "cache": {}, "t0": time.time()}

    def bank(self, st: dict, path: pathlib.Path, meta: dict) -> dict:
        ab = self.abar.cpu().numpy()
        md = self.tab["median"].cpu().numpy()
        dr = torch.stack([self.tab[d] for d in DRAWS]).cpu().numpy()
        mk = self.mask.cpu().numpy()
        z = {**meta, "set_names": np.array(sorted(st["sets"])), "U_layer": np.array([l for l, _ in st["union"]], np.int16),
             "U_neuron": np.array([k for _, k in st["union"]], np.int32)}
        for x, S in st["sets"].items():
            li, ki = np.array([l for l, _ in S], np.int64), np.array([k for _, k in S], np.int64)
            z.update({f"set_{x}_layer": li.astype(np.int16), f"set_{x}_neuron": ki.astype(np.int32), f"set_{x}_abar": ab[li, ki], f"set_{x}_median": md[li, ki],
                      f"set_{x}_draws": dr[:, li, ki], f"set_{x}_in_B": mk[li, ki] if len(S) else np.zeros(0, bool)})
        for s in st["sides"]:
            names = list(st["after"][s])
            z.update({f"{s}_tok": np.asarray(st["tok"][s], np.int64), f"{s}_a_before": st["before"][s], f"{s}_arms": np.array(names),
                      f"{s}_a_after": np.stack([st["after"][s][n] for n in names]).astype(np.float32), f"{s}_U_a": st["U"][s][0], f"{s}_U_c": st["U"][s][1]})
        np.savez_compressed(path, **z)
        return z

    # -------------------------------------------------------------------------------------- pairs
    def pair_pass1(self, p: dict):
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        c, crec = IL.coefficients(self.rows, cap, unit, self.W, A)
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
                "J_own_inv": members_only(sJ, M, W, self.mask), "J_cross_inv": members_only(sO, M, W, self.mask)}
        self.excluded([sets[x] for x in ("J_own_e", "J_cross_e", "J_random_e", "O_random_e")])
        assert not sets["J_members"] or sets["J_single"] == sets["J_members"][:1], "the planted member is not the cross set's largest-|ā| B_t member"
        st = self.new_state(unit, {"J": Jt, "O": Ot}, cap, A, c, sets, ("J", "O"))
        st["c_rec"] = crec
        st["same"] = same_layer_neuron(unit, W)
        self.chk["c"].append(crec["value"])
        self.chk["c_fail"].append(crec["failing_input"])
        self.before(st, "J")
        self.read(st, "J", "identity", [], "mean")
        for x in CASE_SETS:
            self.read(st, "J", f"{x}__mean", sets[f"J_{x}"], "mean")
        self.state[p["unit"]] = st
        self.finish_pair(p, st, ("J",), npass=1)

    def pair_pass2(self, p: dict, sample: bool):
        st = self.state[p["unit"]]
        S = st["sets"]
        for x in CASE_SETS:
            for v in VALUES[1:]:
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
        z = self.finish_pair(p, st, ("J", "O"), npass=2)
        self.gate_rows(p["unit"], z, ("J", "O"))
        if sample:
            self.samples(p, st)
        del self.state[p["unit"]]

    def finish_pair(self, p: dict, st: dict, sides, npass: int) -> dict:
        st["sides"] = sides
        z = self.bank(st, self.out / "e1" / "pairs" / f"{p['unit']}.npz", {"unit": p["unit"], "layer": p["layer"], "tercile": p["tercile"], "kind": "pair",
                                                                           "npass": npass, "same_layer_neuron": st["same"]})
        S = st["sets"]
        self.m["pairs"][p["unit"]] = {"layer": p["layer"], "tercile": p["tercile"], "a": p["a"], "b": p["b"], "lam": p["lam"], "pass": npass,
                                      "npz": f"e1/pairs/{p['unit']}.npz", "sides": record_of(z, sides), "c_check": st["c_rec"],
                                      "set_sizes": {x: len(v) for x, v in S.items()}, "n_in_B": {x: int(RB.in_B(v, self.mask).sum()) if v else 0 for x, v in S.items()},
                                      "single": [[int(l), int(k), float(self.abar[l, k])] for l, k in S["J_single"]],
                                      "same_layer_neuron": [p["layer"], st["same"]],
                                      "seconds": round(time.time() - st["t0"], 1)}
        return z

    def gate_rows(self, key: str, z: dict, sides):
        """Identity and resample-known changes relative to the side's max |a_P before|; the planted responses (amendment 20) on side J."""
        for s in sides:
            names = [str(x) for x in z[f"{s}_arms"]]
            b, aft = z[f"{s}_a_before"].astype(np.float64), z[f"{s}_a_after"].astype(np.float64)
            scale = np.abs(b).max()
            d = lambda n, ref=b: float(np.abs(aft[names.index(n)] - ref).max() / scale)
            self.chk["id"].append(d("identity"))
            if s == "J" and "identity_fail" in names:
                self.chk["id_fail"].append(d("identity_fail"))
                ref = aft[names.index("own_u__mean")]
                self.chk["res"].append(d("resconst_abar", ref))
                self.chk["res_fail"].append(d("resconst_2abar", ref))
                r = {n: planted_response(b, aft[names.index(n)]) for n in ("planted_dose1", "planted", "planted_same_layer_dose1", "planted_same_layer_dose10")}
                self.chk["planted"].append({"unit": key, "member": len(z["set_J_members_layer"]) > 0, **r,
                                            "max_abs_delta_same_layer": float(max(np.abs(aft[names.index(n)] - b).max()
                                                                                  for n in ("planted_same_layer_dose1", "planted_same_layer_dose10")))})

    def samples(self, p: dict, st: dict):
        pr = self.prompts(st["tok"]["J"])
        b = st["before"]["J"]
        tab = {v: self.tab[v].cpu().numpy() for v in ("mean", "median") + DRAWS}
        with open(self.out / "samples" / "e1.jsonl", "a") as fh:
            for v in SAMPLE_VALUES:
                for e in CONS:
                    for x in ("own", "cross"):
                        S = st["sets"][f"J_{x}_{e}"]
                        vals = DRAWS if v == "resample" else (v,)
                        aft = [st["after"]["J"][f"{x}_{e}__{w}"] for w in vals]
                        wv = lambda l, k: [0.0 if w == "zero" else float(tab[w][l, k]) for w in vals]
                        ctx = [{"tok": int(t), **pr[i], "a_before": float(b[i]), "a_after": [float(a[i]) for a in aft],
                                "fall": [float((b[i] - a[i]) / b[i]) for a in aft]} for i, t in enumerate(st["tok"]["J"])]
                        fl = [RA.job_fall(b, a)["mean"] for a in aft]
                        fh.write(json.dumps({"pair": p["unit"], "layer": p["layer"], "side": "J", "value": v, "construction": e, "arm": x,
                                             "draws": list(vals), "members": [[int(l), int(k), float(tab["mean"][l, k]), wv(l, k)] for l, k in S],
                                             "job_fall": fl, "job_fall_mean_over_draws": float(np.mean(fl)), "job_fall_median_over_draws": float(np.median(fl)),
                                             "contexts": ctx}, ensure_ascii=False) + "\n")
                        self.n_samples += 1

    # -------------------------------------------------------------------------------------- cases
    def case(self, j: dict):
        unit = UR.pole_unit(j["pole"])
        assert unit["layer"] == j["layer"], (j["job"], unit["layer"], j["layer"])
        Jt, Ot = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"]))
        capJ, AJ = self.side(Jt, unit)
        _, AO = self.side(Ot, unit)
        c, crec = IL.coefficients(self.rows, capJ, unit, self.W, AJ)
        self.chk["c"].append(crec["value"])
        self.chk["c_fail"].append(crec["failing_input"])
        sJ, sO = AJ.sum(0), AO.sum(0)
        Ju, Ou = RA.top_m(sJ, M, self.W), RA.top_m(sO, M, self.W)
        Je, Oe = self.rank(sJ), self.rank(sO)
        sets = {"J_own_u": Ju, "J_cross_u": Ou, "J_random_u": RA.random_matched(Ou, Ju, self.abar),
                "J_own_e": Je, "J_cross_e": Oe, "J_random_e": RB.random_matched_excluded(Oe, Je, self.abar, self.mask)}
        self.excluded([sets["J_random_e"]])
        st = self.new_state(unit, {"J": Jt}, capJ, AJ, c, sets, ("J",))
        self.before(st, "J")
        self.read(st, "J", "identity", [], "mean")
        for x in CASE_SETS:
            for v in VALUES:
                self.read(st, "J", f"{x}__{v}", sets[f"J_{x}"], v)
        key = f"{j['pole']}__c{j['c']}"
        z = self.bank(st, self.out / "e1" / "cases" / f"{key}.npz", {"unit": key, "job": j["job"], "pole": j["pole"], "layer": j["layer"], "kind": "case"})
        rec = record_of(z, ("J",))
        sm = rec["J"]
        self.m["cases"][j["job"]] = {"pole": j["pole"], "layer": j["layer"], "tercile": j["tercile"], "npz": f"e1/cases/{key}.npz", "sides": rec,
                                     "cancel": case_cancel(sm),
                                     "c_check": crec, "set_sizes": {x: len(v) for x, v in sets.items()}, "seconds": round(time.time() - st["t0"], 1)}
        self.gate_rows(j["job"], z, ("J",))

    # -------------------------------------------------------------------------------------- gates as they become readable
    def known_pairs(self):
        P = self.m["pairs"]
        ter = {u: v["tercile"] for u, v in P.items()}
        I = {e: {u: v["sides"]["J"]["I"][f"mean__{e}"] for u, v in P.items()} for e in CONS}
        Nu, Ne = null_level(I["u"], ter), null_level(I["e"], ter)
        g = self.m["gates"]
        g["known_number"] = {"pairs": {"N_mean_unexcluded": Nu, "N_mean_excluded": Ne, "pass": bool(abs(Nu["value"] - KNOWN["N_mean_unexcluded"]) <= KNOWN_TOL
                                                                                                  and abs(Ne["value"] - KNOWN["N_mean_excluded"]) <= KNOWN_TOL)},
                             "targets": KNOWN, "tol": KNOWN_TOL, "rule": "SPEC4's null on side J (cross D(O) / own D(J)), terciles 1/3 each; the verdict decides"}
        g["room"] = {"value": Nu["value"] - Ne["value"], "line": ROOM_LINE, "pass": bool(Nu["value"] - Ne["value"] >= ROOM_LINE), "failing_input": None,
                     "identity": None, "rule": "G(mean) = N(mean, u) - N(mean, e) >= 0.05"}

    def known_inverted(self):
        P = self.m["pairs"]
        I = {u: v["sides"]["J"]["falls"] for u, v in P.items()}
        Iinv = {u: JR.ratio(f["cross_inv__mean"]["mean"], f["own_inv__mean"]["mean"]) if "own_inv__mean" in f else float("nan") for u, f in I.items()}
        Ni = null_level(Iinv, {u: v["tercile"] for u, v in P.items()})
        kn = self.m["gates"]["known_number"]
        kn["failing_input"] = {"input": "N(mean, e) with the exclusion mask inverted (sets ranked among B_t members only)", "N": Ni,
                               "failed": not abs(Ni["value"] - KNOWN["N_mean_excluded"]) <= KNOWN_TOL}

    def known_cases(self):
        C = self.m["cases"]
        I = [v["sides"]["J"]["I"]["mean__e"] for v in C.values() if not v["cancel"]["mean__e"]["cancelled"]]
        val = float(np.median(I)) if I else float("nan")
        kn = self.m["gates"]["known_number"]
        kn["cases"] = {"I_excluded_mean": val, "n_live": len(I), "n": len(C), "pass": bool(abs(val - KNOWN["I_excluded_mean"]) <= KNOWN_TOL)}
        kn["pass"] = bool(kn["pairs"]["pass"] and kn["cases"]["pass"])

    def gates_pairs(self):
        c = self.chk
        g = self.m["gates"]
        nanmed = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")
        g["resample_known"] = {"value": max(c["res"]), "tol": RES_TOL, "pass": max(c["res"]) <= RES_TOL,
                               "failing_input": {"input": "the constant vector 2ā in place of ā", "value": max(c["res_fail"])},
                               "failing_input_failed": max(c["res_fail"]) > RES_TOL, "identity": {"value": max(c["res"]), "reason": "the gate is an identity check"},
                               "rule": "max over pairs and contexts of |a_P(resample code, constant ā) - a_P(mean)| / max |a_P before| on own_u, side J"}
        rows = [r for r in c["planted"] if r["member"]]
        share = lambda d1, d10: float(np.mean([planted_side(r[d1], r[d10]) for r in rows])) if rows else float("nan")
        v, vf = share("planted_dose1", "planted"), share("planted_same_layer_dose1", "planted_same_layer_dose10")
        g["planted"] = {"value": v, "line": PLANTED_SHARE, "pass": bool(v >= PLANTED_SHARE), "n_sides": len(rows), "n_sides_excluded": len(c["planted"]) - len(rows),
                        "tol_line": 10 * ID_TOL, "dose1_median": nanmed([r["planted_dose1"] for r in rows]), "dose10_median": nanmed([r["planted"] for r in rows]),
                        "failing_input": {"input": "the same two writes into neuron (L, (a + 1) mod W) of the pole's own layer, whose activation does not enter a_P",
                                          "value": vf, "max_abs_delta": max([r["max_abs_delta_same_layer"] for r in rows], default=float("nan"))},
                        "failing_input_failed": not vf >= PLANTED_SHARE, "identity": None, "sides": c["planted"],
                        "rule": "amendment 20: on side J of every pair whose unexcluded cross set holds a B_t member, its largest-|ā| member written alone at ā "
                                "(dose 1) and 10 ā (dose 10); share of such sides with |Δa_P(10)| > |Δa_P(1)| > 10 x 1e-6 >= 0.9; |Δa_P| = mean over the "
                                "side's contexts of |a_P after - a_P before| / max |a_P before|; sides without a member counted, not gated"}

    def summarize(self):
        c, g = self.chk, self.m["gates"]
        g["identity"] = {"value": max(c["id"]), "tol": ID_TOL, "pass": max(c["id"]) <= ID_TOL, "n_sides": len(c["id"]),
                         "failing_input": {"input": "own_u's first member written at ā in place of the empty set", "value": max(c["id_fail"])},
                         "failing_input_failed": max(c["id_fail"]) > ID_TOL, "identity": {"value": max(c["id"]), "reason": "read on the identity arm itself"},
                         "rule": "max over sides and contexts of |a_P after - a_P before| / max |a_P before| on the identity arm"}
        g["coefficient"] = {"value": max(c["c"]), "tol": IL.C_TOL, "pass": max(c["c"]) <= IL.C_TOL, "failing_input": {"input": "c with the next layer's scale",
                            "value": float(np.nanmax(c["c_fail"]))}, "failing_input_failed": float(np.nanmax(c["c_fail"])) > IL.C_TOL}
        g["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0}
        want = N_SAMPLE_PAIRS * len(SAMPLE_VALUES) * len(CONS) * 2
        g["samples"] = {"lines": self.n_samples, "want": want, "pass": self.n_samples == want,
                        "path": "samples/e1.jsonl"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.write()
        assert g["identity"]["pass"] and g["identity"]["failing_input_failed"], f"identity: {g['identity']}"
        assert g["resample_known"]["pass"] and g["resample_known"]["failing_input_failed"], f"resample known: {g['resample_known']}"
        assert g["coefficient"]["failing_input_failed"], f"the coefficient check's failing input passed: {g['coefficient']}"

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P, C = self.items["planted"], self.items["primary"]
        for p in P:
            self.pair_pass1(p)
            self.m["n_done"]["pairs_pass1"] += 1
            self.write()
        self.known_pairs()
        self.m["timing_s"]["pairs_pass1"] = round(time.time() - t0, 1)
        self.write()
        for i, p in enumerate(P):
            self.pair_pass2(p, sample=i < N_SAMPLE_PAIRS)
            self.m["n_done"]["pairs_pass2"] += 1
            self.write()
        self.known_inverted()
        self.gates_pairs()
        self.m["timing_s"]["pairs"] = round(time.time() - t0, 1)
        self.write()
        for j in C:
            self.case(j)
            self.m["n_done"]["cases"] += 1
            self.write()
        self.known_cases()
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
    ap.add_argument("--stage", required=True, choices=["e1"])
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, "instruments_e1_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
    assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha,
                  "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
    job.m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("INSTRUMENTS_E1_OK", json.dumps({"t": a.t}), flush=True)


if __name__ == "__main__":
    main()
