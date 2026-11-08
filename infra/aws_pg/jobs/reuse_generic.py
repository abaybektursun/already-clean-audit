"""SPEC6 GENERIC job, one load at T_15 (studies/reuse/SPEC6_generic.md §2-§5, §9 step 1). A subclass of jobs.reuse_birth.Job: ā, B_t,
high_background.json and the foreign choice are SPEC4's code (JB.Job.background, JB.Job.foreign); the instrument modules reuse_ablate, reuse_birth
and unite_route are imported unchanged; infra/pg/reuse_generic.py adds the held scale, the cuts and the members arms.

    python -m infra.aws_pg.jobs.reuse_generic --stage generic --t T_15 --revision stage1-step1413814 --items <s3 birth_items.json> --corpus <s3> \
        --primary-foreign <s3 primary T_15 foreign.json> --spec4-bank <s3 SPEC4 T_15 step dir> --t14-bank <s3 SPEC5 dating T_14 step dir> --out <dir>

Before any forward: SPEC4's T_15 bank (metrics.json, falls/<case>.npz) and SPEC5's dating T_14 bank (dating/<case>.npz) are fetched, and every
primary case's J contexts are asserted to be the contexts of both (SPEC4 has no T_14 load; SPEC5's dating bank holds a_P at T_14 on the same
contexts, the known number's failing input). Then the parent's known number and completeness, ā over layers 0..30 on UNITE's e1 positions, B_t
(high_background.json) and the cuts g3 (top 3% of |ā|) and pl1 (top 1% within each layer) (cuts.json), Q per case (foreign.json).
Gates read before the units (SPEC6 §5), on the first primary case with L >= 2 and 20 contexts (its J then its O*, first 20): held identity (a
held-scale forward with the empty set equals the clean a_P exactly; failing input: layer l replays layer l-1's inv) and held completeness (the
held norm output at every ablated layer equals weight * y_ablated * inv_clean recomputed by an independent probe, cast as the norm casts, to
1e-4 relative; failing input: the free-scale output; identity value: the same check on the empty set), the set being the case's own set ranked
without exclusion (SPEC2's construction, so the divisor moves). Both assert after their numbers are written.
Units, m 50, direct ranking, mean replacement (ā), every side capped at 12 contexts by rank, each read set: one clean forward that records the
held scale at every layer below the pole, then one forward per (arm, scale); held holds the arm's ablated layers (the identity arm: every layer
below the pole). One npz per (unit, arm, exclusion, scale): falls/<unit>__<arm>__<exclusion>__<scale>.npz, written as soon as it is read.
  primary cases (150): identity (free, held); own', cross', random', foreign' ranked and matched with B_t excluded (g1) x (free, held); read on J.
      prebirth/<case>__J.npz and __O.npz: SPEC4's Part A arrays read at T_15 (JB.Job.bank_part_a, read "generic_t15"; loads on the J side only).
  primary ceiling splits (129 distinct), read on side 2: identity (free, held); own' D'(side 2), cross' D'(side 1) under g1 x (free, held).
  planted pairs (56), read on side 1: identity (free, held); own D(side 1) and cross D(side 2) under none, g1, g3, pl1 x (free, held); the
      members-alone arm (the B_t members of the unexcluded cross set) and the single-largest-member arm (its largest |ā| member) x (free, held).
npz fields: unit, pole, arm, exclusion, scale, tok, a_before, g_before, u_before (the clean readout), a_after, g_after, u_after, set_layer,
set_neuron, set_r (ā), set_in_B (B_t), set_in_cut (the arm's own exclusion), held_layers, n_held_calls.
metrics.json (rewritten after every unit): per case, split and pair its job falls per "<arm>__<exclusion>__<scale>" with the npz path; per case
I and cancel per scale, s, the known-number references (SPEC4's a_P before and SPEC5's T_14 a_P on its J contexts, SPEC4's own' and cross'
falls) and the Part A paths; per pair I per "<exclusion>__<scale>", the members and the single member with ā, the B_t count of every set;
gates: held_identity, held_completeness, known_spec4_a (median over cases of unite_stats.known_ratio against SPEC4's T_15 a_P; failing input
SPEC5's T_14 a_P), known_spec4_b (free own' and cross' of the first 20 cases against SPEC4's falls, median absolute error; failing input the
next case's falls), identity (free; failing input own'), exclusion per cut, memory, timing. Statistics over the bank (the nulls, the positive
control, the room, the cancel share, the known null) are the verdict's (studies/reuse/scripts/verdict_generic.py).
samples/: held_gates.json (the 20 contexts' prompts and a_P clean, held empty, shifted, held and free on the set), cases.json (three named cases:
every context's prompt, marked token, clean a_P and a_P under every arm and scale), members_alone.json (per pair the members with ā and their
layer's median |ā|).
"""
from __future__ import annotations

import argparse
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
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import reuse_generic as RG
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US

M = RB.M
CAP = JB.CAP
N_GATE_CTX = 20
N_KNOWN_B = 20
N_SAMPLE_CASES = 3
CASE_ARMS = ("own", "cross", "random", "foreign")
PAIR_ARMS = ("own", "cross")
SPLIT_ARMS = ("own", "cross")


def key_of(j: dict) -> str:
    return f"{j['pole']}__c{j['c']}"


def arm_name(arm: str, excl: str, scale: str) -> str:
    return f"{arm}__{excl}__{scale}"


def bank_arm(out: pathlib.Path, key: str, unit: dict, arm: dict, scale: str, toks, before: dict, after: dict, abar, mask_B, mask_e, held_layers, n_held: int) -> dict:
    """falls/<key>__<arm>__<exclusion>__<scale>.npz and the job fall of that arm."""
    S = arm["set"]
    ns = len(S)
    name = arm_name(arm["arm"], arm["excl"], scale)
    rel = f"falls/{key}__{name}.npz"
    ab = np.asarray(abar.cpu() if torch.is_tensor(abar) else abar, dtype=np.float32)
    np.savez_compressed(out / rel, unit=key, pole=unit["unit"], arm=arm["arm"], exclusion=arm["excl"], scale=scale, tok=np.asarray(toks, np.int64),
                        a_before=before["a"], g_before=before["g"], u_before=before["u"], a_after=after["a"], g_after=after["g"], u_after=after["u"],
                        set_layer=np.array([l for l, _ in S], np.int16), set_neuron=np.array([k for _, k in S], np.int32),
                        set_r=np.array([ab[l, k] for l, k in S], np.float32), set_in_B=RB.in_B(S, mask_B) if ns else np.zeros(0, bool),
                        set_in_cut=RB.in_B(S, mask_e) if ns else np.zeros(0, bool), held_layers=np.array(sorted(held_layers), np.int16), n_held_calls=n_held)
    return {**RA.job_fall(before["a"], after["a"]), "n_set": ns, "n_in_B": int(RB.in_B(S, mask_B).sum()) if ns else 0, "path": rel}


class Job(JB.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str, refs: dict):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "falls", "samples", "prebirth"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert a.stage == "generic" and a.t == "T_15", (a.stage, a.t)
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        for lyr in self.layers:
            RG.ff_norm(lyr)
        self.refs = refs
        self.start = RA.window_start(self.bos)
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is",
                  "held_rule": "post_feedforward_layernorm output at positions >= the window start = weight * y_ablated * inv recorded on the clean forward",
                  "exclusions": {"none": "no exclusion", "g1": "B_t, the 1% largest |ā| (high_background.json)", "g3": "the 3% largest |ā|",
                                 "pl1": "the 1% largest |ā| within each layer"},
                  "job_fall_rule": "mean over contexts with a_P > 0", "m": M, "cap": CAP, "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0,
                  "primary_foreign_sha256": getattr(a, "primary_foreign_sha256", None), "refs": {k: str(v) for k, v in refs.items()},
                  "gates": {}, "jobs": {}, "ceiling": {}, "planted": {}}
        self.id_rows, self.id_fail_rows, self.idh_rows = [], [], []
        self.excl = {e: {"n_sets": 0, "n_members": 0, "n_in_cut": 0} for e in ("g1", "g3", "pl1")}
        self.write()

    # -------------------------------------------------------------------------------------- references and cuts
    def check_refs(self):
        """SPEC4's T_15 bank and SPEC5's T_14 bank hold every primary case on the same J contexts; read before any forward."""
        r = self.refs
        self.spec4 = json.loads(pathlib.Path(r["spec4_metrics"]).read_text())
        assert self.spec4["t"] == "T_15" and self.spec4["status"] == "done" and self.spec4["items_sha256"] == self.a.items_sha256, \
            ("SPEC4's bank is not the T_15 load of these items", self.spec4["t"], self.spec4["status"], self.spec4["items_sha256"])
        self.ref_a = {}
        for j in self.items["primary"]:
            k, Jt = key_of(j), np.asarray(j["J_tok"])[:CAP]
            z4 = np.load(pathlib.Path(r["spec4_falls"]) / f"{k}.npz")
            assert np.array_equal(z4["tok"], Jt), (k, "SPEC4's contexts differ from the items'")
            z14 = np.load(pathlib.Path(r["t14_dating"]) / f"{k}.npz")
            assert str(z14["t"]) == "T_14" and str(z14["pole"]) == j["pole"], (k, str(z14["t"]), "the T_14 bank is not T_14 or not this pole")
            onJ = z14["side"] == "J"
            by = dict(zip(z14["tok"][onJ].tolist(), z14["aP"][onJ].tolist()))
            assert all(int(t) in by for t in Jt), (k, "the T_14 bank lacks a J context")
            f4 = self.spec4["jobs"][j["job"]]["falls"]
            self.ref_a[j["job"]] = {"spec4_a_before": z4["a_before"].astype(np.float64), "t14_a": np.array([by[int(t)] for t in Jt], np.float64),
                                    "spec4_own": float(f4[f"own_direct_m{M}"]["mean"]), "spec4_cross": float(f4[f"cross_direct_m{M}"]["mean"])}
        self.m["refs_checked"] = {"n_cases": len(self.ref_a), "rule": "every primary case's J contexts equal SPEC4's T_15 bank's and are in SPEC5's T_14 bank"}
        self.write()

    def cuts(self):
        n_up = len(self.layers) - 1
        self.cut = RG.masks(self.abar, n_up)
        assert torch.equal(self.cut["g1"]["mask"], self.mask), "g1 is not the B_t of high_background.json"
        rec = {e: {k: c[k] for k in ("count", "per_layer", "frac", "per_layer_rule", "cut", "n_entries", "n_up")} for e, c in self.cut.items()}
        NH.dump(self.out / "cuts.json", rec)
        self.m["cuts"] = rec
        self.write()

    def rank_e(self, score, e: str) -> list:
        return RB.top_m_excluded(score, M, self.W, self.cut[e]["mask"])

    def check_excl(self, e: str, sets: list):
        for S in sets:
            n = int(RB.in_B(S, self.cut[e]["mask"]).sum())
            self.excl[e]["n_sets"] += 1
            self.excl[e]["n_members"] += len(S)
            self.excl[e]["n_in_cut"] += n
            assert n == 0, (e, "a ranked set holds a member of its own cut")

    # -------------------------------------------------------------------------------------- one read set, every arm under both scales
    def count(self, toks):
        self.m["forwards"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)

    def read_set(self, key: str, unit: dict, toks, arms: list) -> tuple:
        """The clean readout (recording the held scale at every layer below the pole), then one readout per arm and scale, each banked at once."""
        L = unit["layer"]
        kw = self.kw(RA.LENGTH)
        toks = np.asarray(toks)
        res, after_by = {}, {}
        with RG.held_scale(self.layers, range(L), self.start) as hs:
            hs.record()
            before = RA.readout(self.model, toks, unit, None, **kw)
            self.count(toks)
            for arm in arms:
                ident = arm["arm"] == "identity"
                abl = RA.ablation_dict(arm["set"], "mean", self.abar, L, self.device, identity=ident)
                for scale in RG.SCALES:
                    if scale == "held":
                        hs.hold(set(abl))
                    else:
                        hs.free()
                    after = RA.readout(self.model, toks, unit, abl, **kw)
                    n_held = hs.done()
                    self.count(toks)
                    name = arm_name(arm["arm"], arm["excl"], scale)
                    held = set(abl) if scale == "held" else set()
                    res[name] = bank_arm(self.out, key, unit, arm, scale, toks, before, after, self.abar, self.mask, self.cut[arm["excl"]]["mask"], held, n_held)
                    after_by[name] = after["a"]
                    if ident and scale == "free":
                        self.id_rows += [{"unit": key, "tok": int(t), "fall": float(x), "change": float(abs(b - c))}
                                         for t, x, b, c in zip(toks, RA.fall(before["a"], after["a"]), before["a"], after["a"])]
                    if ident and scale == "held":
                        self.idh_rows += [{"unit": key, "tok": int(t), "change": float(abs(b - c))} for t, b, c in zip(toks, before["a"], after["a"])]
                    if arm["arm"] == "own" and scale == "free":
                        self.id_fail_rows += [{"unit": key, "tok": int(t), "fall": float(x)} for t, x in zip(toks, RA.fall(before["a"], after["a"]))]
        return before, res, after_by

    # -------------------------------------------------------------------------------------- the held-scale gates, before the units
    def held_gates(self):
        t0 = time.time()
        j = next(j for j in self.items["primary"] if j["layer"] >= 2)
        unit = UR.pole_unit(j["pole"])
        L = unit["layer"]
        Jt = np.asarray(j["J_tok"])[:CAP]
        toks = np.concatenate([Jt, np.asarray(j["O_tok"])[:CAP]])[:N_GATE_CTX]
        _, AJ = self.side(Jt, unit)
        S = RA.top_m(AJ.sum(0), M, self.W)
        abl = RA.ablation_dict(S, "mean", self.abar, L, self.device)
        ident = RA.ablation_dict([], "mean", self.abar, L, self.device, identity=True)
        kw = self.kw(RA.LENGTH)
        with RG.held_scale(self.layers, range(L), self.start) as hs, RG.norm_probe(self.layers, range(L), self.start) as pr:
            hs.record()
            pr.record()
            clean = RA.readout(self.model, toks, unit, None, **kw)["a"]
            hs.hold(set(ident))
            pr.compare(set(ident))
            a_id = RA.readout(self.model, toks, unit, ident, **kw)["a"]
            hs.done()
            c_id = pr.result()
            hs.hold(set(ident), shift=1)
            pr.off()
            a_sh = RA.readout(self.model, toks, unit, ident, **kw)["a"]
            hs.done()
            hs.hold(set(abl))
            pr.compare(set(abl))
            a_held = RA.readout(self.model, toks, unit, abl, **kw)["a"]
            hs.done()
            c_held = pr.result()
            hs.free()
            pr.compare(set(abl))
            a_free = RA.readout(self.model, toks, unit, abl, **kw)["a"]
            c_free = pr.result()
        for _ in range(5):
            self.count(toks)
        d_id = float(np.abs(a_id.astype(np.float64) - clean).max())
        d_sh = float(np.abs(a_sh.astype(np.float64) - clean).max())
        g = self.m["gates"]
        g["held_identity"] = {"value": d_id, "pass": d_id == 0.0, "failing_input": {"input": "layer l replays the inv recorded at layer l-1", "value": d_sh},
                              "failing_input_failed": d_sh > 0.0, "identity": {"value": d_id, "reason": "the gate is read on the empty set: its value is the identity value"},
                              "rule": "max |a_P(held, empty set) - a_P(clean)| over the contexts is 0", "case": j["job"], "n_contexts": int(len(toks)), "layers_held": L}
        g["held_identity"]["ok"] = g["held_identity"]["pass"] and g["held_identity"]["failing_input_failed"]
        g["held_completeness"] = {"value": c_held["max"], "pass": c_held["max"] <= RG.COMPLETE_TOL,
                                  "failing_input": {"input": "the free-scale norm output in place of the held output", "value": c_free["max"]},
                                  "failing_input_failed": c_free["max"] > RG.COMPLETE_TOL, "identity": {"value": c_id["max"], "reason": "the same check on the empty set"},
                                  "tol": RG.COMPLETE_TOL, "by_layer": c_held["by_layer"], "failing_input_by_layer": c_free["by_layer"],
                                  "rule": "max over ablated layers, contexts and positions >= the window start of max|out - weight y_abl inv_clean| / max|weight y_abl inv_clean|",
                                  "case": j["job"], "set": "own D(J), ranked without exclusion", "n_in_B": int(RB.in_B(S, self.mask).sum()),
                                  "fall_free_mean": RA.job_fall(clean, a_free)["mean"], "fall_held_mean": RA.job_fall(clean, a_held)["mean"]}
        g["held_completeness"]["ok"] = g["held_completeness"]["pass"] and g["held_completeness"]["failing_input_failed"]
        pr_ = self.prompts(toks)
        NH.dump(self.out / "samples" / "held_gates.json", {"case": j["job"], "pole": j["pole"], "set": [[int(l), int(k), float(self.abar[l, k])] for l, k in S],
                                                          "contexts": [{"tok": int(t), **p, "a_clean": float(c), "a_held_empty": float(x), "a_shifted": float(s_),
                                                                        "a_held_set": float(h), "a_free_set": float(f)}
                                                                       for t, p, c, x, s_, h, f in zip(toks, pr_, clean, a_id, a_sh, a_held, a_free)]})
        self.m["timing_s"]["held_gates"] = round(time.time() - t0, 1)
        self.write()
        assert g["held_identity"]["ok"], f"held identity failed: {g['held_identity']}"
        assert g["held_completeness"]["ok"], f"held completeness failed: {g['held_completeness']}"

    # -------------------------------------------------------------------------------------- the units
    def case(self, j: dict, q: dict, sample: bool):
        t0 = time.time()
        unit = UR.pole_unit(j["pole"])
        L = unit["layer"]
        assert L == j["layer"], (j["job"], L, j["layer"])
        Jt, Ot = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"]))
        capJ, AJ = self.side(Jt, unit)
        capO, AO = self.side(Ot, unit)
        sJ, sO = AJ.sum(0), AO.sum(0)
        DJ, DO = self.rank(sJ), self.rank(sO)
        R = RB.random_matched_excluded(DO, DJ, self.abar, self.mask)
        qunit = UR.pole_unit(q["pole"])
        assert qunit["layer"] == L, (j["pole"], q["pole"])
        Qt = np.asarray(q["tok"])[:CAP]
        _, AQ = self.side(Qt, qunit)
        DF = self.rank(AQ.sum(0))
        self.check_excl("g1", [DJ, DO, R, DF])
        key = key_of(j)
        arms = [{"arm": "identity", "excl": "none", "set": []}] + [{"arm": x, "excl": "g1", "set": S} for x, S in zip(CASE_ARMS, (DJ, DO, R, DF))]
        before, res, after_by = self.read_set(key, unit, Jt, arms)
        pa = {"J": self.bank_part_a(key, "J", Jt, capJ, AJ, unit, "generic_t15", DJ), "O": self.bank_part_a(key, "O", Ot, capO, AO, unit, "generic_t15", DO)}
        ref = self.ref_a[j["job"]]
        f = lambda x, s: res[arm_name(x, "g1", s)]
        rec = {"pole": j["pole"], "layer": L, "tercile": j["tercile"], "birth": j["birth"], "label": j["label"], "n_J": int(len(Jt)), "n_O": int(len(Ot)),
               "foreign": {"pole": q["pole"], "c": q["c"], "n": int(len(Qt))}, "falls": res,
               "I": {s: JR.ratio(f("cross", s)["mean"], f("own", s)["mean"]) for s in RG.SCALES},
               "cancel": {s: RA.cancel(f("own", s)["mean"], f("random", s)["mean"], f("own", s)["n_pos"]) for s in RG.SCALES},
               "s": len(set(DJ) & set(DO)) / M, "prebirth": pa,
               "known": {"tok": Jt.tolist(), "spec4_a_before": ref["spec4_a_before"].tolist(), "t14_a": ref["t14_a"].tolist(),
                         "ratio_spec4": US.known_ratio(before["a"], ref["spec4_a_before"]), "ratio_t14": US.known_ratio(before["a"], ref["t14_a"]),
                         "spec4_own": ref["spec4_own"], "spec4_cross": ref["spec4_cross"]}}
        if sample:
            pr = self.prompts(Jt)
            cur = json.loads((self.out / "samples" / "cases.json").read_text()) if (self.out / "samples" / "cases.json").exists() else {}
            cur[j["job"]] = {"pole": j["pole"], "layer": L, "contexts": [{"tok": int(t), **pr[i], "a_clean": float(before["a"][i]),
                                                                           **{n: float(v[i]) for n, v in after_by.items()}} for i, t in enumerate(Jt)],
                             "falls": {n: v["mean"] for n, v in res.items()}}
            NH.dump(self.out / "samples" / "cases.json", cur)
        rec["seconds"] = round(time.time() - t0, 1)
        self.m["jobs"][j["job"]] = rec

    def ceiling_generic(self, name: str, pole: str, c: dict):
        """A ceiling split read on side 2: identity, own' D'(side 2), cross' D'(side 1), under g1 x (free, held)."""
        t0 = time.time()
        unit = UR.pole_unit(pole)
        s1, s2 = (np.asarray(x)[:CAP] for x in c["sides_tok"])
        _, A1 = self.side(s1, unit)
        _, A2 = self.side(s2, unit)
        D1, D2 = self.rank(A1.sum(0)), self.rank(A2.sum(0))
        self.check_excl("g1", [D1, D2])
        key = f"{pole}__c{c['c']}_ceiling_side2"
        arms = [{"arm": "identity", "excl": "none", "set": []}, {"arm": "own", "excl": "g1", "set": D2}, {"arm": "cross", "excl": "g1", "set": D1}]
        _, res, _ = self.read_set(key, unit, s2, arms)
        self.m["ceiling"][name] = {"pole": pole, "c": c["c"], "key": key, "n_sides": [int(len(s1)), int(len(s2))], "s": len(set(D1) & set(D2)) / M, "falls": res,
                                   "I_ceil": {s: JR.ratio(res[arm_name("cross", "g1", s)]["mean"], res[arm_name("own", "g1", s)]["mean"]) for s in RG.SCALES},
                                   "seconds": round(time.time() - t0, 1)}

    def pair(self, p: dict):
        """A planted pair read on side 1: own D(side 1), cross D(side 2) under every exclusion, the members-alone and single-member arms, both scales."""
        t0 = time.time()
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        nJ = len(Jt)
        sJ, sO = A[:nJ].sum(0), A[nJ:].sum(0)
        D = {e: (self.rank_e(sJ, e), self.rank_e(sO, e)) for e in RG.EXCLUSIONS}
        for e in ("g1", "g3", "pl1"):
            self.check_excl(e, list(D[e]))
        cross0 = D["none"][1]
        mem = RG.members_alone(cross0, self.mask, self.abar)
        one = RG.single_largest(cross0, self.abar)
        arms = [{"arm": "identity", "excl": "none", "set": []}]
        for e in RG.EXCLUSIONS:
            arms += [{"arm": "own", "excl": e, "set": D[e][0]}, {"arm": "cross", "excl": e, "set": D[e][1]}]
        if mem:
            arms.append({"arm": "members", "excl": "none", "set": mem})
        arms.append({"arm": "single", "excl": "none", "set": one})
        key = f"{p['unit']}__planted"
        _, res, _ = self.read_set(key, unit, Jt, arms)
        I = {f"{e}__{s}": JR.ratio(res[arm_name("cross", e, s)]["mean"], res[arm_name("own", e, s)]["mean"]) for e in RG.EXCLUSIONS for s in RG.SCALES}
        ab = self.abar.cpu().numpy()
        med = np.median(np.abs(ab), 1)
        desc = lambda S: [[int(l), int(k), float(ab[l, k]), float(abs(ab[l, k]) / med[l])] for l, k in S]
        self.m["planted"][p["unit"]] = {"layer": p["layer"], "tercile": US.tercile(p["layer"]), "a": p["a"], "b": p["b"], "lam": p["lam"], "n_J": int(nJ),
                                        "n_O": int(len(Ot)), "falls": res, "I": I, "members": desc(mem), "single": desc(one), "n_members": len(mem),
                                        "members_read": bool(mem), "n_in_B": {f"{x}__{e}": int(RB.in_B(S, self.mask).sum()) for e in RG.EXCLUSIONS for x, S in zip(PAIR_ARMS, D[e])},
                                        "s": {e: len(set(D[e][0]) & set(D[e][1])) / M for e in RG.EXCLUSIONS}, "seconds": round(time.time() - t0, 1)}
        cur = json.loads((self.out / "samples" / "members_alone.json").read_text()) if (self.out / "samples" / "members_alone.json").exists() else {}
        cur[p["unit"]] = {"layer": p["layer"], "members": desc(mem), "single": desc(one), "cross_unexcluded": desc(cross0),
                          "cross_unexcluded_in_B": RB.in_B(cross0, self.mask).tolist(), "columns": ["layer", "neuron", "abar", "abs_abar_over_layer_median"]}
        NH.dump(self.out / "samples" / "members_alone.json", cur)

    # -------------------------------------------------------------------------------------- the load
    def known_spec4(self):
        """known_spec4_a and known_spec4_b over the cases read (SPEC6 §5 a, b)."""
        J = self.m["jobs"]
        ids = [j["job"] for j in self.items["primary"] if j["job"] in J]
        ra = [J[x]["known"]["ratio_spec4"] for x in ids]
        rw = [J[x]["known"]["ratio_t14"] for x in ids]
        va, wa = float(np.median(ra)), float(np.median(rw))
        g = self.m["gates"]
        g["known_spec4_a"] = {"value": va, "pass": va <= US.KNOWN_TOL, "failing_input": {"input": "SPEC5's dating a_P at T_14 on the same contexts in place of SPEC4's T_15 bank", "value": wa},
                              "failing_input_failed": wa > US.KNOWN_TOL, "identity": {"value": 0.0, "reason": "the bank against itself"}, "tol": US.KNOWN_TOL,
                              "rule": "median over cases of unite_stats.known_ratio(a_P clean here, a_P before in SPEC4's T_15 falls npz) on the case's J contexts", "n_cases": len(ids)}
        g["known_spec4_a"]["ok"] = g["known_spec4_a"]["pass"] and g["known_spec4_a"]["failing_input_failed"]
        b_ids = ids[:N_KNOWN_B]
        ours = [J[x]["falls"][arm_name(a, "g1", "free")]["mean"] for x in b_ids for a in ("own", "cross")]
        theirs = [J[x]["known"][f"spec4_{a}"] for x in b_ids for a in ("own", "cross")]
        nxt = [J[b_ids[(i + 1) % len(b_ids)]]["known"][f"spec4_{a}"] for i in range(len(b_ids)) for a in ("own", "cross")]
        vb, wb = float(np.median(np.abs(np.subtract(ours, theirs)))), float(np.median(np.abs(np.subtract(ours, nxt))))
        g["known_spec4_b"] = {"value": vb, "pass": vb <= US.KNOWN_TOL, "failing_input": {"input": "the next case's SPEC4 falls", "value": wb}, "failing_input_failed": wb > US.KNOWN_TOL,
                              "identity": {"value": 0.0, "reason": "the bank against itself"}, "tol": US.KNOWN_TOL, "cases": b_ids,
                              "rule": "median |fall here - SPEC4's fall| over own' and cross' under g1, free scale, of the first 20 cases"}
        g["known_spec4_b"]["ok"] = g["known_spec4_b"]["pass"] and g["known_spec4_b"]["failing_input_failed"]

    def run(self):
        self.check_refs()
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        self.cuts()
        self.held_gates()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P = self.items["primary"]
        splits = self.splits_of(P)
        self.m["n_to_read"] = {"primary": len(P), "splits": len(splits), "planted": len(self.items["planted"])}
        self.m["n_done"] = {k: 0 for k in self.m["n_to_read"]}
        self.m["samples"] = {"cases": [j["job"] for j in P[:N_SAMPLE_CASES]], "files": ["samples/held_gates.json", "samples/cases.json", "samples/members_alone.json"]}
        q = self.foreign()
        for i, j in enumerate(P):
            self.case(j, q[j["job"]], sample=i < N_SAMPLE_CASES)
            self.m["n_done"]["primary"] += 1
            self.write()
        self.known_spec4()
        self.write()
        for key, (pole, c) in splits.items():
            self.ceiling_generic(key, pole, c)
            self.m["n_done"]["splits"] += 1
            self.write()
        for p in self.items["planted"]:
            self.pair(p)
            self.m["n_done"]["planted"] += 1
            self.write()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.summarize()
        self.m["status"] = "done"
        self.write()

    def summarize(self):
        idf = np.array([r["fall"] for r in self.id_rows])
        idc = np.array([r["change"] for r in self.id_rows])
        fl = np.array([r["fall"] for r in self.id_fail_rows])
        ih = np.array([r["change"] for r in self.idh_rows])
        g = self.m["gates"]
        g["identity"] = {"value": float(np.nanmax(np.abs(idf))), "max_abs_change": float(idc.max()), "n_contexts": int(idf.size), "tol": RA.ID_TOL,
                         "pass": bool((np.abs(idf) <= RA.ID_TOL).all()), "failing_input": {"input": "own (free) in place of the identity arm: contexts over 1e-3",
                                                                                           "value": int((np.abs(fl) > RA.ID_TOL).sum()), "n": int(fl.size)},
                         "failing_input_failed": bool((np.abs(fl) > RA.ID_TOL).any()), "identity": {"value": float(np.nanmax(np.abs(idf))), "reason": "read on the identity arm itself"}}
        g["identity"]["ok"] = g["identity"]["pass"] and g["identity"]["failing_input_failed"]
        g["held_identity"]["every_unit"] = {"max_abs_change": float(ih.max()), "n_contexts": int(ih.size), "pass": bool(ih.max() == 0.0)}
        g["held_identity"]["value"] = max(g["held_identity"]["value"], float(ih.max()))
        g["held_identity"]["pass"] = g["held_identity"]["value"] == 0.0
        g["held_identity"]["ok"] = g["held_identity"]["pass"] and g["held_identity"]["failing_input_failed"]
        self.m["exclusion"] = {**{e: dict(v) for e, v in self.excl.items()}, "pass": all(v["n_in_cut"] == 0 for v in self.excl.values()),
                               "rule": "no member of a set ranked or matched under a cut is in that cut"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.write()
        assert g["identity"]["ok"], f"identity failed: {g['identity']}"
        assert g["held_identity"]["ok"], f"held identity failed on a unit: {g['held_identity']}"


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["generic"])
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--primary-foreign", required=True, help="the primary run's T_15 foreign.json (local or s3), as SPEC4's T_15 load read it")
    ap.add_argument("--spec4-bank", required=True, help="SPEC4's T_15 step directory (local or s3): metrics.json and falls/")
    ap.add_argument("--t14-bank", required=True, help="SPEC5's dating T_14 step directory (local or s3): dating/<case>.npz")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def fetch_refs(a, fetch) -> dict:
    s4, t14 = a.spec4_bank.rstrip("/"), a.t14_bank.rstrip("/")
    return {"spec4_metrics": fetch(f"{s4}/metrics.json", "reuse_generic_spec4_t15/metrics.json"),
            "spec4_falls": fetch(f"{s4}/falls/", "reuse_generic_spec4_t15/falls", recursive=True),
            "t14_dating": fetch(f"{t14}/dating/", "reuse_generic_t14/dating", recursive=True), "spec4_bank": a.spec4_bank, "t14_bank": a.t14_bank}


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    if a.primary_foreign.startswith("s3://"):
        a.primary_foreign = str(fetch(a.primary_foreign, "reuse_generic_primary_t15_foreign.json"))
    a.primary_foreign_sha256 = hashlib.sha256(pathlib.Path(a.primary_foreign).read_bytes()).hexdigest()
    refs = fetch_refs(a, fetch)
    items_path = fetch(a.items, "reuse_generic_items.json")
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
    job = Job(a.out, a, items, model, tok, corpus, "cuda", refs)
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
    print("REUSE_GENERIC_OK", json.dumps({"t": a.t}), flush=True)


if __name__ == "__main__":
    main()
