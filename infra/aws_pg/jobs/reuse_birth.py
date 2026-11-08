"""SPEC4 BIRTH job, one O7 checkpoint per call (studies/reuse/SPEC4_birth.md §2-§4, §7, §11). A subclass of jobs.reuse_ablate.Job; the instrument
modules reuse_ablate and unite_route are imported unchanged and infra/pg/reuse_birth.py adds B_t, the exclusion, the top-200 bank and Part A.

    python -m infra.aws_pg.jobs.reuse_birth --stage birth --arms birth_t15 --t T_15 --revision stage1-step1413814 --items <s3 birth_items.json> \
        --corpus <s3> --primary-foreign <s3 primary T_15 foreign.json> --out <dir>
    python -m infra.aws_pg.jobs.reuse_birth --stage birth --arms prebirth_load --t T_5 --revision stage1-step8000 --items <s3> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.reuse_birth --stage birth --arms birth_load --t T_12 --revision stage1-step320000 --items <s3> --corpus <s3> \
        --from-step <this run's T_15 step directory, local> --out <dir>

Every load: the known number and completeness (the parent's gates), ā over every upstream layer (layers 0..n_layers-2, 31 on O7) on UNITE's e1
positions, B_t = the 1% largest |ā| (high_background.json), the GPU memory peak. Then, with B_t excluded from every ranking, matching and foreign set
(the ablated values stay ā), m 50, direct ranking, mean replacement, every side capped at 12 contexts by rank:
  birth_t15 (T_15): Q per case: the 150 primary cases' from --primary-foreign (asserted to be the same 150 jobs, each Q one of the case's
          candidates), the 93 fresh cases' by the parent's rule (nearest ||G||^2 + ||U||^2 among the case's foreign candidates); merged into
          foreign.json. The 150 primary cases with arms identity, own', cross', random', foreign', R_J' on J (R_J' matched to D'(O*) in mean |a| over
          J, as the parent's active_job builds it); the distinct primary ceiling splits on side 2 (identity, own' D'(side 2), cross' D'(side 1)); the
          56 planted pairs (labels: identity, own', cross', random'; side labels shuffled within the pair: identity, own', cross'). No fresh forward.
  birth_load (T_n): Q read from --from-step/foreign.json. The primary cases born at T_n (6 arms) and their ceiling splits, read once per split; the
          fresh cases born at T_n (identity, own', cross', random') and their ceiling splits; the 56 planted pairs; the Part A birth bank of every case
          born at T_n (from the captures of J and O* the arms already took) and the pre-birth bank of every case whose t_before is T_n (captures of J
          and O* only, no ablation).
  prebirth_load (T_5): the pre-birth banks of the cases whose t_before is T_n. No arm; metrics.json says the identity gate has nothing to read.
Arm names and the falls/<key>.npz format are the parent's (own_direct_m50, ...); each npz adds set_in_B [n_arms, 100] (False for every member), the
top-200 bank of each side (top<side>_idx, _A, _abar) and, for primary cases, rj_energy and match_energy. Part A banks: prebirth/<pole>__c<c>__J.npz
and __O.npz with t, read ("birth" or "prebirth"), side, tok, D' of that side at this checkpoint (D_layer, D_neuron; B_t of this load excluded), its
top-200 bank and the arrays of reuse_birth.part_a (loadsG, loadsU on the J side only: they are weights, the same for both sides).
metrics.json: the parent's record format for primary cases (jobs), primary ceiling splits (ceiling) and planted pairs (planted), so verdict.py's
readers read them unchanged; fresh cases in fresh, their splits in fresh_ceiling, Part A banks in part_a; rewritten after every unit.
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
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

M = RB.M
CAP = 12
_SPEC = {"ms": (M,), "rankings": ("direct",), "m_foreign": M, "zero": False, "draws": 0, "cap": CAP, "id_fail_arm": f"own_direct_m{M}"}
ARMSETS = {"birth_t15": {**_SPEC, "arms": True}, "birth_load": {**_SPEC, "arms": True}, "prebirth_load": {**_SPEC, "arms": False}}
PRIMARY_ARMS = ("identity", "own", "cross", "random", "foreign", "rj")
FRESH_ARMS = ("identity", "own", "cross", "random")


def arm(name: str, S: list) -> dict:
    return {"arm": "identity", "set": [], "replace": "mean"} if name == "identity" else {"arm": f"{name}_direct_m{M}", "set": S, "replace": "mean", "ranking": "direct", "m": M}


def choose_q(j: dict, layers) -> dict:
    """The parent's rule (jobs.reuse_ablate.Job.foreign at T_15): the candidate pole of P's layer and sign nearest in ||G||^2 + ||U||^2, P excluded."""
    mlp = layers[j["layer"]].mlp
    load = lambda n: float(mlp.gate_proj.weight[n].float().pow(2).sum() + mlp.up_proj.weight[n].float().pow(2).sum())
    lp = load(j["neuron"])
    cands = [{**c, "load": load(NH.parse_pole(c["pole"])[1])} for c in j["foreign_candidates"]]
    assert cands, f"{j['job']}: no foreign candidate"
    best = min(cands, key=lambda c: (abs(c["load"] - lp), c["pole"]))
    return {"pole": best["pole"], "c": best["c"], "tok": best["tok"], "load": best["load"], "P_load": lp, "candidates": {c["pole"]: c["load"] for c in cands}}


class Job(JR.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
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
        assert a.stage == "birth" and a.arms in ARMSETS, (a.stage, a.arms)
        assert (a.arms == "birth_t15") == (a.t == "T_15"), f"arm set {a.arms} at {a.t}"
        self.spec = ARMSETS[a.arms]
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.m = {"stage": a.stage, "arms": a.arms, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw,
                  "window": RA.LENGTH, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is",
                  "exclusion_rule": "B_t (high_background.json) removed before top_m ranks every set and before random_matched matches R and R_J",
                  "job_fall_rule": "mean over contexts with a_P > 0", "m": M, "cap": CAP, "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0,
                  "primary_foreign_sha256": getattr(a, "primary_foreign_sha256", None),
                  "jobs": {}, "fresh": {}, "planted": {}, "ceiling": {}, "fresh_ceiling": {}, "part_a": {}}
        self.lin = []
        self.id_rows, self.id_fail_rows = [], []
        self.split_sides = {}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0}
        self.write()

    # -------------------------------------------------------------------------------------- ā, B_t and Q
    def background(self):
        t0 = time.time()
        n_up = len(self.layers) - 1
        Ls = [c["layer"] for c in self.items["primary"] + self.items["fresh"]] + [p["layer"] for p in self.items["planted"]]
        assert max(Ls) <= n_up, (max(Ls), n_up)
        sets = UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)
        self.abar = RA.background_mean_a(self.model, sets["e1"], n_up, **self.kw(RA.LENGTH))
        self.m["tokens_forwarded"] += RA.n_forwarded(sets["e1"], RA.LENGTH, self.bos, self.hw)
        self.m["captures"] += len(sets["e1"])
        ab = self.abar.cpu().numpy()
        np.savez_compressed(self.out / "abar.npz", abar=ab, e1=sets["e1"])
        NH.dump(self.out / "background.json", {"e1": sets["e1"].tolist(), "t_index": self.t_index, "n_layers": n_up,
                                               "abar_abs_median_by_layer": np.median(np.abs(ab), 1).tolist(), "abar_abs_max_by_layer": np.abs(ab).max(1).tolist()})
        self.m["background"] = {"n_positions": int(len(sets["e1"])), "n_layers": n_up, "finite": bool(np.isfinite(ab).all()), "abar_abs_median": float(np.median(np.abs(ab)))}
        hb = RB.high_background(self.abar, n_up)
        self.mask = hb["mask"]
        RB.write_high_background(self.out / "high_background.json", hb, self.a.t)
        self.m["high_background"] = {k: hb[k] for k in ("cut", "count", "n_entries", "n_up", "frac", "per_layer")}
        self.m["high_background"]["cut_over_median_abs_abar"] = hb["cut"] / float(np.median(np.abs(ab)))
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"

    def foreign(self) -> dict:
        """T_15: the primary cases' Q from --primary-foreign, the fresh cases' Q by the parent's rule; merged into foreign.json. Other loads: the parent's
        reader of --from-step/foreign.json."""
        if self.a.t != "T_15":
            return super().foreign()
        prim = json.loads(pathlib.Path(self.a.primary_foreign).read_text())
        pj = {j["job"]: j for j in self.items["primary"]}
        assert set(prim) == set(pj), ("--primary-foreign names other jobs than the primary set", sorted(set(prim) ^ set(pj))[:5])
        for x, q in prim.items():
            cand = {c["pole"]: c for c in pj[x]["foreign_candidates"]}
            assert q["pole"] in cand and (cand[q["pole"]]["c"], cand[q["pole"]]["tok"]) == (q["c"], q["tok"]), (x, q["pole"], "Q is not one of the case's candidates")
        out = {**prim, **{j["job"]: choose_q(j, self.layers) for j in self.items["fresh"]}}
        assert len(out) == len(prim) + len(self.items["fresh"]), "a fresh job id is also a primary job id"
        NH.dump(self.out / "foreign.json", out)
        return out

    # -------------------------------------------------------------------------------------- helpers
    def capture(self, toks, unit):
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)
        return super().capture(toks, unit)

    def side(self, toks, unit):
        cap = self.capture(toks, unit)
        return cap, RA.direct_A(self.rows, cap, unit, self.W)

    def rank(self, score) -> list:
        return RB.top_m_excluded(score, M, self.W, self.mask)

    def tops(self, **scores) -> dict:
        out = {}
        for name, sc in scores.items():
            tb = RB.top_bank(sc, self.abar, self.W)
            out.update({f"top{name}_idx": tb["idx"], f"top{name}_A": tb["A"], f"top{name}_abar": tb["abar"]})
        return out

    def flags(self, arms: list) -> np.ndarray:
        f = RB.set_flags(arms, self.mask)
        self.excl["n_sets"] += sum(1 for x in arms if x["set"])
        self.excl["n_members"] += sum(len(x["set"]) for x in arms)
        self.excl["n_in_B"] += int(f.sum())
        assert not f.any(), "a set holds a B_t neuron"
        return f

    def bank_part_a(self, key: str, side: str, toks, cap: dict, A: torch.Tensor, unit: dict, read: str, D: list) -> str:
        arr = RB.part_a(self.rows, cap, unit, loads=side == "J")
        p = self.out / "prebirth" / f"{key}__{side}.npz"
        np.savez_compressed(p, t=self.a.t, read=read, side=side, pole=unit["unit"], tok=np.asarray(toks),
                            D_layer=np.array([l for l, _ in D], np.int16), D_neuron=np.array([k for _, k in D], np.int32), D_in_B=RB.in_B(D, self.mask),
                            **{k.replace(f"top{side}", "top"): v for k, v in self.tops(**{side: A.sum(0)}).items()}, **arr)
        return str(p.relative_to(self.out))

    # -------------------------------------------------------------------------------------- one case at its read checkpoint
    def case(self, j: dict, q: dict | None, fresh: bool):
        t0 = time.time()
        unit = UR.pole_unit(j["pole"])
        L = unit["layer"]
        assert L == j["layer"] and (q is None) == fresh, (j["job"], L, j["layer"])
        Jt, Ot = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"]))
        capJ, AJ = self.side(Jt, unit)
        capO, AO = self.side(Ot, unit)
        sJ, sO = AJ.sum(0), AO.sum(0)
        DJ, DO = self.rank(sJ), self.rank(sO)
        arms = [arm("identity", []), arm("own", DJ), arm("cross", DO), arm("random", RB.random_matched_excluded(DO, DJ, self.abar, self.mask))]
        extra = self.tops(J=sJ, O=sO)
        rec = {"pole": j["pole"], "layer": L, "tercile": j["tercile"], "birth": j["birth"], "label": j["label"], "n_J": int(len(Jt)), "n_O": int(len(Ot))}
        if not fresh:
            qunit = UR.pole_unit(q["pole"])
            assert qunit["layer"] == L, (j["pole"], q["pole"])
            Qt = np.asarray(q["tok"])[:CAP]
            _, AQ = self.side(Qt, qunit)
            DF = self.rank(AQ.sum(0))
            act = torch.stack([capJ["a"][l].abs().mean(0) for l in range(L)]).cpu().numpy().astype(np.float64)   # [L, W] mean |a_k| over J
            RJ = RB.random_matched_excluded(DO, DJ, act, self.mask)
            arms += [arm("foreign", DF), arm("rj", RJ)]
            extra.update({"rj_energy": act[tuple(np.array(RJ).T)].astype(np.float32), "match_energy": act[tuple(np.array(DO).T)].astype(np.float32)})
            rec["foreign"] = {"pole": q["pole"], "c": q["c"], "n": int(len(Qt))}
        extra["set_in_B"] = self.flags(arms)
        key = f"{j['pole']}__c{j['c']}"
        res = self.evaluate(key, unit, Jt, capJ, AJ, None, arms, linear=True, extra=extra)
        own, cr, rd = (res[f"{x}_direct_m{M}"] for x in ("own", "cross", "random"))
        k = f"direct_m{M}"
        rec.update({"falls": res, "I": {k: JR.ratio(cr["mean"], own["mean"])}, "s": {k: len(set(DJ) & set(DO)) / M}, "cancel": {k: RA.cancel(own["mean"], rd["mean"], own["n_pos"])},
                    "falls_path": f"falls/{key}.npz"})
        if not fresh:
            fo = res[f"foreign_direct_m{M}"]
            rec["s"].update({f"foreign_direct_m{M}": len(set(DJ) & set(DF)) / M, f"cross_foreign_direct_m{M}": len(set(DO) & set(DF)) / M})
            rec["foreign_contrast"] = {"direct": {f"cross_minus_foreign_m{M}": cr["mean"] - fo["mean"], f"I_foreign_m{M}": JR.ratio(fo["mean"], own["mean"])}}
            rec["rj"] = {"energy_median": float(np.median(extra["rj_energy"])), "match_energy_median": float(np.median(extra["match_energy"]))}
        if self.a.t != "T_15":
            rec["part_a"] = {"J": self.bank_part_a(key, "J", Jt, capJ, AJ, unit, "birth", DJ), "O": self.bank_part_a(key, "O", Ot, capO, AO, unit, "birth", DO)}
            self.m["part_a"][j["job"]] = {"read": "birth", "set": "fresh" if fresh else "primary", **rec["part_a"]}
        rec["seconds"] = round(time.time() - t0, 1)
        (self.m["fresh"] if fresh else self.m["jobs"])[j["job"]] = rec

    def prebirth(self, j: dict, fresh: bool):
        """Captures of J and O* at t_b - 1, no ablation: the Part A arrays, and D' of each side ranked here (B_t of this load excluded)."""
        unit = UR.pole_unit(j["pole"])
        assert unit["layer"] == j["layer"] and j["t_before"] == self.t_index, j["job"]
        key = f"{j['pole']}__c{j['c']}"
        paths = {}
        for side, toks in (("J", np.asarray(j["J_tok"])[:CAP]), ("O", np.asarray(j["O_tok"])[:CAP])):
            cap, A = self.side(toks, unit)
            D = self.rank(A.sum(0))
            self.excl["n_sets"] += 1
            self.excl["n_members"] += len(D)
            paths[side] = self.bank_part_a(key, side, toks, cap, A, unit, "prebirth", D)
        self.m["part_a"][j["job"]] = {"read": "prebirth", "set": "fresh" if fresh else "primary", "birth": j["birth"], **paths}

    def ceiling_split(self, name: str, pole: str, c: dict, store: dict):
        """A ceiling split read on side 2: identity, own' D'(side 2), cross' D'(side 1)."""
        t0 = time.time()
        unit = UR.pole_unit(pole)
        s1, s2 = (np.asarray(x)[:CAP] for x in c["sides_tok"])
        _, A1 = self.side(s1, unit)
        cap2, A2 = self.side(s2, unit)
        D1, D2 = self.rank(A1.sum(0)), self.rank(A2.sum(0))
        arms = [arm("identity", []), arm("own", D2), arm("cross", D1)]
        extra = {"set_in_B": self.flags(arms), **self.tops(S1=A1.sum(0), S2=A2.sum(0))}
        key = f"{pole}__c{c['c']}_ceiling_side2"
        res = self.evaluate(key, unit, s2, cap2, A2, None, arms, linear=True, extra=extra)
        I, s = JR.ratio(res[f"cross_direct_m{M}"]["mean"], res[f"own_direct_m{M}"]["mean"]), len(set(D1) & set(D2)) / M
        store[name] = {"pole": pole, "c": c["c"], "n_sides": [int(len(s1)), int(len(s2))], "s": s, "s_m": {f"m{M}": s}, "I_ceil": I, "I_ceil_m": {f"m{M}": I},
                       "side2": {"falls": res, "I": I, "I_m": {f"m{M}": I}, "n_read": int(len(s2)), "falls_path": f"falls/{key}.npz"}, "seconds": round(time.time() - t0, 1)}

    def planted_pair(self, p: dict):
        """The parent's planted read (labels, then side labels shuffled within the pair, rng seed 0) with D' and R' excluded."""
        t0 = time.time()
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        nJ = len(Jt)
        out = {}
        perm = np.random.default_rng(0).permutation(len(pool))
        for tag, jr, orr in (("", np.arange(nJ), np.arange(nJ, len(pool))), ("shuffled", np.sort(perm[:nJ]), np.sort(perm[nJ:]))):
            tj, to = (torch.as_tensor(x, device=A.device) for x in (jr, orr))
            sJ, sO = A[tj].sum(0), A[to].sum(0)
            DJ, DO = self.rank(sJ), self.rank(sO)
            arms = [arm("identity", []), arm("own", DJ), arm("cross", DO)] + ([] if tag else [arm("random", RB.random_matched_excluded(DO, DJ, self.abar, self.mask))])
            extra = {"set_in_B": self.flags(arms), **self.tops(J=sJ, O=sO)}
            key = f"{p['unit']}__planted" + (f"_{tag}" if tag else "")
            res = self.evaluate(key, unit, pool[jr], UR.cap_rows(cap, jr), A[tj], None, arms, linear=False, extra=extra)
            I, s = JR.ratio(res[f"cross_direct_m{M}"]["mean"], res[f"own_direct_m{M}"]["mean"]), len(set(DJ) & set(DO)) / M
            out[tag or "labels"] = {"falls": res, "I": I, "s": s, "I_m": {f"m{M}": I}, "s_m": {f"m{M}": s}, "n_J": int(len(jr)), "n_O": int(len(orr)), "falls_path": f"falls/{key}.npz"}
        self.m["planted"][p["unit"]] = {"layer": p["layer"], "a": p["a"], "b": p["b"], "lam": p["lam"], **out, "seconds": round(time.time() - t0, 1)}

    # -------------------------------------------------------------------------------------- the load
    def splits_of(self, cases: list) -> dict:
        """Distinct ceiling splits "<pole>__c<ceiling c>" of the cases that carry one; cases naming one split carry the same sides."""
        out = {}
        for j in cases:
            if j["ceiling"] is None:
                continue
            key = f"{j['pole']}__c{j['ceiling']['c']}"
            if key in out:
                assert out[key][1]["sides_tok"] == j["ceiling"]["sides_tok"], f"{key}: two cases name this split with different sides"
            else:
                out[key] = (j["pole"], j["ceiling"])
        return out

    def run(self):
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        n = self.t_index
        P, F = self.items["primary"], self.items["fresh"]
        bornP, bornF = [j for j in P if j["birth"] == n], [j for j in F if j["birth"] == n]
        preP, preF = [j for j in P if j["t_before"] == n], [j for j in F if j["t_before"] == n]
        a = self.a.arms
        if a == "birth_t15":
            readP, readF = P, []
            assert not preP and not preF, "a case has t_before T_15"
        elif a == "birth_load":
            readP, readF = bornP, bornF
            assert readP or readF, f"no case is born at {self.a.t}"
        else:
            readP, readF = [], []
            assert not bornP and not bornF and (preP or preF), f"prebirth_load at {self.a.t}: births {len(bornP) + len(bornF)}, pre-birth cases {len(preP) + len(preF)}"
        splitsP, splitsF = self.splits_of(readP), self.splits_of(readF)
        self.m["n_to_read"] = {"primary": len(readP), "fresh": len(readF), "primary_splits": len(splitsP), "fresh_splits": len(splitsF),
                               "planted": len(self.items["planted"]) if self.spec["arms"] else 0, "part_a_birth": 0 if a == "birth_t15" else len(readP) + len(readF),
                               "part_a_prebirth": len(preP) + len(preF)}
        self.m["n_done"] = {k: 0 for k in self.m["n_to_read"]}
        q = self.foreign() if readP else {}
        for fresh, cases in ((False, readP), (True, readF)):
            for j in cases:
                self.case(j, None if fresh else q[j["job"]], fresh)
                if a == "birth_load":
                    key = f"{j['pole']}__c{j['ceiling']['c']}" if j["ceiling"] else None
                    rec = self.m["fresh" if fresh else "jobs"][j["job"]]
                    rec["ceiling_split"] = key
                    if key is not None and key not in (self.m["fresh_ceiling"] if fresh else self.m["ceiling"]):
                        self.ceiling_split(key, *(splitsF if fresh else splitsP)[key], self.m["fresh_ceiling"] if fresh else self.m["ceiling"])
                        self.m["n_done"]["fresh_splits" if fresh else "primary_splits"] += 1
                    self.m["n_done"]["part_a_birth"] += 1
                self.m["n_done"]["fresh" if fresh else "primary"] += 1
                self.write()
        if a == "birth_t15":
            for key, (pole, c) in splitsP.items():
                self.ceiling_split(key, pole, c, self.m["ceiling"])
                self.m["n_done"]["primary_splits"] += 1
                self.write()
        if self.spec["arms"]:
            for p in self.items["planted"]:
                self.planted_pair(p)
                self.m["n_done"]["planted"] += 1
                self.write()
        for fresh, cases in ((False, preP), (True, preF)):
            for j in cases:
                self.prebirth(j, fresh)
                self.m["n_done"]["part_a_prebirth"] += 1
                self.write()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.m["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0, "rule": "every member of every ranked, matched and foreign set is outside B_t"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.summarize()
        self.m["status"] = "done"
        self.write()

    def summarize(self):
        """The parent's summary (identity gate, linearity, medians over the primary cases read here), the fresh cases' medians and R_J'. A pre-birth load
        reads no arm, so its identity gate has nothing to read and says so."""
        if not self.spec["arms"]:
            assert not self.id_rows, "a pre-birth load read an arm"
            self.m["identity"] = {"read": False, "pass": None, "reason": "no arm is read at a pre-birth load; the identity gate has nothing to read"}
            self.m["summary"] = {"part_a_prebirth": len(self.m["part_a"])}
            self.write()
            return
        super().summarize()
        med = lambda xs: float(np.nanmedian(xs)) if len(xs) else float("nan")
        Fr, J = self.m["fresh"], self.m["jobs"]
        k = f"direct_m{M}"
        s = self.m["summary"]
        s["fresh"] = {**{x: med([v["falls"][f"{x}_{k}"]["mean"] for v in Fr.values()]) for x in ("own", "cross", "random")},
                      "I_median": med([v["I"][k] for v in Fr.values()]), "s_median": med([v["s"][k] for v in Fr.values()]),
                      "n_cancelled": sum(v["cancel"][k]["cancelled"] for v in Fr.values()), "n": len(Fr)}
        s[f"rj_direct_m{M}"] = {"median_fall": med([v["falls"][f"rj_direct_m{M}"]["mean"] for v in J.values()]), "n_jobs": len(J)}
        if self.m["fresh_ceiling"]:
            s["fresh_ceiling_I"] = {u: v["I_ceil"] for u, v in self.m["fresh_ceiling"].items()}
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["birth"])
    ap.add_argument("--arms", required=True, choices=sorted(ARMSETS))
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--from-step", default=None, help="birth_load: this run's T_15 step directory (local), whose foreign.json holds every Q")
    ap.add_argument("--primary-foreign", default=None, help="birth_t15: the primary run's T_15 foreign.json (local or s3)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    assert (a.arms == "birth_t15") == (a.primary_foreign is not None), "--primary-foreign is the T_15 load's, and only its"
    assert (a.arms == "birth_load") == (a.from_step is not None), "--from-step is the birth loads', and only theirs"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    if a.from_step and a.from_step.startswith("s3://"):
        a.from_step = str(fetch(a.from_step.rstrip("/") + "/foreign.json", f"reuse_birth_{a.t}_from_step/foreign.json").parent)
    if a.primary_foreign:
        if a.primary_foreign.startswith("s3://"):
            a.primary_foreign = str(fetch(a.primary_foreign, "reuse_birth_primary_t15_foreign.json"))
        a.primary_foreign_sha256 = hashlib.sha256(pathlib.Path(a.primary_foreign).read_bytes()).hexdigest()
    items_path = fetch(a.items, f"reuse_birth_{a.t}_items.json")
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
    print("REUSE_BIRTH_OK", json.dumps({"t": a.t, "arms": a.arms}), flush=True)


if __name__ == "__main__":
    main()
