"""INSTRUMENTS E4 job, one O7 load at T_15 (studies/instruments/SPEC_instruments.md §6 and amendment 40: the collateral forecast on Ward-defined
clusters, exploratory). A subclass of jobs.reuse_birth.Job like jobs.instruments_e1's: the model load, the corpus, the parent's activation gate and
completeness (jobs.reuse_ablate.Job.known), the captures with the direct attribution (JB.Job.side: unite_route.capture + reuse_ablate.direct_A),
the ranking with B_t excluded (JB.Job.rank: reuse_birth.top_m_excluded) and the replacement (reuse_ablate.ablation_dict, replaced, readout,
job_fall) are imported unchanged; the planted gate's pieces are jobs.instruments_e1's (amendment 20).

    python -m infra.aws_pg.jobs.instruments_e4 --t T_15 --revision stage1-step1413814 --items <s3 or local e4_items.json> --corpus <s3> --out <dir>

Background. ā over layers 0..30 on UNITE's e1 positions (reuse_ablate.background_mean_a, as JB.Job.background computes it) and B_t =
reuse_birth.high_background (SPEC4's global top 1 percent); B_t's member set is asserted equal to E1's banked T_15 high_background.json (the items'
B_t) before any case; the comparison with SPEC4's B_t (6 members differ) is recorded as that check's failing input.
Per case (items: one per pole; sides A and B are the pole's two largest Ward clusters, each <= 12 contexts from distinct documents):
  captures of A and B on their own contexts and the direct attribution per side; D'(A), D'(B) = the top 50 of the summed attribution with B_t
      excluded; s_true = |D'(A) & D'(B)| / 50. random = reuse_birth.random_matched_excluded(D'(B), D'(A), ā, B_t).
  the proxy (studies/reuse/results/sanity/s4_sameweights.py): Ã_k = mean_x |a_k(x)| . mean_x s_l(x) . |cg G.d_k + cu U.d_k| over the side's
      contexts, (cg, cu) the pole's mean coefficients over the side (c_g = silu'(g) u, c_u = silu(g)); P(X) = its top 50 with B_t excluded;
      s_proxy = |P(A) & P(B)| / 50; the proxy's overlap with the true set |P(X) & D'(X)| / 50 per side. Reported beside it, not a rule: the same
      with |ā_k| in place of mean_x |a_k(x)| (s_proxy_abar; amendment 40 names "the banked T_15 ā").
  read on A's contexts (a_P before, then one forward per arm): identity (hooks, nothing written), identity_fail (D'(A)'s first member at ā), own
      D'(A), cross D'(B), random, each under zero and under mean (ā); other__zero (the D'(A) of the most recent earlier case on another neuron
      whose layer is <= L: the room gate's failing input; absent for a case with none); and, when the unexcluded top 50 of B (reuse_ablate.top_m)
      holds a B_t member, planted_dose1 / planted_dose10 (that set's largest-|ā| B_t member written alone at ā and 10 ā) and
      planted_same_layer_dose1 / _dose10 (the same values written into neuron (L, (a + 1) mod W) of the pole's own layer: Δ = 0 exactly).
  falls per arm: reuse_ablate.job_fall (the mean over A's contexts with a_P > 0); I_v = fall(cross, v) / fall(own, v) for v in zero, mean;
      SPEC4's cancel flags per value (own < 2 |random| or under 3 contexts with a_P > 0).
  e4/<pole>.npz (written before metrics.json): tok_A, tok_B, ranks, a_before [N_A], arms, a_after [arms, N_A], per set (own, cross, random,
      own_u, cross_u, proxy_A, proxy_B, proxy_abar_A, proxy_abar_B, other, planted) its layer, neuron and ā; s_true, s_proxy, s_proxy_abar and the
      proxy overlaps.
Gates (the verdict decides; the job asserts instrument checks only):
  known_number  on the items' 20 known cases (run first): overlap of the job's D'(A) with D'_bank(A) (the E3 bank's A on U summed over A's ranks,
                E1's B_t removed; built by build_items_e4.py), median >= 0.9, asserted after the 20; failing input the reversed ranks' bank set
                (recorded; the verdict exits 1 if it does not fail).
  B_t           this load's B_t equal to E1's (asserted); failing input SPEC4's member set.
  identity      max over cases and contexts of |a_P(identity) - a_P before| / max |a_P before| <= 1e-6; failing input identity_fail (asserted).
  exclusion     no member of own, cross, random or a proxy set in B_t (asserted per set); failing input the unexcluded own_u and cross_u.
  own_room      median over cases of the own fall under zero >= 0.3; failing input other__zero (the sets of a different pole).
  planted       amendment 20 on the cases with a B_t member in cross_u: share with |Δa_P(10)| > |Δa_P(1)| > 10 x 1e-6 >= 0.9; failing input the
                same-layer writes.
  samples       samples/e4.jsonl: the items' 3 sample cases with contexts (prompt, marked token), sets, a_P before and after per arm, falls.
metrics.json is rewritten after every case. Fail loud: no try/except, no fallback; asserts carry their reason.
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

from infra.aws_pg.jobs import instruments_e1 as JE1
from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_birth as JB
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import reuse_generic as RG
from infra.pg import unite_route as UR

M, CAP = RB.M, JB.CAP
VALUES = ("zero", "mean")
SETS = ("own", "cross", "random")
KNOWN_LINE, ID_TOL, OWN_LINE, DOSE, PLANTED_SHARE = 0.9, JE1.ID_TOL, 0.3, JE1.DOSE, JE1.PLANTED_SHARE
EXCLUDED_SETS = ("own", "cross", "random", "proxy_A", "proxy_B", "proxy_abar_A", "proxy_abar_B", "other")
PLANTED_ARMS = ("planted_dose1", "planted_dose10", "planted_same_layer_dose1", "planted_same_layer_dose10")


# ------------------------------------------------------------------------------------------ pure pieces (the dry run and the verdict name them)
def overlap(A: list, B: list) -> float:
    a, b = {tuple(map(int, x)) for x in A}, {tuple(map(int, x)) for x in B}
    assert len(a) == len(A) == M and len(b) == len(B) == M, (len(a), len(A), len(b), len(B))
    return len(a & b) / M


def arm_names(has_other: bool, has_member: bool) -> list:
    """The arms read on side A, in reading order."""
    return ["identity", "identity_fail"] + [f"{x}__{v}" for x in SETS for v in VALUES] + (["other__zero"] if has_other else []) \
        + (list(PLANTED_ARMS) if has_member else [])


@torch.inference_mode()
def proxy_score(rows: UR.Rows, cap: dict, unit: dict, mag: torch.Tensor | None = None) -> torch.Tensor:
    """[L W]: s4_sameweights' Ã_k = mean_x |a_k(x)| . mean_x s_l(x) . |cg G.d_k + cu U.d_k|, (cg, cu) the pole's mean coefficients over the
    captured contexts; mag [>= L, W] replaces mean_x |a_k(x)| (the |ā| variant). The pole's sign does not enter (an absolute value)."""
    assert len(unit["terms"]) == 1, "a single pole"
    L, n = unit["layer"], unit["terms"][0]["n"]
    gc, uc = UR.term_gu(cap, unit)
    cg, cu = UR.coefficients(gc[:, 0], uc[:, 0])
    cg, cu = cg.mean(), cu.mean()
    lo = rows.loads(L, n)
    act = (lambda l: cap["a"][l].abs().mean(0)) if mag is None else (lambda l: mag[l].abs().float())
    return torch.cat([act(l) * cap["s"][l].mean() * (cg * lo[l][0] + cu * lo[l][1]).abs() for l in range(L)])


def gate_known(per_case: dict, line: float = KNOWN_LINE) -> dict:
    ov = np.array([v["overlap"] for v in per_case.values()])
    ow = np.array([v["failing_input_overlap"] for v in per_case.values()])
    return {"median_overlap": float(np.median(ov)), "min_overlap": float(ov.min()), "n_below_line": int((ov < line).sum()), "line": line, "n_cases": int(ov.size),
            "pass": bool(np.median(ov) >= line), "failing_input": "the reversed ranks' bank set (rank r read as 49 - r)",
            "failing_input_median_overlap": float(np.median(ow)), "failing_input_failed": bool(np.median(ow) < line),
            "rule": f"median over the known cases of |D'(A) & D'_bank(A)| / {M} >= {line}"}


def check_items(items: dict, t: str):
    assert items["t"] == t == "T_15" and items["grid"] == "t15_step1413814" and items["m"] == M and items["cap"] == CAP, (items["t"], items["grid"])
    C = items["cases"]
    assert len(C) == items["counts"]["cases"] and len({c["pole"] for c in C}) == len(C), "a pole repeats or the count differs"
    nk = len(items["known_cases"])
    assert nk == items["counts"]["known_cases"] and [c["pole"] for c in C[:nk]] == [k["pole"] for k in items["known_cases"]], "known cases do not run first"
    assert set(items["sample_cases"]) <= {c["pole"] for c in C} and len(items["B_t"]["members"]) == items["B_t"]["count"]
    for c in C:
        assert c["layer"] >= 1, c["pole"]
        for s in ("A", "B"):
            x = c[s]
            assert 6 <= len(x["tok"]) <= CAP and len(set(x["doc"])) == len(x["doc"]) and x["ranks"] == sorted(x["ranks"]), (c["pole"], s)


# ------------------------------------------------------------------------------------------ the job
class Job(JB.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "e4", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        assert self.bos is not None, "the instrument prepends a BOS; this tokenizer defines none"
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P and the coefficients assume silu"
        self.start = RA.window_start(self.bos)
        self.known_cases = {k["pole"]: k for k in items["known_cases"]}
        self.sample_cases = set(items["sample_cases"])
        self.m = {"stage": "instruments_e4", "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "status": "running", "m": M, "cap": CAP, "values": list(VALUES),
                  "label": "EXPLORATORY (amendment 40: the Ward partition was chosen after E3's unblinding)",
                  "replacement": "the value (0, or ā over UNITE's e1 positions) written into down_proj's input at every window position; BOS left",
                  "exclusion_rule": "B_t removed before top_m ranks D'(A), D'(B) and the proxy sets and before random_matched matches random",
                  "job_fall_rule": "mean over side A's contexts with a_P > 0", "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0,
                  "readouts": 0, "n_to_read": {"cases": len(items["cases"]), "known_cases": len(self.known_cases)}, "n_done": {"cases": 0},
                  "gates": {}, "cases": {}}
        self.chk = {"id": [], "id_fail": [], "planted": [], "known": {}}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0, "unexcluded_n_sets": 0, "unexcluded_n_in_B": 0}
        self.prev = []
        self.n_samples = 0
        self.write()

    # -------------------------------------------------------------------------------------- ā and B_t
    def background(self):
        t0 = time.time()
        n_up = len(self.layers) - 1
        assert max(c["layer"] for c in self.items["cases"]) <= n_up, "a pole above the last upstream layer"
        pos = UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)["e1"]
        self.abar = RA.background_mean_a(self.model, pos, n_up, **self.kw(RA.LENGTH))
        self.m["captures"] += len(pos)
        self.m["tokens_forwarded"] += RA.n_forwarded(pos, RA.LENGTH, self.bos, self.hw)
        ab = self.abar.cpu().numpy()
        np.savez_compressed(self.out / "abar.npz", abar=ab, e1=np.asarray(pos))
        hb = RB.high_background(self.abar, n_up)
        self.mask = hb["mask"]
        RB.write_high_background(self.out / "high_background.json", hb, self.a.t)
        ours = {(l, k) for l, k, _ in hb["members"]}
        theirs = {(int(l), int(k)) for l, k, _ in self.items["B_t"]["members"]}
        spec4 = {(int(l), int(k)) for l, k in self.items["B_t_spec4"]["members"]}
        d = max(abs(float(ab[int(l), int(k)]) - float(v)) / (1 + abs(float(v))) for l, k, v in self.items["B_t"]["members"])
        self.m["background"] = {"n_positions": int(len(pos)), "n_layers": n_up, "finite": bool(np.isfinite(ab).all()), "abar_abs_median": float(np.median(np.abs(ab)))}
        self.m["gates"]["B_t"] = {"value": len(ours ^ theirs), "pass": ours == theirs, "n_ours": len(ours), "n_e1": len(theirs), "cut": hb["cut"],
                                  "abar_at_e1_members_max_scaled_diff": d, "e1_path": self.items["B_t"]["path"], "e1_sha256": self.items["B_t"]["sha256"],
                                  "failing_input": {"input": "SPEC4's B_t member set in place of E1's", "value": len(ours ^ spec4)},
                                  "failing_input_failed": ours != spec4, "identity": {"value": 0, "reason": "a set against itself"},
                                  "rule": "the member set of this load's B_t equals E1's banked T_15 high_background.json (symmetric difference 0)"}
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"
        assert ours == theirs, f"B_t recomputed from this load's ā differs from E1's banked B_t: {len(ours ^ theirs)} members"

    # -------------------------------------------------------------------------------------- readouts
    def readout(self, toks, unit: dict, ablation: dict | None) -> np.ndarray:
        a = RA.readout(self.model, toks, unit, ablation, **self.kw(RA.LENGTH))["a"]
        self.m["readouts"] += 1
        self.m["forwards"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
        return a

    def same_layer_readout(self, toks, unit: dict, k: int, val: torch.Tensor) -> np.ndarray:
        with JE1.same_layer(self.layers, unit["layer"], k, val, self.start) as rec:
            a = self.readout(toks, unit, None)
        assert rec["calls"] == -(-len(toks) // self.a.batch), f"{rec['calls']} same-layer writes for {len(toks)} contexts"
        return a

    def count_sets(self, sets: dict):
        for x, S in sets.items():
            n = int(RB.in_B(S, self.mask).sum())
            if x in EXCLUDED_SETS:
                self.excl["n_sets"] += 1
                self.excl["n_members"] += len(S)
                self.excl["n_in_B"] += n
                assert n == 0, f"the excluded set {x} holds a B_t neuron"
            elif x in ("own_u", "cross_u"):
                self.excl["unexcluded_n_sets"] += 1
                self.excl["unexcluded_n_in_B"] += n

    # -------------------------------------------------------------------------------------- one case
    def case(self, c: dict):
        t0 = time.time()
        unit = UR.pole_unit(c["pole"])
        L, n, sign = unit["layer"], unit["terms"][0]["n"], unit["terms"][0]["sign"]
        assert (L, n, "+" if sign > 0 else "-") == (c["layer"], c["neuron"], c["sign"]), c["pole"]
        tA, tB = (np.asarray(c[s]["tok"], np.int64) for s in ("A", "B"))
        for s, t in (("A", tA), ("B", tB)):
            assert np.array_equal(np.asarray(self.tokens[t], np.int64), c[s]["token_id"]), (c["pole"], s, "the marked token differs from the items'")
        capA, AA = self.side(tA, unit)
        capB, AB = self.side(tB, unit)
        sA, sB = AA.sum(0), AB.sum(0)
        DA, DB = self.rank(sA), self.rank(sB)
        sets = {"own": DA, "cross": DB, "random": RB.random_matched_excluded(DB, DA, self.abar, self.mask), "own_u": RA.top_m(sA, M, self.W),
                "cross_u": RA.top_m(sB, M, self.W), "proxy_A": self.rank(proxy_score(self.rows, capA, unit)), "proxy_B": self.rank(proxy_score(self.rows, capB, unit)),
                "proxy_abar_A": self.rank(proxy_score(self.rows, capA, unit, self.abar)), "proxy_abar_B": self.rank(proxy_score(self.rows, capB, unit, self.abar))}
        del capB, AB
        other = next(((p, D) for l, nrn, p, D in reversed(self.prev) if l <= L and nrn != c["pole"][:-1]), None)
        if other is not None:
            sets["other"] = other[1]
        member = RG.members_alone(sets["cross_u"], self.mask, self.abar)[:1]
        if member:
            sets["planted"] = member
        self.count_sets(sets)
        ab = lambda S, v: RA.ablation_dict(S, v, self.abar, L, self.device)
        before = self.readout(tA, unit, None)
        after = {"identity": self.readout(tA, unit, RA.ablation_dict([], "mean", self.abar, L, self.device, identity=True)),
                 "identity_fail": self.readout(tA, unit, ab(DA[:1], "mean"))}
        for x in SETS:
            for v in VALUES:
                after[f"{x}__{v}"] = self.readout(tA, unit, ab(sets[x], v))
        if other is not None:
            after["other__zero"] = self.readout(tA, unit, ab(sets["other"], "zero"))
        same = -1
        if member:
            (ml, mk), = member
            same = JE1.same_layer_neuron(unit, self.W)
            after["planted_dose1"] = self.readout(tA, unit, ab(member, "mean"))
            after["planted_dose10"] = self.readout(tA, unit, RA.ablation_dict(member, "mean", DOSE * self.abar, L, self.device))
            after["planted_same_layer_dose1"] = self.same_layer_readout(tA, unit, same, self.abar[ml, mk].float())
            after["planted_same_layer_dose10"] = self.same_layer_readout(tA, unit, same, DOSE * self.abar[ml, mk].float())
        names = list(after)
        assert names == arm_names(other is not None, bool(member)), names
        F = {k: RA.job_fall(before, a) for k, a in after.items()}
        I = {v: JR.ratio(F[f"cross__{v}"]["mean"], F[f"own__{v}"]["mean"]) for v in VALUES}
        s = {"s_true": overlap(DA, DB), "s_proxy": overlap(sets["proxy_A"], sets["proxy_B"]), "s_proxy_abar": overlap(sets["proxy_abar_A"], sets["proxy_abar_B"]),
             "proxy_overlap_A": overlap(sets["proxy_A"], DA), "proxy_overlap_B": overlap(sets["proxy_B"], DB),
             "proxy_abar_overlap_A": overlap(sets["proxy_abar_A"], DA), "proxy_abar_overlap_B": overlap(sets["proxy_abar_B"], DB)}
        abn = self.abar.cpu().numpy()
        z = {"pole": c["pole"], "layer": L, "tok_A": tA, "tok_B": tB, "ranks_A": np.asarray(c["A"]["ranks"]), "ranks_B": np.asarray(c["B"]["ranks"]),
             "a_before": before, "arms": np.array(names), "a_after": np.stack([after[k] for k in names]).astype(np.float32),
             "other_pole": other[0] if other is not None else "", "same_layer_neuron": same, **{k: np.float64(v) for k, v in s.items()}}
        for x, S in sets.items():
            li, ki = np.array([l for l, _ in S], np.int64), np.array([k for _, k in S], np.int64)
            z.update({f"set_{x}_layer": li.astype(np.int16), f"set_{x}_neuron": ki.astype(np.int32), f"set_{x}_abar": abn[li, ki].astype(np.float32)})
        np.savez_compressed(self.out / "e4" / f"{c['pole']}.npz", **z)
        rec = {"layer": L, "band": c["band"], "n_A": int(len(tA)), "n_B": int(len(tB)), "npz": f"e4/{c['pole']}.npz",
               "falls": {k: {q: f[q] for q in ("mean", "median", "n_pos")} for k, f in F.items()}, "I": I,
               "cancel": {v: RA.cancel(F[f"own__{v}"]["mean"], F[f"random__{v}"]["mean"], F[f"own__{v}"]["n_pos"]) for v in VALUES}, **s,
               "n_in_B": {x: int(RB.in_B(S, self.mask).sum()) for x, S in sets.items()}, "other_pole": other[0] if other is not None else None,
               "planted_member": [[int(l), int(k), float(abn[l, k])] for l, k in member], "set_sizes": {x: len(S) for x, S in sets.items()}}
        if c["pole"] in self.known_cases:
            k = self.known_cases[c["pole"]]
            rec["known"] = {"overlap": overlap(DA, k["D_bank"]), "failing_input_overlap": overlap(DA, k["D_bank_reversed"])}
            self.chk["known"][c["pole"]] = rec["known"]
        scale = np.abs(before.astype(np.float64)).max()
        dlt = lambda name: float(np.abs(after[name].astype(np.float64) - before).max() / scale)
        self.chk["id"].append(dlt("identity"))
        self.chk["id_fail"].append(dlt("identity_fail"))
        if member:
            r = {x: JE1.planted_response(before, after[x]) for x in PLANTED_ARMS}
            self.chk["planted"].append({"pole": c["pole"], **r, "max_abs_delta_same_layer": max(float(np.abs(after[x] - before).max()) for x in PLANTED_ARMS[2:])})
        if c["pole"] in self.sample_cases:
            self.sample(c, unit, before, after, F, sets, s, I)
        self.prev.append((L, c["pole"][:-1], c["pole"], DA))
        rec["seconds"] = round(time.time() - t0, 1)
        self.m["cases"][c["pole"]] = rec

    def sample(self, c: dict, unit: dict, before, after: dict, F: dict, sets: dict, s: dict, I: dict):
        abn = self.abar.cpu().numpy()
        prA, prB = self.prompts(c["A"]["tok"]), self.prompts(c["B"]["tok"])
        row = {"pole": c["pole"], "layer": c["layer"], "band": c["band"], "label": self.m["label"],
               "A": {"w": c["A"]["w"], "contexts": [{"rank": r, "doc": d, "tok": int(t), **pr, "a_before": float(before[i]),
                                                     "a_after": {k: float(a[i]) for k, a in after.items()}} for i, (r, d, t, pr) in
                                                    enumerate(zip(c["A"]["ranks"], c["A"]["doc"], c["A"]["tok"], prA))]},
               "B": {"w": c["B"]["w"], "contexts": [{"rank": r, "doc": d, "tok": int(t), **pr} for r, d, t, pr in zip(c["B"]["ranks"], c["B"]["doc"], c["B"]["tok"], prB)]},
               "sets": {x: [[int(l), int(k), float(abn[l, k])] for l, k in S] for x, S in sets.items()},
               "written": {"zero": "0 at every member", "mean": "ā of each member (third field of each set entry)"},
               "falls": {k: f["mean"] for k, f in F.items()}, "I": I, **s}
        with open(self.out / "samples" / "e4.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n_samples += 1

    # -------------------------------------------------------------------------------------- gates
    def known_gate(self):
        rec = gate_known(self.chk["known"])
        self.m["gates"]["known_number"] = rec
        self.write()
        assert rec["pass"], f"known number failed: median overlap {rec['median_overlap']}"

    def summarize(self):
        c, g, e = self.chk, self.m["gates"], self.excl
        g["identity"] = {"value": max(c["id"]), "tol": ID_TOL, "pass": max(c["id"]) <= ID_TOL, "n_cases": len(c["id"]),
                         "failing_input": {"input": "D'(A)'s first member written at ā in place of the empty set", "value": max(c["id_fail"])},
                         "failing_input_failed": max(c["id_fail"]) > ID_TOL, "identity": {"value": max(c["id"]), "reason": "read on the identity arm itself"},
                         "rule": "max over cases and contexts of |a_P after - a_P before| / max |a_P before| on the identity arm"}
        g["exclusion"] = {**e, "value": e["n_in_B"], "pass": e["n_in_B"] == 0,
                          "failing_input": {"input": "the unexcluded top 50 of A and of B (reuse_ablate.top_m)", "value": e["unexcluded_n_in_B"]},
                          "failing_input_failed": e["unexcluded_n_in_B"] > 0, "rule": "no member of own, cross, random, other or a proxy set in B_t"}
        cs = self.m["cases"].values()
        own = float(np.nanmedian([r["falls"]["own__zero"]["mean"] for r in cs]))
        oth = [r["falls"]["other__zero"]["mean"] for r in cs if "other__zero" in r["falls"]]
        oth_m = float(np.nanmedian(oth)) if oth else float("nan")
        g["own_room"] = {"value": own, "line": OWN_LINE, "pass": bool(own >= OWN_LINE), "n_cases": len(self.m["cases"]),
                         "failing_input": {"input": "the D'(A) of an earlier case on another neuron at layer <= L (the sets of a different pole), zero", "value": oth_m,
                                           "n_cases": len(oth)}, "failing_input_failed": not oth_m >= OWN_LINE,
                         "rule": "median over cases of the own fall (D'(A) on A, zero) >= 0.3"}
        rows = c["planted"]
        share = lambda d1, d10: float(np.mean([JE1.planted_side(r[d1], r[d10]) for r in rows])) if rows else float("nan")
        v, vf = share("planted_dose1", "planted_dose10"), share("planted_same_layer_dose1", "planted_same_layer_dose10")
        nanmed = lambda x: float(np.median(x)) if x else float("nan")
        g["planted"] = {"value": v, "line": PLANTED_SHARE, "pass": bool(v >= PLANTED_SHARE), "n_cases": len(rows), "n_cases_without_member": len(self.m["cases"]) - len(rows),
                        "dose1_median": nanmed([r["planted_dose1"] for r in rows]), "dose10_median": nanmed([r["planted_dose10"] for r in rows]),
                        "failing_input": {"input": "the same two writes into neuron (L, (a + 1) mod W) of the pole's own layer", "value": vf,
                                          "max_abs_delta": max([r["max_abs_delta_same_layer"] for r in rows], default=float("nan"))},
                        "failing_input_failed": not vf >= PLANTED_SHARE, "sides": rows,
                        "rule": "amendment 20 on side A of every case whose unexcluded cross set holds a B_t member: its largest-|ā| member alone at ā and 10 ā; "
                                "share with |Δa_P(10)| > |Δa_P(1)| > 10 x 1e-6 >= 0.9"}
        g["samples"] = {"lines": self.n_samples, "want": len(self.sample_cases), "pass": self.n_samples == len(self.sample_cases), "path": "samples/e4.jsonl"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.write()
        assert g["identity"]["pass"] and g["identity"]["failing_input_failed"], f"identity: {g['identity']}"
        assert g["exclusion"]["pass"], f"exclusion: {g['exclusion']}"

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        t0, tok0 = time.time(), self.m["tokens_forwarded"]
        nk = len(self.known_cases)
        for c in self.items["cases"]:
            self.case(c)
            self.m["n_done"]["cases"] += 1
            if self.m["n_done"]["cases"] == nk:
                self.m["timing_s"]["known_cases"] = round(time.time() - t0, 1)
                self.known_gate()
            self.write()
        assert self.m["n_done"]["cases"] == self.m["n_to_read"]["cases"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["cases"] = round(dt, 1)
        self.m["cases_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.summarize()
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
    items_path = fetch(a.items, "instruments_e4_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    check_items(items, a.t)
    grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
    assert grid[a.t] == a.revision == items["revision"], f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    assert corpus["tokens_sha256"] == items["corpus"]["tokens_sha256"], "the corpus is not the one the grid scored"
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
    t2 = time.time()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t2, 1)
    job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("INSTRUMENTS_E4_OK", json.dumps({"t": a.t, "cases": job.m["n_done"]["cases"], "captures": job.m["captures"]}), flush=True)


if __name__ == "__main__":
    main()
