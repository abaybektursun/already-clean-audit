"""SPEC10 SHAREDINPUTS job, one load at T_15 (studies/reuse/SPEC10_sharedinputs.md §2-§5, §9). A subclass of jobs.reuse_birth.Job: the known
number and completeness (jobs.reuse_ablate.Job.known), ā and B_t (JB.Job.background: high_background.json), the capture and the direct attribution
(JB.Job.side: unite_route.capture + reuse_ablate.direct_A), the ranking with B_t excluded (JB.Job.rank: reuse_birth.top_m_excluded), the top-200
bank (reuse_birth.top_bank), SPEC4's energy-matched random' (reuse_birth.random_matched_excluded: matched to D'(O*) in |ā|, outside D'(J) and
D'(O*), as JB.Job.case builds it) and the mean-replacement ablation with its banking (jobs.reuse_ablate.Job.evaluate: reuse_ablate.readout under
reuse_ablate.replaced, the free scale) are SPEC4's code, imported unchanged. The capture identity is SPEC9's (jobs.reuse_sameweights.Job.identity),
the exclusion counter SPEC9's (reuse_sameweights.excl_add).

    python -m infra.aws_pg.jobs.reuse_sharedinputs --stage sharedinputs --t T_15 --revision stage1-step1413814 --items <s3 birth_items.json> \
        --corpus <s3> --birth-verdict <s3 verdict_birth.json> --spec6-bank <s3 SPEC6 T_15 falls dir> --out <dir>

Cases (§3; cases.json, written before any forward). Live: SPEC4's live flags at birth (verdict_birth.json -> cases.primary_birth and fresh_birth, not
cancelled). For each live case A in item order (the 150 primary, then the 93 fresh): every cluster index K that is the J (c) or the O* (o_star) of
another live case on A's pole, K != A's c and K != A's o_star, once per index (ascending); cases with no K are dropped. K's contexts are its own
cluster's: its carrier's capped J_tok or O_tok (the items hold one context list per pole and cluster index; asserted). K is a sibling when it is the
J of a live case whose o_star equals A's. On the real items: 64 primary cases on 28 neurons and 43 fresh on 17, each with 1 or 2 K.
Before any forward: SPEC6's T_15 falls (--spec6-bank: <case>__own__g1__free.npz and <case>__cross__g1__free.npz) are fetched and asserted present
for every primary case of the list, of its pole, own and cross under g1 at the free scale, on its capped J contexts; the K-validity gate is read and
asserted (every K a cluster of a live case on A's pole, not J and not O* by index, on its own contexts, no index twice; failing input: every K
replaced by J, which must fail for every case). Then the parent's known number and completeness (asserted), the capture identity (SPEC9's: the first
primary case with L >= 1 and 20 capped J + O* contexts captured twice, a_P equal to 1e-5; failing input the hook on L-1; asserted after it is
written), ā over layers 0..30 on UNITE's e1 positions and B_t (abar.npz, background.json, high_background.json).
Per case, every set capped at 12 contexts by rank, m 50, direct ranking, B_t excluded, mean replacement (ā), free scale:
  captures of J, O* and each K on their own contexts; D'(X) = the top 50 of the summed direct attribution outside B_t; random' as SPEC4.
  on J's contexts: identity, own (D'(J)), crossO (D'(O*)), crossK_c<k> (D'(K), one arm per K), random (random');
  on each K's contexts: Kown_c<k> (D'(K), K's positive control), reverse_c<k> (D'(J)).
  Each arm is one jobs.reuse_ablate.Job.evaluate call (a clean readout, then the ablated one) that writes falls/<case>__<arm>.npz at once, in SPEC4's
  layout with one arm (tok, arm, ranking, m, replace, a_before, g_before, u_before, a_after [1, n], g_after, u_after, set_layer [1, 100], set_neuron,
  set_r, set_A, set_a, set_Ag, lin_*), plus job, pole, read_c (the cluster whose contexts are read), set_c (the cluster whose D' is ablated; -1 for
  identity and random), set_in_B [1, 100] (B_t flags; all False, asserted) and, for a ranked set, its top-200 bank (top_idx, top_A, top_abar; no
  exclusion), from which the verdict rebuilds the unexcluded ranking.
metrics.json (rewritten after every case): per case its K (index, contexts, sibling, carriers), every arm's job fall (mean over contexts with a_P > 0)
with its npz path, I_O = crossO/own, I_K per K = crossK/own, their mean, d = I_O - I_K, SPEC4's cancel flags, the D' overlaps, and for primary cases
SPEC6's own and cross; gates and records: known_number, completeness (the parent's), capture_identity, k_validity, known_spec6 (median |fall here -
SPEC6's fall| over own and crossO of the primary cases, line 0.01; failing input the next case's SPEC6 falls; recorded, the verdict decides),
exclusion (no D' or random' member in B_t; failing input the unexcluded ranking reuse_ablate.top_m of every ranked set against B_t), identity_arm (the
parent's: the empty-set ablation moves no a_P by 1e-3; failing input own'), high_background, memory, timing, counts. samples/cases.json: the first
three primary cases of the list: every J and K context's prompt, marked token, a_P before and after each arm read on it, and every set. Evaluate also
appends samples/arms.jsonl (every non-identity arm's contexts with prompts), as SPEC4's loads did.
Statistics over the bank (X1, X1f, the gates' reads, the reported items) are the verdict's (studies/reuse/scripts/verdict_sharedinputs.py).
Fail loud: no try/except, no fallback; asserts carry their reason.
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
from infra.aws_pg.jobs import reuse_sameweights as JS
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

M = RB.M
CAP = JB.CAP
KNOWN_LINE = 0.01
N_SAMPLE_CASES = 3


# ------------------------------------------------------------------------------------------ pure helpers (the dry run and the verdict's register name these)
def key_of(j: dict) -> str:
    return f"{j['pole']}__c{j['c']}"


def capped(toks) -> np.ndarray:
    return np.asarray(toks, np.int64)[:CAP]


def neuron_of(pole: str) -> str:
    return pole[:-1]


def j_arms(Ks: list) -> list:
    """The arms read on J's contexts, in reading order."""
    return ["identity", "own", "crossO"] + [f"crossK_c{k}" for k in Ks] + ["random"]


def k_arms(k: int) -> list:
    """The arms read on K's contexts."""
    return [f"Kown_c{k}", f"reverse_c{k}"]


def arm_key(name: str) -> str:
    return "identity" if name == "identity" else f"{name}_direct_m{M}"


def live_flags(bv: dict) -> dict:
    """SPEC4's live flags at birth: {job: not cancelled} over cases.primary_birth and cases.fresh_birth."""
    return {r["job"]: not r["cancelled"] for k in ("primary_birth", "fresh_birth") for r in bv["cases"][k]}


def cluster_contexts(items: dict) -> dict:
    """{(pole, cluster index): capped contexts as a list}: a case's J_tok are cluster c's, its O_tok cluster o_star's; one list per cluster."""
    out = {}
    for x in items["primary"] + items["fresh"]:
        for cl, t in ((x["c"], x["J_tok"]), (x["o_star"], x["O_tok"])):
            t = capped(t).tolist()
            assert out.setdefault((x["pole"], cl), t) == t, (x["job"], cl, "the items hold two context lists for one cluster")
    return out


def case_list(items: dict, live: dict) -> list:
    """SPEC10 §3: per live case A (item order, primary then fresh) every K = the c or o_star of another live case on A's pole, K not A's c or o_star,
    once per index (ascending); cases with no K dropped."""
    cases = items["primary"] + items["fresh"]
    by = {x["job"]: x for x in cases}
    assert len(by) == len(cases), "a job id repeats"
    miss = [x["job"] for x in cases if x["job"] not in live]
    assert not miss, ("the birth verdict lacks cases", miss[:5])
    pset = {x["job"] for x in items["primary"]}
    ctx = cluster_contexts(items)
    out = []
    for a in cases:
        if not live[a["job"]]:
            continue
        K = {}
        for b in cases:
            if b["pole"] != a["pole"] or b["job"] == a["job"] or not live[b["job"]]:
                continue
            for side, cl in (("J", b["c"]), ("O", b["o_star"])):
                if cl not in (a["c"], a["o_star"]):
                    K.setdefault(cl, []).append([b["job"], side])
        if not K:
            continue
        out.append({"job": a["job"], "set": "primary" if a["job"] in pset else "fresh", "pole": a["pole"], "neuron": neuron_of(a["pole"]), "layer": a["layer"],
                    "c": a["c"], "o_star": a["o_star"],
                    "K": [{"c": cl, "pole": a["pole"], "carriers": src, "tok": ctx[(a["pole"], cl)],
                           "sibling": any(s == "J" and by[jb]["o_star"] == a["o_star"] for jb, s in src)} for cl, src in sorted(K.items())]})
    return out


def k_problems(c: dict, Ks: list, live_clusters: set, ctx: dict) -> list:
    out = [] if len({k["c"] for k in Ks}) == len(Ks) else ["two K name one cluster"]
    if not Ks:
        out.append("no K")
    for k in Ks:
        tag = f"K c{k['c']}"
        if k["pole"] != c["pole"]:
            out.append(f"{tag} is on pole {k['pole']}, not {c['pole']}")
        if k["c"] == c["c"]:
            out.append(f"{tag} is J by index")
        if k["c"] == c["o_star"]:
            out.append(f"{tag} is O* by index")
        if (k["pole"], k["c"]) not in live_clusters:
            out.append(f"{tag} is not a cluster of a live case on the pole")
        if list(k["tok"]) != ctx.get((k["pole"], k["c"])):
            out.append(f"{tag}'s contexts are not its own cluster's")
    return out


def k_validity(clist: list, items: dict, live: dict) -> dict:
    """The gate record: no problem in any case's K; the failing input (every K replaced by J: J's index, J's contexts) a problem in every case."""
    ctx = cluster_contexts(items)
    lc = {(x["pole"], cl) for x in items["primary"] + items["fresh"] if live[x["job"]] for cl in (x["c"], x["o_star"])}
    bad = {c["job"]: k_problems(c, c["K"], lc, ctx) for c in clist}
    fi = {c["job"]: k_problems(c, [{"c": c["c"], "pole": c["pole"], "tok": ctx[(c["pole"], c["c"])]}], lc, ctx) for c in clist}
    n_bad, n_fi = sum(bool(v) for v in bad.values()), sum(bool(v) for v in fi.values())
    return {"value": n_bad, "pass": n_bad == 0, "n_cases": len(clist), "problems": [f"{j}: {p}" for j, v in bad.items() for p in v][:20],
            "failing_input": {"input": "every K replaced by J (J's cluster index, J's contexts)", "value": n_fi}, "failing_input_failed": n_fi == len(clist) > 0,
            "rule": "every K is a cluster (c or o_star) of a live case on the case's pole, not J and not O* by index, read on its own cluster's capped "
                    "contexts, no index twice"}


def known_spec6(recs: list) -> dict:
    """recs [{job, own, cross, spec6_own, spec6_cross}] in case-list order: median |fall here - SPEC6's| over own' and cross_O; failing input the next
    case's SPEC6 falls (cyclic)."""
    n = len(recs)
    ours = [r[x] for r in recs for x in ("own", "cross")]
    theirs = [r[f"spec6_{x}"] for r in recs for x in ("own", "cross")]
    nxt = [recs[(i + 1) % n][f"spec6_{x}"] for i in range(n) for x in ("own", "cross")]
    v, w = float(np.median(np.abs(np.subtract(ours, theirs)))), float(np.median(np.abs(np.subtract(ours, nxt))))
    return {"value": v, "pass": v <= KNOWN_LINE, "line": KNOWN_LINE, "n_cases": n, "failing_input": {"input": "the next case's SPEC6 falls (case-list order, cyclic)", "value": w},
            "failing_input_failed": w > KNOWN_LINE, "identity": {"value": 0.0, "reason": "SPEC6's falls against themselves"},
            "rule": "median over the primary cases of |fall here - SPEC6's T_15 free-scale fall| over own' and cross_O, at most 0.01 (recorded here; the verdict decides)"}


def overlap(A: list, B: list) -> float:
    return len(set(map(tuple, A)) & set(map(tuple, B))) / M


class Job(JB.Job):
    identity = JS.Job.identity          # SPEC9's capture identity, unchanged

    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str, live: dict, spec6):
        self.model, self.tok = model, tok
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.rows = UR.Rows(model)
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P and the coefficients assume silu"
        self.setup(out, a, items, device, int(self.layers[0].mlp.gate_proj.weight.shape[0]), len(self.layers), str(next(model.parameters()).dtype), live, spec6)

    def setup(self, out, a, items: dict, device: str, W: int, n_layers: int, dtype: str, live: dict, spec6):
        """Everything but the model: the bank directories, the case list and metrics.json (the dry run calls this with no model)."""
        self.out, self.a, self.items, self.device, self.W = pathlib.Path(out), a, items, device, W
        for d in ("", "falls", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.hw = a.harvest_window
        assert a.stage == "sharedinputs" and a.t == "T_15", (a.stage, a.t)
        self.spec = {"cap": CAP, "id_fail_arm": f"own_direct_m{M}"}
        self.live, self.spec6_dir = live, pathlib.Path(spec6)
        self.by_job = {x["job"]: x for x in items["primary"] + items["fresh"]}
        self.cases = case_list(items, live)
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "birth_verdict_sha256": a.birth_verdict_sha256,
                  "harvest_window": self.hw, "window": RA.LENGTH, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": n_layers,
                  "width": W, "dtype": dtype, "m": M, "cap": CAP, "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is",
                  "scale": "free (reuse_ablate.readout; no held scale)",
                  "exclusion_rule": "B_t (high_background.json) removed before top_m ranks every set and before random_matched matches random'",
                  "job_fall_rule": "mean over contexts with a_P > 0",
                  "arms": {"on J": "identity, own D'(J), crossO D'(O*), crossK_c<k> D'(K), random random'", "on K": "Kown_c<k> D'(K), reverse_c<k> D'(J)"},
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "refs": {"spec6_falls": str(spec6)}, "cases_file": "cases.json", "cases": {}}
        self.lin, self.id_rows, self.id_fail_rows = [], [], []
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0, "unexcluded_n_in_B": 0, "unexcluded_n_sets_with_B": 0, "random_n_sets": 0, "random_n_in_B": 0}
        self.write()

    # -------------------------------------------------------------------------------------- before any forward
    def check_refs(self):
        """SPEC6's T_15 falls hold own and cross (g1, free) of every primary case of the list, of its pole, on its capped J contexts."""
        self.spec6 = {}
        for c in self.cases:
            if c["set"] != "primary":
                continue
            j = self.by_job[c["job"]]
            rec = {}
            for x in ("own", "cross"):
                p = self.spec6_dir / f"{key_of(j)}__{x}__g1__free.npz"
                assert p.exists(), (c["job"], f"SPEC6's falls lack {p.name}")
                z = np.load(p)
                assert (str(z["pole"]), str(z["arm"]), str(z["exclusion"]), str(z["scale"])) == (j["pole"], x, "g1", "free"), \
                    (p.name, str(z["pole"]), str(z["arm"]), str(z["exclusion"]), str(z["scale"]))
                assert np.array_equal(z["tok"], capped(j["J_tok"])), (p.name, "SPEC6's contexts differ from the items'")
                rec[x] = RA.job_fall(z["a_before"], z["a_after"])["mean"]
            self.spec6[c["job"]] = rec
        self.m["refs_checked"] = {"n_cases": len(self.spec6), "rule": "every primary case of the list has SPEC6's own and cross npz (g1, free), its pole, its capped J contexts"}
        self.write()

    def validity(self):
        NH.dump(self.out / "cases.json", {"rule": case_list.__doc__, "live": "SPEC4's live flags at birth (verdict_birth.json)", "birth_verdict_sha256": self.a.birth_verdict_sha256,
                                          "n": {s: sum(c["set"] == s for c in self.cases) for s in ("primary", "fresh")}, "cases": self.cases})
        self.m["k_validity"] = k_validity(self.cases, self.items, self.live)
        self.m["n_cases"] = {s: sum(c["set"] == s for c in self.cases) for s in ("primary", "fresh")}
        self.m["n_neurons"] = {s: len({c["neuron"] for c in self.cases if c["set"] == s}) for s in ("primary", "fresh")}
        self.write()
        kv = self.m["k_validity"]
        assert kv["pass"], f"K validity failed: {kv['problems']}"
        assert kv["failing_input_failed"], f"K validity's failing input passed for some case: {kv['failing_input']['value']} of {kv['n_cases']}"

    # -------------------------------------------------------------------------------------- one ranked set, one arm, one case
    def ranked(self, score) -> tuple:
        D = self.rank(score)
        JS.excl_add(self.excl, D, RA.top_m(score, M, self.W), self.mask)
        return D, RB.top_bank(score, self.abar, self.W)

    def read_arm(self, key: str, name: str, unit: dict, toks, cap: dict, A, S: list, job: str, read_c: int, set_c: int, top) -> dict:
        """One arm on `toks`: jobs.reuse_ablate.Job.evaluate writes falls/<key>__<name>.npz at once."""
        arm = JB.arm(name, S)
        assert arm["arm"] == arm_key(name), (arm["arm"], name)
        extra = {"set_in_B": RB.set_flags([arm], self.mask), "job": job, "pole": unit["unit"], "read_c": int(read_c), "set_c": int(set_c)}
        assert not extra["set_in_B"].any(), (key, name, "a set holds a B_t neuron")
        if top is not None:
            extra.update({"top_idx": top["idx"], "top_A": top["A"], "top_abar": top["abar"]})
        res = self.evaluate(f"{key}__{name}", unit, toks, cap, A, None, [arm], linear=False, extra=extra)
        return {**res[arm["arm"]], "read_c": int(read_c), "set_c": int(set_c), "n_set": len(S), "path": f"falls/{key}__{name}.npz"}

    def case(self, c: dict, sample: bool):
        t0 = time.time()
        j = self.by_job[c["job"]]
        unit = UR.pole_unit(c["pole"])
        L = unit["layer"]
        assert L == j["layer"] == c["layer"], (c["job"], L, j["layer"])
        Jt, Ot = capped(j["J_tok"]), capped(j["O_tok"])
        capJ, AJ = self.side(Jt, unit)
        _, AO = self.side(Ot, unit)
        DJ, topJ = self.ranked(AJ.sum(0))
        DO, topO = self.ranked(AO.sum(0))
        Ks = []
        for k in c["K"]:
            tk = np.asarray(k["tok"], np.int64)
            capK, AK = self.side(tk, unit)
            DK, topK = self.ranked(AK.sum(0))
            Ks.append((k["c"], tk, capK, AK, DK, topK))
        R = RB.random_matched_excluded(DO, DJ, self.abar, self.mask)
        self.excl["random_n_sets"] += 1
        self.excl["random_n_in_B"] += int(RB.in_B(R, self.mask).sum())
        key = key_of(j)
        F = {}
        read = lambda name, toks, cap, A, S, rc, sc, top: F.__setitem__(name, self.read_arm(key, name, unit, toks, cap, A, S, j["job"], rc, sc, top))
        read("identity", Jt, capJ, AJ, [], j["c"], -1, None)
        read("own", Jt, capJ, AJ, DJ, j["c"], j["c"], topJ)
        read("crossO", Jt, capJ, AJ, DO, j["c"], j["o_star"], topO)
        for k, _, _, _, DK, topK in Ks:
            read(f"crossK_c{k}", Jt, capJ, AJ, DK, j["c"], k, topK)
        read("random", Jt, capJ, AJ, R, j["c"], -1, None)
        for k, tk, capK, AK, DK, topK in Ks:
            read(f"Kown_c{k}", tk, capK, AK, DK, k, k, topK)
            read(f"reverse_c{k}", tk, capK, AK, DJ, k, j["c"], topJ)
        assert list(F) == j_arms([k for k, *_ in Ks]) + [n for k, *_ in Ks for n in k_arms(k)], list(F)
        own = F["own"]["mean"]
        I_K = {f"c{k}": JR.ratio(F[f"crossK_c{k}"]["mean"], own) for k, *_ in Ks}
        I_O = JR.ratio(F["crossO"]["mean"], own)
        I_Km = float(np.mean(list(I_K.values())))
        rec = {"set": c["set"], "pole": c["pole"], "neuron": c["neuron"], "layer": L, "c": j["c"], "o_star": j["o_star"], "n_J": int(len(Jt)), "n_O": int(len(Ot)),
               "K": [{"c": k["c"], "n": len(k["tok"]), "sibling": k["sibling"], "carriers": k["carriers"]} for k in c["K"]], "falls": F,
               "I_O": I_O, "I_K": I_K, "I_K_mean": I_Km, "d": I_O - I_Km, "cancel": RA.cancel(own, F["random"]["mean"], F["own"]["n_pos"]),
               "s": {"J_O": overlap(DJ, DO), "J_K": {f"c{k}": overlap(DJ, DK) for k, *_, DK, _ in Ks}, "O_K": {f"c{k}": overlap(DO, DK) for k, *_, DK, _ in Ks}}}
        if c["set"] == "primary":
            s6 = self.spec6[c["job"]]
            rec["known"] = {"own": own, "cross": F["crossO"]["mean"], "spec6_own": s6["own"], "spec6_cross": s6["cross"]}
        if sample:
            self.sample(c, j, F, {"J": DJ, "O": DO, "random": R, **{f"K_c{k}": DK for k, *_, DK, _ in Ks}})
        rec["seconds"] = round(time.time() - t0, 1)
        self.m["cases"][c["job"]] = rec

    def sample(self, c: dict, j: dict, F: dict, sets: dict):
        """samples/cases.json: the case's J and K contexts with prompts, a_P before and after every arm read on them (from the npz just written), its sets."""
        p = self.out / "samples" / "cases.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        reads = {}
        for tag, rc in [("J", j["c"])] + [(f"K_c{k['c']}", k["c"]) for k in c["K"]]:
            names = [n for n, v in F.items() if v["read_c"] == rc]
            zs = {n: np.load(self.out / F[n]["path"]) for n in names}
            toks = zs[names[0]]["tok"]
            ctx = [{"tok": int(t), **pr, "a_before": float(zs[names[0]]["a_before"][i]), "a_after": {n: float(zs[n]["a_after"][0, i]) for n in names}}
                   for i, (t, pr) in enumerate(zip(toks, self.prompts(toks)))]
            reads[tag] = {"cluster": int(rc), "arms": names, "falls": {n: F[n]["mean"] for n in names}, "contexts": ctx}
        cur[c["job"]] = {"pole": c["pole"], "layer": c["layer"], "c": c["c"], "o_star": c["o_star"], "K": [{"c": k["c"], "sibling": k["sibling"]} for k in c["K"]],
                         "sets": {x: [[int(l), int(k)] for l, k in S] for x, S in sets.items()}, "reads": reads}
        NH.dump(p, cur)

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.check_refs()
        self.validity()
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.identity()
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P = [c for c in self.cases if c["set"] == "primary"]
        self.m["n_to_read"] = {"primary": len(P), "fresh": len(self.cases) - len(P), "K_reads": sum(len(c["K"]) for c in self.cases)}
        self.m["n_done"] = {k: 0 for k in self.m["n_to_read"]}
        samp = [c["job"] for c in P[:N_SAMPLE_CASES]]
        self.m["samples"] = {"cases": samp, "file": "samples/cases.json", "arms": "samples/arms.jsonl"}
        for c in self.cases:
            self.case(c, c["job"] in samp)
            self.m["n_done"][c["set"]] += 1
            self.m["n_done"]["K_reads"] += len(c["K"])
            if c["set"] == "primary" and self.m["n_done"]["primary"] == len(P):
                self.m["known_spec6"] = known_spec6([{"job": x["job"], **self.m["cases"][x["job"]]["known"]} for x in P])
            self.write()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        e = self.excl
        self.m["exclusion"] = {**e, "pass": e["n_in_B"] == 0 and e["random_n_in_B"] == 0, "failing_input_failed": e["unexcluded_n_in_B"] > 0,
                               "rule": "no member of any D' or random' is in B_t; failing input: the unexcluded top 50 (reuse_ablate.top_m) of every ranked set against B_t"}
        idf = np.array([r["fall"] for r in self.id_rows])
        fl = np.array([r["fall"] for r in self.id_fail_rows])
        self.m["identity_arm"] = {"max_abs_fall": float(np.nanmax(np.abs(idf))), "max_abs_change": float(max(r["change"] for r in self.id_rows)), "n_contexts": int(idf.size),
                                  "tol": RA.ID_TOL, "pass": bool((np.abs(idf) <= RA.ID_TOL).all()), "failing_input": f"own_direct_m{M} in place of the identity arm",
                                  "failing_input_n_over_tol": int((np.abs(fl) > RA.ID_TOL).sum()), "failing_input_n": int(fl.size),
                                  "failing_input_failed": bool((np.abs(fl) > RA.ID_TOL).any())}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()
        assert self.m["identity_arm"]["pass"], f"identity arm failed: {self.m['identity_arm']}"
        assert self.m["identity_arm"]["failing_input_failed"], f"identity arm's failing input passed: {self.m['identity_arm']}"


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["sharedinputs"])
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--birth-verdict", required=True, help="SPEC4's verdict_birth.json (local or s3): the live flags at birth")
    ap.add_argument("--spec6-bank", required=True, help="SPEC6's T_15 falls directory (local or s3): <case>__own__g1__free.npz, <case>__cross__g1__free.npz")
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
    bv_path = fetch(a.birth_verdict, "reuse_sharedinputs_verdict_birth.json")
    a.birth_verdict_sha256 = hashlib.sha256(pathlib.Path(bv_path).read_bytes()).hexdigest()
    live = live_flags(json.loads(pathlib.Path(bv_path).read_text()))
    spec6 = fetch(a.spec6_bank.rstrip("/") + "/", "reuse_sharedinputs_spec6_t15/falls", recursive=True)
    items_path = fetch(a.items, "reuse_sharedinputs_items.json")
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
    job = Job(a.out, a, items, model, tok, corpus, "cuda", live, spec6)
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
    print("REUSE_SHAREDINPUTS_OK", json.dumps({"t": a.t, "cases": job.m["n_cases"]}), flush=True)


if __name__ == "__main__":
    main()
