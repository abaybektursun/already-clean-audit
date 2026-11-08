"""SPEC11 SHAREDBIRTH job, one birth checkpoint per call (studies/reuse/SPEC11_sharedbirth.md §2-§5, §9). A subclass of
jobs.reuse_sharedinputs.Job: SPEC10's instrument is used unchanged (the captures and the direct attribution, D' ranked with B_t excluded, SPEC4's
energy-matched random', the arms identity/own/crossO/crossK_c<k>/random on J and Kown_c<k>/reverse_c<k> on K, each one jobs.reuse_ablate.Job.evaluate
call that writes falls/<case>__<arm>.npz at once, samples/cases.json), and so are ā and B_t recomputed at this checkpoint (JB.Job.background), the
parent's known number and completeness (JR.Job.known) and SPEC9's capture identity (JS.Job.identity). What changes is the checkpoint, the cases read
and their K, and the known number's reference.

    python -m infra.aws_pg.jobs.reuse_sharedbirth --stage sharedbirth --t T_12 --revision stage1-step320000 --items <s3 birth_items.json> \
        --corpus <s3> --birth-verdict <s3 verdict_birth.json> --cases-t15 <s3 SPEC10 cases.json> --birth-bank <s3 SPEC4 birth bank>/t12 --out <dir>

Cases (§2, §3; cases.json, written before any forward). SPEC10's case list (--cases-t15; asserted equal to jobs.reuse_sharedinputs.case_list on the
items and SPEC4's live flags at birth). A case's t_b is its items' birth (J's birth checkpoint index). K exists at t_b when K's own case (the case of
the items on the same pole whose c is K, primary or fresh, live or not) has birth at or before t_b, or when K has no own case (K is only an O*, older
than its carrier's J by construction). K born later is dropped and banked under K_dropped; a case left with no K is dropped and banked under
dropped_cases. This load reads the listed cases whose t_b is its checkpoint. On the real items: 47 primary and 36 fresh cases, 101 K reads, births
3, 4, 4, 10, 8, 24, 30 over T_7..T_13; 34 K and 24 SPEC10 cases dropped (studies/reuse/results/review11/BUILD_NOTES_sharedbirth.md: the brief's
count, 48 and 102, keeps one K born after its case's t_b).
Before any forward: the K-validity gate over the whole list (SPEC10's checks on every kept K, and §5's existence condition from the items' births;
failing input: every K of SPEC10's list put back, the K born later included, which must give a problem in every case that had a K dropped), asserted;
SPEC4's birth-load bank of this checkpoint (--birth-bank: metrics.json and falls/<case>.npz) fetched and asserted: its metrics.json is the birth_load
of this checkpoint on these items, and every primary case read here has its npz with own_direct_m50 and cross_direct_m50 on its capped J contexts.
Then the parent's known number and completeness (asserted), the capture identity (asserted), ā and B_t at this checkpoint, and per case SPEC10's
arms with the case's existing K. The cancel rule (SPEC4's, on own' and random') is recorded per case at this load.
metrics.json (rewritten after every case): SPEC10's per-case record (falls with npz paths, I_O, I_K per K, d, cancel, the D' overlaps) plus t_b, each
K's birth and the case's dropped K, and for primary cases known {own, cross, spec4_own, spec4_cross}; gates and records: known_number, completeness,
capture_identity, k_validity, known_spec4 (median |fall here - SPEC4's birth-load fall| over own' and cross_O of this load's primary cases, line 0.01;
failing input the next case's SPEC4 falls; not read at a load with no primary case; recorded, the verdict pools the loads and decides), exclusion,
identity_arm, high_background, memory, timing, counts. Statistics are the verdict's (studies/reuse/scripts/verdict_sharedbirth.py).
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
from infra.aws_pg.jobs import reuse_sharedinputs as JX
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

M = JX.M
CAP = JX.CAP
KNOWN_LINE = JX.KNOWN_LINE
N_SAMPLE_CASES = JX.N_SAMPLE_CASES
LOADS = ("T_7", "T_8", "T_9", "T_10", "T_11", "T_12", "T_13")


# ------------------------------------------------------------------------------------------ pure helpers (the dry run names these)
def own_births(items: dict) -> dict:
    """{(pole, c): birth} over every case of the items: the birth of the cluster whose own case it is."""
    return {(x["pole"], x["c"]): x["birth"] for x in items["primary"] + items["fresh"]}


def exists(b, t_b: int) -> bool:
    """§2: K exists at t_b when its own case was born at or before t_b, or it has no own case (b None: only an O*)."""
    return b is None or b <= t_b


def birth_list(spec10: list, items: dict) -> list:
    """Every SPEC10 case with t_b (its items' birth), K = its K that exist at t_b (each with its birth) and K_dropped = the K born later."""
    by, kb = {x["job"]: x for x in items["primary"] + items["fresh"]}, own_births(items)
    out = []
    for c in spec10:
        tb = by[c["job"]]["birth"]
        K = [{**k, "birth": kb.get((k["pole"], k["c"]))} for k in c["K"]]
        out.append({**c, "t_b": tb, "K": [k for k in K if exists(k["birth"], tb)], "K_dropped": [k for k in K if not exists(k["birth"], tb)]})
    return out


def born_later(c: dict, Ks: list, kb: dict) -> list:
    return [f"K c{k['c']} was born at T_{kb[(c['pole'], k['c'])]}, after t_b T_{c['t_b']}" for k in Ks if not exists(kb.get((c["pole"], k["c"])), c["t_b"])]


def kb_validity(allc: list, items: dict, live: dict) -> dict:
    """The gate record over the whole list: SPEC10's K checks and §2's existence condition (from the items' births) on every listed case's K; the
    failing input (every K of SPEC10's list put back, the K born later included) a problem in every case that had a K dropped."""
    ctx, kb = JX.cluster_contexts(items), own_births(items)
    lc = {(x["pole"], cl) for x in items["primary"] + items["fresh"] if live[x["job"]] for cl in (x["c"], x["o_star"])}
    listed = [c for c in allc if c["K"]]
    bad = {c["job"]: JX.k_problems(c, c["K"], lc, ctx) + born_later(c, c["K"], kb) for c in listed}
    fi = {c["job"]: born_later(c, c["K"] + c["K_dropped"], kb) for c in allc}
    n_bad, n_fi, n_dropped = sum(bool(v) for v in bad.values()), sum(bool(v) for v in fi.values()), sum(bool(c["K_dropped"]) for c in allc)
    return {"value": n_bad, "pass": n_bad == 0, "n_cases": len(listed), "n_K": sum(len(c["K"]) for c in listed), "problems": [f"{j}: {p}" for j, v in bad.items() for p in v][:20],
            "failing_input": {"input": "every K of SPEC10's list put back (the K born after t_b included)", "value": n_fi, "n_cases_with_a_dropped_K": n_dropped},
            "failing_input_failed": n_fi == n_dropped > 0,
            "rule": "every kept K is a cluster (c or o_star) of a live case on the case's pole, not J and not O* by index, read on its own cluster's capped "
                    "contexts, no index twice, and exists at t_b: its own case's birth at or before t_b, or no own case"}


def known_spec4(recs: list) -> dict:
    """recs [{job, own, cross, spec4_own, spec4_cross}] in case-list order: median |fall here - SPEC4's birth-load fall| over own' and cross_O; failing
    input the next case's SPEC4 falls (cyclic). Not read when the load has no primary case."""
    if not recs:
        return {"read": False, "value": None, "pass": None, "n_cases": 0, "reason": "no primary case is read at this load"}
    n = len(recs)
    ours = [r[x] for r in recs for x in ("own", "cross")]
    theirs = [r[f"spec4_{x}"] for r in recs for x in ("own", "cross")]
    nxt = [recs[(i + 1) % n][f"spec4_{x}"] for i in range(n) for x in ("own", "cross")]
    v, w = float(np.median(np.abs(np.subtract(ours, theirs)))), float(np.median(np.abs(np.subtract(ours, nxt))))
    return {"read": True, "value": v, "pass": v <= KNOWN_LINE, "line": KNOWN_LINE, "n_cases": n,
            "failing_input": {"input": "the next case's SPEC4 falls (case-list order at this load, cyclic; the case itself when it is alone)", "value": w},
            "failing_input_failed": w > KNOWN_LINE, "identity": {"value": 0.0, "reason": "SPEC4's falls against themselves"},
            "rule": "median over this load's primary cases of |fall here - SPEC4's birth-load fall| over own' and cross_O, at most 0.01 (recorded here; the "
                    "verdict pools the loads and decides)"}


class Job(JX.Job):
    def setup(self, out, a, items: dict, device: str, W: int, n_layers: int, dtype: str, live: dict, refs: dict):
        """Everything but the model: the bank directories, the case list of this load and metrics.json (the dry run calls this with no model). `refs`:
        cases_t15 (SPEC10's cases.json, local) and birth_bank (SPEC4's birth-load directory of this checkpoint, local: metrics.json, falls/)."""
        self.out, self.a, self.items, self.device, self.W = pathlib.Path(out), a, items, device, W
        for d in ("", "falls", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.hw = a.harvest_window
        assert a.stage == "sharedbirth" and a.t in LOADS, (a.stage, a.t)
        self.spec = {"cap": CAP, "id_fail_arm": f"own_direct_m{M}"}
        self.live, self.bank = live, pathlib.Path(refs["birth_bank"])
        self.by_job = {x["job"]: x for x in items["primary"] + items["fresh"]}
        self.spec10 = json.loads(pathlib.Path(refs["cases_t15"]).read_text())
        self.all = birth_list(self.spec10["cases"], items)
        self.cases = [c for c in self.all if c["K"] and c["t_b"] == self.t_index]
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "birth_verdict_sha256": a.birth_verdict_sha256,
                  "cases_t15_sha256": a.cases_t15_sha256, "harvest_window": self.hw, "window": RA.LENGTH, "started_utc": NH.utc(),
                  "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": n_layers, "width": W, "dtype": dtype, "m": M, "cap": CAP,
                  "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is", "scale": "free (reuse_ablate.readout; no held scale)",
                  "exclusion_rule": "B_t of this checkpoint (high_background.json) removed before top_m ranks every set and before random_matched matches random'",
                  "job_fall_rule": "mean over contexts with a_P > 0", "existence_rule": exists.__doc__,
                  "arms": {"on J": "identity, own D'(J), crossO D'(O*), crossK_c<k> D'(K), random random'", "on K": "Kown_c<k> D'(K), reverse_c<k> D'(J)"},
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "refs": {k: str(v) for k, v in refs.items()}, "cases_file": "cases.json",
                  "cases": {}}
        self.lin, self.id_rows, self.id_fail_rows = [], [], []
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0, "unexcluded_n_in_B": 0, "unexcluded_n_sets_with_B": 0, "random_n_sets": 0, "random_n_in_B": 0}
        self.write()

    # -------------------------------------------------------------------------------------- before any forward
    def validity(self):
        """SPEC10's list re-derived; cases.json (this load's cases with K kept and dropped, its dropped cases, the whole list's counts by load); the
        K-validity gate over the whole list, asserted."""
        assert self.spec10["birth_verdict_sha256"] == self.a.birth_verdict_sha256, "SPEC10's case list was built on another birth verdict"
        assert self.spec10["cases"] == JX.case_list(self.items, self.live), "SPEC10's banked case list differs from case_list on these items and live flags"
        listed = [c for c in self.all if c["K"]]
        by_load = {f"T_{t}": {"primary": sum(c["set"] == "primary" and c["t_b"] == t for c in listed), "fresh": sum(c["set"] == "fresh" and c["t_b"] == t for c in listed),
                              "K_reads": sum(len(c["K"]) for c in listed if c["t_b"] == t), "K_dropped": sum(len(c["K_dropped"]) for c in self.all if c["t_b"] == t),
                              "cases_dropped": sum(not c["K"] and c["t_b"] == t for c in self.all)} for t in sorted({c["t_b"] for c in self.all})}
        here = by_load.get(self.a.t)
        NH.dump(self.out / "cases.json", {"rule": birth_list.__doc__ + " " + exists.__doc__, "spec10_rule": JX.case_list.__doc__, "t": self.a.t, "t_index": self.t_index,
                                          "cases_t15_sha256": self.a.cases_t15_sha256, "birth_verdict_sha256": self.a.birth_verdict_sha256, "n": here, "by_load": by_load,
                                          "cases": self.cases, "dropped_cases": [c for c in self.all if not c["K"] and c["t_b"] == self.t_index]})
        self.m["k_validity"] = kb_validity(self.all, self.items, self.live)
        self.m["n_cases"] = {s: sum(c["set"] == s for c in self.cases) for s in ("primary", "fresh")}
        self.m["n_neurons"] = {s: len({c["neuron"] for c in self.cases if c["set"] == s}) for s in ("primary", "fresh")}
        self.m["by_load"] = by_load
        self.write()
        kv = self.m["k_validity"]
        assert kv["pass"], f"K validity failed: {kv['problems']}"
        assert kv["failing_input_failed"], f"K validity's failing input did not fail in every case with a dropped K: {kv['failing_input']}"
        assert self.cases, f"no listed case is born at {self.a.t}"

    def check_refs(self):
        """SPEC4's birth-load bank of this checkpoint: the birth_load of these items at this t, done; every primary case read here has its npz with own'
        and cross' on its capped J contexts."""
        bm = json.loads((self.bank / "metrics.json").read_text())
        assert (bm["stage"], bm["arms"], bm["t"], bm["status"], bm["items_sha256"]) == ("birth", "birth_load", self.a.t, "done", self.a.items_sha256), \
            ("SPEC4's bank is not this checkpoint's birth load on these items", bm["stage"], bm["arms"], bm["t"], bm["status"], bm["items_sha256"])
        self.spec4 = {}
        for c in self.cases:
            if c["set"] != "primary":
                continue
            j = self.by_job[c["job"]]
            assert c["job"] in bm["jobs"], (c["job"], "SPEC4 did not read this primary case at this load")
            p = self.bank / "falls" / f"{JX.key_of(j)}.npz"
            assert p.exists(), (c["job"], f"SPEC4's birth-load bank lacks {p.name}")
            z = np.load(p)
            arms = [str(x) for x in z["arm"]]
            assert np.array_equal(z["tok"], JX.capped(j["J_tok"])), (p.name, "SPEC4's contexts differ from the items'")
            self.spec4[c["job"]] = {x: RA.job_fall(z["a_before"], z["a_after"][arms.index(f"{x}_direct_m{M}")])["mean"] for x in ("own", "cross")}
        self.spec6 = self.spec4      # SPEC10's case() reads the known number's reference falls under this name
        self.m["refs_checked"] = {"n_cases": len(self.spec4), "bank": str(self.bank),
                                  "rule": "SPEC4's metrics.json is the birth_load of this t on these items; every primary case read here has falls/<case>.npz with "
                                          "own_direct_m50 and cross_direct_m50 on its capped J contexts"}
        self.write()

    # -------------------------------------------------------------------------------------- one case
    def case(self, c: dict, sample: bool):
        super().case(c, sample)
        rec = self.m["cases"][c["job"]]
        if "known" in rec:
            k = rec["known"]
            rec["known"] = {"own": k["own"], "cross": k["cross"], "spec4_own": k["spec6_own"], "spec4_cross": k["spec6_cross"]}
        rec.update({"t_b": c["t_b"], "K_birth": {f"c{k['c']}": k["birth"] for k in c["K"]}, "K_dropped": [{"c": k["c"], "birth": k["birth"]} for k in c["K_dropped"]]})

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.validity()
        self.check_refs()
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.identity()
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P = [c for c in self.cases if c["set"] == "primary"]
        self.m["n_to_read"] = {"primary": len(P), "fresh": len(self.cases) - len(P), "K_reads": sum(len(c["K"]) for c in self.cases)}
        self.m["n_done"] = {k: 0 for k in self.m["n_to_read"]}
        samp = [c["job"] for c in self.cases[:N_SAMPLE_CASES]]
        self.m["samples"] = {"cases": samp, "file": "samples/cases.json", "arms": "samples/arms.jsonl"}
        if not P:
            self.m["known_spec4"] = known_spec4([])
        for c in self.cases:
            self.case(c, c["job"] in samp)
            self.m["n_done"][c["set"]] += 1
            self.m["n_done"]["K_reads"] += len(c["K"])
            if c["set"] == "primary" and self.m["n_done"]["primary"] == len(P):
                self.m["known_spec4"] = known_spec4([{"job": x["job"], **self.m["cases"][x["job"]]["known"]} for x in P])
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
    ap.add_argument("--stage", required=True, choices=["sharedbirth"])
    ap.add_argument("--t", required=True, choices=list(LOADS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--birth-verdict", required=True, help="SPEC4's verdict_birth.json (local or s3): the live flags at birth")
    ap.add_argument("--cases-t15", required=True, help="SPEC10's cases.json (local or s3): the case list with each K's carriers")
    ap.add_argument("--birth-bank", required=True, help="SPEC4's birth-load directory of this checkpoint (local or s3, ending in t<nn>): metrics.json, falls/")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    tt = f"t{int(a.t.split('_')[1]):02d}"
    bank_src = a.birth_bank.rstrip("/")
    assert bank_src.endswith("/" + tt), f"--birth-bank {a.birth_bank} is not the {tt} load"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    bv_path = fetch(a.birth_verdict, "reuse_sharedbirth_verdict_birth.json")
    a.birth_verdict_sha256 = hashlib.sha256(pathlib.Path(bv_path).read_bytes()).hexdigest()
    live = JX.live_flags(json.loads(pathlib.Path(bv_path).read_text()))
    c15 = fetch(a.cases_t15, "reuse_sharedbirth_cases_t15.json")
    a.cases_t15_sha256 = hashlib.sha256(pathlib.Path(c15).read_bytes()).hexdigest()
    if bank_src.startswith("s3://"):
        bank = fetch(bank_src + "/metrics.json", f"reuse_sharedbirth_spec4_{tt}/metrics.json").parent
        fetch(bank_src + "/falls/", f"reuse_sharedbirth_spec4_{tt}/falls", recursive=True)
    else:
        bank = pathlib.Path(bank_src)
    items_path = fetch(a.items, "reuse_sharedbirth_items.json")
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
    job = Job(a.out, a, items, model, tok, corpus, "cuda", live, {"cases_t15": c15, "birth_bank": bank})
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
    print("REUSE_SHAREDBIRTH_OK", json.dumps({"t": a.t, "cases": job.m["n_cases"], "K_reads": job.m["n_to_read"]["K_reads"]}), flush=True)


if __name__ == "__main__":
    main()
