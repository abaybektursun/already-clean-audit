"""INSTRUMENTS E6c job: the marked-token instrument on the program's real cases (studies/instruments/SPEC_instruments.md, section "E6", part E6c).
One Olmo-3 load at T_15. A subclass of jobs.instruments_e1.Job: the model load, the known activation number and completeness (jobs.reuse_ablate.Job.known),
ā, B_t, the capture and the value tables (jobs.instruments_e1.Job.background), the captures, the sets and the bank format are E1's code, unchanged;
the marked-token write is E5's T2 variant (infra/pg/instruments_e5.readout_marked, unchanged): the value written into down_proj's input at the marked
token of each context only, every other position untouched (its post-hook asserts it on every forward).

    python -m infra.aws_pg.jobs.instruments_e6c --t T_15 --revision stage1-step1413814 --items <s3 e1_items.json> --corpus <s3 corpus> --out <dir>

Items: E1's 150 primary cases (inputs/e1_items.json `primary`; 115 neurons), side J, 12 contexts, m 50, direct ranking. Sets per case as E1 built
them: own D(J), cross D(O*), random (reuse_ablate.random_matched to the cross set) under u = unexcluded and e = B_t excluded.
Arms per case (side J): identity (hooks below the pole, nothing written) and own, cross, random under u and e, each under the values mean and zero,
all with the marked-token write: names "<set>__<value>". The three known cases (positions 0, n // 3 and 2n // 3 of `primary`: one per layer third)
also read E1's free arms ("free__<set>__mean", the value written at every window position, reuse_ablate.readout) and identity_fail (own_e's first
member written at ā at the marked token: the identity gate's failing input).
Order: the three known cases first; then the known-number gate is written to metrics.json before any other case is read; then the other 147.
Known number: E1's I_excluded(mean) over the 150 cases is 0.2944 (results/e1/1/t15/metrics.json, sha256 9e7acee8...); here each known case's
free I_e(mean) = fall(cross_e) / fall(own_e) is re-read and compared with E1's banked value for that case (KNOWN_CASES), tolerance 0.02; failing
input: the banked values rotated by one case. The verdict re-reads the targets from E1's metrics.json.
Gates in metrics.json (the verdict decides; the job asserts the instrument checks): known_number; identity (max over cases and contexts of
|a_P after - a_P before| / max |a_P before| on the identity arm <= 1e-6; failing input identity_fail); marked_write (on the first case, the
down_proj input of the lowest written layer under the marked write differs from the clean input only at (row, marked position, written neuron)
and differs there; failing input: the free write's changes away from the marked position); exclusion (no B_t member in an excluded set); samples.
Bank: e6c/cases/<pole>__c<c>.npz in E1's format (J_tok, J_a_before, J_arms, J_a_after [arms, N], J_U_a, J_U_c, set_<x>_layer/_neuron/_abar/_median/
_draws/_in_B), values.npz, high_background.json, background.json; samples/e6c.jsonl: the three known cases x (own_e, cross_e), mean value, contexts
with the marked token, a_P before, after under the marked write and after under the free write. metrics.json after every case.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_e1 as JE
from infra.aws_pg.jobs import reuse_ablate as JR
from infra.pg import instruments_e5 as E5
from infra.pg import instruments_linear as IL
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

M = RB.M
CAP = JE.CAP
VALUES = ("mean", "zero")
CONS = JE.CONS
CASE_SETS = JE.CASE_SETS
ID_TOL = JE.ID_TOL
KNOWN_I, KNOWN_TOL = 0.2944, 0.02
# E1's banked I_e(mean) of the three known cases: results/e1/1/t15/metrics.json cases[job]["sides"]["J"]["I"]["mean__e"] (sha256 9e7acee8f754...;
# the dry run asserts these equal that file's values)
KNOWN_CASES = {"L04N10274+:c2": 0.30066891935202505, "L11N4690-:c2": 0.28984175989252353, "L22N10231+:c0": -0.0506207554413205}
SAMPLE_SETS = ("own_e", "cross_e")
OLMO = "allenai/Olmo-3-1025-7B"


def known_index(n: int) -> list:
    return [0, n // 3, 2 * n // 3]


def known_gate(got: dict, banked: dict) -> dict:
    """max over the known cases of |I_e(mean) re-read - banked| <= 0.02; failing input: the banked values rotated by one case."""
    ks = list(banked)
    rot = {k: banked[ks[(i + 1) % len(ks)]] for i, k in enumerate(ks)}
    d = {k: abs(got[k] - banked[k]) for k in ks}
    df = {k: abs(got[k] - rot[k]) for k in ks}
    return {"value": max(d.values()), "per_case": {k: {"reread": got[k], "banked": banked[k], "abs_diff": d[k]} for k in ks}, "tol": KNOWN_TOL,
            "pass": max(d.values()) <= KNOWN_TOL, "failing_input": {"input": "the banked values rotated by one case", "value": max(df.values())},
            "failing_input_failed": max(df.values()) > KNOWN_TOL, "identity": None, "anchor": {"I_excluded_mean_E1": KNOWN_I},
            "rule": "each known case's free I_e(mean) re-read within 0.02 of E1's banked value"}


def changed_entries(clean: list, other: list, tokens, toks, bos, hw: int, batch: int, idx: torch.Tensor) -> dict:
    """Per readout batch (rows in readout order): entries of down_proj's input that differ from the clean input at (row, marked position, idx)
    and away from them."""
    order = np.argsort(np.asarray(toks) % hw, kind="stable")
    on = off = 0
    for b, (c, x) in enumerate(zip(clean, other)):
        _, pos = UR.pack(tokens, np.asarray(toks)[order[b * batch:(b + 1) * batch]], RA.LENGTH, bos, hw, c.device)
        rows = torch.arange(len(pos), device=c.device)
        diff = c != x
        on += int(diff[rows[:, None], pos[:, None], idx[None, :]].sum())
        diff[rows[:, None], pos[:, None], idx[None, :]] = False
        off += int(diff.sum())
    return {"on_marked": on, "off_marked": off}


class Job(JE.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str, known_cases: dict = KNOWN_CASES):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "e6c/cases", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert a.t == "T_15", a.t
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.start = RA.window_start(self.bos)
        C = items["primary"]
        self.known_jobs = [C[i]["job"] for i in known_index(len(C))]
        assert list(known_cases) == self.known_jobs, (list(known_cases), self.known_jobs, "the known cases are not the items' positions 0, n // 3, 2n // 3")
        self.known_cases = known_cases
        self.m = {"stage": "e6c", "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "status": "running", "m": M, "cap": CAP, "values": list(VALUES), "side": "J",
                  "job_fall_rule": "mean over contexts with a_P > 0", "write": "the value written into down_proj's input at the marked token only (instruments_e5.readout_marked)",
                  "known_jobs": self.known_jobs, "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "readouts": 0, "readouts_cached": 0,
                  "marked_readouts": 0, "n_to_read": {"cases": len(C)}, "n_done": {"cases": 0}, "gates": {}, "cases": {}}
        self.state = {}
        self.chk = {"id": [], "id_fail": []}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0}
        self.n_samples = 0
        self.write()

    # -------------------------------------------------------------------------------------- readouts
    def readm(self, st: dict, arm: str, S: list, value: str, marked: bool):
        """One readout on side J under the marked write (marked) or E1's free write; cached on set, value and write."""
        key = (tuple(sorted(S)), value, marked, arm == "identity")
        if key in st["cache"]:
            st["after"]["J"][arm] = st["cache"][key]
            self.m["readouts_cached"] += 1
            return
        unit, toks = st["unit"], st["tok"]["J"]
        L = unit["layer"]
        abl = IV.written([], "mean", self.tab, L, self.device, identity=True) if arm == "identity" else IV.written(S, value, self.tab, L, self.device)
        a = (E5.readout_marked if marked else RA.readout)(self.model, toks, unit, abl, **self.kw(RA.LENGTH))["a"]
        st["cache"][key] = st["after"]["J"][arm] = a
        self.m["readouts"] += 1
        self.m["marked_readouts"] += int(marked)
        self.m["forwards"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)

    def marked_probe(self, st: dict) -> dict:
        """The marked_write gate on one case: down_proj's input at the lowest written layer of own_e, clean against marked and against free."""
        S = st["sets"]["J_own_e"]
        unit, toks = st["unit"], st["tok"]["J"]
        L = unit["layer"]
        l0 = min(l for l, _ in S)
        abl = IV.written(S, "mean", self.tab, L, self.device)
        kw = self.kw(RA.LENGTH)
        got = {"x": []}
        h = self.layers[l0].mlp.down_proj.register_forward_hook(lambda m, a, o: got["x"].append(a[0].clone()))
        RA.readout(self.model, toks, unit, None, **kw)
        clean, got["x"] = got["x"], []
        E5.readout_marked(self.model, toks, unit, abl, **kw)
        marked, got["x"] = got["x"], []
        RA.readout(self.model, toks, unit, abl, **kw)
        free = got["x"]
        h.remove()
        idx = abl[l0][0]
        cm = changed_entries(clean, marked, self.tokens, toks, self.bos, self.hw, kw["batch"], idx)
        cf = changed_entries(clean, free, self.tokens, toks, self.bos, self.hw, kw["batch"], idx)
        ok = lambda c: c["off_marked"] == 0 and c["on_marked"] > 0
        return {"value": cm, "pass": ok(cm), "layer": l0, "n_written": int(len(idx)), "failing_input": {"input": "the free write (every window position)", "value": cf},
                "failing_input_failed": not ok(cf), "identity": {"value": cm["off_marked"], "reason": "entries changed away from the marked position"},
                "rule": "on the first case, down_proj's input at the lowest written layer of own_e (mean) differs from the clean input at (row, marked "
                        "position, written neuron) and nowhere else"}

    # -------------------------------------------------------------------------------------- cases
    def case(self, j: dict, known: bool, probe: bool):
        unit = UR.pole_unit(j["pole"])
        assert unit["layer"] == j["layer"], (j["job"], unit["layer"], j["layer"])
        Jt, Ot = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"]))
        capJ, AJ = self.side(Jt, unit)
        _, AO = self.side(Ot, unit)
        c, _ = IL.coefficients(self.rows, capJ, unit, self.W, AJ)
        sJ, sO = AJ.sum(0), AO.sum(0)
        Ju, Ou = RA.top_m(sJ, M, self.W), RA.top_m(sO, M, self.W)
        Je, Oe = self.rank(sJ), self.rank(sO)
        sets = {"J_own_u": Ju, "J_cross_u": Ou, "J_random_u": RA.random_matched(Ou, Ju, self.abar),
                "J_own_e": Je, "J_cross_e": Oe, "J_random_e": RB.random_matched_excluded(Oe, Je, self.abar, self.mask)}
        self.excluded([sets["J_own_e"], sets["J_cross_e"], sets["J_random_e"]])
        st = self.new_state(unit, {"J": Jt}, capJ, AJ, c, sets, ("J",))
        self.before(st, "J")
        self.readm(st, "identity", [], "mean", True)
        for x in CASE_SETS:
            for v in VALUES:
                self.readm(st, f"{x}__{v}", sets[f"J_{x}"], v, True)
        if known:
            for x in CASE_SETS:
                self.readm(st, f"free__{x}__mean", sets[f"J_{x}"], "mean", False)
            self.readm(st, "identity_fail", sets["J_own_e"][:1], "mean", True)
        if probe:
            self.m["gates"]["marked_write"] = self.marked_probe(st)
        key = f"{j['pole']}__c{j['c']}"
        z = self.bank(st, self.out / "e6c" / "cases" / f"{key}.npz", {"unit": key, "job": j["job"], "pole": j["pole"], "layer": j["layer"], "kind": "case"})
        names = [str(x) for x in z["J_arms"]]
        b = z["J_a_before"]
        falls = {n: RA.job_fall(b, z["J_a_after"][i]) for i, n in enumerate(names)}
        fm = lambda n: falls[n]["mean"]
        rec = {"pole": j["pole"], "layer": j["layer"], "tercile": j["tercile"], "npz": f"e6c/cases/{key}.npz",
               "falls": {n: {k: f[k] for k in ("mean", "median", "n_pos")} for n, f in falls.items()},
               "I": {f"{v}__{e}": JR.ratio(fm(f"cross_{e}__{v}"), fm(f"own_{e}__{v}")) for v in VALUES for e in CONS},
               "cancel": {f"{v}__{e}": RA.cancel(fm(f"own_{e}__{v}"), fm(f"random_{e}__{v}"), falls[f"own_{e}__{v}"]["n_pos"]) for v in VALUES for e in CONS},
               "set_sizes": {x: len(v) for x, v in sets.items()}, "seconds": round(time.time() - st["t0"], 1)}
        scale = np.abs(b).max()
        self.chk["id"].append(float(np.abs(z["J_a_after"][names.index("identity")] - b).max() / scale))
        if known:
            rec["I_free"] = {f"mean__{e}": JR.ratio(fm(f"free__cross_{e}__mean"), fm(f"free__own_{e}__mean")) for e in CONS}
            self.chk["id_fail"].append(float(np.abs(z["J_a_after"][names.index("identity_fail")] - b).max() / scale))
            self.samples(j, st)
        self.m["cases"][j["job"]] = rec

    def samples(self, j: dict, st: dict):
        pr = self.prompts(st["tok"]["J"])
        b = st["before"]["J"]
        ab = self.abar.cpu().numpy()
        with open(self.out / "samples" / "e6c.jsonl", "a") as fh:
            for x in SAMPLE_SETS:
                S = st["sets"][f"J_{x}"]
                am, af = st["after"]["J"][f"{x}__mean"], st["after"]["J"][f"free__{x}__mean"]
                ctx = [{"tok": int(t), **pr[i], "a_before": float(b[i]), "a_after_marked": float(am[i]), "a_after_free": float(af[i]),
                        "fall_marked": float((b[i] - am[i]) / b[i]), "fall_free": float((b[i] - af[i]) / b[i])} for i, t in enumerate(st["tok"]["J"])]
                fh.write(json.dumps({"case": j["job"], "pole": j["pole"], "layer": j["layer"], "side": "J", "value": "mean", "set": x,
                                     "written_positions": "marked: the marked token only (position n_tokens - 1 of each row); free: every window position >= 1",
                                     "members": [[int(l), int(k), float(ab[l, k])] for l, k in S], "job_fall_marked": RA.job_fall(b, am)["mean"],
                                     "job_fall_free": RA.job_fall(b, af)["mean"], "contexts": ctx}, ensure_ascii=False) + "\n")
                self.n_samples += 1

    # -------------------------------------------------------------------------------------- gates
    def gate_known(self):
        got = {k: self.m["cases"][k]["I_free"]["mean__e"] for k in self.known_jobs}
        self.m["gates"]["known_number"] = known_gate(got, self.known_cases)
        self.write()

    def summarize(self):
        c, g = self.chk, self.m["gates"]
        g["identity"] = {"value": max(c["id"]), "tol": ID_TOL, "pass": max(c["id"]) <= ID_TOL, "n_cases": len(c["id"]),
                         "failing_input": {"input": "own_e's first member written at ā at the marked token (the three known cases)", "value": max(c["id_fail"])},
                         "failing_input_failed": max(c["id_fail"]) > ID_TOL, "identity": {"value": max(c["id"]), "reason": "read on the identity arm itself"},
                         "rule": "max over cases and contexts of |a_P after - a_P before| / max |a_P before| on the identity arm"}
        g["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0}
        want = len(self.known_jobs) * len(SAMPLE_SETS)
        g["samples"] = {"lines": self.n_samples, "want": want, "pass": self.n_samples == want, "path": "samples/e6c.jsonl"}
        C = [v for v in self.m["cases"].values()]
        self.m["summary"] = {f"own_e_fall_median__{v}": float(np.median([r["falls"][f"own_e__{v}"]["mean"] for r in C])) for v in VALUES}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.write()
        assert g["identity"]["pass"] and g["identity"]["failing_input_failed"], f"identity: {g['identity']}"
        assert g["marked_write"]["pass"] and g["marked_write"]["failing_input_failed"], f"marked write: {g['marked_write']}"

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        C = self.items["primary"]
        first = [j for j in C if j["job"] in self.known_jobs]
        for i, j in enumerate(first):
            self.case(j, known=True, probe=i == 0)
            self.m["n_done"]["cases"] += 1
            self.write()
        self.gate_known()
        self.m["timing_s"]["known_cases"] = round(time.time() - t0, 1)
        for j in C:
            if j["job"] not in self.known_jobs:
                self.case(j, known=False, probe=False)
                self.m["n_done"]["cases"] += 1
                self.write()
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
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--revision", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs import instruments_e5 as J5
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, "instruments_e6c_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
    assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download(OLMO, a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha, **J5.env()})
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("INSTRUMENTS_E6C_OK", json.dumps({"t": a.t, "sha": sha, "known_number": job.m["gates"]["known_number"]["value"]}), flush=True)


if __name__ == "__main__":
    main()
