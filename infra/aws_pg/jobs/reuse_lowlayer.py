"""SPEC12 LOWLAYER capture job, one O7 grid checkpoint per call (studies/reuse/SPEC12_lowlayer.md §2, §5). A subclass of jobs.reuse_ablate.Job like
jobs.reuse_dating's; unite_route.capture and reuse_dating.Job.gu are reused unchanged for the instrument, neurons_harvest.MLPTap and forward for
the harvest condition.

    python -m infra.aws_pg.jobs.reuse_lowlayer --t T_7 --revision stage1-step20000 --items <s3 or local lowlayer_poles.json> --corpus <s3> --out <dir>

For every pole of the items (run order: the 8 known poles, then the 128 new poles) and every one of its 50 positions, two captures, each with the
forward stopped after the pole's layer L (L <= 3, the items' upto_layer; no later layer feeds L):
  (i) the harvest condition (neurons_harvest.run): the 2,048-token window the position sits in (windows at multiples of 2,048 corpus tokens), BOS
      first, the model in bf16; a = act(g) * u recomputed in fp32 from the bf16 outputs of mlp.gate_proj and mlp.up_proj (MLPTap's hooks), read at
      the position's window index -> a_h (float32). Each distinct window of the pole is forwarded once, --harvest-batch windows per forward.
  (ii) SPEC8's instrument (reuse_dating --stage dating/scale): unite_route.capture with the 256-token window ending at the position, BOS first, g and
      u fp32, a_P = sign silu(g) u in float64 (reuse_dating.Job.gu) -> gP, uP, aP. s_P(t) = median aP over the 50 positions.
Banked per pole: lowlayer/<pole>.npz (t, pole, tok, doc, a_h, gP, uP, aP; one row per position in rank order); metrics.json rewritten after every pole.
Gates, asserted (each failing input computed and asserted to fail at every load):
  known number (i): the known poles' a_h against their banked exemplar_a at this load, median over the 400 contexts of |a_h - bank| / |bank| <= 0.02;
      failing input the previous load's bank (T_0: the T_1 bank). At T_15 also the new poles' a_h against the banked topk.a (6,400 contexts, the same
      line); its failing input the same bank in reversed rank order.
  known number (ii): for each known pole with a SPEC8 third-set case, aP on the case's J and O* contexts (each side its own capture, as
      reuse_dating.Job.case) against SPEC8's banked dating/<pole>__c<c>.npz at this load (carried in the items), median relative difference <= 0.05;
      failing input the next load's bank (T_15: the T_14 bank).
  capture identity: the first pole's first 20 positions captured twice by each route; a_h max abs difference 0, aP under 1e-5; failing input the
      same neuron read at layer L + 1 from the second call, which must differ by more than 1e-5.
  completeness: every pole's npz read back from disk: 136 poles x 50 positions, tok equal to the items', a_h, gP, uP and aP finite.
Samples (samples/known.json, samples/new.json): the first two contexts of each known pole and of the first three new poles, with a_h, aP and the bank.
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
from infra.aws_pg.jobs import reuse_dating as RD
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REVISION = RD.REVISION                  # T_0..T_15, stage1-step<S>
N_ID = 20
ID_TOL_H, ID_TOL_P, ID_FAIL = 0.0, 1e-5, 1e-5
KNOWN_H_TOL, KNOWN_P_TOL = 0.02, 0.05
N_SAMPLE, N_SAMPLE_POLES = 2, 3
NPZ_KEYS = ("t", "pole", "tok", "doc", "a_h", "gP", "uP", "aP")


def rel_diff(x, b) -> float:
    """Median over contexts of |x - b| / |b|."""
    x, b = np.asarray(x, np.float64), np.asarray(b, np.float64)
    assert x.shape == b.shape and x.ndim == 1 and len(x) > 0, (x.shape, b.shape)
    return float(np.median(np.abs(x - b) / np.abs(b)))


def gate_known(x, bank, wrong, tol: float) -> dict:
    r, w = rel_diff(x, bank), rel_diff(x, wrong)
    return {"median_rel_diff": r, "failing_input_median_rel_diff": w, "tol": tol, "n_contexts": int(len(x)), "pass": r <= tol, "failing_input_failed": w > tol}


def gate_identity(a1, a2, a_wrong, tol: float) -> dict:
    d, dw = float(np.abs(np.asarray(a2, np.float64) - a1).max()), float(np.abs(np.asarray(a_wrong, np.float64) - a1).max())
    return {"max_abs_diff": d, "tol": tol, "pass": d <= tol, "failing_input_max_abs_diff": dw, "failing_input_line": ID_FAIL, "failing_input_failed": dw > ID_FAIL}


def gate_complete(banked: dict, poles: list, t: str) -> dict:
    """banked: {pole: {key: array}} as read back from lowlayer/<pole>.npz; poles: the items' pole records."""
    missing = [p["pole"] for p in poles if p["pole"] not in banked]
    bad = {}
    for p in poles:
        if p["pole"] in missing:
            continue
        z, n, why = banked[p["pole"]], len(p["tok"]), []
        if sorted(z) != sorted(NPZ_KEYS):
            why.append(f"keys {sorted(z)}")
        elif str(z["t"]) != t or str(z["pole"]) != p["pole"]:
            why.append(f"t {z['t']} pole {z['pole']}")
        elif any(np.asarray(z[k]).shape != (n,) for k in NPZ_KEYS[2:]):
            why.append("shape")
        elif not np.array_equal(z["tok"], p["tok"]) or not np.array_equal(z["doc"], p["doc"]):
            why.append("tok or doc differ from the items")
        elif not all(np.isfinite(z[k]).all() for k in ("a_h", "gP", "uP", "aP")):
            why.append("non-finite a_h, gP, uP or aP")
        if why:
            bad[p["pole"]] = why
    n_ctx = int(sum(len(banked[p["pole"]]["a_h"]) for p in poles if p["pole"] in banked and p["pole"] not in bad))
    return {"n_poles_expected": len(poles), "n_poles_present": len(poles) - len(missing), "n_contexts_complete": n_ctx,
            "n_contexts_expected": int(sum(len(p["tok"]) for p in poles)), "missing": missing, "bad": bad, "pass": not missing and not bad}


class Job(JR.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "lowlayer", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        assert self.bos is not None, "the harvest prepends a BOS; this tokenizer defines none"
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.acts = [l.mlp.act_fn for l in self.layers]
        self.hw = a.harvest_window
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P assumes silu"
        self.known_poles, self.new_poles = items["known"], items["poles"]
        self.upto = items["upto_layer"]
        allp = self.known_poles + self.new_poles
        assert all(p["layer"] <= self.upto for p in allp) and self.upto < len(self.layers), "a pole above the stopping layer"
        assert self.new_poles[0]["layer"] + 1 <= self.upto, "the identity's failing input reads layer L + 1 of the first pole"
        for p in allp:
            toks = np.asarray(p["tok"], np.int64)
            assert np.array_equal(np.searchsorted(self.offsets, toks, side="right") - 1, p["doc"]), (p["pole"], "doc ids differ: another corpus")
        self.m = {"t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "harvest_batch": a.harvest_batch,
                  "window": RA.LENGTH, "batch": a.batch, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers),
                  "width": self.W, "dtype": str(next(model.parameters()).dtype), "upto_layer": self.upto, "timing_s": {}, "tokens_forwarded": 0,
                  "harvest_windows": 0, "captures": 0, "poles": {},
                  "a_h": "act(g) * u in fp32 from the bf16 gate_proj/up_proj outputs at the position, the 2,048-token window holding it, BOS first "
                         "(neurons_harvest.run); forward stopped after the pole's layer",
                  "a_P": "sign * silu(g) * u of the pole at the marked token, g and u fp32 from unite_route.capture (256-token window ending at the "
                         "position, BOS first), a_P float64 from them (reuse_dating.Job.gu)"}
        self.write()

    # -------------------------------------------------------------------------------------- the two captures
    @torch.inference_mode()
    def harvest_a(self, toks, n: int, layers: list) -> dict:
        """{layer: a_h [N] float32} of neuron n at each global position in `toks`, each from the forward of its 2,048-token window."""
        toks = np.asarray(toks, np.int64)
        assert (toks % self.hw != 0).all(), "window position 0 is not a context (the harvest excludes it)"
        win = toks // self.hw
        uw = np.unique(win)
        assert int(uw[-1] + 1) * self.hw <= len(self.tokens), "a window runs past the corpus"
        top = max(layers)
        tap = NH.MLPTap(self.layers[:top + 1])
        got = {l: [] for l in layers}

        def fn(lyr, g, u, a_model):
            if lyr in got:
                got[lyr].append(self.acts[lyr](g[:, 1:, n].float()) * u[:, 1:, n].float())     # [B, hw]: window positions 0..hw-1 after the BOS

        for b0 in range(0, len(uw), self.a.harvest_batch):
            w = uw[b0:b0 + self.a.harvest_batch]
            x = torch.from_numpy(np.stack([np.asarray(self.tokens[i * self.hw:(i + 1) * self.hw]) for i in w]).astype(np.int64))
            x = torch.cat([torch.full((len(w), 1), self.bos, dtype=torch.long), x], 1).to(self.device)
            with UR.upto_layer(self.tower, top):
                tap.run(fn, lambda: NH.forward(self.model, x, False))
        self.m["tokens_forwarded"] += len(uw) * (self.hw + 1)
        self.m["harvest_windows"] += len(uw)
        row = torch.as_tensor(np.searchsorted(uw, win), device=self.device)
        col = torch.as_tensor(toks % self.hw, device=self.device)
        return {l: torch.cat(v)[row, col].cpu().numpy().astype(np.float32) for l, v in got.items()}

    def instrument(self, toks, unit, layers=None) -> dict:
        """unite_route.capture with the 256-token window; forward stopped after max(layers) (default the pole's layer)."""
        layers = set(layers or {unit["layer"]})
        toks = np.asarray(toks, np.int64)
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
        return UR.capture(self.model, toks, max(layers), layers, **self.kw(RA.LENGTH))

    # -------------------------------------------------------------------------------------- one pole
    def pole(self, p: dict, kind: str) -> tuple:
        t0 = time.time()
        unit = UR.pole_unit(p["pole"])
        L, n, sign = unit["layer"], unit["terms"][0]["n"], unit["terms"][0]["sign"]
        assert (L, n, "+" if sign > 0 else "-") == (p["layer"], p["neuron"], p["sign"]), p["pole"]
        toks = np.asarray(p["tok"], np.int64)
        a_h = self.harvest_a(toks, n, [L])[L]
        g, u, aP = RD.Job.gu(self.instrument(toks, unit), unit)
        path = self.out / "lowlayer" / f"{p['pole']}.npz"
        np.savez_compressed(path, t=self.a.t, pole=p["pole"], tok=toks, doc=np.asarray(p["doc"], np.int64), a_h=a_h, gP=g, uP=u, aP=aP)
        self.m["poles"][p["pole"]] = {"kind": kind, "layer": L, "n": int(len(toks)), "median_signed_a_h": float(np.median(sign * a_h)),
                                      "s_P": float(np.median(aP)), "median_rel_diff_aP_vs_signed_a_h": rel_diff(aP, sign * a_h),
                                      "finite": bool(np.isfinite(a_h).all() and np.isfinite(aP).all()), "path": str(path.relative_to(self.out)),
                                      "seconds": round(time.time() - t0, 2)}
        self.m["n_done"]["poles"] += 1
        self.write()
        return a_h, aP

    # -------------------------------------------------------------------------------------- gates
    def known_harvest(self, ah: dict):
        """Known number (i) on the known poles; failing input the previous load's bank (T_0: T_1)."""
        ti = self.t_index
        wrong = f"T_{ti - 1}" if ti > 0 else "T_1"
        ks = self.known_poles
        x = np.concatenate([ah[k["pole"]] for k in ks])
        rec = gate_known(x, np.concatenate([k["a_bank"][self.a.t] for k in ks]), np.concatenate([k["a_bank"][wrong] for k in ks]), KNOWN_H_TOL)
        rec.update({"bank": f"exemplar_a at {self.a.t}", "failing_input": f"exemplar_a at {wrong} (the previous load's bank)",
                    "per_pole": {k["pole"]: rel_diff(ah[k["pole"]], k["a_bank"][self.a.t]) for k in ks}})
        self.m["known_harvest"] = rec
        self.write()
        assert rec["pass"], f"known number (harvest condition) failed: {rec['median_rel_diff']}"
        assert rec["failing_input_failed"], f"known number (harvest condition)'s failing input passed: {rec['failing_input_median_rel_diff']}"

    def known_harvest_new(self, ah: dict):
        """T_15 only: the new poles' a_h against the banked topk.a; failing input the same bank in reversed rank order."""
        ps = self.new_poles
        x = np.concatenate([ah[p["pole"]] for p in ps])
        rec = gate_known(x, np.concatenate([p["a"] for p in ps]), np.concatenate([p["a"][::-1] for p in ps]), KNOWN_H_TOL)
        rec.update({"bank": "topk.a at T_15 (the grid's topk/layer_<L>.npz)", "failing_input": "topk.a in reversed rank order",
                    "max_per_pole": max(rel_diff(ah[p["pole"]], p["a"]) for p in ps)})
        self.m["known_harvest_new_T15"] = rec
        self.write()
        assert rec["pass"], f"known number (harvest condition, new poles at T_15) failed: {rec['median_rel_diff']}"
        assert rec["failing_input_failed"], f"known number (new poles at T_15)'s failing input passed: {rec['failing_input_median_rel_diff']}"

    def known_instrument(self):
        """Known number (ii): SPEC8's banked dating captures of the known poles' third-set cases; failing input the next load's bank (T_15: T_14)."""
        t0 = time.time()
        ti = self.t_index
        wrong = f"T_{ti + 1}" if ti < 15 else "T_14"
        xs, bs, ws, per = [], [], [], {}
        for k in self.known_poles:
            unit = UR.pole_unit(k["pole"])
            for c in k["spec8_cases"]:
                side, tok = np.asarray(c["side"]), np.asarray(c["tok"], np.int64)
                assert list(side) == sorted(side, key=lambda s: s != "J"), (k["pole"], c["c"], "J rows then O rows")
                x = np.concatenate([RD.Job.gu(self.instrument(tok[side == s], unit), unit)[2] for s in ("J", "O")])
                xs.append(x)
                bs.append(np.asarray(c["aP"][self.a.t]))
                ws.append(np.asarray(c["aP"][wrong]))
                per[f"{k['pole']}__c{c['c']}"] = {"median_rel_diff": rel_diff(x, bs[-1]), "n": int(len(x)), "bank_sha256": c["sha256"][self.a.t]}
        if not xs:
            self.m["known_instrument"] = {"n_cases": 0, "pass": None, "note": "no known pole has a SPEC8 third-set case"}
            self.write()
            return
        rec = gate_known(np.concatenate(xs), np.concatenate(bs), np.concatenate(ws), KNOWN_P_TOL)
        rec.update({"n_cases": len(per), "per_case": per, "bank": f"SPEC8 dating/<pole>__c<c>.npz aP at {self.a.t}",
                    "failing_input": f"the same files at {wrong} (the next load's bank)", "seconds": round(time.time() - t0, 2)})
        self.m["known_instrument"] = rec
        self.write()
        assert rec["pass"], f"known number (SPEC8 instrument) failed: {rec['median_rel_diff']}"
        assert rec["failing_input_failed"], f"known number (SPEC8 instrument)'s failing input passed: {rec['failing_input_median_rel_diff']}"

    def identity(self):
        """The first new pole's first 20 positions twice by each route; failing input the same neuron at layer L + 1 from the second call."""
        t0 = time.time()
        p = self.new_poles[0]
        unit = UR.pole_unit(p["pole"])
        L, n = unit["layer"], unit["terms"][0]["n"]
        toks = np.asarray(p["tok"][:N_ID], np.int64)
        assert len(toks) == N_ID, len(toks)
        h1 = self.harvest_a(toks, n, [L])[L]
        h2 = self.harvest_a(toks, n, [L, L + 1])
        a1 = RD.Job.gu(self.instrument(toks, unit), unit)[2]
        cap2 = self.instrument(toks, unit, {L, L + 1})
        a2, aw = RD.Job.gu(cap2, unit)[2], RD.Job.gu(cap2, unit, L + 1)[2]
        rec = {"pole": p["pole"], "layer": L, "n_contexts": N_ID, "failing_input": f"neuron {n} read at layer {L + 1} from the second call (the hook on L + 1)",
               "a_h": gate_identity(h1, h2[L], h2[L + 1], ID_TOL_H), "a_P": gate_identity(a1, a2, aw, ID_TOL_P), "a_P_abs_max": float(np.abs(a1).max()),
               "a_h_abs_max": float(np.abs(h1).max())}
        rec["pass"] = rec["a_h"]["pass"] and rec["a_P"]["pass"]
        rec["failing_input_failed"] = rec["a_h"]["failing_input_failed"] and rec["a_P"]["failing_input_failed"]
        self.m["capture_identity"] = rec
        self.m["timing_s"]["identity"] = round(time.time() - t0, 1)
        self.write()
        assert rec["pass"], f"capture identity failed: {rec['a_h']['max_abs_diff']}, {rec['a_P']['max_abs_diff']}"
        assert rec["failing_input_failed"], f"capture identity's failing input passed: {rec['a_h']['failing_input_max_abs_diff']}, {rec['a_P']['failing_input_max_abs_diff']}"

    def completeness(self):
        allp = self.known_poles + self.new_poles
        banked = {}
        for p in allp:
            path = self.out / "lowlayer" / f"{p['pole']}.npz"
            if path.exists():
                with np.load(path) as z:
                    banked[p["pole"]] = {k: z[k] for k in z.files}
        rec = gate_complete(banked, allp, self.a.t)
        self.m["completeness"] = rec
        self.write()
        assert rec["pass"], f"completeness failed: missing {rec['missing'][:5]}, bad {list(rec['bad'].items())[:5]}"

    def samples(self, ah: dict, ap: dict):
        def rows(p, bank):
            return [{"rank": r, "tok": int(p["tok"][r]), **NH.context(self.tokens, self.offsets, int(p["tok"][r]), self.dec), "a_h": float(ah[p["pole"]][r]),
                     "aP": float(ap[p["pole"]][r]), "bank": None if bank is None else float(bank[r])} for r in range(N_SAMPLE)]
        NH.dump(self.out / "samples" / "known.json", {k["pole"]: rows(k, k["a_bank"][self.a.t]) for k in self.known_poles})
        NH.dump(self.out / "samples" / "new.json", {p["pole"]: rows(p, p["a"] if self.a.t == "T_15" else None) for p in self.new_poles[:N_SAMPLE_POLES]})

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        allp = self.known_poles + self.new_poles
        self.m["n_to_read"] = {"poles": len(allp), "contexts": int(sum(len(p["tok"]) for p in allp))}
        self.m["n_done"] = {"poles": 0}
        ah, ap = {}, {}
        t0 = time.time()
        for k in self.known_poles:
            ah[k["pole"]], ap[k["pole"]] = self.pole(k, "known")
        self.known_harvest(ah)
        self.known_instrument()
        self.m["timing_s"]["known"] = round(time.time() - t0, 1)
        self.identity()
        t0, tok0, win0 = time.time(), self.m["tokens_forwarded"], self.m["harvest_windows"]
        for p in self.new_poles:
            ah[p["pole"]], ap[p["pole"]] = self.pole(p, "new")
        dt = time.time() - t0
        self.m["timing_s"]["new_poles"] = round(dt, 1)
        self.m["new_poles_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.m["new_poles_harvest_windows"] = self.m["harvest_windows"] - win0
        if self.a.t == "T_15":
            self.known_harvest_new(ah)
        self.samples(ah, ap)
        self.completeness()
        assert self.m["n_done"]["poles"] == self.m["n_to_read"]["poles"], (self.m["n_done"], self.m["n_to_read"])
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=32, help="contexts per unite_route.capture forward (reuse_dating's)")
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--harvest-batch", type=int, default=8, help="2,048-token windows per forward (neurons_harvest's batch)")
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def check_items(items: dict, t: str):
    grid = {k: "stage1-step" + v.split("_step")[1] for k, v in items["grid"].items()}
    assert grid == REVISION, ("the items' grid is not this job's revision table", grid)
    assert len(items["poles"]) == 128 and len(items["known"]) == 8, (len(items["poles"]), len(items["known"]))
    assert all(t in k["a_bank"] for k in items["known"]), t


def main():
    a = parser().parse_args()
    assert REVISION[a.t] == a.revision, f"{a.t} is {REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, f"reuse_lowlayer_{a.t}_poles.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    check_items(items, a.t)
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
    print("REUSE_LOWLAYER_OK", json.dumps({"t": a.t}), flush=True)


if __name__ == "__main__":
    main()
