"""SPEC5 DATING job, one O7 grid checkpoint per call (studies/reuse/SPEC5_dating.md §2, §3, §5). A subclass of jobs.reuse_ablate.Job like
jobs.reuse_birth's; unite_route.capture, reuse_ablate's capture conventions (256-token window, BOS first) and the parent's known() are reused unchanged.

    python -m infra.aws_pg.jobs.reuse_dating --stage dating --t T_7 --revision stage1-step20000 --items <s3 or local dating_items.json> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.reuse_dating --stage scale --t T_7 --revision stage1-step20000 --items <s3 or local dating_items.json> --corpus <s3> --out <dir>

Every load: the known number and completeness (the parent's gates, asserted), then for every case of the items a capture of J and a capture of O*
(each side in its own call, as SPEC4's Part A captured them) at the pole's layer L with gu_layers {L}; per context a_P = sign silu(g) u of P at the
marked token (g, u fp32, a_P in float64 from them, as verdict_birth reads Part A). Banked per case: dating/<pole>__c<c>.npz (t, pole, side, tok,
gP, uP, aP; one row per context, J's then O*'s) and in metrics.json the mean of a_P over each side. No ablation, no readout.
Capture identity (asserted): the first case's first 20 contexts (J then O*) captured twice in identical calls; a_P equal to 1e-5 (max abs). Its
failing input, read from the second call: the same neuron's g and u at layer L-1 (the hook on the wrong layer), which must differ by more than 1e-5.
At T_15 only, after the cases: (1) the scale capture, each distinct pole's 50 scale contexts once (scale/<pole>.npz; median a_P beside the items'
s_P(T_15)); (2) SPEC4's background: ā over UNITE's 200 e1 positions on layers 0..n_layers-2 and B_15, its top 1% by |ā| (abar.npz,
background.json, high_background.json); (3) a 1M-token background pass: 2,000 windows, each the first 512 tokens of a document with at least 512
tokens drawn with numpy default_rng(0).choice(n_eligible, 2000, replace=False) in draw order (alwayson.windows' rule on the pre-tokenized corpus),
BOS first when the tokenizer has one, a forward pre-hook on every block's mlp.down_proj input over the same layers, accumulating over window positions
1..511 per neuron the sum of a, the sum of |a|, the count of a > 0 and the counts of |a| > 0.5 |ā_200| and |a| > 2 |ā_200| (background_1m.npz); in
metrics.json the Jaccard of B_15 and the top 1% by |abar_1m|, and for B_15's members the median share_pos and share_over_half. Reported, no rule.
metrics.json is rewritten after every case; status done at the end.
--stage scale (SPEC5 amendment 5b), at any load: the known number and completeness, the capture identity, then only the scale capture above for
every distinct pole (scale/<pole>.npz; in metrics.json the median a_P beside the items' s_P at this load, and n_done.scale_poles); no case captures,
no background. 183 poles x 50 contexts = 9,150 captures per load plus the identity's 2 x 20.
--stage scale --abar (SPEC8 §2, §3): after the scale capture, background() at this load (ā over UNITE's 200 e1 positions of this checkpoint on layers
0..n_layers-2: abar.npz, background.json, high_background.json), the global scale g_l(t) = the layer's median |ā|.
--stage dating+scale (SPEC8's pipeline): one model load runs the dating stage into --out, unchanged, then the scale stage (with --abar if given) into
<out>/scale_stage; the second stage's metrics.json records download and load seconds as 0 (the load is the first stage's).
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

# the O7 grid (dating_items.json -> sources.grid): T_0..T_3 added to the parent's table, T_4..T_15 the parent's
REVISION = {"T_0": "stage1-step0", "T_1": "stage1-step1000", "T_2": "stage1-step2000", "T_3": "stage1-step3000", **JR.REVISION}
CAP = 12
N_ID = 20
ID_TOL = 1e-5
N_SAMPLE = 2


def silu64(g):
    return g / (1 + np.exp(-g))


def windows_tok(tokens: np.ndarray, offsets: np.ndarray, n: int, length: int, seed: int) -> dict:
    """studies/reproductions/alwayson/code/pg/alwayson.windows on a pre-tokenized corpus: documents (tokens[offsets[j]:offsets[j+1]]) with >= length tokens are eligible; n drawn
    with default_rng(seed).choice(n_eligible, n, replace=False) in draw order; each window its document's first length tokens."""
    lens = np.diff(offsets)
    eligible = np.flatnonzero(lens >= length)
    assert len(eligible) >= n, f"{len(eligible)} documents with >= {length} tokens, {n} needed"
    pick = np.random.default_rng(seed).choice(len(eligible), size=n, replace=False)
    doc_ids = eligible[pick]
    win = np.stack([np.asarray(tokens[offsets[d]:offsets[d] + length]) for d in doc_ids]).astype(np.int64)
    assert win.shape == (n, length), win.shape
    return {"tokens": win, "doc_ids": doc_ids.astype(np.int64), "n_eligible": int(len(eligible)), "n_docs": int(len(lens)), "n": n, "length": length,
            "seed": seed, "tokens_sha256": hashlib.sha256(np.ascontiguousarray(win, dtype=np.int32).tobytes()).hexdigest(),
            "doc_ids_sha256": hashlib.sha256(np.ascontiguousarray(doc_ids, dtype=np.int64).tobytes()).hexdigest()}


class Job(JR.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "dating", "samples", "scale") if a.stage == "dating" else ("", "samples", "scale"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert a.stage in ("dating", "scale"), a.stage
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P assumes silu"
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "cap": CAP, "timing_s": {}, "tokens_forwarded": 0, "captures": 0, "cases": {},
                  "a_P": "sign * silu(g) * u of the pole at the marked token, g and u fp32 from unite_route.capture, a_P float64 from them"}
        self.write()

    def capture(self, toks, unit, gu_layers=None):
        L = unit["layer"]
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)
        return UR.capture(self.model, np.asarray(toks), L, gu_layers or {L}, **self.kw(RA.LENGTH))

    @staticmethod
    def gu(cap, unit, layer=None):
        """P's g and u (float32 arrays) at `layer` (default P's own) and a_P = sign silu(g) u in float64."""
        t = unit["terms"][0]
        L = unit["layer"] if layer is None else layer
        g, u = cap["g"][L][:, t["n"]].cpu().numpy().astype(np.float32), cap["u"][L][:, t["n"]].cpu().numpy().astype(np.float32)
        return g, u, t["sign"] * silu64(g.astype(np.float64)) * u.astype(np.float64)

    # -------------------------------------------------------------------------------------- the gates of this job
    def identity(self):
        """Two identical captures of the first case's first 20 contexts; the failing input reads P's neuron at layer L-1 from the second call."""
        t0 = time.time()
        c = self.items["cases"][0]
        unit = UR.pole_unit(c["pole"])
        L = unit["layer"]
        assert L >= 1, (c["pole"], "the wrong-layer failing input needs L >= 1")
        toks = np.asarray(c["J_tok"] + c["O_tok"])[:N_ID]
        assert len(toks) == N_ID, (c["job"], len(toks))
        _, _, a1 = self.gu(self.capture(toks, unit), unit)
        cap2 = self.capture(toks, unit, {L - 1, L})
        _, _, a2 = self.gu(cap2, unit)
        _, _, aw = self.gu(cap2, unit, L - 1)
        d, dw = float(np.abs(a2 - a1).max()), float(np.abs(aw - a1).max())
        self.m["capture_identity"] = {"pole": c["pole"], "job": c["job"], "n_contexts": N_ID, "max_abs_diff": d, "tol": ID_TOL, "pass": d <= ID_TOL,
                                      "failing_input": f"P's neuron read at layer {L - 1} (the hook on L-1)", "failing_input_max_abs_diff": dw,
                                      "failing_input_failed": dw > ID_TOL, "a_P_abs_max": float(np.abs(a1).max())}
        self.m["timing_s"]["identity"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["capture_identity"]["pass"], f"capture identity failed: {d}"
        assert self.m["capture_identity"]["failing_input_failed"], f"capture identity's failing input passed: {dw}"

    # -------------------------------------------------------------------------------------- one case
    def case(self, c: dict):
        t0 = time.time()
        unit = UR.pole_unit(c["pole"])
        assert unit["layer"] == c["layer"], (c["job"], unit["layer"], c["layer"])
        sides = {}
        for side, toks in (("J", np.asarray(c["J_tok"])), ("O", np.asarray(c["O_tok"]))):
            assert 1 <= len(toks) <= CAP, (c["job"], side, len(toks))
            sides[side] = (toks, *self.gu(self.capture(toks, unit), unit))
        key = f"{c['pole']}__c{c['c']}"
        p = self.out / "dating" / f"{key}.npz"
        cat = lambda i: np.concatenate([sides["J"][i], sides["O"][i]])
        np.savez_compressed(p, t=self.a.t, pole=c["pole"], side=np.array(["J"] * len(sides["J"][0]) + ["O"] * len(sides["O"][0])),
                            tok=cat(0).astype(np.int64), gP=cat(1), uP=cat(2), aP=cat(3))
        self.m["cases"][c["job"]] = {"set": c["set"], "pole": c["pole"], "layer": c["layer"], "birth": c["birth"], "n_J": int(len(sides["J"][0])),
                                     "n_O": int(len(sides["O"][0])), "mean_J": float(sides["J"][3].mean()), "mean_O": float(sides["O"][3].mean()),
                                     "path": str(p.relative_to(self.out)), "seconds": round(time.time() - t0, 2)}

    def samples(self):
        """The prompts and marked tokens of each case's first two contexts per side (the same at every checkpoint); banked at T_15."""
        with open(self.out / "samples" / "prompts.jsonl", "w") as fh:
            for c in self.items["cases"]:
                for side in ("J", "O"):
                    toks = np.asarray(c[f"{side}_tok"])[:N_SAMPLE]
                    for t, pr in zip(toks.tolist(), self.prompts(toks)):
                        fh.write(json.dumps({"job": c["job"], "side": side, "tok": int(t), **pr}, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------------------- T_15 only (stage dating); every load (stage scale)
    def scale(self):
        """Each distinct pole's 50 scale contexts, captured once; median a_P beside the items' s_P at this load."""
        t0 = time.time()
        out = {}
        for c in self.items["cases"]:
            p = c["pole"]
            if p in out:
                assert out[p]["median_bank"] == c["s_P"][self.a.t], (p, f"two cases of one pole carry different s_P({self.a.t})")
                continue
            unit = UR.pole_unit(p)
            toks = np.asarray(c["scale_tok"])
            g, u, aP = self.gu(self.capture(toks, unit), unit)
            path = self.out / "scale" / f"{p}.npz"
            np.savez_compressed(path, t=self.a.t, pole=p, tok=toks.astype(np.int64), gP=g, uP=u, aP=aP)
            med = float(np.median(aP))
            sb = c["s_P"][self.a.t]
            out[p] = {"median_captured": med, "median_bank": sb, "rel_diff": abs(med - sb) / abs(sb), "n": int(len(toks)),
                      "path": str(path.relative_to(self.out))}
            self.m["scale"] = out
            self.write()
        self.m["scale_summary"] = {"n_poles": len(out), "max_rel_diff": max(v["rel_diff"] for v in out.values()),
                                   "n_over_5pct": sum(v["rel_diff"] > 0.05 for v in out.values())}
        self.m["timing_s"]["scale"] = round(time.time() - t0, 1)
        self.write()

    def background(self):
        """SPEC4's ā and B_15 (jobs.reuse_birth.Job.background's code path on layers 0..n_layers-2)."""
        t0 = time.time()
        n_up = len(self.layers) - 1
        assert max(c["layer"] for c in self.items["cases"]) <= n_up, "a pole sits above the upstream layers"
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
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"

    @torch.inference_mode()
    def background_1m(self):
        """The 1M-token pass on the down_proj input of layers 0..n_up-1, window positions 1..length-1, against ā_200 of this load."""
        t0 = time.time()
        a = self.a
        n_up = self.abar.shape[0]
        win = windows_tok(self.tokens, self.offsets, a.bg_windows, a.bg_length, 0)
        st = RA.window_start(self.bos)
        ab = self.abar.float().abs().to(self.device)
        acc = {k: torch.zeros(n_up, self.W, dtype=torch.float64, device=self.device) for k in ("sum", "abs_sum")}
        acc.update({k: torch.zeros(n_up, self.W, dtype=torch.int64, device=self.device) for k in ("n_pos", "n_over_half", "n_over_twice")})
        seen = []

        def take(l, x):
            assert x.dim() == 3 and x.shape[-1] == self.W and x.shape[1] == st + a.bg_length, (l, tuple(x.shape))
            body = x[:, st + 1:].float()
            mag = body.abs()
            acc["sum"][l] += body.sum((0, 1)).double()
            acc["abs_sum"][l] += mag.sum((0, 1)).double()
            acc["n_pos"][l] += (body > 0).sum((0, 1))
            acc["n_over_half"][l] += (mag > 0.5 * ab[l]).sum((0, 1))
            acc["n_over_twice"][l] += (mag > 2.0 * ab[l]).sum((0, 1))
            seen.append(l)

        handles = [self.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, args, l=l: take(l, args[0])) for l in range(n_up)]
        prefix = [] if self.bos is None else [self.bos]
        n_batches = 0
        with UR.upto_layer(self.tower, n_up - 1):
            for i in range(0, a.bg_windows, a.bg_batch):
                x = torch.as_tensor([prefix + w for w in win["tokens"][i:i + a.bg_batch].tolist()], dtype=torch.long, device=self.device)
                NH.forward(self.model, x, False)
                n_batches += 1
        for h in handles:
            h.remove()
        assert seen == list(range(n_up)) * n_batches, "the down_proj hooks did not fire once per layer per batch"
        n = a.bg_windows * (a.bg_length - 1)
        self.m["tokens_forwarded"] += a.bg_windows * (a.bg_length + len(prefix))
        r = {k: v.cpu().numpy() for k, v in acc.items()}
        abar_1m = (r["sum"] / n).astype(np.float32)
        out = {"abar_1m": abar_1m, "mabs_1m": (r["abs_sum"] / n).astype(np.float32), "share_pos": (r["n_pos"] / n).astype(np.float32),
               "share_over_half": (r["n_over_half"] / n).astype(np.float32), "share_over_twice": (r["n_over_twice"] / n).astype(np.float32),
               "n_positions": np.int64(n), "doc_ids": win["doc_ids"], "tokens_sha256": win["tokens_sha256"], "doc_ids_sha256": win["doc_ids_sha256"]}
        np.savez_compressed(self.out / "background_1m.npz", **out)
        B15 = self.mask.cpu().numpy()
        B1m = RB.high_background(torch.as_tensor(abar_1m), n_up)["mask"].cpu().numpy()
        self.m["background_1m"] = {"n_windows": a.bg_windows, "length": a.bg_length, "n_positions": int(n), "positions": f"window positions 1..{a.bg_length - 1}",
                                   "bos_first": self.bos is not None, "batch": a.bg_batch, "n_eligible": win["n_eligible"], "n_docs": win["n_docs"], "seed": 0,
                                   "tokens_sha256": win["tokens_sha256"], "doc_ids_sha256": win["doc_ids_sha256"], "finite": bool(np.isfinite(abar_1m).all()),
                                   "jaccard_B15_top1pct_1m": float((B15 & B1m).sum() / (B15 | B1m).sum()), "n_B15": int(B15.sum()), "n_top1pct_1m": int(B1m.sum()),
                                   "B15_median_share_pos": float(np.median(out["share_pos"][B15])),
                                   "B15_median_share_over_half": float(np.median(out["share_over_half"][B15])),
                                   "all_median_share_pos": float(np.median(out["share_pos"])), "all_median_share_over_half": float(np.median(out["share_over_half"])),
                                   "rule": "windows_tok (alwayson.windows' rule on the pre-tokenized corpus); mlp.down_proj input, layers 0..n_up-1; "
                                           "B_15 against the top 1% by |abar_1m| (reuse_birth.high_background on both)"}
        self.m["timing_s"]["background_1m"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background_1m"]["finite"], "abar_1m is not finite"

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.identity()
        cases = self.items["cases"]
        if self.a.stage == "scale":
            self.m["n_to_read"] = {"cases": 0, "scale_poles": len({c["pole"] for c in cases})}
            self.m["n_done"] = {"cases": 0, "scale_poles": 0}
            self.scale()
            self.m["n_done"]["scale_poles"] = len(self.m["scale"])
            if self.a.abar:
                self.background()
        else:
            t0 = time.time()
            tok0 = self.m["tokens_forwarded"]
            self.m["n_to_read"] = {"cases": len(cases), "scale_poles": len({c["pole"] for c in cases}) if self.a.t == "T_15" else 0}
            self.m["n_done"] = {"cases": 0, "scale_poles": 0}
            for c in cases:
                self.case(c)
                self.m["n_done"]["cases"] += 1
                self.write()
            dt = time.time() - t0
            self.m["timing_s"]["cases"] = round(dt, 1)
            self.m["cases_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
            if self.a.t == "T_15":
                self.samples()
                self.scale()
                self.m["n_done"]["scale_poles"] = len(self.m["scale"])
                self.background()
                self.background_1m()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["dating", "scale", "dating+scale"])
    ap.add_argument("--abar", action="store_true", help="scale stage: background() (ā, g_l(t)) at this load too")
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    ap.add_argument("--bg-windows", type=int, default=2000)
    ap.add_argument("--bg-length", type=int, default=512)
    ap.add_argument("--bg-batch", type=int, default=16)
    return ap


def main():
    a = parser().parse_args()
    assert REVISION[a.t] == a.revision, f"{a.t} is {REVISION[a.t]}, not {a.revision}"
    stages = a.stage.split("+")
    assert not a.abar or "scale" in stages, "--abar belongs to the scale stage"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, f"reuse_dating_{a.t}_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
    assert grid == REVISION, ("the items' grid is not this job's revision table", {k: (grid.get(k), REVISION.get(k)) for k in set(grid) | set(REVISION) if grid.get(k) != REVISION.get(k)})
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    for i, st in enumerate(stages):
        sa = argparse.Namespace(**{**vars(a), "stage": st})
        t_st = time.time() if i else t1
        job = Job(pathlib.Path(a.out) / "scale_stage" if i else a.out, sa, items, model, tok, corpus, "cuda")
        job.m.update({"download": rec, "download_and_corpus_s": 0.0 if i else round(t1 - t0, 1), "load_s": 0.0 if i else round(time.time() - t1, 1), "sha": sha,
                      "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
        job.m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        job.write()
        job.run()
        job.m["timing_s"]["wall_after_load"] = round(time.time() - t_st, 1)
        job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
        job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("REUSE_DATING_OK", json.dumps({"t": a.t}), flush=True)


if __name__ == "__main__":
    main()
