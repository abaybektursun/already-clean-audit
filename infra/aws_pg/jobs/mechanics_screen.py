"""MECHANICS screen job, one Olmo-3 load at T_15 (studies/mechanics/SPEC_mechanics.md §2, §3, §7). The science is infra/pg/mechanics.py; the
checkpoint download, the corpus and the model load are neurons_harvest's, as instruments_e1 and reuse_lowlayer use them.

    python -m infra.aws_pg.jobs.mechanics_screen --t T_15 --revision stage1-step1413814 --corpus <s3 corpus> --bank <s3 grid bank dir> --out <dir>

Every forward runs eager attention: the head scores need the weights, and E^abl / E and the ablation identity compare forwards of one attention code.
Order (metrics.json rewritten after every step):
  1 weights    column norms and the moments of W_U w_k over the vocabulary (no forward).
  2 scale      s [L, W] = the larger of the bank's per-pole q999 (stats/layer_<L>.npz); zero or non-finite neurons counted and excluded.
  3 known      (asserted) 16 top contexts of 3 bank + poles (layers L/16, L/2, L-2, neuron by default_rng(13)) at the full harvest-window prefix:
               median over poles of median |a - topk.a| / median |topk.a| <= 0.05; failing input the next pole's bank (cyclic).
  4 identity   (asserted) logits of 2 repeated rows with every hook installed and nothing written equal the unhooked logits to 1e-6 relative;
               failing input one neuron written (+ s_k at every position).
  5 abl. id.   (asserted) the first n_id repeated rows forwarded unablated (the repeated pass's hooks), with the ablation hook and zero heads, and
               with all heads of layer L/4 ablated: E(zero) = E(unablated) exactly (units of s); the failing input changes E.
  6 passes     repeated (position sums, head prefix and copying scores), A (position sums, |a|, head scores), B, narrow (Between on id, prev and their permuted
               labels), natural (|a| and |a| > s over positions >= 1).
  7 heads      induction heads = top 8 by prefix score among heads with a positive copying score (amendment 2); negative-copy heads = top 8 by
               prefix score among negative copying scores; 8 drawn by default_rng(12) from the half of heads with the lowest prefix scores.
  8 signatures E, E', E_early, E_early', P, P', T, T', U, U', R, R_nat, F (NaN at excluded neurons, asserted).
  9 ablation   the 8 induction heads, the random 8 and the 8 negative-copy heads over the repeated set: E_abl, E_rand, E_neg.
 10 detector   repeated row 0 and its pass-2-shuffled copy: the top head's score there (minus its A mean) >= 0.3 and E of the top neuron > 0
               (recorded; the verdict reads it).
 11 families   lines (null 99.9th percentile, E floor 0.2; F line 3), the families (top 300 above the line; unexpected: R >= 0.3 and no line
               cleared, F included), the
               layer-matched random set (default_rng(5)); signatures.npz, heads.npz, families.json, flagged.npz.
 12 contexts   both poles of every flagged neuron: the bank's 50 top contexts, contexts/<pole>.jsonl and contexts/flags.npz.
 13 samples    samples/mechanics.jsonl: 10 random rows and 2 natural windows with 5 named neurons at 8 positions; 5 flagged poles x 3 contexts.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
import torch

from infra.pg import mechanics as M
from infra.pg import neurons_harvest as NH
from infra.pg import unite_route as UR

REPO = "allenai/Olmo-3-1025-7B"
BRANCH = "stage1-step1413814"
REVISION_SHA = "373bad25002f1624757a73235c5ca844c6375c25"
PLAN = {"n_seq": M.N_SEQ, "half": M.HALF, "n_narrow": M.N_NARROW, "min_occ": M.MIN_OCC, "n_nat": M.N_NAT, "n_bins": M.N_BINS, "top_heads": 8,
        "family_cap": 300, "random_n": 300, "r_line": 0.3, "floors": {"E": 0.2}, "null_q": 0.999, "n_id": 64, "batch": 32, "known_ctx": 16,
        "known_batch": 4, "known_tol": 0.05, "id_tol": 1e-6, "detector_line": 0.3, "hw": UR.HARVEST_WINDOW, "n_top_excl": M.N_TOP_EXCL, "early": M.EARLY,
        "f_line": 3.0, "n_copy_rand": M.N_COPY_RAND}
SAMPLE_POS = (5, 40, 80, 120, 133, 168, 208, 248)
SAMPLE_ROWS = (("rep", 0), ("rep", 1), ("rep", 2), ("rep", 3), ("A", 0), ("A", 1), ("B", 0), ("B", 1), ("narrow", 0), ("narrow", 1))
N_SAMPLE_NAT, N_SAMPLE_POLES, N_SAMPLE_CTX = 2, 5, 3
FAMILIES = ("induction", "position", "token", "prevtoken", "first", "unexpected", "random")
SIG_OF = {"induction": "E", "position": "P", "token": "T", "prevtoken": "U", "first": "F", "unexpected": "R"}


def use_eager(model):
    model.set_attn_implementation("eager")
    assert model.config._attn_implementation == "eager", model.config._attn_implementation


def ratio_med(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape and a.ndim == 1 and a.size > 0, (a.shape, b.shape)
    return float(np.median(np.abs(a - b)) / np.median(np.abs(b)))


class Job:
    def __init__(self, out, model, tok, corpus: dict, bank_dir, plan: dict, device: str, meta: dict):
        self.out, self.model, self.tok, self.plan, self.device = pathlib.Path(out), model, tok, plan, device
        for d in ("", "contexts", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bank = pathlib.Path(bank_dir)
        self.bman = json.loads((self.bank / "manifest.json").read_text())
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        cfg = model.config
        self.nL, self.W = len(self.layers), int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.H = int(cfg.num_attention_heads)
        self.hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H)
        self.bos = tok.bos_token_id
        self.hw, self.half = plan["hw"], plan["half"]
        assert cfg.model_type == "olmo3", cfg.model_type
        assert cfg._attn_implementation == "eager", "every forward of this job runs eager attention"
        assert self.layers[0].self_attn.o_proj.in_features == self.H * self.hd, (self.layers[0].self_attn.o_proj.in_features, self.H, self.hd)
        assert all(l.self_attn.o_proj.bias is None for l in self.layers), "the copying score assumes o_proj has no bias"
        b = self.bman
        assert b["revision"] == meta["sha"], f"the bank is {b['revision']}, the weights {meta['sha']}"
        assert b["tokens_sha256"] == corpus["tokens_sha256"], "the corpus is not the one the bank scored"
        assert (b["n_layers"], b["width"], b["bos_id"], b["window"]) == (self.nL, self.W, self.bos, self.hw), (b["n_layers"], b["width"], b["bos_id"], b["window"])
        self.dec = NH.decoder(tok)
        self.V, vrec = M.sampling_ids(tok, self.tokens, plan["n_top_excl"])
        assert int(self.V.max()) < cfg.vocab_size and self.bos not in set(self.V.tolist())
        self.seq = M.sequences(self.V, plan["n_seq"], self.half, plan["n_narrow"])
        self.copy_rand = np.sort(self.V[np.random.default_rng(M.SEEDS["copy_rand"]).choice(self.V.size, plan["n_copy_rand"], replace=False)])
        self.m = {"spec": "studies/mechanics/SPEC_mechanics.md", "status": "running", "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"),
                  "plan": plan, "seeds": M.SEEDS, "n_layers": self.nL, "width": self.W, "n_heads": self.H, "head_dim": self.hd, "bos": self.bos,
                  "attn": cfg._attn_implementation, "dtype": str(next(model.parameters()).dtype), "vocab": {**vrec, "min": int(self.V.min()),
                  "max": int(self.V.max()), "rule": "ids below len(tok), not added tokens, non-zero count in the corpus tokens.npy, not among its n_top most "
                  "frequent ids (amendment 5)"},
                  "bank": {k: b.get(k) for k in ("revision", "branch", "tokens_sha256", "n_windows", "window", "bos_id")}, **meta,
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "gates": {}, "selection": {}}
        self.write()

    # -------------------------------------------------------------------------------------- plumbing
    def write(self):
        self.m["updated_utc"] = NH.utc()
        NH.dump(self.out / "metrics.json", self.m)

    def step(self, name: str, t0: float):
        self.m["timing_s"][name] = round(time.time() - t0, 1)
        self.write()
        print(f"[mechanics] {name} done in {self.m['timing_s'][name]} s", flush=True)

    def x_of(self, ids) -> torch.Tensor:
        ids = torch.as_tensor(np.asarray(ids, np.int64))
        assert ids.dim() == 2
        return torch.cat([torch.full((ids.shape[0], 1), self.bos, dtype=torch.long), ids], 1).to(self.device)

    def fwd(self, x, logits: bool = False):
        self.m["tokens_forwarded"] += int(x.numel())
        self.m["forwards"] += 1
        return NH.forward(self.model, x, logits)

    def pos_sum(self, **kw):
        return M.PosSum(self.nL, 2 * self.half, self.W, self.device, **kw)

    def copy_score(self):
        gf = self.tower.norm.weight
        return M.CopyScore(self.model.get_output_embeddings().weight, [l.self_attn.o_proj.weight for l in self.layers],
                           [gf * l.post_attention_layernorm.weight for l in self.layers], self.H, self.hd, self.half, self.copy_rand)

    @torch.no_grad()
    def run_pass(self, ids, pos=None, heads=None, between=None, labels=None, ablation=None, copy=None):
        """Forward the rows of ids (BOS prepended) in batches with the given accumulators."""
        B, N = self.plan["batch"], len(ids)
        cur = {}

        def fn(L, a):
            if pos is not None:
                pos.update(L, a)
            for k, acc in (between or {}).items():
                acc.update(L, a, cur[k])
        with contextlib.ExitStack() as st:
            st.enter_context(M.tap(self.layers, fn))
            if heads is not None:
                st.enter_context(M.attn_tap(self.layers, heads.update))
            if ablation is not None:
                st.enter_context(M.ablate(self.layers, ablation, self.H, self.hd))
            if copy is not None:
                st.enter_context(M.o_tap(self.layers, lambda L, z: copy.update(L, z, cur["copy"])))
            for b0 in range(0, N, B):
                for k in (between or {}):
                    cur[k] = torch.as_tensor(labels[k][b0:b0 + B], device=self.device)
                if copy is not None:
                    cur["copy"] = torch.as_tensor(M.CopyScore.targets(ids[b0:b0 + B], self.half), device=self.device)
                self.fwd(self.x_of(ids[b0:b0 + B]))

    # -------------------------------------------------------------------------------------- 1, 2: weights and scale
    def weights(self):
        t0 = time.time()
        WU = self.model.get_output_embeddings().weight.float()
        gf = self.tower.norm.weight
        cols = [M.wu_columns(WU, l.mlp.down_proj.weight, gf, l.post_feedforward_layernorm.weight) for l in self.layers]
        self.wcols = {k: np.stack([c[k] for c in cols]) for k in cols[0]}
        del WU
        self.m["weights"] = {"rule": "w_norm = ||W_down[:, k]||; over the vocabulary of W_U w_k, fp32: wu_mean, wu_std_ratio (population std / w_norm), wu_skew "
                                     "(m3 / m2^1.5), wu_kurtosis (m4 / m2^2, normal = 3); _eff: the same for W_U (g_final * g_post * w_k), std / ||g_post * w_k||",
                             "tied_embeddings": bool(self.model.config.tie_word_embeddings)}
        self.step("weights", t0)

    def scale(self):
        t0 = time.time()
        s = []
        for L in range(self.nL):
            with np.load(self.bank / "stats" / f"layer_{L:02d}.npz") as z:
                q = z["q999"]
            assert q.shape == (self.W, 2), q.shape
            s.append(M.scale_from_q999(q))
        self.s = np.stack(s)
        self.valid = np.isfinite(self.s)
        self.s_t = torch.as_tensor(self.s, dtype=torch.float32, device=self.device)
        bad = np.argwhere(~self.valid)
        self.m["gates"]["scale"] = {"field": "stats/layer_<L>.npz q999 [W, 2] (per pole); s_k = the larger of the two (amendment 1); zero or non-finite excluded", "n_read": int(self.s.size), "n_expected": self.nL * self.W,
                                    "n_excluded": int(len(bad)), "excluded": bad[:200].tolist(), "median_s": float(np.nanmedian(self.s)),
                                    "pass": bool(self.s.size == self.nL * self.W and self.valid.any()), "failing_input": None}
        self.step("scale", t0)

    # -------------------------------------------------------------------------------------- 3, 4: known activation and identity
    @torch.no_grad()
    def capture_at(self, toks, L: int, n: int) -> np.ndarray:
        """a of neuron n of layer L at each global position in toks, the full harvest-window prefix, BOS first."""
        x, pos = UR.pack(self.tokens, np.asarray(toks, np.int64), None, self.bos, self.hw, self.device)
        got = []
        kb = self.plan["known_batch"]
        for b0 in range(0, len(toks), kb):
            xb, pb = x[b0:b0 + kb], pos[b0:b0 + kb]
            xb = xb[:, :int(pb.max()) + 1]
            rows = torch.arange(len(pb), device=self.device)

            def fn(i, a):
                if i == L:
                    got.append(a[rows, pb, n].float().cpu())
            with M.tap(self.layers, fn):
                self.fwd(xb)
        return torch.cat(got).numpy()

    def known(self):
        t0 = time.time()
        K = self.plan["known_ctx"]
        rng = np.random.default_rng(M.SEEDS["known_poles"])
        poles = []
        for L in (self.nL // 16, self.nL // 2, self.nL - 2):
            with np.load(self.bank / "topk" / f"layer_{L:02d}.npz") as z:
                tk, ak = z["tok"], z["a"]
            for _ in range(100):
                n = int(rng.integers(self.W))
                if (tk[n, 0, :K] >= 0).all():
                    break
            assert (tk[n, 0, :K] >= 0).all(), f"no full + pole drawn in layer {L}"
            poles.append({"pole": NH.pole_id(L, n, 0), "layer": L, "neuron": n, "tok": tk[n, 0, :K], "a_bank": ak[n, 0, :K].astype(np.float64)})
        for p in poles:
            p["a"] = self.capture_at(p["tok"], p["layer"], p["neuron"])
        r = [ratio_med(p["a"], p["a_bank"]) for p in poles]
        rw = [ratio_med(p["a"], poles[(i + 1) % len(poles)]["a_bank"]) for i, p in enumerate(poles)]
        e1b = [float(np.median(np.abs(p["a"] - p["a_bank"])) / np.abs(p["a_bank"]).max()) for p in poles]
        v, vw, tol = float(np.median(r)), float(np.median(rw)), self.plan["known_tol"]
        self.m["gates"]["known_activation"] = {
            "value": v, "tol": tol, "pass": v <= tol, "failing_input": {"input": "the next pole's bank (cyclic)", "value": vw}, "failing_input_failed": vw > tol,
            "rule": "median over 3 poles of median |a_job - topk.a| / median |topk.a| at 16 top contexts, full harvest-window prefix, BOS first",
            "e1b_ratio": float(np.median(e1b)), "e1b_rule": "median |diff| / max |topk.a| (E1b's units, line 1e-2), reported",
            "per_pole": {p["pole"]: {"ratio": r[i], "wrong_ratio": rw[i], "e1b_ratio": e1b[i]} for i, p in enumerate(poles)}, "n_ctx": K}
        NH.dump(self.out / "samples" / "known_activation.json",
                {p["pole"]: [{"rank": j, "tok": int(p["tok"][j]), **NH.context(self.tokens, self.offsets, int(p["tok"][j]), self.dec), "a_job": float(p["a"][j]),
                              "a_bank": float(p["a_bank"][j])} for j in range(2)] for p in poles})
        self.step("known", t0)
        g = self.m["gates"]["known_activation"]
        assert g["pass"] and g["failing_input_failed"], f"known activation: {g}"

    @torch.no_grad()
    def identity(self):
        t0 = time.time()
        x = self.x_of(self.seq["rep"][:2])
        ref = self.fwd(x, True).logits.float()
        pos, hs, cs = self.pos_sum(), M.HeadScore(self.nL, self.H, self.half, self.device), self.copy_score()
        tg = torch.as_tensor(M.CopyScore.targets(self.seq["rep"][:2], self.half), device=self.device)
        with M.tap(self.layers, pos.update), M.attn_tap(self.layers, hs.update), M.ablate(self.layers, {}, self.H, self.hd), \
                M.o_tap(self.layers, lambda L, z: cs.update(L, z, tg)):
            hooked = self.fwd(x, True).logits.float()
        L = self.nL // 2
        k = int(np.flatnonzero(self.valid[L])[0])
        with M.write_neuron(self.layers, L, k, float(self.s[L, k])):
            wrong = self.fwd(x, True).logits.float()
        den = float(ref.abs().max())
        v, vw, tol = float((hooked - ref).abs().max()) / den, float((wrong - ref).abs().max()) / den, self.plan["id_tol"]
        self.m["gates"]["identity"] = {"value": v, "tol": tol, "pass": v <= tol, "failing_input": {"input": f"neuron ({L}, {k}) written: + s_k at every position", "value": vw},
                                       "failing_input_failed": vw > tol, "rule": "max |logits hooked - unhooked| / max |unhooked| over 2 repeated rows; hooks: "
                                       "every down_proj tap, every attention tap, the ablation hook with zero heads, the copying-score tap on o_proj"}
        self.step("identity", t0)
        g = self.m["gates"]["identity"]
        assert g["pass"] and g["failing_input_failed"], f"identity: {g}"

    # -------------------------------------------------------------------------------------- 5, 6: the passes and the heads
    def passes(self):
        t0 = time.time()
        rep, hr, cs = self.pos_sum(), M.HeadScore(self.nL, self.H, self.half, self.device), self.copy_score()
        self.run_pass(self.seq["rep"], pos=rep, heads=hr, copy=cs)
        self.n_rep, self.S_rep, self.head_rep, self.head_copy = rep.count(), rep.S.cpu().numpy(), hr.mean(), cs.mean()
        del rep, hr, cs
        self.step("pass_repeated", t0)
        t0 = time.time()
        A, ha = self.pos_sum(), M.HeadScore(self.nL, self.H, self.half, self.device)
        self.run_pass(self.seq["A"], pos=A, heads=ha)
        self.n_A, self.S_A, self.Q_A, self.head_A = A.count(), A.S.cpu().numpy(), A.Q.cpu().numpy(), ha.mean()
        del A, ha
        B = self.pos_sum()
        self.run_pass(self.seq["B"], pos=B)
        self.n_B, self.S_B = B.count(), B.S.cpu().numpy()
        del B
        self.step("pass_A_B", t0)
        t0 = time.time()
        self.lab = M.narrow_labels(self.seq["narrow"], self.seq["narrow_ids"], self.plan["min_occ"])
        bt = {k: M.Between(self.nL, self.plan["n_narrow"], self.W, self.device) for k in M.LABELS}
        self.run_pass(self.seq["narrow"], between=bt, labels=self.lab)
        self.ratio = {k: bt[k].ratio() for k in M.LABELS}
        self.bt = {"id": bt["id"], "prev": bt["prev"]}
        del bt
        lab = self.lab
        self.m["narrow"] = {"n_ids": int(self.plan["n_narrow"]), "ids_kept": int((lab["id_count"] >= self.plan["min_occ"]).sum()),
                            "prev_kept": int((lab["prev_count"] >= self.plan["min_occ"]).sum()), "tokens_id": int((lab["id"] >= 0).sum()),
                            "tokens_prev": int((lab["prev"] >= 0).sum()), "count_min_median_max": [int(lab["id_count"].min()), float(np.median(lab["id_count"])),
                                                                                                int(lab["id_count"].max())]}
        self.step("pass_narrow", t0)
        t0 = time.time()
        self.nat_w, self.nat = M.natural_windows(self.tokens, self.bman["n_windows"], self.hw, self.plan["n_nat"], 2 * self.half)
        nat = self.pos_sum(abs_from=1, scale=self.s_t)
        self.run_pass(self.nat, pos=nat)
        n_tok = nat.count() * (2 * self.half - 1)
        self.R_nat = nat.Q[:, 1:].double().sum(1).cpu().numpy() / n_tok / self.s
        self.exceed_nat = np.where(self.valid, nat.X.double().cpu().numpy() / n_tok, np.nan)
        del nat
        self.m["gates"]["scale"]["scale_known"] = {"median_exceedance": float(np.nanmedian(self.exceed_nat)), "mean_exceedance": float(np.nanmean(self.exceed_nat)),
                                                   "expected_at_most": 1 - M.SCALE_Q,
                                                   "rule": "share of natural tokens (harvest window positions 1..255) with |a| > s_k, median and mean over "
                                                           "neurons; reported (s is the larger pole's 99.9th percentile at a bin's upper edge, so the share runs below 0.001)"}
        self.step("pass_natural", t0)

    def heads(self):
        score, copy = self.head_rep - self.head_A, self.head_copy
        order = np.argsort(-score.ravel(), kind="stable")
        k = self.plan["top_heads"]
        cell = lambda i: (int(i // self.H), int(i % self.H))
        self.score = score
        self.top_heads = [cell(i) for i in order if copy.ravel()[i] > 0][:k]
        self.neg_heads = [cell(i) for i in order if copy.ravel()[i] < 0][:k]
        assert self.top_heads and self.neg_heads, "no head with a positive (or a negative) copying score"
        bottom = order[len(order) - len(order) // 2:]
        self.rand_heads = [cell(i) for i in np.sort(np.random.default_rng(M.SEEDS["random_heads"]).choice(bottom, k, replace=False))]
        row = lambda L, h: {"layer": L, "head": h, "prefix": float(score[L, h]), "copy": float(copy[L, h]), "repeated": float(self.head_rep[L, h]),
                            "A": float(self.head_A[L, h])}
        self.m["heads"] = {"induction": [row(*x) for x in self.top_heads], "negative_copy": [row(*x) for x in self.neg_heads],
                           "random_bottom_half": [row(*x) for x in self.rand_heads], "n_positive_copy": int((copy > 0).sum()),
                           "rule": "prefix: mean attention weight from position half + i to i + 1 (i = 1..half-2) on the repeated set minus on A; copy: amendment 2 "
                                   "(mechanics.CopyScore); induction = top 8 by prefix among copy > 0; negative_copy = top 8 by prefix among copy < 0 (reported, "
                                   "ablated, no rule); random = 8 from the half of heads with the lowest prefix score"}
        self.write()

    # -------------------------------------------------------------------------------------- 7, 8, 9: signatures, ablation, detector
    def signatures(self):
        t0 = time.time()
        s, h, nb = self.s, self.half, self.plan["n_bins"]
        self.diff = {"rep": M.half_diff(self.S_rep, self.n_rep, h), "A": M.half_diff(self.S_A, self.n_A, h), "B": M.half_diff(self.S_B, self.n_B, h)}
        stop = h + self.plan["early"]
        early = {k: M.half_diff(S, n, h, stop) for k, S, n in (("rep", self.S_rep, self.n_rep), ("A", self.S_A, self.n_A), ("B", self.S_B, self.n_B))}
        bins = {k: M.bin_means(S, n, nb) for k, S, n in (("rep", self.S_rep, self.n_rep), ("A", self.S_A, self.n_A), ("B", self.S_B, self.n_B))}
        perm = np.random.default_rng(M.SEEDS["bin_perm"]).permutation(2 * h)
        bperm = M.bin_means(self.S_B, self.n_B, nb, perm)
        rng_ = lambda b: (b.max(-1) - b.min(-1)) / s
        nan = lambda x: np.where(self.valid, x, np.nan)
        self.sig = {"E": (self.diff["rep"] - self.diff["A"]) / s, "E_null": (self.diff["B"] - self.diff["A"]) / s, "P": rng_(bins["A"]), "P_null": rng_(bperm),
                    "T": nan(self.ratio["id"]), "T_null": nan(self.ratio["id_perm"]), "U": nan(self.ratio["prev"]), "U_null": nan(self.ratio["prev_perm"]),
                    "R": self.Q_A.astype(np.float64).sum(1) / (self.n_A * 2 * h) / s, "R_nat": self.R_nat, "exceed_nat": self.exceed_nat,
                    "E_early": (early["rep"] - early["A"]) / s, "E_early_null": (early["B"] - early["A"]) / s, "F": nan(M.first_ratio(self.Q_A))}
        self.bins = {f"bins_{k}": v / s[..., None] for k, v in bins.items()}
        for k, v in self.sig.items():
            assert np.isnan(v[~self.valid]).all(), f"{k}: an excluded neuron holds a value"
        self.m["gates"]["scale"]["nan_at_excluded"] = True
        self.m["signatures"] = {k: {"n_finite": int(np.isfinite(v).sum()), "median": float(np.nanmedian(v)), "p999": float(np.nanquantile(v, 0.999))}
                                for k, v in self.sig.items()}
        self.step("signatures", t0)

    def ablation(self):
        t0 = time.time()
        h, s = self.half, self.s
        as_dict = lambda hs: {L: [x for (l, x) in hs if l == L] for L in sorted({l for l, _ in hs})}
        for name, hs in (("abl", self.top_heads), ("rand", self.rand_heads), ("neg", self.neg_heads)):
            p = self.pos_sum()
            self.run_pass(self.seq["rep"], pos=p, ablation=as_dict(hs))
            self.diff[f"rep_{name}"] = M.half_diff(p.S.cpu().numpy(), p.count(), h)
            self.sig[f"E_{name}"] = (self.diff[f"rep_{name}"] - self.diff["A"]) / s
            del p
        self.step("ablation", t0)

    def ablation_identity(self):
        """Before the passes: the first n_id repeated rows with the repeated pass's hooks (unablated), with the ablation hook and zero heads, and
        with all heads of layer L/4 ablated (the failing input); E differences in units of s (D_A cancels)."""
        t0 = time.time()
        h, n_id, Lf = self.half, self.plan["n_id"], self.nL // 4
        D = {}
        for name, kw in (("unablated", {"heads": M.HeadScore(self.nL, self.H, h, self.device), "copy": self.copy_score()}), ("zero", {"ablation": {}}),
                         ("layer", {"ablation": {Lf: list(range(self.H))}})):
            p = self.pos_sum()
            self.run_pass(self.seq["rep"][:n_id], pos=p, **kw)
            D[name] = M.half_diff(p.S.cpu().numpy(), p.count(), h)
            del p
        v = float(np.nanmax(np.abs(D["zero"] - D["unablated"]) / self.s))
        vw = float(np.nanmax(np.abs(D["layer"] - D["unablated"]) / self.s))
        self.m["gates"]["ablation_identity"] = {"value": v, "tol": 0.0, "pass": v == 0.0, "failing_input": {"input": f"all {self.H} heads of layer {Lf} ablated",
                                                "value": vw}, "failing_input_failed": vw > 0.0, "n_rows": n_id,
                                                "rule": "max over valid neurons of |E(zero heads ablated) - E(unablated)| on the first n_id repeated rows (batches as in "
                                                        "the passes), units of s"}
        self.step("ablation_identity", t0)
        g = self.m["gates"]["ablation_identity"]
        assert g["pass"] and g["failing_input_failed"], f"ablation identity: {g}"

    @torch.no_grad()
    def detector(self):
        t0 = time.time()
        Lh, hh = self.top_heads[0]
        i = int(np.nanargmax(np.where(self.valid, self.sig["E"], -np.inf)))
        L, n = i // self.W, i % self.W
        q, k = M.head_index(self.half)
        out = {}
        for name, row in (("perfect", self.seq["rep"][0]), ("shuffled", M.shuffled_pass2(self.seq["rep"][0], self.half))):
            got = {}

            def fa(j, a):
                if j == L:
                    got["a"] = a[0, 1:, n].double().cpu().numpy()

            def fw(j, w):
                if j == Lh:
                    assert w is not None, "no attention weights"
                    got["w"] = float(w[0, hh][torch.as_tensor(q, device=w.device), torch.as_tensor(k, device=w.device)].double().mean())
            with M.tap(self.layers, fa), M.attn_tap(self.layers, fw):
                self.fwd(self.x_of(row[None]))
            a = got["a"]
            out[name] = {"score": got["w"] - float(self.head_A[Lh, hh]), "E_top": float((a[self.half:].mean() - a[:self.half].mean() - self.diff["A"][L, n]) / self.s[L, n])}
        ok = lambda r: bool(r["score"] >= self.plan["detector_line"] and r["E_top"] > 0)
        self.m["gates"]["detector"] = {"top_head": [Lh, hh], "top_neuron": [L, n], "line": self.plan["detector_line"], **out["perfect"], "pass": ok(out["perfect"]),
                                       "failing_input": {"input": "repeated row 0 with pass 2 shuffled (default_rng(14))", **out["shuffled"]},
                                       "failing_input_failed": not ok(out["shuffled"]),
                                       "rule": "on repeated row 0: the top head's score (its attention half + i -> i + 1 minus its A mean) >= 0.3 and E of the top neuron > 0"}
        self.step("detector", t0)

    # -------------------------------------------------------------------------------------- 10: families and the banks
    def families(self):
        t0 = time.time()
        sig, valid, p = self.sig, self.valid, self.plan
        ln = M.lines({k: sig[f"{k}_null"] for k in "EPTU"}, valid, p["null_q"], p["floors"])
        above = {k: valid & (sig[k] > ln[k]) for k in "EPTU"}
        null_above = {k: valid & (sig[f"{k}_null"] > ln[k]) for k in "EPTU"}
        above["F"] = valid & (sig["F"] >= p["f_line"])
        ln["F"] = p["f_line"]
        anyl = above["E"] | above["P"] | above["T"] | above["U"] | above["F"]
        unexp = valid & (sig["R"] >= p["r_line"]) & ~anyl
        fam = {f: M.top_cells(sig[SIG_OF[f]], above[SIG_OF[f]], p["family_cap"]) for f in ("induction", "position", "token", "prevtoken", "first")}
        fam["unexpected"] = M.top_cells(sig["R"], unexp, p["family_cap"])
        fam["random"] = M.layer_matched(fam["induction"], p["random_n"], valid)
        self.fam = fam
        self.flagged = list(dict.fromkeys(c for f in FAMILIES for c in fam[f]))
        sets = {f: set(fam[f]) for f in FAMILIES}
        self.member = {c: [f for f in FAMILIES if c in sets[f]] for c in self.flagged}
        counts = {f"N_{k}": int(above[k].sum()) for k in "EPTUF"}
        counts.update({f"N_{k}_null": int(null_above[k].sum()) for k in "EPTU"})
        counts.update({"N_unexpected": int(unexp.sum()), "N_R_ge_line": int((valid & (sig["R"] >= p["r_line"])).sum()), "N_valid": int(valid.sum())})
        self.m["selection"] = {"lines": ln, "line_rule": f"the {p['null_q']} quantile over valid neurons of the null (E', P', T', U'), floors {p['floors']}; F: the fixed line {p['f_line']}",
                               "counts": counts, "family_sizes": {f: len(v) for f, v in fam.items()}, "n_flagged_neurons": len(self.flagged),
                               "random_rule": "layer counts of the induction family by largest remainder (uniform when empty), neurons drawn without replacement "
                                              "outside it, default_rng(5)"}
        np.savez_compressed(self.out / "signatures.npz", **{k: v.astype(np.float32) for k, v in sig.items()}, **{k: v.astype(np.float32) for k, v in self.bins.items()},
                            **{f"D_{k}": v.astype(np.float32) for k, v in self.diff.items()}, **{k: v.astype(np.float32) for k, v in self.wcols.items()},
                            s=self.s.astype(np.float32), valid=valid)
        np.savez_compressed(self.out / "heads.npz", score=self.score, copy=self.head_copy, repeated=self.head_rep, A=self.head_A,
                            top=np.array(self.top_heads, np.int64).reshape(-1, 2), negative=np.array(self.neg_heads, np.int64).reshape(-1, 2),
                            random=np.array(self.rand_heads, np.int64), copy_rand_ids=self.copy_rand)
        NH.dump(self.out / "families.json", {"lines": ln, "counts": counts, "families": {f: [list(c) for c in v] for f, v in fam.items()}})
        fl = np.array(self.flagged, np.int64).reshape(-1, 2)
        pm = lambda S, n: np.stack([S[L, :, k] / n for L, k in self.flagged]).astype(np.float32) if self.flagged else np.zeros((0, 2 * self.half), np.float32)
        idm = {k: np.zeros((len(self.flagged), p["n_narrow"]), np.float32) for k in ("id", "prev")}
        for L in sorted({c[0] for c in self.flagged}):
            rows = [j for j, c in enumerate(self.flagged) if c[0] == L]
            for k in ("id", "prev"):
                idm[k][rows] = self.bt[k].means(L, torch.as_tensor([self.flagged[j][1] for j in rows], device=self.device))
        np.savez_compressed(self.out / "flagged.npz", layer=fl[:, 0], neuron=fl[:, 1], posmean_rep=pm(self.S_rep, self.n_rep), posmean_A=pm(self.S_A, self.n_A),
                            posmean_B=pm(self.S_B, self.n_B), posabs_A=pm(self.Q_A, self.n_A), idmean=idm["id"], prevmean=idm["prev"], narrow_ids=self.seq["narrow_ids"],
                            id_count=self.lab["id_count"], prev_count=self.lab["prev_count"],
                            member=np.array([[f in self.member[c] for f in FAMILIES] for c in self.flagged], bool).reshape(-1, len(FAMILIES)),
                            families=np.array(FAMILIES))
        del self.bt
        self.step("families", t0)

    # -------------------------------------------------------------------------------------- 11: contexts
    def contexts(self):
        t0 = time.time()
        rec = {"pole": [], "layer": [], "neuron": [], "sign": [], "tok": [], "a": [], "repeat": []}
        n_ctx = 0
        for L in sorted({c[0] for c in self.flagged}):
            with np.load(self.bank / "topk" / f"layer_{L:02d}.npz") as z:
                tk, ak = z["tok"], z["a"]
            for (l, k) in (c for c in self.flagged if c[0] == L):
                for sg in (0, 1):
                    pole = NH.pole_id(L, k, sg)
                    fl = np.full(tk.shape[-1], -1, np.int8)
                    lines = []
                    for r in range(tk.shape[-1]):
                        t = int(tk[k, sg, r])
                        if t < 0:
                            break
                        win = M.window_ids(self.tokens, t, self.hw, 2 * self.half)
                        fl[r] = M.repeat_flag(win)
                        lines.append(json.dumps({"pole": pole, "layer": L, "neuron": k, "sign": "+-"[sg], "families": self.member[(L, k)], "rank": r, "tok": t,
                                                 **NH.context(self.tokens, self.offsets, t, self.dec), "a": float(ak[k, sg, r]), "marked_id": int(self.tokens[t]),
                                                 "window_ids": win.tolist(), "repeat": bool(fl[r])}))
                    (self.out / "contexts" / f"{pole}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
                    n_ctx += len(lines)
                    for key, v in (("pole", pole), ("layer", L), ("neuron", k), ("sign", sg), ("tok", tk[k, sg]), ("a", ak[k, sg]), ("repeat", fl)):
                        rec[key].append(v)
        np.savez_compressed(self.out / "contexts" / "flags.npz", **{k: np.array(v) for k, v in rec.items()})
        self.m["contexts"] = {"n_poles": len(rec["pole"]), "n_contexts": n_ctx, "window": f"the {2 * self.half} tokens ending at the marked token, inside its harvest window",
                              "text": "neurons_harvest.context: 64 before and 16 after, inside the document", "repeat_rule": "the 3-gram (two preceding ids, marked id) occurs earlier in the window"}
        self.step("contexts", t0)

    # -------------------------------------------------------------------------------------- 12: samples
    @torch.no_grad()
    def samples(self):
        t0 = time.time()
        named = {}
        for k in ("E", "P", "T", "U", "F"):
            i = int(np.nanargmax(np.where(self.valid, self.sig[k], -np.inf)))
            named[f"top_{k}"] = (i // self.W, i % self.W)
        rows = [(f"{st}", j, self.seq[st][j]) for st, j in SAMPLE_ROWS] + [("natural", int(self.nat_w[j]), self.nat[j]) for j in range(N_SAMPLE_NAT)]
        got = {c: [] for c in named.values()}

        def fn(L, a):
            for (l, n) in got:
                if l == L:
                    got[(l, n)].append(a[:, 1:, n].double().cpu().numpy())
        with M.tap(self.layers, fn):
            self.fwd(self.x_of(np.stack([r[2] for r in rows])))
        lines = []
        for j, (st, idx, ids) in enumerate(rows):
            neu = []
            for name, (L, n) in named.items():
                a = got[(L, n)][0][j]
                neu.append({"name": name, "layer": L, "neuron": n, "s": float(self.s[L, n]), "a": [float(a[p]) for p in SAMPLE_POS],
                            "a_over_s": [float(a[p] / self.s[L, n]) for p in SAMPLE_POS], **{k: float(self.sig[k][L, n]) for k in ("E", "P", "T", "U", "F", "R")}})
            lines.append({"kind": "natural" if st == "natural" else "sequence", "set": st, "index": idx, "ids": [int(t) for t in ids], "text": self.dec(ids),
                          "positions": list(SAMPLE_POS), "tokens_at": [self.dec(ids[p:p + 1]) for p in SAMPLE_POS], "neurons": neu})
        order = [c for grp in itertools.zip_longest(*[self.fam[f] for f in FAMILIES]) for c in grp if c is not None]
        pool = list(dict.fromkeys(order))[:N_SAMPLE_POLES]
        for (L, k) in pool:
            pole = NH.pole_id(L, k, 0)
            ctx = [json.loads(x) for x in (self.out / "contexts" / f"{pole}.jsonl").read_text().splitlines()[:N_SAMPLE_CTX]]
            lines.append({"kind": "pole", "pole": pole, "families": self.member[(L, k)], **{k2: float(self.sig[k2][L, k]) for k2 in ("E", "E_early", "E_abl", "P", "T", "U", "F", "R")},
                          "contexts": [{x: c[x] for x in ("rank", "before", "token", "after", "a", "repeat")} for c in ctx]})
        (self.out / "samples" / "mechanics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        kinds = {k: sum(x["kind"] == k for x in lines) for k in ("sequence", "natural", "pole")}
        want = {"sequence": len(SAMPLE_ROWS), "natural": N_SAMPLE_NAT, "pole": N_SAMPLE_POLES}
        self.m["gates"]["samples"] = {"lines": kinds, "want": want, "pass": kinds == want, "path": "samples/mechanics.jsonl",
                                      "named_neurons": {k: list(v) for k, v in named.items()}, "failing_input": "the file missing: every rule cancelled (the verdict)"}
        self.step("samples", t0)

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        t0 = time.time()
        self.weights()
        self.scale()
        self.known()
        self.identity()
        self.ablation_identity()
        self.passes()
        self.heads()
        self.signatures()
        self.ablation()
        self.detector()
        self.families()
        self.contexts()
        self.samples()
        self.m["timing_s"]["run"] = round(time.time() - t0, 1)
        self.m["status"] = "done"
        self.write()


def fetch_bank(src: str, fetch) -> pathlib.Path:
    """The grid bank's manifest, stats/ and topk/ (S3 under /work/fetch/bank; a local directory as it is)."""
    if not src.startswith("s3://"):
        return pathlib.Path(src)
    src = src.rstrip("/")
    for d in ("stats", "topk"):
        fetch(f"{src}/{d}", f"bank/{d}", recursive=True)
    return fetch(f"{src}/manifest.json", "bank/manifest.json").parent


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--revision", required=True, choices=[BRANCH])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bank", required=True)
    return ap


def main():
    a = parser().parse_args()
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    t0 = time.time()
    local, sha, rec = download(REPO, a.revision)
    assert sha == REVISION_SHA, f"{a.revision} resolved to {sha}, not {REVISION_SHA}"
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    bank = fetch_bank(a.bank, fetch)
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    use_eager(model)
    meta = {"t": a.t, "revision": a.revision, "sha": sha, "download": rec, "download_corpus_bank_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1),
            "corpus": a.corpus, "bank_src": a.bank, "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()}
    job = Job(a.out, model, tok, corpus, bank, PLAN, "cuda", meta)
    job.run()
    job.m["memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    job.m["wall_s"] = round(time.time() - t0, 1)
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("MECHANICS_SCREEN_OK", json.dumps({"t": a.t, "wall_s": job.m["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
