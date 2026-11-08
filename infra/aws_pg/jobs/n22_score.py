"""N22 job (studies/next/07_PICKS/N22_SPEC.md §9): slot-dependent arithmetic on OLMo-3 7B at the stage-1 checkpoints.

One call visits the points of `--points` in order; each point has a kind:
  full      L1 (T_15) and the N22 lens points of L3: every step below.
  accuracy  L2: steps 1-3, 5 and the pool forward of step 6.
  n25       the N25 lens points of L3 (T_0, the early point): steps 1-6 (readout on every item), the probes, N25.
Per point t:
  1. Download the branch; G-REV, G-CFG; G-TOK (the branch's tokenizer re-encodes 20 registered prompts and has the
     registered vocabulary); load bf16 / sdpa (neurons_harvest.load_model).
  2. K0, the known number: mean next-token NLL on the first 120,000 corpus targets by the banked harvest's rule
     (windows of BOS + 2,048 in order, batches of 8, bf16 logits cast to fp32). Gate: within 1e-6 of the bank where
     the bank was made on this GPU model; at the other-device points (T_7-T_14) within the coarse tier 0.01, and the
     difference reported against 2 x the T_14 device pair. Failing input: the neighbouring point's banked value.
  3. The massive-activation scan: the 24 lens prompts forwarded (bf16); per block the max |h_i| per coordinate, the
     median |h|, the top-3 coordinates; residuals at 128 sampled positions (N25's statistics).
  4. (full, n25) The lens: 24 prompts of BOS + 127 corpus tokens in two halves of 12 (A, B), target block 30,
     source blocks 0-29, the vendored estimator (jlens.fitting.jacobian_for_prompt) under deterministic algorithms
     and the sdpa math kernel (n12_score.grad_mode, lesson 2). I-GRAD: prompt 0 again, relative Frobenius difference
     <= 1e-6 at every layer; failing input: prompt 1's Jacobian.
  5. Upcast to fp32, matmul precision "highest" (lesson 1). Every later forward, logit and log-probability is fp32.
  6. The pool forward (1,200 items): greedy top-1, the log-probabilities of the 1,000 number tokens, the
     last-position residual after every block. Readout through five lenses (full, A, B, the logit lens, the full
     lens with the massive coordinates projected out) at every source layer: ranks of s1, s2, y, the partner's s1,
     s2, y and the four operands. LENS_FIDELITY (gated at full points, reported at n25 points): per item the cosine
     between the full lens's transport of block 29's residual and block 30's own residual, against the same lens fed
     a randomly rotated residual; pass if the paired lower bound is above 0 (failing input: the rotated residual in
     place of the residual). The cosine of the untransported residual (the logit lens) is reported beside it.
  7. (full) Bands and delta_read from the freeze half's answered items (n22.band_rule); the class means
     (n22.class_prompts, last position, every block); the ten raw examples (every arm at v index 0) written before
     any other edit; I-FWD (the pool again in reversed batch order, max |d lp| <= 1e-4; failing input: each item
     against its partner); I-EDIT (the lens swap with sigma the identity equals the clean forward; failing input
     h + 0.01 |h| r on the band); the edits of n22.ARMS on the answered items of each half (first N_SWAP_MAX); the
     DoM sweep (freeze half, v index 0); the interchange control at the operand positions (freeze half at every
     site of n22.IC_SITES, test half at the embedding) with its random-direction floor; G-IC-ID (the patch at the
     embedding equals the counterfactual's own forward; failing input: the item's clean forward); the probes (rule
     5); N25's statistics.
out/n22/metrics.json carries gates, counts, timings and the projection, never an interaction, a share or a label.

    python -m infra.aws_pg.jobs.n22_score --items <uri> --corpus <uri> --points 15:full --budget-min 160 --out <dir>

Fail-loud: no try/except, no fallback, no retry.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np

from infra.aws_pg.jobs import m2_score as M
from infra.aws_pg.jobs.neurons_harvest import fetch, fetch_corpus
from infra.pg import m2, n22

HUB = None
ITEMS_SHA = "dbf0b2251516edeb869053c72f2c9e6fe4bab2c9250799f308fbda489bba6e69"   # studies/next/n22/items.json
CUBLAS_CFG = ":4096:8"
VENDOR = str(pathlib.Path(n22.__file__).resolve().parent / "vendor")
BATCH = 64
NAMES = ["s1", "s2", "y", "s1x", "s2x", "yx", "a", "b", "c", "d"]   # readout targets ("x": the partner's)
VARIANTS = ["full", "A", "B", "logit", "proj"]


class Run(M.Run):
    def save(self, phase: str):
        self.rec["phase"], self.rec["updated_utc"] = phase, M.utc()
        self.rec["elapsed_s"] = round(time.time() - self.t0, 1)
        M.dump(self.out / "n22" / "metrics.json", self.rec)
        print(f"[n22] {phase}", flush=True)


def dump(p, obj):
    M.dump(pathlib.Path(p), obj)


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def hash_seed(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def download(branch: str, revision: str):
    hub = HUB
    if hub is None:
        from huggingface_hub import snapshot_download as hub
    t0 = time.time()
    local = pathlib.Path(hub(m2.REPO, revision=branch, allow_patterns=M.ALLOW))
    dt = time.time() - t0
    n = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    rec = {"branch": branch, "commit": local.name, "bytes": n, "seconds": round(dt, 1),
           "MB_per_s": round(n / 1e6 / max(dt, 1e-9), 1)}
    rec["G_REV"] = {"pass": local.name == revision, "commit": local.name, "registered": revision,
                    "failing_input": {"pass": local.name == "0" * 40}}
    return local, rec


# ------------------------------------------------------------------------------------------------ hooks and edits
class Hooks:
    """Forward hooks on the embedding (site -1) or a block's output (site >= 0); fn(tensor) -> tensor."""

    def __init__(self, tower):
        self.tower, self.h = tower, []

    def on(self, site: int, fn):
        import torch
        mod = self.tower.embed_tokens if site == -1 else self.tower.layers[site]

        def hook(m, args, out):
            t = out if torch.is_tensor(out) else out[0]
            t2 = fn(t)
            return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])
        self.h.append(mod.register_forward_hook(hook))

    def record(self, site: int, into: dict, pos=None):
        def fn(t):
            into[site] = (t if pos is None else t[:, pos]).detach().float().clone()
            return t
        self.on(site, fn)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for x in self.h:
            x.remove()


def unit_random(shape, gen, device):
    import torch
    r = torch.randn(shape, generator=gen, device=device)
    return r / r.norm(dim=-1, keepdim=True)


def lens_edit(t, V, q0, rand=False, gen=None, sigma=True):
    """The paper's coordinate swap at every query position (n22.lens_swap_delta); rand: the same norm per position,
    a random direction."""
    t = t.clone()
    d = n22.lens_swap_delta(t[:, q0:], V, sigma)
    if rand:
        d = d.norm(dim=-1, keepdim=True) * unit_random(d.shape, gen, d.device)
    t[:, q0:] = t[:, q0:] + d
    return t


def dom_edit(t, dv, pos, rand=False, gen=None):
    """The mean-difference patch (the paper's §A.24.1) at one position; rand: the same norm, a random direction."""
    t = t.clone()
    d = dv.norm(dim=-1, keepdim=True) * unit_random(dv.shape, gen, dv.device) if rand else dv
    t[:, pos] = t[:, pos] + d
    return t


def last_lp(model, x):
    import torch
    out = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
    return torch.log_softmax(out.logits[:, -1].float(), -1)


# ------------------------------------------------------------------------------------------------ the known number
def k0_nll(model, tokens: np.ndarray, reg: dict, device: str) -> float:
    """neurons_harvest.run's corpus NLL: windows in order, batches of `batch`, all logits, the first `targets`."""
    import torch
    import torch.nn.functional as F
    from infra.pg.neurons_harvest import forward
    W, B, n_t = reg["k0"]["window"], reg["k0"]["batch"], reg["k0"]["targets"]
    s, n, w0 = 0.0, 0, 0
    with torch.inference_mode():
        while n < n_t:
            x = torch.from_numpy(tokens[w0 * W:(w0 + B) * W].astype(np.int64)).view(B, W)
            x = torch.cat([torch.full((B, 1), reg["bos"], dtype=torch.long), x], 1).to(device)
            res = forward(model, x, True)
            for w in range(B):
                nll = F.cross_entropy(res.logits[w, :-1].float(), x[w, 1:], reduction="none")
                take = min(n_t - n, nll.numel())
                s += float(nll[:take].sum())
                n += take
                if n >= n_t:
                    break
            w0 += B
    return s / n


def k0_gate(t: int, nll: float, reg: dict, device_name: str) -> dict:
    bank = reg["k0_bank"][t]
    nb = reg["k0_bank"][t + 1 if t < 15 else t - 1]
    same = reg["points"][t][4] == device_name
    tol = n22.K0_TOL_SAME if same else n22.K0_TOL_COARSE
    g = {"pass": abs(nll - bank) <= tol, "nll": nll, "bank": bank, "diff": nll - bank, "tol": tol, "same_device": same,
         "failing_input": {"pass": abs(nll - nb) <= tol, "neighbour": nb, "diff": nll - nb}}
    if not same:
        g["cross_device"] = {"ref": n22.K0_REF_CROSS, "within_ref": abs(nll - bank) <= n22.K0_REF_CROSS}
    return g


# ------------------------------------------------------------------------------------------------ the lens
def lens_prompts(tokens: np.ndarray, reg: dict):
    import torch
    rng = np.random.default_rng(n22.SEED_LENS)
    L = reg["lens"]["len"] - 1
    starts = sorted(int(x) * L for x in rng.choice(tokens.size // L, size=reg["lens"]["n"], replace=False))
    bos = torch.tensor([[reg["bos"]]])
    return [torch.cat([bos, torch.from_numpy(tokens[s:s + L].astype(np.int64))[None]], 1) for s in starts], starts


def massive_scan(model, tower, prompts, device, n_pos: int = 128):
    import torch
    L, d = len(tower.layers), tower.norm.weight.numel()
    mx = torch.zeros(L, d)
    absall = [[] for _ in range(L)]
    rng = np.random.default_rng(n22.SEED_LENS + 1)
    pos = sorted((int(rng.integers(0, len(prompts))), int(rng.integers(16, prompts[0].shape[1] - 1)))
                 for _ in range(n_pos))
    keep = torch.zeros(n_pos, L, d)
    with torch.inference_mode():
        for j, x in enumerate(prompts):
            rec = {}
            with Hooks(tower) as H:
                for lyr in range(L):
                    H.record(lyr, rec)
                model(input_ids=x.to(device), use_cache=False, logits_to_keep=1)
            for lyr in range(L):
                a = rec[lyr][0].cpu()
                mx[lyr] = torch.maximum(mx[lyr], a.abs().max(0).values)
                absall[lyr].append(a.abs().flatten())
                for k, (jj, p) in enumerate(pos):
                    if jj == j:
                        keep[k, lyr] = a[p]
    med = torch.stack([torch.cat(x).median() for x in absall])
    ratio = mx.max(1).values / med
    return {"top": mx.topk(n22.MASSIVE_K, dim=1).indices.tolist(), "ratio": ratio.tolist(),
            "max_ratio": float(ratio.max()), "median_abs": med.tolist(), "sample_positions": pos}, keep


class IdsLens:
    """A jlens LensModel over pre-tokenized prompts: encode(key) returns the registered ids."""

    def __init__(self, hfl, prompts):
        self.hfl, self.prompts = hfl, prompts
        self.n_layers, self.d_model, self.layers, self.tokenizer = hfl.n_layers, hfl.d_model, hfl.layers, hfl.tokenizer

    def encode(self, key, *, max_length=512):
        return self.prompts[key][:, :max_length].to(self.hfl.input_device)

    def forward(self, ids):
        return self.hfl.forward(ids)

    def unembed(self, r):
        return self.hfl.unembed(r)


def fit_lens(model, tok, prompts, sources, target, dim_batch):
    """Two half-lenses (prompts [0, n/2) and [n/2, n)); I-GRAD: prompt 0 again (failing input: prompt 1's J)."""
    import torch
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from jlens.fitting import jacobian_for_prompt
    from jlens.hf import from_hf
    from infra.aws_pg.jobs.n12_score import grad_mode
    lm = IdsLens(from_hf(model, tok), prompts)
    half = len(prompts) // 2
    sums = [{lyr: torch.zeros(lm.d_model, lm.d_model) for lyr in sources} for _ in range(2)]
    keep, times = {}, []
    with grad_mode():
        for j in range(len(prompts)):
            t0 = time.time()
            J, _, _ = jacobian_for_prompt(lm, j, sources, target_layer=target, dim_batch=dim_batch,
                                          max_seq_len=prompts[j].shape[1])
            times.append(time.time() - t0)
            for lyr in sources:
                sums[j // half][lyr] += J[lyr]
            if j < 2:
                keep[j] = J
        J2, _, _ = jacobian_for_prompt(lm, 0, sources, target_layer=target, dim_batch=dim_batch,
                                       max_seq_len=prompts[0].shape[1])
    rel = [float((J2[lyr] - keep[0][lyr]).norm() / keep[0][lyr].norm()) for lyr in sources]
    relf = [float((keep[1][lyr] - keep[0][lyr]).norm() / keep[0][lyr].norm()) for lyr in sources]
    g = {"pass": max(rel) <= n22.TOL_GRAD, "max_rel": max(rel), "bit_identical": max(rel) == 0.0, "tol": n22.TOL_GRAD,
         "failing_input": {"pass": max(relf) <= n22.TOL_GRAD, "what": "prompt 1's Jacobian", "min_rel": min(relf)}}
    A = {lyr: sums[0][lyr] / half for lyr in sources}
    B = {lyr: sums[1][lyr] / (len(prompts) - half) for lyr in sources}
    return A, B, g, times


# ------------------------------------------------------------------------------------------------ one point
class Point:
    def __init__(self, t, kind, model, tower, tok, reg, items, out, device):
        self.t, self.kind, self.model, self.tower, self.tok, self.reg = t, kind, model, tower, tok, reg
        self.items, self.dev = items, device
        self.L = len(tower.layers)
        self.target = self.L - 2
        self.sources = list(range(self.target))
        self.upper = list(range(self.L // 2, self.target))
        self.ic_sites = [-1] + list(range(0, self.L, 4))
        self.dom_sweep = list(range(0, self.target, 2))
        if self.L == n22.N_LAYERS:
            assert (self.target, self.sources, self.upper, self.ic_sites, self.dom_sweep) == \
                (n22.TARGET, n22.SOURCES, n22.UPPER, n22.IC_SITES, n22.DOM_SWEEP), "layer constants"
        self.dir = out / "n22" / f"t{t:02d}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def fp32(self):
        import torch
        self.model.float()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.num = torch.tensor(self.reg["number_ids"], device=self.dev)
        self.g = self.tower.norm.weight.detach().float()
        self.WU = self.model.lm_head.weight.detach().float()

    def x(self, rows):
        import torch
        return torch.tensor(rows, dtype=torch.long, device=self.dev)

    def readout_logits(self, h, J):
        """h [n, d] fp32 at a source block; J [d, d] or None (the logit lens). lm_head(norm(J h))."""
        return self.model.lm_head(self.tower.norm(h if J is None else h @ J.T)).float()

    def scored(self, it, v):
        p = self.items[it["partner"]]
        f1, f2 = v * it["c"] + it["d"], v + it["d"]
        vals = {"f1": f1, "f2": f2, "v": v, "y": it["y"], "s1": it["s1"], "s2": it["s2"], "f1m": max(f1 - 1, 0),
                "f1p": min(f1 + 1, 999), "f2m": max(f2 - 1, 0), "f2p": min(f2 + 1, 999),
                "f1x": v * p["c"] + p["d"], "f2x": v + p["d"]}
        return [vals[k] for k in n22.SCORED]

    def rows_lp(self, lp, metas):
        """One record per row: the fp32 log-probabilities of the scored number tokens, the top-1 and its lp."""
        import torch
        top = lp.max(-1)
        out = []
        for b, m in enumerate(metas):
            vals = lp[b, self.num[torch.tensor(m["vals"], device=lp.device)]].tolist()
            out.append({k: v for k, v in m.items() if k != "vals"} | {
                "lp": dict(zip(n22.SCORED, vals)), "top1": int(top.indices[b]), "top1_lp": float(top.values[b])})
        return out

    def forward_rows(self, ids, setup, metas):
        import torch
        x = self.x(ids)
        with torch.inference_mode(), Hooks(self.tower) as H:
            setup(H)
            lp = last_lp(self.model, x)
        return self.rows_lp(lp, metas), lp

    def pool_forward(self, order):
        import torch
        n = len(self.items)
        lp_num, top1, top1_lp = torch.zeros(n, 1000), torch.zeros(n, dtype=torch.long), torch.zeros(n)
        H = torch.zeros(n, self.L, self.g.numel())
        last = n22.QPOS["last"]
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            rec = {}
            with torch.inference_mode(), Hooks(self.tower) as Hk:
                for lyr in range(self.L):
                    Hk.record(lyr, rec, pos=last)
                lp = last_lp(self.model, self.x([self.items[j]["ids"] for j in idx]))
            lp_num[idx] = lp[:, self.num].cpu()
            mx = lp.max(-1)
            top1[idx], top1_lp[idx] = mx.indices.cpu(), mx.values.cpu()
            H[idx] = torch.stack([rec[lyr].cpu() for lyr in range(self.L)], 1)
        return lp_num, top1, top1_lp, H

    def class_means(self, classes, probe_layers):
        """Mean last-position residual per (slot, value) over its class prompts at every block; the residuals at
        `probe_layers` for the probes."""
        import torch
        rows = [(int(k), int(x), q["ids"]) for k, xs in classes.items() for x, qs in xs.items() for q in qs]
        sums, probe = {}, {}
        last = n22.QPOS["last"]
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            rec = {}
            with torch.inference_mode(), Hooks(self.tower) as Hk:
                for lyr in range(self.L):
                    Hk.record(lyr, rec, pos=last)
                self.model(input_ids=self.x([r[2] for r in chunk]), use_cache=False, logits_to_keep=1)
            hs = torch.stack([rec[lyr] for lyr in range(self.L)], 1)        # [b, L, d]
            for b, (k, xv, _) in enumerate(chunk):
                sums.setdefault((k, xv), []).append(hs[b])
        mu = {key: torch.stack(v).mean(0) for key, v in sums.items()}
        probe = {key: torch.stack(v)[:, probe_layers].cpu() for key, v in sums.items()}
        return mu, probe


def edit_rows(P, its, band, lm, lv, mu, vis, arms=n22.ARMS):
    """Every arm on the rows (item, vi) of `its`: the lens swap (s_k <-> v) over the upper half (LENS_U) or the
    matched band (LENS_M) at every query position; the answer substitution (y <-> f_k(v)) over the same layers
    (ANS_U, ANS_M); the mean-difference patch at the last position of block lm (DOM_M); each with its random-direction
    floor of matched norm (RAND_U, RAND_M, RAND_DOM)."""
    import torch
    out = []
    last = n22.QPOS["last"]
    for arm in arms:
        layers = {"LENS_U": P.upper, "ANS_U": P.upper, "RAND_U": P.upper,
                  "LENS_M": band, "ANS_M": band, "RAND_M": band, "DOM_M": [lm], "RAND_DOM": [lm]}[arm]
        for slot in (1, 2):
            rows = [(it, vi) for it in its for vi in vis]
            for bi in range(0, len(rows), BATCH):
                chunk = rows[bi:bi + BATCH]
                metas, src, dst, dv = [], [], [], []
                for it, vi in chunk:
                    v = it["v"][vi]
                    vals = P.scored(it, v)
                    metas.append({"item": it["i"], "vi": vi, "arm": arm, "slot": slot, "vals": vals})
                    a, b = (it["y"], vals[n22.SCORED.index(f"f{slot}")]) if arm.startswith("ANS") else (it[f"s{slot}"], v)
                    src.append(a)
                    dst.append(b)
                    if arm in ("DOM_M", "RAND_DOM"):
                        dv.append(mu[(slot, v)][lm] - mu[(slot, it[f"s{slot}"])][lm])
                gen = torch.Generator(device=P.dev).manual_seed(hash_seed(P.t, arm, slot, bi))

                def setup(H, arm=arm, layers=layers, src=src, dst=dst, dv=dv, gen=gen):
                    for lyr in layers:
                        if arm in ("DOM_M", "RAND_DOM"):
                            d_ = torch.stack(dv)
                            H.on(lyr, lambda t, d_=d_: dom_edit(t, d_, last, arm == "RAND_DOM", gen))
                        else:
                            V = torch.stack([lv[lyr][src], lv[lyr][dst]], -1)
                            H.on(lyr, lambda t, V=V: lens_edit(t, V, n22.Q0, arm.startswith("RAND"), gen))
                out += P.forward_rows([it["ids"] for it, _ in chunk], setup, metas)[0]
    return out


def dom_sweep(P, its, mu):
    """The paper's §A.24.1 layer sweep of the mean-difference patch (v index 0, freeze half): reported."""
    import torch
    out = []
    last = n22.QPOS["last"]
    for lyr in P.dom_sweep:
        for slot in (1, 2):
            for bi in range(0, len(its), BATCH):
                chunk = its[bi:bi + BATCH]
                metas = [{"item": it["i"], "vi": 0, "arm": f"DOM_L{lyr:02d}", "slot": slot, "vals": P.scored(it, it["v"][0])}
                         for it in chunk]
                d_ = torch.stack([mu[(slot, it["v"][0])][lyr] - mu[(slot, it[f"s{slot}"])][lyr] for it in chunk])
                out += P.forward_rows([it["ids"] for it in chunk],
                                      lambda H, lyr=lyr, d_=d_: H.on(lyr, lambda t: dom_edit(t, d_, last)), metas)[0]
    return out


def ic_rows(P, its, sites, half, lp_clean):
    """The interchange control (SPEC §5.1) per (item, v of the first N_V_IC, slot): the counterfactual's forward
    (recording the residual at every site), the patch of the slot's operand positions with it at each site >= 0, the
    random-direction floor (the same norm per position) at every site. G-IC-ID on the first batch: the patch at the
    embedding against the counterfactual's own forward; failing input: against the item's clean forward."""
    import torch
    out, idg = [], None
    jobs = [(it, vi, slot) for it in its for vi in range(n22.N_V_IC) for slot in (1, 2)]
    for bi in range(0, len(jobs), BATCH):
        chunk = jobs[bi:bi + BATCH]
        metas = [{"item": it["i"], "vi": vi, "slot": slot, "vals": P.scored(it, it["ic"][vi]["v"])} for it, vi, slot in chunk]
        rec = {}
        with torch.inference_mode(), Hooks(P.tower) as H:
            for s in sites:
                H.record(s, rec)
            lp_cf = last_lp(P.model, P.x([it["ic"][vi][f"ids{slot}"] for it, vi, slot in chunk]))
        out += [r | {"site": -1, "kind": "IC", "half": half} for r in P.rows_lp(lp_cf, metas)]
        ids = [it["ids"] for it, _, _ in chunk]
        mask = torch.zeros(len(chunk), len(ids[0]), 1, device=P.dev)
        for b, (_, _, slot) in enumerate(chunk):
            mask[b, n22.IC_POS[slot]] = 1.0
        kinds = [(s, k) for s in sites for k in (("IC", "RAND_IC") if s >= 0 else ("RAND_IC",))]
        if bi == 0:
            kinds.append((-1, "IC_ID"))
        for s, kind in kinds:
            gen = torch.Generator(device=P.dev).manual_seed(hash_seed(P.t, half, s, kind, bi))

            def fn(t, s=s, kind=kind, gen=gen):
                hc = rec[s].to(t.dtype)
                if kind == "RAND_IC":
                    return t + mask * ((hc - t).norm(dim=-1, keepdim=True) * unit_random(t.shape, gen, t.device))
                return t * (1 - mask) + hc * mask
            rows, lp = P.forward_rows(ids, lambda H, s=s, fn=fn: H.on(s, fn), metas)
            if kind == "IC_ID":
                d_cf = float((lp[:, P.num] - lp_cf[:, P.num]).abs().max())
                d_cl = float((lp[:, P.num].cpu() - lp_clean[[it["i"] for it, _, _ in chunk]]).abs().max())
                idg = {"pass": d_cf <= n22.TOL_FWD, "max_abs": d_cf, "tol": n22.TOL_FWD, "n": len(chunk),
                       "failing_input": {"pass": d_cl <= n22.TOL_FWD, "what": "the item's clean forward", "max_abs": d_cl}}
                continue
            out += [r | {"site": s, "kind": kind, "half": half} for r in rows]
    return out, idg


def probes(P, probe, layers):
    """Rule 5 (12_SHAPING_NOTE.md §4 item 6): linear and one-hidden-layer MLP probes of the slot value from the class
    prompts' last-position residuals (12 train / 4 test per class), beside the same with the labels permuted."""
    import torch
    res = {}
    for slot in (1, 2):
        keys = sorted(k for k in probe if k[0] == slot)
        X = torch.stack([probe[k] for k in keys])                              # [C, n, layers, d]
        C, n = X.shape[0], X.shape[1]
        y = torch.arange(C).repeat_interleave(n)
        tr = torch.tensor([j < 12 for j in range(n)] * C)
        perm = y[torch.from_numpy(np.random.default_rng(n22.SEED).permutation(y.numel()))]
        for li, lyr in enumerate(layers):
            Z = X[:, :, li].reshape(C * n, -1).to(P.dev)
            Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)
            for lab_name, lab in (("real", y), ("permuted", perm)):
                lab = lab.to(P.dev)
                for kind in ("linear", "mlp"):
                    torch.manual_seed(n22.SEED)
                    d = Z.shape[1]
                    net = (torch.nn.Linear(d, C) if kind == "linear" else
                           torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, C)))
                    net = net.to(P.dev).float()
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
                    with torch.enable_grad():
                        for _ in range(300):
                            opt.zero_grad()
                            torch.nn.functional.cross_entropy(net(Z[tr]), lab[tr.to(P.dev)]).backward()
                            opt.step()
                    with torch.no_grad():
                        res[f"s{slot}_L{lyr:02d}_{lab_name}_{kind}"] = float(
                            (net(Z[~tr]).argmax(-1) == lab[(~tr).to(P.dev)]).float().mean())
        res[f"s{slot}_chance"] = 1.0 / C
    return res


def n25_stats(P, lenses, sample_h):
    """N25 (SPEC §7) per lens variant: the CKA block structure of the lens vectors of 2,000 tokens across source
    blocks, the median excess kurtosis of the lens logits at 128 positions (max over blocks), the MLP gain of 256
    lens vectors against 256 isotropic directions at the block's median residual norm (max over blocks)."""
    import torch
    rng = np.random.default_rng(n22.SEED_TOKS)
    nv = min(P.WU.shape[0], P.reg["bos"])
    toks = torch.from_numpy(rng.choice(nv, size=min(2000, nv), replace=False)).to(P.dev)
    U = P.WU[toks] * P.g
    rnd = unit_random((256, U.shape[1]), torch.Generator(device=P.dev).manual_seed(n22.SEED), P.dev)
    out = {}
    for name, J in lenses.items():
        cg = []
        for lyr in P.sources:
            Mv = (U @ J[lyr].to(P.dev)).double()
            K = Mv @ Mv.T
            K = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
            cg.append(K / K.norm())
        L = len(P.sources)
        cka = np.array([[float((cg[i] * cg[j]).sum()) for j in range(L)] for i in range(L)])
        kurt, gain = [], []
        for lyr in P.sources:
            Jd = J[lyr].to(P.dev)
            h = sample_h[:, lyr].to(P.dev)
            kurt.append(float(n22.excess_kurtosis(P.readout_logits(h, Jd)).median()))
            s = float(h.norm(dim=-1).median())
            vv = U[:256] @ Jd
            vv = vv / vv.norm(dim=-1, keepdim=True)
            mlp = P.tower.layers[lyr + 1].mlp
            with torch.no_grad():
                gain.append(float(mlp(s * vv).norm(dim=-1).median() / mlp(s * rnd).norm(dim=-1).median()))
        out[name] = {"cka_block": n22.cka_block(cka), "kurtosis_max": max(kurt), "gain_max": max(gain),
                     "kurtosis": kurt, "gain": gain, "cka": cka.tolist()}
    return out


def examples(items, correct):
    """The first ten freeze-half items, plus the first of the missing kind (right or wrong) if all ten agree."""
    fz = [it for it in items if it["half"] == "freeze"]
    ex = fz[:10]
    for want in (False, True):
        if all(bool(correct[it["i"]]) != want for it in ex):
            ex = ex + [x for x in fz if bool(correct[x["i"]]) == want][:1]
    return ex


def write_examples(P, ex, rows, ranks, lp_num, top1, top1_lp, bands, lens_full, H):
    import torch
    dec = lambda i: P.tok.decode([int(i)])                       # noqa: E731
    path = P.dir.parent / "samples" / f"raw_examples_t{P.t:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for it in ex:
        i = it["i"]
        rd = {}
        for key in ("l1", "l2", "lm"):
            lyr = bands[key]
            tp = torch.softmax(P.readout_logits(H[i:i + 1, lyr].to(P.dev), lens_full[lyr].to(P.dev))[0], -1).topk(10)
            rd[key] = {"layer": lyr, "top10": [[dec(j), round(float(p), 4)] for p, j in zip(tp.values, tp.indices)],
                       "rank_s1": int(ranks[i, lyr, 0, 0]), "rank_s2": int(ranks[i, lyr, 0, 1])}
        after = {f"{r['arm']}_slot{r['slot']}": {"top1": dec(r["top1"]), "top1_p": round(float(np.exp(r["top1_lp"])), 4),
                                                 "lp": {k: round(v, 3) for k, v in r["lp"].items()}}
                 for r in rows if r["item"] == i}
        v = it["v"][0]
        recs.append({"item": i, "half": it["half"], "prompt": P.tok.decode(it["ids"][1:]),
                     "s1": it["s1"], "s2": it["s2"], "y": it["y"], "v": v, "f1": v * it["c"] + it["d"],
                     "f2": v + it["d"], "model_top1": dec(top1[i]), "model_top1_p": round(float(np.exp(top1_lp[i])), 4),
                     "p_y": round(float(np.exp(lp_num[i, it["y"]])), 4),
                     "correct": int(top1[i]) == P.reg["number_ids"][it["y"]],
                     "rank_s1_by_layer": ranks[i, :, 0, 0].tolist(), "rank_s2_by_layer": ranks[i, :, 0, 1].tolist(),
                     "readout": rd, "after_edits": after})
    write_jsonl(path, recs)


def run_point(R, a, reg, items, corpus, t, kind):
    import torch
    import transformers
    from infra.pg import m1
    from infra.pg import neurons_harvest as NH
    from infra.pg.matrix_models import vocab_sha256
    pt = {"t": t, "kind": kind, "branch": reg["points"][t][2]}
    R.rec["points"][str(t)] = pt
    t_fix = time.time()
    local, pt["download"] = download(reg["points"][t][2], reg["points"][t][3])
    R.gate(f"t{t:02d}_G_REV", pt["download"].pop("G_REV"))
    cfg = json.loads((local / "config.json").read_text())
    g = m1.config_gate(cfg, reg["ref_config"])
    g["pass"] = bool(g["pass"] and g["diff_keys"] == [])
    g["failing_input"] = m1.config_gate({**cfg, "model_type": "olmo2-retrofit"}, reg["ref_config"])
    R.gate(f"t{t:02d}_G_CFG", g)
    tok = transformers.AutoTokenizer.from_pretrained(str(local))
    chk = items[:20]
    got = [n22.prompt_ids(tok, (it["a"], it["b"], it["c"], it["d"]), reg["bos"]) for it in chk]
    R.gate(f"t{t:02d}_G_TOK", {"pass": vocab_sha256(tok) == reg["vocab_sha256"] and
                               all(x == it["ids"] for x, it in zip(got, chk)), "n": len(chk),
                               "failing_input": {"pass": got[0] == chk[1]["ids"], "what": "item 0's ids against item 1's"}})
    t_load = time.time()
    model, _ = NH.load_model(str(local), "O7", device=a.device)
    pt["load_s"] = round(time.time() - t_load, 1)
    tower = NH.text_tower(model)
    g = k0_gate(t, k0_nll(model, corpus["tokens"], reg, a.device), reg, a.device_name)
    R.gate(f"t{t:02d}_K0", g)
    if not g["same_device"]:
        R.report(f"t{t:02d}_K0_cross_device", g["cross_device"] | {"diff": g["diff"]})
    P = Point(t, kind, model, tower, tok, reg, items, R.out, a.device)
    prompts, starts = lens_prompts(corpus["tokens"], reg)
    massive, sample_h = massive_scan(model, tower, prompts, a.device)
    dump(P.dir / "massive.json", massive | {"lens_prompt_starts": starts})
    pt["massive_max_ratio"] = massive["max_ratio"]
    pt["fixed_s"] = round(time.time() - t_fix, 1)
    lenses = {}
    if kind in ("full", "n25"):
        t_l = time.time()
        A, B, gg, times = fit_lens(model, tok, prompts, P.sources, P.target, reg["lens"]["dim_batch"])
        R.gate(f"t{t:02d}_I_GRAD", gg)
        pt["lens_s"], pt["lens_prompt_s"] = round(time.time() - t_l, 1), round(float(np.mean(times)), 3)
        torch.save({"A": {k: v.half() for k, v in A.items()}, "B": {k: v.half() for k, v in B.items()},
                    "n_prompts": len(prompts), "sources": P.sources, "target": P.target}, P.dir / "lens_AB.pt")
        lenses = {"A": A, "B": B, "full": {k: (A[k] + B[k]) / 2 for k in A}}
    P.fp32()
    if a.device == "cuda":
        torch.cuda.empty_cache()
    t_f = time.time()
    order = list(range(len(items)))
    lp_num, top1, top1_lp, H = P.pool_forward(order)
    pt["pool_forward_s"] = round(time.time() - t_f, 1)
    y_ids = torch.tensor([reg["number_ids"][it["y"]] for it in items])
    correct = (top1 == y_ids).numpy()
    if reg.get("dry_all_answered"):             # dry run only (a tiny random model answers nothing); never registered
        correct = np.ones_like(correct)
    np.save(P.dir / "clean_lp_numbers.npy", lp_num.numpy())
    pt["answered"] = {h: int(sum(correct[it["i"]] for it in items if it["half"] == h)) for h in ("freeze", "test")}
    dump(P.dir / "accuracy.json", {
        "correct": correct.astype(int).tolist(), "top1": top1.tolist(), "top1_lp": top1_lp.tolist(),
        "lp_y": [float(lp_num[it["i"], it["y"]]) for it in items],
        "partner_y_is_top1": [int(top1[it["i"]]) == reg["number_ids"][items[it["partner"]]["y"]] for it in items]})
    R.save(f"T_{t} answered {pt['answered']}")
    if kind == "accuracy":
        return pt
    # ---- readout through five lenses at every source block
    tid = torch.tensor([[reg["number_ids"][(items[it["partner"]] if k.endswith("x") else it)[k.rstrip("x")]]
                         for k in NAMES] for it in items])
    top = massive["top"]
    proj = {}
    for lyr in P.sources:
        J = lenses["full"][lyr].clone()
        J[:, top[lyr]] = 0
        J[top[P.target], :] = 0
        proj[lyr] = J
    variants = {"full": lenses["full"], "A": lenses["A"], "B": lenses["B"], "logit": None, "proj": proj}
    assert list(variants) == VARIANTS
    ranks = np.zeros((len(items), len(P.sources), len(VARIANTS), len(NAMES)), np.int32)
    for vi_, Jv in enumerate(variants.values()):
        for lyr in P.sources:
            Jd = None if Jv is None else Jv[lyr].to(P.dev)
            for i in range(0, len(items), 300):
                lg = P.readout_logits(H[i:i + 300, lyr].to(P.dev), Jd)
                ranks[i:i + 300, lyr, vi_] = n22.ranks_of(lg, tid[i:i + 300].to(P.dev)).cpu().numpy()
    np.savez_compressed(P.dir / "readout.npz", ranks=ranks, variants=np.array(VARIANTS), targets=np.array(NAMES))
    hl, ht = H[:, P.sources[-1]].to(P.dev), H[:, P.target].to(P.dev)
    Jl = lenses["full"][P.sources[-1]].to(P.dev)
    rot = torch.linalg.qr(torch.randn(hl.shape[1], hl.shape[1], generator=torch.Generator().manual_seed(n22.SEED)))[0].to(P.dev)
    cos = lambda a_, b_: torch.nn.functional.cosine_similarity(a_, b_, dim=-1).cpu().numpy()   # noqa: E731
    c_l, c_r, c_i = cos(hl @ Jl.T, ht), cos((hl @ rot.T) @ Jl.T, ht), cos(hl, ht)
    d_ = n22.boot_mean(c_l - c_r)
    gf = {"pass": bool(d_["lo"] > 0), "cos_lens": float(c_l.mean()), "cos_rotated": float(c_r.mean()),
          "cos_identity": float(c_i.mean()), "lens_minus_rotated": d_,
          "failing_input": {"pass": bool(n22.boot_mean(c_r - c_r)["lo"] > 0),
                            "what": "the rotated residual in place of the residual"}}
    if kind == "full":
        R.gate(f"t{t:02d}_LENS_FIDELITY", gf)
    else:
        R.report(f"t{t:02d}_LENS_FIDELITY", gf)
    # ---- design: bands and delta_read from the freeze half's answered items (n25 points: every freeze item)
    fz = [it for it in items if it["half"] == "freeze" and (correct[it["i"]] or kind == "n25")]
    te = [it for it in items if it["half"] == "test" and (correct[it["i"]] or kind == "n25")]
    design = {"t": t, "kind": kind, "n_answered": pt["answered"], "n_swap": {}}
    ix = np.array([it["i"] for it in fz])
    if ix.size:
        def net(vname, k):
            v_ = VARIANTS.index(vname)
            return ((ranks[ix, :, v_, NAMES.index(k)] <= n22.K_READ).mean(0) -
                    (ranks[ix, :, v_, NAMES.index(k + "x")] <= n22.K_READ).mean(0))
        b = n22.band_rule(net("full", "s1"), net("full", "s2"))
        design |= {"bands": b, "delta_read": float(max(np.abs(net("A", k) - net("B", k)).max() for k in ("s1", "s2"))),
                   "answer_net_at_lm": float(net("full", "y")[b["lm"]]),
                   "net_freeze": {f"{v}_{k}": net(v, k).tolist() for v in VARIANTS for k in ("s1", "s2", "y")}}
    dump(P.dir / "design.json", design)
    probe_layers = sorted(set(P.sources[::3]) | ({design["bands"][k] for k in ("l1", "l2", "lm")} if ix.size else set()))
    t_c = time.time()
    mu, probe = P.class_means(reg["class_prompts"], probe_layers)
    pt["class_s"] = round(time.time() - t_c, 1)
    dump(P.dir / "probes.json", {"layers": probe_layers, "acc": probes(P, probe, probe_layers)})
    dump(P.dir / "n25.json", n25_stats(P, {**lenses, "proj": proj}, sample_h))
    R.save(f"T_{t} readout, probes, n25")
    min_half = min(len(fz), len(te))
    if kind == "n25" or min_half < reg["n_min"]:
        pt["edits"] = "NOT_RUN" if kind == "full" else "n25 point"
        return pt
    band, lm = design["bands"]["band_m"], design["bands"]["lm"]
    lv = {lyr: (P.WU[P.num] * P.g) @ lenses["full"][lyr].to(P.dev) for lyr in P.sources}   # lens vectors of "0".."999"
    # ---- the ten raw examples first
    t_e = time.time()
    ex = examples(items, correct)
    exrows = edit_rows(P, ex, band, lm, lv, mu, [0])
    write_jsonl(P.dir / "raw_example_edits.jsonl", exrows)
    write_examples(P, ex, exrows, ranks, lp_num, top1, top1_lp, design["bands"], lenses["full"], H)
    pt["examples_s"] = round(time.time() - t_e, 1)
    R.save(f"T_{t} raw examples written")
    # ---- identity pairs
    lp2 = P.pool_forward(order[::-1])[0]
    d_id = float((lp2 - lp_num).abs().max())
    d_fail = float((lp_num[[it["partner"] for it in items]] - lp_num).abs().max())
    R.gate(f"t{t:02d}_I_FWD", {"pass": d_id <= n22.TOL_FWD, "max_abs": d_id, "tol": n22.TOL_FWD,
                               "failing_input": {"pass": d_fail <= n22.TOL_FWD, "what": "the partner's forward",
                                                 "max_abs": d_fail}})
    chunk = fz[:BATCH]
    idx = [it["i"] for it in chunk]
    dz = []
    for fail in (False, True):
        gen = torch.Generator(device=P.dev).manual_seed(n22.SEED)

        def setup(H_, fail=fail, gen=gen):
            for lyr in band:
                if fail:
                    H_.on(lyr, lambda tt: tt + 0.01 * tt.norm(dim=-1, keepdim=True) * unit_random(tt.shape, gen, tt.device))
                else:
                    V = torch.stack([lv[lyr][[it["s1"] for it in chunk]], lv[lyr][[it["v"][0] for it in chunk]]], -1)
                    H_.on(lyr, lambda tt, V=V: lens_edit(tt, V, n22.Q0, sigma=False))
        lp = P.forward_rows([it["ids"] for it in chunk], setup, [{"vals": [0]} for _ in chunk])[1]
        dz.append(float((lp[:, P.num].cpu() - lp_num[idx]).abs().max()))
    R.gate(f"t{t:02d}_I_EDIT", {"pass": dz[0] <= n22.TOL_FWD, "max_abs": dz[0], "tol": n22.TOL_FWD,
                                "failing_input": {"pass": dz[1] <= n22.TOL_FWD, "max_abs": dz[1],
                                                  "what": "h + 0.01 |h| r at every position of the band"}})
    # ---- the edits, the sweep, the interchange control
    for h, its in (("freeze", fz), ("test", te)):
        use = its[:reg["n_swap_max"]]
        design["n_swap"][h] = len(use)
        t_s = time.time()
        rows = edit_rows(P, use, band, lm, lv, mu, list(range(n22.N_V)))
        write_jsonl(P.dir / f"edits_{h}.jsonl", rows)
        pt[f"edits_{h}_s"], pt[f"edits_{h}_n"] = round(time.time() - t_s, 1), len(rows)
        t_s = time.time()
        icr, idg = ic_rows(P, use, P.ic_sites if h == "freeze" else [-1], h, lp_num)
        write_jsonl(P.dir / f"ic_{h}.jsonl", icr)
        pt[f"ic_{h}_s"], pt[f"ic_{h}_n"] = round(time.time() - t_s, 1), len(icr)
        R.gate(f"t{t:02d}_G_IC_ID_{h}", idg)
        if h == "freeze":
            t_s = time.time()
            write_jsonl(P.dir / "dom_sweep_freeze.jsonl", dom_sweep(P, use, mu))
            pt["dom_sweep_s"] = round(time.time() - t_s, 1)
        R.save(f"T_{t} edits {h}")
    dump(P.dir / "design.json", design)
    return pt


def parse_points(s: str) -> list:
    out = []
    for x in s.split(","):
        t, k = x.split(":")
        assert k in ("full", "accuracy", "n25"), k
        out.append((int(t), k))
    return out


def run(a, reg, out):
    import torch
    import transformers
    from infra.pg import neurons_corpus as NC
    R = Run(out, a.budget_min * 60)
    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    R.gate("G_SAME", {"pass": v == reg["versions"], "versions": v, "registered": reg["versions"],
                      "failing_input": {"pass": {**v, "transformers": "0.0.0"} == reg["versions"]}})
    ip = fetch(a.items, "inputs/items.json")
    got = m2.sha256_file(ip)
    R.gate("INPUTS", {"pass": got == reg["items_sha"], "got": got, "failing_input": {"pass": "0" * 64 == reg["items_sha"]}})
    rec = json.loads(pathlib.Path(ip).read_text())
    reg = {**reg, "number_ids": rec["number_ids"], "vocab_sha256": rec["vocab_sha256"],
           "class_prompts": rec["class_prompts"]}
    items = rec["items"]
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    R.gate("G_CORPUS", {"pass": corpus["tokens_sha256"] == reg["tokens_sha256"], "got": corpus["tokens_sha256"],
                        "failing_input": {"pass": "0" * 64 == reg["tokens_sha256"]}})
    pts = parse_points(a.points)
    est = {k: v * 3600 for k, v in reg["estimate_h"].items()}       # registered high per-point estimates (n22.project)
    cost = {}
    for j, (t, kind) in enumerate(pts):
        elapsed = time.time() - R.t0
        if j > 0:
            rem = sum(cost.get(k, est[k]) for _, k in pts[j:])
            R.rec["projection"] = {"at": t, "elapsed_s": round(elapsed), "remaining_s": round(rem), "budget_s": R.budget_s}
            if elapsed + rem > 0.9 * R.budget_s:
                R.rec["parked"] = [x for x, _ in pts[j:]]
                R.save(f"parked from T_{t}")
                break
        t0 = time.time()
        pt = run_point(R, a, reg, items, corpus, t, kind)
        pt["wall_s"] = round(time.time() - t0, 1)
        cost[kind] = max(cost.get(kind, 0.0), pt["wall_s"])
        R.rec["measured"] = {"wall_s_by_kind": cost, "lens_prompt_s": pt.get("lens_prompt_s"),
                             "per_pool_forward_s": round(pt["pool_forward_s"] / len(items), 5)}
        gc.collect()
        if a.device == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m2.REPO.replace("/", "--")))
        R.save(f"done T_{t}")
    R.rec["complete"] = not R.rec["parked"]
    R.save("done")


def registry(path=None) -> dict:
    if path is not None:
        r = json.loads(pathlib.Path(path).read_text())
        r["points"] = {int(k): v for k, v in r["points"].items()}
        r["k0_bank"] = {int(k): v for k, v in r["k0_bank"].items()}
        return r
    from infra.pg import m1
    pr = n22.project()["high"]
    return {"versions": M.VERSIONS, "items_sha": ITEMS_SHA, "tokens_sha256": M.TOKENS_SHA,
            "points": {p[0]: list(p) for p in m2.POINTS}, "ref_config": m1.REF_CONFIG, "bos": m2.BOS,
            "k0_bank": n22.K0_BANK, "k0": {"targets": n22.K0_TARGETS, "window": n22.K0_WINDOW, "batch": n22.K0_BATCH},
            "lens": {"n": n22.N_LENS_PROMPTS, "len": n22.LENS_LEN, "dim_batch": n22.DIM_BATCH},
            "n_swap_max": n22.N_SWAP_MAX, "n_min": n22.N_MIN,
            "estimate_h": {"full": pr["per_point_full_s"] / 3600, "accuracy": pr["per_point_accuracy_s"] / 3600,
                           "n25": pr["per_point_n25_s"] / 3600}}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--points", required=True, help="t:kind,... with kind full | accuracy | n25, in visiting order")
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG
    out = pathlib.Path(a.out)
    (out / "n22").mkdir(parents=True, exist_ok=True)
    if a.registry is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    import torch
    reg = registry(a.registry)
    if a.device == "cuda":
        assert not torch.cuda.is_initialized(), "CUDA started before CUBLAS_WORKSPACE_CONFIG was set"
        assert torch.cuda.is_available(), "no CUDA on this instance"
        a.device_name = torch.cuda.get_device_name(0)
    else:
        a.device_name = reg.get("cpu_device_name", "cpu")
    run(a, reg, out)
    print("N22_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
