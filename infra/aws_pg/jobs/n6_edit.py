"""N6 job (J11, edit the young; studies/next/07_PICKS/N6_SPEC.md section 9). One O7 load at T_15 per launch.

    python -m infra.aws_pg.jobs.n6_edit --stage L1|L2 [--half freeze|test] [--against <L2a's metrics.json>] \
        --inputs <s3 prefix or dir: pairs.json, e1_known.json> --corpus <s3> --budget-min <job stop rule> --out <dir>

Amendment A5: L2 runs as two launches on the same pairs file. L2a (--half freeze): the positive control, its probes, the freeze pairs'
edits, the pair probes and the zero edit, and every gate. L2b (--half test, after freeze 2 with G-POWER passed): the test pairs' edits
and the same pair probes and zero edit; before its first gate it asserts that L2a's metrics.json (--against) is a complete freeze-half
launch with the same pairs content, eval windows and eval tokens (sha256). The edit records are the same in both.

Order (every number is written to metrics.json as it exists; a gate whose failure would make later numbers meaningless asserts):
  1. bf16, the model as E1 and the harvest loaded it: G-SAME, G-REV; K1 (E1's banked zero-replacement value, reuse_ablate.readout
     unchanged); K2 (corpus NLL on the first 120,000 targets against the harvest's 2.675372); G-POLE (the first 10 banked T_15
     exemplar activations of every pole in the pairs file, recomputed on full harvest-window prefixes; tolerance 0.05 by amendment
     A1; g, u and a per exemplar written to g_pole.npz); the bf16 rounding probes (G-EPS's failing input).
  2. fp32: the cast (G-FP32); K3 (the same 120,000 targets in fp32, against K2 and the off-by-one failing input); base pass 1 over
     the band-3 eval windows (96 in L1; 1,024 in L2 by amendment A4) (per-token losses, every pole's E+ and its top-50 eval
     positions), the column norms, the positive control of each layer; G-INST (L2: a pass over L1's 96 windows, the values the final
     pairs were matched on); base pass 2 (the identity pair, and the base log-probabilities at every edited pole's 50 eval exemplars).
  3. The progression: the residual h_l of every eval window is kept in host memory (amendment A4); at each layer the windows are
     taken in chunks of --chunk windows: the chunk's h_l goes to the GPU, its h_(l+1) is computed once, every edit of that layer runs
     layers l..31, the final norm and the head from the chunk's h_l with the edit hook on layers[l].mlp.down_proj, and h_(l+1)
     replaces h_l in host memory. A record is written when every chunk of its layer is done. G-CACHE at the end.
  4. G-EPS, G-ID, G-ZN, G-NORM, G-PC, G-DOSE; the raw examples; metrics.json complete.
G-ZN and G-CACHE compare two computation paths (the progression applies the head to one window, the model's forward to a batch);
amendment A2 gives them tolerances frozen from L1-02's measured values, and the job records the head's input shapes per phase.
The peak GPU memory of each phase (torch.cuda.max_memory_allocated, reset after each phase) is in metrics.json measured.peak_mem_gib.

The edit of pole (l, n, s) with vector delta: at every eval position x >= 1 where s a_n(x) > 0 the down projection's output gains
a_n(x) delta, nowhere else (a rank-one change of column n restricted to the pole's own tokens; delta = -kappa w is the column edit,
||delta|| = c = kappa ||w||). Kinds: col (delta = -(c/||w||) w), rand (c r, r a seeded unit vector), col_half (positive control,
-0.5 w), probe (rand's delta added and then subtracted: no effect in exact arithmetic, the rounding floor), zero (delta = 0).
Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time

import numpy as np

from infra.pg import n6

VERSIONS = {"torch": "2.12.0+cu130", "transformers": "5.9.0", "device": "NVIDIA L40S"}
INPUT_SHA = {"L1": {"pairs.json": "adeca76494aa60f70747e64ce981c2119a874f80147c25474d38e79f75b52617",
                    "e1_known.json": "79da230cc099e6981dad1df00bee7c46dfd3b6e890cb9ca7e9a174937e27db61"},
             "L2": {"pairs_L2.json": "7e3a99480b3a2b939ca6eb0a8ae6fe43ee0e8bad22bee302c1a37ca1e4e1755b",   # A3's split, A4's 1,024 eval windows
                    "e1_known.json": "79da230cc099e6981dad1df00bee7c46dfd3b6e890cb9ca7e9a174937e27db61"}}
BOS = 100257
K1_TOL = 1e-3
K2_TARGET, K2_TOL, K2_FAIL = 2.675372037760417, 1e-3, 2.714761865234375    # T_15 harvest (twice, w1 and w3); T_14 the failing input
K3_TOL = 0.02
POLE_TOL = 5e-2     # amendment A1 (2026-11-06, spec section 14): 1e-2 at freeze 1; L1-01's pass side reached 0.0151 (diag/L1-01)
NORM_TOL, REAL_TOL = 1e-6, 1e-4
INST_TOL = 1e-5
TAU_EPS_L1 = 10 * 1.2210622426951182e-08    # amendment A2: the tie margin 10 F_eps at L1-02 (metrics.json gates.G_EPS.value), nats
CACHE_MAX_TOL = 1e-5                         # amendment A2: 10 x L1-02's largest per-token difference 9.537e-7 nats (G_CACHE)
HARVEST_BATCH = 8
TOPK = 50
POLE_EX = 10
CHUNK = 96          # amendment A4: windows per chunk of the progression (the GPU holds h_l and h_(l+1) of 96 windows, as in L1-02)
HOST_MARGIN = 8 * 2 ** 30   # A4: host memory needed beyond the kept residual, bytes (L1-02's largest RSS 2.41 GiB; exemplar logits)


def registry() -> dict:
    """The registered facts the job checks against; the dry run passes a planted dict of the same shape."""
    return {"versions": VERSIONS, "snapshot": n6.SNAPSHOT, "bos": BOS, "k2_target": K2_TARGET, "k2_fail": K2_FAIL,
            "input_sha": INPUT_SHA}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def side(passed: bool, failing_passed: bool, **kw) -> dict:
    """A gate record: the pass side and its registered failing input. The job stops if the gate fails or its failing input passes."""
    return {"pass": bool(passed), "failing_input": {"pass": bool(failing_passed)}, **kw}


def zn_gate(zero_dL: list, first_col_dL: float) -> dict:
    """G-ZN (amendment A2): every zero-norm edit's |dL| <= TAU_EPS_L1. Failing input: the first column edit's dL."""
    v = max(abs(x) for x in zero_dL)
    rec = side(v <= TAU_EPS_L1, abs(first_col_dL) <= TAU_EPS_L1, value_max_abs=v, tol=TAU_EPS_L1, n_zero=len(zero_dL))
    rec["failing_input"]["value"] = first_col_dL
    return rec


def cache_gate(final, base, planted) -> dict:
    """G-CACHE (amendment A2): the per-token losses from the progression's final residual (head per window) against base pass 1
    (the model's forward, head on a batch): |mean difference| <= TAU_EPS_L1 and largest |difference| <= CACHE_MAX_TOL, nats.
    Failing input: the same with each window scored from another window's final residual (`planted`)."""
    def q(lo):
        d = lo.double() - base.double()
        return float(d.mean()), float(d.abs().max()), float(d.abs().mean())
    m, mx, ma = q(final)
    fm, fmx, _ = q(planted)
    rec = side(abs(m) <= TAU_EPS_L1 and mx <= CACHE_MAX_TOL, abs(fm) <= TAU_EPS_L1 and fmx <= CACHE_MAX_TOL, value_mean=m,
               value_max_abs=mx, value_mean_abs=ma, tol_mean=TAU_EPS_L1, tol_max=CACHE_MAX_TOL)
    rec["failing_input"].update(value_mean=fm, value_max_abs=fmx)
    return rec


def rand_unit(d: int, key: str):
    import torch
    g = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    r = torch.randn(d, generator=g, dtype=torch.float64)
    return (r / r.norm()).float()


# ------------------------------------------------------------------------------------------ forward pieces
class Model:
    """The tower's own pieces, called in the order Olmo3Model.forward calls them, so a forward can start from a kept residual."""

    def __init__(self, model):
        from infra.pg import neurons_harvest as NH
        self.model, self.tower = model, NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.cfg = model.config
        self.L = len(self.layers)

    def prep(self, h):
        import torch
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        B, S = h.shape[:2]
        pos = torch.arange(S, device=h.device)[None]
        kw = dict(config=self.cfg, inputs_embeds=h, attention_mask=torch.ones(B, S, dtype=torch.long, device=h.device),
                  past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        return {"pos": pos, "masks": masks, "pe": self.tower.rotary_emb(h, pos)}

    def run(self, h, l0, l1, pr):
        for i in range(l0, l1):
            h = self.layers[i](h, attention_mask=pr["masks"][self.cfg.layer_types[i]], position_ids=pr["pos"],
                               past_key_values=None, position_embeddings=pr["pe"])
        return h

    def embed(self, x):
        return self.tower.embed_tokens(x)

    def window_logits(self, h):
        """[S, V] for one window's final residual h [S, d]."""
        return self.model.lm_head(self.tower.norm(h[None]))[0]


def token_loss(logits, x):
    """Next-token cross-entropy [S-1] in the logits' precision promoted to fp32: input j predicts input j + 1."""
    import torch.nn.functional as F
    return F.cross_entropy(logits[:-1].float(), x[1:], reduction="none")


def windows(tokens, idx, bos, device):
    import torch
    x = torch.from_numpy(np.stack([tokens[w * n6.HW:(w + 1) * n6.HW].astype(np.int64) for w in idx]))
    return torch.cat([torch.full((len(idx), 1), bos, dtype=torch.long), x], 1).to(device)


# ------------------------------------------------------------------------------------------ the edit hook
class Edit:
    """Forward hook on layers[l].mlp.down_proj: out + 1[s a_n > 0, position >= 1] a_n delta (probe: added, then subtracted)."""

    def __init__(self, layer_mod, n: int, sign: int, delta, mode: str):
        assert mode in ("add", "probe")
        self.mod, self.n, self.s, self.delta, self.mode = layer_mod, n, sign, delta, mode
        self.E = 0.0
        self.sq = 0.0
        self.calls = 0

    def hook(self, m, args, out):
        import torch
        a = args[0][..., self.n].float()
        on = self.s * a > 0
        on[:, 0] = False
        coef = torch.where(on, a, torch.zeros_like(a))
        add = coef[..., None] * self.delta.to(out.device)
        self.E += float((coef.double() ** 2).sum())
        self.sq += float((add.double() ** 2).sum())
        self.calls += 1
        y = (out.float() + add).to(out.dtype)
        if self.mode == "probe":
            y = (y.float() - add).to(out.dtype)
        return y

    def __enter__(self):
        self.h = self.mod.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()


# ------------------------------------------------------------------------------------------ the job
class Job:
    def __init__(self, a, model, tok, tokens: np.ndarray, pairs: dict, e1: dict, out: pathlib.Path, device: str, decode, reg: dict):
        import torch
        self.reg, self.bos = reg, reg["bos"]
        self.a, self.model, self.tok, self.tokens, self.P, self.e1 = a, model, tok, tokens, pairs, e1
        self.out, self.device, self.decode = out, device, decode
        self.M = Model(model)
        self.L, self.d, self.W = self.M.L, int(self.M.layers[0].mlp.down_proj.weight.shape[0]), \
            int(self.M.layers[0].mlp.down_proj.weight.shape[1])
        self.win = pairs["eval_windows"]
        self.B, self.C = a.batch, a.chunk
        assert self.C % self.B == 0, f"chunk {self.C} is not a multiple of the batch {self.B}"
        self.t0 = time.time()
        (out / "samples").mkdir(parents=True, exist_ok=True)
        self.rec_path = out / "edits.jsonl"
        self.rec_path.write_text("")
        self.m = {"stage": a.stage, "half": a.half, "started_utc": utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "gates": {},
                  "measured": {}, "n_done": 0, "parked": [], "complete": False, "w_eval": len(self.win), "batch": self.B, "chunk": self.C,
                  "pairs_sha256": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest(),     # A5: the pairs content
                  "eval_windows_sha256": hashlib.sha256(json.dumps(self.win).encode()).hexdigest(),
                  "eval_tokens_sha256": hashlib.sha256(np.stack([tokens[w * n6.HW:(w + 1) * n6.HW].astype(np.int64)
                                                                 for w in self.win]).tobytes()).hexdigest(),
                  "rule": "dL = mean over the W_EVAL x 2,048 eval targets of (loss edited - loss base), fp32 forward, nats per token"}
        self.torch = torch
        self.phase, self.head_shapes = "bf16", {}           # amendment A2: the head's input shapes seen in each phase
        model.lm_head.register_forward_pre_hook(lambda m, args: self.head_shapes.setdefault(self.phase, set()).add(tuple(args[0].shape)))

    def write(self):
        self.m["updated_utc"] = utc()
        self.m["elapsed_s"] = round(time.time() - self.t0, 1)
        dump(self.out / "metrics.json", self.m)

    def peak(self, phase):
        """The peak GPU memory of the phase that ends here (torch.cuda.max_memory_allocated, GiB); the counter is then reset. On the
        CPU (the dry run) there is no GPU memory and the value is None."""
        v = None
        if self.device == "cuda":
            v = round(self.torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            self.torch.cuda.reset_peak_memory_stats()
        self.m["measured"].setdefault("peak_mem_gib", {})[phase] = v
        self.write()

    def gate(self, name, rec, stop=True):
        self.m["gates"][name] = rec
        self.write()
        if stop:
            assert rec["pass"], f"{name} FAILED: {json.dumps(rec)[:600]}"
            assert not rec["failing_input"]["pass"], f"{name}: its failing input passed: {json.dumps(rec)[:600]}"

    # -------------------------------------------------------------------------------------- 1. bf16
    def k1(self):
        from infra.pg import reuse_ablate as RA
        from infra.pg import unite_route as UR
        e = self.e1
        unit = UR.planted_unit({"unit": e["unit"]["unit"], "layer": e["unit"]["layer"], "a": e["unit"]["a"], "b": e["unit"]["b"],
                                "lam": e["unit"]["lam"]})
        kw = dict(tokens=self.tokens, length=e["length"], bos=self.bos, batch=e["batch"], hw=n6.HW, device=self.device)
        before = RA.readout(self.model, e["toks"], unit, None, **kw)["a"]
        abl = RA.ablation_dict([tuple(m) for m in e["members"]], "zero", None, unit["layer"], self.device)
        after = RA.readout(self.model, e["toks"], unit, abl, **kw)["a"]
        ok = before > 0
        jf = float(((before - after) / before)[ok].mean())
        rec = side(abs(jf - e["job_fall"]) <= K1_TOL, abs(e["failing_input_cross_job_fall"] - e["job_fall"]) <= K1_TOL,
                   value=jf, target=e["job_fall"], tol=K1_TOL, a_before=before.tolist(), a_after=after.tolist(),
                   max_abs_diff_before=float(np.abs(before - np.array(e["a_before"])).max()),
                   max_abs_diff_after=float(np.abs(after - np.array(e["a_after"])).max()))
        rec["failing_input"]["value"] = e["failing_input_cross_job_fall"]
        self.peak("load_K1")
        self.gate("K1", rec)

    def corpus_nll(self, shift: int = 1, batch: int = HARVEST_BATCH):
        """The harvest's rule: windows 0, 1, ... with BOS, batches of 8, targets input[:, shift:] from logits[:, :-shift], the first
        120,000 targets. shift = 2 is K3's failing input (targets one position off)."""
        import torch.nn.functional as F
        s, n, w = 0.0, 0, 0
        with self.torch.inference_mode():
            while n < n6.KNOWN_NLL_TOKENS:
                x = windows(self.tokens, list(range(w, w + batch)), self.bos, self.device)
                lg = self.model(input_ids=x, attention_mask=self.torch.ones_like(x), use_cache=False, logits_to_keep=0).logits
                for i in range(x.shape[0]):
                    nll = F.cross_entropy(lg[i, :-shift].float(), x[i, shift:], reduction="none")
                    take = min(n6.KNOWN_NLL_TOKENS - n, nll.numel())
                    s += float(nll[:take].sum())
                    n += take
                    if n >= n6.KNOWN_NLL_TOKENS:
                        break
                del lg
                w += batch
        return s / n

    def pole_check(self):
        """G-POLE: every edited dated pole's first POLE_EX banked T_15 exemplar activations, recomputed on full harvest-window
        prefixes as the harvest computed a (act(g) u in fp32 from the model's g and u); failing input: neuron n + 1 at the same
        positions, read in the same forward. g_pole.npz holds, per pole and exemplar, the recomputed g, u and a of the neuron
        (column 0) and of neuron n + 1 (column 1) beside the bank's a and the positions; it is written before the gate reads."""
        from infra.pg import reuse_ablate as RA
        worst, worst_fail, rows = 0.0, math.inf, []
        keep = {k: [] for k in ("layer", "neuron", "neighbour", "tok", "a_bank", "g", "u", "a")}
        kw = dict(tokens=self.tokens, length=None, bos=self.bos, batch=HARVEST_BATCH, hw=n6.HW, device=self.device)
        for p, q in sorted(self.P["poles"].items()):
            nb = (q["neuron"] + 1) % self.W
            unit = {"layer": q["layer"], "terms": [{"n": q["neuron"], "w": 1.0, "sign": 1.0}, {"n": nb, "w": 1.0, "sign": 1.0}]}
            r = RA.readout(self.model, q["tok15"][:POLE_EX], unit, None, **kw)
            act = self.M.layers[q["layer"]].mlp.act_fn
            g, u = self.torch.from_numpy(r["g"]), self.torch.from_numpy(r["u"])
            a = (act(g) * u).numpy()
            want = np.array(q["a15"][:POLE_EX])
            scale = float(np.abs(want).max())
            e, ef = float(np.abs(a[:, 0] - want).max() / scale), float(np.abs(a[:, 1] - want).max() / scale)
            worst, worst_fail = max(worst, e), min(worst_fail, ef)
            rows.append([p, e, ef])
            for k, v in (("layer", q["layer"]), ("neuron", q["neuron"]), ("neighbour", nb), ("tok", q["tok15"][:POLE_EX]),
                         ("a_bank", want), ("g", r["g"]), ("u", r["u"]), ("a", a)):
                keep[k].append(v)
        np.savez(self.out / "g_pole.npz", poles=np.array([r[0] for r in rows]), **{k: np.array(v) for k, v in keep.items()})
        self.peak("G_POLE")
        self.gate("G_POLE", side(worst <= POLE_TOL, worst_fail <= POLE_TOL, value=worst, tol=POLE_TOL, n_poles=len(rows),
                                 n_exemplars=POLE_EX, failing_input_min=worst_fail, per_pole=rows))

    def colnorms(self):
        return self.torch.stack([l.mlp.down_proj.weight.detach().float().norm(dim=0) for l in self.M.layers]).cpu().numpy()

    def plan_edits(self, norms, pc: dict | None):
        """The edits of this stage with delta, c and kappa; pc maps layer -> (neuron, sign) (None in L1 and in the bf16 phase)."""
        rows = n6.plan(self.P, self.L, self.a.half)
        if self.a.stage == "L1":
            raw = set(n6.raw_example_pairs(self.P))
            rows = [r for r in rows if (r["role"] == "pair" and r["pair"] in raw and r["kind"] == "col") or
                    r["role"] in ("probe_pair", "zero")]
        if pc is None:
            rows = [r for r in rows if not r["role"].startswith("probe_pc") and r["role"] != "pc"]
        by_pair = {p["pair"]: p for p in self.P["pairs"]}
        out = []
        for r in rows:
            r = dict(r)
            if r["role"] in ("pc", "probe_pc"):
                n, s = pc[r["layer"]]
                r["pole"] = n6.pole_id(r["layer"], n, s)
                nw = float(norms[r["layer"], n])
                c = 0.5 * nw if r["kind"] == "col_half" else nw
            else:
                pr = by_pair[r["pair"]]
                _, n, s = n6.parse_pole(r["pole"])
                ne, nl = (float(norms[pr["layer"], n6.parse_pole(pr[m])[1]]) for m in ("early", "late"))
                nw = float(norms[r["layer"], n])
                c = 0.0 if r["kind"] == "zero" else min(ne, nl)
            w = self.M.layers[r["layer"]].mlp.down_proj.weight[:, n].detach().float().cpu()
            if r["kind"] in ("col", "col_half"):
                delta = -(c / nw) * w
            elif r["kind"] in ("rand", "probe"):
                key = r["pole"] if r["role"] != "pair" else f"{r['pair']}_{r['member']}"
                delta = c * rand_unit(self.d, f"n6 rand {key}")
            else:
                delta = self.torch.zeros(self.d)
            dn = float(delta.double().norm())
            assert abs(dn - c) <= NORM_TOL * max(c, 1e-30), f"{r['edit']}: ||delta|| {dn} is not c {c}"
            r.update(n=n, sign=s, c=c, norm_w=nw, kappa=c / nw, delta=delta, delta_norm=dn)
            out.append(r)
        return out

    def bf16_probes(self, edits):
        """G-EPS's failing input: the same probes in the bf16 model, ordinary full forwards over the eval windows."""
        base = self.full_pass(None)
        out = {}
        for e in [x for x in edits if x["kind"] == "probe"]:
            ed = Edit(self.M.layers[e["layer"]].mlp.down_proj, e["n"], e["sign"], e["delta"], "probe")
            with ed:
                lo = self.full_pass(None)
            out[e["edit"]] = float((lo.double() - base.double()).mean())
        return out

    # -------------------------------------------------------------------------------------- 2. fp32 base passes
    def full_pass(self, track, win=None):
        """Per-token losses [W, 2048] from ordinary model forwards in batches of B over the eval windows (or `win`). track: None, or
        a dict that a pre-hook on every down_proj fills (pass 1: E+ and E- of every pole, the top-50 eval positions of every pole)."""
        torch = self.torch
        win = self.win if win is None else win
        losses = []
        hs = []
        if track is not None:
            for li, lay in enumerate(self.M.layers):
                hs.append(lay.mlp.down_proj.register_forward_pre_hook(self._tracker(li, track)))
        with torch.inference_mode():
            for b0 in range(0, len(win), self.B):
                x = windows(self.tokens, win[b0:b0 + self.B], self.bos, self.device)
                track_b0 = b0
                if track is not None:
                    track["b0"] = track_b0
                lg = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=0).logits
                for i in range(x.shape[0]):
                    losses.append(token_loss(lg[i], x[i]))
                del lg
        for h in hs:
            h.remove()
        return torch.stack(losses).cpu()

    def _tracker(self, li, track):
        torch = self.torch

        def pre(m, args):
            a = args[0].float()                                   # [B, S, W]
            B, S, W = a.shape
            body = a[:, 1:]                                       # positions 1 .. 2048: where an edit applies
            track["E"][li, :, 0] += (torch.relu(body) ** 2).sum((0, 1)).double().cpu()
            track["E"][li, :, 1] += (torch.relu(-body) ** 2).sum((0, 1)).double().cpu()
            cand = a[:, 1:S - 1].reshape(-1, W)                   # exemplar candidates: positions 1 .. 2047 (a target exists)
            pos = (torch.arange(B, device=a.device)[:, None] + track["b0"]) * S + torch.arange(1, S - 1, device=a.device)[None]
            pos = pos.reshape(-1)
            for s, si in ((1, 0), (-1, 1)):
                v = torch.cat([track["topv"][li][:, si].T.to(a.device), s * cand])
                p = torch.cat([track["topp"][li][:, si].T.to(a.device), pos[:, None].expand(-1, W)])
                tv, ti = v.topk(TOPK, dim=0)
                track["topv"][li][:, si] = tv.T.cpu()
                track["topp"][li][:, si] = torch.gather(p, 0, ti).T.cpu()
        return pre

    def pass1(self, win=None):
        torch = self.torch
        tr = {"E": torch.zeros(self.L, self.W, 2, dtype=torch.float64),
              "topv": [torch.full((self.W, 2, TOPK), -math.inf) for _ in range(self.L)],
              "topp": [torch.full((self.W, 2, TOPK), -1, dtype=torch.long) for _ in range(self.L)], "b0": 0}
        lo = self.full_pass(tr, win)
        return lo, tr

    def exemplar_logp(self, poles: dict):
        """Base log-probabilities [50, V] at each edited pole's 50 eval exemplar positions (pass 2's logits)."""
        torch = self.torch
        S = n6.HW + 1
        need = {}
        for p, pos in poles.items():
            for k, g in enumerate(pos):
                need.setdefault(int(g) // S, []).append((p, k, int(g) % S))
        out = {p: torch.zeros(len(pos), self.model.config.vocab_size) for p, pos in poles.items()}
        losses = []
        with torch.inference_mode():
            for b0 in range(0, len(self.win), self.B):
                x = windows(self.tokens, self.win[b0:b0 + self.B], self.bos, self.device)
                lg = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=0).logits
                for i in range(x.shape[0]):
                    losses.append(token_loss(lg[i], x[i]))
                    for p, k, j in need.get(b0 + i, []):
                        out[p][k] = torch.log_softmax(lg[i, j].float(), -1).cpu()
                del lg
        return torch.stack(losses).cpu(), out

    # -------------------------------------------------------------------------------------- 3. the progression
    def edit_acc(self, e, ex_pos) -> dict:
        """The running sums of one edit over the chunks of its layer (amendment A4)."""
        S = n6.HW + 1
        pos_by_w = {}
        for k, g in enumerate(ex_pos):
            pos_by_w.setdefault(int(g) // S, []).append((k, int(g) % S))
        ed = Edit(self.M.layers[e["layer"]].mlp.down_proj, e["n"], e["sign"], e["delta"], "probe" if e["kind"] == "probe" else "add")
        return {"ed": ed, "win_sum": self.torch.zeros(len(self.win), dtype=self.torch.float64), "R2": 0.0, "top": [], "kl": [],
                "dl_ex": [], "pos_by_w": pos_by_w, "n_tok": 0, "equal": True, "sec": 0.0}

    def run_edit(self, e, acc, hc, hn, c0, base, ex_logp):
        """One edit on one chunk: hc and hn are the chunk's h_l and h_(l+1) on the GPU, c0 the index of its first window."""
        torch = self.torch
        t = time.time()
        with torch.inference_mode(), acc["ed"]:
            for b0 in range(0, hc.shape[0], self.B):
                hb = hc[b0:b0 + self.B]
                pr = self.M.prep(hb)
                y = self.M.run(hb, e["layer"], e["layer"] + 1, pr)
                acc["R2"] += float(((y.double() - hn[b0:b0 + self.B].double()) ** 2).sum())
                y = self.M.run(y, e["layer"] + 1, self.L, pr)
                x = windows(self.tokens, self.win[c0 + b0:c0 + b0 + self.B], self.bos, self.device)
                for i in range(y.shape[0]):
                    w = c0 + b0 + i
                    lg = self.M.window_logits(y[i])
                    lo = token_loss(lg, x[i]).cpu()
                    acc["n_tok"] += lo.numel()
                    acc["equal"] = acc["equal"] and bool(torch.equal(lo, base[w]))
                    dlt = (lo.double() - base[w].double())
                    acc["win_sum"][w] = dlt.sum()
                    v, j = dlt.abs().topk(5)
                    acc["top"] += [(float(v[q]), w, int(j[q]) + 1, float(dlt[j[q]]), float(base[w][j[q]])) for q in range(5)]
                    for k, jj in acc["pos_by_w"].get(w, []):
                        lp = torch.log_softmax(lg[jj].float(), -1).cpu()
                        b = ex_logp[k]
                        acc["kl"].append(float((b.exp() * (b - lp)).sum()))
                        acc["dl_ex"].append(float(dlt[jj]))
                    del lg
        acc["sec"] += time.time() - t

    def edit_record(self, e, acc, ex_pos) -> dict:
        ed, kl = acc["ed"], acc["kl"]
        top = sorted(acc["top"], reverse=True)[:5]
        rec = {k: e[k] for k in ("edit", "layer", "role", "kind", "pole", "n", "sign", "c", "norm_w", "kappa", "delta_norm")}
        rec.update({k: e[k] for k in ("pair", "member", "half") if k in e})
        rec.update(dL=float(acc["win_sum"].sum() / acc["n_tok"]), dL_win=(acc["win_sum"] / n6.HW).tolist(), R=math.sqrt(acc["R2"]),
                   E_plus=ed.E, realized_sq=ed.sq, realized_rel_err=abs(ed.sq - e["c"] ** 2 * ed.E) / max(e["c"] ** 2 * ed.E, 1e-300),
                   kl_eff=float(np.mean(kl)), dl_eff=float(np.mean(acc["dl_ex"])), n_ex=len(kl), hook_calls=ed.calls,
                   bitwise_equal_base=acc["equal"],
                   top5=[{"abs": v, "window": w, "target_pos": j, "dl": d, "base_loss": bl} for v, w, j, d, bl in top],
                   seconds=round(acc["sec"], 2))
        assert acc["n_tok"] == len(self.win) * n6.HW, f"{e['edit']}: {acc['n_tok']} targets read"
        assert ed.calls == math.ceil(len(self.win) / self.B), f"{e['edit']}: the hook ran {ed.calls} times"
        assert len(kl) == len(ex_pos), f"{e['edit']}: {len(kl)} of {len(ex_pos)} exemplars read"
        return rec

    def progression(self, edits, base, ex, E):
        """Layer by layer from the embeddings; returns the records and the base losses recomputed from h_L (G-CACHE). Amendment A4:
        h_l of every window is kept in host memory (fp32); at each layer the windows go to the GPU in chunks of C, and each chunk's
        h_(l+1) is computed once, read by every edit of the layer, and written back over h_l. Every window is computed in the same
        batches of B as without chunks, so the records do not depend on C."""
        torch = self.torch
        W, B, C = len(self.win), self.B, self.C
        with torch.inference_mode():
            h = torch.empty((W, n6.HW + 1, self.d), dtype=torch.float32)
            for b0 in range(0, W, B):
                h[b0:b0 + B].copy_(self.M.embed(windows(self.tokens, self.win[b0:b0 + B], self.bos, self.device)))
        recs, sec_per_unit = [], None
        for lyr in range(self.L):
            todo = [e for e in edits if e["layer"] == lyr]
            remaining = [e for e in edits if e["layer"] >= lyr]
            if sec_per_unit is not None:
                proj = sec_per_unit * sum(self.L - e["layer"] + 2 for e in remaining)
                self.m["measured"]["projection_remaining_s"] = round(proj, 1)
                if time.time() - self.t0 + proj > 0.9 * self.a.budget_min * 60:
                    self.m["parked"] = [e["edit"] for e in remaining]
                    self.write()
                    return recs, None
            acc = [self.edit_acc(e, ex[e["pole"]]["pos"]) for e in todo]
            with torch.inference_mode():
                for c0 in range(0, W, C):
                    hc = h[c0:c0 + C].to(self.device)
                    hn = torch.empty_like(hc)
                    for b0 in range(0, hc.shape[0], B):
                        hn[b0:b0 + B] = self.M.run(hc[b0:b0 + B], lyr, lyr + 1, self.M.prep(hc[b0:b0 + B]))
                    for e, a in zip(todo, acc):
                        self.run_edit(e, a, hc, hn, c0, base, ex[e["pole"]]["logp"])
                    h[c0:c0 + C].copy_(hn)
                    del hc, hn
            for e, a in zip(todo, acc):
                r = self.edit_record(e, a, ex[e["pole"]]["pos"])
                expect = float(E[e["pole"]])
                r["E_plus_pass1"] = expect
                r["E_rel_err"] = abs(r["E_plus"] - expect) / max(expect, 1e-300)
                recs.append(r)
                with open(self.rec_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if todo:
                done = {y["edit"] for y in recs}
                units = sum(self.L - x["layer"] + 2 for x in edits if x["edit"] in done)
                sec_per_unit = (time.time() - self.t_prog) / max(units, 1)
                self.m["n_done"] = len(recs)
                self.m["measured"]["sec_per_edit_layer_unit"] = sec_per_unit
                self.write()
        with torch.inference_mode():
            lo, other = [], []                              # other: G-CACHE's failing input, another window's final residual
            for b0 in range(0, W, B):
                x = windows(self.tokens, self.win[b0:b0 + B], self.bos, self.device)
                for i in range(x.shape[0]):
                    lo.append(token_loss(self.M.window_logits(h[b0 + i].to(self.device)), x[i]).cpu())
                    other.append(token_loss(self.M.window_logits(h[(b0 + i + 1) % W].to(self.device)), x[i]).cpu())
        return recs, (torch.stack(lo), torch.stack(other))

    # -------------------------------------------------------------------------------------- the run
    def same_comparison(self):
        """Amendment A5, L2b only, before the first gate: L2a's metrics.json (--against) must be a complete freeze-half launch of
        stage L2 with nothing parked, reading the same eval windows, eval tokens and pairs content (sha256) as this launch."""
        prior = json.loads(pathlib.Path(self.a.against).read_text())
        for k in n6.SAME_KEYS:
            assert prior[k] == self.m[k], f"A5: L2b's {k} {self.m[k]} is not L2a's {prior[k]}: the two launches are not one comparison"
        assert prior["stage"] == "L2" and prior["half"] == "freeze", f"A5: --against is stage {prior['stage']} half {prior['half']}"
        assert prior["complete"] and not prior["parked"], f"A5: L2a is not complete ({len(prior['parked'])} edits parked): L2b does not run"
        self.m["against"] = {"path": str(self.a.against), "pg_git_sha": prior["pg_git_sha"], "finished_utc": prior["finished_utc"],
                             **{k: prior[k] for k in n6.SAME_KEYS}}
        self.write()

    def run(self, versions: dict, snapshot: str):
        torch = self.torch
        assert self.P.get("stage") == self.a.stage, f"pairs file of stage {self.P.get('stage')} given to stage {self.a.stage}"
        if self.a.half == "test":
            self.same_comparison()
        want = self.reg["versions"]
        g = side(all(versions[k] == want[k] for k in want), all(versions[k] == "x" + want[k] for k in want), got=versions, want=want)
        self.gate("G_SAME", g)
        sn = self.reg["snapshot"]
        self.gate("G_REV", side(snapshot == sn, "0" * 40 == sn, got=snapshot, want=sn))
        assert next(self.model.parameters()).dtype == torch.bfloat16, "the bf16 phase needs the model as loaded"
        self.k1()
        k2 = self.corpus_nll()
        self.peak("K2")
        kt, kf = self.reg["k2_target"], self.reg["k2_fail"]
        self.gate("K2", side(abs(k2 - kt) <= K2_TOL, abs(kf - kt) <= K2_TOL, value=k2, target=kt, tol=K2_TOL, failing_value_T14=kf))
        self.pole_check()
        norms = self.colnorms()
        np.save(self.out / "colnorms.npy", norms)
        bf_edits = self.plan_edits(norms, None)
        F_bf16 = self.bf16_probes(bf_edits)
        self.m["measured"]["bf16_probe_dL"] = F_bf16
        self.peak("bf16_probes")
        # ---- fp32
        self.model.float()
        torch.set_float32_matmul_precision("highest")
        self.peak("cast")
        dts = {str(p.dtype) for p in self.model.parameters()}
        self.gate("G_FP32", side(dts == {"torch.float32"} and torch.get_float32_matmul_precision() == "highest"
                                 and not torch.backends.cuda.matmul.allow_tf32, False, dtypes=sorted(dts)))
        self.phase = "K3"
        k3 = self.corpus_nll(batch=self.B)
        k3f = self.corpus_nll(shift=2, batch=self.B)
        self.peak("K3")
        self.gate("K3", side(abs(k3 - k2) <= K3_TOL, abs(k3f - k2) <= K3_TOL, value=k3, bf16=k2, tol=K3_TOL,
                             failing_value_off_by_one=k3f))
        t = time.time()
        self.phase = "pass1"
        base1, tr = self.pass1()
        self.m["measured"]["pass_s"] = round(time.time() - t, 1)
        self.peak("pass1")
        E = {}
        Eall = tr["E"].numpy()
        np.save(self.out / "energies.npy", Eall)
        pc = {}
        for lyr in range(self.L):
            i, si = np.unravel_index(np.argmax(Eall[lyr]), Eall[lyr].shape)
            pc[lyr] = (int(i), 1 if si == 0 else -1)
        self.m["pc"] = {lyr: n6.pole_id(lyr, *pc[lyr]) for lyr in pc}
        if self.a.stage == "L2":
            self.phase = "inst"
            _, tri = self.pass1(self.P["inst_windows"])
            self.peak("inst")
            self.instance_check(norms, tri["E"].numpy())
        edits = self.plan_edits(norms, pc if self.a.stage == "L2" else None)
        ex_pos = {}
        for e in edits:
            lyr, n, s = n6.parse_pole(e["pole"])
            si = 0 if s > 0 else 1
            E[e["pole"]] = float(Eall[lyr, n, si])
            ex_pos[e["pole"]] = tr["topp"][lyr][n, si].numpy()
            assert (ex_pos[e["pole"]] >= 0).all() and (tr["topv"][lyr][n, si] > 0).all(), f"{e['pole']}: fewer than 50 eval tokens"
        self.phase = "pass2"
        base2, logp = self.exemplar_logp(ex_pos)
        self.peak("pass2")
        self.gate("G_ID", side(torch.equal(base1, base2), False, value_max_abs=float((base1 - base2).abs().max())), stop=False)
        ex = {p: {"pos": ex_pos[p], "logp": logp[p]} for p in ex_pos}
        self.m["measured"]["base_mean_loss"] = float(base1.double().mean())
        self.write()
        self.t_prog = time.time()
        self.phase = "progression"
        recs, final = self.progression(edits, base1, ex, E)
        self.peak("progression")
        self.finish(recs, final, base1, edits, F_bf16)

    def instance_check(self, norms, Eall):
        """G-INST (L2): the final pairs were matched offline on L1's column norms and E+ (n6.final_pairs); this instance must see the
        same values (rel 1e-5). Eall is E+ over L1's 96 windows (the pairs file's inst_windows, amendment A4), not over L2's eval set.
        Failing input: each pole compared with its partner's values."""
        worst, worst_fail = 0.0, math.inf
        for p in self.P["pairs"]:
            for m, o in (("early", "late"), ("late", "early")):
                lyr, n, s = n6.parse_pole(p[m])
                got = (float(norms[lyr, n]), float(Eall[lyr, n, 0 if s > 0 else 1]))
                want, other = self.P["instance"][p[m]], self.P["instance"][p[o]]
                worst = max(worst, abs(got[0] / want["colnorm"] - 1), abs(got[1] / want["E_plus"] - 1))
                worst_fail = min(worst_fail, max(abs(got[0] / other["colnorm"] - 1), abs(got[1] / other["E_plus"] - 1)))
        self.gate("G_INST", side(worst <= INST_TOL, worst_fail <= INST_TOL, value=worst, tol=INST_TOL, failing_input_min=worst_fail))

    def finish(self, recs, final, base, edits, F_bf16):
        torch = self.torch
        by = {r["edit"]: r for r in recs}
        shapes = {k: sorted(v) for k, v in self.head_shapes.items()}
        self.m["measured"]["head_input_shapes"] = shapes
        paths = {"base_pass1 (model forward)": shapes.get("pass1"), "progression (head per window)": shapes.get("progression")}
        if final is not None:
            self.gate("G_CACHE", {**cache_gate(final[0], base, final[1]), "bitwise_equal": bool(torch.equal(final[0], base)),
                                  "head_input_shapes": paths}, stop=False)
        probes = {k: v["dL"] for k, v in by.items() if v["kind"] == "probe"}
        F = max(abs(v) for v in probes.values())
        Fb = max(abs(v) for v in F_bf16.values())
        self.m["gates"]["G_EPS"] = {**side(F <= n6.EPS_ABS, Fb <= n6.EPS_ABS, value=F, bound=n6.EPS_ABS), "fp32": probes,
                                    "bf16": F_bf16}
        zero = [v for v in by.values() if v["kind"] == "zero"]
        first_col = next(v for v in recs if v["kind"] == "col")
        self.m["gates"]["G_ZN"] = {**zn_gate([v["dL"] for v in zero], first_col["dL"]), "failing_input_edit": first_col["edit"],
                                   "n_bitwise_equal_base": sum(v["bitwise_equal_base"] for v in zero), "head_input_shapes": paths}
        self.m["gates"]["G_ID"]["failing_input"]["pass"] = bool(first_col["bitwise_equal_base"])
        bad = [v["edit"] for v in recs if v["kind"] != "zero" and (v["realized_rel_err"] > REAL_TOL or v["E_rel_err"] > REAL_TOL)]
        self.m["gates"]["G_NORM"] = side(not bad, False, n_edits=len(recs), bad=bad,
                                         max_E_rel_err=max(v["E_rel_err"] for v in recs if v["kind"] != "zero"))
        if self.a.stage == "L2" and self.a.half != "test" and not self.m["parked"]:      # A5: the positive control is L2a's
            self.m["gates"]["G_PC"], self.m["gates"]["G_DOSE"] = pc_gates(by, self.L, n6.TIE_MULT * F)
        if self.a.half != "test":                                                          # A5: the raw examples are freeze pairs
            self.raw_examples(by)
        self.m["complete"] = not self.m["parked"]
        self.m["finished_utc"] = utc()
        self.write()

    def raw_examples(self, by):
        out = []
        pairs = {p["pair"]: p for p in self.P["pairs"]}
        for pid in n6.raw_example_pairs(self.P):
            p = pairs[pid]
            for m, o in (("early", "late"), ("late", "early")):
                r = by.get(f"{pid}_{m}_col")
                if r is None:
                    continue
                toks = []
                for t in r["top5"]:
                    w, j = t["window"], t["target_pos"]
                    g = self.win[w] * n6.HW
                    seq = [self.bos] + self.tokens[g:g + n6.HW].astype(np.int64).tolist()
                    toks.append({**t, "context": self.decode(seq[max(0, j - 32):j]), "target": self.decode([seq[j]])})
                rr = by.get(f"{pid}_{m}_rand")
                out.append({"pair": pid, "member": m, "pole": p[m], "partner": p[o], "age": p[f"age_{m}"], "partner_age": p[f"age_{o}"],
                            "frobenius_norm_c": r["c"], "kappa": r["kappa"], "norm_w": r["norm_w"], "dL_nats_per_token": r["dL"],
                            "dL_rand_same_norm": None if rr is None else rr["dL"], "kl_eff": r["kl_eff"], "E_plus": r["E_plus"],
                            "top5_tokens": toks})
        with open(self.out / "samples" / "raw_examples.jsonl", "w") as f:
            for x in out:
                f.write(json.dumps(x) + "\n")
        self.m["gates"]["G_SAMPLES"] = side(len(out) == 10, False, n=len(out))


def pc_gates(by: dict, L: int, tau: float) -> tuple[dict, dict]:
    """G-PC (amendment A4): in the positive-control layers (n6.pc_layers: 0, 4, ..., 28) the layer's highest-energy pole's column
    edit costs more than its random-direction edit of the same norm (by more than tau) in at least PC_MIN of 8; G-DOSE: the full
    column edit more than the half one, at the same count. Failing input: the labels swapped."""
    layers = n6.pc_layers(L)
    need = n6.PC_MIN * len(layers) // 8
    d = {k: [by[f"PC{l:02d}_{k}"]["dL"] for l in layers] for k in ("col", "rand", "col_half")}
    n_col = sum(c > r + tau for c, r in zip(d["col"], d["rand"]))
    n_rev = sum(r > c + tau for c, r in zip(d["col"], d["rand"]))
    n_dose = sum(c > h + tau for c, h in zip(d["col"], d["col_half"]))
    n_dose_rev = sum(h > c + tau for c, h in zip(d["col"], d["col_half"]))
    return ({**side(n_col >= need, n_rev >= need, n_col_gt_rand=int(n_col), n_layers=len(layers), layers=layers, need=need, tau=tau),
             "failing_input_swapped": int(n_rev)},
            {**side(n_dose >= need, n_dose_rev >= need, n_full_gt_half=int(n_dose), n_layers=len(layers), layers=layers, need=need,
                    tau=tau), "failing_input_swapped": int(n_dose_rev)})


def host_mem_available() -> int:
    """MemAvailable of /proc/meminfo, bytes (the instance's host memory a new allocation can take)."""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("no MemAvailable line in /proc/meminfo")


def fetch_inputs(src: str, reg: dict) -> tuple[dict, dict]:
    from infra.aws_pg.jobs.neurons_harvest import fetch
    out = {}
    names = sorted(reg)
    for name in names:
        p = fetch(src.rstrip("/") + "/" + name, f"n6_inputs/{name}")
        got = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
        assert reg[name] is not None, f"{name} is not registered (freeze 1 sets INPUT_SHA)"
        assert got == reg[name], f"{name}: sha256 {got} is not the registered {reg[name]}"
        out[name] = json.loads(pathlib.Path(p).read_text())
    return out[[n for n in names if n.startswith("pairs")][0]], out["e1_known.json"]


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["L1", "L2"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--half", choices=["freeze", "test"], help="amendment A5, stage L2: freeze (L2a) or test (L2b)")
    ap.add_argument("--against", help="A5, L2b only: L2a's metrics.json (s3 or local)")
    return ap


def check_args(a):
    """Amendment A5: stage L2 runs as L2a (--half freeze) and L2b (--half test --against <L2a's metrics.json>); L1 has no half."""
    assert (a.stage == "L2") == (a.half is not None), f"stage {a.stage}, half {a.half}: L2 runs as L2a (freeze) and L2b (test), A5"
    assert (a.half == "test") == (a.against is not None), "A5: --against (L2a's metrics.json) is given to L2b and only to L2b"


def main(argv=None):
    a = parser().parse_args(argv)
    check_args(a)
    out = pathlib.Path(a.out) / "n6"
    out.mkdir(parents=True, exist_ok=True)
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    assert torch.cuda.is_available(), "no CUDA on this instance"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    reg = registry()
    pairs, e1 = fetch_inputs(a.inputs, reg["input_sha"][a.stage])
    if a.against is not None:
        a.against = str(fetch(a.against, "n6_inputs/L2a_metrics.json"))
    need = len(pairs["eval_windows"]) * (n6.HW + 1) * n6.HIDDEN * 4 + HOST_MARGIN     # amendment A4: the kept residual, fp32
    avail = host_mem_available()
    assert avail >= need, f"host memory: {avail / 2 ** 30:.1f} GiB available, the kept residual needs {need / 2 ** 30:.1f} GiB"
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    assert corpus["band_end"][3] == n6.BAND3_TOK[1], corpus["band_end"]
    t0 = time.time()
    local, sha, rec = download(n6.REPO, n6.REVISION)
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    assert tok.bos_token_id == reg["bos"], tok.bos_token_id
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "device": torch.cuda.get_device_name(0)}
    job = Job(a, model, tok, corpus["tokens"], pairs, e1, out, "cuda", NH.decoder(tok), reg)
    job.m.update(download=rec, load_s=round(time.time() - t0, 1), versions=versions,
                 host_mem={"available_gib_at_start": round(avail / 2 ** 30, 2), "needed_gib": round(need / 2 ** 30, 2)})
    job.run(versions, sha)
    print("N6_JOB_OK", json.dumps({"stage": a.stage, "half": a.half, "n_done": job.m["n_done"], "parked": len(job.m["parked"])}),
          flush=True)


if __name__ == "__main__":
    main()
