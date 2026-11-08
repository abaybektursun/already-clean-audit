"""MENDELEEV N9 — the graft job: G (R_G, D_G, cuts) + W (weight transplant) for ONE row.
Contract: infra/pg/MATRIX_CONTRACT.md §6; spec §1.2 (Graft, Transplant), §6.4, §6.13, §11.6, §12.3.

  python -m infra.aws_pg.jobs.matrix_graft --row 2 --half discovery --pass 1 --stride 2 --out DIR
  python -m infra.aws_pg.jobs.matrix_graft --row 2 --half confirmation --pass 2 --layers 14,15,16 --out DIR
      pass 2 scores the stride grid ∪ the bracketed layers, all on the confirmation half (§6.13): the
      layer CHOICE comes from the discovery band, every number in the curve from confirmation.
  ... --then-pass2                          pass 1 on discovery, then pass 2 on confirmation on the same
                                            bank: bracket = the discovery band's interior, cut = the best
                                            discovery cut at its layer, transplant band = the discovery band
  ... --transplant [--L-lo a --L-hi b]      W configs (co-resident pair only)
  ... --cached-donor [--cache-dir D]        one model resident at a time (spec §11.6)
  ... --equivalence                         §11.6 gate on a co-resident row (cached == co-resident)
  ... --smoke                               1 item/subject, 4 sweep points, k=8 only
  python -m infra.aws_pg.jobs.matrix_graft --dryrun     weight-free on a 2-layer stub pair

Stages, each appended to out/metrics.json on completion (`kind: "graft"`):
  manifest → identity (self-graft at 3 layers, both models) → wiring (donor last-layer residual
  through the DONOR's norm + head reproduces the donor's logits; D2) → hooks (resolver names ==
  named_modules) → baselines (S, both models, both halves) → R_G_last (last-layer prefix graft read
  through the RECIPIENT head — a RESULT, banked as `R_G_last_recipient_head`) → nulls (≥ 5
  mismatched-item cuts → `null_Rs`) → sweep (R_G per layer, then D_G converse) → cuts (final-position
  only; top-k directions of the healthy − broken residual difference, k ∈ {8, 64, 512}) → transplant
  (--transplant; carrel_core's runbook generalised by the family resolver: none/all sanity, prefix
  0..L_lo, two outside-band controls, band, band+tail, singles, attn-only, MLP-only, final norm;
  restore-after-each exact).
Models come from the contract loader `matrix_models.load_pair(row, device, which=("before","after"))`
→ {"before": member, "after": member}: one MEMBER DICT per side, {"fwd": Forward, "model", "tok",
"proc", "repo", "which", "shell", "info", "local"} (`matrix_models.assert_member`), and NO top-level
manifest — provenance lives on each member and is banked here by `member_manifest`. Healthy = before,
broken = after (spec §1.2 roles); cached mode loads `which=("before",)` / `which=("after",)` one at a
time and gets `{"before": member}` / `{"after": member}` back. `controls.
prefix_control_applicable` (embed + head identical across the pair, fingerprints compared through
`key_remap`) tells the verdict script when the D17 prefix control is 'not applicable' (rows 1b, 3b,
3c-Act, 4). The job BANKS inputs only. Every verdict (identity gate, T, band, reader_verdict, wiring gate,
weight_band_verdict) is computed offline by scripts/matrix_verdicts.py from `controls`, `curves`,
`transplant`; the job-side T / band written here are labelled `job_side` and used only to pick the
cut layer, the suggested pass-2 bracket, and the transplant band when --L-lo/--L-hi are not given.

Order notes (both benign, stated): baselines run before R_G_last because R needs them; the
confirmation half re-runs the gates and nulls on its own items so each metrics.json is complete.
R_G(L) = (acc_graft − acc_broken)/(acc_healthy − acc_broken), healthy donor → broken recipient.
D_G(L) = (acc_healthy − acc_graft)/(acc_healthy − acc_broken), broken donor → healthy recipient.
All graft evaluations run at the fixed batch 4 of the identity gate (D22). Fail loud; no try/except.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import time

BATCH = 4                      # D22: the identity gate's batch; every graft eval uses it
SHOTS = 5
N_SELF = 3
N_NULL = 5
CUT_KS = (8, 64, 512)
HALVES = ("discovery", "confirmation")
CUT_NAMES = ("final_position",) + tuple(f"top{k}" for k in CUT_KS)


# ---------------------------------------------------------------------------------------- banking
class Bank:
    """metrics.json (atomic, after every stage) + controls.json + per_item/*.json + events.jsonl."""

    def __init__(self, out, meta: dict):
        from infra.pg.matrix_recovery_core import atomic_write_json
        self._w = atomic_write_json
        self.out = pathlib.Path(out)
        (self.out / "per_item").mkdir(parents=True, exist_ok=True)
        self.m = {"kind": "graft", **meta, "stages": [], "curves": {}, "cuts": {}, "baselines": {}}
        self.t0 = time.time()
        self.write()

    def write(self):
        self._w(self.out / "metrics.json", self.m)
        _write_samples(str(self.out), self.m, "graft")      # spec §12.11: every eval carries real data
        if "controls" in self.m:
            self._w(self.out / "controls.json", self.m["controls"])

    def stage(self, name: str, **fields):
        from infra.pg.util_probe import log_event
        self.m["stages"].append(name)
        self.m.setdefault("stage_s", {})[name] = round(time.time() - self.t0, 1)
        log_event(self.out / "events.jsonl", stage=name, elapsed_s=round(time.time() - self.t0, 1), **fields)
        self.write()
        print(f"[GRAFT][{name}] {json.dumps(fields, default=str)[:300]}", flush=True)

    def per_item(self, name: str, rec: dict) -> str:
        self._w(self.out / "per_item" / f"{name}.json", rec)
        return f"per_item/{name}.json"


# ---------------------------------------------------------------------------------------- pure helpers
def sweep_layers(n_layers: int, stride: int) -> list:
    from infra.pg.matrix_recovery_core import sweep_layers as sl
    return sl(n_layers, stride)


def null_cuts(n_layers: int, k: int = N_NULL, batch: int = BATCH) -> list:
    """k (layer, shift) mismatched-item cuts: k spread layers at shift 1 (BROCA convention); when the
    model has fewer layers than cuts (the stub), layers cycle and the roll shift grows (< batch)."""
    from infra.pg.matrix_recovery_core import null_layers
    base = null_layers(n_layers, k) if n_layers - 1 >= k else list(range(n_layers))
    cuts, shift = [], 1
    while len(cuts) < k:
        for L in base:
            if len(cuts) < k:
                cuts.append((int(L), shift))
        shift += 1
        assert shift < batch or len(cuts) >= k, f"cannot place {k} null cuts with {n_layers} layers at batch {batch}"
    return cuts


def self_layers(n_layers: int, k: int = N_SELF) -> list:
    """k interior layers spread over depth for the self-graft identity test (all layers on a stub)."""
    import numpy as np
    if n_layers <= k:
        return list(range(n_layers))
    ls = sorted({int(round(x)) for x in np.linspace(0, n_layers - 1, k + 2)[1:-1]})
    assert len(ls) == k, (n_layers, ls)
    return ls


def restoration(acc_graft: float, acc_broken: float, acc_healthy: float) -> float:
    from infra.pg.broca_core import restoration_fraction
    return restoration_fraction(acc_graft, acc_broken, acc_healthy)


def damage(acc_graft: float, acc_broken: float, acc_healthy: float) -> float:
    from infra.pg.noether_core import damage_fraction
    return damage_fraction(acc_graft, acc_broken, acc_healthy)


def best_layer(layers, R_G, T: float, n_layers: int) -> tuple:
    """Cut layer: L_hi of the job-side band when the sweep can define one (≥ 4 points), else the
    sweep's argmax R_G. Descriptive selection only (spec §6.4 cuts are descriptive)."""
    from infra.pg.matrix_core import graft_band
    if len(layers) >= 4:
        g = graft_band(layers, R_G, T)
        if g["L_hi"] is not None:
            return int(g["L_hi"]), "L_hi(job-side band)"
    i = max(range(len(layers)), key=lambda j: R_G[j])
    return int(layers[i]), "argmax R_G (no band)"


def transplant_configs(n_layers: int, L_lo: int, L_hi: int) -> list:
    """carrel_core.CONFIGS generalised to a band found by the graft: band = (L_lo, L_hi] (the layers
    whose WEIGHTS lie between the two prefix points), tail = (L_lo, n), prefix = [0, L_lo] (D17
    control), two outside-band windows of the band's width (before the band; the second moves after
    the band when the prefix is too short for two disjoint windows)."""
    assert 0 <= L_lo < L_hi < n_layers, (L_lo, L_hi, n_layers)
    band = list(range(L_lo + 1, L_hi + 1)); w = len(band)
    prefix = list(range(0, L_lo + 1)); tail = list(range(L_lo + 1, n_layers))
    ctrl_a = list(range(0, min(w, L_lo + 1)))
    ctrl_b = list(range(max(0, L_lo + 1 - w), L_lo + 1))
    if ctrl_b == ctrl_a:
        ctrl_b = list(range(L_hi + 1, min(n_layers, L_hi + 1 + w)))
    both = ("attn", "mlp")
    cfg = [("none", {"layers": [], "modules": both}),
           ("all", {"layers": list(range(n_layers)), "modules": both, "final_norm": True, "embed": True}),
           (f"prefix_0_{L_lo}", {"layers": prefix, "modules": both}),
           (f"ctrl_{ctrl_a[0]}_{ctrl_a[-1]}", {"layers": ctrl_a, "modules": both})]
    if ctrl_b:
        cfg.append((f"ctrl_{ctrl_b[0]}_{ctrl_b[-1]}", {"layers": ctrl_b, "modules": both}))
    cfg += [("band", {"layers": band, "modules": both}), ("band_tail", {"layers": tail, "modules": both})]
    cfg += [(f"solo_{l}", {"layers": [l], "modules": both}) for l in band]
    cfg += [("attn_band", {"layers": band, "modules": ("attn",)}), ("mlp_band", {"layers": band, "modules": ("mlp",)}),
            ("final_norm", {"layers": [], "modules": (), "final_norm": True})]
    names = [c[0] for c in cfg]
    assert len(names) == len(set(names)), names
    return cfg


# ---------------------------------------------------------------------------------------- the run
class Ctx:
    """One row's pair + items; every stage reads from here. `fwd_h` / `fwd_b` are Forward shells
    (contract §3) over the healthy and broken models; `res_*` their family resolvers."""

    def __init__(self, bank: Bank, items: dict, dev_shots, shots: int, cache_dir, cached: bool):
        self.bank, self.items, self.dev_shots, self.shots = bank, items, dev_shots, shots
        self.cache_dir, self.cached = pathlib.Path(cache_dir), cached
        self.fwd_h = self.fwd_b = self.res_h = self.res_b = None

    def attach(self, side: str, member: dict):
        """`member` is ONE `matrix_models.load_pair` member dict (contract §1), not a Forward: the
        loader returns {"fwd", "model", "tok", "proc", "repo", "which", "shell", "info", "local"} per
        side, on both the co-resident and the cached-donor path. Asserted, so a loader or a stub that
        drifts from that shape fails here with the shape in the message instead of an AttributeError
        three minutes into a GPU rental (2026-09-03). The member itself is NOT retained: cached mode
        frees a side by dropping `fwd_*`/`res_*`, and a reference held here would pin the weights."""
        from infra.pg.family_resolvers import resolve
        from infra.pg.matrix_models import assert_member
        fwd = assert_member(member)["fwd"]
        fwd.batch_size = BATCH
        res = resolve(fwd.model)
        setattr(self, "fwd_" + side[0], fwd)
        setattr(self, "res_" + side[0], res)
        return res

    def prompts(self, half: str) -> list:
        from infra.pg.format_readouts import s_prompt
        return [s_prompt(it, list(self.dev_shots(it["subject"], self.shots))) for it in self.items[half]]

    def S(self, fwd, model, half: str) -> dict:
        from infra.pg import format_readouts as FR
        from infra.pg.cached_donor import swap_model
        with swap_model(fwd, model):
            return FR.S(fwd, self.items[half], self.shots, self.dev_shots)

    def cache_path(self, side: str, L: int) -> pathlib.Path:
        return self.cache_dir / side / f"L{int(L)}"

    def graft(self, direction: str, L: int, mismatch: int = 0, mix=None):
        """'h2b': healthy donor → broken recipient (R_G); 'b2h': broken donor → healthy recipient (D_G)."""
        from infra.pg.cached_donor import CachedPatchedLM, PatchedLM
        assert direction in ("h2b", "b2h"), direction
        d, r = ("h", "b") if direction == "h2b" else ("b", "h")
        fwd_r, res_r = getattr(self, "fwd_" + r), getattr(self, "res_" + r)
        if self.cached:
            return CachedPatchedLM(fwd_r.model, res_r, L, self.cache_path("healthy" if d == "h" else "broken", L),
                                   mismatch=mismatch, mix=mix)
        fwd_d, res_d = getattr(self, "fwd_" + d), getattr(self, "res_" + d)
        return PatchedLM(fwd_d.model, fwd_r.model, res_d.layers, res_r.layers, L, mismatch=mismatch, mix=mix)


def assert_hook_names(model, res) -> list:
    """Per-layer hook-name assertion: the resolver's names must be in named_modules() and be the SAME
    module objects the grafts hook."""
    named = dict(model.named_modules())
    for L, (nm, ly) in enumerate(zip(res.layer_names, res.layers)):
        assert nm in named, f"layer {L}: hook name {nm!r} not in named_modules()"
        assert named[nm] is ly, f"layer {L}: {nm!r} in named_modules() is not the resolver's layer module"
    assert len(res.layers) == res.n_layers
    return list(res.layer_names)


def identity_stage(ctx: Ctx, side: str, half: str, layers) -> dict:
    """Self-graft (donor == recipient == this model) at each layer: 100 % item match, max|Δlogit| ≤ 1e-2."""
    from infra.pg.cached_donor import PatchedLM, logit_agreement
    fwd, res = getattr(ctx, "fwd_" + side[0]), getattr(ctx, "res_" + side[0])
    prompts = ctx.prompts(half)
    per = {}
    for L in layers:
        per[str(L)] = logit_agreement(fwd, fwd.model, PatchedLM(fwd.model, fwd.model, res.layers, res.layers, L), prompts)
    return {"layers": [int(L) for L in layers], "per_layer": per,
            "item_match_frac": min(v["item_match_frac"] for v in per.values()),
            "max_abs_dlogit": max(v["max_abs_dlogit"] for v in per.values())}


def wiring_stage(ctx: Ctx, half: str) -> dict:
    """D2: the DONOR's last-layer residual through the DONOR's own final norm + head vs the donor's
    logits from the shell — per item argmax match and max|Δlogit| (the positive hook test)."""
    import torch
    from infra.pg.cached_donor import _hidden
    fwd, res = ctx.fwd_h, ctx.res_h
    prompts = ctx.prompts(half)
    L = res.n_layers - 1
    box = []
    hd = res.layers[L].register_forward_hook(lambda m, a, o: box.append(res.readout(_hidden(o)).float().cpu()))
    try:
        ref = torch.cat([fwd.next_token_logits(prompts[s:s + BATCH]) for s in range(0, len(prompts), BATCH)], 0)
    finally:
        hd.remove()
    got = torch.cat(box, 0)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    match = (got.argmax(-1) == ref.argmax(-1)).float()
    return {"layer": L, "hook_name": res.layer_names[L], "item_match_frac": float(match.mean()),
            "max_abs_dlogit": float((got - ref).abs().max()), "n": int(ref.shape[0]),
            "softcap": res.softcap, "pre_head": type(res.pre_head).__name__ if res.pre_head is not None else None}


def _samples_only(rec: dict) -> dict:
    """The `extra.samples` block of a readout record, and nothing else (spec §12.11).

    Every graft readout already ran real inference and `format_readouts` already banked three real
    samples inside `rec["extra"]` — but `baseline_stage`, `graft_eval` and the transplant loop each kept
    only {acc, n, R, file} and dropped `extra`, so `_write_samples`' `collect_samples` walk found nothing
    and every graft cell's `samples.json` came out `{"cols": {}}`. Six such empty shells are banked on S3
    over data that was already paid for. Only `samples` is carried up: the full `extra` is 15 KB per
    readout (9.9 KB of it the `alt_surface` per-item score matrix) against 3 KB for the samples, it is
    already banked per readout under `per_item/`, and metrics.json is written after every stage."""
    sm = (rec.get("extra") or {}).get("samples")
    assert sm, f"readout {rec.get('col')} banked no extra.samples — §12.11 makes its number unquotable"
    return {"col": rec.get("col"), "acc": rec.get("acc"), "n": rec.get("n"), "floor": rec.get("floor"),
            "samples": sm}


def baseline_stage(ctx: Ctx, side: str) -> dict:
    fwd = getattr(ctx, "fwd_" + side[0])
    out = {}
    for half in HALVES:
        if half not in ctx.items:
            continue
        rec = ctx.S(fwd, fwd.model, half)
        out[half] = {"acc": rec["acc"], "n": rec["n"], "file": ctx.bank.per_item(f"S_{side}_{half}", rec),
                     "per_item": rec["per_item"], "per_item_hash": rec["per_item_hash"],
                     "extra": _samples_only(rec)}
    return out


def _base(ctx: Ctx, half: str):
    b = ctx.bank.m["baselines"]
    assert b["healthy"][half]["per_item_hash"] == b["broken"][half]["per_item_hash"], "item pairing broke"
    ah, ab = b["healthy"][half]["acc"], b["broken"][half]["acc"]
    assert ah - ab > 1e-6, f"no lost margin on {half}: healthy {ah} broken {ab} — the graft is undefined"
    return ah, ab


def graft_eval(ctx: Ctx, direction: str, L: int, half: str, tag: str, mismatch: int = 0, mix=None,
               items_subset=None) -> dict:
    """One graft readout. Returns {acc, R or D, n, file}; per-item vector banked under per_item/."""
    from infra.pg import format_readouts as FR
    from infra.pg.cached_donor import swap_model
    fwd_r = ctx.fwd_b if direction == "h2b" else ctx.fwd_h
    items = ctx.items[half] if items_subset is None else items_subset
    with swap_model(fwd_r, ctx.graft(direction, L, mismatch=mismatch, mix=mix)):
        rec = FR.S(fwd_r, items, ctx.shots, ctx.dev_shots)
    b = ctx.bank.m["baselines"]
    hashes = rec["per_item_hash"]
    base_hashes = b["healthy"][half]["per_item_hash"]
    pos = {h: i for i, h in enumerate(base_hashes)}
    assert len(pos) == len(base_hashes), "baseline item hashes are not unique — pairing would be ambiguous"
    idx = [pos[h] for h in hashes]
    ah = sum(b["healthy"][half]["per_item"][i] for i in idx) / len(idx)
    ab = sum(b["broken"][half]["per_item"][i] for i in idx) / len(idx)
    out = {"layer": int(L), "acc": rec["acc"], "n": rec["n"], "acc_healthy": ah, "acc_broken": ab,
           "file": ctx.bank.per_item(tag, rec), "extra": _samples_only(rec)}
    if direction == "h2b":
        out["R"] = restoration(rec["acc"], ab, ah)
    else:
        out["D"] = damage(rec["acc"], ab, ah)
    return out


def null_stage(ctx: Ctx, half: str, n_null: int) -> dict:
    """≥ 5 mismatched-item cuts (healthy donor rolled within the batch → broken recipient). A tail
    batch too small to roll is DROPPED from that cut (never a self-graft leak); R is computed on the
    items actually cut, against the same items' baselines."""
    items = ctx.items[half]
    cuts, out = null_cuts(ctx.res_b.n_layers, n_null), []
    for L, shift in cuts:
        n = len(items); r = n % BATCH
        keep = items[: n - r] if (r and r <= shift) else items
        g = graft_eval(ctx, "h2b", L, half, f"null_L{L}_s{shift}_{half}", mismatch=shift, items_subset=keep)
        out.append({**g, "shift": shift, "n_items": len(keep)})
    return {"cuts": out, "null_Rs": [c["R"] for c in out], "n_cuts": len(out)}


def sweep_stage(ctx: Ctx, half: str, layers, curve: dict, reuse_last: dict | None) -> dict:
    """R_G(L) for every layer (per_item/G_L<L>_<half>.json), then D_G(L) converse (D_L<L>_<half>.json)."""
    for L in layers:
        if reuse_last is not None and L == ctx.res_b.n_layers - 1 and str(L) not in curve["R"]:
            curve["R"][str(L)] = reuse_last
        elif str(L) not in curve["R"]:
            curve["R"][str(L)] = graft_eval(ctx, "h2b", L, half, f"G_L{L}_{half}")
        _publish(ctx, half, curve)
        ctx.bank.stage(f"R_G_L{L}_{half}", layer=L, R=curve["R"][str(L)]["R"])
    for L in layers:
        if str(L) not in curve["D"]:
            curve["D"][str(L)] = graft_eval(ctx, "b2h", L, half, f"D_L{L}_{half}")
        _publish(ctx, half, curve)
        ctx.bank.stage(f"D_G_L{L}_{half}", layer=L, D=curve["D"][str(L)]["D"])
    return curve


def _publish(ctx: Ctx, half: str, curve: dict):
    """curves[half] in the layout scripts/matrix_verdicts.graft_verdicts reads: aligned lists."""
    Ls = sorted(int(k) for k in curve["R"])
    c = {"layers": Ls, "R_G": [curve["R"][str(L)]["R"] for L in Ls], "acc_graft": [curve["R"][str(L)]["acc"] for L in Ls],
         "files": [curve["R"][str(L)]["file"] for L in Ls]}
    Ld = sorted(int(k) for k in curve["D"])
    if Ld:
        c["D_G_layers"] = Ld; c["D_G"] = [curve["D"][str(L)]["D"] for L in Ld]
        c["acc_converse"] = [curve["D"][str(L)]["acc"] for L in Ld]
    ctx.bank.m["curves"][half] = c


def iter_captures(fwd, model, layer, prompts):
    """Per batch: (layer output hidden, attention_mask) for `model` driven through `fwd` — the hidden
    from a layer hook, the mask from a pre-hook on the outer model (shell-agnostic)."""
    from infra.pg.cached_donor import _hidden, swap_model
    prompts = list(prompts)
    for s in range(0, len(prompts), BATCH):
        box = {}
        h1 = model.register_forward_pre_hook(lambda m, a, kw: box.__setitem__("mask", kw.get("attention_mask")), with_kwargs=True)
        h2 = layer.register_forward_hook(lambda m, a, o: box.__setitem__("h", _hidden(o)))
        try:
            with swap_model(fwd, model):
                fwd.next_token_logits(prompts[s:s + BATCH])
        finally:
            h1.remove(); h2.remove()
        assert "h" in box and box.get("mask") is not None, "capture needs input attention_mask + a layer hit"
        yield box["h"], box["mask"]


def direction_basis(ctx: Ctx, L: int, half: str, ks) -> dict:
    """Top-k principal directions (eigenvectors of the centred second moment, fp64) of the
    healthy − broken layer-L residual difference over every real position of the half's prompts
    (streams flattened into samples on 4-stream models). Unsupervised — no labels touch it."""
    import numpy as np
    import torch
    prompts = ctx.prompts(half)
    d = None; n = 0; sx = None; xx = None
    pairs = zip(iter_captures(ctx.fwd_h, ctx.fwd_h.model, ctx.res_h.layers[L], prompts),
                iter_captures(ctx.fwd_b, ctx.fwd_b.model, ctx.res_b.layers[L], prompts))
    for (hh, mh), (hb, mb) in pairs:
        assert hh.shape == hb.shape and torch.equal(mh, mb), "pair encodings differ (shell parity broken)"
        diff = (hh.float() - hb.float())
        m = mh.bool()
        if diff.dim() == 4:                                   # [B, S, streams, D] → mask over (B, S)
            m = m[:, :, None].expand(-1, -1, diff.shape[2])
        X = diff[m].reshape(-1, diff.shape[-1]).double().cpu().numpy()
        if d is None:
            d = X.shape[1]; sx = np.zeros(d); xx = np.zeros((d, d))
        n += X.shape[0]; sx += X.sum(0); xx += X.T @ X
    assert n > 1, n
    mean = sx / n
    cov = xx / n - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
    total = float(evals.sum())
    out = {"d": int(d), "n_positions": int(n), "bases": {}, "explained_variance_frac": {}, "k_effective": {}}
    for k in ks:
        ke = min(int(k), int(d))
        out["bases"][k] = torch.tensor(evecs[:, :ke].copy(), dtype=torch.float32)
        out["explained_variance_frac"][str(k)] = float(evals[:ke].sum() / total) if total > 0 else float("nan")
        out["k_effective"][str(k)] = ke
    return out


def cut_mixes(basis: dict | None, ks, names, device):
    """The `mix(donor, recipient) -> tensor` hooks of spec §6.4's cuts."""
    import torch
    mixes = {}
    if "final_position" in names:
        mixes["final_position"] = lambda dnr, h: torch.cat([h[:, :-1], dnr[:, -1:]], 1)
    for k in ks:
        if f"top{k}" in names:
            U = basis["bases"][k].to(device)

            def mk(U):
                return lambda dnr, h: (h.float() + (dnr.float() - h.float()) @ U @ U.T).to(h.dtype)
            mixes[f"top{k}"] = mk(U)
    return mixes


def cuts_stage(ctx: Ctx, half: str, L: int, names, ks) -> dict:
    device = next(ctx.fwd_b.model.parameters()).device
    basis = direction_basis(ctx, L, half, ks) if any(n.startswith("top") for n in names) else None
    out = {"layer": int(L), "cuts": {}}
    if basis is not None:
        out["basis"] = {k: v for k, v in basis.items() if k != "bases"}
    for name, mix in cut_mixes(basis, ks, names, device).items():
        g = graft_eval(ctx, "h2b", L, half, f"cut_{name}_L{L}_{half}", mix=mix)
        out["cuts"][name] = g
        ctx.bank.m["cuts"][half] = out
        ctx.bank.stage(f"cut_{name}_{half}", layer=L, R=g["R"])
    return out


def transplant_stage(ctx: Ctx, half: str, L_lo: int, L_hi: int) -> dict:
    """W (spec §1.2 Transplant, §6.4 W): copy the healthy model's weights for a config's parameter set
    INTO the broken model (resolver.key_remap), evaluate the broken model ALONE, restore from the
    snapshot and assert every tensor is back exactly. Banks R_prefix_Llo / R_G_Llo / R_band / R_band_tail."""
    import torch
    from infra.pg import family_resolvers as FRS
    assert not ctx.cached, "the transplant needs the co-resident pair (weights of both sides)"
    res_h, res_b = ctx.res_h, ctx.res_b
    model_b = ctx.fwd_b.model
    part_h, part_b = FRS.partition(res_h), FRS.partition(res_b)
    sd_h, sd_b = ctx.fwd_h.model.state_dict(), model_b.state_dict()
    r2d = {}
    for dk in FRS.lm_keys(part_h):
        rk = res_b.key_remap(dk)
        assert rk not in r2d, f"two donor keys map to {rk}"
        r2d[rk] = dk
    universe = FRS.lm_keys(part_b)
    assert set(universe) == set(r2d), f"LM key universes differ across the pair: {sorted(set(universe) ^ set(r2d))[:5]}"
    for k in universe:
        assert sd_h[r2d[k]].shape == sd_b[k].shape and sd_h[r2d[k]].dtype == sd_b[k].dtype, (k, sd_h[r2d[k]].shape, sd_b[k].shape)
    snap = {k: sd_b[k].detach().clone().cpu() for k in universe}
    head_keys = part_b["embed"] + part_b["lm_head"]
    embed_same = all(torch.equal(sd_h[r2d[k]], sd_b[k]) for k in head_keys)

    def load(names, src):
        res = model_b.load_state_dict({k: src(k) for k in names}, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        assert set(res.missing_keys) == set(sd_b) - set(names), "missing-key set drifted"

    def transplant(names):
        load(names, lambda k: sd_h[r2d[k]])
        for k in names:
            assert torch.equal(sd_b[k], sd_h[r2d[k]]), f"transplant did not land: {k}"

    def restore(names):
        load(names, lambda k: snap[k].to(sd_b[k].device))
        for k in names:
            assert torch.equal(sd_b[k].cpu(), snap[k]), f"restore did not land: {k}"

    ah, ab = _base(ctx, half)
    base = ctx.bank.m["baselines"]
    out = {"half": half, "L_lo": int(L_lo), "L_hi": int(L_hi), "n_lm_keys": len(universe),
           "embed_and_head_identical_across_pair": bool(embed_same), "configs": {}}
    ctx.bank.m["transplant"] = out
    for name, spec in transplant_configs(res_b.n_layers, L_lo, L_hi):
        names = FRS.config_keys(part_b, spec)
        transplant(names)
        rec = ctx.S(ctx.fwd_b, model_b, half)
        restore(names)
        row = {"layers": [int(l) for l in spec["layers"]], "modules": list(spec["modules"]),
               "final_norm": bool(spec.get("final_norm")), "embed": bool(spec.get("embed")), "n_keys": len(names),
               "acc": rec["acc"], "R": restoration(rec["acc"], ab, ah), "file": ctx.bank.per_item(f"W_{name}_{half}", rec),
               "extra": _samples_only(rec)}
        if name == "none":
            row["sanity_equals_broken"] = bool(rec["per_item"] == base["broken"][half]["per_item"])
            assert row["sanity_equals_broken"], "SANITY: config 'none' != broken baseline per item"
        if name == "all":
            row["sanity_equals_healthy"] = bool(rec["per_item"] == base["healthy"][half]["per_item"])
            assert row["sanity_equals_healthy"], "SANITY: config 'all' != healthy baseline per item (shared shell assumed)"
        out["configs"][name] = row
        ctx.bank.stage(f"W_{name}_{half}", n_keys=len(names), R=row["R"])
    rec = ctx.S(ctx.fwd_b, model_b, half)
    assert rec["per_item"] == base["broken"][half]["per_item"], "WIRING: the restored broken model drifted from its baseline"
    out["restored_equals_broken"] = True
    cv = ctx.bank.m["curves"].get(half, {"layers": [], "R_G": []})
    if int(L_lo) in cv["layers"]:
        R_G_Llo = cv["R_G"][cv["layers"].index(int(L_lo))]
    else:
        R_G_Llo = graft_eval(ctx, "h2b", L_lo, half, f"G_L{L_lo}_{half}")["R"]
    out.update({"R_prefix_Llo": out["configs"][f"prefix_0_{L_lo}"]["R"], "R_G_Llo": float(R_G_Llo),
                "R_band": out["configs"]["band"]["R"], "R_band_tail": out["configs"]["band_tail"]["R"]})
    return out


def equivalence_stage(ctx: Ctx, half: str) -> dict:
    """§11.6 gate (co-resident row): cached self-graft and cached contrast graft reproduce the
    co-resident ones to the item."""
    from infra.pg.cached_donor import CachedPatchedLM, PatchedLM, cache_layer, equivalence_check
    n = ctx.res_b.n_layers
    prompts = ctx.prompts(half)
    out = {}
    Ls = n // 2, n - 1
    cache_layer(ctx.fwd_b, ctx.res_b, prompts, Ls[0], ctx.cache_path("broken", Ls[0]))
    out["self_graft"] = equivalence_check(ctx.fwd_b, PatchedLM(ctx.fwd_b.model, ctx.fwd_b.model, ctx.res_b.layers, ctx.res_b.layers, Ls[0]),
                                    CachedPatchedLM(ctx.fwd_b.model, ctx.res_b, Ls[0], ctx.cache_path("broken", Ls[0])), prompts)
    cache_layer(ctx.fwd_h, ctx.res_h, prompts, Ls[1], ctx.cache_path("healthy", Ls[1]))
    out["contrast"] = equivalence_check(ctx.fwd_b, PatchedLM(ctx.fwd_h.model, ctx.fwd_b.model, ctx.res_h.layers, ctx.res_b.layers, Ls[1]),
                                        CachedPatchedLM(ctx.fwd_b.model, ctx.res_b, Ls[1], ctx.cache_path("healthy", Ls[1])), prompts)
    out["pass"] = bool(out["self_graft"]["pass"] and out["contrast"]["pass"])
    assert out["pass"], f"EQUIVALENCE FAILED: {out}"
    return out


def job_side_T(null_Rs) -> float:
    from infra.pg.matrix_core import null_spread, threshold_T
    return threshold_T(null_spread(null_Rs))


def head_fingerprint(model, res) -> dict:
    """sha256 of every embedding + lm_head tensor (shape, dtype, bytes), keyed by this model's keys —
    compared across the pair through `key_remap`, so it also works when the two models are never
    co-resident (cached mode)."""
    import hashlib
    import torch
    from infra.pg import family_resolvers as FRS
    part = FRS.partition(res)
    sd = model.state_dict()
    out = {}
    for k in part["embed"] + part["lm_head"]:
        t = sd[k].detach().cpu().contiguous()
        raw = t.view(getattr(torch, f"int{8 * t.element_size()}")).numpy().tobytes()
        out[k] = hashlib.sha256(f"{tuple(t.shape)}|{t.dtype}|".encode() + raw).hexdigest()
    return out


def prefix_control_applicable(fp_h: dict, fp_b: dict, res_b) -> bool:
    """Spec §6.4: True iff embedding AND head are identical across the pair — only then can the weight
    prefix control 0..L_lo reproduce the activation prefix graft (D17). False on rows whose head
    changed (1b, 3b, 3c-Act, 4): the verdict script prints 'not applicable'."""
    if len(fp_h) != len(fp_b):
        return False
    return all(fp_b.get(res_b.key_remap(hk)) == v for hk, v in fp_h.items())


def run(ctx: Ctx, a, load_side=None) -> dict:
    """The runbook for ONE half. Co-resident: both Forwards attached by `launch`. Cached
    (`a.cached_donor`): `load_side(side) -> one load_pair MEMBER DICT` (not a Forward — `Ctx.attach`
    takes the member and asserts it; this docstring said "Forward" while the code passed a member, the
    same prose-vs-contract drift that cost three GPU packs on 2026-09-03) loads one model at a time; the healthy model is
    cached at every needed layer, released, the broken model evaluated from the cache and itself
    cached for the converse, released, the healthy model reloaded for D_G (it stays resident for a
    following half). Pass 2 scores the stride grid ∪ the bracketed layers on the confirmation half
    (§6.13) — the layer CHOICE comes from discovery, every number in the curve from confirmation."""
    bank, half = ctx.bank, a.half
    layers_req = None if a.layers is None else [int(x) for x in a.layers.split(",") if x != ""]

    def plan(n_layers):
        stride = max(a.stride, n_layers // 3) if a.smoke else a.stride
        grid = sweep_layers(n_layers, stride)
        if a.pass_ == 2:
            assert half == "confirmation", "pass 2 scores the confirmation half (§6.13)"
            assert layers_req is not None, "--pass 2 needs --layers (the bracket from the discovery band; may be empty)"
            return sorted(set(grid) | set(layers_req))
        return grid

    if ctx.cached and ctx.fwd_h is None:
        assert load_side is not None
        ctx.attach("healthy", load_side("healthy"))
    n = ctx.res_h.n_layers
    layers = plan(n)
    bank.m.update({"n_layers": n, "family": ctx.res_h.family, "shell": ctx.fwd_h.shell, "batch": BATCH, "shots": ctx.shots})
    bank.m.setdefault("runs", {})[half] = {"pass": a.pass_, "sweep_layers": layers, "layers_given": layers_req, "stride": a.stride}
    selfs = self_layers(n)
    ctrl = {"half": half, "self_layers": selfs, "null_cuts": null_cuts(n, a.n_null)}
    bank.m.setdefault("controls_by_half", {})[half] = ctrl
    bank.m.setdefault("controls", ctrl)                 # the verdict script reads the FIRST half's gates + nulls
    bank.stage(f"manifest_{half}", n_layers=n, pass_=a.pass_, sweep_layers=layers, family=ctx.res_h.family)
    fp_h = head_fingerprint(ctx.fwd_h.model, ctx.res_h)

    # ---- gates on the healthy side ----
    idh = identity_stage(ctx, "healthy", half, selfs)
    ctrl["selfgraft_healthy"] = idh
    bank.stage(f"identity_healthy_{half}", **{k: idh[k] for k in ("item_match_frac", "max_abs_dlogit")})
    w = wiring_stage(ctx, half)
    ctrl.update({"wiring": w, "donorhead_item_match_frac": w["item_match_frac"], "donorhead_max_abs_dlogit": w["max_abs_dlogit"]})
    bank.stage(f"wiring_{half}", **{k: w[k] for k in ("item_match_frac", "max_abs_dlogit", "hook_name")})
    if not (w["item_match_frac"] >= 0.99 and w["max_abs_dlogit"] <= 1e-2):      # §8b: diagnostics saved, then stop
        bank.m["void_wiring"] = half
        bank.stage(f"EXIT_VOID_WIRING_{half}", reason="donor residual through the donor head does not reproduce the donor logits")
        print("GRAFT_VOID_WIRING " + json.dumps(w), flush=True)
        return bank.m
    ctrl["hook_names_healthy"] = assert_hook_names(ctx.fwd_h.model, ctx.res_h)
    if "healthy" not in bank.m["baselines"]:
        bank.m["baselines"]["healthy"] = baseline_stage(ctx, "healthy")
        bank.stage(f"baselines_healthy_{half}", **{h: v["acc"] for h, v in bank.m["baselines"]["healthy"].items()})

    if ctx.cached:
        need = sorted(set(layers) | {n - 1} | {L for L, _ in null_cuts(n, a.n_null)})
        from infra.pg.cached_donor import cache_layer
        prompts = ctx.prompts(half)
        for L in need:
            idx = cache_layer(ctx.fwd_h, ctx.res_h, prompts, L, ctx.cache_path("healthy", L))
            bank.stage(f"cache_healthy_L{L}_{half}", n_batches=len(idx))
        ctx.fwd_h = ctx.res_h = None                    # the resolver holds the layer modules too
        _release()
        ctx.attach("broken", load_side("broken"))

    # ---- gates on the broken side ----
    assert ctx.res_b.n_layers == n, "pair depth differs"
    idb = identity_stage(ctx, "broken", half, selfs)
    ctrl["selfgraft_broken"] = idb
    ctrl["selfgraft_item_match_frac"] = min(idh["item_match_frac"], idb["item_match_frac"])
    ctrl["selfgraft_max_abs_dlogit"] = max(idh["max_abs_dlogit"], idb["max_abs_dlogit"])
    bank.stage(f"identity_broken_{half}", **{k: idb[k] for k in ("item_match_frac", "max_abs_dlogit")})
    ctrl["hook_names_broken"] = assert_hook_names(ctx.fwd_b.model, ctx.res_b)
    ctrl["prefix_control_applicable"] = prefix_control_applicable(fp_h, head_fingerprint(ctx.fwd_b.model, ctx.res_b), ctx.res_b)
    if "broken" not in bank.m["baselines"]:
        bank.m["baselines"]["broken"] = baseline_stage(ctx, "broken")
        bank.stage(f"baselines_broken_{half}", **{h: v["acc"] for h, v in bank.m["baselines"]["broken"].items()})
        bank.m["baselines"]["lost_margin"] = {h: _base(ctx, h)[0] - _base(ctx, h)[1] for h in bank.m["baselines"]["broken"]}
    _base(ctx, half)

    # ---- R_G(last) through the RECIPIENT head (a result, never a gate) ----
    last = graft_eval(ctx, "h2b", n - 1, half, f"G_L{n - 1}_{half}")
    ctrl["R_G_last_recipient_head"] = last["R"]
    ctrl["R_G_last"] = last
    bank.stage(f"R_G_last_{half}", R=last["R"], acc=last["acc"])

    # ---- nulls → job-side T ----
    nl = null_stage(ctx, half, a.n_null)
    ctrl.update({"nulls": nl["cuts"], "null_Rs": nl["null_Rs"]})
    T = job_side_T(nl["null_Rs"])
    ctrl["T_job_side"] = T
    bank.stage(f"nulls_{half}", null_Rs=nl["null_Rs"], T_job_side=T)

    # ---- sweep: R_G then D_G ----
    curve = {"R": {}, "D": {}}
    if ctx.cached:
        for L in layers:
            curve["R"][str(L)] = last if L == n - 1 else graft_eval(ctx, "h2b", L, half, f"G_L{L}_{half}")
            _publish(ctx, half, curve)
            bank.stage(f"R_G_L{L}_{half}", layer=L, R=curve["R"][str(L)]["R"])
    else:
        sweep_stage(ctx, half, layers, curve, reuse_last=last)
    cv = bank.m["curves"][half]
    from infra.pg.matrix_core import graft_band
    if len(cv["layers"]) >= 4:
        # The curve's LAST point is R_G(last) (set from `last` above / `sweep_stage(reuse_last=…)`) — the
        # full-prefix graft, i.e. the wiring control, 1.0 by construction on an intact recipient head.
        # It stays IN the banked curve so the figure and the head-stage diagnostic keep it, and
        # `graft_band` excludes it from band-finding and max_R by default (2026-09-03 audit).
        gb = graft_band(cv["layers"], cv["R_G"], T)
        jb = {**gb, "T": T, "pass2_suggested": (list(range(gb["L_lo"] + 1, gb["L_hi"])) if gb["band"] else None)}
    else:
        jb = {"L_lo": None, "L_hi": None, "band": None, "note": "< 4 sweep points", "T": T}
    bank.m.setdefault("job_side_band", {})[half] = jb
    bank.stage(f"sweep_R_G_done_{half}", **{k: v for k, v in jb.items() if k != "note"})

    # ---- cuts at the best layer ----
    names = [] if a.cuts == "none" else (list(CUT_NAMES) if a.cuts == "all" else [x for x in a.cuts.split(",") if x])
    ks = [k for k in CUT_KS if f"top{k}" in names] if not a.smoke else [k for k in CUT_KS[:1] if f"top{k}" in names]
    names = [x for x in names if x == "final_position" or int(x[3:]) in ks]
    if names and not ctx.cached:
        Lc, rule = (int(a.cut_layer), "--cut-layer") if a.cut_layer is not None else best_layer(cv["layers"], cv["R_G"], T, n)
        bank.m["cuts"][half] = {"layer": Lc, "rule": rule}
        cuts_stage(ctx, half, Lc, names, ks)
        bank.m["cuts"][half]["rule"] = rule
        bank.stage(f"cuts_done_{half}", layer=Lc, rule=rule)
    elif names:
        bank.stage(f"cuts_skipped_{half}", reason="cuts need both models resident (cached-donor mode)")

    # ---- converse in cached mode (broken cached, healthy reloaded) ----
    if ctx.cached:
        from infra.pg.cached_donor import cache_layer
        prompts = ctx.prompts(half)
        for L in layers:
            idx = cache_layer(ctx.fwd_b, ctx.res_b, prompts, L, ctx.cache_path("broken", L))
            bank.stage(f"cache_broken_L{L}_{half}", n_batches=len(idx))
        ctx.fwd_b = ctx.res_b = None
        _release()
        ctx.attach("healthy", load_side("healthy"))
        for L in layers:
            curve["D"][str(L)] = graft_eval(ctx, "b2h", L, half, f"D_L{L}_{half}")
            _publish(ctx, half, curve)
            bank.stage(f"D_G_L{L}_{half}", layer=L, D=curve["D"][str(L)]["D"])

    # ---- W transplant ----
    if a.transplant:
        if a.L_lo is not None or a.L_hi is not None:
            assert a.L_lo is not None and a.L_hi is not None, "--L-lo and --L-hi go together"
            lo, hi = int(a.L_lo), int(a.L_hi)
        else:
            assert jb.get("band"), f"--transplant without --L-lo/--L-hi needs a job-side band; got {jb}"
            lo, hi = int(jb["L_lo"]), int(jb["L_hi"])
        t = transplant_stage(ctx, half, lo, hi)
        assert t["embed_and_head_identical_across_pair"] == ctrl["prefix_control_applicable"]
        bank.stage(f"transplant_done_{half}", **{k: t[k] for k in ("R_prefix_Llo", "R_G_Llo", "R_band", "R_band_tail")})

    if a.equivalence:
        assert not ctx.cached, "--equivalence runs on a co-resident row"
        bank.m["equivalence"] = equivalence_stage(ctx, half)
        bank.stage(f"equivalence_{half}", **{k: bank.m["equivalence"][k]["max_abs_dlogit"] for k in ("self_graft", "contrast")})

    bank.m["elapsed_s"] = round(time.time() - bank.t0, 1)
    bank.stage(f"done_{half}")
    print("GRAFT_OK " + json.dumps({"row": bank.m.get("row"), "half": half, "pass": a.pass_, "R_G_last": last["R"],
                                    "T_job_side": T, "band": jb.get("band"),
                                    "prefix_control_applicable": ctrl["prefix_control_applicable"]}), flush=True)
    return bank.m


SIDE = {"healthy": "before", "broken": "after"}     # spec §1.2 roles ↔ the contract loader's names


def member_manifest(member: dict) -> dict:
    """The JSON-able provenance of one loaded member. `load_pair` has no top-level "manifest" key —
    `pair.get("manifest")` was silently None here, so every banked graft record would have carried an
    empty `pair_manifest` even on a successful run. Same fields matrix_behaviour banks per member."""
    return {k: member[k] for k in ("which", "repo", "shell", "local", "info")}


def launch(ctx: Ctx, a, load_pair, device) -> dict:
    """Load through the contract loader `load_pair(row, device, which=...)` → one member dict per
    requested side, {"before": member, "after": member} (healthy = before, broken = after), run
    pass 1, and with --then-pass2 run pass 2 on the confirmation half on the SAME bank: bracket =
    the discovery band's interior, cut = the single best discovery cut at its layer (§6.13, k = 1),
    transplant band = the discovery band (W is confirmatory, so it runs on the confirmation pass)."""
    import copy
    bank = ctx.bank
    load_side = None
    if ctx.cached:
        def load_side(side):
            member = load_pair(a.row, device, which=(SIDE[side],))[SIDE[side]]
            bank.m.setdefault("pair_manifest", {})[side] = member_manifest(member)
            return member
    else:
        pair = load_pair(a.row, device, which=("before", "after"))
        bank.m["pair_manifest"] = {side: member_manifest(pair[w]) for side, w in SIDE.items()}
        ctx.attach("healthy", pair["before"])
        ctx.attach("broken", pair["after"])
    if not a.then_pass2:
        return run(ctx, a, load_side)
    assert a.pass_ == 1 and a.half == "discovery", "--then-pass2 starts from pass 1 on the discovery half"
    a1 = copy.copy(a)
    a1.transplant = False
    run(ctx, a1, load_side)
    if bank.m.get("void_wiring"):
        return bank.m
    jb = bank.m["job_side_band"]["discovery"]
    bracket = list(range(jb["L_lo"] + 1, jb["L_hi"])) if jb.get("band") else []
    a2 = copy.copy(a)
    a2.half, a2.pass_, a2.layers, a2.then_pass2, a2.equivalence = "confirmation", 2, ",".join(map(str, bracket)), False, False
    cuts = bank.m["cuts"].get("discovery", {}).get("cuts", {})
    if cuts:
        a2.cuts = max(cuts, key=lambda k: cuts[k]["R"])
        a2.cut_layer = bank.m["cuts"]["discovery"]["layer"]
    else:
        a2.cuts = "none"
    if a.transplant and a.L_lo is None:
        assert jb.get("band"), f"--transplant needs a band; discovery found none: {jb}"
        a2.L_lo, a2.L_hi = jb["L_lo"], jb["L_hi"]
    bank.m["then_pass2"] = {"bracket": bracket, "cuts": a2.cuts, "cut_layer": a2.cut_layer, "L_lo": a2.L_lo, "L_hi": a2.L_hi,
                            "chosen_on": "discovery"}
    return run(ctx, a2, load_side)


def _release():
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------- CLI
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", default="")
    ap.add_argument("--half", default="discovery", choices=HALVES)
    ap.add_argument("--pass", dest="pass_", type=int, default=1, choices=(1, 2))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--layers", default=None, help="pass 2: the bracketed layers (grid ∪ bracket is scored), comma-separated")
    ap.add_argument("--then-pass2", dest="then_pass2", action="store_true",
                    help="after pass 1 on discovery, run pass 2 on confirmation with the discovery-chosen bracket/cut/band")
    ap.add_argument("--out", default="")
    ap.add_argument("--cached-donor", dest="cached_donor", action="store_true")
    ap.add_argument("--cache-dir", dest="cache_dir", default="")
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--transplant", action="store_true")
    ap.add_argument("--L-lo", dest="L_lo", type=int, default=None)
    ap.add_argument("--L-hi", dest="L_hi", type=int, default=None)
    ap.add_argument("--cuts", default="all", help="all | none | comma list of " + ",".join(CUT_NAMES))
    ap.add_argument("--cut-layer", dest="cut_layer", type=int, default=None)
    ap.add_argument("--n-null", dest="n_null", type=int, default=N_NULL)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def _smoke_items(items: list) -> list:
    seen, out = set(), []
    for it in items:
        if it["subject"] not in seen:
            seen.add(it["subject"]); out.append(it)
    return out


MODULE = "infra.aws_pg.jobs.matrix_graft"


def main(argv=None):
    # `python -m <job>` calls main() with argv=None, so the pin re-exec must resolve the real argv
    # before handing it to the child — `subprocess.run([py, "-m", mod, *None])` raises
    # "TypeError: Value after * must be an iterable, not NoneType". matrix_behaviour normalises on its
    # first line (`argv = list(sys.argv[1:] if argv is None else argv)`); these two did not, so my own
    # pin fix crashed the 3b recovery pack on its next launch.
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse(argv)
    if a.dryrun:
        return _dryrun()
    assert a.row and a.out, "--row and --out are required"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # §8b transformers pin. `matrix_behaviour` has always done this; these two jobs never did, so a
    # Molmo2 row's remote modeling code ran against the AMI's transformers 5.9.0 instead of its declared
    # 4.57.1. It fails as `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS[self.rope_type]` inside
    # `modeling_molmo2.py` — an error that names neither transformers nor the pin. The 2026-09-03
    # confound audit flagged this exact gap as item 30 ("matrix_graft.py / matrix_recovery.py never call
    # it"); it cost the 3b recovery packs two launches on 2026-09-04 before the prediction was matched to
    # the traceback. A row with no pin returns None here and falls straight through.
    from infra.pg.matrix_models import ensure_transformers_pin
    if ensure_transformers_pin(a.row, MODULE, argv) is not None:
        return

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import matrix_items as MI
    from infra.pg.matrix_models import load_pair          # contract §1: load_pair(row, device, which=...)
    assert torch.cuda.is_available(), "the graft job runs on a GPU box"
    device = "cuda"
    items = {h: MI.mmlu_eval(h) for h in HALVES}
    if a.smoke:
        items = {h: _smoke_items(v) for h, v in items.items()}
    import importlib.metadata as md
    meta = {"row": a.row, "argv": list(argv) if argv is not None else None, "smoke": a.smoke, "cached_donor": a.cached_donor,
            "items": {h: {"n": len(v), "sha256": MI._sha256(v)} for h, v in items.items()},
            "env": {"torch": torch.__version__, "transformers": md.version("transformers"),
                    "git_sha": os.environ.get("PG_GIT_SHA") or os.environ.get("GIT_SHA"),
                    "instance_type": os.environ.get("PG_INSTANCE_TYPE")}}
    bank = Bank(a.out, meta)
    from infra.pg.util_probe import DmonSampler
    DmonSampler(bank.out / "events.jsonl").start()
    cache_dir = a.cache_dir or str(pathlib.Path(a.out) / "donor_cache")
    ctx = Ctx(bank, items, MI.mmlu_dev_shots, a.shots, cache_dir, cached=a.cached_donor)
    launch(ctx, a, load_pair, device)


# ---------------------------------------------------------------------------------------- dry-run
_STUB = {}


def _stub_pair(n_layers=2, seed=0):
    """Healthy = stub; broken = a copy with the LAST layer's weights sign-flipped. Items: the gold
    letter is the healthy model's own argmax, so acc_healthy = 1 and the lesion costs items. Memoised
    so the stub loader returns the same objects on every call."""
    if "pair" in _STUB:
        return _STUB["pair"]
    import copy
    import torch
    from infra.pg import family_resolvers as FRS
    from infra.pg.format_readouts import s_prompt
    from infra.pg.text_eval import CHOICES
    healthy = FRS.stub_model("llama", n_layers=n_layers, seed=seed)
    broken = copy.deepcopy(healthy)
    with torch.no_grad():                      # sign-flip lesion of the last layer (moves the letter argmax)
        ly = broken.model.layers[-1]
        ly.mlp.weight.mul_(-3.0)
        ly.self_attn.weight.mul_(-3.0)
    fwd_h, fwd_b = FRS.StubForward(healthy), FRS.StubForward(broken)
    subjects = [f"subj{i}" for i in range(6)]
    items = {}
    for half, salt in (("discovery", "d"), ("confirmation", "c")):
        # StubForward encodes only the LAST 24 characters, so the option texts carry the item identity
        raw = [{"subject": s, "question": f"Q{salt}{i} of {s}: what?", "choices": [f"{s[-1]}{i}{j}{salt}" for j in range(4)],
                "answer": 0, "hash": f"{s}-{salt}-{i}"} for s in subjects for i in range(3)]
        ids = fwd_h.letter_ids(CHOICES, True)
        lg = fwd_h.next_token_logits([s_prompt(it, []) for it in raw])
        for it, row in zip(raw, lg):
            it["answer"] = int(row[ids].argmax())
        items[half] = raw
    _STUB["pair"] = (fwd_h, fwd_b, items)
    return _STUB["pair"]


def _stub_member(fwd, which: str, row="stub", device="cpu") -> dict:
    """One `matrix_models.load_pair` member over a stub Forward, checked against the real contract."""
    from infra.pg.matrix_forward import _StubTok
    from infra.pg.matrix_models import assert_member
    return assert_member({"fwd": fwd, "model": fwd.model, "tok": _StubTok(), "proc": None,
                          "repo": f"stub/{row}/{which}", "which": which, "shell": fwd.shell,
                          "info": {"row": row, "which": which, "device": device, "stub": True},
                          "local": None}, which)


def _stub_load_pair(row, device, which=("before", "after")) -> dict:
    """The stub loader is a CONTRACT, not a convenience: it must return EXACTLY what
    matrix_models.load_pair returns — a member dict per side (`assert_member`), keyed by `which` when
    several are asked for, bare when one string is. Until 2026-09-03 it returned the Forward itself
    plus keys (`raw`, `manifest`) the real loader has never had; `--dryrun` passed while every real
    graft died in `ctx.attach`. A stub that does not match the interface it stands for turns a
    dry-run into a rubber stamp, so both sides now go through `matrix_models.assert_member`."""
    fwd_h, fwd_b, _ = _stub_pair()
    full = {"before": fwd_h, "after": fwd_b}
    if isinstance(which, str):                       # the real loader's single-member form
        assert which in full, which
        return _stub_member(full[which], which, row, device)
    assert set(which) <= set(full), which
    return {w: _stub_member(full[w], w, row, device) for w in which}


def _dryrun():
    import copy
    import dataclasses
    import tempfile
    from infra.pg import matrix_core as MC
    from infra.pg.cached_donor import PatchedLM
    fwd_h, fwd_b, items = _stub_pair()
    dev = lambda subject, k: []
    # ---- the stub loader is a CONTRACT: its shape is the real loader's, down to the documented keys.
    #      `_stub_load_pair` used to return the Forward itself where load_pair returns a member dict;
    #      the dry-run passed and every real graft died in `ctx.attach` (2026-09-03). These three
    #      lines are what makes a green dry-run mean anything about the real path's shape.
    import re as _re
    from infra.pg import matrix_models as MM
    _doc = MM.load_pair.__doc__.split("Returns")[1].split("per member")[0]
    assert set(_re.findall(r'"(\w+)"', _doc)) == set(MM.MEMBER_KEYS), (_doc, MM.MEMBER_KEYS)
    assert set(_stub_load_pair("stub", "cpu", "before")) == set(MM.MEMBER_KEYS)          # single-member form
    _sp = _stub_load_pair("stub", "cpu", ("before", "after"))                            # keyed form
    assert set(_sp) == {"before", "after"} and all(MM.assert_member(v, w) for w, v in _sp.items())
    with tempfile.TemporaryDirectory() as td:
        # ---- A: pass 1 on discovery, then pass 2 on confirmation on the same bank (co-resident)
        out = pathlib.Path(td) / "out"
        a = parse(["--row", "stub", "--half", "discovery", "--pass", "1", "--stride", "1", "--out", str(out), "--then-pass2",
                   "--transplant", "--L-lo", "0", "--L-hi", "1", "--equivalence", "--shots", "0"])
        ctx = Ctx(Bank(out, {"row": "stub"}), items, dev, 0, out / "donor_cache", cached=False)
        m = launch(ctx, a, _stub_load_pair, "cpu")
        pm = m["pair_manifest"]                      # keyed by SIDE, one member_manifest per side
        assert set(pm) == {"healthy", "broken"} and [pm[s]["which"] for s in ("healthy", "broken")] == ["before", "after"]
        assert set(pm["healthy"]) == {"which", "repo", "shell", "local", "info"} and pm["healthy"]["shell"] == "text"
        # --- artifacts + incremental writes
        disk = json.loads((out / "metrics.json").read_text())
        assert disk["kind"] == "graft" and disk["stages"][-1] == "done_confirmation" and (out / "controls.json").exists()
        ev = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
        assert [e["stage"] for e in ev] == disk["stages"] and all("t" in e for e in ev)
        assert (out / "per_item" / "G_L1_discovery.json").exists() and (out / "per_item" / "D_L0_confirmation.json").exists()
        # --- the verdict script's inputs are all present and the frozen rules run on them
        c = disk["controls"]
        assert c["half"] == "discovery" and set(disk["controls_by_half"]) == {"discovery", "confirmation"}
        assert c["prefix_control_applicable"] is True and json.loads((out / "controls.json").read_text())["half"] == "discovery"
        g = MC.graft_identity_gate(c["selfgraft_item_match_frac"], c["selfgraft_max_abs_dlogit"],
                                   c["donorhead_item_match_frac"], c["donorhead_max_abs_dlogit"], c["R_G_last_recipient_head"])
        assert g["gate"] == "VALID" and g["identity_ok"] and g["wiring_ok"], g
        assert c["selfgraft_max_abs_dlogit"] == 0.0 and c["donorhead_max_abs_dlogit"] == 0.0
        T = MC.threshold_T(MC.null_spread(c["null_Rs"])); assert len(c["null_Rs"]) == 5 and T >= 0.10
        for half in ("discovery", "confirmation"):
            cv = disk["curves"][half]
            assert cv["layers"] == [0, 1] and abs(cv["R_G"][1] - 1.0) < 1e-9, cv     # last-layer graft restores fully (shared head)
            assert abs(cv["R_G"][0]) < 1e-9, cv                                          # layer-0 graft: the broken last layer still runs
            assert cv["D_G_layers"] == [0, 1] and abs(cv["D_G"][1] - 1.0) < 1e-9 and abs(cv["D_G"][0]) < 1e-9, cv
            assert disk["baselines"]["healthy"][half]["acc"] == 1.0 > disk["baselines"]["broken"][half]["acc"]
        assert disk["runs"]["discovery"]["pass"] == 1 and disk["runs"]["confirmation"] == {"pass": 2, "sweep_layers": [0, 1], "layers_given": [], "stride": 1}
        assert disk["then_pass2"]["bracket"] == [] and disk["then_pass2"]["cuts"] in CUT_NAMES and disk["then_pass2"]["chosen_on"] == "discovery"
        # --- cuts on discovery (all) at the last layer; ONE cut (the best) re-scored on confirmation
        cuts = disk["cuts"]["discovery"]["cuts"]
        assert set(cuts) == set(CUT_NAMES) and abs(cuts["final_position"]["R"] - 1.0) < 1e-9, cuts
        assert disk["cuts"]["discovery"]["basis"]["k_effective"] == {"8": 8, "64": 16, "512": 16}
        assert abs(cuts["top64"]["R"] - 1.0) < 1e-9 and abs(cuts["top512"]["R"] - 1.0) < 1e-9, cuts
        assert abs(disk["cuts"]["discovery"]["basis"]["explained_variance_frac"]["64"] - 1.0) < 1e-9
        assert list(disk["cuts"]["confirmation"]["cuts"]) == [disk["then_pass2"]["cuts"]] and disk["cuts"]["confirmation"]["rule"] == "--cut-layer"
        # --- transplant ran on the CONFIRMATION pass: none == broken, all == healthy, band (= the lesioned
        #     last layer) restores fully, prefix (layer 0, identical across the pair) == the activation graft
        t = disk["transplant"]; tc = t["configs"]
        assert t["half"] == "confirmation" and t["embed_and_head_identical_across_pair"] is True
        assert tc["none"]["sanity_equals_broken"] and tc["all"]["sanity_equals_healthy"]
        assert abs(tc["band"]["R"] - 1.0) < 1e-9 and abs(tc["band_tail"]["R"] - 1.0) < 1e-9 and abs(tc["solo_1"]["R"] - 1.0) < 1e-9
        assert abs(tc["prefix_0_0"]["R"]) < 1e-9 and abs(t["R_G_Llo"]) < 1e-9
        assert abs(tc["final_norm"]["R"]) < 1e-9 and "ctrl_0_0" in tc and "attn_band" in tc and "mlp_band" in tc
        assert MC.transplant_wiring_gate(t["R_prefix_Llo"], t["R_G_Llo"])["gate"] == "VALID"
        assert MC.weight_band_verdict(t["R_band"], t["R_band_tail"])["verdict"] == "H_weight-band"
        assert disk["equivalence"]["pass"] and t["restored_equals_broken"]
        # ---- B: standalone pass 2 = grid ∪ bracket on confirmation; pass 2 without --layers / on discovery fails loud
        out2 = pathlib.Path(td) / "out2"
        a2 = parse(["--row", "stub", "--half", "confirmation", "--pass", "2", "--stride", "2", "--layers", "1,0", "--out", str(out2),
                    "--cuts", "none", "--shots", "0"])
        ctx2 = Ctx(Bank(out2, {"row": "stub"}), items, dev, 0, out2 / "donor_cache", cached=False)
        m2 = launch(ctx2, a2, _stub_load_pair, "cpu")
        assert m2["curves"]["confirmation"]["layers"] == [0, 1] and m2["runs"]["confirmation"]["pass"] == 2 and m2["controls"]["half"] == "confirmation"
        assert "discovery" not in m2["curves"] and abs(m2["curves"]["confirmation"]["R_G"][1] - 1.0) < 1e-9 and not m2["cuts"]
        for bad in (["--row", "stub", "--half", "confirmation", "--pass", "2", "--out", str(out2), "--shots", "0"],
                    ["--row", "stub", "--half", "discovery", "--pass", "2", "--layers", "1", "--out", str(out2), "--shots", "0"]):
            ctxb = Ctx(Bank(out2, {"row": "stub"}), items, dev, 0, out2 / "c", cached=False)
            try:
                launch(ctxb, parse(bad), _stub_load_pair, "cpu"); raise SystemExit("pass-2 misuse accepted: " + " ".join(bad))
            except AssertionError:
                pass
        # ---- C: cached-donor mode through the contract loader, one side at a time; reproduces the co-resident curve
        out3 = pathlib.Path(td) / "out3"
        a3 = parse(["--row", "stub", "--half", "discovery", "--pass", "1", "--stride", "1", "--out", str(out3),
                    "--cached-donor", "--cuts", "none", "--shots", "0"])
        ctx3 = Ctx(Bank(out3, {"row": "stub"}), items, dev, 0, out3 / "donor_cache", cached=True)
        calls = []
        m3 = launch(ctx3, a3, lambda row, device, which: (calls.append(tuple(which)), _stub_load_pair(row, device, which))[1], "cpu")
        assert calls == [("before",), ("after",), ("before",)], calls
        cv = disk["curves"]["discovery"]
        assert m3["curves"]["discovery"]["R_G"] == cv["R_G"] and m3["curves"]["discovery"]["D_G"] == cv["D_G"]
        assert m3["controls"]["null_Rs"] == c["null_Rs"] and m3["controls"]["prefix_control_applicable"] is True
        assert (out3 / "donor_cache" / "healthy" / "L1" / "index.json").exists() and (out3 / "donor_cache" / "broken" / "L0" / "index.json").exists()
        assert m3["pair_manifest"]["healthy"]["which"] == "before" and m3["pair_manifest"]["broken"]["which"] == "after"
        # ---- D: wiring FAIL — a resolver hooked at the wrong module fails the donor-head test and the
        #      hook-name assertion while the self-graft identity passes trivially; the run banks and stops
        out4 = pathlib.Path(td) / "out4"
        ctx4 = Ctx(Bank(out4, {"row": "stub"}), items, dev, 0, out4 / "c", cached=False)
        ctx4.attach("healthy", _stub_member(copy.copy(fwd_h), "before")); ctx4.attach("broken", _stub_member(fwd_b, "after"))
        bad = dataclasses.replace(ctx4.res_h, layers=[l.self_attn for l in ctx4.res_h.layers])
        ctx4.res_h = bad
        w = wiring_stage(ctx4, "discovery")
        assert w["item_match_frac"] < 0.99 or w["max_abs_dlogit"] > 1e-2, w
        idw = identity_stage(ctx4, "healthy", "discovery", [0, 1])
        assert idw["item_match_frac"] == 1.0
        assert MC.graft_identity_gate(idw["item_match_frac"], idw["max_abs_dlogit"], w["item_match_frac"], w["max_abs_dlogit"], 1.0)["gate"] == "VOID-wiring"
        try:
            assert_hook_names(fwd_h.model, bad); raise SystemExit("wrong hook passed the name assertion")
        except AssertionError:
            pass
        a4 = parse(["--row", "stub", "--half", "discovery", "--pass", "1", "--stride", "1", "--out", str(out4), "--shots", "0"])
        m4 = run(ctx4, a4)
        assert m4["stages"][-1] == "EXIT_VOID_WIRING_discovery" and m4["void_wiring"] == "discovery" and not m4["curves"]
        # ---- E: prefix_control_applicable goes False when the head differs across the pair
        fp_h = head_fingerprint(fwd_h.model, ctx.res_h)
        other = copy.deepcopy(fwd_b.model)
        import torch
        with torch.no_grad():
            other.lm_head.weight[0, 0] += 1.0
        from infra.pg.family_resolvers import resolve
        assert prefix_control_applicable(fp_h, head_fingerprint(fwd_b.model, ctx.res_b), ctx.res_b)
        assert not prefix_control_applicable(fp_h, head_fingerprint(other, resolve(other)), resolve(other))
        # ---- helpers
        assert null_cuts(26) == [(1, 1), (7, 1), (13, 1), (19, 1), (25, 1)] and null_cuts(2) == [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]
        assert self_layers(26) == [6, 12, 19] and self_layers(2) == [0, 1] and self_layers(43) == [10, 21, 32]
        names = [n for n, _ in transplant_configs(26, 15, 22)]
        assert names[:5] == ["none", "all", "prefix_0_15", "ctrl_0_6", "ctrl_9_15"] and "solo_22" in names and "solo_15" not in names
        assert transplant_configs(26, 15, 22)[5][1]["layers"] == list(range(16, 23))
        assert best_layer([0, 8, 16, 24], [0.0, 0.0, 0.7, 0.9], 0.1, 26) == (16, "L_hi(job-side band)")
        assert best_layer([0, 1], [0.2, 0.9], 0.1, 2) == (1, "argmax R_G (no band)")
        assert fwd_b.model is ctx.fwd_b.model and isinstance(PatchedLM(fwd_h.model, fwd_b.model, ctx.res_h.layers, ctx.res_b.layers, 1), PatchedLM)
    print("PASS matrix_graft dry-run (2-layer stub pair via the contract loader: identity, wiring pass+fail, hook assertion, "
          "nulls, pass-1 → then-pass2 (grid ∪ bracket, best cut, band), standalone pass-2, cuts, transplant sanity, "
          "cached-donor equivalence, prefix_control_applicable, incremental writes)")
    return True


def _write_samples(out_dir, metrics, kind):
    """Spec §12.11: one samples.json per cell, covering every eval this job ran (the readout records it
    banked already carry `extra.samples`)."""
    import json as _json, os as _os
    from infra.pg.format_readouts import collect_samples
    sm = collect_samples(metrics)
    tmp = _os.path.join(out_dir, "samples.json.tmp")
    with open(tmp, "w") as f:
        _json.dump({"row": metrics.get("row"), "kind": kind, "smoke": metrics.get("smoke"),
                    "cols": sm}, f, indent=1, default=float)
    _os.replace(tmp, _os.path.join(out_dir, "samples.json"))
    return len(sm)


if __name__ == "__main__":
    main()


