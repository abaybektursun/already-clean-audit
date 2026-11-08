"""MENDELEEV N6 — graft primitives: the co-resident `PatchedLM`, the per-layer donor cache, and
`CachedPatchedLM` reading from it. Contract: infra/pg/MATRIX_CONTRACT.md §6 (graft job) and spec §11.6.

A graft (spec §1.2) runs donor and recipient on the SAME batch and replaces the recipient's
decoder-layer-L output residual (all positions) with the donor's. Every patched model here has the
call contract of `infra/aws_pg/jobs/broca_sweep.PatchedLM`: `model(**kw)` with the exact kwargs a
`Forward` shell passes (input_ids / attention_mask / pixel_values / logits_to_keep …) → the
recipient's output object (`.logits`). A `Forward` (matrix_forward) is driven through a graft by
swapping its `.model` (`swap_model`); attribute reads the shell needs (`dtype`, `config`) fall
through to the recipient.

  PatchedLM(donor, recipient, donor_layers, recipient_layers, L, mismatch=0, mix=None)
      co-resident: donor forward captures layer-L output; recipient forward runs with it replaced.
      `mismatch=s` rolls the donor batch by s items (the mismatched-item null, s < batch).
      `mix(donor_out, recipient_out) -> tensor` is the cut hook (None = full replacement); the
      per-position and per-direction cuts of spec §6.4 are `mix` functions in matrix_graft.
  cache_layer(donor_fwd, donor_res, prompts, L, path)
      donor pass over `prompts` in the Forward's own batches; layer-L output saved per batch to
      `path/<key>.pt`, key = sha1 of the batch's input_ids (so the recipient's batch must have the
      SAME prompts, order, batch size and padding — anything else is a loud KeyError). Saved in the
      residual's NATIVE dtype (bf16 on every row): the same bytes as the contract's fp16 and, unlike
      fp16, no overflow on Gemma-class residual outliers (|x| > 65504 at BOS positions).
  CachedPatchedLM(recipient, resolver, L, cache_path, mismatch=0, mix=None)
      the same contract, donor residual read from the cache — one model resident.
  logit_agreement(fwd, model_a, model_b, prompts) → per-item argmax match fraction + max|Δlogit|
  equivalence_check(fwd, coresident, cached, prompts) → item match 100 % and max|Δlogit| ≤ 1e-2

Fail-loud; no try/except; heavy imports inside functions (py_compiles without torch).
`python -m infra.pg.cached_donor` runs the weight-free dry-run on family_resolvers' stubs.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib

EQUIV_MAX_DLOGIT = 1e-2


def _hidden(out):
    """The hidden-state tensor of a decoder layer's output (tensor, or the first element of a tuple —
    Molmo2 returns `(hidden_states,)`; DeepSeek-V4 returns the [B, S, 4, D] stream tensor)."""
    return out[0] if isinstance(out, tuple) else out


def _with_hidden(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


def batch_key(input_ids) -> str:
    """Cache key of one batch: sha1 over the (B, S) int64 input_ids and their shape."""
    import numpy as np
    a = np.ascontiguousarray(input_ids.detach().cpu().numpy().astype(np.int64))
    return hashlib.sha1(repr(a.shape).encode() + a.tobytes()).hexdigest()[:24]


@contextlib.contextmanager
def swap_model(fwd, model):
    """Run a Forward shell through a patched model; the original is always put back."""
    orig = fwd.model
    fwd.model = model
    try:
        yield fwd
    finally:
        fwd.model = orig


class _Patch:
    """Shared replacement machinery. Subclasses provide `_donor_residual(kw) -> tensor`."""

    def __init__(self, recipient, recipient_layer, mismatch: int = 0, mix=None):
        self.recipient = recipient
        self.rl = recipient_layer
        self.mismatch = int(mismatch)
        self.mix = mix
        assert self.mismatch >= 0

    def __getattr__(self, name):            # dtype / config / device … for the Forward shells
        return getattr(self.recipient, name)

    def _donor_residual(self, kw):
        raise NotImplementedError

    def __call__(self, **kw):
        import torch
        cap = self._donor_residual(kw)
        if self.mismatch:
            assert cap.shape[0] > self.mismatch, \
                f"mismatched-item roll {self.mismatch} needs batch > {self.mismatch}, got {cap.shape[0]}"
            cap = torch.roll(cap, shifts=self.mismatch, dims=0)
        fired = []

        def rep_hook(m, a, out):
            h = _hidden(out)
            assert h.shape == cap.shape, f"patch shape mismatch: recipient {tuple(h.shape)} vs donor {tuple(cap.shape)}"
            new = cap if self.mix is None else self.mix(cap, h)
            assert new.shape == h.shape and new.dtype == h.dtype, (new.shape, new.dtype, h.shape, h.dtype)
            fired.append(1)
            return _with_hidden(out, new)

        hd = self.rl.register_forward_hook(rep_hook)
        try:
            with torch.inference_mode():
                out = self.recipient(**kw)
        finally:
            hd.remove()
        assert len(fired) == 1, f"replacement hook fired {len(fired)} times (expected exactly once)"
        return out


class PatchedLM(_Patch):
    """Co-resident graft (broca_sweep.PatchedLM contract + `mix`)."""

    def __init__(self, donor, recipient, donor_layers, recipient_layers, L, mismatch: int = 0, mix=None):
        super().__init__(recipient, recipient_layers[L], mismatch, mix)
        self.donor, self.dl, self.L = donor, donor_layers[L], int(L)

    def _donor_residual(self, kw):
        import torch
        box = []
        hd = self.dl.register_forward_hook(lambda m, a, out: box.append(_hidden(out)))
        try:
            with torch.inference_mode():
                self.donor(**kw)
        finally:
            hd.remove()
        assert len(box) == 1, f"donor capture fired {len(box)} times"
        return box[0]


def capture_layer(model, layer, call_kwargs):
    """One forward of `model` with the given kwargs; returns (layer output hidden, model output)."""
    import torch
    box = []
    hd = layer.register_forward_hook(lambda m, a, out: box.append(_hidden(out)))
    try:
        with torch.inference_mode():
            out = model(**call_kwargs)
    finally:
        hd.remove()
    assert len(box) == 1, f"capture fired {len(box)} times"
    return box[0], out


def cache_layer(donor_fwd, donor_res, prompts, L, path) -> dict:
    """Donor pass over `prompts` (the Forward's own batching); the layer-L output of every batch is
    saved to `path/<batch_key>.pt` in its native dtype. Returns the index {key: shape} (also written
    to `path/index.json`). Batches are keyed by input_ids captured with a forward pre-hook on the
    OUTER model, so any shell (text / PaliGemma / Molmo2) works without touching its encoder."""
    import torch
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layer = donor_res.layers[int(L)]
    index, seen_keys = {}, []

    def pre_hook(m, args, kwargs):
        assert "input_ids" in kwargs, "the shell must call the model with input_ids as a keyword"
        seen_keys.append(batch_key(kwargs["input_ids"]))

    def cap_hook(m, a, out):
        h = _hidden(out).detach()
        assert torch.isfinite(h).all(), f"non-finite donor residual at layer {L}"
        key = seen_keys[-1]
        assert key not in index, f"duplicate batch in cache_layer ({key}): identical batches twice"
        torch.save(h.cpu().contiguous(), path / f"{key}.pt")
        index[key] = list(h.shape)

    h1 = donor_fwd.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(cap_hook)
    try:
        prompts = list(prompts)
        for s in range(0, len(prompts), int(donor_fwd.batch_size)):
            donor_fwd.next_token_logits(prompts[s:s + int(donor_fwd.batch_size)])
    finally:
        h1.remove()
        h2.remove()
    n_batches = -(-len(prompts) // int(donor_fwd.batch_size))
    assert len(index) == n_batches, f"{len(index)} cached batches for {n_batches} expected"
    (path / "index.json").write_text(json.dumps({"layer": int(L), "n_prompts": len(prompts),
                                                  "batch_size": int(donor_fwd.batch_size), "batches": index}))
    return index


def cache_bytes(index: dict, dtype_bytes: int = 2) -> int:
    n = 0
    for shape in index.values():
        k = 1
        for s in shape:
            k *= int(s)
        n += k * dtype_bytes
    return n


class CachedPatchedLM(_Patch):
    """Graft from a `cache_layer` directory: the donor residual of the incoming batch is looked up by
    its input_ids key (loud KeyError when the batch was not cached)."""

    def __init__(self, recipient, resolver, L, cache_path, mismatch: int = 0, mix=None):
        super().__init__(recipient, resolver.layers[int(L)], mismatch, mix)
        self.L = int(L)
        self.path = pathlib.Path(cache_path)
        idx = json.loads((self.path / "index.json").read_text())
        assert idx["layer"] == self.L, f"cache at {self.path} holds layer {idx['layer']}, asked for {self.L}"
        self.index = idx["batches"]

    def _donor_residual(self, kw):
        import torch
        key = batch_key(kw["input_ids"])
        if key not in self.index:
            raise KeyError(f"batch {key} not in the layer-{self.L} cache ({len(self.index)} batches): "
                           "prompts / order / batch size / padding differ from cache_layer")
        return torch.load(self.path / f"{key}.pt", map_location=kw["input_ids"].device)


def logits_by_batch(fwd, model, prompts):
    """Next-token logits through `fwd` with `model` swapped in, chunked by fwd.batch_size exactly as
    format_readouts._letter_read chunks (so cache keys and numerics match the readouts)."""
    import torch
    prompts, bs, outs = list(prompts), int(fwd.batch_size), []
    with swap_model(fwd, model):
        for s in range(0, len(prompts), bs):
            outs.append(fwd.next_token_logits(prompts[s:s + bs]).float())
    return torch.cat(outs, 0)


def logit_agreement(fwd, model_a, model_b, prompts) -> dict:
    """Per-item full-vocab argmax agreement and max |Δlogit| between two models driven through the same
    Forward shell on the same prompts (the identity / wiring / equivalence statistic)."""
    import torch
    la, lb = logits_by_batch(fwd, model_a, prompts), logits_by_batch(fwd, model_b, prompts)
    assert la.shape == lb.shape and la.shape[0] == len(prompts), (la.shape, lb.shape)
    assert torch.isfinite(la).all() and torch.isfinite(lb).all(), "non-finite logits in agreement check"
    match = (la.argmax(-1) == lb.argmax(-1)).float()
    return {"item_match_frac": float(match.mean()), "max_abs_dlogit": float((la - lb).abs().max()),
            "n": int(la.shape[0])}


def equivalence_check(fwd, coresident, cached, prompts) -> dict:
    """§11.6 gate: the cached graft reproduces the co-resident graft to the item (100 % argmax match)
    and to 1e-2 in every logit."""
    ag = logit_agreement(fwd, coresident, cached, prompts)
    ag["pass"] = bool(ag["item_match_frac"] >= 1.0 - 1e-12 and ag["max_abs_dlogit"] <= EQUIV_MAX_DLOGIT)
    return ag


# ===========================================================================================
# dry-run (weight-free, on family_resolvers' stub layouts)
# ===========================================================================================
def _dryrun():
    import copy
    import tempfile
    import torch
    from infra.pg import family_resolvers as FRS
    prompts = ["alpha beta gamma", "delta", "epsilon zeta eta theta", "iota kappa", "lambda"]
    for layout in ("llama", "molmo2", "deepseek_v4"):
        donor = FRS.stub_model(layout, n_layers=2, seed=0)
        recip = copy.deepcopy(donor)
        with torch.no_grad():                       # lesion the recipient's last layer
            recip_layers = FRS.resolve(recip).layers
            recip_layers[1].mlp.weight.mul_(-3.0)
        rd, rr = FRS.resolve(donor), FRS.resolve(recip)
        fwd = FRS.StubForward(recip)
        ref_d = FRS.StubForward(donor).next_token_logits(prompts)
        ref_r = fwd.next_token_logits(prompts)
        assert (ref_d.argmax(-1) != ref_r.argmax(-1)).any(), "lesion did nothing"
        # full graft at the last layer through the recipient's own head == the donor's logits exactly
        # (the head is shared, so replacing the last residual reproduces the donor bit-for-bit)
        g = PatchedLM(donor, recip, rd.layers, rr.layers, 1)
        with swap_model(fwd, g):
            got = fwd.next_token_logits(prompts)
        assert torch.equal(got, ref_d), layout
        # self-graft == identity
        ag = logit_agreement(fwd, recip, PatchedLM(recip, recip, rr.layers, rr.layers, 0), prompts)
        assert ag["item_match_frac"] == 1.0 and ag["max_abs_dlogit"] == 0.0, ag
        # mismatched roll changes the residual for every item of a batch (batch 4, then a singleton
        # batch — which cannot be rolled and must fail loud)
        fwd.batch_size = 4
        gm = PatchedLM(donor, recip, rd.layers, rr.layers, 1, mismatch=1)
        with swap_model(fwd, gm):
            lm = fwd.next_token_logits(prompts[:4])
        assert not torch.equal(lm, ref_d[:4])
        try:
            with swap_model(fwd, gm):
                fwd.next_token_logits(prompts[4:])
            raise SystemExit("singleton batch rolled silently")
        except AssertionError:
            pass
        # mix hook: final-position-only replacement at the LAST layer == full graft (only the last
        # position feeds the head); a no-op mix == unpatched recipient
        last_only = lambda d, h: torch.cat([h[:, :-1], d[:, -1:]], 1)
        with swap_model(fwd, PatchedLM(donor, recip, rd.layers, rr.layers, 1, mix=last_only)):
            assert torch.equal(fwd.next_token_logits(prompts), ref_d), layout
        with swap_model(fwd, PatchedLM(donor, recip, rd.layers, rr.layers, 1, mix=lambda d, h: h)):
            assert torch.equal(fwd.next_token_logits(prompts), ref_r), layout
        # cache → CachedPatchedLM reproduces the co-resident graft exactly; wrong batching fails loud
        with tempfile.TemporaryDirectory() as td:
            dfwd = FRS.StubForward(donor); dfwd.batch_size = 4
            idx = cache_layer(dfwd, rd, prompts, 1, td)
            assert len(idx) == 2 and cache_bytes(idx) > 0 and (pathlib.Path(td) / "index.json").exists()
            c = CachedPatchedLM(recip, rr, 1, td)
            eq = equivalence_check(fwd, g, c, prompts)
            assert eq["pass"] and eq["max_abs_dlogit"] == 0.0, eq
            fwd.batch_size = 2                       # different batch composition → not cached
            try:
                with swap_model(fwd, c):
                    fwd.next_token_logits(prompts)
                raise SystemExit("uncached batch served")
            except KeyError:
                pass
            fwd.batch_size = 4
            try:
                CachedPatchedLM(recip, rr, 0, td); raise SystemExit("wrong layer accepted")
            except AssertionError:
                pass
            # cached null cut == co-resident null cut
            cm = CachedPatchedLM(recip, rr, 1, td, mismatch=1)
            with swap_model(fwd, cm):
                lcm = fwd.next_token_logits(prompts[:4])
            assert torch.equal(lcm, lm)
        # attribute fall-through for the shells
        assert g.config.num_hidden_layers == 2
    print("cached_donor dry-run: ALL PASS (llama / molmo2-tuple / deepseek-4-stream layouts)")


if __name__ == "__main__":
    _dryrun()
