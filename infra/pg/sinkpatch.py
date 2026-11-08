"""SINKPATCH — residual-stream position patching (studies/sinkpatch/SPEC.md §2.3, §2.4).

Recorder        forward hooks on every block of a member's decoder; stores each block's output residual for the
                ONE sequence of the forward as fp32 [S, d] on the model device (batch 1 asserted).
PatchPositions  forward hooks on the named blocks that overwrite the residual at absolute positions with given
                vectors (fp32 arithmetic, cast back). Plan: {layer: (positions [k], vectors [k, d])}. Batch 1 asserted:
                with left padding a position index would not be the absolute position.
plans(...)      the arm table of SPEC §2.3 built from a parent capture and a child capture of the same prompt.
window_ids      the STAGEDIFF B3 windows (wikitext-2-raw test text, 2048 tokens, first 120 000 tokens); `window_nll`
                scores one window through a member's own StageForward exactly as `text_eval.shell_text_perplexity`
                does (targets = the window's own tokens located by `_shell_text_block`), optionally with a logit
                transform (calibration) applied before the cross-entropy.
fit_bias / fit_temperature   the calibration baseline: b = mean(logits_P − logits_C) over the fit set; T on a grid.

Fail-loud; heavy imports inside functions (py_compiles without torch).
"""
from __future__ import annotations

import random

from infra.pg.stagediff_models import BAND, V_SHARED

T_GRID = [round(0.5 + 0.05 * i, 2) for i in range(31)]           # 0.50 .. 2.00
UNRELATED_TEXT = "The committee will meet on Tuesday to review the budget."
ARMS_PATCH = ("bos_self", "bos_parent", "bos_parent_unrelated", "bos_rescaled", "bos_random", "pos1_parent",
              "posrand_parent", "bos_pos1_parent", "all_parent", "bos_parent_alllayers")
ARMS_CALIB = ("calib_bias", "calib_bias_temp")
ARMS = ("clean",) + ARMS_PATCH + ARMS_CALIB


def _residual(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


class Recorder:
    """acts[L] = block-L output residual, fp32 [S, d], for the single sequence of the forward."""

    def __init__(self, layers):
        self.layers = layers
        self.acts = {}
        self.handles = []

    def _hook(self, L):
        def h(module, args, output):
            hs = _residual(output)
            assert hs.shape[0] == 1, f"Recorder needs batch 1, got {tuple(hs.shape)}"
            self.acts[L] = hs[0].detach().float()
        return h

    def __enter__(self):
        self.acts = {}
        for L, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def stack(self):
        import torch
        return torch.stack([self.acts[L] for L in range(len(self.layers))], 0)      # [L, S, d]


class PatchPositions:
    """Overwrite the residual at absolute positions of block L's output with the plan's vectors."""

    def __init__(self, layers, plan: dict):
        self.layers = layers
        self.plan = {int(L): (pos, vec) for L, (pos, vec) in plan.items()}
        self.handles = []
        self.n_calls = 0

    def _hook(self, L):
        pos, vec = self.plan[L]

        def h(module, args, output):
            hs = _residual(output)
            assert hs.shape[0] == 1, f"PatchPositions needs batch 1, got {tuple(hs.shape)}"
            assert int(pos.max()) < hs.shape[1], (int(pos.max()), hs.shape)
            hf = hs.float().clone()
            hf[0, pos.to(hs.device)] = vec.to(hs.device).float()
            self.n_calls += 1
            return _rewrap(output, hf.to(hs.dtype))
        return h

    def __enter__(self):
        for L in sorted(self.plan):
            self.handles.append(self.layers[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []


def _plan(layers, positions, source):
    """{L: (positions, source[L][positions])} for the given layers; `source` is [L, S, d]."""
    import torch
    pos = torch.as_tensor(list(positions), dtype=torch.long)
    return {int(L): (pos, source[int(L)][pos]) for L in layers}


def random_unit(d: int, seed: int, device):
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(d, generator=g)
    return (v / v.norm()).to(device)


def rand_position(n: int, seed: int):
    """One content position p ~ U{2 .. n−2} for a sequence of n tokens (start token included); None when n < 4."""
    if n < 4:
        return None
    return random.Random(int(seed)).randint(2, n - 2)


def plans(actsP, actsC, s0_unrelated, band=BAND, n_layers=None, seed_rand: int = 1337, seed_pos: int = 0):
    """The patch arms of SPEC §2.3 for ONE sequence. actsP / actsC: [L, S, d] parent / child captures of the same
    prompt; s0_unrelated: [L, d] the parent's position-0 state captured on UNRELATED_TEXT. Returns
    ({arm: plan}, meta) with the random position (None → arm skipped) and the random-vector seeds."""
    import torch
    L_all, S, d = actsP.shape
    assert actsC.shape == actsP.shape, (actsC.shape, actsP.shape)
    n_layers = n_layers or L_all
    band = [int(L) for L in band]
    nP = actsP[:, 0].norm(dim=-1)                                            # [L]
    nC = actsC[:, 0].norm(dim=-1).clamp_min(1e-12)
    rescaled = actsC.clone()
    rescaled[:, 0] = actsC[:, 0] * (nP / nC).unsqueeze(-1)
    rnd = actsP.clone()
    for L in range(L_all):
        rnd[L, 0] = random_unit(d, seed_rand + L, actsP.device) * nP[L]
    unrel = actsP.clone()
    unrel[:, 0] = s0_unrelated.to(actsP.device)
    p = rand_position(S, seed_pos)
    out = {"bos_self": _plan(band, [0], actsC),
           "bos_parent": _plan(band, [0], actsP),
           "bos_parent_unrelated": _plan(band, [0], unrel),
           "bos_rescaled": _plan(band, [0], rescaled),
           "bos_random": _plan(band, [0], rnd),
           "pos1_parent": _plan(band, [1], actsP),
           "bos_pos1_parent": _plan(band, [0, 1], actsP),
           "all_parent": _plan(band, range(S), actsP),
           "bos_parent_alllayers": _plan(range(n_layers), [0], actsP)}
    if p is not None:
        out["posrand_parent"] = _plan(band, [p], actsP)
    meta = {"rand_position": p, "n_tokens": int(S), "rescale_factor_band_mean": float((nP / nC)[band].mean()),
            "seed_rand": seed_rand, "seed_pos": seed_pos}
    return out, meta


# ------------------------------------------------------------------------------------- forwards
def forward_last(fwd, enc):
    """Last-position logits [V] fp32 for a batch-1 encoding, through the member's own call kwargs."""
    import torch
    with torch.inference_mode():
        out = fwd.model(**fwd._call_kwargs(enc), logits_to_keep=1)                  # noqa: SLF001
    return out.logits[0, -1].float()


def forward_all(fwd, enc):
    """All-position logits [S, V] fp32 for a batch-1 encoding."""
    import torch
    with torch.inference_mode():
        out = fwd.model(**fwd._call_kwargs(enc))                                    # noqa: SLF001
    return out.logits[0].float()


def encode1(fwd, text: str):
    enc = fwd._encode([text])                                                       # noqa: SLF001
    assert enc["input_ids"].shape[0] == 1 and bool((enc["attention_mask"] == 1).all()), "batch-1 encoding has padding"
    return enc


def capture(fwd, layers, text: str):
    """(enc, last-position logits [V], residuals [L, S, d] fp32) for one text through the member's forward."""
    enc = encode1(fwd, text)
    with Recorder(layers) as rec:
        lg = forward_last(fwd, enc)
    return enc, lg, rec.stack()


# ------------------------------------------------------------------------------------- B3 windows
def window_ids(tok, text: str, window: int = 2048, max_tokens: int = 120_000) -> list:
    """The STAGEDIFF B3 windows of a text: token-id chunks of `window` over the first `max_tokens` ids."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]
    out = []
    for start in range(0, ids.size(0), window):
        chunk = ids[start:start + window]
        if chunk.numel() < 8:
            break
        out.append(chunk)
    return out


def wikitext_text() -> str:
    from datasets import load_dataset
    from infra.pg.text_eval import WIKITEXT_CONFIG, WIKITEXT_REPO
    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    return "\n\n".join(t for t in ds["text"] if t.strip())


def window_targets(fwd, win_text: str):
    """(enc, lo, hi, targets): logits rows lo..hi−1 predict `targets` — the window's own tokens located inside the
    shell's input_ids (`_shell_text_block`), first target scored from the preceding token; a no-start-token shell
    drops its first target as `shell_text_perplexity` does."""
    from infra.pg.text_eval import _shell_text_block
    enc = encode1(fwd, win_text)
    in_ids = enc["input_ids"][0]
    pos, block = _shell_text_block(fwd.tok, win_text, in_ids)
    off = 1 if pos == 0 else 0
    n_text = int(block.shape[0])
    assert n_text - off >= 1, (n_text, off)
    return enc, pos + off - 1, pos + n_text - 1, block[off:], off


def nll_of(logits, lo: int, hi: int, targets, transform=None):
    """(sum NLL, n, top id and prob at the last scored row) for all-position logits [S, V]; `transform` maps the
    scored logits [n, V] (fp32) before the cross-entropy (calibration arms)."""
    import torch
    lg = logits[lo:hi].float()
    if transform is not None:
        lg = transform(lg)
    tg = targets.to(lg.device)
    assert lg.shape[0] == tg.shape[0], (lg.shape, tg.shape)
    loss = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
    lp = torch.log_softmax(lg[-1], -1)
    top = int(lp.argmax())
    return float(loss), int(tg.numel()), top, float(lp[top].exp())


# ------------------------------------------------------------------------------------- calibration
class BiasAccumulator:
    """Running mean of (logits_P − logits_C) over fit rows: `add([n, V], [n, V])`, `.bias` → [V]."""

    def __init__(self):
        self.sum = None
        self.n = 0

    def add(self, lp, lc):
        d = (lp.float() - lc.float())
        if d.dim() == 1:
            d = d.unsqueeze(0)
        self.sum = d.sum(0) if self.sum is None else self.sum + d.sum(0)
        self.n += int(d.shape[0])

    @property
    def bias(self):
        assert self.n > 0, "no fit rows"
        return self.sum / self.n


def fit_temperature(rows: list, bias, grid=T_GRID) -> dict:
    """rows: list of (logits [n, V] fp32, targets [n]) of the fit set; picks T minimising the mean NLL of
    (logits + bias) / T. Returns {"T", "nll_by_T", "n"}."""
    import torch
    tot = {T: 0.0 for T in grid}
    n = 0
    for lg, tg in rows:
        lgb = lg.float() + bias.to(lg.device)
        tg = tg.to(lg.device)
        for T in grid:
            tot[T] += float(torch.nn.functional.cross_entropy(lgb / T, tg, reduction="sum"))
        n += int(tg.numel())
    assert n > 0
    nll = {T: tot[T] / n for T in grid}
    best = min(grid, key=lambda T: nll[T])
    return {"T": best, "nll_by_T": {str(T): nll[T] for T in grid}, "n": n, "nll_at_T1": nll[1.0]}


def calib_transform(bias, T: float = 1.0):
    def f(lg):
        return (lg + bias.to(lg.device)) / T
    return f


# ------------------------------------------------------------------------------------- dry-run
def _dryrun():
    import torch
    from infra.pg.stagediff_models import stub_member
    P, C = stub_member(bos_id=1, seed=0), stub_member(bos_id=1, seed=1)
    lp, lc = P["an"]["layers"], C["an"]["layers"]
    prompt = "Q: capital of France? Answer:"
    encP, lgP, aP = capture(P["fwd"], lp, prompt)
    encC, lgC, aC = capture(C["fwd"], lc, prompt)
    assert torch.equal(encP["input_ids"], encC["input_ids"]) and aP.shape == aC.shape and aP.shape[0] == 4
    S = aP.shape[1]
    _, _, u = capture(P["fwd"], lp, UNRELATED_TEXT)
    # the stub adds 1e-3·position to the embedding, so position 0 is prompt-independent here too
    assert torch.allclose(u[:, 0], aP[:, 0]), (u[:, 0] - aP[:, 0]).abs().max()
    pl, meta = plans(aP, aC, u[:, 0], band=(1, 2), n_layers=4, seed_pos=7)
    assert set(pl) == set(ARMS_PATCH) and meta["rand_position"] is not None and 2 <= meta["rand_position"] <= S - 2
    # identity: patching the child's own position-0 state changes nothing, at any position
    with PatchPositions(lc, pl["bos_self"]) as pp, Recorder(lc) as rec:
        lg_self = forward_last(C["fwd"], encC)
    assert pp.n_calls == 2 and torch.allclose(lg_self, lgC, atol=1e-5) and torch.allclose(rec.stack(), aC, atol=1e-5)
    # the patch writes exactly the named position: position 0 equals the parent's, every other position at the
    # patched layer's OUTPUT is unchanged (later layers may move)
    with PatchPositions(lc, {1: pl["bos_parent"][1]}), Recorder(lc) as rec:
        forward_last(C["fwd"], encC)
    a = rec.stack()
    assert torch.allclose(a[1, 0], aP[1, 0]) and torch.allclose(a[1, 1:], aC[1, 1:]) and not torch.allclose(a[2], aC[2])
    # all positions at all layers → the child's last-layer residual is the parent's
    with PatchPositions(lc, _plan(range(4), range(S), aP)), Recorder(lc) as rec:
        forward_last(C["fwd"], encC)
    assert torch.allclose(rec.stack()[3], aP[3], atol=1e-5)
    # rescaled: position-0 norm equals the parent's, direction is the child's
    pos, vec = pl["bos_rescaled"][1]
    assert abs(float(vec.norm()) - float(aP[1, 0].norm())) < 1e-4
    assert float(torch.nn.functional.cosine_similarity(vec[0], aC[1, 0], dim=0)) > 0.999
    pos, vec = pl["bos_random"][1]
    assert abs(float(vec.norm()) - float(aP[1, 0].norm())) < 1e-4
    assert torch.equal(pl["all_parent"][2][0], torch.arange(S)) and len(pl["bos_parent_alllayers"]) == 4
    assert plans(aP, aC, u[:, 0], band=(1, 2), n_layers=4, seed_pos=7)[1]["rand_position"] == meta["rand_position"]   # seeded
    # posrand: no admissible position for a 3-token sequence
    assert rand_position(3, 0) is None and rand_position(4, 0) == 2
    # windows: the round trip locates the block; a planted bias is recovered; the temperature grid picks the truth
    text = ("the quick brown fox jumps over the lazy dog. " * 40)
    wins = window_ids(C["tok"], text, window=32, max_tokens=200)
    assert len(wins) == 7 and all(w.numel() == 32 for w in wins[:6])
    enc, lo, hi, tg, off = window_targets(C["fwd"], C["tok"].decode(wins[0].tolist()))
    assert off == 0 and hi - lo == 32 == tg.numel()
    lg_all = forward_all(C["fwd"], enc)
    loss, n, top, p = nll_of(lg_all, lo, hi, tg)
    assert n == 32 and loss > 0 and 0 < p <= 1
    acc = BiasAccumulator()
    true_b = torch.randn(lg_all.shape[-1])
    acc.add(lg_all[lo:hi] + true_b, lg_all[lo:hi])
    assert torch.allclose(acc.bias, true_b, atol=1e-4) and acc.n == 32
    rows = [(lg_all[lo:hi] * 1.5 - true_b, tg)]                 # logits over-dispersed by 1.5 and shifted by −b
    ft = fit_temperature(rows, true_b)
    assert abs(ft["nll_by_T"]["1.5"] - loss / 32) < 1e-4, (ft["nll_by_T"]["1.5"], loss / 32)   # T = 1.5 recovers the clean NLL exactly
    assert ft["nll_by_T"][str(ft["T"])] <= ft["nll_by_T"]["1.5"] and ft["n"] == 32
    peaked = torch.full_like(lg_all[lo:hi], -5.0); peaked[torch.arange(32), tg] = 5.0
    assert fit_temperature([(peaked, tg)], torch.zeros_like(true_b))["T"] == 0.5     # confident-and-right → sharpest T wins
    loss_c, _, _, _ = nll_of(lg_all[lo:hi] * 1.5 - true_b, 0, 32, tg, transform=calib_transform(true_b, 1.5))
    assert abs(loss_c - loss) < 1e-3, (loss_c, loss)
    print("PASS sinkpatch dry-run (Recorder, PatchPositions exact/identity/all-layer, plans, windows, bias, temperature)")


if __name__ == "__main__":
    _dryrun()
