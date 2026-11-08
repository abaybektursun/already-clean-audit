"""STAGEDIFF lenses — spec §2 B1-int, ROT, TRACE readout; §3 `AnatomyLensModel`; §4 corpora; §5 shuffle control.

`AnatomyLensModel` implements the `jlens.protocol.LensModel` interface over `matrix_forward.lm_anatomy`, so
the vendored `jlens` (anthropics/jacobian-lens @ 581d3986, `infra/pg/vendor/jlens`, sha256 in
`vendor/jlens/VENDOR.md`) fits and applies unchanged on Qwen3 (`model.layers`) and on Molmo2/MolmoAct2
(`model.transformer.blocks`). `forward` runs the WHOLE model with an all-ones mask and `position_ids`
0..S-1 (`logits_to_keep=1` keeps the head cheap) — the same forward the behaviour readouts use, so lens
and behaviour see identical residuals. `unembed` = `lm_head(ln_f(h))[..., :V_SHARED]` (151 669 shared ids) in fp32.

Layer convention: layer ℓ = output of block ℓ (0..35). J_ℓ maps it to block 35's output; at ℓ = 35 both
lenses are the identity. `lens=None` is the logit lens (J = I). Ranks are 1-based (rank 1 = top).
Residual arithmetic is fp32 (code fact, CLAUDE.md §Machine 4).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time

from infra.pg.stagediff_models import BAND, N_LAYERS, V_SHARED

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1")
FIT_SEQS, FIT_LEN, TRACE_SEQS, TRACE_LEN, CORPUS_SEED = 64, 128, 200, 32, 0
TRACE_POSITIONS = 5           # content positions 0..4 = "j = 1..5 after BOS" when a BOS is present
_HEADING = re.compile(r"^ = [^=].*[^=] = $")


# ------------------------------------------------------------------------------------- LensModel
class AnatomyLensModel:
    def __init__(self, member: dict):
        import torch
        self.member = member
        self.model = member["model"]
        self.an = member["an"]
        self.layers = self.an["layers"]
        self.tokenizer = member["tok"]
        self.bos_id = member["bos_id"]
        self.n_layers = int(self.an["n_layers"])
        self.d_model = int(self.an["lm_head"].weight.shape[1])
        self.device = self.an["lm_head"].weight.device
        self.seqs: dict = {}
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    # registered token-id sequences (no BOS) addressed as "seq:<name>" prompts, so jlens.fit averages over
    # EXACT token windows instead of re-tokenised text (the corpus hash is over these ids)
    def register(self, name: str, ids):
        self.seqs[f"seq:{name}"] = [int(x) for x in ids]

    def ids_for(self, prompt: str) -> list:
        ids = self.seqs[prompt] if prompt in self.seqs else [int(x) for x in self.tokenizer.encode(prompt, add_special_tokens=False)]
        return ([int(self.bos_id)] if self.bos_id is not None else []) + ids

    @property
    def bos_offset(self) -> int:
        return 0 if self.bos_id is None else 1

    def encode(self, text: str, *, max_length: int = 512):
        ids = self.ids_for(text)[:max_length]
        return self._torch.tensor([ids], dtype=self._torch.long, device=self.device)

    def forward(self, input_ids):
        torch = self._torch
        B, S = input_ids.shape
        am = torch.ones((B, S), dtype=torch.long, device=input_ids.device)
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        return self.model(input_ids=input_ids, attention_mask=am, position_ids=pos, use_cache=False, logits_to_keep=1)

    def unembed(self, residual):
        W = self.an["lm_head"].weight
        h = residual.to(W.device, W.dtype)
        return self.an["lm_head"](self.an["norm"](h))[..., :V_SHARED].float()

    def head(self):
        """(gamma fp32 [d], W_U bf16/fp32 [V_SHARED, d]) — the final norm scale and the shared-prefix unembedding."""
        return self.an["norm"].weight.detach().float(), self.an["lm_head"].weight.detach()[:V_SHARED]


# ------------------------------------------------------------------------------------- corpora
def wikitext_docs(split: str) -> list:
    """Documents of wikitext-103-raw-v1: a document starts at a top-level heading line ' = Title = '."""
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT[0], WIKITEXT[1], split=split)
    docs, cur = [], []
    for line in ds["text"]:
        if _HEADING.match(line.rstrip("\n")):
            if cur:
                docs.append("".join(cur))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        docs.append("".join(cur))
    assert len(docs) >= 50, len(docs)      # train 28 k articles; validation and test hold 60 each
    return docs


def build_corpus(tok, docs: list, n_seqs: int, seq_len: int, seed: int, min_doc_tokens: int,
                 windows_per_doc: int = 1, max_chars: int = 8000) -> dict:
    """`n_seqs` token windows of exactly `seq_len` ids (no BOS): documents shuffled by `Random(seed)`, a
    document qualifies when its first `max_chars` chars hold ≥ `min_doc_tokens` tokens; up to
    `windows_per_doc` consecutive NON-OVERLAPPING windows are taken from its start. Returns ids + (doc, window)
    indices + sha256 over the ids."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    seqs, used = [], []
    for i in order:
        ids = [int(x) for x in tok.encode(docs[i][:max_chars], add_special_tokens=False)]
        if len(ids) < min_doc_tokens:
            continue
        for w in range(windows_per_doc):
            chunk = ids[w * seq_len:(w + 1) * seq_len]
            if len(chunk) < seq_len:
                break
            seqs.append(chunk); used.append((i, w))
            if len(seqs) == n_seqs:
                break
        if len(seqs) == n_seqs:
            break
    assert len(seqs) == n_seqs, f"only {len(seqs)} of {n_seqs} windows available ({len(docs)} documents)"
    return {"seqs": seqs, "doc_window_index": used, "n_docs_used": len({i for i, _ in used}), "n_seqs": n_seqs,
            "seq_len": seq_len, "seed": seed, "min_doc_tokens": min_doc_tokens, "windows_per_doc": windows_per_doc,
            "sha256": corpus_sha256(seqs)}


def corpus_sha256(seqs) -> str:
    return hashlib.sha256(json.dumps([list(map(int, s)) for s in seqs]).encode()).hexdigest()


def fit_corpus(tok, n_seqs=FIT_SEQS, seq_len=FIT_LEN, seed=CORPUS_SEED, docs=None) -> dict:
    c = build_corpus(tok, docs if docs is not None else wikitext_docs("train"), n_seqs, seq_len, seed, 200)
    c["split"] = "train"
    return c


def trace_corpus(tok, n_seqs=TRACE_SEQS, seq_len=TRACE_LEN, seed=CORPUS_SEED, docs=None) -> dict:
    """Validation split: 60 articles, so 200 × 32-token windows are 4 consecutive non-overlapping windows per
    article from the shuffled article order (spec §15.11); every window starts at a 32-token boundary."""
    c = build_corpus(tok, docs if docs is not None else wikitext_docs("validation"), n_seqs, seq_len, seed,
                     min_doc_tokens=4 * seq_len, windows_per_doc=4, max_chars=20000)
    c["split"] = "validation"
    return c


def top_tokens(seqs, k: int = 5000) -> list:
    """Token ids of the corpus ranked by frequency (ties by id), capped at k and at V_SHARED."""
    from collections import Counter
    cnt = Counter(int(t) for s in seqs for t in s if int(t) < V_SHARED)
    return [t for t, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ------------------------------------------------------------------------------------- fitting
def fit_lens(lm: AnatomyLensModel, seqs: list, dim_batch: int, ckpt_path: str, sizes=(), save_dir=None) -> tuple:
    """jlens.fit over registered windows; intermediate lenses at `sizes` prompts (the fit-size sweep) come
    from the SAME running sum via resume, saved as `lens_<n>.pt` (fp16). Returns (lens, info)."""
    import torch
    from jlens import fit as jl_fit
    names = []
    for i, s in enumerate(seqs):
        lm.register(f"fit{i}", s); names.append(f"seq:fit{i}")
    max_len = max(len(s) for s in seqs) + lm.bos_offset
    info = {"n_seqs": len(seqs), "dim_batch": dim_batch, "max_seq_len": max_len, "stages": []}
    lens = None
    for n in sorted(set([s for s in sizes if s < len(seqs)] + [len(seqs)])):
        t0 = time.perf_counter()
        lens = jl_fit(lm, names[:n], dim_batch=dim_batch, max_seq_len=max_len, checkpoint_path=ckpt_path,
                      checkpoint_every=4, resume=True)
        dt = time.perf_counter() - t0
        info["stages"].append({"n": n, "seconds": dt, "n_prompts_fitted": lens.n_prompts})
        if save_dir is not None:
            lens.save(os.path.join(save_dir, f"lens_{n}.pt"), dtype=torch.float16)
    assert lens is not None and lens.n_prompts == len(seqs), (lens, len(seqs))
    return lens, info


def load_lens(path: str):
    from jlens import JacobianLens
    return JacobianLens.load(path)


def lens_to(lens, device):
    """Move every J_ℓ to `device` ONCE. `JacobianLens.transport` calls `.to(residual.device)` per call, which
    on a CPU-resident lens would stream 26 MB per layer per readout (538 items × 35 layers ≈ 0.5 TB over
    PCIe for one profile). Do this after `save()` (which is written from wherever the tensors live)."""
    lens.jacobians = {L: J.to(device) for L, J in lens.jacobians.items()}
    return lens


def merge_lenses(paths: list):
    from jlens import JacobianLens
    return JacobianLens.merge([JacobianLens.load(p) for p in paths])


def lens_stats(lens) -> dict:
    import torch
    out = {}
    for L in lens.source_layers:
        J = lens.jacobians[L]
        d = J.shape[0]
        out[int(L)] = {"diag_mean": float(J.diagonal().mean()), "fro_over_sqrt_d": float(J.norm() / d ** 0.5),
                       "identity_distance": float((J - torch.eye(d)).norm() / d ** 0.5), "finite": bool(torch.isfinite(J).all())}
    return out


# ------------------------------------------------------------------------------------- residual capture
def capture(lm: AnatomyLensModel, ids: list, positions=None):
    """Residuals at every block output for ONE sequence (content ids, BOS added here): tensor
    [n_layers, n_pos, d] fp32 on the model device; `positions` index CONTENT tokens (None = all)."""
    import torch
    from jlens import ActivationRecorder
    full = ([int(lm.bos_id)] if lm.bos_id is not None else []) + [int(x) for x in ids]
    x = torch.tensor([full], dtype=torch.long, device=lm.device)
    with torch.inference_mode(), ActivationRecorder(lm.layers, at=range(lm.n_layers)) as rec:
        lm.forward(x)
        acts = torch.stack([rec.activations[L][0].float() for L in range(lm.n_layers)], 0)   # [L, S, d]
    if positions is None:
        return acts
    idx = [int(p) + lm.bos_offset for p in positions]
    assert all(0 <= i < acts.shape[1] for i in idx), (idx, acts.shape)
    return acts[:, idx, :]


def lens_logits(lm: AnatomyLensModel, h, layer: int, lens=None):
    """Lens logits for residual(s) `h` [.., d] at `layer`: J_ℓ h then unembed (lens given and ℓ < last),
    plain unembed otherwise. fp32, shared prefix."""
    x = h.float()
    if lens is not None and layer < lm.n_layers - 1:
        x = lens.transport(x, layer)
    return lm.unembed(x)


def ranks_of(logits, ids: list):
    """1-based rank of each id in a 1-d logit vector (rank 1 = argmax); ties count against the id."""
    import torch
    ids_t = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
    v = logits[ids_t]
    return 1 + (logits.unsqueeze(0) > v.unsqueeze(1)).sum(1)


def item_acts(lm: AnatomyLensModel, items: list) -> dict:
    """{hash: [n_layers, d] fp16 CPU} at each item's readout position."""
    import torch
    out = {}
    for it in items:
        ids = lm.tokenizer.encode(it["prompt"], add_special_tokens=False)
        a = capture(lm, ids, positions=[it["readout_idx"]])[:, 0, :]
        out[it["hash"]] = a.to(torch.float16).cpu()
    return out


def rank_profile(lm: AnatomyLensModel, items: list, acts: dict, lens=None, partner=None) -> list:
    """Per item, per scored intermediate, the rank at every layer (min over the intermediate's ids).
    `partner` {hash: hash} scores item i's ids on item partner[i]'s activations (the shuffle control).
    A reading under ANOTHER checkpoint's head is done by transplanting that head into the resident model
    (`stagediff_interv.transplant_head`) around this call — exact by construction, no second formula."""
    import torch
    out = []
    for it in items:
        src = acts[partner[it["hash"]] if partner else it["hash"]].to(lm.device).float()   # [L, d]
        rows = []
        per_layer = []
        for L in range(lm.n_layers):
            x = src[L]
            if lens is not None and L < lm.n_layers - 1:
                x = lens.transport(x, L)
            per_layer.append(lm.unembed(x))
        lg = torch.stack(per_layer, 0)                                                     # [L, V]
        for inter in it["intermediates_scored"]:
            ids_t = torch.as_tensor(inter["ids"], device=lg.device)
            v = lg[:, ids_t]                                                               # [L, n_ids]
            r = 1 + (lg.unsqueeze(2) > v.unsqueeze(1)).sum(1)                              # [L, n_ids]
            rows.append(r.min(1).values.tolist())
        out.append({"hash": it["hash"], "set": it["set"], "split": it["split"], "ranks": rows})
    return out


def pass_at_k(profile: list, band=BAND, k: int = 10, split=None) -> dict:
    band = list(band)
    per_item, per_set = {}, {}
    for p in profile:
        if split and p["split"] != split:
            continue
        hits = [min(r[L] for L in band) <= k for r in p["ranks"]]
        frac = sum(hits) / len(hits)
        per_item[p["hash"]] = frac
        per_set.setdefault(p["set"], []).append(frac)
    n = len(per_item)
    return {"k": k, "band": [band[0], band[-1]], "n_items": n,
            "overall": (sum(per_item.values()) / n) if n else None,
            "per_set": {s: sum(v) / len(v) for s, v in per_set.items()}, "per_item": per_item}


def layer_curve(profile: list, k: int = 10, split=None) -> list:
    """Fraction of intermediates with rank ≤ k at each layer (the F2 profile)."""
    n_layers = len(profile[0]["ranks"][0])
    tot, hit = 0, [0] * n_layers
    for p in profile:
        if split and p["split"] != split:
            continue
        for r in p["ranks"]:
            tot += 1
            for L in range(n_layers):
                hit[L] += r[L] <= k
    return [h / tot for h in hit] if tot else []


# ------------------------------------------------------------------------------------- ROT (lens vectors)
def token_vectors(lens, gamma, W, layer: int, token_ids: list):
    """Rows: the direction at `layer` the lens reads as token t: (γ ⊙ W_U[t]) J_ℓ  (= J_ℓᵀ g_t as a column);
    at the last layer (the one layer above the fitted sources) the direction is g_t itself. fp32 on W's device."""
    import torch
    ids_t = torch.as_tensor(token_ids, device=W.device, dtype=torch.long)
    g = (W[ids_t].float() * gamma.to(W.device).float())
    if layer not in lens.jacobians:
        assert layer == max(lens.source_layers) + 1, (layer, lens.source_layers)
        return g
    return g @ lens.jacobians[layer].to(W.device)


def rot_by_layer(lens_p, lens_c, head_p, head_c, token_ids: list, layers=range(N_LAYERS)) -> dict:
    """Median cosine per layer between parent and child lens vectors of the same token, under each model's
    own head ('own_heads') and under the PARENT head for both ('parent_head'); token bootstrap CIs."""
    import torch
    from infra.pg.stagediff_stats import boot_median
    gp, Wp = head_p
    gc, Wc = head_c
    out = {}
    for L in layers:
        vp = token_vectors(lens_p, gp, Wp, L, token_ids)
        vc_own = token_vectors(lens_c, gc, Wc, L, token_ids)
        vc_ph = token_vectors(lens_c, gp, Wp.to(vc_own.device), L, token_ids)
        cos_own = torch.nn.functional.cosine_similarity(vp, vc_own.to(vp.device), dim=1).cpu().numpy()
        cos_ph = torch.nn.functional.cosine_similarity(vp, vc_ph.to(vp.device), dim=1).cpu().numpy()
        out[int(L)] = {"own_heads": boot_median(cos_own), "parent_head": boot_median(cos_ph)}
    return out


# ------------------------------------------------------------------------------------- TRACE readout
def topk_ids(logits, k: int = 25) -> list:
    return logits.topk(k).indices.tolist()


def relevance(ids: list, robot_ids: set) -> float:
    return sum(1 for i in ids if i in robot_ids) / len(ids)


def trace_readout(lm: AnatomyLensModel, delta, robot_ids: set, band=BAND, lens=None, k: int = 25) -> dict:
    """`delta` [n_layers, n_pos, d] fp32 → relevance grid over (band layer, position) and the best cell,
    with its top-k tokens decoded."""
    import torch
    grid, best = {}, None
    for L in band:
        for j in range(delta.shape[1]):
            lg = lens_logits(lm, delta[L, j].to(lm.device), L, lens)
            ids = topk_ids(lg, k)
            rel = relevance(ids, robot_ids)
            grid[f"{L},{j}"] = rel
            if best is None or rel > best["relevance"]:
                best = {"layer": int(L), "position": int(j), "relevance": rel, "top_ids": ids,
                        "top_tokens": [lm.tokenizer.decode([i]) for i in ids]}
    return {"grid": grid, "best": best, "k": k}


# ------------------------------------------------------------------------------------- dry-run
def _dryrun():
    import torch
    from infra.pg.stagediff_models import stub_member
    from infra.pg import stagediff_items as SI
    m = stub_member(bos_id=1)
    lm = AnatomyLensModel(m)
    assert lm.n_layers == 4 and lm.d_model == 16 and lm.bos_offset == 1
    docs = [f" = Doc {i} = \n" + ("word " * 300) + f"x{i}\n" for i in range(12)]
    fc = fit_corpus(lm.tokenizer, n_seqs=3, seq_len=40, seed=0, docs=docs)
    tc = trace_corpus(lm.tokenizer, n_seqs=4, seq_len=24, seed=0, docs=docs)
    assert all(len(s) == 40 for s in fc["seqs"]) and all(len(s) == 24 for s in tc["seqs"])
    assert fc["sha256"] == corpus_sha256(fc["seqs"]) and len(fc["doc_window_index"]) == 3 and fc["n_docs_used"] == 3
    tc2 = trace_corpus(lm.tokenizer, n_seqs=8, seq_len=24, seed=0, docs=docs[:3])          # 3 docs → 4 windows each
    assert tc2["n_docs_used"] == 2 and tc2["doc_window_index"][:4] == [(tc2["doc_window_index"][0][0], w) for w in range(4)]
    assert tc2["seqs"][0] != tc2["seqs"][1]                                                  # consecutive, not repeated
    assert fit_corpus(lm.tokenizer, n_seqs=3, seq_len=40, seed=0, docs=docs)["sha256"] == fc["sha256"]   # deterministic
    tt = top_tokens(fc["seqs"], k=5)
    assert len(tt) == 5
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        lens, info = fit_lens(lm, fc["seqs"], dim_batch=4, ckpt_path=os.path.join(d, "ck.pt"), sizes=(2,), save_dir=d)
        assert lens.n_prompts == 3 and [s["n"] for s in info["stages"]] == [2, 3]
        assert os.path.exists(os.path.join(d, "lens_2.pt")) and os.path.exists(os.path.join(d, "lens_3.pt"))
        st = lens_stats(lens)
        assert set(st) == {0, 1, 2} and all(v["finite"] for v in st.values())
        l2 = load_lens(os.path.join(d, "lens_3.pt"))
        assert l2.source_layers == [0, 1, 2]
        mg = merge_lenses([os.path.join(d, "lens_2.pt"), os.path.join(d, "lens_3.pt")])
        assert mg.n_prompts == 5
    # capture + lens logits + ranks
    ids = lm.tokenizer.encode("Q: capital of France? Answer:", add_special_tokens=False)
    acts = capture(lm, ids)
    assert acts.shape == (4, len(ids) + 1, 16)
    last = capture(lm, ids, positions=[len(ids) - 1])
    assert torch.allclose(last[:, 0], acts[:, -1])
    lg_model = lm.member["fwd"].next_token_logits(["Q: capital of France? Answer:"])[0]
    lg_lens = lens_logits(lm, acts[3, -1], 3, lens)        # last layer == identity == the model's own logits
    assert torch.allclose(lg_model, lg_lens, atol=1e-4), (lg_model - lg_lens).abs().max()
    r = ranks_of(lg_lens, [int(lg_lens.argmax()), int(lg_lens.argmin())])
    assert r[0].item() == 1 and r[1].item() == lg_lens.numel()
    # profiles on the real item builder with the stub tokenizer
    sets = SI.load_sets()
    items = SI.smoke_subset(SI.int_items(lm.tokenizer, sets), 2)
    A = item_acts(lm, items)
    prof = rank_profile(lm, items, A, lens=lens)
    prof_ll = rank_profile(lm, items, A, lens=None)
    assert len(prof) == len(items) and all(len(r) == 4 for p in prof for r in p["ranks"])
    assert [p["ranks"][0][3] for p in prof] == [p["ranks"][0][3] for p in prof_ll]      # identity at the last layer
    pk = pass_at_k(prof, band=(1, 2), k=256)
    assert pk["overall"] == 1.0 and pass_at_k(prof, band=(1, 2), k=0)["overall"] == 0.0
    partner = SI.derange_within_set(items)
    sh = rank_profile(lm, items, A, lens=lens, partner=partner)
    assert len(sh) == len(prof)
    g, W = lm.head()
    curve = layer_curve(prof, k=256)
    assert curve == [1.0] * 4
    # ROT: a lens against itself is cosine 1 under both heads
    rot = rot_by_layer(lens, lens, (g, W), (g, W), tt, layers=range(4))
    assert all(abs(v["own_heads"]["point"] - 1) < 1e-5 and abs(v["parent_head"]["point"] - 1) < 1e-5 for v in rot.values())
    # TRACE readout on a random delta
    delta = torch.randn(4, 5, 16)
    tr = trace_readout(lm, delta, set(range(0, 256, 2)), band=(1, 2), lens=lens, k=10)
    assert len(tr["grid"]) == 10 and 0 <= tr["best"]["relevance"] <= 1 and len(tr["best"]["top_tokens"]) == 10
    print("PASS stagediff_lens dry-run (LensModel over anatomy, corpora, jlens fit + sweep + merge, capture, ranks, profiles, shuffle, ROT, TRACE)")


if __name__ == "__main__":
    _dryrun()
