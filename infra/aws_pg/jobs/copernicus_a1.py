"""CO-A1 — does the sink carry load-bearing cargo, or is it a parking lot? (queue items 27 + 87)

Spec §2.1 / §6.1 / §9.3. Consumes the hashed grid banks published by `copernicus_a1_banks.py`.

THE OPERATOR is the banked FUNC instrument: a rank-k subspace projected OUT of the q/k INPUT at
every layer, x' = x − QQᵀx. That ablates attention ROUTING only — v_proj, o_proj and the MLP read
the unmodified stream — so every sentence this job writes carries that site scope, and one SCOPE
cell at the decisive k repeats the projection on the FULL residual stream as the honest
"load-bearing" test.

THE ADJUDICATED SET is {core, anti, topenergy}. topsigma and bottomsigma are INSTRUMENT CONTROLS
with a dynamic-range gate and a floor gate; they never appear in a verdict clause. That split is
the round-1 fix: with topsigma as a verdict rival, H-P1 was unconfirmable by construction.

WHY EQUAL-k IS NOT ENOUGH. core-128 already holds ~28% of the input activation energy (recomputed
from the anchor at run time, never quoted) and topenergy is the width-wise energy MAXIMISER, so an
equal-k comparison very nearly forces the traffic reading arithmetically. The adjudicating cells
are therefore the equal-ENERGY contrast (comparator width k_b matched on removed D-half energy)
and the set differences core⊖topenergy / topenergy⊖core.

BLINDING is a discovery/confirmation construction, stated honestly: bands and the m_sup/m_eq
scales are calibrated on the D half, frozen to thresholds.json with a content hash, and only then
is the C half computed — once — by the scoring pass that asserts that hash.

    pg run infra.aws_pg.jobs.copernicus_a1 \
        --args "--bank-s3 s3://.../results/copernicus_a1_banks/out --manifest-sha256 <sha>" \
        --no-weights --type g6e.8xlarge,p5.4xlarge \
        --s3-prefix results/copernicus_a1 --budget-min 480
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import conserved_core_probes as P

from infra.aws_pg.jobs.invariant_core import PG1_REPO, N_LAYERS
from infra.aws_pg.jobs.invariant_core_func import (
    _blank_image, _decoder_layers, _load_full_pg1, _locate_text_block, _processor_text_ids,
)

ARM = "CO-A1"
QUEUE_ITEM = "27"
WAVE = "W0"
N_IN = 2048
DECISIVE_K = 128                     # §2.1 — kept at 128 for anchor continuity
SMALL_K_GATE = 8                     # §6.1.7 — the one SCORED small-k gate
N_RAND_DRAWS = 5                     # §2.1 — nested, >=5
N_EM_DRAWS = 3                       # §2.1 — energy-matched, >=3
N_TWIN_REF = 5                       # §6.1.2 — same-construction reference draws per twin
SPLIT_SALT = "copernicus-a1-2026-08-17"          # frozen at authoring; the D/C key
ANCHOR_S3 = "s3://pg-lineage-769063704696/results/invcore_func/out/invariant_core_func.json"


def is_article_marker(row: str) -> bool:
    """§2.1's splitter rule, corrected PRE-FREEZE (journaled; see §15 v0.7).

    The spec froze the marker as the regex `^ = .+ = $`. That regex also matches wikitext SECTION
    headings — ' = = Gameplay = = ' satisfies it, because `.+` happily swallows '= Gameplay ='.
    Blocks would then have been sections, not articles, and two consecutive sections of the SAME
    article could land on opposite sides of the D/C firewall while sharing topic and vocabulary:
    precisely the adjacent-context leak the block split exists to prevent (§2.1 rejects a
    window-level split for the same reason).

    Articles carry one '=' per side, sections two or more, so the corrected rule keeps the frozen
    INTENT — article granularity — and drops the unintended matches. Rows are stripped first
    because wikitext rows retain their trailing newline through the anchor's `t.strip()` filter.
    """
    s = row.strip()
    return s.startswith("= ") and s.endswith(" =") and not s.startswith("= = ")


# ===========================================================================
# Corpus: byte-identical windowing to the anchor + the frozen D/C block split.
# ===========================================================================
def build_corpus(processor, window: int):
    """The anchor's wikitext construction, byte-identical, plus article-block ids per window.

    §2.1: the article marker is a RAW row (see `is_article_marker`) tested BEFORE the join; a
    block is the token span between consecutive markers; a window that straddles a boundary
    belongs to the block of its FIRST token. Splitting at WINDOW level would leak adjacent
    context across the D/C firewall, and there are no document ids in this instrument to split
    on instead.
    """
    from datasets import load_dataset
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    rows = [t for t in ds["text"] if t.strip()]          # the anchor's filter, unchanged
    text = "\n\n".join(rows)                             # the anchor's join, unchanged

    # char offset where each row starts in the joined text, and the marker rows among them
    starts, pos = [], 0
    for r in rows:
        starts.append(pos)
        pos += len(r) + 2                                # the "\n\n" separator
    block_starts = [starts[i] for i, r in enumerate(rows) if is_article_marker(r)]
    assert len(block_starts) >= 20, f"only {len(block_starts)} article markers — splitter is wrong"

    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids[0]
    offsets = enc.offset_mapping[0].tolist()
    assert len(offsets) == ids.shape[0], "tokenizer did not return one offset per token"

    windows = []
    for start in range(0, ids.shape[0], window):
        chunk = ids[start:start + window]
        if chunk.shape[0] < 8:
            break
        wtext = tok.decode(chunk.tolist())
        if len(tok(wtext, add_special_tokens=False).input_ids) < 8:
            continue
        first_char = offsets[start][0]
        blk = int(np.searchsorted(block_starts, first_char, side="right") - 1)
        windows.append({"idx": len(windows), "text": wtext, "block": blk, "tok_start": int(start)})
    return windows, {"n_rows": len(rows), "n_chars": len(text), "n_tokens": int(ids.shape[0]),
                     "n_blocks": len(block_starts), "n_windows": len(windows)}


def split_dc(windows, salt: str):
    """Salted-hash split of contiguous window BLOCKS. Thresholds are set on D; the headline is
    computed once on C."""
    D, Cc = [], []
    for w in windows:
        h = hashlib.sha256(f"{salt}|{w['block']}".encode()).hexdigest()
        (D if int(h[:8], 16) % 2 == 0 else Cc).append(w)
    return D, Cc


# ===========================================================================
# Readouts. One projection hook family, one ppl routine, used by every arm.
# ===========================================================================
class _InputProject:
    """forward_pre_hook on q_proj AND k_proj of every layer: x' = x − Q(Qᵀx).

    THE PROJECTION IS COMPUTED IN fp32 and cast back to the model dtype. The anchor built Q in the
    MODEL's dtype (bf16) and projected in bf16; with k=128 that leaves an orthogonality residue
    large enough to add damage of its own, which lands squarely on the two gates that read small
    numbers — the bottomsigma FLOOR and the set-difference cells. The preflight measures the
    bf16-vs-fp32 gap and banks it rather than assuming it away.
    """

    def __init__(self, layers, Q_by_layer, device, sites=("q_proj", "k_proj")):
        import torch
        self.layers = layers
        self.sites = sites
        self.Q = {L: torch.as_tensor(np.ascontiguousarray(Q_by_layer[L]), device=device,
                                     dtype=torch.float32) for L in Q_by_layer}
        self._handles = []

    def _mk(self, L):
        Q = self.Q[L]

        def pre_hook(_m, args):
            x = args[0]
            x32 = x.float()
            x2 = x32 - (x32 @ Q) @ Q.transpose(0, 1)
            return (x2.to(x.dtype),) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            if L not in self.Q:
                continue
            for site in self.sites:
                self._handles.append(
                    getattr(layer.self_attn, site).register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class _ResidualProject(_InputProject):
    """The SCOPE cell (§2.1): the same projection on the WHOLE residual stream entering the
    decoder layer, so v_proj/o_proj/MLP see it too. This is the honest 'load-bearing' test; the
    q/k cell alone can only speak about routing."""

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            if L not in self.Q:
                continue
            self._handles.append(layer.register_forward_pre_hook(self._mk(L)))
        return self


class _InputCapture:
    """Accumulate the fp64 Gram of the q_proj input per layer. Used ONLY on the D half."""

    def __init__(self, layers):
        self.layers = layers
        self.gram = {L: None for L in range(len(layers))}
        self.n_tok = {L: 0 for L in range(len(layers))}
        self._handles = []

    def _mk(self, L):
        def pre_hook(_m, args):
            flat = args[0].reshape(-1, args[0].shape[-1]).double()
            g = (flat.transpose(0, 1) @ flat).cpu().numpy()
            self.gram[L] = g if self.gram[L] is None else self.gram[L] + g
            self.n_tok[L] += flat.shape[0]
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    __exit__ = _InputProject.__exit__


def encode_windows(processor, device, windows, legacy_tokenization=False):
    """Run the processor ONCE per window and cache exactly what the forward needs.

    This job scores ~190 arm-passes over the same windows. Calling the processor inside the scoring
    loop would re-tokenise and re-run SigLIP's image preprocessing for every window on every pass —
    tens of thousands of redundant CPU-side preprocessings on an 8-vCPU box, where they, not the
    GPU, would set the runtime. The blank image is IDENTICAL for every window, so its pixel tensor
    is built once and shared.

    `legacy_tokenization=True` reproduces the anchor's exact target selection (tokenise the window
    WITHOUT the processor's appended newline). That path raises inside `_locate_text_block`
    whenever the tokenizer merges the newline into the final token — the anchor's 120 windows
    happened not to trigger it, and this job scores more than twice as many. The default path uses
    `_processor_text_ids`, which is correct either way; the preflight scores both so the difference
    is a measured systematic rather than an assumption.
    """
    import torch
    tok = processor.tokenizer
    blank = _blank_image()
    enc = []
    pixel = None
    for w in windows:
        wtext = w["text"]
        win_ids = (tok(wtext, return_tensors="pt", add_special_tokens=False).input_ids[0]
                   if legacy_tokenization else _processor_text_ids(tok, wtext))
        n_text = int(win_ids.shape[0])
        assert n_text >= 8, f"window {w['idx']} tokenised to {n_text} tokens"
        inp = processor(text=[wtext], images=[blank], return_tensors="pt", padding=False)
        in_ids = inp["input_ids"][0]
        assert in_ids.shape[0] > n_text, "text span >= prompt — image prefix missing"
        pos = _locate_text_block(in_ids, win_ids)
        if pixel is None:
            pixel = inp["pixel_values"].to(device).to(torch.bfloat16)
        enc.append({"input_ids": inp["input_ids"].to(device),
                    "attention_mask": inp["attention_mask"].to(device),
                    "pos": pos, "n_text": n_text,
                    "targets": win_ids[1:].to(device), "idx": w["idx"]})
    assert enc, "no windows encoded"
    return {"rows": enc, "pixel_values": pixel}


def ppl_encoded(model, enc) -> np.ndarray:
    """Per-window [nll_sum, n_tok] from a cached encoding. Forward only — no CPU preprocessing."""
    import torch
    per_window = []
    pixel = enc["pixel_values"]
    for r in enc["rows"]:
        with torch.inference_mode():
            logits = model(input_ids=r["input_ids"], attention_mask=r["attention_mask"],
                           pixel_values=pixel).logits[0]
        text_logits = logits[r["pos"]:r["pos"] + r["n_text"] - 1, :].float()
        assert text_logits.shape[0] == r["targets"].shape[0], (text_logits.shape, r["targets"].shape)
        loss = torch.nn.functional.cross_entropy(text_logits, r["targets"], reduction="sum")
        per_window.append([float(loss.item()), int(r["targets"].numel())])
    a = np.asarray(per_window, dtype=np.float64)
    assert a.shape[0] == len(enc["rows"]) and a[:, 1].sum() > 0
    return a


def ppl_windows(model, processor, device, windows, legacy_tokenization=False) -> np.ndarray:
    """Encode-then-score in one call. Used where a window set is scored ONCE (the preflight legs
    and CO-A2's final ppl); the battery uses `encode_windows` + `ppl_encoded` instead."""
    return ppl_encoded(model, encode_windows(processor, device, windows, legacy_tokenization))


# ===========================================================================
# Banks
# ===========================================================================
def fetch_banks(bank_s3: str, dest: pathlib.Path):
    """M-D99: the consumer takes the producer's prefix as an ARGUMENT and asserts on an empty
    dependency — a default that silently resolves to a stale prefix is how a run scores last
    month's banks."""
    assert bank_s3.startswith("s3://"), f"--bank-s3 must be an s3 uri, got {bank_s3!r}"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", bank_s3, str(dest), "--region", "us-east-1"], check=True)
    npz, man = dest / "a1_banks.npz", dest / "a1_banks_manifest.json"
    assert npz.exists() and man.exists(), f"bank dependency empty at {bank_s3} (looked in {dest})"
    meta = json.loads(man.read_text())
    z = np.load(npz)
    banks = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    for row in meta["banks"]:
        key = f"{row['bank']}/{row['k']}/{row['layer']}"
        assert key in banks, f"manifest names {key}, npz does not carry it"
        got = C.bank_hash(banks[key])
        assert got == row["sha256"], f"{key}: bank hash {got[:12]} != manifest {row['sha256'][:12]}"
    print(f"[a1] banks: {len(banks)} arrays, all hashes verified against the producer manifest",
          flush=True)
    return banks, meta


def by_layer(banks, bank: str, k: int, layers):
    return {L: banks[f"{bank}/{k}/{L}"] for L in layers}


def act_energy(gram: np.ndarray, Q: np.ndarray) -> float:
    """Fraction of activation energy inside span(Q): trace(Qᵀ G Q)/trace(G)."""
    tot = float(np.trace(gram))
    assert tot > 0, "activation Gram has zero trace"
    return float(np.trace(Q.T @ gram @ Q) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-s3", required=True, help="producer prefix from copernicus_a1_banks")
    ap.add_argument("--anchor-s3", default=ANCHOR_S3)
    ap.add_argument("--k-grid", default="2,4,8,16,32,64,128")
    ap.add_argument("--n-ppl-seq", type=int, default=340,
                    help="windows scanned; D and C each need >=120 after the block split")
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--n-act", type=int, default=120, help="D-half windows for the ACT Gram")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = sorted(int(x) for x in a.k_grid.split(","))
    layers = list(range(N_LAYERS))
    if a.smoke:
        k_grid = [8, 128]
        layers = [0, 9, 17]
        a.n_ppl_seq, a.n_act = 24, 8

    H.hf_setup()
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required for the pt-224 forward"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"

    banks, bmeta = fetch_banks(a.bank_s3, pathlib.Path(H.cache_dir(out, "a1_banks")))
    assert set(k_grid) <= set(bmeta["k_grid"]), f"banks lack k in {set(k_grid) - set(bmeta['k_grid'])}"
    base_sha = bmeta["base_sha"]

    anchor = json.loads(subprocess.run(
        ["aws", "s3", "cp", a.anchor_s3, "-", "--region", "us-east-1"],
        check=True, capture_output=True, text=True).stdout)
    anchor_clean = float(anchor["capability_baseline"]["ppl"])
    anchor_k128 = next(r for r in anchor["ablation"] if r["k"] == DECISIVE_K)
    anchor_tol = 3.0 * float(anchor_k128["contrast"]["ci_unt_minus_rand"]["half_width"]) \
        if "half_width" in anchor_k128["contrast"]["ci_unt_minus_rand"] else \
        3.0 * (anchor_k128["contrast"]["ci_unt_minus_rand"]["hi"]
               - anchor_k128["contrast"]["ci_unt_minus_rand"]["lo"]) / 2.0

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "primary", "director §11: TRAFFIC at the decisive k=128 on the equal-ENERGY "
            "contrast (topenergy >= core); the small-k residual fires", confidence="low-mid",
            basis="spec §11 row CO-A1", family="payload")
    exp.log(ARM, "small_k", "core-minus-topenergy at k=8 exits its random band by > m_sup(8)",
            confidence="low-mid", basis="spec §6.1.7", family="payload")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_a1", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "primary"))
    gates = C.Gates(out)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, device)
    mlayers = _decoder_layers(model)

    rep = {"object": "copernicus_a1_v1", "arm": ARM, "program": C.PROGRAM, "queue_item": QUEUE_ITEM,
           "base_repo": PG1_REPO, "base_sha": base_sha, "bank_s3": a.bank_s3,
           "manifest_sha256": bmeta["manifest_sha256"], "k_grid": k_grid, "layers": layers,
           "decisive_k": DECISIVE_K, "seed": a.seed, "smoke": a.smoke,
           "site_scope": "q/k read-path input at every layer (attention ROUTING only)",
           "provenance": C.provenance(), "utc": C.utc_now(),
           "anchor": {"s3": a.anchor_s3, "clean_ppl": anchor_clean,
                      "dppl_core_k128": anchor_k128["contrast"]["dppl_unt"],
                      "tolerance_ci_x3": anchor_tol}}

    def save():
        # default=float so a stray numpy scalar cannot kill a multi-hour run at its final
        # write; every value in this report is genuinely numeric.
        (out / "copernicus_a1.json").write_text(
            json.dumps(rep, indent=2, sort_keys=True, default=float))

    # ---- corpus + D/C -------------------------------------------------------------------
    windows, cinfo = build_corpus(processor, a.ppl_window)
    windows = windows[:a.n_ppl_seq]
    Dw, Cw = split_dc(windows, SPLIT_SALT)
    rep["corpus"] = {**cinfo, "scanned": len(windows), "n_D": len(Dw), "n_C": len(Cw),
                     "salt": SPLIT_SALT, "window": a.ppl_window,
                     "splitter": "article-block; marker = a raw pre-join row that is '= X =' with a SINGLE '=' per side (section headings excluded, §15 v0.7); window -> block of its first token"}
    print(f"[a1] corpus: {len(windows)} windows -> D={len(Dw)} C={len(Cw)} "
          f"over {cinfo['n_blocks']} article blocks", flush=True)
    if not a.smoke:
        assert len(Dw) >= 120 and len(Cw) >= 120, (
            f"D={len(Dw)} C={len(Cw)}: §2.1 requires >=120 windows per half — raise --n-ppl-seq")
    save()

    # ---- PREFLIGHT A (§6.1.1) ----------------------------------------------------------
    Qt = torch.as_tensor(banks[f"core/{DECISIVE_K}/{layers[0]}"], device=device, dtype=torch.float32)
    xr = torch.randn(3, 5, N_IN, device=device, dtype=torch.float32)
    x1 = xr - (xr @ Qt) @ Qt.T
    x2 = x1 - (x1 @ Qt) @ Qt.T
    proj_resid = float((x2 - x1).abs().max())
    inside = float((x1 @ Qt).abs().max())
    xb = xr.to(torch.bfloat16)
    Qb = Qt.to(torch.bfloat16)
    inside_bf16 = float(((xb - (xb @ Qb) @ Qb.T).float() @ Qt).abs().max())
    assert proj_resid < 1e-4 and inside < 1e-4, f"projector not clean: {proj_resid} {inside}"
    print(f"[a1][PRE] projector fp32 resid={proj_resid:.2e} inside={inside:.2e} | "
          f"bf16 inside={inside_bf16:.2e}", flush=True)

    anchor_windows = windows[:120]
    pw_clean_legacy = ppl_windows(model, processor, device, anchor_windows, legacy_tokenization=True)
    pw_clean_fixed = ppl_windows(model, processor, device, anchor_windows)
    core_dec = by_layer(banks, "core", DECISIVE_K, layers)
    with _InputProject(mlayers, core_dec, device):
        pw_core_legacy = ppl_windows(model, processor, device, anchor_windows, legacy_tokenization=True)
    clean_legacy, clean_fixed = C.ppl(pw_clean_legacy), C.ppl(pw_clean_fixed)
    dppl_core = C.ppl(pw_core_legacy) - clean_legacy
    d_clean = abs(clean_legacy - anchor_clean)
    d_dmg = abs(dppl_core - float(anchor_k128["contrast"]["dppl_unt"]))
    repro = bool(d_clean <= 3.0 * 0.05 * anchor_clean and d_dmg <= anchor_tol)
    rep["preflight"] = {
        "projector_fp32_resid": proj_resid, "projector_fp32_inside": inside,
        "projector_bf16_inside": inside_bf16,
        "clean_ppl_legacy_tok": clean_legacy, "clean_ppl_fixed_tok": clean_fixed,
        "tokenization_systematic_dppl": clean_fixed - clean_legacy,
        "anchor_clean_ppl": anchor_clean, "delta_clean": d_clean,
        "dppl_core_k128": dppl_core, "anchor_dppl_core_k128": anchor_k128["contrast"]["dppl_unt"],
        "delta_damage": d_dmg, "tolerance": anchor_tol, "reproduced": repro,
        "note": ("clean ppl and the core-k128 damage leg are reproduced in the ANCHOR's exact "
                 "configuration (legacy tokenization, its 120 windows). The battery below runs the "
                 "corrected tokenization; the gap between the two is banked above, not assumed away.")}
    print(f"[a1][PRE] clean={clean_legacy:.4f} (anchor {anchor_clean:.4f}, |d|={d_clean:.4f}) "
          f"dppl_core_k128={dppl_core:+.3f} (anchor {anchor_k128['contrast']['dppl_unt']:+.3f}, "
          f"|d|={d_dmg:.3f} tol={anchor_tol:.3f}) -> reproduced={repro}", flush=True)
    save()
    gates.record("a1_anchor_reproduction", repro, delta_clean=d_clean, delta_damage=d_dmg,
                 tolerance=anchor_tol, clean_ppl=clean_legacy, anchor_clean_ppl=anchor_clean,
                 band="anchor bootstrap CI x3 on the damage leg; 15% on clean ppl")
    if not (repro or a.smoke):
        rep["verdict"] = C.VOID_INSTRUMENT
        rep["verdict_reason"] = ("§6.1.1 preflight: the anchor configuration did not reproduce. A "
                                 "damage mismatch is operator drift, NEVER an H-P0 verdict.")
        save()
        at.close()
        print(f"COPERNICUS_A1_OK verdict={C.VOID_INSTRUMENT} (preflight)", flush=True)
        return

    # ---- ACT pass on the D half (§2.1 re-point) ----------------------------------------
    cap = _InputCapture(mlayers)
    with cap, torch.inference_mode():
        blank = _blank_image()
        for w in Dw[:a.n_act]:
            inp = processor(text=[w["text"]], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
    gram = {L: cap.gram[L] for L in layers}
    for L in layers:
        assert gram[L] is not None and cap.n_tok[L] > 0, f"no activation captured at L{L}"
    gains = {L: 1.0 + mlayers[L].input_layernorm.weight.detach().float().cpu().numpy()
             for L in layers}
    rep["act"] = {"corpus": "wikitext D-half (the READOUT corpus, §2.1)", "n_windows": len(Dw[:a.n_act]),
                  "n_tok": {str(L): int(cap.n_tok[L]) for L in layers}}

    # topenergy + its γ-corrected twin, D-half only, hashed before any contrast
    topen, topen_g = {}, {}
    for k in k_grid:
        topen[k] = {L: C.topenergy_bank(gram[L], k)[0] for L in layers}
        topen_g[k] = {L: C.topenergy_bank(gram[L], k, gains=gains[L])[0] for L in layers}
    rep["topenergy_hashes"] = {str(k): {str(L): C.bank_hash(topen[k][L]) for L in layers}
                               for k in k_grid}

    # PUBLISHED FOR CONSUMPTION, not just hashed (§13 propagation note (b); director request
    # 2026-08-17). SHACKLETON's V1(a) cells must CONSUME this artifact and never rebuild it: a
    # rebuilt topenergy bank would be a different object unless the consumer reproduced this exact
    # corpus half, this exact ACT Gram and this exact γ convention, and "same name, two objects" is
    # the f_perp lesson the program already paid for once.
    np.savez_compressed(out / "topenergy_banks.npz",
                        **{f"topenergy/{k}/{L}": topen[k][L] for k in k_grid for L in layers},
                        **{f"topenergy_gamma/{k}/{L}": topen_g[k][L] for k in k_grid for L in layers})
    topen_manifest = {
        "object": "copernicus_topenergy_banks_v1", "arm": ARM, "program": C.PROGRAM,
        "base_repo": PG1_REPO, "base_sha": base_sha, "k_grid": k_grid, "layers": layers,
        "site": "q_proj input (post-input_layernorm residual)",
        "construction": ("one-hot top-k coordinates by activation energy, i.e. the diagonal of the "
                         "ACT Gram at the q_proj input; the mercator_e2 construction"),
        "corpus": ("READOUT corpus wikitext-2 test, D HALF ONLY — an activation-derived bank built "
                   "on the full corpus would leak the confirmation half"),
        "corpus_detail": {**rep["corpus"], "n_act_windows": len(Dw[:a.n_act])},
        "gamma_convention": ("gamma = 1 + norm.weight (Gemma family). The RAW bank is the one the "
                            "verdicts read; the gamma-corrected twin is banked beside it so a "
                            "consumer's interpretation does not ride on CO-F1's outcome"),
        "hashes": {"topenergy": rep["topenergy_hashes"],
                   "topenergy_gamma": {str(k): {str(L): C.bank_hash(topen_g[k][L]) for L in layers}
                                       for k in k_grid}},
        "consumption_contract": ("consume these arrays by hash; do NOT regenerate. A regenerated "
                                 "bank is only the same object if the corpus half, the ACT Gram "
                                 "and the gamma convention all match exactly"),
        "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "topenergy_manifest.json").write_text(json.dumps(topen_manifest, indent=2, default=float))
    print(f"[a1] topenergy banks PUBLISHED: {len(k_grid) * len(layers) * 2} arrays "
          f"(raw + gamma twin) + manifest with hashes", flush=True)
    save()

    # ---- the arm table -----------------------------------------------------------------
    # ONE Haar draw per (seed, layer), sliced to every k — the nested null the nested banks need.
    rand_draws = {}
    for d in range(N_RAND_DRAWS):
        full = {L: C.nested_random_subspaces(N_IN, max(k_grid), a.seed * 7919 + 1000 * d + L)
                for L in layers}
        for k in k_grid:
            rand_draws.setdefault(k, []).append({L: np.ascontiguousarray(full[L][:, :k])
                                                 for L in layers})

    def energy_frac(Qby):
        # Over the ARM's OWN layers, not all 18: the layer-group cells are defined on 6, and
        # averaging over layers they do not define would raise rather than mislead — but only at
        # run time, hours in.
        return float(np.mean([act_energy(gram[L], Qby[L]) for L in Qby]))

    # energy-matched random draws at the decisive k (conserved_core precedent)
    eig = {L: np.linalg.eigh(gram[L]) for L in layers}
    em_draws = []
    for d in range(N_EM_DRAWS):
        cell = {}
        for L in layers:
            w, V = eig[L]
            Q, _ = P.energy_matched_subspace(V[:, ::-1], w[::-1],
                                             act_energy(gram[L], banks[f"core/{DECISIVE_K}/{L}"]),
                                             DECISIVE_K, seed=a.seed * 104729 + 31 * d + L)
            cell[L] = Q
        em_draws.append(cell)

    # permutation twins + their OWN same-construction reference draws (§6.1.2)
    twin_core = {L: C.coordinate_permutation_twin(banks[f"core/{DECISIVE_K}/{L}"],
                                                  a.seed + L)[0] for L in layers}
    twin_topen = {L: C.coordinate_permutation_twin(topen[DECISIVE_K][L], a.seed + 7 + L)[0]
                  for L in layers}
    twin_ref_dense = [{L: C.coordinate_permutation_twin(banks[f"core/{DECISIVE_K}/{L}"],
                                                        a.seed + 500 * (r + 1) + L)[0]
                       for L in layers} for r in range(N_TWIN_REF)]
    twin_ref_onehot = []
    for r in range(N_TWIN_REF):
        cell = {}
        for L in layers:
            rng = np.random.default_rng(a.seed + 900 * (r + 1) + L)
            idx = rng.choice(N_IN, DECISIVE_K, replace=False)
            Q = np.zeros((N_IN, DECISIVE_K)); Q[idx, np.arange(DECISIVE_K)] = 1.0
            cell[L] = Q
        twin_ref_onehot.append(cell)

    # layer groups: 6 highest vs 6 lowest core ACT ratio at the decisive k (§6.1.7 context)
    ratio = {L: act_energy(gram[L], banks[f"core/{DECISIVE_K}/{L}"]) / (DECISIVE_K / N_IN)
             for L in layers}
    order = sorted(layers, key=lambda L: -ratio[L])
    hi6, lo6 = order[:6], order[-6:]
    rep["act"]["core_ratio_by_layer"] = {str(L): ratio[L] for L in layers}
    rep["act"]["layer_groups"] = {"high": hi6, "low": lo6}

    arms = {}
    for k in k_grid:
        arms[f"core/{k}"] = by_layer(banks, "core", k, layers)
        arms[f"anti/{k}"] = by_layer(banks, "anti", k, layers)
        arms[f"topenergy/{k}"] = topen[k]
        for d in range(N_RAND_DRAWS):
            arms[f"random{d}/{k}"] = rand_draws[k][d]
    for bid in ("topsigma", "bottomsigma"):
        arms[f"{bid}/{DECISIVE_K}"] = by_layer(banks, bid, DECISIVE_K, layers)
    arms[f"complement/{H.BANK_K}"] = by_layer(banks, "complement", H.BANK_K, layers)
    arms[f"topenergy_gamma/{DECISIVE_K}"] = topen_g[DECISIVE_K]
    for d, cell in enumerate(em_draws):
        arms[f"ematch{d}/{DECISIVE_K}"] = cell
    arms[f"twin_core/{DECISIVE_K}"] = twin_core
    arms[f"twin_topenergy/{DECISIVE_K}"] = twin_topen
    for r in range(N_TWIN_REF):
        arms[f"twinref_dense{r}/{DECISIVE_K}"] = twin_ref_dense[r]
        arms[f"twinref_onehot{r}/{DECISIVE_K}"] = twin_ref_onehot[r]
    arms[f"layergroup_hi/{DECISIVE_K}"] = {L: banks[f"core/{DECISIVE_K}/{L}"] for L in hi6}
    arms[f"layergroup_lo/{DECISIVE_K}"] = {L: banks[f"core/{DECISIVE_K}/{L}"] for L in lo6}

    rep["arm_energy_D"] = {name: energy_frac(Q) for name, Q in arms.items()}

    # ---- D half: run every arm, set the bands and the margins -------------------------
    enc_cache = {}

    def encoding_for(which, wins):
        """Encode each half ONCE; D/D2 and C/C2 score the same windows."""
        half = which[0]
        if half not in enc_cache:
            enc_cache[half] = encode_windows(processor, device, wins)
            print(f"[a1] encoded {len(wins)} windows for half {half} (processor run once)",
                  flush=True)
        return enc_cache[half]

    def run_half(which, wins, arm_table, scope_cells=()):
        enc = encoding_for(which, wins)
        pw_clean = ppl_encoded(model, enc)
        res = {"clean_ppl": C.ppl(pw_clean), "n_windows": len(wins), "arms": {}}
        print(f"[a1][{which}] clean ppl = {res['clean_ppl']:.4f} over {len(wins)} windows",
              flush=True)
        for i, (name, Qby) in enumerate(arm_table.items()):
            with _InputProject(mlayers, Qby, device):
                pw = ppl_encoded(model, enc)
            ci = C.paired_log_ratio_ci(pw, pw_clean, seed=a.seed)
            res["arms"][name] = {"dnll": ci["point"], "ci": ci, "ppl": C.ppl(pw),
                                 "cap_sat": bool(C.ppl(pw) > C.CAP_SAT_MULT * res["clean_ppl"]),
                                 "per_window": pw.tolist()}
            print(f"[a1][{which}] {i + 1}/{len(arm_table)} {name}: dNLL={ci['point']:+.4f} "
                  f"ppl={C.ppl(pw):.3f}{' SAT' if res['arms'][name]['cap_sat'] else ''}", flush=True)
        for name, Qby in scope_cells:
            with _ResidualProject(mlayers, Qby, device):
                pw = ppl_encoded(model, enc)
            ci = C.paired_log_ratio_ci(pw, pw_clean, seed=a.seed)
            res["arms"][name] = {"dnll": ci["point"], "ci": ci, "ppl": C.ppl(pw),
                                 "cap_sat": bool(C.ppl(pw) > C.CAP_SAT_MULT * res["clean_ppl"]),
                                 "per_window": pw.tolist(), "scope": "full residual stream"}
            print(f"[a1][{which}] SCOPE {name}: dNLL={ci['point']:+.4f}", flush=True)
        res["clean_per_window"] = pw_clean.tolist()
        return res

    scope = [(f"scope_core/{DECISIVE_K}", by_layer(banks, "core", DECISIVE_K, layers)),
             (f"scope_topenergy/{DECISIVE_K}", topen[DECISIVE_K])]
    Dres = run_half("D", Dw, arms, scope)
    rep["D"] = {k: v for k, v in Dres.items() if k != "clean_per_window"}
    save()

    # bands + margins from D
    thresholds = {"k_grid": k_grid, "decisive_k": DECISIVE_K, "salt": SPLIT_SALT,
                  "bands": {}, "margins": {}}
    for k in k_grid:
        b = C.band([Dres["arms"][f"random{d}/{k}"]["dnll"] for d in range(N_RAND_DRAWS)])
        m = C.margins(b["width"], Dres["arms"][f"core/{k}"]["dnll"])
        thresholds["bands"][str(k)] = b
        thresholds["margins"][str(k)] = m
        print(f"[a1][D] k={k}: core dNLL={Dres['arms'][f'core/{k}']['dnll']:.4f} "
              f"rand band=[{b['lo']:.4f},{b['hi']:.4f}] w={b['width']:.5f} "
              f"m_sup={m['m_sup']:.4f} m_eq={m['m_eq']:.4f}", flush=True)
    thresholds["twin_ref_band_dense"] = C.band(
        [Dres["arms"][f"twinref_dense{r}/{DECISIVE_K}"]["dnll"] for r in range(N_TWIN_REF)])
    thresholds["twin_ref_band_onehot"] = C.band(
        [Dres["arms"][f"twinref_onehot{r}/{DECISIVE_K}"]["dnll"] for r in range(N_TWIN_REF)])
    thresholds["em_band"] = C.band(
        [Dres["arms"][f"ematch{d}/{DECISIVE_K}"]["dnll"] for d in range(N_EM_DRAWS)], )

    # ---- STAGE 2 banks: functions of the D-half energy, hashed into thresholds.json ----
    e_core = rep["arm_energy_D"][f"core/{DECISIVE_K}"]
    # k_b is matched on ENERGY, and is NOT restricted to the 7-point k-grid: the grid exists to
    # give a dose curve, and forcing the comparator onto it would leave the "equal-energy" contrast
    # matched only to the nearest factor of two — which at these widths is not equal energy at all.
    # Search every integer width and take the closest mean removed energy.
    # A one-hot bank's removed energy is EXACTLY the top-k diagonal mass of the Gram, so one sorted
    # cumulative sum per layer gives the energy at every width at once — no projection, no matmul.
    _cum = {L: np.cumsum(np.sort(np.diag(gram[L]))[::-1]) / float(np.trace(gram[L]))
            for L in layers}
    _widths = np.arange(1, N_IN + 1)
    _mean_e = np.mean([_cum[L] for L in layers], axis=0)
    kb = int(_widths[int(np.argmin(np.abs(_mean_e - e_core)))])
    kb_e = float(_mean_e[kb - 1])
    kb_bank = {L: C.topenergy_bank(gram[L], kb)[0] for L in layers}
    kb_degenerate = bool(kb == DECISIVE_K)
    sd_core, sd_te, dims = {}, {}, []
    for L in layers:
        A, keep_a, _ = C.set_difference_bank(banks[f"core/{DECISIVE_K}/{L}"], topen[DECISIVE_K][L])
        B, keep_b, _ = C.set_difference_bank(topen[DECISIVE_K][L], banks[f"core/{DECISIVE_K}/{L}"])
        dims.append(min(keep_a, keep_b))
    common = int(min(dims))
    for L in layers:
        A, _, _ = C.set_difference_bank(banks[f"core/{DECISIVE_K}/{L}"], topen[DECISIVE_K][L])
        B, _, _ = C.set_difference_bank(topen[DECISIVE_K][L], banks[f"core/{DECISIVE_K}/{L}"])
        sd_core[L], sd_te[L] = np.ascontiguousarray(A[:, :common]), np.ascontiguousarray(B[:, :common])
    sd_small = {}
    for L in layers:
        A, _, _ = C.set_difference_bank(banks[f"core/{SMALL_K_GATE}/{L}"], topen[SMALL_K_GATE][L])
        sd_small[L] = A
    small_dim = int(min(sd_small[L].shape[1] for L in layers))
    sd_small = {L: np.ascontiguousarray(sd_small[L][:, :small_dim]) for L in layers}

    stage2 = {f"kb_topenergy/{kb}": kb_bank, f"core_minus_topenergy/{common}": sd_core,
              f"topenergy_minus_core/{common}": sd_te,
              f"core_minus_topenergy_smallk/{small_dim}": sd_small}
    # matched-DIMENSION null bands from the SAME nested draws truncated to the realized width
    for dim, tag in ((common, "sd"), (small_dim, "sd_small")):
        for d in range(N_RAND_DRAWS):
            stage2[f"randdim_{tag}{d}/{dim}"] = {
                L: np.ascontiguousarray(
                    C.nested_random_subspaces(N_IN, max(k_grid), a.seed * 7919 + 1000 * d + L)[:, :dim])
                for L in layers}
    thresholds["stage2"] = {
        "k_b": kb, "k_b_energy_D": kb_e, "core_energy_D": e_core,
        "set_difference_dim": common, "small_k_set_difference_dim": small_dim,
        "k_b_degenerate_equals_decisive_k": kb_degenerate,
        "hashes": {name: {str(L): C.bank_hash(Q) for L, Q in cell.items()}
                   for name, cell in stage2.items()}}

    Dres2 = run_half("D2", Dw, stage2)
    # Keyed by TAG, not by width alone: the decisive and small-k set differences can realise the
    # SAME dimension, and a width-only key would let one band silently overwrite the other.
    for dim, tag, core_k in ((common, "sd", DECISIVE_K), (small_dim, "sd_small", SMALL_K_GATE)):
        thresholds["bands"][f"{tag}_dim{dim}"] = C.band(
            [Dres2["arms"][f"randdim_{tag}{d}/{dim}"]["dnll"] for d in range(N_RAND_DRAWS)])
        thresholds["margins"][f"{tag}_dim{dim}"] = C.margins(
            thresholds["bands"][f"{tag}_dim{dim}"]["width"], Dres["arms"][f"core/{core_k}"]["dnll"])
    rep["D2"] = {k: v for k, v in Dres2.items() if k != "clean_per_window"}

    thr_hash = C.write_thresholds(out / "thresholds.json", thresholds)
    rep["thresholds_sha256"] = thr_hash
    rep["stage2"] = thresholds["stage2"]
    save()
    print(f"[a1] thresholds FROZEN sha256={thr_hash[:16]} k_b={kb} sd_dim={common} "
          f"small_sd_dim={small_dim}", flush=True)

    # ---- bank-overlap matrix, BEFORE the C half is touched ----------------------------
    overlap = {}
    for k in k_grid:
        pairs = {"core~anti": (f"core/{k}", f"anti/{k}"), "core~topenergy": (f"core/{k}", f"topenergy/{k}"),
                 "anti~topenergy": (f"anti/{k}", f"topenergy/{k}")}
        overlap[str(k)] = {name: float(np.mean([C.mean_cos2(arms[x][L], arms[y][L]) for L in layers]))
                           for name, (x, y) in pairs.items()}
    rep["bank_overlap_mean_cos2"] = overlap
    non_adjudicating = bool(overlap[str(DECISIVE_K)]["core~topenergy"] > C.BANK_OVERLAP_CSTAR)
    anti_non_adjudicating = bool(overlap[str(DECISIVE_K)]["core~anti"] > C.BANK_OVERLAP_CSTAR)
    gates.record("a1_equal_k_adjudicating", not non_adjudicating,
                 mean_cos2_core_topenergy=overlap[str(DECISIVE_K)]["core~topenergy"],
                 c_star=C.BANK_OVERLAP_CSTAR, k=DECISIVE_K,
                 band="frozen ceiling c*=0.25 (§2.1)")
    save()

    # ---- C half, computed ONCE ---------------------------------------------------------
    thresholds = C.load_thresholds(out / "thresholds.json", thr_hash)
    Cres = run_half("C", Cw, arms, scope)
    Cres2 = run_half("C2", Cw, stage2)
    rep["C"] = {k: v for k, v in Cres.items() if k != "clean_per_window"}
    rep["C2"] = {k: v for k, v in Cres2.items() if k != "clean_per_window"}
    save()

    # ---- instrument gates (§6.1.6) ------------------------------------------------------
    pwC = {n: np.asarray(v["per_window"]) for n, v in {**Cres["arms"], **Cres2["arms"]}.items()}
    # THE SCORED BAND IS THE FROZEN ONE. §6.1.2 calibrates the matched-random band on D and
    # freezes it to thresholds.json before C is computed; scoring against a band recomputed from
    # C's own draws would quietly return the confirmation half's own noise to the decision and
    # defeat the blind. The C-half draws are measured anyway and ship as a context row.
    bandD = {k: thresholds["bands"][str(k)] for k in k_grid}
    bandC = {k: C.band([Cres["arms"][f"random{d}/{k}"]["dnll"] for d in range(N_RAND_DRAWS)])
             for k in k_grid}
    w_rand_dec = thresholds["bands"][str(DECISIVE_K)]["width"]
    g_dyn = bool(Cres["arms"][f"topsigma/{DECISIVE_K}"]["dnll"]
                 > bandD[DECISIVE_K]["hi"] + C.TOPSIGMA_DYNAMIC_RANGE * w_rand_dec)
    g_floor = bool(Cres["arms"][f"bottomsigma/{DECISIVE_K}"]["dnll"] <= bandD[DECISIVE_K]["hi"])
    g_sane = bool(Cres["clean_ppl"] < C.SANE_PPL_MULT * anchor_clean)
    g_twin_core = bool(Cres["arms"][f"twin_core/{DECISIVE_K}"]["dnll"]
                       <= thresholds["twin_ref_band_dense"]["hi"])
    g_twin_te = bool(Cres["arms"][f"twin_topenergy/{DECISIVE_K}"]["dnll"]
                     <= thresholds["twin_ref_band_onehot"]["hi"])
    gates.record("a1_topsigma_dynamic_range", g_dyn,
                 topsigma_dnll=Cres["arms"][f"topsigma/{DECISIVE_K}"]["dnll"],
                 random_band_hi=bandD[DECISIVE_K]["hi"], w_rand=w_rand_dec,
                 required_excess=C.TOPSIGMA_DYNAMIC_RANGE * w_rand_dec,
                 band="frozen D-half band (thresholds.json): " + bandD[DECISIVE_K]["construction"])
    gates.record("a1_bottomsigma_floor", g_floor,
                 bottomsigma_dnll=Cres["arms"][f"bottomsigma/{DECISIVE_K}"]["dnll"],
                 random_band_hi=bandD[DECISIVE_K]["hi"],
                 band="frozen D-half band (thresholds.json): " + bandD[DECISIVE_K]["construction"])
    gates.record("a1_sane_ppl", g_sane, clean_ppl=Cres["clean_ppl"], anchor_clean=anchor_clean,
                 bound=C.SANE_PPL_MULT * anchor_clean, band="frozen 2x the anchor's clean value")
    gates.record("a1_permutation_twins_inside_reference", bool(g_twin_core and g_twin_te),
                 twin_core=Cres["arms"][f"twin_core/{DECISIVE_K}"]["dnll"],
                 twin_core_ref_hi=thresholds["twin_ref_band_dense"]["hi"],
                 twin_topenergy=Cres["arms"][f"twin_topenergy/{DECISIVE_K}"]["dnll"],
                 twin_topenergy_ref_hi=thresholds["twin_ref_band_onehot"]["hi"],
                 band="each twin against its OWN same-construction reference draws (§6.1.2)")
    instrument_ok = bool(g_dyn and g_floor and g_sane and g_twin_core and g_twin_te)

    # ---- decisive k (§6.1.4): largest k at which no ADJUDICATED bank is saturation-flagged
    adjudicated = ("core", "anti", "topenergy")
    unflagged = [k for k in k_grid if not any(Cres["arms"][f"{b}/{k}"]["cap_sat"] for b in adjudicated)]
    dec_k = DECISIVE_K if DECISIVE_K in unflagged else (max(unflagged) if unflagged else None)
    rep["decisive_k_used"] = dec_k
    rep["saturation_flags"] = {f"{b}/{k}": Cres["arms"][f"{b}/{k}"]["cap_sat"]
                               for b in adjudicated for k in k_grid}

    verdict, fired, detail = None, [], {}
    if not instrument_ok:
        verdict = C.VOID_INSTRUMENT
        detail["reason"] = "an §6.1.6 instrument gate did not pass"
    elif dec_k is None:
        verdict = C.VOID_INSTRUMENT
        detail["reason"] = "every k saturation-flagged on an adjudicated bank"
    else:
        m = thresholds["margins"][str(dec_k)]
        m_sup, m_eq = m["m_sup"], m["m_eq"]
        rand_avg = np.mean([pwC[f"random{d}/{dec_k}"] for d in range(N_RAND_DRAWS)], axis=0)
        ci_core_rand = C.paired_log_ratio_ci(pwC[f"core/{dec_k}"], rand_avg, seed=a.seed)
        ci_core_te = C.paired_log_ratio_ci(pwC[f"core/{dec_k}"], pwC[f"topenergy/{dec_k}"], seed=a.seed)
        ci_core_anti = C.paired_log_ratio_ci(pwC[f"core/{dec_k}"], pwC[f"anti/{dec_k}"], seed=a.seed)
        ci_anti_core = C.paired_log_ratio_ci(pwC[f"anti/{dec_k}"], pwC[f"core/{dec_k}"], seed=a.seed)
        ci_eqE = C.paired_log_ratio_ci(pwC[f"core/{dec_k}"], pwC[f"kb_topenergy/{kb}"], seed=a.seed)
        sd_dim = thresholds["stage2"]["set_difference_dim"]
        sd_band = thresholds["bands"][f"sd_dim{sd_dim}"]
        sd_m = thresholds["margins"][f"sd_dim{sd_dim}"]
        randdim_avg = np.mean([pwC[f"randdim_sd{d}/{sd_dim}"] for d in range(N_RAND_DRAWS)], axis=0)
        ci_sd_rand = C.paired_log_ratio_ci(pwC[f"core_minus_topenergy/{sd_dim}"], randdim_avg,
                                           seed=a.seed)
        ci_sd_vs = C.paired_log_ratio_ci(pwC[f"core_minus_topenergy/{sd_dim}"],
                                         pwC[f"topenergy_minus_core/{sd_dim}"], seed=a.seed)
        te_sd_out = bool(Cres2["arms"][f"topenergy_minus_core/{sd_dim}"]["dnll"] > sd_band["hi"])

        # The frozen §6.1.5 decision procedure lives in copernicus_core as a PURE function, so
        # every branch and the precedence table are exercised against planted inputs by the
        # §14.2 dry-run before this job ever holds a GPU.
        vd = C.a1_verdict({
            "instrument_ok": instrument_ok, "unflagged_k": unflagged, "dec_k": dec_k,
            "dnll": {b: {k: Cres["arms"][f"{b}/{k}"]["dnll"] for k in k_grid} for b in adjudicated},
            "bands": bandD, "m_sup": m_sup, "m_eq": m_eq,
            "non_adjudicating": non_adjudicating, "anti_non_adjudicating": anti_non_adjudicating,
            "sd_dnll_core": Cres2["arms"][f"core_minus_topenergy/{sd_dim}"]["dnll"],
            "sd_band": sd_band, "sd_m_sup": sd_m["m_sup"], "te_sd_outside": te_sd_out,
            "ci_core_rand": ci_core_rand, "ci_core_te": ci_core_te, "ci_core_anti": ci_core_anti,
            "ci_anti_core": ci_anti_core, "ci_eqE": ci_eqE, "ci_sd_rand": ci_sd_rand,
            "ci_sd_vs": ci_sd_vs})
        verdict, fired = vd["verdict"], vd["fired"]
        detail = {"legs": vd["legs"], "m_sup": m_sup, "m_eq": m_eq, "k": dec_k, "k_b": kb,
                  "non_adjudicating_equal_k": non_adjudicating,
                  "anti_non_adjudicating": anti_non_adjudicating,
                  "ci_core_vs_random": ci_core_rand, "ci_core_vs_topenergy": ci_core_te,
                  "ci_core_vs_anti": ci_core_anti, "ci_anti_minus_core": ci_anti_core,
                  "ci_equal_energy_core_minus_kb": ci_eqE,
                  "ci_setdiff_vs_random": ci_sd_rand, "ci_setdiff_core_vs_te": ci_sd_vs,
                  "topenergy_minus_core_outside_band": te_sd_out,
                  "set_difference_dim": sd_dim, "sd_band": sd_band,
                  # The stage-2 banks are functions of the D-half energy at k=128 and were hashed
                  # into thresholds.json before C was touched, so they CANNOT be rebuilt if
                  # saturation pushes the decisive k below 128. When that happens the equal-energy
                  # and set-difference cells still speak about k=128 while the grid cells speak
                  # about dec_k, and the row must say so rather than let the mismatch pass.
                  "stage2_pinned_at_k": DECISIVE_K,
                  "stage2_matches_decisive_k": bool(dec_k == DECISIVE_K)}
        if dec_k != DECISIVE_K:
            gates.record("a1_stage2_decisive_k_mismatch", False, decisive_k_used=dec_k,
                         stage2_pinned_at=DECISIVE_K,
                         band="§6.1.2 subjects stage-2 banks to the decisive-k fallback, but they "
                              "are frozen pre-C and cannot be rebuilt; the mismatch is flagged")

        # §6.1.7 the one SCORED small-k gate
        sdim = thresholds["stage2"]["small_k_set_difference_dim"]
        sband = thresholds["bands"][f"sd_small_dim{sdim}"]
        smallrand = np.mean([pwC[f"randdim_sd_small{d}/{sdim}"] for d in range(N_RAND_DRAWS)], axis=0)
        ci_small = C.paired_log_ratio_ci(pwC[f"core_minus_topenergy_smallk/{sdim}"], smallrand,
                                         seed=a.seed)
        small_fired = C.ci_separated(ci_small, thresholds["margins"][f"sd_small_dim{sdim}"]["m_sup"])
        gates.record("a1_small_k_residual_sink_specific", bool(small_fired),
                     k=SMALL_K_GATE, dim=sdim, dnll=Cres2["arms"][f"core_minus_topenergy_smallk/{sdim}"]["dnll"],
                     ci_lo_one_sided=ci_small["lo_one_sided"],
                     m_sup=thresholds["margins"][f"sd_small_dim{sdim}"]["m_sup"],
                     band=sband["construction"], prediction="fires (§11, low-mid)")
        detail["small_k_gate"] = {"fired": bool(small_fired), "ci": ci_small}

    # The primary verdict is a RECORDED GATE with its quantities, not just a field: §6.5.8 forbids
    # bare-flag verdicts, and the §11 grid pairs each pre-logged expectation with the gate that
    # spends it — a verdict that is not a gate leaves its expectation unspendable.
    gates.record("a1_primary", verdict == C.A1_SINK_SPECIFIC, verdict=verdict, fired=fired,
                 decisive_k=dec_k, legs=detail.get("legs"),
                 m_sup=detail.get("m_sup"), m_eq=detail.get("m_eq"),
                 non_adjudicating_equal_k=non_adjudicating,
                 band=("§6.1.5 branch set + precedence table, scored against the frozen D-half "
                       "bands in thresholds.json"),
                 prediction="TRAFFIC at the decisive k on the equal-ENERGY contrast (§11, low-mid)")
    rep["verdict"] = verdict
    rep["verdict_branches_fired"] = fired
    rep["verdict_detail"] = detail
    rep["instrument_gates"] = {"topsigma_dynamic_range": g_dyn, "bottomsigma_floor": g_floor,
                               "sane_ppl": g_sane, "twin_core": g_twin_core,
                               "twin_topenergy": g_twin_te, "all_ok": instrument_ok}
    rep["context"] = {
        "dose_curve_dnll_C": {f"{b}/{k}": Cres["arms"][f"{b}/{k}"]["dnll"]
                              for b in adjudicated for k in k_grid},
        "random_band_C": {str(k): bandC[k] for k in k_grid},
        "removed_energy_D": rep["arm_energy_D"],
        "dnll_per_unit_energy_C": {n: (Cres["arms"][n]["dnll"] / rep["arm_energy_D"][n])
                                   for n in rep["arm_energy_D"] if rep["arm_energy_D"][n] > 0},
        "scope_cells_C": {n: Cres["arms"][n]["dnll"] for n in Cres["arms"] if n.startswith("scope_")},
        "layer_group_C": {n: Cres["arms"][n]["dnll"] for n in Cres["arms"] if n.startswith("layergroup")},
        "note": ("dNLL-per-unit-energy is a CONTEXT row; no verdict clause reads it. The dose "
                 "curve is reported as measured — there is no monotone-dose gate.")}

    for k in k_grid:
        for b in adjudicated:
            at.add("a1_dnll", value=Cres["arms"][f"{b}/{k}"]["dnll"], bank_id=f"{b}{k}",
                   model_id=PG1_REPO, model_sha=base_sha, n=len(Cw),
                   ci_lo=Cres["arms"][f"{b}/{k}"]["ci"]["lo"],
                   ci_hi=Cres["arms"][f"{b}/{k}"]["ci"]["hi"],
                   corpus_id="wikitext2:window512:C", seed=a.seed,
                   flags={"k": k, "half": "C", "site": "q/k read-path",
                          "removed_energy_D": rep["arm_energy_D"][f"{b}/{k}"],
                          "cap_sat": Cres["arms"][f"{b}/{k}"]["cap_sat"],
                          "band": "frozen D: " + bandD[k]["construction"]})
    at.close()
    at.to_parquet()
    save()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_A1_OK verdict={verdict} fired={fired} decisive_k={dec_k} "
          f"instrument_ok={instrument_ok} n_D={len(Dw)} n_C={len(Cw)}", flush=True)


if __name__ == "__main__":
    main()
