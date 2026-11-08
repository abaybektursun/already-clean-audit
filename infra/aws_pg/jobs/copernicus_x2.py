"""CO-X2 (item 53 + item 82 mandate) — the Llama-3.2-3B row-Gram treatment, per type.

QUESTION. M-B1 asked, on PG1, whether the row-side suppressed subspace is a SECOND object or the
input-side spectrum re-read. Its statistic is rho = Spearman, across cells, of the input-side
avoidance ratio against the row-side one: if the row side were the input spectrum transposed, the
two numbers would move together. Item 53 asks the same of a foreign family; item 82 mandates that
it be answered PER TYPE from the start, never pooled.

THE STATISTIC AND ITS LEGACY LABEL ARE READ AT RUN TIME from `mercator_b1_types.json`, never
retyped, with path and sha256 recorded in the artifact (§6.3). The 0.8 same-object threshold is a
LEGACY LABEL and is reported as one; adjudication is null-relative.

TWO NULLS, AND THE SECOND IS THE ONE THAT ADJUDICATES.

    N1  both bases Haar-randomised per delta. Bounds the pure construction pull — how much
        input/row avoidance would track even with no shared structure at all.
    N2  ONE side randomised: keep each delta's real right-singular basis, randomise the left.
        Preserves the real input-side consensus and destroys only the row<->column linkage, so it
        is the null that isolates the thing the arm is about.

VERDICT per type = (rho - N2 median) against N2's 2.5-97.5 band. If a type's N2 upper bound reaches
the legacy 0.8, that cell is VOID-INSTRUMENT for the legacy rule and ships null-relative only.

THE NULLS ARE BUILT FROM SPECTRA, NOT FROM MATERIALISED RANDOM MATRICES, which is what makes them
affordable. For a null delta A = U_r S V^T:

    A A^T = U_r S^2 U_r^T          and          A^T A = V S^2 V^T

so N1 accumulates U_r S^2 U_r^T on the row side and V_r S^2 V_r^T on the input side with
independent Haar frames, and N2 accumulates U_r S^2 U_r^T on the row side while its input side is
EXACTLY the real input Gram (V is kept, so V S^2 V^T is the real term). Only the squared singular
values are needed per delta — `svd(compute_uv=False)` — and no null matrix is ever formed.

A ZERO DELTA IS A DATUM, NOT A CORRUPT INPUT. A community fine-tune may leave a matrix bit-identical
to the base — `DevQuasar/llama3.2_3b_chat_brainstorm-v3.3.2` has an untouched `down_proj` at layer 0
— and the PG1 path this streaming loop was adapted from never sees that, because all 24 official
PaliGemma fine-tunes move every matrix. Such a fine-tune has no direction to contribute in that
cell and cannot be unit-Frobenius normalised there. It is therefore dropped FROM THAT CELL ONLY,
`n_terms` is counted PER CELL rather than globally, and the drops are recorded in a ledger: "how
many of the 24 actually wrote here" is a fact about the cloud, not an exception to swallow.

AMBIENT DIMENSION IS ON EVERY ROW and types never pool: Llama-3.2-3B is MQA, so k_proj/v_proj row
Grams are 1024-wide against q_proj/o_proj's 3072 and gate/up's 8192. Pooling those would compare
avoidance ratios whose flat expectations differ by 8x.

THE NULL BAND MUST HAVE SPREAD, AND THIS ARM'S FIRST RUN SHOWED IT MAY NOT. rho is a Spearman over
one point per LAYER, so with 7 layers it takes only a handful of discrete values — and the avoidance
ratio turns out to be fixed almost entirely by the SPECTRUM, which every null draw holds constant.
The result is a null that is nearly deterministic: in the 2026-08-18 run, 5 of 7 types had all 50
N2 draws return the identical rho, giving a 2.5-97.5 band of ZERO width. Against a zero-width band
a single rank swap flips the verdict, so "rho vs the band" is not a test. That is the one-sided
threshold the program has paid for before (DARWIN's `twin_band_degenerate`), and more DRAWS cannot
fix it — only more layers can. A cell whose null band is degenerate is VOID-INSTRUMENT and ships
descriptive only; the verdict consumes that gate rather than recording it alongside.

DRAW COUNT SCALES WITH WIDTH because an eigendecomposition is O(n^3): a wide type gets a wider band
from fewer draws rather than no band at all. The count used is recorded per cell, and a band from
few draws is a wide band, not a wrong one.

TWO STAGES, BECAUSE THE TWO HALVES WANT DIFFERENT MACHINES. Streaming 24 Llama checkpoints is
download-bound — measured at 1.4-17 MB/s from the Hub, i.e. tens of minutes per checkpoint — while
the null pass is GPU-bound and short. Run as one job on a GPU box, the L40S idles for most of a
day. So `--stage stream` accumulates the Grams and per-delta spectra on a CPU box and writes them
to S3, and `--stage score` loads those on a GPU box and does the avoidance, nulls and adjudication.
`--stage all` keeps the single-box path for a smoke.

Nothing about the estimand changes between the two: the Grams handed over are the same tensors the
single-box path would have held in memory.

    python3 -m infra.aws_pg.jobs.copernicus_x2 --out out --manifest-sha256 <sha> \
        --stage stream --stage-s3 s3://.../results/copernicus_x2stream/out
    python3 -m infra.aws_pg.jobs.copernicus_x2 --out out --manifest-sha256 <sha> \
        --stage score  --stage-s3 s3://.../results/copernicus_x2stream/out
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import darwin_census as CEN
from infra.pg import mercator_rowgram as RG
from infra.pg.mercator_erosion import _spearman
from infra.aws_pg.jobs.invariant_core import _delete_cache
from infra.aws_pg.jobs.copernicus_a1_banks import load_manifest

ARM = "CO-X2"
QUEUE_ITEM = "53+82"
WAVE = "WF"
LLAMA_MANIFEST = "infra/pg/copernicus_llama_delta_manifest.json"
# The launcher ships infra/ ONLY, so an artifact read at run time has to live under it. This is a
# byte-identical copy of studies/.../mercator_b1_types.json, and a dry-run test asserts the two
# stay identical — one source of truth, enforced by the gate rather than by a convention. The sha
# recorded in the artifact is therefore the same number either copy would give.
B1_TYPES = "infra/pg/mercator_b1_types.json"
B1_TYPES_ORIGIN = "studies/lineage/paligemma1-pi0/research/mercator_b1_types.json"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYERS = (0, 4, 9, 13, 18, 22, 27)          # 7 of 28; >= 4 per type is what a per-type rho needs
K = H.BANK_K
SEED = 1337
# an eigendecomposition is O(n^3); a wide type gets a wider band from fewer draws, never no band
DRAWS_BY_WIDTH = ((4096, 50), (8192, 20), (10 ** 9, 10))


def n_draws(width):
    """Draws for the null band at this width.

    Sized from a measured smoke, not a guess: the dominant cost per null cell is one
    eigendecomposition, and the counts here keep the whole null pass to a few GPU-hours. A band
    from 50 draws has noisy tails and is a WIDE band, not a wrong one — and the count used rides on
    every cell so a reader can see which it is."""
    return next(n for w, n in DRAWS_BY_WIDTH if width <= w)


def avoidance_gpu(G, torch, k=K):
    """`avoidance` without leaving the GPU.

    The null pass computes one of these per (cell, draw), and routing each through
    `.cpu().numpy()` + LAPACK made the smoke's null pass the whole job's cost centre — a
    CPU eigendecomposition of an 8192-wide Gram is ~a minute, and there are thousands of them.
    torch.linalg.eigvalsh on an L40S is two orders faster. The arithmetic is the same: bottom-k
    eigenvalue sum over the exact diagonal trace, against the flat expectation k/m. A dry-run test
    asserts it agrees with the CPU path.
    """
    m = int(G.shape[0])
    lam = torch.linalg.eigvalsh(G.to(torch.float64))
    lam = torch.clamp(lam, min=0.0)
    trace = float(torch.diagonal(G).to(torch.float64).sum())
    assert trace > 0, "degenerate Gram (zero trace)"
    lam_b = lam[:k]
    frac = float(lam_b.sum()) / trace
    flat = float(k) / float(m)
    return {"bottom_k_energy_fraction": frac, "flat_expectation": flat, "ratio": frac / flat,
            "m": m, "k": k, "trace": trace,
            "degenerate": bool(float(lam_b.max()) <= 1e-12 * max(trace / m, 1e-300))}


def read_b1_types(path=B1_TYPES):
    """§6.3: the SAME_OBJECT statistic and its 0.8 label are READ, with path + sha cited."""
    p = pathlib.Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    thresholds = {t: float(v["threshold"]) for t, v in d["per_type"].items()}
    assert len(set(thresholds.values())) == 1, f"per-type thresholds disagree: {thresholds}"
    return {"path": str(p), "origin": B1_TYPES_ORIGIN,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "legacy_threshold": next(iter(thresholds.values())),
            "pg1_rho_per_type": {t: float(v["rho"]) for t, v in d["per_type"].items()}}


def avoidance(G, k=K, n_terms=None):
    """The avoidance ratio for one Gram, via the house extremes path (bottom-k + exact diagonal
    trace) so a wide type reports the same number without an eigendecomposition it cannot afford.

    FLAGS RANK DEGENERACY. A consensus Gram of n unit-Frobenius deltas has rank at most
    n * min(m, n_in); when that is below the Gram's width the bottom-k eigenvalues are EXACTLY zero
    and the ratio is exactly 0 — a property of the cloud's size, not of where the cloud writes. The
    2-delta smoke produced exactly this at down_proj (input width 8192, rank <= 6144) and it would
    enter a rank correlation as a real datum. Reported as `degenerate`, never silently."""
    lam_b, _Vb, _lt, _Vt, trace, _full = RG.eig_extremes(G, k)
    out = RG.row_avoidance_from_extremes(lam_b, trace, int(G.shape[0]), k)
    scale = float(trace) / float(G.shape[0])
    out["degenerate"] = bool(float(np.max(lam_b)) <= 1e-12 * max(scale, 1e-300))
    if n_terms is not None:
        out["n_terms"] = int(n_terms)
    return out


def spearman_by_type(rows, key_x="input_avoidance", key_y="row_avoidance"):
    out = {}
    for t in sorted({r["type"] for r in rows}):
        sel = [r for r in rows if r["type"] == t]
        if len(sel) < 4:
            out[t] = None
            continue
        out[t] = float(_spearman([r[key_x] for r in sel], [r[key_y] for r in sel]))
    return out


def haar_gram(dev, torch, m, s2, gen):
    """U S^2 U^T for a Haar frame U (m x r) and squared singular values s2 (length r)."""
    r = int(s2.shape[0])
    A = torch.randn(m, r, generator=gen, device=dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return (Q * s2.to(Q.dtype)) @ Q.T


def null_draw(dev, torch, spectra, m_row, n_in, real_in_gram, gen, kind):
    """One (input_gram, row_gram) pair under N1 or N2, accumulated over the cloud's spectra."""
    row = torch.zeros(m_row, m_row, device=dev, dtype=torch.float32)
    inp = None if kind == "N2" else torch.zeros(n_in, n_in, device=dev, dtype=torch.float32)
    for s2 in spectra:
        row += haar_gram(dev, torch, m_row, s2, gen)
        if kind == "N1":
            inp += haar_gram(dev, torch, n_in, s2, gen)
    if kind == "N2":
        # N2 keeps every delta's real right-singular basis, so its input Gram IS the real one
        inp = real_in_gram
    return inp, row


class _CellGrams:
    """`gram[k]` reads that cell's Gram from the synced stage file, so memory stays O(one cell).

    A dict-shaped facade rather than a change to every call site: the streaming path builds real
    dicts and `_score` should not care which it was handed."""

    def __init__(self, loader, which):
        self._loader, self._which = loader, which

    def __getitem__(self, k):
        return self._loader(k, self._which)


def stage_paths(keys):
    """One file per cell, named by (type, layer) — a single 7 GB blob would have to be rewritten
    whole on every incremental save, and the stream stage saves after every checkpoint."""
    return {k: f"gram__{k[0]}__L{k[1]}.npz" for k in keys}


def put_stage(local_dir, uri):
    r = subprocess.run(["aws", "s3", "sync", str(local_dir), uri, "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stage upload failed: {r.stderr[-400:]}"


def get_stage(uri, local_dir):
    r = subprocess.run(["aws", "s3", "sync", uri, str(local_dir), "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stage download failed: {r.stderr[-400:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=LLAMA_MANIFEST)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--stage", choices=("all", "stream", "score"), default="all")
    ap.add_argument("--stage-s3", default="",
                    help="where `stream` writes and `score` reads the Grams + spectra")
    ap.add_argument("--layers", default="",
                    help="comma list; default LAYERS. The 7-layer default gives a 7-point "
                         "Spearman whose null band is degenerate — pass all 28 to fix it.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 models, 1 layer, 3 null draws — validates the whole path cheaply")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import torch
    # the stream stage is download- and BLAS-bound and runs on a CPU box; only scoring needs CUDA
    need_gpu = a.stage in ("all", "score")
    assert not need_gpu or torch.cuda.is_available(), "no CUDA"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert a.stage == "all" or a.stage_s3, "--stage stream/score needs --stage-s3"

    man = load_manifest(a.manifest, a.manifest_sha256)
    repos, shas, base_sha = man["repos"], man["shas"], man["base_sha"]
    base_repo = man["base_repo"]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(LAYERS)
    if a.smoke:
        layers = layers[:1]
    assert len(set(layers)) == len(layers), f"duplicate layers: {layers}"
    if a.smoke:
        repos = repos[:2]
    b1 = read_b1_types()

    rep = {"object": "copernicus_x2_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "smoke": bool(a.smoke),
           "base_repo": base_repo, "base_sha": base_sha, "n_ft": len(repos),
           "manifest": a.manifest, "manifest_sha256": man["sha256"],
           "types": list(TYPES), "layers": layers, "k": K,
           "b1_types_source": b1,
           "statistic": ("Spearman across cells of input-side avoidance ratio vs row-side "
                         "avoidance ratio; high |rho| means the row side is the input spectrum "
                         "re-read (M-B1's definition, read from the cited artifact)"),
           "provenance": C.provenance(), "utc": C.utc_now(), "cells": []}

    def save():
        (out / "copernicus_x2.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    # CEN, not M: mercator_core.load_decoder_matrices routes through a PaliGemma-scoped loader
    # (it exists because PG1's SigLIP tower duplicates the decoder key suffixes) and dies on a
    # Llama config. darwin_census's reads the safetensors shards directly, which is the
    # single-tower path AND the one that tolerates a resized vocabulary — the exact failure its
    # own docstring records as defect D10.
    scratch = pathlib.Path(H.cache_dir(out, "x2_stage"))
    scratch.mkdir(parents=True, exist_ok=True)

    if a.stage == "score":
        get_stage(a.stage_s3, scratch)
        meta = json.loads((scratch / "stage_meta.json").read_text())
        assert meta["manifest_sha256"] == man["sha256"], "stage was built from a different cloud"
        assert meta["n_streamed"] == len(repos), (
            f"stage holds {meta['n_streamed']} deltas, manifest has {len(repos)} — "
            f"the stream stage did not finish")
        keys = [tuple(k) for k in meta["keys"]]
        # LAZY, and bounded. Holding every Gram resident is ~65 GB in fp64 at 28 layers x 7 types:
        # too big for an L40S's 48 GB VRAM, and too big for a g6e.xlarge's 32 GB of host RAM — the
        # first 28-layer score died OOM in six minutes doing exactly that. Nothing needs them all
        # at once: the cell loop reads one cell, and the null loop needs only the spectra plus (for
        # N2) one TYPE's input Grams. So the spectra and term counts are preloaded (they are tiny)
        # and the Grams are read from the synced stage files on demand.
        paths = stage_paths(keys)
        spectra, n_terms = {}, {}
        for k in keys:
            z = np.load(scratch / paths[k])
            spectra[k] = [torch.tensor(r, dtype=torch.float32) for r in z["spectra"]]
            n_terms[k] = int(z["n_terms"][0])
            assert n_terms[k] == len(spectra[k]), (
                f"{k}: stage says {n_terms[k]} terms but carries {len(spectra[k])} spectra")
            del z

        def _gram(k, which):
            z = np.load(scratch / paths[k])
            return torch.tensor(z[which], dtype=torch.float64)

        gram_in = _CellGrams(_gram, "gin")
        gram_row = _CellGrams(_gram, "grow")
        rep["n_streamed"] = int(meta["n_streamed"])
        rep["n_terms_per_cell"] = meta.get("n_terms_per_cell")
        rep["zero_deltas"] = meta.get("zero_deltas")
        rep["stage_s3"] = a.stage_s3
        print(f"[x2] loaded stage: {len(keys)} cells, n_terms "
              f"{min(n_terms.values())}-{max(n_terms.values())} per cell", flush=True)
        save()
        return _score(a, rep, out, save, keys, gram_in, gram_row, spectra, n_terms, torch,
                      dev, b1)

    base_mats = CEN.load_decoder_matrices(base_repo, base_sha, TYPES, layers)
    keys = sorted(base_mats)
    print(f"[x2] {len(keys)} cells x {len(repos)} deltas", flush=True)

    gram_in, gram_row, spectra = {}, {}, {k: [] for k in keys}
    n_terms = {k: 0 for k in keys}          # PER CELL: a fine-tune can leave one matrix untouched
    zero_deltas = []
    n_streamed = 0
    for i, repo in enumerate(repos):
        print(f"[x2] {repo} ({i + 1}/{len(repos)})", flush=True)
        ft = CEN.load_decoder_matrices(repo, shas[repo], TYPES, layers)
        for k in keys:
            dW = (ft[k].to(torch.float64) - base_mats[k].to(torch.float64)).to(dev)
            f2 = float((dW ** 2).sum())
            if f2 <= 0.0:
                # bit-identical to the base: no direction to contribute, and not normalisable
                zero_deltas.append({"repo": repo, "type": k[0], "layer": k[1]})
                del dW
                continue
            dW = dW / np.sqrt(f2)                      # unit Frobenius, as M-B1 normalises
            if k not in gram_in:
                gram_in[k] = torch.zeros(dW.shape[1], dW.shape[1], device=dev,
                                         dtype=torch.float64)
                gram_row[k] = torch.zeros(dW.shape[0], dW.shape[0], device=dev,
                                          dtype=torch.float64)
            gram_in[k] += dW.T @ dW
            gram_row[k] += dW @ dW.T
            sv = torch.linalg.svdvals(dW)
            spectra[k].append((sv ** 2).to(torch.float32).cpu())
            n_terms[k] += 1
            del dW
        n_streamed += 1
        del ft
        torch.cuda.empty_cache()
        _delete_cache(repo)
        rep["n_streamed"] = n_streamed
        rep["zero_deltas"] = zero_deltas
        rep["n_terms_per_cell"] = {f"{k[0]}|L{k[1]}": v for k, v in n_terms.items()}
        save()
        if a.stage == "stream":
            names = stage_paths(keys)
            for k in keys:
                np.savez(scratch / names[k],
                         gin=gram_in[k].cpu().numpy().astype(np.float32),
                         grow=gram_row[k].cpu().numpy().astype(np.float32),
                         spectra=np.stack([t.numpy() for t in spectra[k]]),
                         n_terms=np.array([n_terms[k]]))
            (scratch / "stage_meta.json").write_text(json.dumps(
                {"manifest_sha256": man["sha256"], "n_streamed": n_streamed,
                 "n_terms_per_cell": {f"{k[0]}|L{k[1]}": v for k, v in n_terms.items()},
                 "zero_deltas": zero_deltas,
                 "keys": [list(k) for k in keys], "layers": layers, "types": list(TYPES)}))
            put_stage(scratch, a.stage_s3)
            print(f"[x2] stage written to {a.stage_s3} after {n_streamed} deltas "
                  f"({len(zero_deltas)} zero cells)", flush=True)

    if a.stage == "stream":
        rep["stage_s3"] = a.stage_s3
        rep["stage_complete"] = bool(n_streamed == len(repos))
        save()
        print(f"[x2] STREAM COMPLETE n={n_streamed} zero_cells={len(zero_deltas)}", flush=True)
        return

    return _score(a, rep, out, save, keys, gram_in, gram_row, spectra, n_terms, torch, dev, b1)


def _score(a, rep, out, save, keys, gram_in, gram_row, spectra, n_terms, torch, dev, b1):
    gen = torch.Generator(device=dev); gen.manual_seed(SEED)
    rows = []
    for k in keys:
        t, L = k
        nt = n_terms[k]
        Gin = (gram_in[k] / nt)
        Grow = (gram_row[k] / nt)
        ain = avoidance(Gin.cpu().numpy(), n_terms=nt)
        arow = avoidance(Grow.cpu().numpy(), n_terms=nt)
        width, in_width = int(Grow.shape[0]), int(Gin.shape[0])
        del Gin, Grow                      # the normalised copies, not the banked Grams
        rows.append({"type": t, "layer": L, "input_avoidance": ain["ratio"],
                     "row_avoidance": arow["ratio"], "row_width": width,
                     "input_width": in_width, "n_terms": nt,
                     "degenerate": bool(ain["degenerate"] or arow["degenerate"])})
        rep["cells"] = rows
        save()
        print(f"[x2] {t:10s} L{L:<2d} in={ain['ratio']:.4f} row={arow['ratio']:.4f} "
              f"m_row={width}", flush=True)

    # a degenerate cell is not a datum: its ratio is fixed by the cloud's rank, not its geometry
    live = [r for r in rows if not r["degenerate"]]
    rep["n_degenerate_cells"] = len(rows) - len(live)
    rep["rho_per_type"] = spearman_by_type(live)
    rep["rho_per_type_including_degenerate"] = spearman_by_type(rows)
    save()

    # ---- nulls, per type, drawn on the same cells --------------------------------------------
    nulls = {}
    for kind in ("N1", "N2"):
        draws_by_type = {}
        for t in sorted({r["type"] for r in rows}):
            tk = [k for k in keys if k[0] == t]
            # shapes come from the CELL ROWS, not from re-reading a Gram: under the lazy loader
            # `gram_row[k].shape` is a disk read, and these are needed once per type.
            dims = {(r["type"], r["layer"]): (r["row_width"], r["input_width"]) for r in rows}
            width = max(dims[k][0] for k in tk)
            B = 3 if a.smoke else n_draws(width)
            # HOISTED OUT OF THE DRAW LOOP. N2 reuses each cell's REAL input Gram on every draw;
            # reading it per draw would be ~1400 reads of up to 537 MB per type under the lazy
            # loader. Held as float32 (the value is cast to float32 in use anyway) so a whole
            # type's input Grams stay well inside the card.
            real_in = ({k: (gram_in[k] / n_terms[k]).to(torch.float32).to(dev) for k in tk}
                       if kind == "N2" else {})
            per_draw = []
            for b in range(B):
                nrows = []
                for k in tk:
                    m_row, n_in = dims[k]
                    sp = [s.to(dev) for s in spectra[k]]
                    nt = n_terms[k]
                    ni, nr = null_draw(dev, torch, sp, m_row, n_in, real_in.get(k), gen, kind)
                    nrows.append({"type": t, "layer": k[1],
                                  "input_avoidance": avoidance_gpu(
                                      ni / (1.0 if kind == "N2" else nt), torch)["ratio"],
                                  "row_avoidance": avoidance_gpu(nr / nt, torch)["ratio"]})
                    del ni, nr
                torch.cuda.empty_cache()
                if len(nrows) >= 4:
                    per_draw.append(float(_spearman([r["input_avoidance"] for r in nrows],
                                                    [r["row_avoidance"] for r in nrows])))
            draws_by_type[t] = {"n_draws": B, "width": width,
                                "rho": per_draw,
                                "n_distinct": len(set(round(x, 9) for x in per_draw)),
                                "median": float(np.median(per_draw)) if per_draw else None,
                                "lo": float(np.percentile(per_draw, 2.5)) if per_draw else None,
                                "hi": float(np.percentile(per_draw, 97.5)) if per_draw else None}
            real_in.clear()
            torch.cuda.empty_cache()
            print(f"[x2] {kind} {t}: n={B} median={draws_by_type[t]['median']} "
                  f"distinct={draws_by_type[t]['n_distinct']}", flush=True)
            nulls[kind] = draws_by_type
            rep["nulls"] = nulls
            save()

    # ---- adjudication: per type, against N2 ---------------------------------------------------
    legacy = b1["legacy_threshold"]
    verdicts = {}
    for t, rho in rep["rho_per_type"].items():
        n2 = nulls["N2"].get(t) or {}
        if rho is None or n2.get("median") is None:
            verdicts[t] = {"verdict": "VOID-COVERAGE", "reason": "fewer than 4 cells for a rho"}
            continue
        excess = rho - n2["median"]
        void_legacy = n2["hi"] >= legacy
        # THE BAND'S SPREAD IS A VALIDITY GATE, consumed here rather than recorded beside the
        # verdict. A band of zero width cannot separate anything: every rho that is not exactly
        # the null's single value reads as "escapes" or "follows" with no tolerance at all.
        band_width = n2["hi"] - n2["lo"]
        if band_width <= 0.0 or n2.get("n_distinct", 0) < 3:
            verdicts[t] = {
                "verdict": "VOID-INSTRUMENT", "rho": rho,
                "excess_over_n2_median": excess,
                "n2_band": [n2["lo"], n2["hi"]], "n2_band_width": band_width,
                "n2_distinct_values": n2.get("n_distinct"), "n2_draws": n2["n_draws"],
                "row_width": n2["width"], "legacy_threshold": legacy,
                "legacy_rule_void": bool(void_legacy),
                "reason": ("the N2 band is degenerate — rho is a Spearman over one point per "
                           "layer and the null holds the spectra fixed, so the null barely "
                           "varies; more draws cannot widen it, only more layers can"),
                "note": "descriptive only; no directional claim is made for this type"}
            continue
        if rho > n2["hi"]:
            v = "follows the pull"
        elif rho < n2["lo"]:
            v = "escapes the pull"
        else:
            v = "inside the null band"
        verdicts[t] = {"verdict": v, "rho": rho, "excess_over_n2_median": excess,
                       "n2_band": [n2["lo"], n2["hi"]], "n2_band_width": band_width,
                       "n2_distinct_values": n2.get("n_distinct"), "n2_draws": n2["n_draws"],
                       "row_width": n2["width"],
                       "legacy_threshold": legacy,
                       "legacy_rule_void": bool(void_legacy),
                       "note": ("VOID-INSTRUMENT for the legacy 0.8 rule; ships null-relative only"
                                if void_legacy else "legacy label reportable")}
    rep["verdicts"] = verdicts
    n_void = sum(1 for v in verdicts.values() if v["verdict"] == "VOID-INSTRUMENT")
    rep["adjudication_summary"] = {
        "n_types": len(verdicts), "n_void_instrument": n_void,
        "arm_adjudicable": bool(n_void < len(verdicts)),
        "note": ("a type whose N2 band has zero width is not adjudicated; if every type voids, "
                 "the arm has measured rho but adjudicated nothing")}
    # An ARM-LEVEL token, so this row reads like every other arm in the ledger and the §9.7
    # re-derivation has something to check. §6.3 forbids pooling types, so the arm-level token is
    # a statement about COVERAGE — how many types were adjudicable — never an average of them.
    rep["verdict"] = ("VOID-INSTRUMENT" if n_void == len(verdicts)
                      else ("PARTIAL-COVERAGE" if n_void else "ADJUDICATED"))
    rep["verdict_basis"] = (f"{len(verdicts) - n_void} of {len(verdicts)} types had an N2 band "
                            f"with usable spread; types are never pooled (§6.3)")
    save()
    print("[x2] " + json.dumps(verdicts, default=float), flush=True)


if __name__ == "__main__":
    main()
