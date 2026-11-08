"""GRAIN shared job glue — the one place the grain jobs bind the pure instrument
(grain_bsf / inherit_core / ctrliface_core) to artifacts, items and sites, so every stage
shares a single verified path (spec §2/§3). Heavy imports inside functions.

Frozen provenance rules:
  * Sites are RESOLVED from artifacts, never typed in: primary = L0 (the sealed ENDOW-O origin
    locus, asserted against the endow_o controls selection); secondary = the S_vl home layer
    read from the inherit_svl artifact's own metrics.json.
  * Read banks are LOADED, never re-estimated (B1): S_action = ctrliface_discovery bases.pt
    (validate_bank), S_vl = inherit_svl svl_bases.pt.
  * Item halves are FRESH to GRAIN: held (sealed) iff sha1('grain|'+content_hash) is odd —
    no other program's seal is reproduced or touched (§2).
  * h̄ / floors / ppl reference are PG1-native, recomputed from the GRAIN harvest + the
    invcore-FUNC artifact — never quoted from prose.
"""
from __future__ import annotations

import hashlib
import json

from infra.aws_pg.jobs._inherit_shared import (  # noqa: F401  (re-exported for grain jobs)
    CTRLIFACE_BANK,
    INVCORE_FUNC_JSON,
    S3_BUCKET,
    install_pg,
    load_bank,
)
from infra.aws_pg.jobs._inherit_shared import fetch as _s3_fetch


def fetch(uri, dst):
    """S3 URIs download via the house fetch; plain paths (pipeline-local step outputs, e.g.
    /work/out/<step>/…) resolve directly — the single-instance pipeline chains steps through
    the local disk, not through the 30s S3 streamer (a sync race, never rely on it)."""
    import pathlib
    if str(uri).startswith("s3://"):
        return _s3_fetch(uri, dst)
    p = pathlib.Path(uri)
    assert p.exists(), f"local artifact missing: {p}"
    return p

SVL_OUT = f"{S3_BUCKET}/results/inherit_svl/out"
ENDOW_O_CONTROLS = f"{S3_BUCKET}/results/endow_o_controls/out/controls.json"
# comwrite_a_anatomy2 is the LIVE anatomy artifact (k_star=128, full 126-pair grid, LOO banks
# (2048,128) fp32); comwrite_a_anatomy/ is the superseded first run (no k_star key).
COMWRITE_ANATOMY = f"{S3_BUCKET}/results/comwrite_a_anatomy2/out"
FULL_BANK = f"{S3_BUCKET}/results/inherit_pi05_bank/out/bases_full.pt"   # S_action redraw (draw 2)

K_READ = 128                       # the frozen read-bank rank (full-read ablation cell)
PPL_TOL = 0.5                      # the house ppl-gate tolerance (inherit_gate0 convention)
POS_IMAGE, POS_TEXT, POS_BOS = 0, 1, 2


def grain_split(content_hashes, name):
    """held (SEALED) iff sha1('grain|'+hash) is odd — the COMWRITE salt pattern with the
    grain| salt (spec §2). Routed through enforce_split (fail-loud on an empty half)."""
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"grain|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


def resolve_sites(workdir="/tmp/grain"):
    """(L_primary, L_secondary) from artifacts: primary = the endow_o_controls selection's
    primary (the sealed origin locus — L0 by that artifact's construction, asserted, never
    assumed); secondary = the inherit_svl artifact's own layer field."""
    ctrl = json.loads(fetch(ENDOW_O_CONTROLS, f"{workdir}/endow_o_controls.json").read_text())
    Lp = int(ctrl["selection"]["primary"])
    svl = json.loads(fetch(f"{SVL_OUT}/metrics.json", f"{workdir}/inherit_svl_metrics.json").read_text())
    Ls = int(svl["L"])
    assert Lp != Ls, f"primary and secondary sites coincide (L{Lp})"
    return Lp, Ls


def load_read_banks(workdir="/tmp/grain"):
    """{'saction': {'L': L0, 'V': (2048,128) fp64}, 'svl': {'L': L_home, 'V': (2048,128) fp64}}
    — both loaded frozen + orthonormality-asserted (validate_bank / QR-check)."""
    import torch
    from infra.pg.ctrliface_core import orthonormalize
    from infra.pg.inherit_core import slice_k
    Lp, Ls = resolve_sites(workdir)
    bases, _h, _t = load_bank(workdir)
    V_sa = slice_k(bases[Lp], K_READ)
    svl = torch.load(fetch(f"{SVL_OUT}/svl_bases.pt", f"{workdir}/svl_bases.pt"),
                     map_location="cpu", weights_only=False)
    V_vl = torch.as_tensor(svl["Svl"], dtype=torch.float64)[:, :K_READ]
    err = float((V_vl.T @ V_vl - torch.eye(K_READ, dtype=torch.float64)).abs().max())
    if err > 1e-6:                       # fp32-banked eigvecs — re-orthonormalize, disclose
        V_vl = orthonormalize(V_vl)
    return {"saction": {"L": Lp, "V": V_sa}, "svl": {"L": Ls, "V": V_vl}}


def pinned_pg1_sha(workdir="/tmp/grain"):
    """The invcore-FUNC-pinned PG1 sha (exact instrument reproduction) + its ppl reference."""
    func = json.loads(fetch(INVCORE_FUNC_JSON, f"{workdir}/invcore_func.json").read_text())
    return func["base_sha"], float(func["capability_baseline"]["ppl"])


def position_labels(input_ids, image_token_id, bos_token_id):
    """(B, S) int64 labels: 0=image, 1=text, 2=BOS (the §1 pre-mortem covariate)."""
    import torch
    lab = torch.full_like(input_ids, POS_TEXT)
    lab[input_ids == image_token_id] = POS_IMAGE
    lab[input_ids == bos_token_id] = POS_BOS
    return lab


def make_row_capture(store, d):
    """Capture-only pre-hook: bank the entering residual as fp16 CPU rows per call (paired with
    the caller's per-batch labels). Streams per batch — never one dump (x_cap OOM lesson)."""
    from infra.pg.inherit_core import _split_hidden

    def hook(_module, args, kwargs):
        hs, _ = _split_hidden(args, kwargs)
        store.append(hs.detach().to("cpu", dtype=__import__("torch").float16).reshape(-1, d))
    return hook


def manifest_sha(hashes, cfg):
    """sha256 of the sorted item content hashes + the frozen config — the training-data
    manifest identity banked with every dictionary (T4)."""
    blob = json.dumps({"hashes": sorted(hashes), "cfg": cfg}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def load_shards(prefix_s3, meta, site_key, workdir="/tmp/grain/shards"):
    """Fetch + concatenate a site's fp16 shards (+ labels). Returns (X (N,d) fp16, labels)."""
    import torch
    Xs, Ls = [], []
    for name in meta["shards"][site_key]:
        p = fetch(f"{prefix_s3}/{name}", f"{workdir}/{name}")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        Xs.append(blob["X"])
        Ls.append(blob["labels"])
    X = torch.cat(Xs, 0)
    lab = torch.cat(Ls, 0)
    assert X.shape[0] == lab.shape[0] and X.shape[0] == meta["n_rows"][site_key]
    return X, lab


def disc_holdout_split(n, frac=0.1, seed=1337):
    """Held-out-WITHIN-discovery rows for FVU (T2) — never the sealed item half."""
    import torch
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_hold = max(1, int(frac * n))
    return perm[n_hold:], perm[:n_hold]          # (train_idx, holdout_idx)


def dict_bank_name(site_key, m, seed):
    return f"dict_{site_key}_m{m}_s{seed}.pt"


def eval_cell(model, proc, task, rows, ctx_factory, device="cuda", batch_size=16):
    """One behavioral cell under a context-manager factory (fresh ctx per call)."""
    import numpy as np
    from infra.pg import inherit_pg1 as IP
    r = IP.eval_pg1(model, proc, task, rows, device, ctx=ctx_factory(), batch_size=batch_size)
    return np.asarray(r["per_item"], float), r
