"""CONFLUX shared job glue — banked-tensor loaders for the CPU-only overlap test.

Everything is REUSED from the CARRY/HANDLE stack (`load_q_bank`, `load_roster`,
`load_promoted_spikes`, `fetch`, `StageTimer`, `install_pg`); this module adds only the CONFLUX-
specific reads: the pre-spilled raw o_proj ΔW (`comwrite_a_anatomy2/out/scratch/o_proj__L{L}__{ridx}`),
the frozen 12/12 discovery/confirmation split (journal A3), the U_wr/U_unt partners, the top-8 spike
carrier, and the banked per-FT behavioural gains. No model, no GPU. No try/except hiding (CLAUDE.md).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from infra.aws_pg.jobs._carry_shared import (  # noqa: F401
    StageTimer,
    fetch,
    install_pg,
    load_promoted_spikes,
    load_q_bank,
    load_roster,
)
from infra.aws_pg.jobs._grain_shared import S3_BUCKET

COMWRITE_ANATOMY = f"{S3_BUCKET}/results/comwrite_a_anatomy2/out"
SCRATCH = f"{COMWRITE_ANATOMY}/scratch"
SPECTER_O = f"{S3_BUCKET}/results/specter_o/out"
CARRY_SCOPE = f"{S3_BUCKET}/results/carry_scope/out"
CARRY_C_ASSIGN = f"{S3_BUCKET}/results/carry_c_assign/out"
CARRY_C_SEALED = f"{S3_BUCKET}/results/carry_c_sealed/out"
HANDLE_T = f"{S3_BUCKET}/results/handle_t/out"
CONFLUX_DISC = f"{S3_BUCKET}/results/conflux_disc/out"


def slug_of(repo):
    """`google/paligemma-3b-ft-<slug>-224` → <slug>."""
    return repo.split("-ft-")[1].rsplit("-", 1)[0]


def roster_index(workdir):
    """{slug: ridx} and {ridx: slug} for the 24-FT roster (ridx = the scratch-npy suffix)."""
    anat = load_roster(str(workdir))
    slug2i, i2slug = {}, {}
    for ridx, repo in enumerate(anat["roster"]):
        s = slug_of(repo)
        slug2i[s] = ridx
        i2slug[ridx] = s
    assert len(slug2i) == 24, f"roster produced {len(slug2i)} distinct slugs"
    return slug2i, i2slug, anat


def conflux_split(slugs):
    """Frozen, unbiased 12/12 split of the 24 FT slugs (journal A3): rank by
    sha1('conflux_split|'+slug); first 12 = discovery, last 12 = confirmation. Deterministic, no
    discretion — U_bb is weight-only so every FT has usable ΔW; blind-safe because U_sus is
    base-model-only."""
    keyed = sorted(slugs, key=lambda s: hashlib.sha1(f"conflux_split|{s}".encode()).hexdigest())
    assert len(keyed) == 24, f"need 24 slugs, got {len(keyed)}"
    return sorted(keyed[:12]), sorted(keyed[12:])


def load_dW_oproj(workdir, ridx, L=0):
    """The raw o_proj ΔW for roster index `ridx` at layer L — banked fp16 (2048,2048), returned fp64.
    This IS the diff (wf − wb), already spilled by comwrite_a_anatomy (dW = wf − wb)."""
    import numpy as np
    dst = pathlib.Path(workdir) / f"dW_o_proj_L{L}_{ridx}.npy"
    fetch(f"{SCRATCH}/o_proj__L{L}__{ridx}.npy", dst)
    return np.load(dst).astype(np.float64)


def load_dW_list(workdir, ridxs, L=0):
    """List of raw o_proj ΔW (fp64) for the given roster indices at layer L."""
    return [load_dW_oproj(workdir, int(i), L) for i in ridxs]


def load_partners(workdir):
    """U_wr (consensus write, o_proj) and U_unt (conserved core, residual) from the SPECTER library
    bank — the two known shared structures (reference partners). fp64 (2048, k). Slice to r at use."""
    import torch
    lib = torch.load(fetch(f"{SPECTER_O}/library_bank.pt", pathlib.Path(workdir) / "library_bank.pt"),
                     map_location="cpu", weights_only=False)
    U_wr = torch.as_tensor(lib["U_wr_o_proj"], dtype=torch.float64).contiguous().numpy()
    U_unt = torch.as_tensor(lib["U_unt_residual"], dtype=torch.float64).contiguous().numpy()
    return {"U_wr": U_wr, "U_unt": U_unt}


def load_carrier(workdir):
    """The top-8 spike carrier (HANDLE spike_p8): the 8 promoted spike eigvecs, orthonormalised to a
    (2048, 8) frame. fp64."""
    import numpy as np
    sp = load_promoted_spikes(str(workdir))
    V = np.stack([s["evec"].double().numpy() for s in sp], axis=1)   # (2048, 8)
    Q, _R = np.linalg.qr(V)
    return np.ascontiguousarray(Q[:, :8])


def load_gains(workdir):
    """Per-FT banked behavioural quantity for Arm B: the CARRY assignment-battery gain_frac on the
    assigned task (falls back to `gain`). Returns {slug: value or None}. Not a forgetting battery
    (none banked — journal M0.4); Arm B reports INSTRUMENT-NULL if < 3 finite."""
    wd = pathlib.Path(workdir)
    assign = json.loads(fetch(f"{CARRY_C_ASSIGN}/metrics.json", wd / "carry_assign.json").read_text())
    out = {}
    for slug, rec in assign.get("assign", {}).items():
        task = rec.get("assigned_task")
        cell = rec.get(task) if isinstance(rec.get(task), dict) else None
        val = None
        if isinstance(rec.get("stiff_energy_fraction"), (int, float)):
            val = float(rec["stiff_energy_fraction"])
        if cell and isinstance(cell.get("gain"), (int, float)):
            val = float(cell["gain"])
        out[slug] = val
    return out


def fetch_census(workdir):
    """The banked HANDLE census (preflight target): cross_overlap_med / haar_p95 / core_uwr / core_uunt."""
    c = json.loads(fetch(f"{HANDLE_T}/metrics.json",
                         pathlib.Path(workdir) / "handle_t.json").read_text())
    return c["census"]
