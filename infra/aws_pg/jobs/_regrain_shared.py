"""REGRAIN shared job glue — the ONE place the REGRAIN jobs bind the corrected instrument
(regrain_core) to the GRAIN banks. Everything GRAIN-side is REUSED from _grain_shared; this
module only adds the v1 bank pointers, the v2 gate artifact, and the frozen program constants.

Frozen provenance rules (spec §0.5 A5a):
  * The v1 m=1 gate verdict is read from `grain_t3diag/out/gates.json` (the em-calibrated v2
    rule, m1_s0.gate_pass=true) — NEVER from `grain_t2/out/train_primary/gates.json`, whose v1
    rule was unpassable-by-construction (gate_pass=false, pivot=instrument_null).
  * The dictionary WEIGHTS live in `grain_t2/out/train_primary/`; only seed 0 exists.
  * Every ppl leg is recomputed at n_seq=120 (the program-wide freeze). GRAIN's T3 ran n_seq=60;
    its Δppl numbers are NOT comparable and are never quoted.
"""
from __future__ import annotations

import json

from infra.aws_pg.jobs._grain_shared import (  # noqa: F401  (re-exported)
    COMWRITE_ANATOMY,
    K_READ,
    POS_BOS,
    POS_IMAGE,
    POS_TEXT,
    PPL_TOL,
    dict_bank_name,
    disc_holdout_split,
    eval_cell,
    fetch,
    grain_split,
    install_pg,
    load_read_banks,
    load_shards,
    make_row_capture,
    manifest_sha,
    pinned_pg1_sha,
    position_labels,
    resolve_sites,
)
from infra.aws_pg.jobs._grain_shared import S3_BUCKET

# ---- GRAIN v1 banks (read-only inputs; never rewritten) --------------------
GRAIN_GATE0 = f"{S3_BUCKET}/results/grain_gate0/out"
GRAIN_HARVEST = f"{S3_BUCKET}/results/grain_t/out/harvest"
GRAIN_DICTS_L0 = f"{S3_BUCKET}/results/grain_t2/out/train_primary"
GRAIN_DICTS_L14 = f"{S3_BUCKET}/results/grain_ts2/out"
GRAIN_GATES_V2 = f"{S3_BUCKET}/results/grain_t3diag/out/gates.json"     # the v2 rule (A5a)
GRAIN_GATES_L14 = f"{S3_BUCKET}/results/grain_ts2/out/gates.json"
GRAIN_GDISC = f"{S3_BUCKET}/results/grain_gdisc_saction/out/g_disc.json"
GRAIN_SATLAS = f"{S3_BUCKET}/results/grain_s_atlas/out/metrics.json"

# ---- frozen program constants (spec §2, §4, §5, §7) -----------------------
N_PPL_SEQ = 120                    # program-wide; closes v1's 60/120 inconsistency
EM_SEEDS = (500, 501, 502)         # the em-band seeds GRAIN used (spans reproduce exactly)
ROT_SEED = 4242                    # the Haar rotation for every rotated-error cell
RHO = 0.5                          # NON-BINDING, reported (A4)
CAP_ATOMS = 48                     # B* cap (spec §1)
TOP_ALIGN = 24                     # alignment-ranked screening pool (v1 used 16; A4 widens)
N_SCREEN = 320                     # A4: screen-slice size (v1 used 160)
N_DISC = 640
N_SEALED = 400
BH_Q = 0.10
TOP8 = 8
K_CAN = 16


def load_shards_stream(prefix_s3, meta, site_key, workdir="/tmp/regrain/shards"):
    """REGRAIN shards carry a per-row `stream` provenance (0 = VL forward, 1 = blank-image text
    forward) beside `labels`, so Arm H selects pure-text / pure-VL rows without storing anything
    twice. Returns (X fp16, labels int8, stream int8)."""
    import torch
    Xs, Ls, Ss = [], [], []
    for name in meta["shards"][site_key]:
        blob = torch.load(fetch(f"{prefix_s3}/{name}", f"{workdir}/{name}"),
                          map_location="cpu", weights_only=False)
        Xs.append(blob["X"])
        Ls.append(blob["labels"])
        Ss.append(blob["stream"])
    X, lab, st = torch.cat(Xs, 0), torch.cat(Ls, 0), torch.cat(Ss, 0)
    assert X.shape[0] == lab.shape[0] == st.shape[0] == meta["n_rows"][site_key]
    return X, lab, st


def load_v2_gates(workdir="/tmp/regrain"):
    """The v2 (em-calibrated) gate artifact for the L0 grid. Fail-loud if it is not the
    re-adjudicated object (guards against silently reading the v1 gates)."""
    g = json.loads(fetch(GRAIN_GATES_V2, f"{workdir}/gates_v2.json").read_text())
    assert g["object"] == "grain_train_v2+t3v2", f"not the v2 gate artifact: {g['object']}"
    assert g["pivot"] == "grain1_only", f"unexpected v1 pivot {g['pivot']}"
    assert g["configs"]["m1_s0"]["gate_pass"] is True, "banked m=1 dict is not v2-faithful"
    return g


def em_span(r_e, seed, d):
    """Reproduce a GRAIN em-random span EXACTLY. `energy_matched_random(d, E, tr, seed)` draws
    `random_basis(d, r_e, seed)` where r_e = round(E·d/tr); passing (E=r_e, tr=d) recovers the
    same rank, hence the same span, without re-measuring E on new activations (the t3_diag
    trick, reused verbatim)."""
    import torch
    from infra.pg.ctrliface_core import energy_matched_random
    Q, r = energy_matched_random(d, float(r_e), float(d), int(seed), "cpu", torch.float64)
    assert int(r) == int(r_e), f"em span rank drift {r} != {r_e}"
    return Q


def effect_from(per_item, intact_per_item, floor):
    """RF drop (1 − RF) on paired items, dead-model-floor convention."""
    import numpy as np
    pi = np.asarray(per_item, float)
    ii = np.asarray(intact_per_item, float)
    denom = ii.mean() - floor
    assert denom > 1e-3, f"no dynamic range vs dead floor ({denom:.5f}) — halt"
    return float(1.0 - (pi.mean() - floor) / denom)


def effect_se(per_item, intact_per_item, floor, n_boot=2000, seed=1337):
    """Bootstrap SE of `effect` on the screen slice — journaled BEFORE B* is fixed (A4)."""
    import numpy as np
    pi = np.asarray(per_item, float)
    ii = np.asarray(intact_per_item, float)
    rng = np.random.default_rng(seed)
    n = pi.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = ii[idx].mean() - floor
        if denom > 1e-6:
            vals.append(1.0 - (pi[idx].mean() - floor) / denom)
    return float(np.std(np.asarray(vals)))
