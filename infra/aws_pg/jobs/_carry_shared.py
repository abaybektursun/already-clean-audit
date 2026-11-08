"""CARRY shared job glue — freeze-assert mechanics, parent-artifact pointers, the `carry|`
item split, and the Q-builder every arm shares.

§0.5 amendments (journal E2, decided pre-freeze, dated 2026-07-12):
  A1  per-FT gain tasks: COMWRITE wired only 3 harnesses; the repo's wired menu is 4
      (vqav2, textvqa, coco_captions, refcoco). An ASSIGNMENT BATTERY (ft cell × 4 tasks,
      discovery items) fixes each FT's gain task := argmax |gain|; frozen before any
      component cell. Transfer-readout scope disclosed; non-transferring FTs VOID at the
      range guard (§2 intercurrent-event rule).
  A2  stiffness metric = the gated spectral completion of F̄ at (L0, dc, ppl) ONLY
      (fidelity 0.927); K-FAC path invalid (0.103), never used. Q materialized densely
      from banked ingredients; single-metric scope disclosed; NO extra operator pass.
  A3  per-position Rayleigh terms are NOT banked → Arm P's pre-approved recompute battery
      runs; sink positions = attention-telemetry rule (≥ 10× uniform received mass) ∪
      {first position, BOS}; sink_frac = q(δ·1_sink)/q(δ), cross term reported.
  A4  Gemma (1+γ) RMSNorm: fold diag(1+γ) into residual-reading weights, then Hadamard
      rotation is exact (S18); D-suite proves it on a tiny block before Arm W2 trusts it.
  A5  Arm C realized grid: 9 cells/FT (ft, stiff=shared(U_stiff), flat=residual(U_stiff),
      3×(randA, randB) matched-rank eigsplits) + base/floor per task per half; range-guard
      noise := paired-item bootstrap SE of the full-write gain (corpus tasks: paired CIDEr
      bootstrap).
  A6  Arm N(b) substrate: REGRAIN harvest shards (L0/L14, ~1M fp16 rows/site) + ATLAS L8.

Freeze mechanics mirror SPECTER verbatim: control plane checks the per-arm tag
(`scripts/carry_check_freeze.py`); the driver half here recomputes sha256 of the frozen
spec copy shipped in the tarball against `carry_freeze.json` and refuses to run on
mismatch. No retroactive tags.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from infra.aws_pg.jobs._regrain_shared import (  # noqa: F401  (re-exported for carry drivers)
    N_PPL_SEQ,
    S3_BUCKET,
    effect_from,
    effect_se,
    fetch,
    install_pg,
    pinned_pg1_sha,
)
from infra.aws_pg.jobs._specter_shared import (  # noqa: F401
    StageTimer,
    assert_gpu_seconds,
    batch_manifest,
    build_ppl_item,
    build_vqa_item,
    clean_forward,
    emit_family,
    safe_log1p_dppl,
    safe_ppl_gate,
    wikitext_windows,
)

# ---- parent artifacts (read-only; every load-bearing number RECOMPUTED from these) ----------
SPECTER_GATE0 = f"{S3_BUCKET}/results/specter_gate0/out"
SPECTER_O = f"{S3_BUCKET}/results/specter_o/out"
SPECTER_O_L8L14 = f"{S3_BUCKET}/results/specter_o_l8l14/out"
SPECTER_K = f"{S3_BUCKET}/results/specter_k/out"
SPECTER_S = f"{S3_BUCKET}/results/specter_s/out"
SPECTER_H = f"{S3_BUCKET}/results/specter_h/out"
SPECTER_M_FIT = f"{S3_BUCKET}/results/specter_m_fit/out"
COMWRITE_ANATOMY = f"{S3_BUCKET}/results/comwrite_a_anatomy2/out"
COMWRITE_GATE0 = f"{S3_BUCKET}/results/comwrite_gate0_cpu2/out"
COMWRITE_B_DISC = f"{S3_BUCKET}/results/comwrite_b_disc/out"
GAUGE_CS = f"{S3_BUCKET}/results/gauge_full/out/cs/gauge_cs.json"
REGRAIN_HARVEST = f"{S3_BUCKET}/results/regrain_pipeA/out/harvest"
GRAIN_DICTS_L0 = f"{S3_BUCKET}/results/grain_t2/out/train_primary"   # dict_primary_m1_s0.pt = m=1
ATLAS_GATE0 = f"{S3_BUCKET}/results/atlas_gate0/out"
BISPEC_SIGMA = f"{S3_BUCKET}/results/bispec_sigma/out"
CARRY_SCOPE = f"{S3_BUCKET}/results/carry_scope/out"
CARRY_GATE0 = f"{S3_BUCKET}/results/carry_gate0/out"
CARRY_C_ASSIGN = f"{S3_BUCKET}/results/carry_c_assign/out"

_JOBS_DIR = pathlib.Path(__file__).resolve().parent
SPEC_FROZEN = _JOBS_DIR / "carry_spec_frozen.md"
FREEZE_MANIFEST = _JOBS_DIR / "carry_freeze.json"

WIRED_TASKS = ("vqav2", "textvqa", "coco_captions", "refcoco")
N_ITEMS_TASK = 200            # items per readout per half (spec §4.1)
N_STREAM = 1500               # stream window per task at Gate-0 (below refcoco's 1975-item
                              # eval split; > 4× the 400/task the two halves need)
STIFF_METRIC_KEY = "L0/dc/ppl"   # A2: the single frozen stiffness metric


def assert_frozen(arm):
    """Driver half of freeze-or-refuse (the _specter_shared.assert_frozen contract)."""
    assert FREEZE_MANIFEST.exists(), f"no freeze manifest — arm '{arm}' refuses to launch"
    man = json.loads(FREEZE_MANIFEST.read_text())
    assert arm in man, f"arm '{arm}' has no freeze entry — refuses to launch (no retroactive tags)"
    assert SPEC_FROZEN.exists(), "frozen spec copy missing from the packaged tree"
    h = hashlib.sha256(SPEC_FROZEN.read_bytes()).hexdigest()
    assert h == man[arm]["spec_sha256"], (
        f"spec hash mismatch for arm '{arm}': tree={h[:12]} manifest={man[arm]['spec_sha256'][:12]}")
    return man[arm]


def carry_split(content_hashes, name="carry"):
    """Fresh `carry|` salt (spec §2/§4.1) — pure sha1 bucketing on item content hashes.
    Even bucket = discovery, odd = sealed (the comwrite_split convention, re-salted)."""
    import numpy as np

    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"carry|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


# ---------------------------------------------------------------------------
# The operator-completion environment (A2) — fetched + rebuilt exactly as
# specter_k.completion_env, then materialized densely and split.
# ---------------------------------------------------------------------------
def load_completion_env(workdir, key=STIFF_METRIC_KEY):
    """Fetch the SPECTER operator + library banks and rebuild the completion ingredients
    for `key`. Returns (U or None, lam list, band_bulk list, basis (d,d) fp64, sb_slices,
    bank_entry). Every ingredient is the banked artifact — nothing re-derived."""
    import torch

    from infra.pg import specter_core as SC
    wd = pathlib.Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    bank = torch.load(fetch(f"{SPECTER_O}/operator_bank.pt", wd / "operator_bank.pt"),
                      map_location="cpu", weights_only=False)
    if key not in bank:
        bank2 = torch.load(fetch(f"{SPECTER_O_L8L14}/operator_bank.pt",
                                 wd / "operator_bank2.pt"),
                           map_location="cpu", weights_only=False)
        bank2.pop("_meta", None)
        bank.update(bank2)
    assert key in bank, f"operator key {key!r} missing from both banks"
    lib = torch.load(fetch(f"{SPECTER_O}/library_bank.pt", wd / "library_bank.pt"),
                     map_location="cpu", weights_only=False)
    meta = lib["band_projectors_meta"]
    sb = [tuple(x) for x in meta["sb_slices"]]
    basis = lib["sigma_basis_L0"].double()
    e = bank[key]
    sp = e["spikes"][:8]
    if sp:
        U = e["evecs_A1"][:, [s["idx"] for s in sp]].double()
        lam = [s["eval_a1"] for s in sp]
    else:
        U, lam = None, []
    bulk = SC.band_bulk_from_traces(e["traces_sb"], sb, U, lam, basis)
    return U, lam, bulk, basis, sb, e


def load_q_bank(workdir):
    """The carry_scope-built Q bank: evals (d,) fp64, evecs (d,d) fp32, k_stiff, plus the
    ingredient/content hashes. Every arm consuming the split reads THIS artifact."""
    import torch
    qb = torch.load(fetch(f"{CARRY_SCOPE}/carry_q_bank.pt",
                          pathlib.Path(workdir) / "carry_q_bank.pt"),
                    map_location="cpu", weights_only=False)
    for k in ("evals", "evecs", "k_stiff", "metric_key", "sha_ingredients"):
        assert k in qb, f"q bank missing {k!r}"
    assert qb["metric_key"] == STIFF_METRIC_KEY
    return qb


def load_roster(workdir):
    """The 24-FT corpus roster + pinned SHAs + excluded pairs from the COMWRITE anatomy
    artifact (never from prose). Returns the anatomy metrics dict."""
    anat = json.loads(fetch(f"{COMWRITE_ANATOMY}/metrics.json",
                            pathlib.Path(workdir) / "anatomy_metrics.json").read_text())
    assert len(anat["roster"]) == 24, f"roster has {len(anat['roster'])} entries, want 24"
    return anat


def load_promoted_spikes(workdir):
    """The 8 promoted spikes: (key, idx, eval) from specter_s + their eigvecs from the
    operator bank. Returns list of {key, idx, eval, evec (d,) fp32}."""
    import torch
    wd = pathlib.Path(workdir)
    sm = json.loads(fetch(f"{SPECTER_S}/metrics.json", wd / "specter_s_metrics.json")
                    .read_text())
    bank = torch.load(fetch(f"{SPECTER_O}/operator_bank.pt", wd / "operator_bank.pt"),
                      map_location="cpu", weights_only=False)
    out = []
    for row in sm["promoted"]:
        key, idx = row["key"], int(row["idx"])
        assert key in bank, f"promoted spike key {key!r} not in the operator bank"
        out.append({"key": key, "idx": idx, "eval": float(row["eval"]),
                    "evec": bank[key]["evecs_A1"][:, idx].float()})
    assert len(out) == 8, f"expected 8 promoted spikes, got {len(out)}"
    return out


# ---------------------------------------------------------------------------
# Item banking (Gate-0 writes; every later arm aligns by content hash).
# ---------------------------------------------------------------------------
def rows_for_banked(task, wanted_hashes, n_stream):
    """Re-stream + align to a banked half by content hash (comwrite_b_cells.rows_for,
    re-hosted so carry drivers do not import a comwrite JOB module)."""
    from infra.pg.projector_ablation import _content_hash, load_eval_items
    wanted = set(wanted_hashes)
    rows = load_eval_items(task, n_stream)
    by_hash = {}
    for r in rows:
        h = _content_hash(task, r)
        if h in wanted and h not in by_hash:
            by_hash[h] = r
    missing = len(wanted) - len(by_hash)
    assert missing == 0, f"{missing}/{len(wanted)} banked items not re-streamed"
    return [by_hash[h] for h in wanted_hashes]


def load_gate0_items(workdir, task_name, half, n=N_ITEMS_TASK):
    """The Gate-0-banked carry| halves for one task: returns (hashes, n_stream)."""
    it = json.loads(fetch(f"{CARRY_GATE0}/items_carry_{task_name}.json",
                          pathlib.Path(workdir) / f"items_carry_{task_name}.json")
                    .read_text())
    sealed = set(it["sealed_idx"])
    pick = [h for i, h in enumerate(it["hashes"]) if (i in sealed) == (half == "sealed")]
    assert len(pick) >= n, f"{task_name} {half} half has {len(pick)} < {n}"
    return pick[:n], len(it["hashes"])


def gain_of(res_cell, res_base, task_is_corpus, refs=None, cand_cell=None, cand_base=None):
    """Uniform gain interface over per-item and corpus tasks (A5): returns {gain, se}."""
    from infra.pg import carry_core as CC
    if task_is_corpus:
        return {k: v for k, v in CC.cider_paired_gain(cand_cell, cand_base, refs).items()
                if k in ("gain", "se")}
    return CC.paired_gain(res_cell["per_item"], res_base["per_item"])
