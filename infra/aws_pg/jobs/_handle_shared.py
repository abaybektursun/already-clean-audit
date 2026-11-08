"""HANDLE shared job glue — freeze-assert mechanics (HANDLE's own tag + spec sha256), the
`handle|` item salt, the CARRY-FT roster RECOMPUTE (§0.5-3), the ladder-basis builder, and the
parent-artifact pointers. Every load-bearing number is RECOMPUTED from its named artifact at run
time (baseline discipline); nothing is transcribed from spec prose.

The swap-shell, the Q bank, the readout, and the gain instrument are all REUSED from the CARRY
job stack (`_carry_shared` → `_regrain_shared`/`_specter_shared`/`_grain_shared`); this module
adds only what is new to HANDLE. Heavy imports inside functions so the module py_compiles on the
control plane. No try/except hiding (CLAUDE.md).

Freeze mechanics mirror CARRY/SPECTER verbatim: the control plane checks the per-arm tag
(`scripts/handle_check_freeze.py`); the driver half here recomputes sha256 of the frozen spec
copy shipped in the tarball against `handle_freeze.json` and refuses to run on mismatch. No
retroactive tags.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

# Reuse the entire CARRY helper surface (which itself re-exports regrain/specter/grain glue).
from infra.aws_pg.jobs._carry_shared import (  # noqa: F401
    N_ITEMS_TASK,
    N_STREAM,
    STIFF_METRIC_KEY,
    WIRED_TASKS,
    StageTimer,
    assert_gpu_seconds,
    batch_manifest,
    emit_family,
    fetch,
    install_pg,
    load_gate0_items,
    load_promoted_spikes,
    load_q_bank,
    load_roster,
    pinned_pg1_sha,
    rows_for_banked,
    safe_log1p_dppl,
    safe_ppl_gate,
)
from infra.aws_pg.jobs._grain_shared import S3_BUCKET

# ---- parent artifacts (read-only; every load-bearing number RECOMPUTED from these) ----------
CARRY_SCOPE = f"{S3_BUCKET}/results/carry_scope/out"
CARRY_GATE0 = f"{S3_BUCKET}/results/carry_gate0/out"
CARRY_C_ASSIGN = f"{S3_BUCKET}/results/carry_c_assign/out"
CARRY_C_SEALED = f"{S3_BUCKET}/results/carry_c_sealed/out"
CARRY_E = f"{S3_BUCKET}/results/carry_e/out"
CARRY_N_AB = f"{S3_BUCKET}/results/carry_n_ab/out"
CARRY_N_CD = f"{S3_BUCKET}/results/carry_n_cd/out"
CARRY_B2 = f"{S3_BUCKET}/results/carry_b2/out"
SPECTER_H = f"{S3_BUCKET}/results/specter_h/out"
SPECTER_S = f"{S3_BUCKET}/results/specter_s/out"
SPECTER_O = f"{S3_BUCKET}/results/specter_o/out"
BISPEC_SIGMA = f"{S3_BUCKET}/results/bispec_sigma/out"
BISPEC_COHALIGN = f"{S3_BUCKET}/results/bispec_cohalign/out"
GRAIN_DICT_L0 = f"{S3_BUCKET}/results/grain_t2/out/train_primary/dict_primary_m1_s0.pt"  # m=1 L0

HANDLE_GATE0 = f"{S3_BUCKET}/results/handle_gate0/out"          # HANDLE's own handle|-salted items

_JOBS_DIR = pathlib.Path(__file__).resolve().parent
SPEC_FROZEN = _JOBS_DIR / "handle_spec_frozen.md"
FREEZE_MANIFEST = _JOBS_DIR / "handle_freeze.json"

N_PPL_SEQ = 120


def assert_frozen(arm):
    """Driver half of freeze-or-refuse (the _carry_shared.assert_frozen contract, HANDLE tags)."""
    assert FREEZE_MANIFEST.exists(), f"no freeze manifest — arm '{arm}' refuses to launch"
    man = json.loads(FREEZE_MANIFEST.read_text())
    assert arm in man, f"arm '{arm}' has no freeze entry — refuses to launch (no retroactive tags)"
    assert SPEC_FROZEN.exists(), "frozen spec copy missing from the packaged tree"
    h = hashlib.sha256(SPEC_FROZEN.read_bytes()).hexdigest()
    assert h == man[arm]["spec_sha256"], (
        f"spec hash mismatch for arm '{arm}': tree={h[:12]} manifest={man[arm]['spec_sha256'][:12]}")
    return man[arm]


def handle_split(content_hashes, name="handle"):
    """Fresh `handle|` salt (spec §2/§4.1) — pure sha1 bucketing on item content hashes. Even
    bucket = discovery, odd = sealed. Gate-0 audits zero overlap of the odd (sealed) items with
    any prior program's sealed exposures on the same task (spec §4.1)."""
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"handle|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


# ---------------------------------------------------------------------------
# CARRY FT roster RECOMPUTE (spec §0.5-3) — mechanical, no discretion.
# ---------------------------------------------------------------------------
def load_carry_fts(workdir):
    """Recompute the 6 discovery / 12 scored-sealed FT lists and each FT's assigned gain task
    from carry_c_assign (the assignment battery + frozen discovery/sealed pools) and
    carry_c_sealed (the sealed cells' `range_ok` flags). Returns:
      {anat, base_sha, shas, discovery(6), sealed_pool(18), sealed_scored(≈12),
       assigned_task:{slug:task}, repo_of:{slug:repo}}.
    The 'scored' set = the sealed-pool FTs whose CARRY sealed full-write cell passed the range
    guard (range_ok). HANDLE re-checks its OWN range guard under the handle| salt at run time
    (§2); this recompute only fixes WHICH FTs enter the population."""
    wd = pathlib.Path(workdir)
    anat = load_roster(str(wd))                                   # 24 repos + shas + base_sha
    assign = json.loads(fetch(f"{CARRY_C_ASSIGN}/metrics.json", wd / "carry_assign.json").read_text())
    sealed = json.loads(fetch(f"{CARRY_C_SEALED}/metrics.json", wd / "carry_sealed.json").read_text())
    assert assign.get("green") and sealed.get("green"), "a CARRY Arm-C phase is not green"
    disc = list(assign["discovery_fts"])
    pool = list(assign["sealed_fts"])
    assert len(disc) == 6, f"CARRY discovery pool has {len(disc)} FTs, want 6"
    scored = sorted(s for s in pool if sealed["cells"].get(s, {}).get("range_ok"))
    assigned = {s: assign["assign"][s]["assigned_task"] for s in disc + pool}
    repo_of = {}
    for repo in anat["roster"]:
        slug = repo.split("-ft-")[1].rsplit("-", 1)[0]
        repo_of[slug] = repo
    return {"anat": anat, "base_sha": anat["base_sha"], "shas": anat["shas"],
            "discovery": disc, "sealed_pool": pool, "sealed_scored": scored,
            "assigned_task": assigned, "repo_of": repo_of,
            "n_sealed_scored": len(scored)}


def load_handle_items(workdir, task_name, half, n=N_ITEMS_TASK):
    """The Gate-0-banked handle| halves for one task: returns (hashes, n_stream). Mirrors
    _carry_shared.load_gate0_items but reads HANDLE's own item bank."""
    it = json.loads(fetch(f"{HANDLE_GATE0}/items_handle_{task_name}.json",
                          pathlib.Path(workdir) / f"items_handle_{task_name}.json").read_text())
    sealed = set(it["sealed_idx"])
    pick = [h for i, h in enumerate(it["hashes"]) if (i in sealed) == (half == "sealed")]
    assert len(pick) >= n, f"{task_name} {half} half has {len(pick)} < {n}"
    return pick[:n], len(it["hashes"])


def ft_decoder_sd(repo, sha):
    """The FT language-model state-dict on CPU (one snapshot at a time; drop after use). Reused
    load path — PaliGemma-1 pt-224, bf16, eager (the inherit_pg1 contract)."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    from infra.pg.entities import resolve_submodule_paths
    ft = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, revision=sha, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager").eval()
    sd = {k: v.detach().clone()
          for k, v in ft.get_submodule(resolve_submodule_paths(ft)["language_model"])
          .state_dict().items()}
    del ft
    return sd


def drop_snapshot(repo):
    """Delete the HF cache entry for `repo` (one-snapshot-on-disk protocol; Arm-W lesson)."""
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    _delete_cache(repo)


# ---------------------------------------------------------------------------
# The ladder-basis builder — the ONE place a rung's per-(L,t) basis dict is made.
# ---------------------------------------------------------------------------
def rung_bases(U, grid):
    """{(L, mtype): U} for every grid pair — the swap-shell contract (a single global d×k
    activation-space basis applied on each matrix's residual-facing side, exactly as CARRY's
    stiff/flat cells). `U` = the (d,k) core / band / band^stiff slice for this rung."""
    return {lt: U for lt in grid}
