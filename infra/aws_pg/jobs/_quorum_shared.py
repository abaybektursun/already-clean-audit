"""QUORUM shared job glue — the capability bank loader (CARRY's FT repos), the multimodal task
readout, and the necessary-rank rank-sweep driver.

The instrument is CLEAVE's PROTECT swap-shell, swept over REMOVAL RANK instead of α (A-QUORUM):
per capability c (a CARRY fine-tune), install the full ΔW → `gain_full`; then at each ladder rank r,
install `ΔW_off(r) = ΔW − proj_{U_policy[:, :r]} ΔW` (`build_decoder("residual", …)`) → `gain_off(r)`;
retention `R(r) = gain_off(r)/gain_full`; n* = the first r where R < ρ_break.

Readout = the capability's OWN assigned multimodal task (A-QUORUM-3), scored by `inherit_pg1.eval_pg1`
+ `carry_core.paired_gain` on identical items (streamed once per task, base score cached). NOT MMLU.

Bases from CLEAVE's `cleave_banks2` (A-QUORUM-2): energy = Σₓ eigenbasis at all 18 layers (R1-exempt),
Q = restricted to its certified support.

Heavy imports inside functions so the module py_compiles on the control plane. No try/except (CLAUDE.md).
"""
from __future__ import annotations

from infra.aws_pg.jobs._cleave_shared import (  # noqa: F401  (swap-shell reused verbatim)
    D,
    N_LAYERS,
    N_PPL_BATTERY,
    S3_BUCKET,
    SEED,
    assert_grid,
    decoder_sd,
    fetch,
    fetch_s3,
    install_pg,
    lm_bundle,
    load_model,
    make_installer,
    reset_model,
    snapshot_base,
)

# the CARRY corpus anatomy (roster + pinned SHAs) and the assignment (gain per donor + assigned task)
COMWRITE_ANATOMY = f"{S3_BUCKET}/results/comwrite_a_anatomy2/out/metrics.json"
CARRY_ASSIGN = f"{S3_BUCKET}/results/carry_c_assign/out/metrics.json"
N_ITEMS = 128                 # eval items per task (streamed once; base + every cell scored identically)


# ===========================================================================
# The capability bank — CARRY's scoring FT repos + their assigned tasks.
# ===========================================================================
def _slug_of(repo):
    """'google/paligemma-3b-ft-ai2d-224' → 'ai2d' (CARRY's slug convention)."""
    return repo.split("/")[-1].replace("paligemma-3b-ft-", "").replace("-224", "")


def load_capability_bank(workdir="/tmp/quorum"):
    """The scoring capability bank: every CARRY donor with `range_ok=True` (measurable positive
    swap-shell gain), joined to its pinned SHA and assigned multimodal task. Returns a list of
    {repo, sha, slug, task, carry_gain}, deterministically ordered (largest carry-gain first).

    A-QUORUM-1: the spec's "12" was CARRY's sealed-half count; the eligible pool is the 18 `range_ok`
    donors, each re-screened by QUORUM's own MIN_GAIN_FULL downstream."""
    import json
    import pathlib
    anat = json.loads(pathlib.Path(fetch(COMWRITE_ANATOMY, pathlib.Path(workdir) / "anatomy.json")).read_text())
    assign = json.loads(pathlib.Path(fetch(CARRY_ASSIGN, pathlib.Path(workdir) / "assign.json")).read_text())
    roster, shas = anat["roster"], anat["shas"]
    a = assign["assign"]
    bank = []
    for repo in roster:
        slug = _slug_of(repo)
        row = a.get(slug)
        if row is None or not row.get("range_ok"):
            continue
        t = row["assigned_task"]
        bank.append({"repo": repo, "sha": shas[repo], "slug": slug, "task": t,
                     "carry_gain": float(row[t]["gain"])})
    bank.sort(key=lambda c: -c["carry_gain"])
    assert bank, "no scoring capabilities in the CARRY assignment"
    return bank, anat.get("base_sha")


def hash_split(bank, salt="quorum_split", n_discovery=4):
    """Frozen slug-hash split into discovery / confirmation (spec §4). Discovery fixes ρ_break, the
    ladder, and the isotonic knots; confirmation is sealed. Deterministic; discovery = the
    `n_discovery` donors with the smallest hash (never a behavioural criterion)."""
    import hashlib
    keyed = sorted(bank, key=lambda c: hashlib.sha1(f"{salt}|{c['slug']}".encode()).hexdigest())
    disc = keyed[:n_discovery]
    conf = keyed[n_discovery:]
    return {"discovery": disc, "confirm": conf}


def ft_decoder_sd(repo, sha, base_sha, device=None, max_attempts=4):
    """Download a CARRY fine-tune and return its LANGUAGE-MODEL decoder state-dict (the ΔW carrier),
    on `device`. The CARRY `ft_decoder_sd` pattern: eager, bf16, cpu load, then the language_model
    submodule. HF_HUB_DISABLE_XET is set by the job (`reference_hf_xet_wedge`).

    A ~5GB checkpoint download over HF occasionally drops mid-stream (`RemoteProtocolError`). This is a
    TRANSIENT network fault, not a bug, so it is retried (the partial file resumes from the HF cache) —
    but only a bounded number of times, after which it re-RAISES (fail-loud; a persistent failure is a
    real error, never hidden — CLAUDE.md). No try/except swallows anything: the last attempt's
    exception propagates."""
    import time

    import torch
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg.entities import resolve_submodule_paths
    ft = None
    for attempt in range(1, max_attempts + 1):
        try:
            ft = PaliGemmaForConditionalGeneration.from_pretrained(
                repo, revision=sha, torch_dtype=torch.bfloat16, device_map="cpu",
                attn_implementation="eager").eval()
            break
        except Exception as e:                       # transient network only; re-raise on the last try
            if attempt == max_attempts:
                raise
            print(f"[ft_load] {repo} attempt {attempt}/{max_attempts} failed ({type(e).__name__}); "
                  f"retrying (resume from cache)", flush=True)
            time.sleep(5 * attempt)
    lm = ft.get_submodule(resolve_submodule_paths(ft)["language_model"])
    dev = device or "cpu"
    sd = {k: v.detach().to(dev).clone() for k, v in lm.state_dict().items()}
    del ft
    return sd


def drop_ft_cache(repo):
    """Free the downloaded checkpoint from disk (CARRY's between-cell hygiene — 18 × 6GB otherwise)."""
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    _delete_cache(repo)


# ===========================================================================
# The multimodal task readout.
# ===========================================================================
def build_task_map():
    """{name: BTask} for the 4 CARRY WIRED_TASKS (textvqa/coco_captions/vqav2/refcoco)."""
    from infra.pg import projector_ablation as PA
    return {t.name: t for t in PA.build_tasks()}


def task_rows(task, n=N_ITEMS):
    """`n` eval rows for a task, streamed deterministically (first-n) — the SAME rows for base and
    every installed cell, so the paired gain is valid."""
    from infra.pg import projector_ablation as PA
    return PA.load_eval_items(task, n)


def score(model, proc, task, rows, device):
    """eval_pg1 on `rows` → {score, per_item, ci95}. Eager forward."""
    from infra.pg import inherit_pg1 as IP
    return IP.eval_pg1(model, proc, task, rows, device)


def gain_of(cell_res, base_res):
    """{gain, se} vs the cached base result — paired-item bootstrap for per-item tasks, the CARRY
    conservative combination for corpus (COCO) tasks (`carry_core.paired_gain` / ci-combine)."""
    from infra.pg import carry_core as CC
    if cell_res.get("per_item") is not None and base_res.get("per_item") is not None:
        return CC.paired_gain(cell_res["per_item"], base_res["per_item"])
    se = ((cell_res["ci95"][1] - cell_res["ci95"][0]) ** 2
          + (base_res["ci95"][1] - base_res["ci95"][0]) ** 2) ** 0.5 / (2 * 1.96)
    return {"gain": cell_res["score"] - base_res["score"], "se": se}


# ===========================================================================
# The rank-sweep driver — the necessary-rank instrument.
# ===========================================================================
def energy_bases(banks, rank, seed, device, n_layers=N_LAYERS):
    """`build_decoder` bases removing the top-`rank` ENERGY subspace at ALL 18 layers (R1-exempt).
    Returns (bases, excluded=())."""
    from infra.pg import cleave_core as CL
    return CL.off_bases(banks, "energy", int(rank), seed=seed, n_layers=n_layers, device=device), ()


def policy_bases(banks, policy, rank, seed, device, support_layers=None, n_layers=N_LAYERS):
    """bases + excluded for a removal at `rank` on `policy`. Energy/random/shuffle span all 18 layers;
    a Q ('susceptibility'/'susceptibility_shuffle') removal is restricted to `support_layers` (the
    certified layers AT THIS RANK — a list) via `off_bases_restricted` — the R1 caveat made mechanical.
    Returns (None, None) if a Q rank has no certified support (the caller logs+skips)."""
    from infra.pg import cleave_core as CL
    if policy in ("susceptibility", "susceptibility_shuffle"):
        if not support_layers:
            return None, None
        return CL.off_bases_restricted(banks, policy, int(rank), support_layers, seed=seed,
                                       n_layers=n_layers, device=device)
    return CL.off_bases(banks, policy, int(rank), seed=seed, n_layers=n_layers, device=device), ()


def fold_adapter_decoder_sd(lm, device=None):
    """Fold every attached rank-r adapter into a plain decoder state-dict: for each wrapped module,
    `W_eff = W_base + scale·(A @ B)` (relearn.make_adapter's forward is `base(x)+scale·(x@Bᵀ)@Aᵀ`, i.e.
    an effective weight `W + scale·A@B`). Non-adapter keys pass through as the frozen base weight.
    Returns a state-dict keyed IDENTICALLY to `decoder_sd(lm)` (so build_decoder can diff it)."""
    import torch
    dev = device or "cpu"
    out = {}
    modmap = dict(lm.named_modules())
    for k, v in lm.state_dict().items():
        # a wrapped module's weight lives at '<mod>.base.weight'; the adapter factors at '<mod>.A/.B'
        if k.endswith(".base.weight"):
            mod_name = k[: -len(".base.weight")]
            mod = modmap.get(mod_name)
            if mod is not None and hasattr(mod, "A") and hasattr(mod, "B"):
                W = mod.base.weight.detach().to(torch.float32)
                delta = float(mod.scale) * (mod.A.detach().to(torch.float32) @ mod.B.detach().to(torch.float32))
                out[f"{mod_name}.weight"] = (W + delta).to(mod.base.weight.dtype).to(dev)
                continue
        # skip the ADAPTER's own params + the '.base.' re-key (handled above); keep everything else
        if ".base.weight" in k or k.rsplit(".", 1)[-1] in ("A", "B"):
            continue
        out[k] = v.detach().to(dev).clone()
    return out


def install_full_delta(base_lm_sd, ft_lm_sd, banks, *, install, read, reset, seed, device):
    """Install the FULL ΔW ('ft' cell, through the SAME build_decoder pipeline every removal cell
    uses) and read → the retention denominator `gain_full`. Returns (res, total_fro2). The 'ft' cell
    ignores the projector, but build_decoder requires a bases grid; pass a rank-5 energy grid."""
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    reset()
    bases = CL.off_bases(banks, "energy", 5, seed=seed, n_layers=N_LAYERS, device=device)
    cell_sd, book = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases, n_layers=N_LAYERS)
    install(cell_sd, "full_delta")
    del cell_sd, bases
    return read("full_delta"), book["total_fro2"]


def rank_sweep(base_lm_sd, ft_lm_sd, banks, *, policy, ladder, install, read, reset, gain_full,
               seed, device, support=None, log=print):
    """Sweep removal of the top-r `policy` subspace across `ladder`; return per-rank
    [{policy, rank, retention, gain_off, removed_energy_fraction, n_support_layers}].

    R(r) = gain_off(r) / gain_full. At r=d the energy off-projection annihilates ΔW (retention 0),
    so the curve reaches 0 — the necessary-rank crossing is guaranteed for the energy arm (A-QUORUM-4).

    `support` (Q arms only) = {rank: [certified layers]} — the per-rank support is looked up here so
    the removal projects only where U_sus is certified at THAT rank.
    """
    from infra.pg import comwrite_surgery as SURG
    is_q = policy in ("susceptibility", "susceptibility_shuffle")
    rows = []
    for r in ladder:
        reset()
        layers_r = (support.get(int(r)) if isinstance(support, dict) else support) if is_q else None
        bases, excluded = policy_bases(banks, policy, r, seed, device, support_layers=layers_r)
        if bases is None:
            log(f"[sweep] {policy:>22s} r={r:<5d} SKIP — no certified support at this rank")
            continue
        cell_sd, book = SURG.build_decoder("residual", base_lm_sd, ft_lm_sd, bases,
                                           excluded=excluded, n_layers=N_LAYERS)
        install(cell_sd, f"{policy}/r{r}")
        del cell_sd, bases
        res = read(f"{policy}/r{r}")
        removed = (book["total_captured_fro2"] / book["total_fro2"]) if book["total_fro2"] > 0 else 0.0
        retention = float(res["gain"] / gain_full) if abs(gain_full) > 1e-9 else 0.0
        rows.append({"policy": policy, "rank": int(r), "retention": retention,
                     "gain_off": float(res["gain"]), "removed_energy_fraction": removed,
                     "n_support_layers": len(layers_r) if (is_q and layers_r) else N_LAYERS})
        log(f"[sweep] {policy:>22s} r={r:<5d} retention={retention:+.3f} gain_off={res['gain']:+.4f} "
            f"removedE={removed:.3f}")
    return rows
