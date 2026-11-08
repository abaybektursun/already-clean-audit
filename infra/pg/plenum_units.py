"""PLENUM — the ONE loader every unit enters the battery through (spec §1.1, §4).

The map's value is uniformity: any two rows of the grid must differ ONLY in the unit (§3.2). That
is a property of this module — if each unit reached the battery by its own path, a difference
between two rows could always be a difference between two harnesses.

So there is exactly one shell (the PG1 VLM architecture, eager, bf16) and every unit is installed
into it through `lacuna_shell.install_canon`:

  pg1              native PG1                                     (CEILING)
  ft-<slug>        the fine-tune's own full VLM state dict         (native weights, no surgery)
  pi0 / pi05       policy VLM half re-shelled                      (LACUNA R4/R5)
  u3 / u4 / u3c    banked policy units, same re-shell path
  stitch0          SigLIP release + random projector + gemma-2b    (LACUNA R2)
  matched_dw       PG1 LM + Frobenius-matched Gaussian ΔW          (the FLOOR)
  rand_dec         PG1 LM → std-matched Gaussian                   (scale zero)
  gemma-2b         text-only forward, no image path at all         (LACUNA R1)

A fine-tune "loads natively" in the §2.1 sense: its OWN weights fill every VLM slot, with no
compartment surgery and no borrowed tensors. Installing them into the shared shell rather than
instantiating 24 separate models is what keeps the harness identical across rows (and keeps 24 ×
12 GB off the disk at once — each snapshot is deleted after use).

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

from infra.pg.plenum_map import FT_REPO, FT_SLUGS

PG1_REPO = "google/paligemma-3b-pt-224"
# Banked LACUNA Arm-3 F0 units live on S3 (`pg` ships only infra/, so repo artifacts are read from
# their S3 copies — §4). u3c is written by PLENUM's own co-training job.
S3_UNITS = "s3://pg-lineage-769063704696/weights/lacuna_units"
POLICY_UNITS = {"pi0": "hf:lerobot/pi0_base", "pi05": "hf:lerobot/pi05_base",
                "u3": f"{S3_UNITS}/U3", "u4": f"{S3_UNITS}/U4",
                "u3c": f"{S3_UNITS}/U3c"}
# §4 key-count gate: an FT is a fine-tune of PG1, so its VLM key inventory must match PG1's exactly.
# A repo that does not is NOT a member of the population and must fail loudly rather than be
# quietly evaluated as if it were.
FT_EXPECT_KEYS = 603


def ft_units():
    return [f"ft-{s}" for s in FT_SLUGS]


def _delete_cache(repo_id):
    """Free the snapshot after use — 24 × ~12 GB does not fit (invariant_core pattern, REUSED)."""
    import os
    import shutil
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    d = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d)          # no try/except — a cleanup failure must fail loud


def ft_state_dict(slug):
    """{canonical_key: tensor} of one fine-tune's full VLM + its resolved sha.

    Read straight from safetensors (never `from_pretrained`) so the tensors land on CPU in their
    stored dtype and the model class is never instantiated a second time."""
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon_map
    repo = FT_REPO.format(slug=slug)
    sd = {}
    for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
        sd.update(load_file(hf_hub_download(repo, f)))
    sha = HfApi().model_info(repo, revision="main").sha
    c = canon_map(sd)
    assert len(c) == FT_EXPECT_KEYS, (
        f"{repo}: {len(c)} VLM keys ≠ PG1's {FT_EXPECT_KEYS} — this repo is not a same-architecture "
        f"fine-tune of PG1 and must not enter the grid as one")
    return c, sha, repo


def s3_policy_sd(uri, workdir):
    """A banked lerobot unit's state dict from S3 (u3 / u4 / u3c), with its sha256 asserted."""
    import hashlib
    import pathlib
    import subprocess
    from safetensors.torch import load_file
    d = pathlib.Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", uri, str(d)], check=True)
    shards = sorted(d.glob("*.safetensors"))
    assert shards, f"no safetensors under {uri}"
    sd, digests = {}, {}
    for p in shards:
        sd.update(load_file(str(p)))
        digests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return sd, digests


def install_unit(uid, model, proc, pg1_sd, device, seeds, workdir="/tmp/plenum_units"):
    """Put `uid` into the shared shell and return the install manifest.

    The shell is ALWAYS reset to native PG1 first, so no row can inherit another's weights (the
    CLEAVE reset contract, inherited via `lacuna_shell.restore_pg1`).
    """
    from infra.pg import lacuna_shell as LS
    LS.restore_pg1(model, pg1_sd)
    if uid == "pg1":
        return {"unit": uid, "kind": "native", "note": "PG1 in shell (CEILING)"}
    if uid.startswith("ft-"):
        slug = uid[len("ft-"):]
        sd, sha, repo = ft_state_dict(slug)
        # The FT's OWN weights must fill every shell slot — nothing may remain PG1's, or the row
        # is a hybrid rather than the fine-tune.
        #
        # `lm_head.weight` is the ONE legitimate exception, and it is not a loophole: PaliGemma
        # TIES the head to `embed_tokens`, so a tied checkpoint does not ship a separate head
        # tensor (this is why `lacuna_shell.build_row_policy` lists it in `allow_missing`). An
        # earlier version of this guard demanded an explicit write for it and failed the whole
        # grid on ft-refcoco-seg.
        #
        # The exception is discharged by PROOF rather than by assumption: after `tie_weights()`
        # the head must share storage with the embedding we just wrote, and its values must equal
        # the fine-tune's own table. If the tie silently did not happen, the head would still hold
        # PG1's values and the row WOULD be the hybrid this guard exists to catch.
        info = LS.install_canon(model, sd, f"ft/{slug}", allow_missing=("lm_head",))
        model.tie_weights()
        assert info["n_untouched"] == 0, (
            f"{repo}: {info['n_untouched']} shell tensors kept PG1 values "
            f"({info['untouched_examples']}) — this row would be a hybrid, not the fine-tune")
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings()
        assert head.weight.data_ptr() == emb.weight.data_ptr(), (
            f"{repo}: lm_head is NOT tied to embed_tokens after install — the head still holds "
            f"PG1's values and this row is a hybrid")
        ft_emb = sd.get("language_model.embed_tokens.weight")
        assert ft_emb is not None, f"{repo}: no embed_tokens in the fine-tune's own weights"
        import torch
        # Compare in the SHELL's dtype, not the checkpoint's. `install_canon` casts on write (the
        # shell is bf16), so casting the shell's tensor BACK to the checkpoint dtype and demanding
        # bit-equality would fail on a correct install — it asks for precision the write never
        # promised. Casting the FT's table forward is exact: it is precisely what was written.
        assert torch.equal(emb.weight.detach().cpu(), ft_emb.to(emb.weight.dtype)), (
            f"{repo}: the installed embedding does not equal the fine-tune's own table "
            f"(compared in the shell dtype {emb.weight.dtype})")
        info["lm_head"] = "tied to embed_tokens (verified by storage identity + value equality)"
        _delete_cache(repo)
        return {"unit": uid, "kind": "native_finetune", "repo": repo, "sha": sha, **info}
    # THE FILM (§2.3: "the full Tier-1 battery at EVERY checkpoint"). A checkpoint is not a map
    # unit, so it has no POLICY_UNITS entry — but it is exactly a policy, and the film's whole
    # validity rests on the checkpoint battery being the SAME battery the grid rows were measured
    # with. Rather than write a second battery inside a film job (which would let the film and the
    # grid drift apart silently), a checkpoint enters through THIS loader like everything else:
    #
    #   plenum_fc --units ckpt:s3://.../plenum_film/U3/seed0/00000512,ckpt:...
    #
    # so `plenum_fc`, `plenum_text`, `plenum_census` and `plenum_control` all film for free.
    if uid.startswith("ckpt:"):
        uri = uid[len("ckpt:"):]
        assert uri.startswith("s3://"), f"{uid}: film checkpoints are read from S3"
        import hashlib
        wd = f"{workdir}/ckpt_{hashlib.sha256(uri.encode()).hexdigest()[:12]}"
        full, digests = s3_policy_sd(uri, wd)
        if not any(k.startswith(LS.PALI_PREFIX) for k in full):
            full = {LS.canon(k): v for k, v in full.items()}
            assert any(k.startswith(LS.PALI_PREFIX) for k in full), (
                f"{uid}: no {LS.PALI_PREFIX!r} compartment even after canonicalisation")
        for k in [x for x in full
                  if x.endswith("language_model.embed_tokens.weight") and LS.PALI_PREFIX in x]:
            del full[k]
        info = LS.build_row_policy(model, "pi0", embed="own", full_sd=full)
        return {"unit": uid, "kind": "film_checkpoint", "source": uri,
                "shard_sha256": digests, **info}
    if uid in POLICY_UNITS:
        src = POLICY_UNITS[uid]
        if src.startswith("hf:"):
            full, sha = LS.policy_full_sd(uid)
            digests = None
        else:
            full, digests = s3_policy_sd(src, f"{workdir}/{uid}")
            sha = None
            # A banked unit was written by `policy.save_pretrained`, which nests the compartments
            # under `model.` — so `policy_vlm_sd`'s exact-prefix match on
            # `paligemma_with_expert.paligemma.` finds nothing. Drop the `model.` path COMPONENT
            # (never the substring: `vision_model` must survive) exactly as `lacuna_shell.canon`
            # does, so a banked unit and an HF policy present the same key namespace.
            if not any(k.startswith(LS.PALI_PREFIX) for k in full):
                full = {LS.canon(k): v for k, v in full.items()}
                assert any(k.startswith(LS.PALI_PREFIX) for k in full), (
                    f"{uid}: no {LS.PALI_PREFIX!r} compartment even after canonicalisation; "
                    f"sample keys {sorted(full)[:4]}")
            # A banked unit was written from a LIVE `PI0Policy`, which carries BOTH `lm_head.weight`
            # and a tied `language_model.embed_tokens.weight`; the published π0 safetensors drop the
            # latter under the openpi tied-embedding convention (`lacuna_units` documents this).
            # So a banked unit presents 164 LM keys where `policy_vlm_sd` expects 163. Drop the tied
            # duplicate — NOT a truncation: `build_row_policy` installs the table from `lm_head`
            # (the `own_head` path), so the embedding still arrives, once.
            emb_dupes = [k for k in full
                         if k.endswith("language_model.embed_tokens.weight")
                         and LS.PALI_PREFIX in k]
            for k in emb_dupes:
                del full[k]
            if emb_dupes:
                digests = {**(digests or {}), "dropped_tied_embed": len(emb_dupes)}
        name_for_counts = "pi05" if uid in ("pi05",) else "pi0"
        info = LS.build_row_policy(model, name_for_counts, embed="own", full_sd=full)
        return {"unit": uid, "kind": "policy_reshell", "source": src, "sha": sha,
                "shard_sha256": digests, **info}
    if uid == "stitch0":
        return {"unit": uid, "kind": "stitch",
                **LS.build_row_stitch0(model, seeds["stitch"], pg1_sd)}
    if uid == "matched_dw":
        full, _ = LS.policy_full_sd("pi0")
        lm, _h, _c = LS.policy_vlm_sd("pi0", full_sd=full)
        target = {k[len("language_model."):]: v for k, v in lm.items()
                  if k.startswith("language_model.")}
        del full
        return {"unit": uid, "kind": "floor",
                **LS.build_row_matched_dw(model, pg1_sd, target, seeds["matched_dw"])}
    if uid == "rand_dec":
        return {"unit": uid, "kind": "scale_zero",
                **LS.build_row_random_decoder(model, pg1_sd, seeds["rand_dec"])}
    raise RuntimeError(f"unknown unit {uid!r}")


def unit_order(uids):
    """Group units so each heavy checkpoint is fetched ONCE (the ladder's ROW_GROUPS discipline).
    Cheap shell rows first, then the FT roster, then the policies (13 GB each, one at a time)."""
    rank = {"pg1": 0, "matched_dw": 1, "rand_dec": 2, "stitch0": 3}
    def key(u):
        if u in rank:
            return (0, rank[u], u)
        if u.startswith("ft-"):
            return (1, 0, u)
        if u.startswith("ckpt:"):
            # film checkpoints last, in STEP order (the string is zero-padded, so lexical == numeric)
            return (3, 0, u)
        return (2, 0, u)
    return sorted(uids, key=key)
