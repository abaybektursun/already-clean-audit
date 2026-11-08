"""TESSERA Arm X — crosscoder features across the lineage. Spec §1.2 X, §2 X, §5 P1/P2/N4, §6 X.

THE QUESTION. π0's behavioural floor is established. Arm X asks what it IS at feature grain:

  present-but-unread   PG1's multimodal features still live in π0's residual stream; what died is
                       the readout. The boundary theory (summaries preserved / computation
                       rewritten) predicts this.
  erased               the features are gone from the stream. This would wound the boundary theory
                       where it has been safest.

A frozen text SAE cannot tell these apart — a genuinely absent feature and a feature the SAE never
had a latent for look identical. A crosscoder trained JOINTLY on both sides' activations can,
because its dictionary is fit to both.

THE PAIRS (§2 X). Each is (before, after) with before = the parent:
  identity   (PG1, PG1)          P2 — the only calibration that can fail. Same model both sides, so
                                 any "emergent" fraction it reports is manufactured by the
                                 instrument, and it is what every other pair's band is read against.
  install    (PG1⟨gemma-2b dec⟩, PG1)   the creation hop, run through an IDENTICAL multimodal
                                 forward on both sides — only the 18 decoder layers differ. This is
                                 the port source's own construction, and it is why the pair is
                                 paired: both sides see the same image tokens.
  pi0        (PG1, π0)           the VLA hop whose behaviour is at the floor.
  pi05       (PG1, π0.5)         the VLA hop that retains VQA.
  ft-max / ft-min                the frozen FT rule: the max- and min-mean-survival units per
                                 `plenum_agg_fc`, computed mechanically before any harvest.

HARVEST. Both sides are installed into the SAME shell in turn and forwarded over the SAME batches,
so token i is the same token on both sides — row-aligned by construction rather than by alignment
code. The input ids are hashed on each pass and the hashes asserted equal; a pair whose two passes
did not see identical inputs is not paired, and the assert says so rather than producing a subtly
scrambled dictionary.

Only DISCOVERY-half tokens train the crosscoder (`held` mask). Dropping that mask costs an
out-of-sample claim and is called out in the infra notes as a recurring critic finding.

GATES. P1 (synthetic planted pair) runs in `tessera_x_p1`; P2 is the identity pair here, first in
the order; N4 patches each side's crosscoder reconstruction back into that side's forward and
requires the battery score to land inside the item-bootstrap 95% CI of the unpatched score
(amendment A7) — FVU health is not identity, which is the PLENUM feature lesson.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_x --args "--pairs identity,pi0" \
      --type g6e.2xlarge,p5.4xlarge --s3-prefix results/tessera_x_a --budget-min 900

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

SITES = (9, 14)                 # spec §2 X
GEMMA1_REPO = "google/gemma-2b"
PAIR_ORDER = ("identity", "install", "pi05", "pi0", "ft-max", "ft-min", "sibling")
XC_SEEDS_DEFAULT = (0, 1)


def _read_json(p):
    import subprocess
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def ft_pair_units(agg_fc):
    """The frozen FT rule: max- and min-MEAN-survival units per `plenum_agg_fc`.

    Mechanical and computed from banked labels, so it is fixed before any harvest and cannot be
    steered by what the crosscoder shows.
    """
    law_readout = agg_fc["law_readout_per_faculty"]
    by_unit = {}
    for c in agg_fc["cells"]:
        if law_readout.get(c["faculty"]) == c["readout"] and str(c["unit"]).startswith("ft-"):
            by_unit.setdefault(c["unit"], []).append(float(c["S"]))
    means = {u: sum(v) / len(v) for u, v in by_unit.items() if v}
    assert len(means) >= 2, f"need ≥2 FT units with survival scores, got {len(means)}"
    hi = max(means, key=lambda u: means[u])
    lo = min(means, key=lambda u: means[u])
    return {"ft-max": hi, "ft-min": lo, "mean_survival": means}


def swap_decoder_to_gemma1(model, g1_sd):
    """Reset the shell's 18 decoder layers + final norm to base gemma-2b, KEEPING the vision tower,
    projector and the extended-vocab embedding — `contrast_a1._lm_swap_state_dict`'s semantics.

    This is what makes the install pair PAIRED: both sides run the identical multimodal forward
    over the identical image tokens, and only the decoder weights differ.
    """
    import torch
    from infra.pg import inherit_pg1 as IP
    lm = IP.language_model(model) if hasattr(IP, "language_model") else None
    layers = IP.decoder_layers(model)
    n_written, missing = 0, []
    for i, blk in enumerate(layers):
        for name, p in blk.named_parameters():
            key = f"model.layers.{i}.{name}"
            alt = f"layers.{i}.{name}"
            src = g1_sd.get(key, g1_sd.get(alt))
            if src is None:
                missing.append(key)
                continue
            assert tuple(src.shape) == tuple(p.shape), (
                f"gemma-2b {key} shape {tuple(src.shape)} ≠ shell {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(src.to(p.dtype))
            n_written += 1
    assert not missing, f"decoder swap missing {len(missing)} keys, e.g. {missing[:4]}"
    assert n_written > 0, "decoder swap wrote nothing"
    return {"n_written": n_written, "lm": str(type(lm))}


def harvest_sites(model, layers, mm_batch, sites, device, micro=4):
    """{site: (n_kept, d) fp32 CPU acts} plus the input hash of this pass.

    The hash is the pairing guarantee: two passes that produce the same hash saw the same tokens in
    the same order, so row i of one side is row i of the other.
    """
    import torch
    from infra.pg.gemma_scope import ResidualHarvest
    ii, am, pv = mm_batch["input_ids"], mm_batch["attention_mask"], mm_batch["pixel_values"]
    keep = mm_batch["keep_mask"]
    h = hashlib.sha256(ii.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    chunks = {L: [] for L in sites}
    hs = {L: ResidualHarvest(layers, L) for L in sites}
    for L in sites:
        hs[L].__enter__()
    try:
        for s in range(0, ii.shape[0], micro):
            km = keep[s:s + micro].reshape(-1)
            with torch.inference_mode():
                model(input_ids=ii[s:s + micro], attention_mask=am[s:s + micro],
                      pixel_values=pv[s:s + micro])
            for L in sites:
                cap = hs[L].captured
                chunks[L].append(cap.reshape(-1, cap.shape[-1])[km].float().cpu())
    finally:
        for L in sites:
            hs[L].__exit__(None, None, None)
    return {L: torch.cat(chunks[L], 0) for L in sites}, h


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg-fc", required=True)
    ap.add_argument("--pairs", default=",".join(PAIR_ORDER))
    ap.add_argument("--n-items", type=int, default=512)
    # A14 — Arm X trains at the setting P1 PASSED at. The A9 port defaults (4000 / 1e-4) produce a
    # degenerate all-latents-live dictionary whose decoder norms all sit at r ~ 0.5, on which P1
    # could not recover a PLANTED 0.1111 emergent fraction. Running Arm X there would produce
    # emergent ~ 0 on every lineage pair for instrument reasons.
    ap.add_argument("--steps", type=int, default=20000, help="A14 (supersedes A9 CC_TRAIN_STEPS)")
    ap.add_argument("--lr", type=float, default=1e-3, help="A14 (supersedes A9 CC_LR)")
    ap.add_argument("--sites", default=",".join(str(s) for s in SITES))
    # Train ONLY the shuffle-pairing leg. Exists to recover the LOST-fraction null for cells whose
    # real leg is already banked (v1 discarded shuffle `lost_fraction`), without paying to retrain
    # a real crosscoder whose numbers are already on S3. The real leg's `seeds` list comes out
    # EMPTY under this flag, which the merge must treat as "null-only", never as a measured cell.
    ap.add_argument("--shuffle-only", action="store_true",
                    help="skip the real leg; measure the shuffle null only")
    # §8/§12 density-cut ladder: X is the pre-registered cut-density arm, seeds 2->1 and sites 2->1
    # as the FLOORS. A cut is recorded in metrics so a reader sees the realised density, never
    # inferring full density from a table that does not say.
    ap.add_argument("--seeds", default=",".join(str(s) for s in XC_SEEDS_DEFAULT),
                    help="crosscoder seeds; the §8 floor is a single seed")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import plenum_units as PU
    from infra.pg import tessera_core as TC
    from infra.pg import tessera_crosscoder as XC

    device = "cuda"
    sites = [int(s) for s in a.sites.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    agg = _read_json(a.agg_fc)
    ftp = ft_pair_units(agg)

    rep = {"object": "TESSERA Arm X — crosscoder features across the lineage",
           "sites": sites, "ft_pair_rule": {"ft-max": ftp["ft-max"], "ft-min": ftp["ft-min"]},
           # Record what the TRAINER WAS ACTUALLY GIVEN (a.steps / a.lr), not the module constants.
           # v1 wrote XC.CC_TRAIN_STEPS and omitted lr entirely, so an artifact trained at the A14
           # setting reported the superseded A9 setting (4000) — the exact reading under which a
           # careful reader would discard the arm as run on the instrument P1 proved blind.
           # The measurements were unaffected (the override reaches `train_crosscoder`); only the
           # provenance record was wrong. Both port defaults are kept alongside so the supersession
           # is visible in the artifact rather than only in the spec.
           "hyperparams": {"dict_expansion": XC.DICT_EXPANSION, "k": XC.BATCHTOPK_K,
                           "steps": a.steps, "lr": a.lr, "batch": XC.CC_BATCH,
                           "seeds": None,
                           "min_paired_tokens": XC.MIN_PAIRED_TOKENS,
                           "port_defaults_superseded_by_A14": {"steps": XC.CC_TRAIN_STEPS,
                                                               "lr": XC.CC_LR}},
           "pairs": {}}
    rep["hyperparams"]["seeds"] = seeds
    rep["density_cut"] = {"seeds": len(seeds), "sites": len(sites),
                          "full_density": {"seeds": 2, "sites": 2},
                          "cut_applied": bool(len(seeds) < 2 or len(sites) < 2),
                          "rule": "§8/§12 pre-registered cut ladder: X is the density-cut arm, seeds 2→1 and sites 2→1 are the FLOORS. Recorded so a reader never infers full density from a table that does not say."}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    model, proc, layers, sha, base_ppl = SS.load_model("/tmp/tessera_x")
    rep["pinned_sha"], rep["base_ppl"] = sha, base_ppl      # P5 anchor, banked on every load
    # `restore_pg1` -> `install_canon` expects the SHELL's canonical key namespace, not
    # `model.state_dict()`'s `model.`-prefixed one. Passing the raw state dict makes every one of
    # the 603 VLM keys miss its shell counterpart and the install asserts. `pg1_baseline_sd` is the
    # constructor every PLENUM job uses for exactly this argument (REUSED).
    from infra.pg import lacuna_shell as _LS
    pg1_sd = _LS.pg1_baseline_sd(model)
    _w()

    # ---- the harvest batch: PG1's own VQA items, discovery half only -------------------------
    vqa_task, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa_task, a.n_items)
    # sha1-bucketed discovery/confirmation split — the confirmation half is never harvested, so no
    # instrument in this arm has seen the data the behavioural claims rest on
    disc = [i for i, h in enumerate(hashes)
            if int(hashlib.sha1(f"tessera_x|{h}".encode()).hexdigest(), 16) % 4 != 0]
    rows = [rows[i] for i in disc]
    rep["n_items_discovery"] = len(rows)
    print(f"[X] harvest items: {len(rows)} discovery of {len(hashes)}", flush=True)

    imgs = [vqa_task.image_fn(r).convert("RGB") for r in rows]
    prompts = [vqa_task.prompt_fn(r) for r in rows]
    enc = proc(text=prompts, images=imgs, return_tensors="pt", padding="longest")
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["pixel_values"] = enc["pixel_values"].to(next(model.parameters()).dtype)
    img_id = model.config.image_token_index
    is_image = enc["input_ids"] == img_id
    keep = (enc["attention_mask"].bool())
    mm_batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "pixel_values": enc["pixel_values"], "keep_mask": keep}
    rep["n_kept_tokens_estimate"] = int(keep.sum())
    rep["n_image_tokens"] = int(is_image.sum())
    _w()

    def install(side):
        """Put one side into the shell. Always resets to native PG1 first (the CLEAVE contract)."""
        from infra.pg import lacuna_shell as LS
        if side == "gemma1dec":
            LS.restore_pg1(model, pg1_sd)
            from huggingface_hub import hf_hub_download, list_repo_files
            from safetensors.torch import load_file
            g1 = {}
            for f in [x for x in list_repo_files(GEMMA1_REPO) if x.endswith(".safetensors")]:
                g1.update(load_file(hf_hub_download(GEMMA1_REPO, f)))
            return {"side": side, **swap_decoder_to_gemma1(model, g1)}
        return PU.install_unit(side, model, proc, pg1_sd, device,
                               {"stitch": 0, "matched_dw": 0, "rand_dec": 0})

    pair_units = {"identity": ("pg1", "pg1"), "install": ("gemma1dec", "pg1"),
                  "pi0": ("pg1", "pi0"), "pi05": ("pg1", "pi05"),
                  "ft-max": ("pg1", ftp["ft-max"]), "ft-min": ("pg1", ftp["ft-min"]),
                  # TESSERA-FIX C1 — the SEVENTH pair: two PG1 siblings with NO descent relation
                  # between them. Every other pair here is (parent, child), so a nonzero reading
                  # has only the identity band to clear — and C0 shows that band is [0,0] because
                  # identity's live latents all sit within +/-0.02 of r=0.5 and cannot reach the
                  # 0.10/0.90 classification cuts at all. This pair is the achievable null: two
                  # real, differently-trained models whose difference is NOT descent, giving a
                  # nonzero-width reference band that Arm X's nonzero readings must actually clear.
                  "sibling": (ftp["ft-max"], ftp["ft-min"])}

    identity_band = None
    for pname in [p for p in PAIR_ORDER if p in a.pairs.split(",")]:
        before_u, after_u = pair_units[pname]
        print(f"[X] pair {pname}: before={before_u} after={after_u}", flush=True)
        acts, hsh = {}, {}
        for role, uid in (("before", before_u), ("after", after_u)):
            info = install(uid)
            acts[role], hsh[role] = harvest_sites(model, layers, mm_batch, sites, device)
            print(f"[X]   {role}={uid} installed ({str(info)[:80]}) "
                  f"hash={hsh[role]} n={acts[role][sites[0]].shape}", flush=True)
        assert hsh["before"] == hsh["after"], (
            f"{pname}: the two harvest passes saw different inputs "
            f"({hsh['before']} vs {hsh['after']}) — the pair is not paired")

        prec = {"units": {"before": before_u, "after": after_u}, "input_hash": hsh["before"],
                "sites": {}}
        for L in sites:
            ab, aa = acts["before"][L], acts["after"][L]
            assert ab.shape == aa.shape, f"{pname}@L{L}: {tuple(ab.shape)} vs {tuple(aa.shape)}"
            n_tok = ab.shape[0]
            if n_tok < XC.MIN_PAIRED_TOKENS:
                prec["sites"][L] = {"VOID": "below MIN_PAIRED_TOKENS", "n_paired_tokens": n_tok,
                                    "floor": XC.MIN_PAIRED_TOKENS}
                print(f"[X]   L{L} VOID: {n_tok} < {XC.MIN_PAIRED_TOKENS} paired tokens",
                      flush=True)
                continue
            seeds_out, shuf_out = [], []
            for seed in seeds:
                if not a.shuffle_only:
                    ae, cfg = XC.train_crosscoder(ab, aa, L, seed, device,
                                                  steps_override=a.steps, lr_override=a.lr)
                    fr = XC.pair_fractions(ae, ab, aa, device)
                    fv = XC.fvu(ae, ab, aa, device)
                    seeds_out.append({"seed": seed, "cfg": cfg, **fr, **fv})
                    print(f"[X]   L{L} s{seed} emergent={fr['emergent_fraction']:.4f} "
                          f"survival={fr['survival_fraction']:.4f} n_live={fr['n_live']} "
                          f"fvu_after={fv['fvu_after']:.3f}", flush=True)
                    del ae
                ae_s, _ = XC.train_crosscoder(ab, aa, L, seed, device,
                                              shuffle_pairing=True,
                                              steps_override=a.steps, lr_override=a.lr)
                fr_s = XC.pair_fractions(ae_s, ab, aa, device)
                # v1 stored ONLY `emergent_fraction` here and discarded the rest of `fr_s`. That
                # left `lost_fraction` -- the single non-zero quantity Arm X measures, and the one
                # the π0 erasure question turns on -- with NO matched null, while its sibling
                # `emergent_fraction` had one. Store the whole record: the shuffle leg costs the
                # same training either way, so throwing the numbers away bought nothing.
                shuf_out.append({"seed": seed, **fr_s})
                print(f"[X]   L{L} s{seed} SHUFFLE emergent={fr_s['emergent_fraction']:.4f} "
                      f"lost={fr_s['lost_fraction']:.4f} n_live={fr_s['n_live']}", flush=True)
                del ae_s
                torch.cuda.empty_cache()
            if a.shuffle_only:
                # No `seeds` key: downstream code keys "this cell was measured" off `seeds`, and an
                # empty list under that key would read as a measured cell with no seeds rather than
                # as a null-only run.
                prec["sites"][L] = {"n_paired_tokens": n_tok, "shuffle_null": shuf_out,
                                    "shuffle_null_only": True}
            else:
                prec["sites"][L] = {"n_paired_tokens": n_tok, "seeds": seeds_out,
                                    "shuffle_null": shuf_out,
                                    "emergent_seed_agree": abs(seeds_out[0]["emergent_fraction"]
                                                               - seeds_out[-1]["emergent_fraction"])}
            rep["pairs"][pname] = prec
            _w()
        rep["pairs"][pname] = prec
        _w()

        # A --shuffle-only run has no real leg by construction, so there are no fractions to build
        # a band from and nothing to label. v1 of this flag left the band block reachable, and the
        # (correct) non-empty assert then killed the run after identity and before the pairs it was
        # launched for. The flag must skip the whole labelling stage, not fail inside it.
        if pname == "identity" and not a.shuffle_only:
            def _band(field):
                fr = [s[field] for L in prec["sites"]
                      for s in prec["sites"][L].get("seeds", [])]
                assert fr, f"the identity pair produced no {field} — every pair's band is gone"
                b = TC.boot_ci(fr)
                return {"p2.5": b["lo95"], "p50": b["point"], "p97.5": b["hi95"], "n": len(fr)}
            identity_band = _band("emergent_fraction")
            # v1 built a band for `emergent_fraction` only, and scored only that quantity. LOST --
            # the quantity the erasure question actually turns on -- got no band and no verdict,
            # so it was never scored against ANY null: not the identity band (never computed) and
            # not the shuffle (computed, then discarded at storage). Both legs are restored here.
            identity_band_lost = _band("lost_fraction")
            rep["identity_band"] = identity_band
            rep["identity_band_lost"] = identity_band_lost
            _w()

    # ---- label every pair against the identity band (§6 X) ------------------------------------
    if identity_band and not a.shuffle_only:
        for pname, prec in rep["pairs"].items():
            if pname == "identity":
                continue
            for L, srec in prec.get("sites", {}).items():
                if "seeds" not in srec:
                    continue
                em = sum(s["emergent_fraction"] for s in srec["seeds"]) / len(srec["seeds"])
                sv = sum(s["survival_fraction"] for s in srec["seeds"]) / len(srec["seeds"])
                lo = sum(s["lost_fraction"] for s in srec["seeds"]) / len(srec["seeds"])
                srec["emergent_verdict"] = TC.fraction_verdict(em, identity_band)
                srec["lost_verdict"] = TC.fraction_verdict(lo, identity_band_lost)
                srec["mean_emergent"] = em
                srec["mean_survival"] = sv
                srec["mean_lost"] = lo
                # Range check, not a threshold. The identity band is the null a fraction must
                # CLEAR (fraction_verdict's own docstring); the shuffle null exists to show the
                # statistic CAN fire. W4's verdict was void precisely because its shuffle could
                # not, so a LOST verdict with no demonstrated range is recorded as such.
                shl = [z["lost_fraction"] for z in srec.get("shuffle_null", [])
                       if "lost_fraction" in z]
                srec["lost_range_verified"] = bool(shl and max(shl) > 10 * float(
                    identity_band_lost["p97.5"]) and max(shl) > 0.01)
                srec["shuffle_lost_max"] = (max(shl) if shl else None)
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TESSERA_X_OK", json.dumps({"pairs": list(rep["pairs"]),
                                      "identity_band": rep.get("identity_band")}, default=str),
          flush=True)


if __name__ == "__main__":
    main()
