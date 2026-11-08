"""PLENUM Arm 1 — the weight/geometry census, fanned out over the whole roster (spec §2.1, §13).

The §13 ledger calls this a "thin loop over `lacuna_a2_bytes`", and that is exactly what it is:
`_delta_census` is imported verbatim, including the interpretive control it already carries
(`srank_base`, `n_base_below_band`) — the N2 base-matrix control the LACUNA 2c lesson mandates.
A band that the BASE matrices also fail is not a null, so every spectral row here ships with the
base-matrix comparison attached and the aggregator withdraws any band that fails it (§5 N2).

The census also produces the law's ‖ΔW‖ predictor (§2.1): `rel_frob` of the LM component per unit,
recomputed from weights, never quoted from a prior report.

Census cells are EXPLORATORY and labelled as such (§7) — they are never in a BH family.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_census --type r7i.4xlarge --no-weights \
      --args "--units all-ft" --s3-prefix results/plenum_census --budget-min 600
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

# component -> canonical-key prefix in the PG1 VLM namespace
COMPONENTS = {
    "lm": "language_model.layers",
    "vision": "vision_tower",
    "projector": "multi_modal_projector",
    "embedding": "language_model.embed_tokens",
}


def _read_json(p):
    p = str(p)
    if p.startswith("s3://"):
        import subprocess
        return json.loads(subprocess.run(["aws", "s3", "cp", p, "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def unit_etag(uri):
    """S3 ETag of a unit's weights, recorded so a ROW is self-verifying.

    Two cells of the same unit's row must have read the same weights. Today a battery straddled a
    re-publish — `fc`/`gen` synced the previous run's weights and `text`/`ctl` synced the new ones — and
    NOTHING in the artifacts flagged it, because every job recorded the same unit *URI* while reading
    different *objects*. A URI names a location; an ETag names the bytes.
    """
    import subprocess
    if not str(uri).startswith("s3://"):
        return None
    b = str(uri)[len("s3://"):].split("/", 1)
    r = subprocess.run(["aws", "s3api", "head-object", "--bucket", b[0],
                        "--key", f"{b[1].rstrip('/')}/model.safetensors",
                        "--query", "ETag", "--output", "text"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def _sub(sd, prefix):
    return {k: v for k, v in sd.items() if k.startswith(prefix)}


def _absolute_census(sd, tag, topk=32, seed=0):
    """srank / ‖W‖ of the matrices THEMSELVES, for a component with no pre-install reference.

    Spec §2.3 defines the action-expert census as "srank/‖W‖/coupling stats per checkpoint (**no
    'before' exists** — census, exploratory)". `_delta_census` cannot express that: with no base to
    difference, every ΔW is zero, `live` is empty, and every median collapses to None — an empty
    cell dressed up as a measurement. The expert is a real compartment with real weights, so the
    honest census is the absolute one.

    Also used for PG1's own row, where the delta IS exactly zero by construction and the absolute
    profile is the reference every other unit's delta is measured against.

    Carries the same N2 base-matrix control as the delta census: `n_below_band` counts matrices
    whose own stable rank falls under the Frobenius-matched Gaussian band, which is what makes
    "low-rank" a checkable claim about these matrices rather than about trained matrices in general.
    """
    import numpy as np
    from infra.aws_pg.jobs.lacuna_a2_bytes import SRANK_DRAWS, _sigma1
    from infra.pg import lacuna_core as LC
    rows = []
    for k in sorted(k for k in sd if sd[k].ndim == 2):
        A = sd[k].detach().double().numpy()
        fro = float(np.linalg.norm(A))
        s1 = _sigma1(A)
        rows.append({"key": k, "shape": list(A.shape), "frob": fro,
                     "srank": float((A ** 2).sum() / max(s1 ** 2, 1e-300)),
                     # SRANK_DRAWS, not a local number: the band is cached per
                     # (shape, n_draws, seed), so a different draw count here would silently compute
                     # a SECOND band and the absolute and delta censuses would be comparing against
                     # different nulls.
                     "srank_null": LC.srank_random_band(A.shape, n_draws=SRANK_DRAWS,
                                                        seed=seed)})
    assert rows, f"{tag}: no 2-D matrices — an absolute census over nothing is not a measurement"
    med = float(np.median([r["srank"] for r in rows]))
    lo = float(np.median([r["srank_null"]["p2_5"] for r in rows]))
    hi = float(np.median([r["srank_null"]["p97_5"] for r in rows]))
    return {"tag": tag, "absolute": True, "n_matrices": len(rows), "n_nonzero": len(rows),
            "median_srank": med, "median_null_p2_5": lo, "median_null_p97_5": hi,
            "median_frob": float(np.median([r["frob"] for r in rows])),
            "total_frob": float(np.sqrt(sum(r["frob"] ** 2 for r in rows))),
            "n_below_band": int(sum(1 for r in rows if r["srank"] < r["srank_null"]["p2_5"])),
            "fullrank_like": bool(lo <= med <= hi),
            "below_band": bool(med < lo),
            "reference": "NONE — no pre-install state exists for this component (§2.3)",
            "per_matrix": rows}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="all-ft")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--resume", default=None,
                    help="prior metrics.json (local or s3://); units already banked are skipped")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs.lacuna_a2_bytes import _delta_census
    from infra.pg import lacuna_shell as LS
    from infra.pg import lacuna_shell as _LS
    from infra.pg import plenum_units as PU

    units = (PU.ft_units() if a.units == "all-ft"
             else [u.strip() for u in a.units.split(",") if u.strip()])
    # RESUME rather than try/except. A 24-unit census pulls ~290 GB from HF and a single dropped
    # connection killed run 1 after one unit. Wrapping the download would HIDE a real failure; a
    # resumable job re-runs cheaply and still dies loudly on anything that is not transient.
    done_cells, skip = [], set()
    if a.resume:
        prior = _read_json(a.resume)
        done_cells = prior.get("cells", [])
        skip = {c["unit"] for c in done_cells}
        print(f"[CEN] resuming: {len(skip)} units already banked", flush=True)
    t0 = time.time()
    rep = {"object": "plenum_census_v1", "units": units, "topk": a.topk,
           "label": "EXPLORATORY census (§7) — never in a BH family",
           "n2_control": "every spectral row carries srank_base + n_base_below_band; a band the "
                         "BASE matrices also fail is withdrawn by the aggregator (§5 N2)"}
    cells = list(done_cells)
    rep["cells"] = cells                          # bound BEFORE the loop (fleet lesson)
    rep["resumed_units"] = sorted(skip)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # PG1 is the reference every unit is differenced against — loaded ONCE, on CPU.
    from transformers import PaliGemmaForConditionalGeneration
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        LS.PG1_REPO, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    base = LS.canon_map({k: v.detach().cpu() for k, v in pg.state_dict().items()})
    del pg
    rep["base"] = {"repo": LS.PG1_REPO, "n_keys": len(base)}
    _w()

    for uid in units:
        if uid in skip:
            continue
        # The roster is not only fine-tunes: the map needs weights/geometry for the policies and
        # the banked units too, or ~80 cells stay PENDING for want of a loader rather than for want
        # of a measurement. A policy's VLM half is differenced against PG1 exactly as an FT is —
        # same reference, same components, so the rows stay comparable.
        expert_sd = None
        if uid == "pg1":
            # PG1 IS the reference, so its delta is exactly zero by construction — which makes this
            # row two things at once, and neither is a placeholder. (a) An INSTRUMENT CHECK: any
            # nonzero entry here would mean the load/canon path is not deterministic, and every
            # other unit's ΔW would be measuring that instead. (b) The absolute spectral profile of
            # PG1's own matrices — the reference the whole census is stated relative to, which the
            # N2 control needs in order to be interpretable at all.
            sd, repo, sha = base, LS.PG1_REPO, None
            rec_abs = True
        else:
            rec_abs = False
        if uid.startswith("ft-"):
            slug = uid[len("ft-"):]
            sd, sha, repo = PU.ft_state_dict(slug)
        elif uid == "pg1":
            pass                      # sd already bound to the reference above
        elif uid in ("stitch0", "matched_dw", "rand_dec", "gemma-2b"):
            # I previously called these "no measurement defined" and left ~54 cells pending. That
            # was wrong. The synthetics are CONSTRUCTED in the shell, so their state dict is
            # obtainable and differencing it against PG1 is exactly the same operation as for a
            # fine-tune. gemma-2b shares the Gemma-1 decoder architecture with PG1's language_model,
            # so its LM and embedding are differenceable; its vision/projector are already
            # structurally impossible and stay that way.
            import torch as _t
            from infra.pg import lacuna_shell as _LS
            from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
            if uid == "gemma-2b":
                g = AutoModelForCausalLM.from_pretrained(
                    _LS.GEMMA1_REPO, torch_dtype=_t.float32, device_map="cpu")
                gsd = _LS.canon_map({k: v.detach().cpu() for k, v in g.state_dict().items()})
                del g
                sd = {f"language_model.{k}": v for k, v in gsd.items()}
                repo, sha = _LS.GEMMA1_REPO, None
            else:
                shell = PaliGemmaForConditionalGeneration.from_pretrained(
                    _LS.PG1_REPO, torch_dtype=_t.float32, device_map="cpu",
                    attn_implementation="eager")
                base_sd = _LS.canon_map({k: v.detach().cpu().clone()
                                         for k, v in shell.state_dict().items()})
                if uid == "stitch0":
                    _LS.build_row_stitch0(shell, 1, base_sd)
                elif uid == "matched_dw":
                    full, _ = _LS.policy_full_sd("pi0")
                    lm, _h, _c = _LS.policy_vlm_sd("pi0", full_sd=full)
                    tgt = {k[len("language_model."):]: v for k, v in lm.items()
                           if k.startswith("language_model.")}
                    del full
                    _LS.build_row_matched_dw(shell, base_sd, tgt, 20260727)
                else:
                    _LS.build_row_random_decoder(shell, base_sd, 1337)
                sd = _LS.canon_map({k: v.detach().cpu() for k, v in shell.state_dict().items()})
                del shell
                repo, sha = f"constructed:{uid}", None
        else:
            from infra.pg import lacuna_shell as _LS
            src = PU.POLICY_UNITS.get(uid)
            assert src, f"{uid}: no source known (expected an ft-* or a policy/banked unit)"
            if src.startswith("hf:"):
                full, sha = _LS.policy_full_sd(uid)
            else:
                full, _dig = PU.s3_policy_sd(src, f"/tmp/census_{uid}")
                sha = None
                if not any(k.startswith(_LS.PALI_PREFIX) for k in full):
                    full = {_LS.canon(k): v for k, v in full.items()}
                # Same tied-embed duplicate the unit loader already handles: a banked unit was
                # written from a LIVE PI0Policy and carries BOTH lm_head and a tied embed_tokens,
                # giving 164 LM keys where the published policies have 163. Drop the duplicate —
                # the head is reinstated as the embedding below, so nothing is lost.
                for k in [x for x in full
                          if x.endswith("language_model.embed_tokens.weight")
                          and _LS.PALI_PREFIX in x]:
                    del full[k]
            vlm, own_head, _c = _LS.policy_vlm_sd(
                "pi05" if uid == "pi05" else "pi0", full_sd=full)
            # the policy's tied head IS its embedding table; put it back under the shell's name so
            # the embedding component is differenced rather than silently skipped
            sd = dict(vlm)
            sd["language_model.embed_tokens.weight"] = own_head
            repo = src
            # THE ACTION EXPERT. `policy_vlm_sd` deliberately keeps the expert OUT of the VLM shell
            # (it asserts on leakage), which is right for the battery — and it is why the expert
            # never reached the census at all, leaving weights/expert + geometry/expert PENDING for
            # all four policy units over weights that were already in memory.
            #
            # The expert is everything the policy owns that is NOT the PaliGemma half: the
            # gemma_expert decoder plus the action/state projections and the time MLP. Defined by
            # exclusion rather than by an enumerated prefix list, so a checkpoint that names its
            # projections differently is still covered; the matched key inventory is recorded in the
            # artifact so the composition is auditable rather than assumed.
            expert_sd = {k: v for k, v in full.items() if not k.startswith(_LS.PALI_PREFIX)}
            assert expert_sd, f"{uid}: no non-PaliGemma keys — this checkpoint has no action expert"
            expert_sd = {k: v.detach().cpu().float() for k, v in expert_sd.items()}
            del full
        rec = {"unit": uid, "repo": repo, "sha": sha,
               "weights_etag": unit_etag(PU.POLICY_UNITS.get(uid, "")), "components": {}}
        for comp, prefix in COMPONENTS.items():
            b, u = _sub(base, prefix), _sub(sd, prefix)
            if not u:
                # gemma-2b has no vision tower or projector: those cells are already `impossible`
                # in the frozen map, so a missing component here is expected, not an error.
                rec["components"][comp] = {"absent": True,
                                           "why": f"{uid} has no {comp} component"}
                continue
            assert b, f"{uid}/{comp}: empty base component"
            # VOCAB-WIDTH TRAP (lacuna_shell AMENDMENT-2, documented): PG1's embedding is 257216
            # padded rows while a policy's tied table is 257152. The extra 64 are the `<image>`
            # placeholder plus padding, which the policy architecture does not use. Differencing
            # them elementwise cannot broadcast. Compare the SHARED rows only, and record the
            # truncation rather than letting a silently-reshaped tensor into the census.
            trunc = {}
            for k in list(b):
                if k in u and b[k].ndim == 2 and b[k].shape[0] != u[k].shape[0]:
                    n = min(b[k].shape[0], u[k].shape[0])
                    trunc[k] = {"base_rows": int(b[k].shape[0]), "unit_rows": int(u[k].shape[0]),
                                "compared_rows": int(n)}
                    b = {**b, k: b[k][:n]}
                    u = {**u, k: u[k][:n]}
            if trunc:
                rec.setdefault("vocab_truncation", {})[comp] = trunc
            # KEY-NAME RESOLUTION. `_delta_census` matches keys EXACTLY (`k in base`), but a policy
            # checkpoint and the PG1 shell differ in INTERNAL nesting depth for the same tensors:
            # `vision_tower.vision_model.encoder.layers.0.…` vs `vision_tower.encoder.layers.0.…`
            # (documented on `lacuna_shell.resolve_keys`, which the shell install uses for exactly
            # this reason). Exact matching therefore found ZERO shared vision matrices for EVERY
            # policy unit — pi0, pi05, u3, u4, u3c all reported `n_matrices: 0`, and the aggregator
            # turned that into a `weights/vision` verdict of "unmoved" computed from no comparison at
            # all. Worse than unsupported: LACUNA measured these towers as NOT byte-frozen
            # (max|Δ| 0.94 / 1.90), so "unmoved" contradicts a known result.
            #
            # Resolve the unit's keys onto the shell's canonical names before differencing, using the
            # same compartment-restricted longest-suffix match the install path uses. Its numeric-
            # address guard is what keeps this safe (a suffix short enough to drop the layer index
            # cannot silently map layer 9 onto layer 0).
            if not (set(b) & set(u)) and b and u:
                _m = _LS.resolve_keys(b, list(u), f"{uid}/{comp}")
                _u2 = {_m[k]: v for k, v in u.items() if _m.get(k)}
                rec.setdefault("key_resolution", {})[comp] = {
                    "unit_keys": len(u), "resolved": len(_u2),
                    "why": "exact key match found nothing; nesting depth differs between the policy "
                           "checkpoint and the shell"}
                assert _u2, (f"{uid}/{comp}: {len(u)} unit keys and {len(b)} shell keys share no "
                             f"names and none resolve — refusing to emit a verdict from no comparison")
                u = _u2
            rec["components"][comp] = _delta_census(b, u, f"{uid}/{comp}", topk=a.topk)
            # A component that compared NOTHING must not produce a verdict.
            assert rec["components"][comp]["n_matrices"] > 0, (
                f"{uid}/{comp}: zero matrices compared — a cell here would assert 'unmoved' from no "
                f"data")
        if rec_abs:
            for comp, prefix in COMPONENTS.items():
                blk = rec["components"][comp]
                assert blk.get("median_rel_frob") == 0.0, (
                    f"pg1/{comp}: PG1 differenced against itself gave "
                    f"{blk.get('median_rel_frob')!r}, not an exact zero — the load path is not "
                    f"deterministic and every other unit's ΔW is contaminated by that")
                blk["self_reference_check"] = "PASS (delta is an exact structural zero)"
                blk["absolute_profile"] = _absolute_census(
                    _sub(base, prefix), f"pg1/{comp}/absolute", topk=a.topk)
        if expert_sd is not None:
            rec["components"]["expert"] = _absolute_census(expert_sd, f"{uid}/expert",
                                                           topk=a.topk)
            rec["expert_keys"] = sorted(expert_sd)
            del expert_sd
        # the law's ‖ΔW‖ predictor: the LM component's median relative Frobenius norm
        rec["dw_norm_lm"] = rec["components"]["lm"]["median_rel_frob"]
        cells.append(rec)
        if uid.startswith("ft-"):
            PU._delete_cache(repo)
        print(f"[CEN] {uid:>20s} dW_lm={rec['dw_norm_lm']:.5f} "
              f"srank_delta={rec['components']['lm']['median_srank_delta']} "
              f"base_below_band={rec['components']['lm']['n_base_below_band']}"
              f"/{rec['components']['lm']['n_nonzero']} t={time.time()-t0:.0f}s", flush=True)
        _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("PLENUM_CENSUS_OK", json.dumps({c["unit"]: c["dw_norm_lm"] for c in cells}), flush=True)


if __name__ == "__main__":
    main()
