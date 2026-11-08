"""ERLANGEN Arm T — the dissociation-fingerprint screen, ONE unit per job (spec §2-T).

Measures Δ(readout battery) for every component × every intervention on one unit, and banks the
raw fingerprint tensor. Clustering, stability, transfer and the type inventory are computed by
`erlangen_aggregate` over the banked tensors — the screen itself makes no claim (it is a census,
§7).

    python -m infra.aws_pg.jobs.erlangen_t --out DIR --unit pg1 [--n-items 32]
        [--ft-repo google/paligemma-3b-ft-...] [--no-per-head] [--layer-stride 1]

Every cell banks its two item-half means, so the noise floor (and therefore the `unmeasurable`
VOID branch) is available for every component without a second pass.

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--n-items", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ft-repo", default="")
    ap.add_argument("--ft-sha", default="")
    ap.add_argument("--no-per-head", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--curiosity-core", action="store_true",
                    help="amendment A6: also fingerprint the untouched-core ACTIVATION subspace "
                         "as if it were a component")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch

    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_shell import load_pg1, pg1_baseline_sd, policy_full_sd

    assert torch.cuda.is_available(), "Arm T needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "erlangen_t_v1", "unit": a.unit, "n_items": a.n_items,
           "readouts": list(P.READOUT_NAMES), "interventions": list(P.INTERVENTIONS),
           "ft_repo": a.ft_repo, "layer_stride": a.layer_stride,
           "include_per_head": not a.no_per_head, "include_vision": not a.no_vision}

    def bank():
        (out / f"erlangen_t_{a.unit}.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    gemma1_donor = P.donor_gemma1(pg1_sd)
    donors = {"pg1": pg1_sd, "gemma1": gemma1_donor}
    policy_sd = policy_full_sd("pi05")[0] if a.unit == "pi05" else None
    print(f"[T/{a.unit}] shell + donors ready ({time.time()-t0:.0f}s)", flush=True)

    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    comps = P.enumerate_components(model, include_per_head=not a.no_per_head,
                                   include_vision=not a.no_vision,
                                   layer_stride=a.layer_stride)
    rep["components"] = list(comps)
    rep["n_components"] = len(comps)
    print(f"[T/{a.unit}] grid = {len(comps)} components × {len(P.INTERVENTIONS)} interventions "
          f"= {len(comps)*len(P.INTERVENTIONS)} cells", flush=True)
    bank()

    info = P.install_unit(model, a.unit, pg1_sd, gemma1_donor=gemma1_donor,
                          ft_repo=a.ft_repo or None, ft_sha=a.ft_sha or None,
                          policy_sd=policy_sd)
    rep["install"] = {k: v for k, v in info.items() if k != "restore"}
    fv = P.forward_validity(model, proc, device, packs)
    rep["forward_validity"] = fv
    print(f"[T/{a.unit}] N3 forward validity: {fv['note']} wiki={fv['wiki_nll']:.4f} "
          f"dperc={fv['delta_perception']:+.4f}", flush=True)
    bank()

    base = P.readout_vector(model, proc, device, packs, batch_size=a.batch)
    ba, bb = P.half_vecs(base)
    rep["cells"] = {"__base__": {"vec": P.vec(base), "half_a": ba, "half_b": bb}}
    rep["identity_controls"] = {}
    bank()

    tcell = time.time()
    for ci, cname in enumerate(comps):
        parts = comps[cname]
        for kind in P.INTERVENTIONS:
            saved = P.apply_intervention(model, parts, kind, donors, seed=20260803)
            r = P.readout_vector(model, proc, device, packs, batch_size=a.batch)
            P.restore(model, saved)
            ha, hb = P.half_vecs(r)
            rep["cells"][f"{cname}|{kind}"] = {"vec": P.vec(r), "half_a": ha, "half_b": hb}
            if (a.unit, kind) in (("pg1", "swap_pg1"), ("gemma1", "swap_gemma1")):
                d = max(abs(x - y) for x, y in zip(P.vec(r), P.vec(base)))
                rep["identity_controls"][f"{cname}|{kind}"] = d
                assert d < 1e-4, (
                    f"IDENTITY CONTROL FAILED {a.unit}/{cname}/{kind}: drift {d} — surgery bug")
        if (ci + 1) % 5 == 0 or ci == 0:
            per = (time.time() - tcell) / (ci + 1)
            print(f"[T/{a.unit}] {ci+1}/{len(comps)} {cname} {per:.1f}s/comp "
                  f"eta {per*(len(comps)-ci-1)/60:.0f}min", flush=True)
            bank()
    bank()

    # --- amendment A6 (curiosity reserve): does the untouched core have a TYPE? ----------------
    # The core is an ACTIVATION subspace, not a matrix, so it cannot be a row of the weight grid.
    # It is fingerprinted with the same readout axis under the activation-space analogue of the
    # intervention set (mean-ablate span(Q) → h̄ at each probe layer), so its row lands in the same
    # space as the weight components and can be clustered beside them.
    if a.curiosity_core:
        from infra.pg import erlangen_algebra as A
        rep["curiosity_core"] = A.core_fingerprint(model, proc, device, packs, batch_size=a.batch,
                                                   base_vec=P.vec(base))
        bank()

    rep["timing"] = {"secs": time.time() - t0,
                     "secs_per_cell": (time.time() - tcell) / max(len(comps) * len(P.INTERVENTIONS), 1)}
    rep["DONE"] = True
    bank()
    print(f"[T/{a.unit}] DONE {len(comps)} components in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
