"""ERLANGEN MDE — P1, the gate that can kill the typing paradigm for ~1 GPU-h (spec §3).

The clustering of magnitude-normalized response fingerprints must recover the three KNOWN
content-free types (interface / knowledge / addressing) on the frozen 12-component grid, against a
label-permutation null, with split-half stability over the readout axis (amendment A1). It also
runs the harder LM-internal secondary grid (addressing vs knowledge vs routing, where compartment
identity is unavailable as a shortcut) and the A5 construct-validity gate against the generation
battery.

    python -m infra.aws_pg.jobs.erlangen_mde --out DIR [--units pg1,pi05] [--n-items 64]

PASS → Arm T proceeds. FAIL → H_types-converge is dead and the program pivots to D/M/B.
VOID → components below the repeat-measurement noise floor; reported as an instrument fact.

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
    ap.add_argument("--units", default="pg1,pi05")
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-split", type=int, default=200)
    ap.add_argument("--skip-genbattery", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check only: 8 items, 2 components, 1 unit, no clustering")
    a = ap.parse_args()
    if a.smoke:
        a.n_items, a.units, a.n_perm, a.n_split = 8, "pg1", 200, 10
        a.skip_genbattery = True
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch

    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_shell import load_pg1, pg1_baseline_sd, policy_full_sd

    assert torch.cuda.is_available(), "the MDE needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "erlangen_mde_v1", "units": a.units.split(","), "n_items": a.n_items,
           "readouts": list(P.READOUT_NAMES), "interventions": list(P.INTERVENTIONS)}

    def bank():
        (out / "erlangen_mde.json").write_text(json.dumps(rep, indent=2))

    bank()

    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    print(f"[mde] PG1 shell up ({time.time()-t0:.0f}s), {len(pg1_sd)} canonical keys", flush=True)

    gemma1_donor = P.donor_gemma1(pg1_sd)
    donors = {"pg1": pg1_sd, "gemma1": gemma1_donor}
    print(f"[mde] donors ready: pg1={len(pg1_sd)} gemma1={len(gemma1_donor)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    print(f"[mde] items {rep['items_sha256'][:12]} ({time.time()-t0:.0f}s)", flush=True)
    bank()

    # the union of both frozen grids — measured once, scored twice
    grid = []
    for name, _ in list(P.P1_PRIMARY) + list(P.P1_SECONDARY):
        if name not in grid:
            grid.append(name)
    if a.smoke:
        grid = grid[:2] + ["multi_modal_projector.linear"]
    rep["grid"] = grid
    rep["p1_primary"] = {k: v for k, v in P.P1_PRIMARY}
    rep["p1_secondary"] = {k: v for k, v in P.P1_SECONDARY}
    comps_all = P.enumerate_components(model, include_per_head=False, include_vision=True)
    missing = [g for g in grid if g not in comps_all]
    assert not missing, f"MDE grid names absent from the shell: {missing}"

    policy_sd = None
    if "pi05" in rep["units"]:
        policy_sd, pi05_sha = policy_full_sd("pi05")
        rep["pi05_sha"] = pi05_sha
        print(f"[mde] pi05 checkpoint {pi05_sha[:8]} loaded ({time.time()-t0:.0f}s)", flush=True)

    rep["cells"] = {}
    rep["identity_controls"] = {}
    rep["forward_validity"] = {}
    rep["timing"] = {}
    for unit in rep["units"]:
        info = P.install_unit(model, unit, pg1_sd, gemma1_donor=gemma1_donor,
                              policy_sd=policy_sd)
        fv = P.forward_validity(model, proc, device, packs)
        rep["forward_validity"][unit] = fv
        print(f"[mde] unit={unit} install={info.get('install', {}).get('n_installed', 0)} "
              f"valid={fv['valid']} wiki_nll={fv['wiki_nll']:.4f} dperc={fv['delta_perception']:.4f}",
              flush=True)
        bank()
        base = P.readout_vector(model, proc, device, packs, batch_size=a.batch)
        ba, bb = P.half_vecs(base)
        rep["cells"][unit] = {"__base__": {"vec": P.vec(base), "half_a": ba, "half_b": bb}}
        bank()
        tcell = time.time()
        for ci, cname in enumerate(grid):
            parts = comps_all[cname]
            for kind in P.INTERVENTIONS:
                saved = P.apply_intervention(model, parts, kind, donors, seed=20260803)
                r = P.readout_vector(model, proc, device, packs, batch_size=a.batch)
                P.restore(model, saved)
                ha, hb = P.half_vecs(r)
                rep["cells"][unit][f"{cname}|{kind}"] = {
                    "vec": P.vec(r), "half_a": ha, "half_b": hb}
                # in-line P3: swap_pg1 on the pg1 unit (and swap_gemma1 on gemma1) is IDENTITY
                if (unit, kind) in (("pg1", "swap_pg1"), ("gemma1", "swap_gemma1")):
                    d = max(abs(x - y) for x, y in zip(P.vec(r), P.vec(base)))
                    rep["identity_controls"][f"{unit}|{cname}|{kind}"] = d
                    assert d < 1e-4, (
                        f"IDENTITY CONTROL FAILED {unit}/{cname}/{kind}: max readout drift {d} — "
                        f"the swap is supposed to be a no-op; state-dict surgery is wrong")
            print(f"[mde] {unit} {ci+1}/{len(grid)} {cname} "
                  f"({(time.time()-tcell)/(ci+1):.1f}s/comp)", flush=True)
            bank()
        rep["timing"][unit] = {"secs": time.time() - tcell, "n_comp": len(grid),
                               "secs_per_component": (time.time() - tcell) / len(grid),
                               "secs_per_cell": (time.time() - tcell) / (len(grid) * len(P.INTERVENTIONS))}
        bank()

    # ---- fingerprints + the P1 verdicts -------------------------------------------------------
    import numpy as np

    def fingerprints(unit, names):
        """(F, F_half_a, F_half_b) with rows = components, cols = intervention × readout."""
        cells = rep["cells"][unit]
        base = cells["__base__"]
        F, A, B = [], [], []
        for cname in names:
            row, ra, rb = [], [], []
            for kind in P.INTERVENTIONS:
                c = cells[f"{cname}|{kind}"]
                row += [x - y for x, y in zip(c["vec"], base["vec"])]
                ra += [x - y for x, y in zip(c["half_a"], base["half_a"])]
                rb += [x - y for x, y in zip(c["half_b"], base["half_b"])]
            F.append(row); A.append(ra); B.append(rb)
        return np.asarray(F), np.asarray(A), np.asarray(B)

    rep["p1"] = {}
    ref_unit = rep["units"][0]
    grids = () if a.smoke else (("primary", P.P1_PRIMARY), ("secondary", P.P1_SECONDARY))
    for gname, gmap in grids:
        names = [n for n, _ in gmap]
        types = sorted({t for _, t in gmap})
        labels = np.array([types.index(t) for _, t in gmap])
        for unit in rep["units"]:
            F, A, B = fingerprints(unit, names)
            # the column scale is fixed on the REFERENCE unit and applied everywhere (A1/A9)
            scale = EC.column_scale(fingerprints(ref_unit, names)[0])["scale"]
            v = EC.p1_verdict(A, B, labels, scale, k=len(types),
                              n_perm=a.n_perm, n_split=a.n_split)
            v["grid"] = gname
            v["unit"] = unit
            v["types"] = types
            v["components"] = names
            rep["p1"][f"{gname}|{unit}"] = v
            print(f"[mde] P1 {gname}/{unit}: {v['verdict']} ARI={v['ari_normalized']:.3f} "
                  f"p={v['perm_p']:.4f} stab={v['stability_ari_median']:.3f} "
                  f"(raw {v['ari_raw_sensitivity']:.3f})", flush=True)
            bank()
        # cross-unit transfer: do the SAME components cluster the same way in another unit?
        if len(rep["units"]) > 1:
            u0, u1 = rep["units"][0], rep["units"][1]
            s = EC.column_scale(fingerprints(u0, names)[0])["scale"]
            l0 = EC.cluster(EC.preprocess(fingerprints(u0, names)[0], s), len(types))
            l1 = EC.cluster(EC.preprocess(fingerprints(u1, names)[0], s), len(types))
            rep["p1"][f"{gname}|transfer"] = {
                "units": [u0, u1], "ari_between_units": EC.adjusted_rand(l0, l1),
                "labels_u0": [int(x) for x in l0], "labels_u1": [int(x) for x in l1]}
            print(f"[mde] transfer {gname} {u0}↔{u1} ARI="
                  f"{rep['p1'][f'{gname}|transfer']['ari_between_units']:.3f}", flush=True)
    bank()

    # ---- A5 construct-validity: NLL damage vs GENERATION-battery damage -----------------------
    if not a.skip_genbattery:
        from infra.pg.inherit_pg1 import eval_pg1, load_items, pg1_tasks
        P.install_unit(model, ref_unit, pg1_sd, gemma1_donor=gemma1_donor, policy_sd=policy_sd)
        vqa_task, _ = pg1_tasks()
        gen_rows, _ = load_items(vqa_task, a.n_items)
        base_gen = eval_pg1(model, proc, vqa_task, gen_rows, device, batch_size=8)["score"]
        names = [n for n, _ in P.P1_PRIMARY]
        gen_dmg, nll_dmg = [], []
        rep["construct_validity_rows"] = []
        for cname in names:
            saved = P.apply_intervention(model, comps_all[cname], "zero", donors, seed=20260803)
            s = eval_pg1(model, proc, vqa_task, gen_rows, device, batch_size=8)["score"]
            P.restore(model, saved)
            cells = rep["cells"][ref_unit]
            d_nll = float(np.mean([x - y for x, y in zip(cells[f"{cname}|zero"]["vec"],
                                                         cells["__base__"]["vec"])]))
            gen_dmg.append(base_gen - s)
            nll_dmg.append(d_nll)
            rep["construct_validity_rows"].append(
                {"component": cname, "gen_score": s, "gen_damage": base_gen - s,
                 "nll_damage": d_nll})
            print(f"[mde] genbattery {cname}: score={s:.4f} damage={base_gen - s:+.4f} "
                  f"nll_damage={d_nll:+.4f}", flush=True)
            bank()
        rep["base_gen_score"] = base_gen
        rep["construct_validity"] = EC.construct_validity_rho(nll_dmg, gen_dmg)
        print(f"[mde] A5 construct validity rho={rep['construct_validity']['rho']:.3f} "
              f"CI={rep['construct_validity']['lo95']:.3f},{rep['construct_validity']['hi95']:.3f} "
              f"passes={rep['construct_validity']['passes']}", flush=True)

    # ---- the gate, flags LAST ------------------------------------------------------------------
    prim = rep["p1"].get(f"primary|{ref_unit}")
    rep["MDE_VERDICT"] = "SMOKE-OK" if a.smoke else prim["verdict"]
    rep["MDE_MEANS"] = "wiring check only" if a.smoke else prim["failure_means"]
    rep["total_secs"] = time.time() - t0
    bank()
    print(f"[mde] DONE verdict={rep['MDE_VERDICT']} in {rep['total_secs']/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
