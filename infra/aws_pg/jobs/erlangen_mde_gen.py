"""ERLANGEN MDE re-scope — P1 on the GENERATION battery (the pre-committed amendment-A5 branch).

A5 froze this remedy before any data: if the teacher-forced NLL readout fails its rank-agreement
gate against the generation battery, T re-scopes to the generation battery on a reduced grid and
the re-scope is reported. The gate fired (ρ = −0.238, CI [−0.825, +0.566]; no individual channel
passes either), so this job runs the SAME frozen P1 rule with the SAME frozen grids and label maps
— only the readout axis changes:

    channels = {vqav2, textvqa} × {real image, gray-128 image}

Both tasks are per-item scored (VQA soft-accuracy / ANLS), so the two disjoint item halves still
give the repeat-measurement noise floor and the VOID branch stays available. Nothing else moves:
same components, same interventions, same ARI ≥ 0.50 / permutation p < 0.05 / stability ≥ 0.50.

Why the NLL readout failed is itself banked by the MDE and is a finding, not an excuse: zeroing the
projector collapses generation (+0.578, i.e. the whole score) while IMPROVING teacher-forced NLL
(−1.78 on the VQA channel), because a model that cannot see falls back on a strong short-answer
language prior. Likelihood and capability dissociate exactly at the components that break the image
path — the readout-level form of the GraftLM KL≠capability lesson.

    python -m infra.aws_pg.jobs.erlangen_mde_gen --out DIR [--n-items 64] [--unit pg1]

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
    ap.add_argument("--unit", default="pg1")
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-split", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.inherit_pg1 import eval_pg1
    from infra.pg.lacuna_shell import gray_task, load_pg1, pg1_baseline_sd
    from infra.pg.projector_ablation import build_tasks, load_eval_items

    assert torch.cuda.is_available(), "the re-scoped MDE needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "erlangen_mde_gen_v1", "unit": a.unit, "n_items": a.n_items,
           "trigger": "amendment A5 construct-validity gate FAILED on the NLL readout",
           "interventions": list(P.INTERVENTIONS)}

    def bank():
        (out / "erlangen_mde_gen.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    gemma1_donor = P.donor_gemma1(pg1_sd)
    donors = {"pg1": pg1_sd, "gemma1": gemma1_donor}
    print(f"[gen] shell + donors ready ({time.time()-t0:.0f}s)", flush=True)

    bt = {t.name: t for t in build_tasks()}
    rows = {nm: load_eval_items(bt[nm], a.n_items) for nm in ("vqav2", "textvqa")}
    channels = [("vqa_real", bt["vqav2"], "vqav2"),
                ("vqa_gray", gray_task(bt["vqav2"]), "vqav2"),
                ("txtvqa_real", bt["textvqa"], "textvqa"),
                ("txtvqa_gray", gray_task(bt["textvqa"]), "textvqa")]
    rep["channels"] = [c[0] for c in channels]
    print(f"[gen] {len(channels)} generation channels × {a.n_items} items", flush=True)

    def battery():
        """per-channel score + per-item vector (both tasks are per-item scored)."""
        o = {}
        for nm, task, rk in channels:
            r = eval_pg1(model, proc, task, rows[rk], device, batch_size=a.batch)
            assert r["per_item"] is not None, f"{nm} is not per-item scored"
            o[nm] = {"mean": r["score"], "per_item": r["per_item"]}
        return o

    def vec(o):
        return [o[c[0]]["mean"] for c in channels]

    def halves(o):
        aa, bb = [], []
        for c in channels:
            pi = o[c[0]]["per_item"]
            aa.append(float(np.mean(pi[0::2])))
            bb.append(float(np.mean(pi[1::2])))
        return aa, bb

    comps_all = P.enumerate_components(model, include_per_head=False, include_vision=True)
    grid = []
    for name, _ in list(P.P1_PRIMARY) + list(P.P1_SECONDARY):
        if name not in grid:
            grid.append(name)
    missing = [g for g in grid if g not in comps_all]
    assert not missing, f"grid names absent from the shell: {missing}"
    rep["grid"] = grid

    P.install_unit(model, a.unit, pg1_sd, gemma1_donor=gemma1_donor)
    base = battery()
    ba, bb = halves(base)
    rep["cells"] = {"__base__": {"vec": vec(base), "half_a": ba, "half_b": bb}}
    rep["identity_controls"] = {}
    print(f"[gen] baseline: " + " ".join(f"{c[0]}={base[c[0]]['mean']:.4f}" for c in channels),
          flush=True)
    bank()

    tcell = time.time()
    for ci, cname in enumerate(grid):
        for kind in P.INTERVENTIONS:
            saved = P.apply_intervention(model, comps_all[cname], kind, donors, seed=20260803)
            r = battery()
            P.restore(model, saved)
            ha, hb = halves(r)
            rep["cells"][f"{cname}|{kind}"] = {"vec": vec(r), "half_a": ha, "half_b": hb}
            if (a.unit, kind) in (("pg1", "swap_pg1"), ("gemma1", "swap_gemma1")):
                d = max(abs(x - y) for x, y in zip(vec(r), vec(base)))
                rep["identity_controls"][f"{cname}|{kind}"] = d
                assert d < 1e-4, f"IDENTITY CONTROL FAILED {cname}/{kind}: drift {d}"
        print(f"[gen] {ci+1}/{len(grid)} {cname} ({(time.time()-tcell)/(ci+1):.1f}s/comp)",
              flush=True)
        bank()

    # ---- the frozen P1 rule, unchanged --------------------------------------------------------
    def fingerprints(names):
        cells = rep["cells"]
        b = cells["__base__"]
        F, A_, B_ = [], [], []
        for c in names:
            row, ra, rb = [], [], []
            for kind in P.INTERVENTIONS:
                cell = cells[f"{c}|{kind}"]
                row += [x - y for x, y in zip(cell["vec"], b["vec"])]
                ra += [x - y for x, y in zip(cell["half_a"], b["half_a"])]
                rb += [x - y for x, y in zip(cell["half_b"], b["half_b"])]
            F.append(row); A_.append(ra); B_.append(rb)
        return np.asarray(F), np.asarray(A_), np.asarray(B_)

    rep["p1"] = {}
    for gname, gmap in (("primary", P.P1_PRIMARY), ("secondary", P.P1_SECONDARY)):
        names = [n for n, _ in gmap]
        types = sorted({t for _, t in gmap})
        labels = np.array([types.index(t) for _, t in gmap])
        F, A_, B_ = fingerprints(names)
        scale = EC.column_scale(F)["scale"]
        v = EC.p1_verdict(A_, B_, labels, scale, k=len(types), n_perm=a.n_perm,
                          n_split=a.n_split)
        v.update({"grid": gname, "unit": a.unit, "types": types, "components": names})
        rep["p1"][gname] = v
        print(f"[gen] P1 {gname}: {v['verdict']} ARI={v['ari_normalized']:.3f} "
              f"p={v['perm_p']:.4f} stab={v['stability_ari_median']:.3f} "
              f"raw={v['ari_raw_sensitivity']:.3f} unmeas={v['noise_floor']['n_unmeasurable']}",
              flush=True)
        bank()

    rep["MDE_GEN_VERDICT"] = rep["p1"]["primary"]["verdict"]
    rep["MDE_GEN_MEANS"] = rep["p1"]["primary"]["failure_means"]
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[gen] DONE verdict={rep['MDE_GEN_VERDICT']} in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
