"""ERLANGEN Arm X — the transplant compatibility matrix + mirror-v2 (spec §2-X).

For every (component group, donor→host) the donor's group is installed into the host and the
readout battery is re-measured. Compatibility is normalised BY ITS OWN CONTROL: the same group
zero-ablated in the same host. So

    compat = 1 − max(0, mean_r[NLL_r(transplant) − NLL_r(host)]) / mean_r[NLL_r(host, group zeroed) − NLL_r(host)]

    1  : the transplant costs the host nothing
    0  : the transplant is as bad as deleting the group outright
    <0 : WORSE than deleting it (reported, never clipped — the ceiling direction is a real finding)

P3 (self-transplant identity) runs FIRST and on every cell: donor == host must be a bit-exact
no-op, and a non-zero delta HALTS the job rather than being averaged into a matrix.

    python -m infra.aws_pg.jobs.erlangen_x --out DIR --ft-max-repo … --ft-min-repo …
        --ft-max-task textvqa --ft-min-task vqav2 [--n-items 32]

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
    ap.add_argument("--n-items", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ft-max-repo", required=True)
    ap.add_argument("--ft-min-repo", required=True)
    ap.add_argument("--ft-max-task", required=True)
    ap.add_argument("--ft-min-task", required=True)
    ap.add_argument("--units", default="pg1,pi0,pi05,ft_max,ft_min")
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

    from infra.pg import erlangen_annotation as AN
    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_shell import load_pg1, pg1_baseline_sd

    assert torch.cuda.is_available(), "Arm X needs a GPU"
    device = "cuda"
    t0 = time.time()
    units = a.units.split(",")
    task_of = {"pg1": "pg1", "pi0": "pi0", "pi05": "pi05",
               "ft_max": a.ft_max_task, "ft_min": a.ft_min_task}
    rep = {"object": "erlangen_x_v1", "units": units, "n_items": a.n_items,
           "task_of_unit": task_of, "annotation": AN.manifest(),
           "ft_max_repo": a.ft_max_repo, "ft_min_repo": a.ft_min_repo}

    def bank():
        (out / "erlangen_x.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    groups = P.component_groups(model)
    rep["groups"] = list(groups)
    print(f"[X] shell + {len(groups)} groups ready ({time.time()-t0:.0f}s)", flush=True)
    bank()

    # every unit's FULL state-dict, bf16 — one object serving both the host and the donor role
    sds = {}
    for u in units:
        sds[u] = P.unit_sd(u, pg1_sd,
                           ft_repo=(a.ft_max_repo if u == "ft_max" else
                                    a.ft_min_repo if u == "ft_min" else None))
        print(f"[X] donor {u}: {len(sds[u])} keys ({time.time()-t0:.0f}s)", flush=True)
    donors = {f"unit_{u}": sds[u] for u in units}

    rep["cells"] = {}
    rep["identity_controls"] = {}
    rep["forward_validity"] = {}
    # N3's absolute ceiling is anchored on PG1-in-shell, measured first (critic finding 10)
    P.install_unit(model, "pg1", pg1_sd, sd=sds["pg1"])
    anchor = P.forward_validity(model, proc, device, packs)["wiki_nll"]
    abs_ceiling = P.N3_ABS_CEILING_MULT * anchor
    rep["n3_anchor"] = {"pg1_wiki_nll": anchor, "mult": P.N3_ABS_CEILING_MULT,
                        "abs_ceiling": abs_ceiling}
    print(f"[X] N3 absolute ceiling = {abs_ceiling:.3f} ({P.N3_ABS_CEILING_MULT}x PG1's "
          f"{anchor:.3f})", flush=True)
    for host in units:
        P.install_unit(model, host, pg1_sd, sd=sds[host])
        fv = P.forward_validity(model, proc, device, packs, abs_ceiling=abs_ceiling)
        rep["forward_validity"][host] = fv
        # N3: an invalid host forward makes every cell in its row uninterpretable
        print(f"[X] host={host} valid={fv['valid']} wiki={fv['wiki_nll']:.4f} "
              f"dperc={fv['delta_perception']:+.4f}", flush=True)
        base = P.vec(P.readout_vector(model, proc, device, packs, batch_size=a.batch))
        rep["cells"][f"{host}|__base__"] = base
        bank()
        for g, parts in groups.items():
            # the per-(group, host) damage scale: delete the group in this host
            sv = P.apply_intervention(model, parts, "zero", donors, seed=20260803)
            zero = P.vec(P.readout_vector(model, proc, device, packs, batch_size=a.batch))
            P.restore(model, sv)
            dref = float(np.mean(np.array(zero) - np.array(base)))
            rep["cells"][f"{host}|{g}|__zero__"] = {"vec": zero, "d_ref": dref}
            for donor in units:
                sv = P.apply_intervention(model, parts, f"swap_unit_{donor}", donors,
                                          seed=20260803)
                v = P.vec(P.readout_vector(model, proc, device, packs, batch_size=a.batch))
                P.restore(model, sv)
                dmg = float(np.mean(np.array(v) - np.array(base)))
                compat = 1.0 - max(0.0, dmg) / dref if dref > 0 else float("nan")
                rep["cells"][f"{host}|{g}|{donor}"] = {"vec": v, "damage": dmg, "compat": compat}
                if donor == host:
                    d = max(abs(x - y) for x, y in zip(v, base))
                    rep["identity_controls"][f"{host}|{g}"] = d
                    assert d < 1e-4, (
                        f"P3 SELF-TRANSPLANT FAILED {host}/{g}: readout drift {d} — the surgery "
                        f"is not a no-op; HALT (spec §5-P3)")
            print(f"[X] {host} {g}: d_ref={dref:+.4f} compat="
                  + " ".join(f"{d}:{rep['cells'][f'{host}|{g}|{d}']['compat']:+.2f}"
                             for d in units), flush=True)
            bank()

    # ---- the compatibility matrix + mirror-v2 --------------------------------------------------
    C = []
    for host in units:
        row = []
        for donor in units:
            vals = [rep["cells"][f"{host}|{g}|{donor}"]["compat"] for g in groups]
            vals = [v for v in vals if v == v]
            row.append(float(np.mean(vals)) if vals else float("nan"))
        C.append(row)
    rep["compat_matrix"] = C
    rep["compat_matrix_units"] = units
    rep["compat_by_group"] = {
        g: [[rep["cells"][f"{h}|{g}|{d}"]["compat"] for d in units] for h in units]
        for g in groups}

    keys = [f"{u}:{task_of[u]}" for u in units]
    Csym = (np.array(C) + np.array(C).T) / 2.0          # Mantel needs a symmetric association
    rep["mirror_v2"] = {}
    for tag, ann in (("A", AN.ANNOTATION_A), ("B", AN.ANNOTATION_B)):
        S = np.array(AN.similarity_matrix(keys, ann))
        rep["mirror_v2"][tag] = EC.mantel(S, Csym, n_perm=10000, seed=0)
        rep["mirror_v2"][tag]["similarity_matrix"] = S.tolist()
        print(f"[X] mirror-v2 ({tag}): r={rep['mirror_v2'][tag]['r']:+.3f} "
              f"p={rep['mirror_v2'][tag]['p_two_sided']:.4f}", flush=True)
    rep["p3"] = EC.self_transplant_gate(list(rep["identity_controls"].values()), 1e-4)
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[X] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
