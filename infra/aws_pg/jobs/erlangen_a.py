"""ERLANGEN Arm A — commutation / absorption over K = 24 activation interventions (amendment A2).

Banks the full ordered K×K behaviour tensor plus both nulls, then applies the frozen P2 rule:

  positive control : layer-ORDERED pairs must commute LESS than the exact-commutation floor, and
                     the directed parameter shift must carry the information-flow sign
  Null-1 (floor)   : same-locus ORTHOGONAL random subspaces, parameters harvested once on the base
                     — algebraically commuting, so whatever is measured is pipeline noise
  Null-2 (ceiling) : the same pair WITH re-harvest. If re-harvest alone produces commutators as
                     large as the positive control, the control could not have failed → arm VOID.

    python -m infra.aws_pg.jobs.erlangen_a --out DIR [--n-items 32] [--loci 0,2,4,6,9,12,15,17]

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
    ap.add_argument("--k-sub", type=int, default=32)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = the full ordered K×K grid")
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

    from infra.pg import erlangen_algebra as A
    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.inherit_pg1 import decoder_layers
    from infra.pg.lacuna_shell import load_pg1

    assert torch.cuda.is_available(), "Arm A needs a GPU"
    device = "cuda"
    t0 = time.time()
    A.K_SUB = a.k_sub
    ivs = A.intervention_ids()
    rep = {"object": "erlangen_a_v1", "K": len(ivs), "interventions": ivs,
           "loci": list(A.LOCI), "kinds": list(A.KINDS), "k_sub": a.k_sub,
           "n_items": a.n_items, "readouts": list(P.READOUT_NAMES)}

    def bank():
        (out / "erlangen_a.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    layers = decoder_layers(model)
    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    print(f"[A] shell + items ready ({time.time()-t0:.0f}s)", flush=True)

    harvest_items = packs["cap_real"][:16]
    hbatches = [harvest_items[i:i + 8] for i in range(0, len(harvest_items), 8)]

    def forward_fn(batch):
        inp = proc(text=[b["prompt"] for b in batch], images=[b["image"] for b in batch],
                   return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            model(**inp)

    def b_of(seq, reharvest=True, fixed=None):
        handles, params = A.realize(model, layers, seq, forward_fn, hbatches,
                                    seed=20260803, reharvest=reharvest, fixed=fixed)
        r = P.readout_vector(model, proc, device, packs, batch_size=a.batch)
        A.remove(handles)
        return P.vec(r), params

    base, _ = b_of([])
    rep["base"] = base
    bank()

    # ---- singles ------------------------------------------------------------------------------
    rep["singles"] = {}
    base_Q = {}
    for iv in ivs:
        v, params = b_of([iv])
        rep["singles"][iv] = v
        base_Q[iv] = params[iv]["Q"].cpu()
        print(f"[A] single {iv}: ‖Δb‖={np.linalg.norm(np.array(v)-np.array(base)):.4f}", flush=True)
        bank()

    # ---- the ordered K×K grid ------------------------------------------------------------------
    pairs = [(i, j) for i in ivs for j in ivs if i != j]
    if a.max_pairs:
        pairs = pairs[: a.max_pairs]
    rep["n_ordered_pairs"] = len(pairs)
    rep["pairs"] = {}
    rep["param_shift"] = {}
    tp = time.time()
    for n, (i, j) in enumerate(pairs):
        # i∘j: j is applied FIRST, then i is harvested under j
        v, params = b_of([j, i])
        rep["pairs"][f"{i}|{j}"] = v
        rep["param_shift"][f"{i}|{j}"] = A.subspace_shift(base_Q[i], params[i]["Q"].cpu())
        if (n + 1) % 20 == 0 or n == 0:
            per = (time.time() - tp) / (n + 1)
            print(f"[A] pair {n+1}/{len(pairs)} {i}∘{j} {per:.1f}s/pair "
                  f"eta {per*(len(pairs)-n-1)/60:.0f}min", flush=True)
            bank()
    bank()

    # ---- nulls ---------------------------------------------------------------------------------
    rep["null_exact"], rep["null_reharvest"] = {}, {}
    for L in A.LOCI:
        Qa, Qb = A.orthogonal_random_pair(k=a.k_sub, seed=1000 + L)
        ia, ib = f"L{L}|randA", f"L{L}|randB"
        fixed = {ia: Qa, ib: Qb}
        for tag, rh in (("null_exact", False), ("null_reharvest", True)):
            v1, _ = b_of([ia, ib], reharvest=rh, fixed=fixed)
            v2, _ = b_of([ib, ia], reharvest=rh, fixed=fixed)
            rep[tag][f"L{L}"] = EC.commutator(v1, v2)
        print(f"[A] L{L} null_exact={rep['null_exact'][f'L{L}']:.5f} "
              f"null_reharvest={rep['null_reharvest'][f'L{L}']:.5f}", flush=True)
        bank()

    # ---- matrices + the frozen verdict -----------------------------------------------------------
    comm, absorb, ordered_c, flow = {}, {}, [], []
    for i in ivs:
        for j in ivs:
            if i == j or f"{i}|{j}" not in rep["pairs"] or f"{j}|{i}" not in rep["pairs"]:
                continue
            c = EC.commutator(rep["pairs"][f"{i}|{j}"], rep["pairs"][f"{j}|{i}"])
            comm[f"{i}|{j}"] = c
            absorb[(i, j)] = EC.absorption(rep["pairs"][f"{j}|{i}"], rep["singles"][i],
                                           rep["singles"][j], base)
            Li, Lj = A.parse_id(i)[0], A.parse_id(j)[0]
            if Li < Lj:
                ordered_c.append(c)
                # flow sign: does the DOWNSTREAM member's own subspace move more when harvested
                # under the upstream one than vice versa?
                down = rep["param_shift"].get(f"{j}|{i}")     # j (downstream) harvested under i
                up = rep["param_shift"].get(f"{i}|{j}")       # i (upstream) harvested under j
                if down is not None and up is not None:
                    flow.append(1.0 if down > up else -1.0)
    rep["commutators"] = comm
    rep["absorption"] = {f"{i}|{j}": v for (i, j), v in absorb.items()}
    rep["dependency_graph"] = EC.dependency_graph(absorb)
    rep["verdict"] = EC.algebra_verdict(ordered_c, list(rep["null_exact"].values()),
                                        list(rep["null_reharvest"].values()), flow or [0.0])
    rep["n_ordered_commutators"] = len(ordered_c)
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[A] DONE verdict={rep['verdict']['verdict']} "
          f"median_pos={rep['verdict']['median_positive']:.4f} "
          f"floor={rep['verdict']['null_exact_p97_5']:.5f} "
          f"reharvest={rep['verdict']['median_null_reharvest']:.4f} "
          f"edges={rep['dependency_graph']['n_edges']} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
