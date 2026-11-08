"""RELAW reads — the cheap weight reads {R_prin(r), R_dist} per (PG1-base, FT) pair over the
banked SAV family (the MDE predictor matrix; prereg §3.1/§2.3).

G (= S_recover@16k) and R_ovl (= L9 overlap) are ALREADY banked in the SAV-C pool
(savc_metrics.json); this job adds only the NEW read R_prin = pooled principal-subspace survival
(relaw.principal_survival) by SVD-ing each FT decoder against the shared PG1 base decoder
(SVD'd ONCE, truncated to the top r_max columns, reused across every unit and every r in the
sweep). Pure fp64 SVD + state_dict loads — NO model forward, NO GPU. Generation firewall: every
decoder asserted 18-layer (Gemma-1). Fail-loud.

    python -m infra.aws_pg.jobs.relaw_reads --out <dir> \
        --pool studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json \
        --r-grid 8,16,32,64,128,256,512 [--include-chain pi0,pi05] [--pi05-sealed-s3 s3://...]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess


def _free_hf_cache(unit, kind):
    """Drop a model's HF cache snapshot after its decoder is extracted (only the decoder mattered) —
    keeps peak disk at base + one model (the 200GB root cannot hold all 15 at once). Best-effort
    cleanup, not science: rmtree is tolerant; the SVD/IO that matters already failed loud upstream."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
    name = ("models--google--" + unit) if kind != "chain" else ""
    if name:
        d = os.path.join(hub, name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", required=True, help="savc_metrics.json (local path or s3://)")
    ap.add_argument("--r-grid", default="8,16,32,64,128,256,512")
    ap.add_argument("--include-chain", default="",
                    help="comma list of chain units to ALSO compute (PG1 always on; e.g. pi0,pi05)")
    ap.add_argument("--pi05-sealed-s3", default="")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r_grid = [int(x) for x in args.r_grid.split(",")]
    r_max = max(r_grid)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import relaw as RL
    from infra.pg import w15_carrier as C
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.savb_unit import extract_ft_decoder
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _extract_decoder

    # ---- the SAV-C pool (G + R_ovl + l2_base banked; we add R_prin) ----
    pool_path = args.pool
    if pool_path.startswith("s3://"):
        pool_path = str(out / "savc_metrics.json")
        subprocess.run(["aws", "s3", "cp", args.pool, pool_path, "--region", "us-east-1"], check=True)
    pool = json.loads(pathlib.Path(pool_path).read_text())["pool"]
    by_unit = {u["unit"]: u for u in pool}

    chain_extra = [c.strip() for c in args.include_chain.split(",") if c.strip()]
    # base = PG1 (paligemma-3b-pt-224): R_prin/R_dist isolate the TASK fine-tune's effect on PG1's
    # principal substrate (within-family dynamic range — the point). Family = every fine-tune OF PG1
    # (n5c/n5c_new HF repos) + opt-in chain FTs (pi0/pi05); the PG1 base itself is excluded (overlap
    # with self = 1). R_ovl stays the banked SAV-B activation gauge (vs gemma-2b) — the comparator.
    units = [u["unit"] for u in pool if u["kind"] in ("n5c", "n5c_new")]
    units += [u for u in chain_extra if u in by_unit and u != "PG1"]
    print(f"[RELAW-READS] base=PG1 (paligemma-3b-pt-224); {len(units)} fine-tune units, "
          f"r_grid={r_grid}", flush=True)

    pi05_local = ""
    if args.pi05_sealed_s3:
        pi05_local = str(out / "pi05_decoder.safetensors")
        subprocess.run(["aws", "s3", "cp", args.pi05_sealed_s3, pi05_local, "--region", "us-east-1"],
                       check=True)

    def truncated_svd(W):
        """fp64 SVD truncated to the top r_max columns of U/V (all we ever slice). float64."""
        U, _S, V = C.svd_fp64(W)
        k = min(r_max, U.shape[1], V.shape[1])
        return U[:, :k].contiguous(), V[:, :k].contiguous()

    # ---- base PG1 (paligemma-3b-pt-224) decoder: extracted the SAME way as the FTs (one loader,
    # one key convention), SVD'd once, truncated factors cached + reused across all units + all r ----
    ent = ENTITIES["PG1"]
    dec_b_raw, base_sha = extract_ft_decoder(ent.repo_id)
    dec_b = {k: v.detach() for k, v in dec_b_raw.items()}
    del dec_b_raw
    base2d = sorted(k for k, v in dec_b.items() if v.ndim == 2)
    nlb = RL.n_decoder_layers(base2d)
    assert nlb == 18, f"generation firewall: PG1 base must be 18-layer (Gemma-1), got {nlb}"
    print(f"[RELAW-READS] base PG1: {len(base2d)} 2-D mats, {nlb} layers — SVD-ing once...", flush=True)
    base_fac, base_fro2 = {}, {}
    for k in base2d:
        Wb = dec_b[k].float().double()
        Ub, Vb = truncated_svd(Wb)
        base_fac[k] = (Ub, Vb)
        base_fro2[k] = float((Wb ** 2).sum())
    _free_hf_cache(ent.repo_id.split("/")[-1], "n5c")     # base decoder tensors live in RAM now
    print(f"[RELAW-READS] base SVD cached (top-{r_max}); base disk cache freed.", flush=True)

    def load_ft(unit):
        if by_unit[unit]["kind"] == "chain":
            dec, sha = _extract_decoder(unit, "cpu", pi05_local)
            assert dec is not None, f"chain decoder {unit} returned None"
            return dec, str(sha)
        dec, sha = extract_ft_decoder(f"google/{unit}")
        return dec, str(sha)

    rows = []
    for unit in units:
        dec_u, src_sha = load_ft(unit)
        dec_u = {k: v.detach() for k, v in dec_u.items()}
        shared = [k for k in base2d if k in dec_u and dec_u[k].ndim == 2
                  and tuple(dec_u[k].shape) == tuple(dec_b[k].shape)]
        nlu = RL.n_decoder_layers(shared)
        assert nlu == 18, f"{unit}: firewall — expected 18 layers, got {nlu}"
        assert len(shared) == len(base2d), \
            f"{unit}: {len(shared)}/{len(base2d)} 2-D mats shared — decoder key drift, HALT"
        rprin_sum = {r: 0.0 for r in r_grid}
        dW2, W2 = 0.0, 0.0
        for k in shared:
            Wf = dec_u[k].float().double()
            Uf, Vf = truncated_svd(Wf)
            Ub, Vb = base_fac[k]
            for r in r_grid:
                rprin_sum[r] += RL.overlap_at_rank(Ub, Vb, Uf, Vf, r)
            dW2 += float(((Wf - dec_b[k].float().double()) ** 2).sum())
            W2 += base_fro2[k]
            del Wf, Uf, Vf
        n_mat = len(shared)
        rprin = {str(r): rprin_sum[r] / n_mat for r in r_grid}
        r_dist = float(np.sqrt(dW2) / np.sqrt(W2))
        banked = by_unit[unit]
        row = {"unit": unit, "kind": banked["kind"], "source_sha": src_sha,
               "G": float(banked["S_16k"]), "G_S1k": float(banked.get("S_1k", float("nan"))),
               "R_ovl": float(banked["overlap"]), "R_dist_banked": float(banked["l2_base"]),
               "R_dist": r_dist, "R_prin": rprin, "n_mat": n_mat}
        rows.append(row)
        print(f"[RELAW-READS] {unit:38s} G={row['G']:+.3f} R_ovl={row['R_ovl']:.3f} "
              f"R_dist={r_dist:.4f} R_prin@{r_max}={rprin[str(r_max)]:.4f} "
              f"R_prin@8={rprin['8']:.4f}", flush=True)
        del dec_u
        _free_hf_cache(unit, by_unit[unit]["kind"])      # free disk before the next download

    rep = {"object": "relaw_reads_v1", "family": "sav_pg1", "generation": "gemma1_18L",
           "n_units": len(rows), "r_grid": r_grid, "base_repo": ent.repo_id,
           "base_revision": ent.revision, "base_sha": str(base_sha), "pool_src": args.pool,
           "units": rows, "torch": torch.__version__}
    (out / "reads_sav.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[RELAW-READS] OK — {len(rows)} units → reads_sav.json", flush=True)
    print(f"RELAW_READS_OK n_units={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
