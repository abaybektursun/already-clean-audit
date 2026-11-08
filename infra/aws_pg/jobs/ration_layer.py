"""RATION Arm L (GPU) — the labelled TRANSFER arm (A-RATION-1): apply the L0-Q susceptibility BASIS at
every one of the 18 decoder layers and ask where the L0 map retains leverage. Each layer gets its own
layer-matched quantiser ranges + its own Σ_x eigenbasis (fair basis-transfer test); only the
susceptibility basis+λ are the fixed L0 object. Reports per-layer excess E = susceptibility −
sink-aware variance at the frozen b̄* (ppl + vqa). NOT a per-layer matched operator (none is banked).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_layer --type p5.4xlarge,g6e.8xlarge \
      --s3-prefix results/ration_layer --budget-min 90
"""
import argparse
import json
import os
import pathlib
import time


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _ration_shared as RS
    from infra.pg import ration_core as RC

    t00 = time.time()
    rep = {"object": "ration_layer_v1"}

    def dump():
        (out / "ration_layer.json").write_text(json.dumps(rep, indent=2, default=float))

    def stage(m):
        print(f"[ration-layer] {m} (+{time.time() - t00:.0f}s)", flush=True)

    banks0 = torch.load(RS.fetch(f"{RS.RATION_DISC}/ration_banks.pt",
                                 pathlib.Path("/tmp/ration/ration_banks.pt")),
                        map_location="cpu", weights_only=False)
    fr = banks0["freeze"]
    bstar_ppl, bstar_vqa = fr["bstar_ppl"], fr["bstar_vqa"]
    model, proc, layers, sha, base_ppl = RS.load_model()
    ppl_disc, ppl_conf = RS.wikitext_split(proc)
    task, vqa_disc, vqa_conf = RS.vqa_task_and_split()
    rep["freeze"] = fr
    stage(f"loaded banks; per-layer Σ_x pass over {len(ppl_disc)} windows")

    # per-layer Σ_x in ONE pass, MIX calibration (ppl + VQA), discovery only (no confirmation peek)
    sig_all = RS.accumulate_sigma_all_layers(model, proc, layers, ppl_disc, vqa_rows=vqa_disc, task=task)

    # clean reference (passthrough at each layer is identical — one clean pass suffices)
    sub_i0, sub_h0 = RC.make_residual_substitutor(layers, 0)
    sub_h0["spec"] = None
    clean_vqa = RS.vqa_per_item(model, proc, task, vqa_conf, sub_i0, n=RS.N_VQA_CELL)
    clean_ppl_pw, _ = RS.ppl_per_window(model, proc, ppl_conf, sub_i0, sub_h0, collect_mse=False)
    g_clean_ppl, g_clean_vqa = RC.ppl_goodness(clean_ppl_pw), [float(x) for x in clean_vqa]

    def banks_for_layer(L):
        s = sig_all[L]
        mean_x, C_x, absmean_x = RC.finalize_sigma(s["gram"], s["sum"], s["abs"], s["n"])
        sev, sea = RC._sigma_eig(C_x)
        b = {"Q_evecs": banks0["Q_evecs"], "Q_evals": banks0["Q_evals"],
             "sigma_evecs": sev, "sigma_evals": sea.numpy(), "mean_x": mean_x, "C_x": C_x,
             "absmean_x": absmean_x.numpy(), "seed": banks0["seed"]}
        b["sink_dirs"] = RC.sink_eigendirections(sev)
        return b

    per_layer = {}
    for L in range(len(layers)):
        bl = banks_for_layer(L)
        cells = {}
        for policy in ("susceptibility", "sink_aware_variance", "variance", "uniform"):
            basis = RS.precompute_basis_dev(policy, bl)
            raw = {}
            for tag, b in (("ppl", bstar_ppl), ("vqa", bstar_vqa)):
                if b is None:
                    continue
                cell = RS.run_cell(policy, b, bl, model, proc, task, ppl_conf, vqa_conf, layers, L=L,
                                   precomputed_basis=basis, n_vqa=RS.N_VQA_CELL,
                                   do_vqa=(tag == "vqa"))
                if tag == "ppl":
                    raw["ppl"] = RC.ppl_goodness(cell["ppl_per_window"])
                else:
                    raw["vqa"] = [float(x) for x in cell["vqa_per_item"]]
            cells[policy] = raw
        row = {"L": L, "n_sink": len(bl["sink_dirs"])}
        if bstar_ppl is not None:
            row["E_ppl"] = RC.paired_excess(cells["susceptibility"]["ppl"],
                                            cells["sink_aware_variance"]["ppl"])
            row["susc_ppl_loss"] = RC.loss_ci(g_clean_ppl, cells["susceptibility"]["ppl"])["E"]
            row["uniform_ppl_loss"] = RC.loss_ci(g_clean_ppl, cells["uniform"]["ppl"])["E"]
        if bstar_vqa is not None:
            row["E_vqa"] = RC.paired_excess(cells["susceptibility"]["vqa"],
                                            cells["sink_aware_variance"]["vqa"])
            row["susc_vqa_loss"] = RC.loss_ci(g_clean_vqa, cells["susceptibility"]["vqa"])["E"]
            row["uniform_vqa_loss"] = RC.loss_ci(g_clean_vqa, cells["uniform"]["vqa"])["E"]
        per_layer[str(L)] = row
        stage(f"  L{L}: E_ppl={row.get('E_ppl',{}).get('E') if 'E_ppl' in row else None} "
              f"E_vqa={row.get('E_vqa',{}).get('E') if 'E_vqa' in row else None}")
        dump()
    rep["per_layer"] = per_layer
    rep["wall_seconds"] = round(time.time() - t00, 1)
    dump()
    print("RATION_LAYER_OK", json.dumps({"layers": len(per_layer)}))


if __name__ == "__main__":
    main()
