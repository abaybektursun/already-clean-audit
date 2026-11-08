"""RATION confirmation (GPU) — spec §9 step 4+5: load the FROZEN discovery banks/selections, run the
full rate-distortion sweep on the SEALED confirmation half (the headline figure), compute the sealed
paired excess E at (L*, b̄*) for both readouts, the mirror MSE, the sink adjudication, the random
null, the oracle headroom, and apply `adjudicate_ration` (§6 frozen order). Writes ration_conf.json +
ration_rate.json. The analyst logged predictions in the journal BEFORE this job launched.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_conf --type p5.4xlarge,g6e.8xlarge \
      --s3-prefix results/ration_conf --budget-min 100
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
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC

    t00 = time.time()
    rep = {"object": "ration_conf_v1", "L_star": RS.L_STAR}

    def dump():
        (out / "ration_conf.json").write_text(json.dumps(rep, indent=2, default=float))

    def stage(m):
        print(f"[ration-conf] {m} (+{time.time() - t00:.0f}s)", flush=True)

    # ---- load frozen banks + model ----
    bpath = RS.fetch(f"{RS.RATION_DISC}/ration_banks.pt", pathlib.Path("/tmp/ration/ration_banks.pt"))
    banks = torch.load(bpath, map_location="cpu", weights_only=False)
    fr = banks["freeze"]
    rep["freeze_loaded"] = fr
    bstar_ppl, bstar_vqa = fr["bstar_ppl"], fr["bstar_vqa"]
    assert bstar_ppl is not None or bstar_vqa is not None, "no sensitive budget frozen — VOID-no-range at disc"
    model, proc, layers, sha, base_ppl = RS.load_model()
    assert sha == banks["pinned_sha"], f"sha drift disc={banks['pinned_sha'][:8]} conf={sha[:8]}"
    stage(f"loaded frozen banks (b̄*_ppl={bstar_ppl} b̄*_vqa={bstar_vqa} sink={len(fr['sink_dirs'])})")

    # ---- confirmation half (deterministic split — same salt as discovery) ----
    ppl_disc, ppl_conf = RS.wikitext_split(proc)
    task, vqa_disc, vqa_conf = RS.vqa_task_and_split()
    rep["n"] = {"ppl_conf": len(ppl_conf), "vqa_conf": len(vqa_conf)}

    # clean reference on the SEALED half (passthrough)
    sub_i, sub_h = RC.make_residual_substitutor(layers, RS.L_STAR)
    sub_h["spec"] = None
    clean_vqa = RS.vqa_per_item(model, proc, task, vqa_conf, sub_i, n=None)
    clean_ppl_pw, _ = RS.ppl_per_window(model, proc, ppl_conf, sub_i, sub_h, collect_mse=False)
    g_clean_ppl = RC.ppl_goodness(clean_ppl_pw)
    g_clean_vqa = [float(x) for x in clean_vqa]
    rep["clean"] = {"vqa_acc": float(np.mean(clean_vqa)),
                    "ppl": float(np.exp(sum(s for s, _ in clean_ppl_pw) / sum(n for _, n in clean_ppl_pw)))}
    stage(f"sealed clean vqa={rep['clean']['vqa_acc']:.4f} ppl={rep['clean']['ppl']:.4f}")

    # ---- full confirmation sweep (7 policies × 5 budgets); keep raw at the b̄* cells ----
    sweep, raw = {}, {}
    for policy in RC.POLICIES:
        basis = RS.precompute_basis_dev(policy, banks)
        sweep[policy] = {}
        for b in RC.BUDGET_GRID:
            cell = RS.run_cell(policy, b, banks, model, proc, task, ppl_conf, vqa_conf, layers,
                               precomputed_basis=basis, n_vqa=None)
            g_ppl = RC.ppl_goodness(cell["ppl_per_window"])
            g_vqa = [float(x) for x in cell["vqa_per_item"]]
            sweep[policy][str(b)] = {"ppl_loss": RC.loss_ci(g_clean_ppl, g_ppl),
                                     "vqa_loss": RC.loss_ci(g_clean_vqa, g_vqa),
                                     "mse": cell["mse"], "vqa_acc": float(np.mean(g_vqa))}
            if b in (bstar_ppl, bstar_vqa):
                raw.setdefault(policy, {})[str(b)] = {"g_ppl": g_ppl, "g_vqa": g_vqa, "mse": cell["mse"]}
            stage(f"  {policy}@{b}b ppl_loss={sweep[policy][str(b)]['ppl_loss']['E']:+.4f} "
                  f"vqa_loss={sweep[policy][str(b)]['vqa_loss']['E']:+.4f} mse={cell['mse']:.4g}")
        dump()
    rep["confirmation_sweep"] = sweep
    (out / "ration_rate.json").write_text(json.dumps(
        {"sweep": sweep, "clean": rep["clean"], "freeze": fr, "budget_grid": list(RC.BUDGET_GRID)},
        indent=2, default=float))

    # ---- sealed adjudication per readout ----
    def adjudicate(readout, bstar):
        if bstar is None:
            return {"verdict": "VOID-no-range", "reason": "no sensitive budget frozen for this readout"}
        gk = "g_ppl" if readout == "ppl" else "g_vqa"
        s = str(bstar)
        E_sa = RC.paired_excess(raw["susceptibility"][s][gk], raw["sink_aware_variance"][s][gk])
        E_nv = RC.paired_excess(raw["susceptibility"][s][gk], raw["variance"][s][gk])
        uni = sweep["uniform"][s][f"{readout}_loss"]
        orc = {"E": uni["E"] - sweep["oracle"][s][f"{readout}_loss"]["E"]}   # uniform − oracle > 0
        dr = fr[f"delta_range_{readout}"]
        rnd = sweep["random"][s][f"{readout}_loss"]["E"]
        random_saturated = bool(abs(rnd) < dr and abs(uni["E"]) < dr)
        # mirror holds (variance is MSE-optimal) when variance_mse < susceptibility_mse, i.e. the
        # excess susceptibility_mse − variance_mse is POSITIVE. adjudicate halts if val ≤ 0.
        mirror = {"val": raw["susceptibility"][s]["mse"] - raw["variance"][s]["mse"]}
        verdict = RC.adjudicate_ration(E_vs_sinkaware=E_sa, E_vs_naive=E_nv, uniform_loss=uni,
                                       oracle_headroom=orc, delta_range=dr,
                                       random_saturated=random_saturated, mirror_mse_excess=mirror)
        # fraction of oracle headroom captured by susceptibility (diagnostic)
        susc_gain = uni["E"] - sweep["susceptibility"][s][f"{readout}_loss"]["E"]
        verdict["oracle_headroom"] = orc
        verdict["susceptibility_headroom_fraction"] = (susc_gain / orc["E"]) if orc["E"] > 0 else None
        verdict["bstar"] = bstar
        return verdict

    rep["verdict_ppl"] = adjudicate("ppl", bstar_ppl)
    rep["verdict_vqa"] = adjudicate("vqa", bstar_vqa)
    rep["wall_seconds"] = round(time.time() - t00, 1)
    dump()
    stage(f"VERDICT ppl={rep['verdict_ppl']['verdict']} · vqa={rep['verdict_vqa']['verdict']}")
    print("RATION_CONF_OK", json.dumps({"ppl": rep["verdict_ppl"]["verdict"],
                                        "vqa": rep["verdict_vqa"]["verdict"]}))


if __name__ == "__main__":
    main()
