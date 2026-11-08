"""SPECTER Arm M sealed (GPU, run ONCE) — the rematch. Runs the frozen ordered cell list on the
`specter|` SEALED half until 20 valid cells (VOIDs from the registered 30-backup list, every
VOID logged), then unblinds once: Δρ = ρ(M4) − ρ(M0) per readout (10k pair bootstrap), BH
q=0.10 over {ppl, vqa} with emitted p-values, R-M1 adjudication. The LAST GPU event of the
program (spec §3.3). No rule swap; the sealed half is touched exactly here.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_m_sealed --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/specter_m_sealed --budget-min 300
"""
import argparse
import json
import os
import pathlib

N_SCORED = 20
N_SEALED_ITEMS = 200


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("m_sealed")
    RS.install_pg()
    rep = {"object": "specter_m_sealed_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        fz = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_m_fit/out/m_freeze.json",
                                 "/tmp/specter/m_freeze.json").read_text())
        mf = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_m_fit/out/metrics.json",
                                 "/tmp/specter/mf.json").read_text())
        g0 = json.loads(SS.fetch(f"{SS.SPECTER_GATE0}/gate0.json",
                                 "/tmp/specter/g0.json").read_text())
        assert mf.get("green") and g0.get("green"), "parents not green — refuse"
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        sig = SS.load_sigma_bank("L0", workdir="/tmp/specter")
        bands = BC.eig_bands(sig["Sigma"])
    d = IP.D
    dev = "cuda"
    sb = [tuple(x) for x in fz["subband_slices"]]
    rms = g0["identity"]["site_rms"]

    with tm.stage("scaffold"):
        vqa_task, _ = IP.pg1_tasks()
        from infra.aws_pg.jobs._regrain_shared import GRAIN_GATE0
        it = json.loads(SS.fetch(f"{GRAIN_GATE0}/items_vqa.json",
                                 "/tmp/specter/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa_task, it["stream_window"])
        split = SS.specter_split(hashes)
        sealed_rows = [rows[i] for i in split["conf_idx"]][:N_SEALED_ITEMS]
        intact = IP.eval_pg1(model, proc, vqa_task, sealed_rows, dev)
        floor = IP.eval_pg1(model, proc, vqa_task, sealed_rows, dev, scramble_seed=4321)["score"]
        base_ppl = IP.ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)["ppl"]
        assert abs(base_ppl - SS.FUNC_BASE_PPL) < SS.PPL_TOL
        rep["reference"] = {"intact_vqa": intact["score"], "floor": floor, "base_ppl": base_ppl}
        runner = SS.SpecterCellRunner(model, proc, vqa_task, sealed_rows, intact, floor,
                                      base_ppl, rep, _w)
        env = {"L": 0, "layers": layers, "basis": bands["basis"], "sb_slices": sb,
               "hbar": torch.as_tensor(sig["hbar"]), "rms": rms, "d": d,
               "channel_rms": g0["channel_rms_at_site"]["measured"],
               "head_dim": g0["model"]["head_dim"], "n_heads": g0["model"]["n_heads"],
               "rope_theta": g0["model"]["rope_theta"], "seed": 33}
    _w()

    # ---- the battery: ordered list until 20 valid --------------------------------------
    with tm.stage("battery"):
        scored, voids = [], []
        span = intact["score"] - floor
        for entry in fz["sealed_ordered"]:
            if len(scored) >= N_SCORED:
                break
            cell = entry["cell"]
            ctx = AS.grammar_cell_ctx(cell, env)
            rec = runner.run(f"sealed/{entry['key']}", ctx, family="sealed", arm="sealed",
                             rank=entry["rank"])
            if bool(rec.get("saturated_dead")):
                voids.append({"rank": entry["rank"], "key": entry["key"],
                              "reason": "saturated_dead (vqa at/below scrambled floor)"})
                continue
            scored.append(dict(entry, measured_log1p_dppl=rec["log1p_dppl"],
                               measured_vqa_effect=rec["effect"]))
        rep["scored_n"] = len(scored)
        rep["voids"] = voids
        rep["scored"] = scored
        assert len(scored) + len(voids) <= len(fz["sealed_ordered"])
    _w()

    # ---- unblind once -------------------------------------------------------------------
    with tm.stage("unblind"):
        res = {}
        pvals, labels = [], []
        for ro, mkey in (("ppl", "measured_log1p_dppl"), ("vqa", "measured_vqa_effect")):
            y = np.array([s[mkey] for s in scored], float)
            m4 = np.array([s["pred"][f"M4_{ro}"] for s in scored], float)
            m0 = np.array([s["pred"][f"M0_{ro}"] for s in scored], float)
            drho, lo, hi, r4, r0 = AF.pair_bootstrap_drho(m4, m0, y)
            _, p4 = AF.perm_p_spearman(m4, y)
            pvals.append(p4)
            labels.append(ro)
            res[ro] = {"drho": drho, "ci": [lo, hi], "rho_m4": r4, "rho_m0": r0,
                       "p_m4_orders": p4,
                       "adjudication": SC.adjudicate_rm1(
                           drho=drho, drho_cilo=lo, drho_cihi=hi, rho_m4=r4, rho_m0=r0,
                           n_void_backups=len(voids))}
            print(f"[m] {ro}: drho={drho:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"rho_m4={r4:.3f} rho_m0={r0:.3f} -> {res[ro]['adjudication']['verdict']}",
                  flush=True)
        SS.emit_family(rep, "rm1", pvals, labels)
        rep["rm1"] = res
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_M_SEALED_OK", json.dumps({ro: res[ro]["adjudication"]["verdict"]
                                             for ro in res}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
