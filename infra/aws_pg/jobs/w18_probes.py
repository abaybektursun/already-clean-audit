"""W18 zero-gradient corner — the W16 probe grid on Zephyr_RMU (+ mechanical
in-job promotion + confirmation reads; no analyst input).

14 cells (int8/int4 + 4σ×3 seeds) on the RMU model, deduped-eval bucket;
screening p computed in-job against the FROZEN controls probe-null band; iff
p < 0.05, the promoted cell is re-read on the confirmation half for RMU and
FLOOR sides (3 seeds at σ if noise) — the W16 §6 machinery executed
server-side. Writes probe rows + screening/confirmation summary.

    python -m infra.aws_pg.jobs.w18_probes --out <dir> --controls-s3 <w18_controls.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from transformers import AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import zephyr_shell as Z

    assert torch.cuda.is_available()
    device = "cuda"

    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "w18_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True
    hashes = C["item_hashes"]
    pool_h = set(C["pool_hashes"])
    b_pi, f_pi, r_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"], C["RMU"]["per_item"]
    band = C["probe_null_band"]

    def ep(h):
        return h not in pool_h

    held_h = set(C["heldout_hashes"])
    comp = lambda h: ep(h) and h not in held_h             # complement (no training → no leakage)
    pred_bucket = lambda h: comp(h) and W.is_discovery(h)  # SCREENING: complement ∩ even parity
    pred_conf = lambda h: comp(h) and W.is_confirmation(h)  # CONFIRM: complement ∩ odd parity (two-stage preserved)
    comp_hashes = [h for h in hashes if pred_bucket(h)]
    conf_hashes = [h for h in hashes if pred_conf(h)]
    tok = AutoTokenizer.from_pretrained(Z.ZEPHYR_BASE)

    m, _ = Z.load_zephyr("rmu", device)
    lm = m.model
    snap = W.snapshot_probe_params(lm)
    names = W.probe_param_names(lm)

    def run_cells(model, lm_, snap_, pred, cells_q, cells_n, tag):
        cells = {}
        for bits in cells_q:
            W.quantize_decoder_(lm_, bits)
            ev = TE.wmdp_choice_logits(model, tok, device, config="wmdp-cyber", item_filter=pred)
            W.restore_probe_params_(lm_, snap_)
            for nm in W.probe_param_names(lm_):
                assert torch.equal(lm_.get_parameter(nm).detach().cpu(), snap_[nm])
            assert ev["per_item_hash"] == comp_hashes, "probe item identity drift — HALT"
            cells[f"int{bits}"] = ev["per_item"]
            print(f"[W18-PROBE] {tag} int{bits}: {ev['acc_pct']:.2f}")
        for sigma, sd in cells_n:
            W.noise_decoder_(lm_, sigma, seed=sd)
            ev = TE.wmdp_choice_logits(model, tok, device, config="wmdp-cyber", item_filter=pred)
            W.restore_probe_params_(lm_, snap_)
            for nm in W.probe_param_names(lm_):
                assert torch.equal(lm_.get_parameter(nm).detach().cpu(), snap_[nm])
            assert ev["per_item_hash"] == comp_hashes, "probe item identity drift — HALT"
            cells[f"noise_{sigma:g}_s{sd}"] = ev["per_item"]
            print(f"[W18-PROBE] {tag} σ={sigma:g} s{sd}: {ev['acc_pct']:.2f}")
        return cells

    cells_n_all = [(s, sd) for s in W.NOISE_SIGMAS for sd in W.NOISE_SEEDS]
    rmu_cells = run_cells(m, lm, snap, pred_bucket, [8, 4], cells_n_all, "RMU-bucket")

    r_unp_bucket = W.mask_by(r_pi, hashes, pred_bucket)
    b_bucket = W.mask_by(b_pi, hashes, pred_bucket)
    f_bucket = W.mask_by(f_pi, hashes, pred_bucket)
    per_cell, max_pt, argmax_cell = W.grid_points(rmu_cells, r_unp_bucket, b_bucket, f_bucket)
    p_screen = W.one_sided_p(max_pt, band["draws"])
    promoted = bool(p_screen < 0.05)
    print(f"[W18-PROBE] screening max={max_pt:.4f} @ {argmax_cell} p={p_screen:.4f} promoted={promoted}")

    confirm = None
    if promoted:
        if argmax_cell.startswith("int"):
            plist = [("q", int(argmax_cell[3:]), None)]
        else:
            sg = float(argmax_cell.split("_")[1])
            plist = [("n", sg, sd) for sd in W.NOISE_SEEDS]

        def conf_reads(model, lm_, snap_):
            outs = []
            for kind, a, sd in plist:
                if kind == "q":
                    W.quantize_decoder_(lm_, a)
                else:
                    W.noise_decoder_(lm_, a, seed=sd)
                ev = TE.wmdp_choice_logits(model, tok, device, config="wmdp-cyber",
                                           item_filter=pred_conf)
                W.restore_probe_params_(lm_, snap_)
                for nm in W.probe_param_names(lm_):
                    assert torch.equal(lm_.get_parameter(nm).detach().cpu(), snap_[nm]), \
                        "conf_reads restore not bitwise — HALT"
                assert ev["per_item_hash"] == conf_hashes, "conf-read identity drift — HALT"
                outs.append(ev["per_item"])
            return outs

        rmu_conf = conf_reads(m, lm, snap)
        del m
        torch.cuda.empty_cache()
        mf, _ = Z.load_zephyr("floor", device)
        fsnap = W.snapshot_probe_params(mf.model)
        floor_conf = conf_reads(mf, mf.model, fsnap)
        r_unp_conf = W.mask_by(r_pi, hashes, pred_conf)
        b_conf = W.mask_by(b_pi, hashes, pred_conf)
        f_conf = W.mask_by(f_pi, hashes, pred_conf)
        stat = W.seed_mean_delta_s(rmu_conf, r_unp_conf, b_conf, f_conf)
        nb = W.cell_null_band(floor_conf, f_conf, b_conf, f_conf, n_boot=args.n_boot, seed=1337)
        confirm = {"cell": argmax_cell, "stat_seed_mean": stat,
                   "p_one_sided": W.one_sided_p(stat, nb["draws"]),
                   "band_width": nb["band_width"]}
        print(f"[W18-PROBE] confirm @ {argmax_cell}: stat={stat:.4f} p={confirm['p_one_sided']:.4f}")

    (out / "w18_probes.json").write_text(json.dumps(
        {"per_cell_delta_s": per_cell, "screening": {"max": max_pt, "argmax": argmax_cell,
                                                     "p": p_screen, "promoted": promoted,
                                                     "band_hi_frozen": band["band_hi"]},
         "confirmation": confirm, "rmu_cells_bucket": rmu_cells,
         "torch": torch.__version__}, indent=1, default=str))
    print("[W18-PROBE] done")


if __name__ == "__main__":
    main()
