"""N6 L2 projection at L1's measured fp32 rate (spec section 9.5 S3: center at the measured rate, high 1.25 times the center; the
hard kill 1.3 times the high projection; the job's stop rule --budget-min 0.9 times the hard kill), on L2's eval set (amendment A4:
1,024 windows) and L2's roster (A4: the positive control in 8 layers).

The edit rate is the FLOP count of L1's edits (n6.flops_from, the spec's FLOP model) over their measured seconds (edits.jsonl). L1's
seconds outside the edits are split into a part that scales with the eval windows and a fixed part (boot, download, load, K1, K2,
G-POLE, cast, K3, records). The window part at 96 windows: base pass 1 (measured), base pass 2 (taken equal to pass 1), the three bf16
passes of the rounding probes (FLOPs at four times the edit rate, the spec's rule), the progression's layer advance and G-CACHE's two
heads per window (FLOPs at the edit rate). L2 = the fixed part + the window part x (W_L2 / 96) + G-INST's pass over L1's 96 windows
(taken equal to pass 1) + the host-device copies of the kept residual (A4; not measured: 6 GB/s assumed) + L2's edit FLOPs at the
edit rate.

Amendment A5: L2 runs as two launches, L2a (the freeze half, n6.plan(half="freeze")) and L2b (the test half). Each launch pays the
whole setup (the fixed part, the window part, the G-INST pass, the residual copies) and its own edits at the edit rate; the two pair
probes and the zero edit run in both. The same rule gives each launch's high projection, hard kill and job stop rule.

    PYTHONPATH=. python3 studies/next/n6/projection_l1.py --launch <EC2 LaunchTime> --end <STATUS time> [--bank studies/next/n6/banks/<attempt>]
        [--pairs studies/next/n6/pairs_L2.json] [--out <json>]
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402
from infra.aws_pg.jobs import n6_edit as J  # noqa: E402

HERE = pathlib.Path("studies/next/n6")
COPY_BYTES_PER_S = 6e9          # host-device copies of the kept residual, pageable memory (assumed, not measured)
GIB = 2 ** 30


def t(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--bank", default=str(HERE / "banks/L1-02"))
    ap.add_argument("--pairs", default=str(HERE / "pairs_L2.json"))
    ap.add_argument("--out", default=str(HERE / "projection_L1.json"))
    a = ap.parse_args()
    BANK = pathlib.Path(a.bank)
    m = json.loads((BANK / "metrics.json").read_text())
    recs = [json.loads(x) for x in open(BANK / "edits.jsonl") if x.strip()]
    pairs = json.loads(pathlib.Path(a.pairs).read_text())
    w1, w2 = m["w_eval"], len(pairs["eval_windows"])
    assert w1 == n6.W_EVAL and w2 == pairs["w_eval"] == n6.W_EVAL_L2, (w1, w2)
    tok1, tok2 = w1 * (n6.HW + 1), w2 * (n6.HW + 1)
    fl = sum(n6.flops_from(r["layer"], tok1) for r in recs)
    sec = sum(r["seconds"] for r in recs)
    rate = fl / sec
    pass_s = m["measured"]["pass_s"]
    rate_pass = n6.flops_from(0, tok1) / pass_s
    up = (t(a.end) - t(a.launch)).total_seconds()
    win1 = {"base pass 1 (measured)": pass_s, "base pass 2 (taken equal to pass 1)": pass_s,
            "bf16 rounding probes, 3 bf16 passes (FLOPs at 4 x the edit rate)": 3 * n6.flops_from(0, tok1) / (4 * rate),
            "progression layer advance, 32 layers (FLOPs at the edit rate)": tok1 * n6.N_LAYERS * (n6.FLOP_LAYER + n6.FLOP_ATTN) / rate,
            "G-CACHE, two heads per window (FLOPs at the edit rate)": 2 * tok1 * n6.FLOP_HEAD / rate}
    fixed = up - sec - sum(win1.values())
    resid = w2 * (n6.HW + 1) * n6.HIDDEN * 4
    copies = (2 * n6.N_LAYERS + 3) * resid                     # 32 layers in and out, the embedding written, G-CACHE's two reads
    add = {"G-INST pass over L1's 96 windows (taken equal to pass 1)": pass_s,
           f"host-device copies of the kept residual, {copies / 1e9:.0f} GB at {COPY_BYTES_PER_S / 1e9:.0f} GB/s (assumed)": copies / COPY_BYTES_PER_S}
    pl = n6.plan(pairs)
    roles = {}
    for e in pl:
        roles[e["role"]] = roles.get(e["role"], 0) + 1
    fl2 = sum(n6.flops_from(e["layer"], tok2) for e in pl)
    units = sum(n6.N_LAYERS - e["layer"] + 2 for e in pl)
    win2 = {k: v * w2 / w1 for k, v in win1.items()}
    center = fixed + sum(win2.values()) + sum(add.values()) + fl2 / rate

    def at(w):                                                  # the same model at w eval windows (for the head; not the plan)
        return fixed + sum(win1.values()) * w / w1 + pass_s + (2 * n6.N_LAYERS + 3) * w * (n6.HW + 1) * n6.HIDDEN * 4 / \
            COPY_BYTES_PER_S + sum(n6.flops_from(e["layer"], w * (n6.HW + 1)) for e in pl) / rate
    pv = json.loads((HERE / "diag/L1-02/a4_preview.json").read_text())["G_POWER"]["ratio"]
    table = [{"w_eval": w, "center_gpu_h": round(at(w) / 3600, 2), "high_gpu_h": round(1.25 * at(w) / 3600, 2),
              "sd_scale": round(math.sqrt(w1 / w), 3), "G_POWER_ratio_if_D_held": round(pv * math.sqrt(w1 / w), 3)}
             for w in (96, 256, 512, 774, 1024)]
    high = 1.25 * center
    kill = math.ceil(1.3 * high / 60)
    setup = fixed + sum(win2.values()) + sum(add.values())     # every L2 launch pays this besides its edits (A5)

    def launch(half):                                           # amendment A5: one launch of the split
        p_ = n6.plan(pairs, half=half)
        r_ = {}
        for e in p_:
            r_[e["role"]] = r_.get(e["role"], 0) + 1
        es = sum(n6.flops_from(e["layer"], tok2) for e in p_) / rate
        c_ = setup + es
        k_ = math.ceil(1.3 * 1.25 * c_ / 60)
        return {"half": half, "n_edits": len(p_), "by_role": r_, "edit_layer_units": sum(n6.N_LAYERS - e["layer"] + 2 for e in p_),
                "edit_s_center": es, "setup_s": setup, "center_s": c_, "high_s": 1.25 * c_, "center_gpu_h": round(c_ / 3600, 3),
                "high_gpu_h": round(1.25 * c_ / 3600, 3), "usd_center_at_2.24": round(2.24 * c_ / 3600, 2),
                "usd_high_at_2.24": round(2.24 * 1.25 * c_ / 3600, 2), "usd_high_at_3.00424": round(3.00424 * 1.25 * c_ / 3600, 2),
                "hard_kill_min": k_, "job_budget_min": math.floor(0.9 * k_)}
    l2a, l2b = launch("freeze"), launch("test")
    both = sorted({e["edit"] for e in n6.plan(pairs, half="freeze")} & {e["edit"] for e in n6.plan(pairs, half="test")})
    rep_s = sum(n6.flops_from(e["layer"], tok2) for e in pl if e["edit"] in both) / rate
    a5 = {"center_s": l2a["center_s"] + l2b["center_s"], "high_s": l2a["high_s"] + l2b["high_s"],
          "center_gpu_h": round((l2a["center_s"] + l2b["center_s"]) / 3600, 3), "high_gpu_h": round((l2a["high_s"] + l2b["high_s"]) / 3600, 3),
          "usd_center_at_2.24": round(2.24 * (l2a["center_s"] + l2b["center_s"]) / 3600, 2),
          "usd_high_at_2.24": round(2.24 * (l2a["high_s"] + l2b["high_s"]) / 3600, 2),
          "A4_one_launch_center_s": center, "extra_setup_s": setup, "edits_in_both": both, "edits_in_both_s": rep_s,
          "rule": "L2a + L2b = A4's one launch + one more setup + the edits that run in both (the two pair probes and the zero edit, "
                  "which L2b's G-EPS and G-ZN read)",
          "rule_holds": math.isclose(l2a["center_s"] + l2b["center_s"], center + setup + rep_s, rel_tol=1e-12)}
    assert a5["rule_holds"], a5
    pk = m["measured"]["peak_mem_gib"]
    n_poles = len({e["pole"] for e in pl if "pole" in e}) + len({e["layer"] for e in pl if e["role"] in ("pc", "probe_pc")})
    logp = n_poles * J.TOPK * n6.VOCAB * 4
    rss = max(json.loads(x)["rss_kb"] for x in open(HERE / "diag/L1-02/heartbeat.jsonl")) * 1024
    out = {"what": "N6 L2 projection at L1's measured fp32 rate (spec section 9.5 S3), amendment A4: 1,024 eval windows, the positive "
                   "control in 8 layers", "pairs": {"path": a.pairs, "sha256": n6.sha256_file(a.pairs), "n_pairs": len(pairs["pairs"]),
                                                      "w_eval": w2},
           "L1": {"launch_utc": a.launch, "end_utc": a.end, "uptime_s": up, "n_edits": len(recs), "w_eval": w1, "edit_seconds": sec,
                  "edit_flop": fl, "non_edit_s": up - sec, "pass_s": pass_s,
                  "sec_per_edit_layer_unit_job": m["measured"]["sec_per_edit_layer_unit"]},
           "rate_fp32_flop_per_s": {"edits": rate, "base_pass_1_with_trackers": rate_pass,
                                    "spec_assumed": {k: v for k, v in n6.RATES.items()}},
           "L1_non_edit_split_s": {"window_part_at_96": win1, "fixed": fixed},
           "L2": {"n_edits": len(pl), "by_role": roles, "edit_layer_units": units, "edit_flop": fl2, "edit_s_center": fl2 / rate,
                  "edit_s_by_unit_rate": units * m["measured"]["sec_per_edit_layer_unit"] * w2 / w1,
                  "window_part_s": win2, "added_s": add, "fixed_s": fixed, "center_s": center, "high_s": high,
                  "center_gpu_h": round(center / 3600, 3), "high_gpu_h": round(high / 3600, 3),
                  "usd_center_at_2.24": round(2.24 * center / 3600, 2), "usd_high_at_2.24": round(2.24 * high / 3600, 2),
                  "usd_high_at_3.00424": round(3.00424 * high / 3600, 2),
                  "hard_kill_min": kill, "job_budget_min": math.floor(0.9 * kill),
                  "freeze1_projection_gpu_h": json.loads((HERE / "projection.json").read_text())["center"]["L2_gpu_h"],
                  "A3_projection_96_windows": {"n_edits": 281, "center_gpu_h": 4.466, "high_gpu_h": 5.582, "hard_kill_min": 436,
                                               "job_budget_min": 392},
                  "above_12_gpu_h_rule": high / 3600 > 12,
                  "by_window_count": {"note": "the same model with the A4 roster at other eval-set sizes; G_POWER_ratio_if_D_held is "
                                              "diag/L1-02/a4_preview.json's 96-window ratio x sqrt(96 / w) (bar 0.5)", "rows": table}},
           "L2a": l2a, "L2b": l2b, "A5_sum": a5,
           "memory": {"gpu_peak_gib_L1_02_measured": pk,
                      "gpu_peak_gib_L2_projected": {**{k: v for k, v in pk.items() if k != "progression"}, "inst": pk["pass1"],
                                                    "progression": pk["progression"]},
                      "gpu_basis": "every phase but the progression runs the same batch shapes as L1-02 (batch 4; G-INST is base pass 1's "
                                   "forward on L1's 96 windows); in the progression the GPU holds h_l and h_(l+1) of at most one chunk of "
                                   f"{J.CHUNK} windows, as L1-02 held them for its 96 windows, and the concatenation copy of L1-02 is gone, so "
                                   "the progression's peak is at most L1-02's; the L40S has 48 GB (44.7 GiB)",
                      "host_kept_residual_gib": resid / GIB, "host_exemplar_logprobs_gib": logp / GIB, "n_edited_poles": n_poles,
                      "host_L1_02_largest_rss_gib": rss / GIB, "host_projected_gib": (resid + logp + rss) / GIB,
                      "host_job_check_needs_gib": (resid + J.HOST_MARGIN) / GIB,
                      "host_instances_gib": {"g6e.2xlarge": 64, "g6e.4xlarge": 128}},
           "rule": "center = fixed + window part x (W_L2 / 96) + G-INST pass + residual copies + L2 edit FLOPs / L1 edit rate; "
                   "high = 1.25 x center; hard kill = ceil(1.3 x high); job --budget-min = floor(0.9 x hard kill)"}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"rate_TFLOPs": rate / 1e12, "edits": len(pl), "by_role": roles, **{k: out["L2"][k] for k in
                      ("edit_s_center", "edit_s_by_unit_rate", "center_gpu_h", "high_gpu_h", "usd_center_at_2.24", "usd_high_at_2.24",
                       "hard_kill_min", "job_budget_min", "above_12_gpu_h_rule")}}))
    print(json.dumps({"fixed_s": fixed, "win1": win1, "add": add}))
    for k in ("L2a", "L2b"):
        print(k, json.dumps({x: out[k][x] for x in ("n_edits", "by_role", "edit_s_center", "setup_s", "center_gpu_h", "high_gpu_h",
                                                     "usd_center_at_2.24", "usd_high_at_2.24", "hard_kill_min", "job_budget_min")}))
    print("A5_sum", json.dumps({x: a5[x] for x in ("center_gpu_h", "high_gpu_h", "usd_center_at_2.24", "usd_high_at_2.24",
                                                    "extra_setup_s", "edits_in_both", "edits_in_both_s", "rule_holds")}))
    print(json.dumps({k: out["memory"][k] for k in ("host_kept_residual_gib", "host_exemplar_logprobs_gib", "n_edited_poles",
                                                    "host_L1_02_largest_rss_gib", "host_projected_gib", "host_job_check_needs_gib")}))


if __name__ == "__main__":
    main()
