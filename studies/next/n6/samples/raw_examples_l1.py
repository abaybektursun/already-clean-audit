"""N6's ten raw examples with L1's numbers (spec section 10): the zero-GPU part (raw_examples_zero_gpu.json) with its PENDING
fields filled from L1's records (banks/<attempt>/samples/raw_examples.jsonl and banks/<attempt>/edits.jsonl): the Frobenius norm c of the
edit, kappa, the corpus loss change in nats per token, E+, the efficacy KL at the pole's 50 eval exemplars, the realized residual
change R, and the five tokens whose loss changed most with 32 tokens of context. The random-direction twin is L2's.

    PYTHONPATH=. python3 studies/next/n6/samples/raw_examples_l1.py [studies/next/n6/banks/<attempt>]
"""
import json
import pathlib
import sys

HERE = pathlib.Path("studies/next/n6/samples")
BANK = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "studies/next/n6/banks/L1-02")


def main():
    zero = json.loads((HERE / "raw_examples_zero_gpu.json").read_text())
    gpu = {(r["pair"], r["member"]): r for r in map(json.loads, open(BANK / "samples/raw_examples.jsonl"))}
    rec = {r["edit"]: r for r in map(json.loads, open(BANK / "edits.jsonl"))}
    assert len(zero) == 10 and len(gpu) == 10, (len(zero), len(gpu))
    rows = []
    for z in zero:
        g = gpu[(z["pair"], z["member"])]
        e = rec[f"{z['pair']}_{z['member']}_col"]
        assert g["pole"] == z["pole"] == e["pole"] and g["partner"] == z["partner"], z["pole"]
        assert g["dL_nats_per_token"] == e["dL"] and g["frobenius_norm_c"] == e["c"]
        rows.append({**z, "frobenius_norm_c": e["c"], "norm_w": e["norm_w"], "kappa": e["kappa"], "dL_nats_per_token": e["dL"],
                     "E_plus": e["E_plus"], "E_plus_pass1": e["E_plus_pass1"], "kl_eff": e["kl_eff"], "dl_eff": e["dl_eff"],
                     "R_residual_change": e["R"], "seconds": e["seconds"], "dL_rand_same_norm": "L2",
                     "top5_tokens": g["top5_tokens"]})
    (HERE / "raw_examples_L1.json").write_text(json.dumps(rows, indent=1))
    lines = ["# N6 raw examples with L1's numbers", "",
             "Written by `studies/next/n6/samples/raw_examples_l1.py` from `raw_examples_zero_gpu.json` and L1's records "
             f"(`{BANK}/samples/raw_examples.jsonl`, `{BANK}/edits.jsonl`). Age is the main cluster's birth index T_k. "
             "c is the Frobenius norm of the rank-one edit (c = min of the two column norms of the pair), kappa = c / ||w||. "
             "dL is the mean change of next-token loss over the 196,608 eval targets, fp32, nats per token. KL_eff is the mean "
             "KL(base || edited) at the pole's 50 eval exemplars, nats. The random-direction twin is L2's.", ""]
    for r in rows:
        lines.append(f"## {r['pair']} {r['member']}: {r['pole']} (partner {r['partner']})")
        lines.append("")
        lines.append(f"layer {r['layer']}, {r['half']} half; age T_{r['age']} (partner T_{r['partner_age']}); adjusted age "
                     f"{r['adjusted_age']:+.2f}; max|a| {r['max_abs']:.4g}; density {r['density']:,}")
        lines.append("")
        lines.append(f"c {r['frobenius_norm_c']:.6g} (||w|| {r['norm_w']:.6g}, kappa {r['kappa']:.4f}); "
                     f"**dL {r['dL_nats_per_token']:+.4e} nats per token**; E+ {r['E_plus']:.6g}; KL_eff {r['kl_eff']:.4e}; "
                     f"R {r['R_residual_change']:.4g}")
        lines.append("")
        lines.append("Strongest T_15 contexts (bank):")
        lines.append("")
        for t in r["top3_T15"]:
            lines.append(f"- rank {t['rank']}, a {t['a']:.4g}: …{json.dumps(t['before_tail'])[1:-1]} **[{json.dumps(t['token'])[1:-1]}]** "
                         f"{json.dumps(t['after_head'])[1:-1]}…")
        lines.append("")
        lines.append("Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):")
        lines.append("")
        for t in r["top5_tokens"]:
            lines.append(f"- w {t['window']}, pos {t['target_pos']}, Δℓ {t['dl']:+.4f}, base {t['base_loss']:.3f}: "
                         f"…{json.dumps(t['context'][-120:])[1:-1]} **[{json.dumps(t['target'])[1:-1]}]**")
        lines.append("")
    (HERE / "raw_examples_L1.md").write_text("\n".join(lines))
    print("written", len(rows), "rows")


if __name__ == "__main__":
    main()
