"""CO-F1 (item 74) — the γ retrofit: is the sink an address, or a gain artifact?

Spec §3 Tier F / §6.3 CO-F1(b). CPU-only, banked artifacts plus the base checkpoint's norm gains.

WHY THIS EXISTS. Every reader whose input is a normed residual sees x·γ, not x, so a coordinate
with a large γ carries a large post-LN activation for free. Any "address" statistic read at a
post-LN site is therefore γ-contaminated, and the question is whether the sink survives the
correction or was partly the gains all along.

γ = 1 + norm.weight — the Gemma-family offset. γ² WITHOUT the +1 is simply wrong on this family,
and getting it wrong would manufacture exactly the relocation this arm is looking for.

THE SITES ARE NOT INTERCHANGEABLE. q/k/v read the post-input_layernorm stream; gate/up read the
post-attention_layernorm stream; o_proj reads the ATTENTION OUTPUT and has no γ at its input at
all. o_proj is therefore EXCLUDED from the (b) gate rather than given a borrowed gain, and the
banked comparison distribution is restricted to the same sites so the contrast stays
apples-to-apples.

THE (b) GATE reuses `mercator_predict` verbatim — the same `measured_address` (top-k by leverage),
the same `predict_address`, the same `score_prediction`. A re-implementation would make the
comparison against M-8's 72 banked cells meaningless.

BOTTOM-DECILE-γ FLAGS ride on every corrected cell: dividing by a tiny γ is how a γ correction
manufactures spikes, so the cells where that could happen are named on the row (§8 threat row).

    pg run infra.aws_pg.jobs.copernicus_f1 \
        --args "--m8-s3 s3://.../results/mercator_w0gpu4/out/M-8/atlas.jsonl" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_f1 --budget-min 90
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_predict as MP

ARM = "CO-F1"
QUEUE_ITEM = "74"
WAVE = "WF"
BANK = "core32"
K_ADDR = 32
# The reader sites that see a NORMED stream, and which norm supplies their gain.
GAMMA_SITE = {"q_proj": "input_layernorm", "k_proj": "input_layernorm",
              "v_proj": "input_layernorm", "gate_proj": "post_attention_layernorm",
              "up_proj": "post_attention_layernorm"}
NO_GAMMA = ("o_proj",)                 # reads the attention output: no input-side gain exists


def load_m8(m8_s3: str, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["aws", "s3", "cp", m8_s3, str(dest), "--region", "us-east-1", "--quiet"],
                       check=True)
    rows = [json.loads(l) for l in dest.read_text().splitlines() if l.strip()]
    j = [r for r in rows if r["statistic"] == "address_overlap_jaccard"]
    assert j, f"M-8 dependency at {m8_s3} carries no address_overlap_jaccard rows"
    out = {}
    for r in j:
        out.setdefault(json.loads(r["flags"])["scorer"], []).append(r)
    for s in ("activation_magnitude", "gradient_magnitude", "predictor", "random"):
        assert s in out, f"M-8 anchor is missing the {s} scorer"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m8-s3",
                    default="s3://pg-lineage-769063704696/results/mercator_w0gpu4/out/M-8/atlas.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    import torch
    from huggingface_hub import HfApi
    from infra.aws_pg.jobs.invariant_core import PG1_REPO, _load_decoder

    m8 = load_m8(a.m8_s3, pathlib.Path(H.cache_dir(out, "m8")) / "atlas.jsonl")
    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "f1b_gamma2_matches_activation_scorer",
            "γ² correlates with topenergy, NOT with the sink coordinates; the sink is tiny-γ at "
            "most reader sites (pass-through)", confidence="low-mid",
            basis="spec §11 row CO-F1", family="mechanism")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_f1", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "f1b_gamma2_matches_activation_scorer"))
    gates = C.Gates(out)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    sd = _load_decoder(PG1_REPO, base_sha)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    gamma = {}
    for L in range(H.N_LAYERS):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            key = f"layers.{L}.{norm}.weight"
            assert key in sd, f"missing {key} in the decoder state dict"
            gamma[(L, norm)] = 1.0 + sd[key].to(torch.float64).cpu().numpy()   # Gemma offset

    rows, gate_cells = [], []
    for t in list(GAMMA_SITE) + list(NO_GAMMA):
        for L in range(H.N_LAYERS):
            U = WT.load_bank(bank_dir, t, L, BANK)
            n = U.shape[0]
            lev = H.leverage(U)
            truth = MP.measured_address(U, K_ADDR)
            row = {"type": t, "layer": L, "n": int(n), "has_gamma": t in GAMMA_SITE}
            if t in GAMMA_SITE:
                g = gamma[(L, GAMMA_SITE[t])]
                assert g.shape[0] == n, f"{t} L{L}: gamma dim {g.shape[0]} != bank dim {n}"
                g2 = g ** 2
                # (b) γ² as an address predictor, scored with M-8's own machinery
                sp = MP.score_prediction(MP.predict_address(g2, K_ADDR), truth, n)
                # (a) does γ² track the sink's leverage at all?
                from scipy import stats
                rho = float(stats.spearmanr(g2, lev).statistic)
                # (d) is the sink pass-through or read-hard at this site?
                on = float(np.mean(g2[list(truth)]))
                off = float(np.mean(np.delete(g2, list(truth))))
                dec = float(np.percentile(g, 10))
                row.update({"jaccard_gamma2": sp["jaccard"], "spearman_gamma2_leverage": rho,
                            "gamma2_on_address": on, "gamma2_off_address": off,
                            "on_over_off": on / off if off > 0 else float("inf"),
                            "bottom_decile_gamma": dec,
                            "bottom_decile_flag": bool(dec < 0.1),
                            "n_address_in_bottom_decile": int(np.sum(g[list(truth)] < dec))})
                gate_cells.append(sp["jaccard"])
                at.add("f1_gamma2_address_jaccard", value=sp["jaccard"], bank_id=BANK, layer=L,
                       type=t, n=K_ADDR,
                       flags={"scorer": "gamma_squared", "site": GAMMA_SITE[t],
                              "spearman_gamma2_leverage": rho, "gamma2_on_over_off": row["on_over_off"],
                              "bottom_decile_gamma": dec,
                              "bottom_decile_flag": row["bottom_decile_flag"],
                              "gamma_convention": "gamma = 1 + norm.weight (Gemma family)"})
            else:
                row["excluded_reason"] = ("o_proj reads the attention output; no input-side gain "
                                          "exists, so no gamma cell is defined here")
            rows.append(row)

    # THE GATE (§6.3 CO-F1(b)). Comparison restricted to the SAME sites — M-8 banked q/k/v/o_proj,
    # and o_proj has no gamma, so pooling it in would compare different populations.
    act_same_site = [r["value"] for r in m8["activation_magnitude"]
                     if r["type"] in GAMMA_SITE]
    assert act_same_site, "no same-site activation_magnitude cells in the M-8 anchor"
    act_ci = H.mean_ci(act_same_site, seed=a.seed)
    g_ci = H.mean_ci(gate_cells, seed=a.seed)
    beats_or_matches = bool(g_ci["mean"] >= act_ci["ci_lo"])
    rand_same_site = [r["value"] for r in m8["random"] if r["type"] in GAMMA_SITE]
    rand_ci = H.mean_ci(rand_same_site, seed=a.seed)
    above_random = bool(g_ci["ci_lo"] > rand_ci["ci_hi"])
    gates.record("f1b_gamma2_matches_activation_scorer", beats_or_matches,
                 gamma2_mean_jaccard=g_ci["mean"], gamma2_ci=[g_ci["ci_lo"], g_ci["ci_hi"]],
                 activation_mean_jaccard=act_ci["mean"],
                 activation_ci=[act_ci["ci_lo"], act_ci["ci_hi"]],
                 random_mean_jaccard=rand_ci["mean"], n_cells_gamma=len(gate_cells),
                 n_cells_activation=len(act_same_site),
                 sites=sorted(GAMMA_SITE), excluded=list(NO_GAMMA),
                 band=("bootstrap CI of the mean over the banked same-site cells; BEATS/MATCHES = "
                       "gamma-squared's mean lies within or above it (§6.3)"),
                 prediction="gamma-squared does NOT track the sink address (§11, low-mid)")
    gates.record("f1b_gamma2_above_random", above_random,
                 gamma2_ci_lo=g_ci["ci_lo"], random_ci_hi=rand_ci["ci_hi"],
                 band="M-8's own random scorer over the same sites")

    flagged = [r for r in rows if r.get("bottom_decile_flag")]
    rep = {"object": "copernicus_f1_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "base_repo": PG1_REPO, "base_sha": base_sha,
           "m8_s3": a.m8_s3, "k_address": K_ADDR, "bank": BANK,
           "gamma_convention": "gamma = 1 + norm.weight (Gemma family; gamma^2 without the +1 is wrong)",
           "sites": GAMMA_SITE, "excluded_sites": list(NO_GAMMA),
           "gate_b": {"gamma2": g_ci, "activation_magnitude": act_ci, "random": rand_ci,
                      "beats_or_matches": beats_or_matches, "above_random": above_random},
           "bottom_decile_flagged_cells": len(flagged),
           "verdict": (C.REFUTER_DID_NOT_FIRE if beats_or_matches else C.REFUTER_FIRED),
           "rows": rows, "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_f1.json").write_text(json.dumps(rep, indent=2, default=float))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_F1_OK gamma2_jaccard={g_ci['mean']:.4f} "
          f"activation={act_ci['mean']:.4f} [{act_ci['ci_lo']:.4f},{act_ci['ci_hi']:.4f}] "
          f"matches={beats_or_matches} above_random={above_random} "
          f"bottom_decile_flags={len(flagged)}", flush=True)


if __name__ == "__main__":
    main()
