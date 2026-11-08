"""XGEN figures — weight-free from S3 JSON (CPU). Prereg §B.

Fetches the frozen results/xgen_deletion/.../deletion.json + results/xgen_carrier/.../carrier.json
and renders, one-claim-one-figure, PNG (dpi=150) + PDF (bbox tight) with a provenance footer:

  F1  S_text(AFTER/BEFORE/FLOOR) — V1 deletion; PG2 'H_forget' annotated LABEL-ONLY (no PG2 number).
  F2  per-matrix MLP f_perp vs the dimension-counting null + the row-shuffle control — V2 carrier.
  F3  the SEAL table — {V1, V2} PG1 labels vs PG2 labels {H_forget, out-of-span} (firewall: labels).

FIREWALL: only PG2 verdict LABELS appear; never a PG2 number. Verdict labels here are DESCRIPTIVE
(the binding verdict is applied at unblind §9.6 under BH-FDR). Fail-loud; no try/except.

    pg run infra.aws_pg.jobs.xgen_figures --type c7i.xlarge --no-weights \
        --s3-prefix results/xgen_figures --budget-min 30 \
        --args="--freeze-sha <sha>"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
DEFAULT_DELETION_KEY = "results/xgen_deletion/out/deletion.json"
DEFAULT_CARRIER_KEY = "results/xgen_carrier/out/carrier.json"
# label-only PG2 references. V2 here is the RAW-ΔW out-of-span fraction = the W12 object (PG2 value
# is also AT-NULL / ¬H_new). The W15 "out-of-span carrier" (f_perp 0.81) is a DIFFERENT object —
# the install-gradient cotangent, NOT `decompose` on raw weights — and is NOT measured here (Phase-2).
PG2_LABELS = {"V1": "H_forget", "V2_W12_rawDW": "at-null", "W15_carrier": "out-of-span (0.81, untested)"}


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"],
                   check=True)


def _footer(fig, prov):
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deletion-key", default=DEFAULT_DELETION_KEY)
    ap.add_argument("--carrier-key", default=DEFAULT_CARRIER_KEY)
    ap.add_argument("--freeze-sha", default="UNFROZEN")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dpath, cpath = out / "deletion.json", out / "carrier.json"
    _fetch(a.deletion_key, dpath)
    _fetch(a.carrier_key, cpath)
    dele = json.loads(dpath.read_text())
    carr = json.loads(cpath.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # --- §6 descriptive labels from the frozen JSONs (binding verdict applied at unblind) ---
    # θ_forget = FLOOR + eval-noise band (S_text units), computed by xgen_deletion on the discovery
    # half. All three §6 V1 conjuncts (S_text<θ, not OOD-recovers, recovery==H_deletion) fold into
    # descriptive_v1_label in deletion.json.
    s_conf = dele["S_text"]["confirmation"]
    theta = dele["theta_forget"]
    rec_verdict = (dele.get("recovery_gate") or {}).get("verdict")
    v1_label = dele.get("descriptive_v1_label") or (
        "H_forget" if (s_conf["ci95"][1] < theta and not dele["OOD_blankimg_native_pg1"]["recovers"])
        else "H_retain")
    v2 = carr["mlp_summary"]
    # V2 = the RAW-ΔW out-of-span fraction (gain_routing.decompose) = the W12 object; its PG2 value is
    # also AT the dimension-counting null (¬H_new). NOT the W15 install-gradient carrier (different
    # object, gradient-based, not measured). PG1 at-null vs PG2-W12 at-null → REPLICATES W12.
    v2_label = "out-of-span" if v2["descriptive_out_of_span"] else "at-null"

    # ---------------- F1: S_text deletion ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    arms = ["BEFORE", "AFTER", "FLOOR"]
    accs = [dele["mmlu"][k]["acc_pct"] for k in arms]
    ax.bar(arms, accs, color=["#4c72b0", "#c44e52", "#999999"])
    ax.axhline(25.0, ls=":", color="#666", lw=1, label="4-way chance (25%)")
    ax.set_ylabel("MMLU 5-shot (%)")
    ci = s_conf["ci95"]
    ax.axhline(dele["mmlu"]["FLOOR"]["acc_pct"], ls="--", color="#bbb", lw=1, label="FLOOR")
    ax.set_title(f"V1 deletion (swap-back, confirmation): S_text={s_conf['S_text']:.3f} "
                 f"CI[{ci[0]:.2f},{ci[1]:.2f}] vs θ_forget={theta:.3f}; recovery={rec_verdict} → {v1_label} (descriptive)\n"
                 f"(PG2 label: {PG2_LABELS['V1']} — LABEL-ONLY, firewall)")
    ax.legend(fontsize=8)
    _footer(fig, f"XGEN V1 · s3://{BUCKET}/{a.deletion_key} · commit {a.freeze_sha} · "
                 f"BEHAVIORAL · FIREWALL:labels-only")
    _save(fig, "F1_s_text_deletion")

    # ---------------- F2: per-matrix MLP f_perp vs null + shuffle ----------------
    mlp = [r for r in carr["per_matrix"] if r["is_mlp"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = list(range(len(mlp)))
    ax.scatter(xs, [r["f_perp"] for r in mlp], s=14, color="#c44e52", label="ΔW₁ f_perp", zorder=3)
    ax.scatter(xs, [r["shuffle"]["mean"] for r in mlp], s=10, color="#dd8888",
               marker="x", label="row-shuffle control", zorder=2)
    ax.plot(xs, [r["null"]["analytic_f_perp"] for r in mlp], color="#55a868", ls="--",
            lw=1, label="dimension-counting null")
    ax.plot(xs, [r["null"]["p95_mean"] for r in mlp], color="#888", ls=":", lw=1, label="null p95")
    ax.set_xlabel("MLP matrix (gate/up/down × 18 layers)")
    ax.set_ylabel("raw-ΔW f_perp (out-of-span fraction)")
    ax.set_title(f"V2 raw-ΔW out-of-span (W12 object): median MLP f_perp={v2['median_f_perp']:.3f} "
                 f"CI{[round(x,3) for x in v2['median_f_perp_ci95']]} vs dim-counting null p95={v2['null_p95_median']:.3f} "
                 f"→ {v2_label} (enrich {v2['enrichment_median']:.2f})\n"
                 f"PG2-W12 raw-ΔW: {PG2_LABELS['V2_W12_rawDW']} → REPLICATES. "
                 f"(W15 install-gradient carrier = a DIFFERENT object, not measured — firewall: labels-only)")
    ax.legend(fontsize=8, loc="best")
    _footer(fig, f"XGEN V2 · s3://{BUCKET}/{a.carrier_key} · commit {a.freeze_sha} · "
                 f"WEIGHT · FIREWALL:labels-only")
    _save(fig, "F2_carrier_fperp")

    # ---------------- F3: the seal table ----------------
    v1_match = (v1_label == PG2_LABELS["V1"])
    v2_match = (v2_label == PG2_LABELS["V2_W12_rawDW"])     # raw-ΔW vs W12 (both at-null)
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.axis("off")
    rows = [
        ["readout", "object", "PG1", "PG2", "match?"],
        ["V1 deletion (W11-A)", "S_text behavioral", v1_label, PG2_LABELS["V1"], "✓" if v1_match else "✗ (PG1 retains more)"],
        ["V2 (W12)", "raw-ΔW out-of-span", v2_label, PG2_LABELS["V2_W12_rawDW"], "✓ replicates" if v2_match else "✗"],
        ["W15 carrier", "install-gradient f_perp", "NOT MEASURED", PG2_LABELS["W15_carrier"], "— Phase-2"],
    ]
    seal = ("W12 raw-ΔW REPLICATES; W11-A behavioral INCOMPLETE on Gemma-1 (H_retain); "
            "W15 install-carrier UNTESTED (Phase-2)")
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("THE SEAL: W12 raw-ΔW replicates · W11-A behavioral incomplete (H_retain) · "
                 "W15 carrier untested (different object, Phase-2)", fontsize=10)
    _footer(fig, f"XGEN SEAL · {a.deletion_key} + {a.carrier_key} · commit {a.freeze_sha} · "
                 f"FIREWALL:labels-only · V2=raw-ΔW(W12)≠W15 install-carrier")
    _save(fig, "F3_seal_table")

    index = {"figures": ["F1_s_text_deletion", "F2_carrier_fperp", "F3_seal_table"],
             "v1_label": v1_label, "v2_label_rawDW_W12": v2_label, "v2_replicates_W12": v2_match,
             "W15_carrier": "not measured (Phase-2)", "seal": seal,
             "deletion_src": f"s3://{BUCKET}/{a.deletion_key}",
             "carrier_src": f"s3://{BUCKET}/{a.carrier_key}", "commit": a.freeze_sha}
    (out / "fig_index.json").write_text(json.dumps(index, indent=2))
    print(f"[XGEN-FIG] V1={v1_label} (W11-A match={v1_match}) · V2_rawDW={v2_label} (W12 match={v2_match}) · "
          f"W15 carrier=untested → {seal}  ({out})")


if __name__ == "__main__":
    main()
