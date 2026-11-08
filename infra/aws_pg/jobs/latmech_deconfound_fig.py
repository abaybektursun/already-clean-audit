"""LATMECH headline de-confound figure — RF_MCQ(16k) for ALL conditions incl. the top-σ + random-σ-matched
de-confounds, so the figure tells the FINAL story (principal-subspace, H_store CONFOUNDED), not the
pre-critic by-rule verdict (post-critic-artifact-desync discipline). Reads the specific cells by S3 path
(the de-confound cells live in separate prefixes with colliding ids). Pure, CPU.

    python -m infra.aws_pg.jobs.latmech_deconfound_fig --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

B = "s3://pg-lineage-769063704696/results"
# (label, s3 path to the cell json, recover?) — seed 1337, n=16000, sealed half.
CELLS = [
    ("deleted\n(ceiling)", f"{B}/latmech-grid-s1337/out/deleted_n16000/cell_deleted_cap_r512_n16000_s1337_confirmation.json"),
    ("content\n(grad top-4)", f"{B}/latmech-grid-s1337/out/content_n16000/cell_content_cap_r512_n16000_s1337_confirmation.json"),
    ("top-σ\n(top-4 by σ)", f"{B}/latmech-topsig/out/topsig_ablated/cell_content_cap_r512_n16000_s1337_confirmation.json"),
    ("randspan\n(rand σ-matched)", f"{B}/latmech-randspan/out/randspan_ablated/cell_content_cap_r512_n16000_s1337_confirmation.json"),
    ("emrand\n(low-σ tail)", f"{B}/latmech-grid-s1337/out/emrand_n16000/cell_emrand_cap_r512_n16000_s1337_confirmation.json"),
    ("shrink-perturb\n(basin)", f"{B}/latmech-grid-s1337/out/shrink_perturb_n16000/cell_shrink_perturb_cap_r512_n16000_s1337_confirmation.json"),
    ("random_floor", f"{B}/latmech-grid-s1337/out/random_floor_n16000/cell_random_floor_cap_r512_n16000_s1337_confirmation.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import subprocess
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from infra.pg import reprice_stats as RS

    def load(p):
        loc = out / (p.split("/")[-1])
        subprocess.run(["aws", "s3", "cp", p, str(loc), "--region", "us-east-1"], check=True)
        return json.loads(loc.read_text())

    rows = [(lab, load(p)) for lab, p in CELLS]
    floor = np.array(dict((lab, d) for lab, d in rows)["random_floor"]["mcq"]["per_item"], float)
    ceil = np.array([d for lab, d in rows if "deleted" in lab][0]["mcq"]["per_item"], float)

    labels, pts, los, his = [], [], [], []
    for lab, d in rows:
        S = np.array(d["mcq"]["per_item"], float)
        b = RS.bootstrap_recovery_fraction(S, floor, ceil, n=10000, seed=1337)
        labels.append(lab); pts.append(b["point"]); los.append(b["point"] - b["lo"]); his.append(b["hi"] - b["point"])

    # color: tank (RF<0.3) red, recover (RF>0.6) green, anchors blue/grey
    cols = []
    for lab, p in zip(labels, pts):
        if "deleted" in lab: cols.append("#0072B2")
        elif "random_floor" in lab: cols.append("#777777")
        elif p < 0.3: cols.append("#D55E00")   # tank
        else: cols.append("#009E73")           # recover

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, pts, yerr=[los, his], capsize=4, color=cols, alpha=0.88)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="#0072B2", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RF_MCQ(16k)  (1 = deleted ceiling, 0 = random floor)")
    ax.set_title("LATMECH — re-learnability is scaffolded by the top-σ PRINCIPAL subspace, not cloze-content\n"
                 "content & top-σ ablation CRATER re-entry; random-σ-matched, low-σ-tail, basin all RECOVER "
                 "⇒ H_store (by rule) is a principal-subspace confound")
    fig.text(0.01, 0.005, "sealed half n=7453, seed 1337, n=16000, r=512 | content↔top-σ subspace overlap 0.35 | "
             "tank=red, recover=green | DIAGNOSTIC (¬novel/¬practical)", fontsize=6)
    fig.savefig(out / "F1_deconfound.png", bbox_inches="tight", dpi=150)
    fig.savefig(out / "F1_deconfound.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "deconfound_fig.json").write_text(json.dumps(
        dict(zip([l.replace("\n", " ") for l in labels], [round(p, 3) for p in pts])), indent=1))
    print("[latmech-deconfound-fig] OK —", dict(zip([l.split(chr(10))[0] for l in labels], [round(p, 3) for p in pts])))


if __name__ == "__main__":
    main()
