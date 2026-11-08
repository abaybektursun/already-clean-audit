"""policy-content-readout — figure suite (prereg §10). F1 the policy ladder, F2 paired per-item
conditional retention. Weight-free: reads policy_content_readout.json + policy_content_readout_banks.json
only. Every panel: item-bootstrap CI, CEILING/FLOOR bands overlaid, plain-language title, provenance
footer. PNG (review) + PDF (paper), bbox_inches=tight. Run:

  python -m infra.aws_pg.jobs.pcr_figures --result-dir <dir> --out-dir <dir> --sha <freeze_sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

# colourblind-safe (Okabe–Ito)
CLOZE_C = "#009E73"   # green  — content axis
MCQ_C = "#0072B2"     # blue   — letter axis
FLOOR_C = "#D55E00"   # orange — floor band
REF_C = "#999999"

ARM_LABEL = {"base": "base\n(CEILING)", "floor": "random-dec\n(FLOOR)", "pi0": "π0", "pi05": "π0.5",
             "matched_dw": "matched-\nrandom-ΔW", "pg1": "PG1"}
LADDER = ["base", "pg1", "pi0", "pi05", "matched_dw", "floor"]


def _boot_mean(x, n_boot=1000, seed=1337):
    import numpy as np
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = [x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fig1_ladder(result, banks, out_dir, prov):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in LADDER if a in banks]
    cloze = {a: _boot_mean([int(v) for v in banks[a]["cloze_per_item"]]) for a in arms}
    mcq = {a: _boot_mean([int(v) for v in banks[a]["mcq_per_item"]]) for a in arms}
    dr = result.get("cloze_dynamic_range", {})
    mq = result.get("mcq_span", {})

    x = np.arange(len(arms))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    # CEILING/FLOOR bands
    if dr:
        ax.axhspan(dr["floor"], dr["ceiling"], color=CLOZE_C, alpha=0.08, zorder=0)
        ax.axhline(dr["ceiling"], color=CLOZE_C, ls=":", lw=1, alpha=0.7)
        ax.axhline(dr["floor"], color=FLOOR_C, ls=":", lw=1, alpha=0.7)
    for i, a in enumerate(arms):
        cm, clo, chi = cloze[a]
        mm, mlo, mhi = mcq[a]
        ax.bar(x[i] - w / 2, cm, w, color=CLOZE_C, label="cloze (content)" if i == 0 else None,
               yerr=[[cm - clo], [chi - cm]], capsize=3, ecolor="#333", error_kw={"lw": 1})
        ax.bar(x[i] + w / 2, mm, w, color=MCQ_C, label="MCQ (letter)" if i == 0 else None,
               yerr=[[mm - mlo], [mhi - mm]], capsize=3, ecolor="#333", error_kw={"lw": 1})
    ax.axhline(0.25, color=REF_C, ls="--", lw=1, alpha=0.6)
    ax.text(len(arms) - 0.5, 0.255, "4-way chance", color=REF_C, fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=9)
    ax.set_ylabel("MMLU accuracy"); ax.set_ylim(0, max(0.5, (dr.get("ceiling", 0.45) + 0.07)))
    ax.legend(loc="upper right", frameon=False)
    verdicts = []
    for name in ("pi0", "pi05"):
        if name in result.get("policies", {}):
            verdicts.append(f"{name}: {result['policies'][name]['confirmation']['verdict'].split('(')[0].strip()}")
    title = "Cloze (content) vs MCQ (letter) on the π0/π0.5 policies, own-embed"
    if verdicts:
        title += "\n" + " · ".join(verdicts)
    ax.set_title(title, fontsize=10.5)
    ax.text(0.0, -0.20, "How to read: green = content ranking, blue = letter argmax; shaded band = cloze [FLOOR, CEILING].\n"
            "Cloze near CEILING with MCQ at FLOOR ⇒ letter-readout collapse (content survives); BOTH at FLOOR ⇒ genuine deletion.",
            transform=ax.transAxes, fontsize=7.5, color="#444")
    ax.text(1.0, -0.28, prov, transform=ax.transAxes, fontsize=6, color="#888", ha="right")
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"F1_policy_ladder.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"cloze": cloze, "mcq": mcq}


def fig2_conditional(result, banks, out_dir, prov):
    """Among items the base CEILING ranks cloze-correct, what fraction does the policy still rank
    correctly (cloze) vs answer by letter (MCQ)? Isolates: of the content the base KNOWS, how much
    survives into the policy under each readout."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_correct = np.asarray(banks["base"]["cloze_per_item"], dtype=bool)
    names = [n for n in ("pi0", "pi05") if n in banks]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    x = np.arange(len(names))
    w = 0.36
    for i, n in enumerate(names):
        cz = np.asarray(banks[n]["cloze_per_item"], dtype=int)[base_correct]
        mq = np.asarray(banks[n]["mcq_per_item"], dtype=int)[base_correct]
        cm, clo, chi = _boot_mean(cz)
        mm, mlo, mhi = _boot_mean(mq)
        ax.bar(x[i] - w / 2, cm, w, color=CLOZE_C, label="cloze (content)" if i == 0 else None,
               yerr=[[cm - clo], [chi - cm]], capsize=3, ecolor="#333", error_kw={"lw": 1})
        ax.bar(x[i] + w / 2, mm, w, color=MCQ_C, label="MCQ (letter)" if i == 0 else None,
               yerr=[[mm - mlo], [mhi - mm]], capsize=3, ecolor="#333", error_kw={"lw": 1})
    # floor reference: same conditional for the random decoder
    fl_cz = np.asarray(banks["floor"]["cloze_per_item"], dtype=int)[base_correct].mean() if "floor" in banks else None
    if fl_cz is not None:
        ax.axhline(fl_cz, color=FLOOR_C, ls=":", lw=1.2, alpha=0.8)
        ax.text(len(names) - 0.5, fl_cz + 0.01, "FLOOR cloze | base-correct", color=FLOOR_C, fontsize=8, ha="right")
    ax.axhline(0.25, color=REF_C, ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL.get(n, n) for n in names])
    ax.set_ylabel("retention | base ranks it cloze-correct"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False)
    n_base = int(base_correct.sum())
    ax.set_title(f"Of the content the base CEILING knows (n={n_base} cloze-correct items),\n"
                 "how much survives into the policy — by content vs letter readout", fontsize=10.5)
    ax.text(1.0, -0.16, prov, transform=ax.transAxes, fontsize=6, color="#888", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"F2_conditional_retention.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_base_cloze_correct": n_base, "floor_cloze_given_base": fl_cz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sha", default="UNFROZEN")
    a = ap.parse_args()
    rd = pathlib.Path(a.result_dir)
    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((rd / "policy_content_readout.json").read_text())
    banks = json.loads((rd / "policy_content_readout_banks.json").read_text())
    banks_hash = hashlib.sha1((rd / "policy_content_readout_banks.json").read_bytes()).hexdigest()[:12]
    prov = f"policy-content-readout · SHA {a.sha} · banks {banks_hash} · CONFIRMATORY"

    f1 = fig1_ladder(result, banks, out_dir, prov)
    f2 = fig2_conditional(result, banks, out_dir, prov)
    registry = {
        "F1_policy_ladder": {"script": "infra.aws_pg.jobs.pcr_figures:fig1_ladder",
                             "inputs": ["policy_content_readout.json", f"banks@{banks_hash}"],
                             "tag": "CONFIRMATORY", "summary": f1},
        "F2_conditional_retention": {"script": "infra.aws_pg.jobs.pcr_figures:fig2_conditional",
                                     "inputs": [f"banks@{banks_hash}"], "tag": "CONFIRMATORY", "summary": f2},
    }
    (out_dir / "figures_registry.json").write_text(json.dumps(registry, indent=2, default=str))
    print("[PCR-FIG] wrote F1, F2 + figures_registry.json to", out_dir)


if __name__ == "__main__":
    main()
