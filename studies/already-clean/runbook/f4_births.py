"""F4 of SPEC_D1 §10, redrawn on 2026-11-05 after the retag of the §6.4 birth-order claim (results/RESULTS.md, Retag).

The two panels are those the old f4 of infra/pg/neurons_figures.py drew (its helpers are imported; since the housekeeping of
2026-11-05, studies/next/11_HOUSEKEEPING.md, that f4 calls draw() here): cluster births per grid point, main and non-main, from
bank/w5/verdict/metrics.json q2.births. Added on each bar: the chance level,
the mean count of random same-size groups born at that point over 300 random regroupings of every pole's non-echo
top-50 exemplars (studies/next/06b_verify_births/out/s03_histogram_tv.json, main_random_mean and non_main_random_mean;
the random group in main's slot has main's size). The birth-order line now states b against the random-group reference
of the third recount (studies/next/10_recount2/out/05_chance.json, summary.R1t) beside the registered permutation band,
with b's interval from the same recount (04_b.json, b_ci95_bootstrap_poles), as in the Retag of results/RESULTS.md; the
instrument's own interval (metrics.json, seed 0) is one pole wider at the top (follow-ups of 2026-11-05, item 4).
Writes F4_births.svg; the PDF and PNG are made from it with headless Chrome and sips (JOURNAL 2026-11-05).

    python3 -I studies/already-clean/runbook/f4_births.py --out studies/already-clean/results/figures
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from infra.pg.neurons_figures import (GRID_LABELS, INK, NULL, SERIES, Svg, grid_ticks, grid_x,  # noqa: E402
                                      median_index, need, wrap, y_axis)
from infra.pg.neurons_verdict import file_sha256  # noqa: E402

METRICS = Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json")  # local copy
METRICS_SHA16 = "a95bfa57bfe65246"            # the seventh unblind run (RESULTS.md, REGISTRY.md)
CHANCE = REPO / "studies/next/10_recount2/out/05_chance.json"
BOOT = REPO / "studies/next/10_recount2/out/04_b.json"
HIST = REPO / "studies/next/06b_verify_births/out/s03_histogram_tv.json"


def draw(m: dict, source: Path, shown: str) -> tuple[str, dict]:
    """F4 from the metrics `m` read from `source` (named `shown` in the footer) and the three chance files; refuses births
    or b other than those the chance files were computed on. infra/pg/neurons_figures.py f4 calls this."""
    births = need(m, source, "q2", "births")
    bo = need(m, source, "q2", "birth_order")
    r1t = json.loads(CHANCE.read_text())["summary"]["R1t"]
    boot = json.loads(BOOT.read_text())
    h = json.loads(HIST.read_text())
    chance = {"main": h["main_random_mean"], "non_main": h["non_main_random_mean"]}
    # the chance file must describe the same clusters and the same b as the metrics
    assert h["main_real"] == births["main"]["counts"] and h["non_main_real"] == births["non_main"]["counts"], "births"
    assert abs(r1t["b_real"] - bo["b"]) < 1e-12, (r1t["b_real"], bo["b"])
    assert abs(boot["b"] - bo["b"]) < 1e-12 and boot["n_poles_m_ge_2"] == bo["n_poles"], (boot["b"], boot["n_poles_m_ge_2"])
    blo, bhi = boot["b_ci95_bootstrap_poles"]

    x0, width = 80, 780
    x, step = grid_x(x0, width)
    hi = max(1, max(max(births[k]["counts"]) for k in births if k in chance),
             math.ceil(max(max(v) for v in chance.values())))
    ref, (rlo, rhi) = r1t["b_ref_mean"], r1t["b_ref_ci95_over_replicates"]
    where = "below" if bo["b"] < rlo else "inside" if bo["b"] <= rhi else "above"
    caption = [
        "birth: first grid point where the median exemplar reaches the pole's own q99.9 and stays there "
        "(percentile reading); echo clusters left out",
        *wrap("chance level (black line on each bar): each pole's non-echo top-50 exemplars shuffled into groups of "
              f"the real sizes and dated by the same rule; mean of {h['n_draws']} regroupings "
              "(studies/next/06b_VERIFY_BIRTHS.md §3)", 150),
        *wrap(f"birth order (§6.4): b = {bo['b']:.3f} [{blo:.3f}, {bhi:.3f}] over {bo['n_poles']} "
              f"poles (recount bootstrap, 04_b.json); random same-size groups with main's group holding the top exemplar give {ref:.3f} "
              f"[{rlo:.3f}, {rhi:.3f}] (studies/next/10_RECOUNT2.md §7), so b is {where} chance. The §6.4 permutation "
              f"band (mean {bo['null_mean']:.3f}, 97.5th {bo['null_97_5']:.3f}) keeps the real clusters' ties and is "
              "not a chance level for size and rank; 'main oldest' retagged 2026-11-05 (results/RESULTS.md, Retag).",
              150),
    ]
    sources = (f"source: {shown} (sha256 {file_sha256(source)[:16]}); "
               f"studies/next/10_recount2/out/05_chance.json (sha256 {file_sha256(CHANCE)[:16]}); "
               f"studies/next/10_recount2/out/04_b.json (sha256 {file_sha256(BOOT)[:16]}); "
               f"studies/next/06b_verify_births/out/s03_histogram_tv.json (sha256 {file_sha256(HIST)[:16]})")
    footer = wrap(sources, 160) + ["read: q2.births, q2.birth_order; summary.R1t; b_ci95_bootstrap_poles; main_random_mean, "
                                   "non_main_random_mean; drawn by studies/already-clean/runbook/f4_births.py"]
    cap_top = 510
    height = cap_top + 14 * len(caption) + 10 + 12 * len(footer) + 6
    svg = Svg(900, height)

    for i, (key, top) in enumerate((("main", 70), ("non_main", 280))):
        b = births[key]
        y = y_axis(svg, x0, x0 + width, top, 160, 0, hi, sorted({0, hi // 2, hi}), fmt="{:.0f}",
                   label=f"{key.replace('_', '-')} clusters born (n = {b['n']}, never present {b['never_born']})")
        for t, c in enumerate(b["counts"]):
            svg.rect(x(t) - step * 0.35, y(c), step * 0.7, y(0) - y(c), SERIES[i], value=c)
        for t, c in enumerate(chance[key]):
            svg.line(x(t) - step * 0.42, y(c), x(t) + step * 0.42, y(c), stroke=INK, width=1.5)
    grid_ticks(svg, x, 450, step)
    svg.line(x0 + width - 250, 46, x0 + width - 232, 46, stroke=INK, width=1.5)
    svg.text(x0 + width - 226, 49, "chance level: random same-size groups", size=9, fill=NULL)

    mi, nmi = median_index(births["main"]["counts"]), median_index(births["non_main"]["counts"])
    title = (f"Main clusters are born at median {GRID_LABELS[mi] if mi is not None else 'none'}, non-main clusters at "
             f"median {GRID_LABELS[nmi] if nmi is not None else 'none'}")
    for i, line in enumerate(wrap(title, (svg.width - 40) // 8)):
        svg.text(20, 26 + 18 * i, line, size=14, weight="bold")
    for i, line in enumerate(caption):
        svg.text(x0, cap_top + 14 * i, line, size=9, fill=NULL)
    for i, line in enumerate(footer):
        svg.text(20, cap_top + 14 * len(caption) + 10 + 12 * i, line, size=9, fill=NULL)

    return svg.render(), {"height": height, "b": bo["b"], "chance_b": ref, "where": where}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert file_sha256(METRICS).startswith(METRICS_SHA16), f"{METRICS} is not the seventh run's metrics.json"
    text, info = draw(json.loads(METRICS.read_text()), METRICS, "bank/w5/verdict/metrics.json")
    out = a.out / "F4_births.svg"
    out.write_text(text)
    print(json.dumps({"wrote": str(out), **info, "sha256": file_sha256(out)}))


if __name__ == "__main__":
    main()
