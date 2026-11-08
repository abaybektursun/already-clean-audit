"""F1-F6 (SPEC_D1 §10) as self-contained SVG files, drawn to scale from one unblind's verdict/metrics.json
and nothing else. matplotlib is not in the pinned stack (pyproject.toml), so the SVG is written by hand.
Each figure's title states the finding from the numbers, its footer names the source file (with its
SHA-256) and the keys read, and a missing or PENDING input is refused, naming the file and the key.
F4 is the exception since 2026-11-05: draw() of studies/already-clean/runbook/f4_births.py draws it, which also
reads the chance level of random same-size groups from three files under studies/next and refuses births or b
other than those of the seventh unblind run, on which that chance level was computed.

    python -m infra.pg.neurons_figures --unblind bank/w5 --out bank/w5/figures
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import math
from pathlib import Path

from infra.pg.neurons_verdict import CLASSES, GRID_LABELS, GRID_STEPS, file_sha256, read_json

INK, GRID, NULL = "#1b1a17", "#d9d6cf", "#7a7670"
SERIES = ("#3b6ea8", "#c26a1b", "#5c8a4a", "#8a4f7d")     # one colour per model or series
ESSAY = {"ONE": (0.90, 0.90), "TWO-REL": (0.07, 0.07), "UNREL": (0.02, 0.04)}   # essay.md 29, its own thresholds
FIGURES = ("F1_audit_classes", "F2_strong_rates", "F3_grid_population", "F4_births", "F5_parent_test", "F6_gallery")
F4_BIRTHS = Path(__file__).resolve().parents[2] / "studies" / "already-clean" / "runbook" / "f4_births.py"


class Svg:
    def __init__(self, width: int, height: int):
        self.width, self.height, self.parts = width, height, []

    def rect(self, x, y, w, h, fill, value=None, opacity=1.0):
        data = f' data-value="{value:.6g}"' if value is not None else ""
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" '
                          f'fill-opacity="{opacity}"{data}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
                          f'stroke-width="{width}"{d}/>')

    def circle(self, cx, cy, r, fill):
        self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}"/>')

    def text(self, x, y, s, size=11, anchor="start", weight="normal", fill=INK):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" fill="{fill}">{html.escape(str(s))}</text>')

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
                f'viewBox="0 0 {self.width} {self.height}" font-family="Helvetica, Arial, sans-serif">'
                f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>' + "".join(self.parts) + "</svg>")


def need(metrics: dict, path: Path, *keys):
    """metrics[k1][k2]...; refuses a missing key or a recorded PENDING, naming the file and the key."""
    node = metrics
    for i, k in enumerate(keys):
        if not isinstance(node, dict) or k not in node:
            raise ValueError(f"{path}: no {'.'.join(map(str, keys[:i + 1]))}; the figure cannot be drawn")
        node = node[k]
    if isinstance(node, str) and node.startswith("PENDING"):
        raise ValueError(f"{path}: {'.'.join(map(str, keys))} is {node!r}; the figure cannot be drawn")
    return node


def wrap(text: str, n: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        if cur and len(cur) + 1 + len(word) > n:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return lines + [cur]


def frame(svg: Svg, title: str, source: Path, keys: str) -> None:
    """The title (the finding, wrapped to the width) and the footer naming the source file and the keys read."""
    for i, line in enumerate(wrap(title, (svg.width - 40) // 8)):
        svg.text(20, 26 + 18 * i, line, size=14, weight="bold")
    svg.text(20, svg.height - 24, f"source: {source} (sha256 {file_sha256(source)[:16]})", size=9, fill=NULL)
    svg.text(20, svg.height - 12, f"read: {keys}", size=9, fill=NULL)


def y_axis(svg: Svg, x, right, top, height, lo, hi, ticks, fmt="{:.2f}", label=""):
    """A vertical axis at x from lo (bottom) to hi (top), gridlines to `right`; returns value -> pixel y."""
    def y(v):
        return top + height * (1 - (v - lo) / (hi - lo))
    for t in ticks:
        svg.line(x, y(t), right, y(t), stroke=GRID, width=0.5)
        svg.text(x - 6, y(t) + 4, fmt.format(t), size=9, anchor="end")
    svg.line(x, top, x, top + height)
    if label:
        svg.text(x - 40, top - 8, label, size=10)
    return y


def legend(svg: Svg, x, y, names) -> None:
    for i, name in enumerate(names):
        svg.rect(x, y + 16 * i - 9, 10, 10, SERIES[i % len(SERIES)])
        svg.text(x + 16, y + 16 * i, name, size=10)


def bars_with_intervals(svg, groups, series, value, interval, x0, top, height, width, y):
    """Grouped bars to scale on `y`; each bar's height is value * the axis scale, whiskers are the interval."""
    gw = width / len(groups)
    bw = gw * 0.8 / len(series)
    for gi, g in enumerate(groups):
        for si, s in enumerate(series):
            v, (lo, hi) = value(g, s), interval(g, s)
            bx = x0 + gi * gw + gw * 0.1 + si * bw
            svg.rect(bx, y(v), bw * 0.9, y(0) - y(v), SERIES[si % len(SERIES)], value=v)
            cx = bx + bw * 0.45
            svg.line(cx, y(lo), cx, y(hi), width=1.2)
            svg.line(cx - 3, y(lo), cx + 3, y(lo), width=1.2)
            svg.line(cx - 3, y(hi), cx + 3, y(hi), width=1.2)
        svg.text(x0 + gi * gw + gw / 2, top + height + 16, g, size=10, anchor="middle")
    return gw


def pct(x) -> str:
    return f"{100 * x:.1f}%"


# --- figures -----------------------------------------------------------------------------------

def f1(metrics: dict, source: Path) -> str:
    q1 = need(metrics, source, "q1")
    if not q1:
        raise ValueError(f"{source}: q1 is empty; F1 cannot be drawn")
    models = sorted(q1)
    frac = {m: need(metrics, source, "q1", m, "fractions") for m in models}
    svg = Svg(820, 470)
    top, height, x0, width = 70, 300, 70, 620
    y = y_axis(svg, x0, x0 + width, top, height, 0, 1, [0, 0.25, 0.5, 0.75, 1.0], label="share of poles")
    gw = bars_with_intervals(svg, CLASSES, models, lambda c, m: frac[m][c]["p"], lambda c, m: frac[m][c]["wilson95"],
                             x0, top, height, width, y)
    for gi, c in enumerate(CLASSES):
        if c in ESSAY:
            lo, hi = ESSAY[c]
            gx = x0 + gi * gw
            if hi > lo:
                svg.rect(gx + 4, y(hi), gw - 8, y(lo) - y(hi), NULL, opacity=0.35)
            svg.line(gx + 4, y(hi), gx + gw - 4, y(hi), stroke=NULL, width=1.5, dash="5 3")
    legend(svg, x0 + width + 16, top + 10, models)
    svg.text(x0 + width + 16, top + 30 + 16 * len(models), "dashed: the essay's", size=9, fill=NULL)
    svg.text(x0 + width + 16, top + 42 + 16 * len(models), "90 / 7 / 2–4, its own", size=9, fill=NULL)
    svg.text(x0 + width + 16, top + 54 + 16 * len(models), "thresholds, not ours", size=9, fill=NULL)
    title = "One job under passing explanations: " + "; ".join(
        f"{m} {pct(frac[m]['ONE']['p'])} [{pct(frac[m]['ONE']['wilson95'][0])}, {pct(frac[m]['ONE']['wilson95'][1])}]"
        for m in models)
    frame(svg, title, source, "q1.<model>.fractions (Wilson 95%)")
    return svg.render()


def f2(metrics: dict, source: Path) -> str:
    q1 = need(metrics, source, "q1")
    if not q1:
        raise ValueError(f"{source}: q1 is empty; F2 cannot be drawn")
    models = sorted(q1)
    rates = {m: need(metrics, source, "q1", m, "strong_rates") for m in models}
    kinds = ("pos", "near", "distant", "shuffled")
    for m in models:
        for k in kinds:
            if "rate" not in rates[m].get(k, {}):
                raise ValueError(f"{source}: q1.{m}.strong_rates.{k} has no rate; F2 cannot be drawn")
    svg = Svg(820, 470)
    top, height, x0, width = 70, 300, 70, 620
    y = y_axis(svg, x0, x0 + width, top, height, 0, 1, [0, 0.25, 0.5, 0.75, 1.0], label="share of texts strong (score ≥ θ_s)")
    bars_with_intervals(svg, kinds, models, lambda k, m: rates[m][k]["rate"], lambda k, m: rates[m][k]["wilson95"],
                        x0, top, height, width, y)
    svg.line(x0, y(0.01), x0 + width, y(0.01), stroke=NULL, width=1.5, dash="5 3")
    legend(svg, x0 + width + 16, top + 10, models)
    svg.text(x0 + width + 16, top + 30 + 16 * len(models), "dashed: 1%, θ_s is the", size=9, fill=NULL)
    svg.text(x0 + width + 16, top + 42 + 16 * len(models), "99th percentile of", size=9, fill=NULL)
    svg.text(x0 + width + 16, top + 54 + 16 * len(models), "discovery shuffled scores", size=9, fill=NULL)
    title = "Strong positives against strong shuffled texts: " + "; ".join(
        f"{m} {pct(rates[m]['pos']['rate'])} vs {pct(rates[m]['shuffled']['rate'])}" for m in models)
    frame(svg, title, source, "q1.<model>.strong_rates (non-RARE poles, Wilson 95%)")
    return svg.render()


def grid_x(x0, width):
    step = width / len(GRID_STEPS)
    return lambda t: x0 + step * (t + 0.5), step


def grid_ticks(svg, x, top, step):
    for t, s in enumerate(GRID_STEPS):
        svg.text(x(t), top + 14, GRID_LABELS[t], size=9, anchor="middle")
        svg.text(x(t), top + 25, f"{s // 1000}k" if s >= 1000 else str(s), size=8, anchor="middle", fill=NULL)


def boxes(svg, x, y, q, step, color):
    """q = [q05, q25, q50, q75, q95] per grid point; a box from q25 to q75, whiskers to q05 and q95."""
    for t, v in enumerate(q):
        svg.line(x(t), y(v[0]), x(t), y(v[4]), stroke=color)
        svg.rect(x(t) - step * 0.25, y(v[3]), step * 0.5, y(v[1]) - y(v[3]), color, opacity=0.35)
        svg.line(x(t) - step * 0.25, y(v[2]), x(t) + step * 0.25, y(v[2]), stroke=color, width=2)


def f3(metrics: dict, source: Path) -> str:
    gp = need(metrics, source, "q2", "grid_population")
    pts = gp["per_grid_point"]
    if len(pts) != len(GRID_STEPS):
        raise ValueError(f"{source}: q2.grid_population has {len(pts)} grid points, expected {len(GRID_STEPS)}")
    svg = Svg(900, 720)
    x0, width = 90, 760
    x, step = grid_x(x0, width)
    dead = [p["dead_share"] for p in pts]
    y1 = y_axis(svg, x0, x0 + width, 70, 130, 0, 1, [0, 0.5, 1.0], label="dead share (max|a| = 0)")
    for t in range(1, len(pts)):
        svg.line(x(t - 1), y1(dead[t - 1]), x(t), y1(dead[t]), stroke=SERIES[0], width=1.5)
    for t, v in enumerate(dead):
        svg.circle(x(t), y1(v), 3, SERIES[0])
    y2 = y_axis(svg, x0, x0 + width, 250, 160, 0, 1, [0, 0.25, 0.5, 0.75, 1.0], label="double-negative share of the top-50")
    boxes(svg, x, y2, [p["double_negative_share"] for p in pts], step, SERIES[1])
    dens = [[math.log10(max(v, 1.0)) for v in p["density"]] for p in pts]
    hi = max(1, math.ceil(max(v for q in dens for v in q)))
    y3 = y_axis(svg, x0, x0 + width, 460, 160, 0, hi, list(range(0, hi + 1)), fmt="1e{:.0f}", label="density (tokens ≥ 0.01·max|a|), log scale")
    boxes(svg, x, y3, dens, step, SERIES[2])
    grid_ticks(svg, x, 620, step)
    title = (f"From T_0 to T_15, dead share {dead[0]:.3f} to {dead[-1]:.3f}; median double-negative share "
             f"{pts[0]['double_negative_share'][2]:.2f} to {pts[-1]['double_negative_share'][2]:.2f}")
    svg.text(x0, 680, f"boxes: quartiles; whiskers: 5th and 95th percentiles over live poles of every layer "
                      f"(n = {pts[-1]['n_poles']} poles per grid point); exploratory", size=9, fill=NULL)
    frame(svg, title, source, "q2.grid_population.per_grid_point")
    return svg.render()


def median_index(counts):
    n = sum(counts)
    if n == 0:
        return None
    run = 0
    for t, c in enumerate(counts):
        run += c
        if run >= n / 2:
            return t


def load_f4_births():
    """The F4 generator of the 2026-11-05 retag, loaded from its file ("already-clean" is not a package name)."""
    spec = importlib.util.spec_from_file_location("f4_births", F4_BIRTHS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f4(metrics: dict, source: Path) -> str:
    # 2026-11-05: the birth-order claim the old F4 stated ("main oldest", above the §6.4 band) is retagged, not supported
    # (studies/already-clean/results/RESULTS.md, Retag). The F4 of the retag adds the chance level of random same-size
    # groups, which is not in metrics.json; its generator reads it from the recount files and is called here.
    return load_f4_births().draw(metrics, source, str(source))[0]


def f5(metrics: dict, source: Path) -> str:
    pu = need(metrics, source, "q2", "p_u")
    gate = need(metrics, source, "judge_gate")
    svg = Svg(920, 480)
    top, height, x0, width = 70, 300, 70, 480
    y = y_axis(svg, x0, x0 + width, top, height, 0, 1, [0, 0.25, 0.5, 0.75, 1.0], label="judge accuracy")
    if pu.get("valid"):
        hard = need(metrics, source, "q2", "hard")
        bars = [("late-born, random", pu["A"]), ("late-born, hard", hard["accuracy"]),
                ("control, random", pu["a_ctrl"]), ("control, hard", hard["ctrl_accuracy"])]
        title = (f"p_u = {pu['point']:.3f} [{pu['ci95'][0]:.3f}, {pu['ci95'][1]:.3f}]: {pu['rule']}"
                 + (" (underpowered)" if pu["underpowered"] else "")
                 + f"; {pu['n_clusters']} late-born clusters on {pu['n_poles']} poles")
    else:
        bars = [("gate control, random", gate["a_ctrl"]), ("gate control, hard", gate["hard_ctrl"])]
        title = f"Parent test not valid: {pu['reason']}"
    bw = width / len(bars)
    for i, (name, v) in enumerate(bars):
        svg.rect(x0 + i * bw + bw * 0.2, y(v), bw * 0.6, y(0) - y(v), SERIES[0 if i < 2 else 1], value=v)
        svg.text(x0 + i * bw + bw / 2, y(v) - 4, f"{v:.3f}", size=10, anchor="middle")
        svg.text(x0 + i * bw + bw / 2, top + height + 16, name, size=10, anchor="middle")
    svg.line(x0, y(0.5), x0 + width, y(0.5), stroke=NULL, width=1.5, dash="5 3")
    svg.text(x0 + width - 4, y(0.5) - 4, "chance", size=9, anchor="end", fill=NULL)
    px = x0 + width + 120
    yp = y_axis(svg, px, px + 110, top, height, 0, 1, [0, 0.1, 0.2, 0.5, 1.0], label="p_u")
    for v, name in ((0.10, "H_RG if upper < 0.10"), (0.20, "H_SP if lower > 0.20")):
        svg.line(px, yp(v), px + 110, yp(v), stroke=NULL, dash="5 3")
        svg.text(px + 112, yp(v) + 3, name, size=8, fill=NULL)
    if pu.get("valid"):
        svg.line(px + 50, yp(pu["ci95"][0]), px + 50, yp(pu["ci95"][1]), stroke=SERIES[0], width=2)
        svg.circle(px + 50, yp(pu["point"]), 4, SERIES[0])
    else:
        svg.text(px + 10, top + height / 2, "not computed", size=10, fill=NULL)
    frame(svg, title, source, "q2.p_u, q2.hard, judge_gate")
    return svg.render()


def snippet(marked: str, before: int = 60, after: int = 30) -> str:
    """The part of a marked context around the ⟦ ⟧ marker."""
    i, j = marked.find("⟦"), marked.find("⟧")
    if i < 0 or j < 0:
        raise ValueError(f"no ⟦ ⟧ marker in a gallery exemplar: {marked[:80]!r}")
    return ("…" if i > before else "") + marked[max(0, i - before):j + 1 + after] + ("…" if j + 1 + after < len(marked) else "")


def f6(metrics: dict, source: Path) -> str:
    entries = need(metrics, source, "q2", "gallery")
    heights = [80 + 58 * sum(not c["echo"] for c in e["clusters"]) if e["pole"] else 70 for e in entries]
    svg = Svg(1000, 100 + sum(heights) + 60)
    x0, width = 60, 330
    x, step = grid_x(x0, width)
    top = 70
    for e, h in zip(entries, heights):
        if e["pole"] is None:
            svg.text(x0, top + 20, f"{e['group']}: {e['note']}", size=11, weight="bold")
            top += h
            continue
        jobs = sorted((c for c in e["clusters"] if not c["echo"]),
                      key=lambda c: (not c["is_main"], c["birth_pct"] if c["birth_pct"] is not None else 99))
        svg.text(x0, top + 14, f"{e['pole']} ({e['class']}; group {e['group']}): {len(jobs)} clusters, "
                               f"{sum(c['echo'] for c in e['clusters'])} echo", size=11, weight="bold")
        rows_top = top + 34
        main_y = rows_top + 10
        for k, c in enumerate(jobs):
            ry = rows_top + 10 + 58 * k
            b = c["birth_pct"]
            if b is not None:
                if not c["is_main"]:
                    svg.line(x(b), main_y, x(b), ry, stroke=NULL, dash="3 2")
                svg.line(x(b), ry, x(len(GRID_STEPS) - 1), ry, stroke=SERIES[0] if c["is_main"] else SERIES[1], width=2)
                for t, present in enumerate(c["present_pct"]):
                    if present:
                        svg.circle(x(t), ry, 2.2, SERIES[0] if c["is_main"] else SERIES[1])
                svg.circle(x(b), ry, 5, INK)
            r = f"r(c) = {c['r_at_birth']:.2f}" if c["r_at_birth"] is not None else "never present"
            label = f"{'main: ' if c['is_main'] else ''}{c['name']} · born {GRID_LABELS[b] if b is not None else '—'} · {r} · {c['status']}"
            svg.text(x0 + width + 20, ry + 4, label, size=10, weight="bold" if c["is_main"] else "normal")
            for i, ex in enumerate(c["exemplars"]):
                svg.text(x0 + width + 20, ry + 18 + 12 * i, snippet(ex), size=8, fill=NULL)
        for t in range(len(GRID_STEPS)):
            svg.text(x(t), top + h - 12, GRID_LABELS[t].replace("T_", ""), size=7, anchor="middle", fill=NULL)
        top += h
    shown = [e for e in entries if e["pole"]]
    title = "Three poles over training: " + "; ".join(
        f"{e['pole']} ({e['class']}) {sum(not c['echo'] for c in e['clusters'])} clusters" for e in shown)
    svg.text(x0, svg.height - 44, "each row a cluster: dot = present (percentile reading), black node = birth, "
                                  "dashed = branch from the main cluster's line at the birth; grid index below", size=9, fill=NULL)
    frame(svg, title, source, "q2.gallery")
    return svg.render()


def build(unblind: Path, out: Path) -> list[Path]:
    source = unblind / "verdict" / "metrics.json"
    metrics = read_json(source)
    paths = [out / f"{name}.svg" for name in FIGURES]
    existing = [p for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"figures exist: {existing}")
    svgs = [fn(metrics, source) for fn in (f1, f2, f3, f4, f5, f6)]     # every figure is drawn before any is written
    out.mkdir(parents=True, exist_ok=True)
    for p, s in zip(paths, svgs):
        p.write_text(s, encoding="utf-8")
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="neurons_figures")
    ap.add_argument("--unblind", required=True, help="the unblind --out directory (holding verdict/metrics.json)")
    ap.add_argument("--out", required=True, help="directory for F1-F6 (.svg)")
    a = ap.parse_args(argv)
    for p in build(Path(a.unblind), Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
