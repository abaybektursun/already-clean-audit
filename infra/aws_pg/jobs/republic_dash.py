"""republic_dash — self-contained dashboard.html from the tidy results frame.

Design system (validated with the dataviz palette checker on both surfaces):
- Origin is the categorical channel (4 slots max): US blue, CN gold, Europe violet,
  Other green. Per-model identity comes from hover-highlight + the table, never from
  more hues.
- The A-profile is a dot plot (rows = values, x = A in [-1,+1]); CI whiskers render
  only for the hovered/focused model, so 18 models don't paint 18 overlapping bars.
- Instrument health is a tile grid with per-model mean-S-by-depth sparklines
  (replaces the unreadable 18-line chart). Models failing min S >= 0.75 at their best
  layer are FLAGGED: hollow dots in the profile, labeled chip in the grid.
- No CDN, no framework; JS only enhances (tooltip + highlight); table view under
  <details> keeps every plotted number reachable without hover.
"""
import datetime as dt
import json

ORIGIN_GROUP = {"US": "United States", "CN": "China", "EU": "Europe", "CH": "Europe",
                "UAE": "Other", "KR": "Other", "JP": "Other", "RU": "Other"}
GROUPS = ["United States", "China", "Europe", "Other"]
LIGHT = {"United States": "#2a78d6", "China": "#a86e00", "Europe": "#4a3aa7", "Other": "#008300"}
DARK = {"United States": "#3987e5", "China": "#c98500", "Europe": "#9085e9", "Other": "#0ca30c"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CSS = """
:root { --surface:#faf7f0; --page:#f5f1e8; --ink:#1c1b18; --ink2:#57544c; --muted:#8a867c;
  --grid:#e6e1d4; --axis:#c9c4b4; --card:#fffdf8;
  --us:#2a78d6; --cn:#a86e00; --eu:#4a3aa7; --ot:#008300; --hi:#1c1b18; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --page:#111110; --ink:#f2f0ea; --ink2:#c3c0b4; --muted:#8a867c;
    --grid:#2c2c29; --axis:#3a3a35; --card:#201f1d;
    --us:#3987e5; --cn:#c98500; --eu:#9085e9; --ot:#0ca30c; --hi:#f2f0ea; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:40px 28px 80px; }
.kicker { font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); }
h1 { font-size:26px; font-weight:650; margin:10px 0 6px; }
.note { color:var(--ink2); max-width:72ch; margin:0 0 8px; }
.stamp { font:11px ui-monospace,Menlo,monospace; color:var(--muted); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:28px 0 8px; }
.tile { background:var(--card); border:1px solid var(--grid); border-radius:10px; padding:14px 16px; }
.tile .l { font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); }
.tile .v { font-size:34px; font-weight:620; margin-top:6px; }
.tile .s { font-size:12px; color:var(--ink2); }
section { margin-top:36px; }
h2 { font-size:17px; font-weight:640; margin:0 0 2px; }
.sub { color:var(--ink2); font-size:13px; margin:0 0 14px; max-width:78ch; }
.legend { display:flex; flex-wrap:wrap; gap:18px; align-items:center; margin:0 0 10px;
  font:12px system-ui,sans-serif; color:var(--ink2); }
.key { display:inline-flex; align-items:center; gap:7px; }
.key .sw { width:14px; height:3px; border-radius:2px; display:inline-block; }
.key .dotk { width:9px; height:9px; border-radius:50%; display:inline-block;
  border:2px solid var(--muted); background:transparent; }
.hint { margin-left:auto; color:var(--muted); font-size:12px; }
.chart-card { background:var(--card); border:1px solid var(--grid); border-radius:10px;
  padding:18px 18px 10px; overflow-x:auto; }
svg { display:block; }
svg text { font:11.5px system-ui,sans-serif; fill:var(--ink2); }
svg .lab { font-size:13px; fill:var(--ink); }
svg .mono { font:10px ui-monospace,Menlo,monospace; fill:var(--muted); letter-spacing:.08em; }
.dot { stroke:var(--card); stroke-width:2; cursor:pointer; transition:opacity .12s; }
.dot.hollow { fill:var(--card); stroke-width:2.5; }
.hit { fill:transparent; cursor:pointer; }
.whisker { stroke-width:2; opacity:0; transition:opacity .12s; }
body.hl .dot { opacity:.18; }
body.hl .dot.on { opacity:1; }
body.hl .whisker.on { opacity:.9; }
.grid-h { display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:12px; }
.hcard { background:var(--card); border:1px solid var(--grid); border-radius:10px; padding:13px 15px; }
.hcard .name { font-weight:630; font-size:14px; }
.hcard .meta { font:11px ui-monospace,Menlo,monospace; color:var(--ink2); margin:5px 0 7px; }
.chip { display:inline-flex; align-items:center; gap:6px; font:600 10px/1 ui-monospace,Menlo,monospace;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink2); }
.chip .sw { width:10px; height:10px; border-radius:50%; }
.flag { color:#a33b00; font-weight:700; }
@media (prefers-color-scheme: dark) { .flag { color:#ec835a; } }
details { margin-top:26px; }
summary { cursor:pointer; color:var(--ink2); font-size:14px; }
table { border-collapse:collapse; font-size:12px; margin-top:12px; }
td,th { padding:3px 10px; border-bottom:1px solid var(--grid); text-align:right;
  font-variant-numeric:tabular-nums; }
th { color:var(--muted); font-weight:600; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
#tip { position:fixed; pointer-events:none; background:var(--ink); color:var(--surface);
  border-radius:7px; padding:8px 11px; font-size:12px; line-height:1.45; opacity:0;
  transition:opacity .1s; max-width:260px; z-index:10; }
#tip .big { font-size:14px; font-weight:650; }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function show(el, x, y) {
    tip.textContent = '';
    var big = document.createElement('div'); big.className = 'big';
    big.textContent = el.dataset.a;
    var name = document.createElement('div'); name.textContent = el.dataset.name;
    var rest = document.createElement('div'); rest.textContent = el.dataset.rest;
    tip.append(big, name, rest);
    tip.style.left = Math.min(x + 14, window.innerWidth - 280) + 'px';
    tip.style.top = (y + 16) + 'px';
    tip.style.opacity = 1;
    document.body.classList.add('hl');
    var m = el.dataset.m;
    document.querySelectorAll('.dot,.whisker').forEach(function (d) {
      d.classList.toggle('on', d.dataset.m === m);
    });
  }
  function hide() {
    tip.style.opacity = 0;
    document.body.classList.remove('hl');
    document.querySelectorAll('.on').forEach(function (d) { d.classList.remove('on'); });
  }
  document.querySelectorAll('.hit').forEach(function (h) {
    h.addEventListener('pointermove', function (e) { show(h, e.clientX, e.clientY); });
    h.addEventListener('pointerleave', hide);
    h.setAttribute('tabindex', '0');
    h.addEventListener('focus', function () {
      var r = h.getBoundingClientRect(); show(h, r.left, r.top);
    });
    h.addEventListener('blur', hide);
  });
})();
"""


def _prep(df):
    """Per model: best layer, per-value stats at that layer, mean-S depth curve, flag."""
    out = []
    for model, m in df.groupby("model"):
        s = m[m.metric == "S_auroc"]
        depth = s.groupby("layer").estimate.mean()
        best = int(depth.idxmax())
        at = m[m.layer == best].set_index(["value", "metric"]).estimate
        lohi = m[m.layer == best].set_index(["value", "metric"])
        vals = sorted({v for v, _ in at.index})
        group = ORIGIN_GROUP[m.origin.iloc[0]]
        min_s = min(at[(v, "S_auroc")] for v in vals)
        out.append(dict(
            model=model, origin=m.origin.iloc[0], group=group, best=best,
            n_layers=int(m.layer.max()), min_s=min_s,
            gap=float(sum(at[(v, "specificity_gap")] for v in vals) / len(vals)),
            flagged=min_s < 0.75,
            curve=[float(x) for x in depth.sort_index().tolist()],
            values={v: dict(A=float(at[(v, "A_proj")]),
                            lo=float(lohi.ci_lo[(v, "A_proj")]),
                            hi=float(lohi.ci_hi[(v, "A_proj")]),
                            S=float(at[(v, "S_auroc")]),
                            gap=float(at[(v, "specificity_gap")])) for v in vals}))
    return out


def _profile_svg(models, values):
    lp, rp, tp, rh = 190, 26, 34, 34
    w = 1020
    h = tp + rh * len(values) + 30
    X = lambda a: lp + (max(-1.15, min(1.15, a)) + 1.15) / 2.3 * (w - lp - rp)
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Alignment profile dot plot">']
    p.append(f'<text class="mono" x="{X(-1)}" y="{tp-18}" text-anchor="middle">-1 VIOLATE POLE</text>')
    p.append(f'<text class="mono" x="{X(0)}" y="{tp-18}" text-anchor="middle">0</text>')
    p.append(f'<text class="mono" x="{X(1)}" y="{tp-18}" text-anchor="middle">+1 AFFIRM POLE</text>')
    for gx in (-1, -0.5, 0, 0.5, 1):
        col = "var(--axis)" if gx == 0 else "var(--grid)"
        p.append(f'<line x1="{X(gx)}" y1="{tp-8}" x2="{X(gx)}" y2="{h-22}" stroke="{col}" stroke-width="1"/>')
    for i, v in enumerate(values):
        yc = tp + i * rh + rh / 2
        p.append(f'<text class="lab" x="{lp-14}" y="{yc+4}" text-anchor="end">{v.replace("_", " ")}</text>')
        if i:
            p.append(f'<line x1="{lp-4}" y1="{tp+i*rh}" x2="{w-rp}" y2="{tp+i*rh}" stroke="var(--grid)" stroke-width="0.5"/>')
        marks = []
        for j, mo in enumerate(models):
            d = mo["values"].get(v)
            if d is None:
                continue
            y = yc + ((j % 5) - 2) * 3.2
            c = f'var(--{ {"United States":"us","China":"cn","Europe":"eu","Other":"ot"}[mo["group"]] })'
            marks.append(
                f'<line class="whisker" data-m="{mo["model"]}" x1="{X(d["lo"])}" y1="{y}" '
                f'x2="{X(d["hi"])}" y2="{y}" stroke="{c}"/>')
            shape = (f'<circle class="dot hollow" data-m="{mo["model"]}" cx="{X(d["A"])}" cy="{y}" r="5" stroke="{c}"/>'
                     if mo["flagged"] else
                     f'<circle class="dot" data-m="{mo["model"]}" cx="{X(d["A"])}" cy="{y}" r="5" fill="{c}"/>')
            flag = " · instrument FLAGGED (read with caution)" if mo["flagged"] else ""
            marks.append(shape)
            marks.append(
                f'<circle class="hit" data-m="{mo["model"]}" data-a="A {d["A"]:+.2f}" '
                f'data-name="{mo["model"]} · {mo["group"]}" '
                f'data-rest="95% CI [{d["lo"]:+.2f}, {d["hi"]:+.2f}] · S {d["S"]:.2f} · '
                f'specificity {d["gap"]:+.2f} · layer {mo["best"]}{flag}" '
                f'cx="{X(d["A"])}" cy="{y}" r="13"/>')
        p += marks
    p.append("</svg>")
    return "".join(p)


def _spark(curve, flagged):
    w, h = 180, 30
    n = len(curve)
    pts = " ".join(f"{i/(n-1)*w:.1f},{h - (max(0.5, min(1, s)) - 0.5) / 0.5 * (h-3):.1f}"
                   for i, s in enumerate(curve))
    col = "var(--muted)" if flagged else "var(--us)"
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<line x1="0" y1="{h-1}" x2="{w}" y2="{h-1}" stroke="var(--grid)"/>'
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
            f'stroke-linejoin="round"><title>mean S by layer (0.5 to 1.0)</title></polyline></svg>')


def _heatmap(m_df, values):
    s = m_df[m_df.metric == "S_auroc"]
    layers = sorted(s.layer.unique())
    cw = max(7, min(15, 760 // len(layers)))
    lp, tp, rh = 185, 8, 17
    w, h = lp + cw * len(layers) + 10, tp + rh * len(values) + 24
    piv = s.pivot(index="value", columns="layer", values="estimate")
    p = [f'<svg viewBox="0 0 {w} {h}">']
    for i, v in enumerate(values):
        p.append(f'<text x="{lp-10}" y="{tp+i*rh+12}" text-anchor="end">{v.replace("_", " ")}</text>')
        for j, l in enumerate(layers):
            t = max(0.0, min(1.0, (piv.loc[v, l] - 0.5) / 0.5))
            c = SEQ[round(t * (len(SEQ) - 1))]
            p.append(f'<rect x="{lp+j*cw}" y="{tp+i*rh}" width="{cw-1}" height="{rh-1}" fill="{c}">'
                     f'<title>{v} · layer {l} · S={piv.loc[v, l]:.3f}</title></rect>')
    for j, l in enumerate(layers):
        if j % max(1, len(layers) // 10) == 0:
            p.append(f'<text class="mono" x="{lp+j*cw+cw/2}" y="{h-8}" text-anchor="middle">{l}</text>')
    p.append("</svg>")
    return "".join(p)


def render_dashboard(df, out_dir, battery_version):
    prim = df[~df.metric.str.endswith("_raw")]
    models = sorted(_prep(prim), key=lambda m: (GROUPS.index(m["group"]), m["model"]))
    values = sorted({r for r in prim["value"].unique()})
    # most contested first: sort by cross-model A range among cleared instruments
    cleared = [m for m in models if not m["flagged"]] or models
    values.sort(key=lambda v: -(max(m["values"][v]["A"] for m in cleared if v in m["values"])
                                - min(m["values"][v]["A"] for m in cleared if v in m["values"])))

    n_cells = sum(len(m["values"]) for m in models)
    n_sep = sum(1 for m in models for v in m["values"].values() if v["S"] >= 0.75)
    med_gap = sorted(m["gap"] for m in models)[len(models) // 2]
    n_flag = sum(1 for m in models if m["flagged"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tiles = [("models assayed", str(len(models)), f"{n_flag} flagged"),
             ("civic values", str(len(values)), f"battery {battery_version}"),
             ("linearly separable", f"{100*n_sep//n_cells}%", "of model x value cells, S >= 0.75"),
             ("median specificity", f"{med_gap:+.2f}", "gap vs held-out controls")]
    tile_html = "".join(f'<div class="tile"><div class="l">{l}</div><div class="v">{v}</div>'
                        f'<div class="s">{s}</div></div>' for l, v, s in tiles)

    legend = "".join(
        f'<span class="key"><span class="sw" style="background:var(--{k})"></span>{g}</span>'
        for g, k in [("United States", "us"), ("China", "cn"), ("Europe", "eu"), ("Other", "ot")])
    legend += '<span class="key"><span class="dotk"></span>flagged instrument</span>'

    health = []
    for m in models:
        k = {"United States": "us", "China": "cn", "Europe": "eu", "Other": "ot"}[m["group"]]
        status = ('<span class="flag">&#9888; FLAGGED &mdash; probes unreliable</span>'
                  if m["flagged"] else '<span>&#10003; cleared</span>')
        health.append(
            f'<div class="hcard"><div class="name">{m["model"]}</div>'
            f'<div class="chip"><span class="sw" style="background:var(--{k})"></span>{m["group"]} · {m["origin"]}</div>'
            f'<div class="meta">best layer {m["best"]}/{m["n_layers"]} · min S {m["min_s"]:.2f} · '
            f'gap {m["gap"]:+.2f}<br>{status}</div>{_spark(m["curve"], m["flagged"])}</div>')

    heat = "".join(f"<h2 style='margin-top:20px'>{m['model']}</h2>{_heatmap(prim[prim.model == m['model']], sorted(values))}"
                   for m in models)

    tab = ['<table><tr><th>model</th><th>value</th><th>A</th><th>ci lo</th><th>ci hi</th>'
           '<th>S</th><th>gap</th><th>layer</th></tr>']
    for m in models:
        for v in sorted(m["values"]):
            d = m["values"][v]
            tab.append(f'<tr><td>{m["model"]}</td><td>{v}</td><td>{d["A"]:+.3f}</td>'
                       f'<td>{d["lo"]:+.3f}</td><td>{d["hi"]:+.3f}</td><td>{d["S"]:.3f}</td>'
                       f'<td>{d["gap"]:+.3f}</td><td>{m["best"]}</td></tr>')
    tab.append("</table>")

    html = f"""<style>{CSS}</style>
<div class="wrap">
<div class="kicker">Republic Assay · weights-level observatory</div>
<h1>Civic Alignment Profile</h1>
<p class="note">Linear probes read each model's residual stream on a matched contrastive battery
of twelve civic values. <b>A</b> is the model's default reading of neutral civic dilemmas
(&minus;1 = violate pole, +1 = affirm pole); <b>S</b> is probe separability (held-out AUROC);
specificity is measured against held-out non-civic controls after projecting out the generic
good/bad valence axis.</p>
<div class="stamp">battery {battery_version} · generated {stamp} · every number binds to a pinned model revision</div>
<div class="tiles">{tile_html}</div>

<section>
<h2>Default-state alignment, value by value</h2>
<p class="sub">One dot per model at its best-separability layer, sorted by cross-model disagreement.
Hover or tab to a dot to follow one model across all values and see its confidence interval.</p>
<div class="legend">{legend}<span class="hint">hover a dot &mdash; tooltip + follow the model</span></div>
<div class="chart-card">{_profile_svg(models, values)}</div>
</section>

<section>
<h2>Instrument health, per model</h2>
<p class="sub">Sparkline: how linearly readable the twelve values are, by depth (0.5 = chance, top = 1.0).
A model is flagged when any value fails S &ge; 0.75 at its best layer &mdash; its A dots are hollow above
and should not be compared until resolved.</p>
<div class="grid-h">{"".join(health)}</div>
</section>

<details><summary>Separability by layer, per model (heatmaps)</summary>{heat}</details>
<details><summary>Table view &mdash; every plotted number (full data: results.parquet)</summary>{"".join(tab)}</details>
</div>
<div id="tip" role="status"></div>
<script>{JS}</script>
"""
    (out_dir / "dashboard.html").write_text(html)
