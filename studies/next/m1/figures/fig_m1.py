"""M1 figure (M1_SPEC.md section 10), rendered from verdict_m1.json only. Regenerate after any correction.
    python3 -I studies/next/m1/figures/fig_m1.py studies/next/m1/verdict_m1.json studies/next/m1/figures/fig_m1.png
x = layer band; y = C, the band mean of each pole's code+math share of its top-50 slots (absolute).
start_a and start_b are the same weights harvested twice. G, R, M carry the 95 % bootstrap interval over neurons of
their paired shift from start_a, added to C(start_a). The grey box is C(start_a) +- bar(b)."""
import hashlib, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
v = json.loads(raw)
sha = hashlib.sha256(raw).hexdigest()
C, sh, bar = v["exploratory"]["C"], v["exploratory"]["shift_from_start"], v["bar"]
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e1"
series = [("start_a", "start (stage1-step476838)", MUTED, "o", -0.24), ("start_b", "start, second harvest", MUTED, "o", -0.12),
          ("G", "G  gen-mc-only, 0 % code+math", "#2a78d6", "s", 0.0), ("R", "R  round5, 40 %", "#eb6834", "D", 0.12),
          ("M", "M  math-code-reasoning-web, 65 %", "#1baf7a", "^", 0.24)]
bands = ["layers 0-7", "8-15", "16-23", "24-31"]
fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
for b in range(4):
    c0 = C["start_a"][b]
    ax.add_patch(plt.Rectangle((b - 0.36, c0 - bar[b]), 0.72, 2 * bar[b], color=GRID, zorder=0, lw=0))
for k, name, col, mk, dx in series:
    xs = [b + dx for b in range(4)]
    ys = C[k]
    if k in sh:
        lo = [C["start_a"][b] + sh[k][b][1] for b in range(4)]
        hi = [C["start_a"][b] + sh[k][b][2] for b in range(4)]
        ax.vlines(xs, lo, hi, color=col, lw=2, zorder=2)
    face = "none" if k == "start_b" else col
    ax.scatter(xs, ys, s=64, marker=mk, facecolor=face, edgecolor=col if k != "start_b" else MUTED, linewidth=1.6,
               zorder=3, label=name)
for k, name, col, mk, dx in series[1:]:
    ax.annotate(k if k != "start_b" else "start", (3 + dx, C[k][3]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
lead = [x["mean"] for x in v["lead"]]
ax.set_xticks(range(4)); ax.set_xticklabels(bands, color=INK2)
ax.set_xlim(-0.5, 3.6)
ax.set_ylabel("C: mean code+math share of a pole's top-50 slots", color=INK2)
ax.tick_params(colors=INK2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("In every layer band M's code+math share is above G's, by %.3f to %.3f,\nand G is below the shared start"
             % (min(lead), max(lead)), loc="left", fontsize=12, color=INK)
ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
fig.text(0.01, 0.01, ("Verdict: %s. Attempts k0-01, mix-a-01, mix-b-01; verdict_m1.json sha256 %s.\n"
                      "Bars: 95 %% bootstrap over neurons of the shift from start_a, added to C(start_a); they are narrower than the markers. "
                      "Grey box: C(start_a) +- bar(b). start_b equals start_a (G-ID I = 0).")
         % (v["headline"], sha[:16]), fontsize=7.2, color=INK2, va="bottom")
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.17)
fig.savefig(out, facecolor=SURF)
print("wrote", out, "from", src, sha[:16])
