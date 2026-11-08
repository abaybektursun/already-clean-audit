"""B recomputed from the job's activations a and the banked grid histograms, for every (unit, role) whose
(point, layer) histogram was fetched (fetch_stats.py --hist), over all 22,016 poles. This reader's percentile, written
from M2_SPEC.md section 0.5: v = numpy median of a+ over the positions; B = (tokens in bins below v's bin + the
log-linear share of v's bin) / all own-sign tokens; 0 when v = 0. Also the dating rule at the pole: median a+
against the banked q999 at T_(k-1) and T_k."""
import argparse, json, pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arrays", required=True)
ap.add_argument("--stats", required=True)
a = ap.parse_args()
here = pathlib.Path(__file__).resolve().parent
root, sd = pathlib.Path(a.arrays), pathlib.Path(a.stats)
units = json.load(open(here.parent / "units.json"))["units"]


def my_B(acts, hist, edges):
    x = acts.astype(np.float64)
    v = np.stack([np.median(np.maximum(x, 0), 0), np.median(np.maximum(-x, 0), 0)], -1)  # [N, 2]
    le = np.log10(edges.astype(np.float64))
    lv = np.log10(np.where(v > 0, v, 1.0))
    b = np.clip(np.searchsorted(le, lv, "right") - 1, 0, hist.shape[-1] - 1)
    h = hist.astype(np.float64)
    cum = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(h, -1)], -1)
    below = np.take_along_axis(cum, b[..., None], -1)[..., 0]
    inbin = np.take_along_axis(h, b[..., None], -1)[..., 0]
    frac = np.clip((lv - le[b]) / (le[b + 1] - le[b]), 0, 1)
    n = h.sum(-1)
    return np.where((v > 0) & (n > 0), (below + frac * inbin) / np.maximum(n, 1), 0.0), v


out = []
for f in sorted(sd.glob("hist_t*_L*.npz")):
    t, layer = int(f.name[6:8]), int(f.name[10:12])
    st = np.load(f)
    for u in units:
        for role in ("pre", "birth", "end"):
            if u["layer"] != layer or u["roles"].get(role) != t:
                continue
            z = np.load(root / f"t{t:02d}" / f"{u['unit'].replace(':', '_')}_{role}.npz")
            B, v = my_B(z["a"], st["hist"], st["hist_edges"])
            d = np.abs(B - z["B"])
            n, s = u["carrier"]
            out.append({"unit": u["unit"], "role": role, "t": t, "layer": layer, "max_abs_dB_all_poles": float(d.max()),
                        "n_poles_dB_gt_1e-12": int((d > 1e-12).sum()),
                        "pole_v_over_q999": float(v[n, s] / st["q999"][n, s]), "pole_B": float(z["B"][n, s])})
(here / "r6_bcheck.json").write_text(json.dumps(out, indent=1))
for r in out:
    print(r)
