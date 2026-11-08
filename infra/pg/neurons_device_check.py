"""Device check for the W3 birth readings: one checkpoint harvested on two GPUs (LOCAL, banked arrays only).

    python -m infra.pg.neurons_device_check --a <dir A> --b <dir B> --poles <grid poles.json> --out <report.json>
        [--clusters <explain>/explain/clusters.jsonl]

A and B are two harvest directories of the same checkpoint and step (e.g. grid/1b/out/t14_step1000000 from an
RTX PRO 5000 and grid/1c/out/t14_step1000000 from an L40S). Asserted equal before anything is compared:
poles/positions.npz `tok`, poles/poles.json, manifest tokens_sha256 and n_tokens_scored.

Per pole (layer l, neuron n, sign s) it compares:
  exemplar_a  poles/exemplar_a.npz `a` [P, 50] at the T_15 positions: relative difference over every filled
              entry, and the share of entries whose sign differs.
  presence    the two birth-rule readings on each device, as the verdict (6d26acff) computes them; the verdict
              module is not in this tree, so the rule is copied: med = median (float64) over the filled exemplars
              of a+ = max(s*a, 0), s = +1 for the + pole and -1 for the - pole; percentile, med > 0 and
              med >= stats q999[n, s]; raw-ratio, med > 0 and med >= 0.3 * stats max_abs[n, s]. The unit here is
              the pole, a proxy: the verdict reads presence per cluster (median over one cluster's sample-0
              exemplar_ranks). With --clusters the same rule is also read per cluster.
  stats       q999 and max_abs at [n, s] from stats/layer_{l:02d}.npz.
  topk        overlap of the top-20 `tok` entries of topk/layer_{l:02d}.npz at [n, s] (the records the judge sees).
Relative difference = |x_A - x_B| / max(|x_A|, |x_B|) (0 when both are 0), so it is symmetric and at most 2.
Fail-loud: no try/except; numpy only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

POLE_RE = re.compile(r"^L(\d+)N(\d+)([+-])$")         # the pattern of neurons_harvest.POLE_RE
RAW_FRAC = 0.3
TOP = 20
OVERLAP_FLOOR = 15


def parse_pole(p: str) -> tuple[int, int, int]:
    m = POLE_RE.match(p)
    assert m, f"bad pole id {p!r}"
    return int(m[1]), int(m[2]), "+-".index(m[3])


def rel_diff(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    den = np.maximum(np.abs(x), np.abs(y))
    return np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1), 0.0)


def summary(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


class Dir:
    def __init__(self, path):
        self.root = pathlib.Path(path)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.poles = json.loads((self.root / "poles" / "poles.json").read_text())
        self.row = {p: j for j, p in enumerate(self.poles)}
        self.positions = np.load(self.root / "poles" / "positions.npz")["tok"]
        self.a = np.load(self.root / "poles" / "exemplar_a.npz")["a"]
        self._stats, self._topk = {}, {}

    def stats(self, layer):
        if layer not in self._stats:
            st = np.load(self.root / "stats" / f"layer_{layer:02d}.npz")
            self._stats[layer] = {"q999": st["q999"], "max_abs": st["max_abs"]}
        return self._stats[layer]

    def top(self, layer):
        if layer not in self._topk:
            self._topk[layer] = np.load(self.root / "topk" / f"layer_{layer:02d}.npz")["tok"]
        return self._topk[layer]


def presence(x, sign: float, q: float, mx: float) -> dict:
    """The verdict's birth-rule reading on exemplar values x: median of a+ = max(sign * a, 0) in float64."""
    med = float(np.median(np.maximum(sign * x.astype(np.float64), 0.0)))   # a+ (verdict signed_magnitude)
    return {"median_a_plus": med, "q999": q, "max_abs": mx,
            "percentile": med > 0 and med >= q, "raw_ratio": med > 0 and med >= RAW_FRAC * mx}


def cluster_ranks(path) -> dict:
    """{pole: [exemplar_ranks of each sample-0 non-echo cluster]}; declined rows skipped (pole leaves Q2)."""
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["sample"] == 0 and "declined" not in r:
            out[r["pole"]] = [c["exemplar_ranks"] for c in r["clusters"] if not c["echo"]]
    return out


def compare(a_dir, b_dir, poles: list, clusters=None) -> dict:
    A, B = Dir(a_dir), Dir(b_dir)
    for k in ("tokens_sha256", "n_tokens_scored"):
        assert A.manifest[k] == B.manifest[k], f"manifest {k} differs: {A.manifest[k]} vs {B.manifest[k]}"
    assert A.poles == B.poles, "poles/poles.json differs between the two directories"
    assert np.array_equal(A.positions, B.positions), "poles/positions.npz differs: not the same exemplar columns"
    missing = [p for p in poles if p not in A.row]
    assert not missing, f"{len(missing)} poles not in the harvest's pole list, e.g. {missing[:3]}"

    ex_rel, sign_diff, n_entries = [], 0, 0
    q_rel, m_rel, overlaps, low_overlap = [], [], [], []
    flips = {"percentile": [], "raw_ratio": []}
    present = {"percentile": [0, 0], "raw_ratio": [0, 0]}
    cflips = {"percentile": [], "raw_ratio": []}
    n_clusters, cluster_poles = 0, 0
    for p in poles:
        lyr, n, s = parse_pole(p)
        j = A.row[p]
        a, b = A.a[j], B.a[j]
        fa, fb = ~np.isnan(a), ~np.isnan(b)
        assert np.array_equal(fa, fb), f"{p}: exemplar fill pattern differs"
        assert fa.any(), f"{p}: no filled exemplar"
        ex_rel.append(rel_diff(a[fa], b[fb]))
        sign_diff += int((np.sign(a[fa]) != np.sign(b[fb])).sum())
        n_entries += int(fa.sum())
        sign = 1.0 if s == 0 else -1.0
        thr = {name: (float(d.stats(lyr)["q999"][n, s]), float(d.stats(lyr)["max_abs"][n, s])) for name, d in
               (("A", A), ("B", B))}
        reading = {"A": presence(a[fa], sign, *thr["A"]), "B": presence(b[fb], sign, *thr["B"])}
        if clusters is not None and p in clusters:
            cluster_poles += 1
            for ci, ranks in enumerate(clusters[p]):
                ranks = np.asarray(ranks)
                ca, cb = a[ranks], b[ranks]
                keep = ~np.isnan(ca)
                assert keep.any(), f"{p} cluster {ci}: no filled exemplar"
                cr = {"A": presence(ca[keep], sign, *thr["A"]), "B": presence(cb[keep], sign, *thr["B"])}
                n_clusters += 1
                for rule in cflips:
                    if cr["A"][rule] != cr["B"][rule]:
                        cflips[rule].append({"pole": p, "cluster_index": ci, "A": cr["A"], "B": cr["B"]})
        for rule in flips:
            present[rule][0] += reading["A"][rule]
            present[rule][1] += reading["B"][rule]
            if reading["A"][rule] != reading["B"][rule]:
                flips[rule].append({"pole": p, **{d: reading[d] for d in ("A", "B")}})
        q_rel.append(rel_diff(reading["A"]["q999"], reading["B"]["q999"]))
        m_rel.append(rel_diff(reading["A"]["max_abs"], reading["B"]["max_abs"]))
        ta, tb = A.top(lyr)[n, s, :TOP], B.top(lyr)[n, s, :TOP]
        ov = len(set(ta[ta >= 0].tolist()) & set(tb[tb >= 0].tolist()))
        overlaps.append(ov)
        if ov < OVERLAP_FLOOR:
            low_overlap.append({"pole": p, "overlap": ov})

    P = len(poles)
    return {
        "a": str(a_dir), "b": str(b_dir), "n_poles": P, "unit": "pole (proxy; the verdict reads presence per cluster)",
        "devices": {"A": A.manifest.get("device_name"), "B": B.manifest.get("device_name")},
        "checks": {"positions_identical": True, "tokens_sha256": A.manifest["tokens_sha256"],
                   "n_tokens_scored": A.manifest["n_tokens_scored"], "branch": [A.manifest.get("branch"),
                                                                                 B.manifest.get("branch")]},
        "definitions": {"rel_diff": "|x_A - x_B| / max(|x_A|, |x_B|), 0 when both are 0",
                        "a_plus": "a+ = max(s*a, 0), s = +1 for the + pole and -1 for the - pole; median in float64",
                        "percentile": "median over filled exemplars of a+ = max(s*a, 0), and median > 0, and "
                                      "median >= stats q999[n, s] (verdict 6d26acff, copied)",
                        "raw_ratio": "median over filled exemplars of a+ = max(s*a, 0), and median > 0, and "
                                     f"median >= {RAW_FRAC} * stats max_abs[n, s] (verdict 6d26acff, copied)",
                        "topk_overlap": f"|top-{TOP} tok of A ∩ top-{TOP} tok of B| per pole"},
        "exemplar_a": {"rel_diff": summary(np.concatenate(ex_rel)), "n_entries": n_entries,
                       "sign_differs_share": sign_diff / n_entries},
        "presence": {rule: {"present_A": present[rule][0], "present_B": present[rule][1],
                            "n_flipped": len(flips[rule]), "flip_share": len(flips[rule]) / P,
                            "flipped": [f["pole"] for f in flips[rule]], "detail": flips[rule]}
                     for rule in flips},
        "stats": {"q999_rel_diff": summary(q_rel), "max_abs_rel_diff": summary(m_rel)},
        "topk_top20": {"overlap_median": float(np.median(overlaps)),
                       f"share_below_{OVERLAP_FLOOR}": len(low_overlap) / P, "below": low_overlap},
        "clusters": None if clusters is None else {
            "unit": "cluster (sample-0, non-echo; cluster_index = position in that list)",
            "n_poles_with_clusters": cluster_poles, "n_clusters": n_clusters,
            **{rule: {"n_flipped": len(cflips[rule]), "flip_share": len(cflips[rule]) / max(n_clusters, 1),
                      "flipped": [[f["pole"], f["cluster_index"]] for f in cflips[rule]], "detail": cflips[rule]}
               for rule in cflips}},
    }


def print_summary(r: dict):
    e, st, t = r["exemplar_a"], r["stats"], r["topk_top20"]
    fmt = lambda s: f"median {s['median']:.4f}  p90 {s['p90']:.4f}  p99 {s['p99']:.4f}  max {s['max']:.4f}"
    print(f"A {r['devices']['A']}  vs  B {r['devices']['B']}   {r['n_poles']} poles, branch {r['checks']['branch']}")
    print(f"checks: positions identical, tokens_sha256 {r['checks']['tokens_sha256'][:12]}, "
          f"n_tokens_scored {r['checks']['n_tokens_scored']}")
    print(f"exemplar_a rel diff over {e['n_entries']} entries: {fmt(e['rel_diff'])}; sign differs "
          f"{e['sign_differs_share']:.4%}")
    for rule, v in r["presence"].items():
        print(f"presence per pole (proxy) {rule:10s}: present A {v['present_A']}, B {v['present_B']}; flipped {v['n_flipped']} "
              f"({v['flip_share']:.2%}) {v['flipped'][:12]}{' ...' if v['n_flipped'] > 12 else ''}")
    print(f"stats q999 rel diff:    {fmt(st['q999_rel_diff'])}")
    print(f"stats max_abs rel diff: {fmt(st['max_abs_rel_diff'])}")
    print(f"top-20 overlap: median {t['overlap_median']:.1f}; below {OVERLAP_FLOOR}/20 in "
          f"{t[f'share_below_{OVERLAP_FLOOR}']:.2%} of poles")
    c = r["clusters"]
    if c is not None:
        for rule in ("percentile", "raw_ratio"):
            print(f"per cluster {rule:10s}: {c[rule]['n_flipped']} of {c['n_clusters']} clusters flipped "
                  f"({c[rule]['flip_share']:.2%}) {c[rule]['flipped'][:8]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="harvest dir on device A")
    ap.add_argument("--b", required=True, help="harvest dir on device B, same checkpoint and step")
    ap.add_argument("--poles", required=True, help="JSON list of pole ids to compare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clusters", default=None, help="<explain>/explain/clusters.jsonl: also read presence per cluster")
    args = ap.parse_args(argv)
    poles = json.loads(pathlib.Path(args.poles).read_text())
    report = compare(args.a, args.b, poles, None if args.clusters is None else cluster_ranks(args.clusters))
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print_summary(report)
    return report


if __name__ == "__main__":
    main()
