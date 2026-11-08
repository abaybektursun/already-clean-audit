"""GUTENBERG GB-2 — reader gain at 14B (CPU range-read), behind the g2b reproduction gate.

Spec §3 GB-2 + AM-0.5.4: the amundsen_reader_gain.json generator was never committed, so this
is NEW code that must FIRST reproduce the banked g2b table (deterministic rows to rel<=1e-3;
fresh haar within +-10% of the banked haar) before any 14B number is written. Then the same
instrument runs on Qwen2.5-14B (SHACKLETON census substrate): gains of the census core32/anti32,
the v1_ablation-seeded haar banks (rng(1337) sequence reproduced so the gains belong to the
ACTUAL ablated banks), base-W sigma banks, topenergy coordinate banks, and (optionally,
--extra-banks-s3, after GB-3 lands) the banked energymatched/specmatched vectors. GB-4 consumes
the q_proj table. Weight-only convention (biases ignored), fp64, ||W U||_F^2 / k.

    pg run infra.aws_pg.jobs.gutenberg_gb2 --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --s3-prefix results/gutenberg_gb2 --budget-min 180
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G
from infra.pg.humboldt_core import hf_setup

BANKED_GAIN_S3 = "s3://pg-lineage-769063704696/results/gutenberg_inputs/amundsen_reader_gain.json"
CENSUS_G2B = "s3://pg-lineage-769063704696/results/shackleton_census_g2b/out/banks/"
CENSUS_14B = "s3://pg-lineage-769063704696/results/shackleton_census14b_banks/out/banks/"
TOPENERGY_14B = "s3://pg-lineage-769063704696/results/shackleton_r_v1_14b/topenergy_14b.npz"
SITES = ("q_proj", "k_proj", "v_proj", "gate_proj")


def _fetch(s3_uri, dst, recursive=False):
    cmd = ["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1", "--only-show-errors"]
    if recursive:
        cmd.insert(3, "--recursive")
    subprocess.run(cmd, check=True)
    return dst


class HFRangeReader:
    """fetch(start, stop) over HF-hosted safetensors shards, resolving tensor -> shard via the
    index file. One requests.Session; token from the environment (hf_setup ran first)."""

    def __init__(self, repo):
        import os
        import requests
        from huggingface_hub import hf_hub_download, hf_hub_url
        self.repo = repo
        self.sess = requests.Session()
        tok = os.environ.get("HF_TOKEN")
        if tok:
            self.sess.headers["Authorization"] = f"Bearer {tok}"
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        self.weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
        self._urls = {f: hf_hub_url(repo, f) for f in set(self.weight_map.values())}
        self._hdrs = {}

    def _fetcher(self, fname):
        url = self._urls[fname]

        def fetch(start, stop):
            r = self.sess.get(url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=120)
            assert r.status_code in (200, 206), (r.status_code, url)
            data = r.content
            assert len(data) == stop - start, (len(data), stop - start)
            return data
        return fetch

    def tensor(self, name):
        fname = self.weight_map[name]
        fetch = self._fetcher(fname)
        if fname not in self._hdrs:
            self._hdrs[fname] = G.safetensors_header(fetch)
        hdr, d0 = self._hdrs[fname]
        return G.range_read_tensor(fetch, hdr, d0, name)


def gain_table(reader, prefix, n_layers, banks_dir, seed, extra_banks=None,
               tope_npz=None, haar_ablation_dim=None, label=None):
    """Per-layer per-site gain rows. banks_dir holds q_proj__L{L}__{core32,anti32}.npy.
    haar_ablation_dim: if set, reproduce v1_ablation's haar sequence (rng(1337), 3 draws of
    (d, 32) per layer in layer order) and record their q_proj gains as haar_ablation_s{s}."""
    rows = []
    rng_fresh = np.random.default_rng(seed)
    rng_abl = np.random.default_rng(1337) if haar_ablation_dim else None
    for L in range(n_layers):
        core = np.linalg.qr(np.asarray(
            np.load(banks_dir / f"q_proj__L{L}__core32.npy"), dtype=np.float64))[0]
        anti = np.linalg.qr(np.asarray(
            np.load(banks_dir / f"q_proj__L{L}__anti32.npy"), dtype=np.float64))[0]
        n = core.shape[0]
        abl_haars = []
        if rng_abl is not None:
            assert haar_ablation_dim == n, (haar_ablation_dim, n)
            for _ in range(3):
                abl_haars.append(np.linalg.qr(rng_abl.standard_normal((n, 32)))[0])
        row = {"layer": L}
        for site in SITES:
            W = reader.tensor(f"{prefix}.layers.{L}.self_attn.{site}.weight"
                              if site != "gate_proj"
                              else f"{prefix}.layers.{L}.mlp.gate_proj.weight")
            assert W.shape[1] == n, (site, W.shape, n)
            haars = [G.haar_frame(n, 32, rng_fresh) for _ in range(3)]
            row[site] = {
                "core32": G.reader_gain(W, core), "anti32": G.reader_gain(W, anti),
                "haar32": float(np.median([G.reader_gain(W, h) for h in haars])),
                "allmean": G.allmean_gain(W),
            }
            if site == "q_proj":
                _, _, Vh = np.linalg.svd(W, full_matrices=False)
                Vs = Vh.T
                r = Vs.shape[1]
                row[site].update({
                    "bottomsigma32": G.reader_gain(W, Vs[:, r - 32:]),
                    "topsigma32": G.reader_gain(W, Vs[:, :32]),
                    "overlap_core_bottom32": G.overlap_with_singular_subspace(
                        core, Vs[:, ::-1], 32),
                    "overlap_core_bottom256": G.overlap_with_singular_subspace(
                        core, Vs[:, ::-1], 256),
                    "overlap_core_top32": G.overlap_with_singular_subspace(core, Vs, 32),
                })
                for s, h in enumerate(abl_haars):
                    row[site][f"haar_ablation_s{s}"] = G.reader_gain(W, h)
                if tope_npz is not None:
                    for k in (32, 128):
                        idx = tope_npz[f"L{L:02d}_top{k}"]
                        row[site][f"topenergy{k}"] = G.reader_gain(
                            W, G.coordinate_bank(n, idx))
                if extra_banks:
                    for bid, arr_by_layer in extra_banks.items():
                        row[site][bid] = G.reader_gain(W, arr_by_layer[L])
        rows.append(row)
        # label, not prefix: both models are loaded with prefix "model", so the prefix
        # cannot distinguish the g2b gate pass from the 14B pass in the log.
        print(f"[gb2] {label or prefix} L{L}: q core={row['q_proj']['core32']:.4g} "
              f"haar={row['q_proj']['haar32']:.4g}", flush=True)
    return rows


def g2b_reproduction_gate(rows, banked):
    """AM-0.5.4: deterministic per-layer q_proj rows to rel<=1e-3; fresh haar within +-10%
    of the banked per-layer haar; per-site medians (core/anti/all) to rel<=1e-3."""
    def relok(a, b, tol=1e-3):
        return abs(a - b) <= tol * max(abs(b), 1e-12)

    checked = 0
    for got, exp in zip(rows, banked["per_layer"]):
        assert got["layer"] == exp["layer"]
        q = got["q_proj"]
        for kk, bk in (("core32", "gain_core32"), ("anti32", "gain_anti32"),
                       ("allmean", "gain_allmean"), ("bottomsigma32", "gain_bottomsigma32"),
                       ("topsigma32", "gain_topsigma32"),
                       ("overlap_core_bottom32", "overlap_core_bottom32"),
                       ("overlap_core_bottom256", "overlap_core_bottom256"),
                       ("overlap_core_top32", "overlap_core_top32")):
            assert relok(q[kk], exp[bk]), \
                f"g2b gate FAILED L{got['layer']} {kk}: {q[kk]} vs banked {exp[bk]}"
            checked += 1
        assert 0.9 * exp["gain_haar32"] <= q["haar32"] <= 1.1 * exp["gain_haar32"], \
            f"g2b haar gate FAILED L{got['layer']}: {q['haar32']} vs {exp['gain_haar32']}"
    for site, key in (("k_proj", "k_proj"), ("v_proj", "v_proj"), ("gate_proj", "gate_proj")):
        for stat in ("core", "anti", "all"):
            got_med = float(np.median([r[site]["core32" if stat == "core" else
                                               "anti32" if stat == "anti" else "allmean"]
                                       for r in rows]))
            exp_med = banked["per_site_median"][key][stat]
            assert relok(got_med, exp_med), \
                f"g2b gate FAILED {site} median {stat}: {got_med} vs {exp_med}"
            checked += 1
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--skip-14b", action="store_true", help="gate-only smoke")
    ap.add_argument("--extra-banks-s3", default="",
                    help="optional GB-3 banked ematch/spec vectors npz (14B q_proj space)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb2")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()

    banked = json.loads(_fetch(BANKED_GAIN_S3, wk / "banked_gain.json").read_text())
    _fetch(CENSUS_G2B, wk / "banks_g2b", recursive=True)
    g2b_rows = gain_table(HFRangeReader("google/gemma-2b"), "model", 18,
                          wk / "banks_g2b", seed=a.seed, label="gemma-2b")
    n_checked = g2b_reproduction_gate(g2b_rows, banked)
    print(f"[gb2] g2b reproduction gate PASSED ({n_checked} checks)", flush=True)

    result = {"g2b_gate": {"passed": True, "n_checks": n_checked,
                           "banked_source": BANKED_GAIN_S3},
              "g2b_rows": g2b_rows, "seed": a.seed,
              "convention": "gain = ||W U||_F^2 / k, fp64, weight-only; haar = median of 3",
              "code": G.provenance(), "utc": C.utc_now()}

    if not a.skip_14b:
        _fetch(CENSUS_14B, wk / "banks_14b", recursive=True)
        tope = np.load(_fetch(TOPENERGY_14B, wk / "topenergy_14b.npz"))
        extra = None
        if a.extra_banks_s3:
            xnpz = np.load(_fetch(a.extra_banks_s3, wk / "extra_banks.npz"))
            extra = {}
            for key in xnpz.files:              # "{bid}/{L}" (+ gram_eigvals_desc/{L}, skipped)
                bid, L = key.rsplit("/", 1)
                if xnpz[key].ndim != 2:
                    continue
                extra.setdefault(bid, {})[int(L)] = np.asarray(xnpz[key], dtype=np.float64)
        rows14 = gain_table(HFRangeReader("Qwen/Qwen2.5-14B"), "model", 48,
                            wk / "banks_14b", seed=a.seed + 1, extra_banks=extra,
                            tope_npz=tope, haar_ablation_dim=5120, label="Qwen2.5-14B")
        med = {site: {stat: float(np.median([r[site][stat] for r in rows14]))
                      for stat in ("core32", "anti32", "haar32", "allmean")}
               for site in SITES}
        med["q_proj_core_over_haar"] = med["q_proj"]["core32"] / med["q_proj"]["haar32"]
        result.update({"qwen14b_rows": rows14, "qwen14b_median": med,
                       "topenergy_source": TOPENERGY_14B,
                       "banks_14b": CENSUS_14B,
                       "extra_banks": a.extra_banks_s3 or None})
        print(f"[gb2] 14B q_proj median core/haar = {med['q_proj_core_over_haar']:.4f}",
              flush=True)

    (out / "gutenberg_gb2_reader_gain.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float))
    print("GUTENBERG_GB2_OK", flush=True)


if __name__ == "__main__":
    main()
