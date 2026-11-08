"""DARWIN W0-1b — the massive-activation coordinate lists (§4/W0-1; journal amendment A2). GPU.

Why this arm exists. §4/W0-1 freezes the criterion "coordinates with median |h_i| > 5x the global
median, computed per layer per model from existing B1 series", and §8 makes W0-1 the FIRST arm
because it decides whether the core is the already-known massive-activation/outlier-dimension
object. But HUMBOLDT's B1 recorded, per token, only e_U = ‖UUᵀh‖²/‖h‖² — a per-BANK scalar. No
per-COORDINATE activation magnitude was written, so the list is not derivable from
`humboldt/humboldt_b1/` at any cost. This arm harvests the missing object.

Token-set identity, not similarity. The windows are replayed from B1's own frozen
`meta__<model>__<corpus>.npz` token ids, grouped by `window_idx`, and fed to the model as
input_ids. So the tokens these medians are computed over are B1's tokens — the same objects the
e_U series was computed over — rather than a re-tokenisation that would merely resemble them.

Sampling. An exact per-coordinate median needs every token resident: 50k tokens x 2048 dims x 18
layers x 3 sites is ~22 GB per model. A seeded reservoir of MAX_TOKENS tokens per (layer, site)
is used instead; `n_tokens_sampled` and the reservoir seed ride in every row's flags, and the
sampling is deterministic given B1's token order.

  pg run infra.aws_pg.jobs.darwin_w0_outliers --args "--arm W0-1b" --type g6e.2xlarge ...
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H

MAX_TOKENS = 4096                 # reservoir size per (layer, site)
B1_S3 = f"s3://{D.BUCKET}/humboldt/humboldt_b1"
# The text corpora. VQAv2 is excluded deliberately: its windows carry 256 image tokens whose
# activation statistics are a different population, and mixing them into a per-coordinate median
# would make the "global median" of the frozen rule a blend of two regimes.
CORPORA = ("wikitext", "code", "parallel", "repetition", "longdoc")
CRITERION_CORPUS = "wikitext"     # the corpus the frozen 5x-median rule is applied to


class SiteSample:
    """forward_pre_hooks that reservoir-sample raw residual rows at each (layer, site).

    Stores |h| rows, not h: the criterion is on magnitude, and keeping the sign would double the
    memory for a statistic nothing reads.
    """

    def __init__(self, layers, device, seed, sites=H.SITES_PHASE_B, max_tokens=MAX_TOKENS):
        import numpy as np
        self.layers, self.sites = layers, tuple(sites)
        self.max_tokens = int(max_tokens)
        self.rng = np.random.default_rng(seed)
        self.buf = {}          # (L, site) -> (m, n) fp32 reservoir
        self.seen = {}         # (L, site) -> tokens seen
        self._handles = []

    def _mk(self, L, site):
        import numpy as np

        def pre_hook(_m, args):
            x = args[0]
            assert x.shape[0] == 1, f"SiteSample requires batch 1, got {x.shape[0]}"
            rows = x.reshape(-1, x.shape[-1]).abs().float().detach().cpu().numpy()
            k = (L, site)
            cur, seen = self.buf.get(k), self.seen.get(k, 0)
            if cur is None:
                cur = np.empty((0, rows.shape[1]), dtype=np.float32)
            take = min(self.max_tokens - cur.shape[0], rows.shape[0])
            if take > 0:
                cur = np.concatenate([cur, rows[:take]], axis=0)
            # Reservoir (Vitter R) for the remainder, so late windows are represented too.
            for i in range(take, rows.shape[0]):
                j = int(self.rng.integers(0, seen + i + 1))
                if j < self.max_tokens:
                    cur[j] = rows[i]
            self.buf[k] = cur
            self.seen[k] = seen + rows.shape[0]
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            for site in self.sites:
                mod = getattr(layer.self_attn if site in ("q_proj_in", "o_proj_in") else layer.mlp,
                              A.SITE_MODULE[site])
                self._handles.append(mod.register_forward_pre_hook(self._mk(L, site)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def fetch_b1_meta(dest, model_id, corpus):
    """Pull one B1 meta npz. Absence is fatal: the arm's whole point is token-set identity."""
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    name = f"meta__{model_id}__{corpus}.npz"
    p = d / name
    if not p.exists():
        subprocess.run(["aws", "s3", "cp", f"{B1_S3}/{name}", str(p), "--region", "us-east-1"],
                       check=True)
    return p


def windows_from_meta(path):
    """Reconstruct B1's token windows: [(window_idx, [token_id, ...]), ...] in recorded order."""
    import numpy as np
    z = np.load(path)
    tid, widx = z["token_id"], z["window_idx"]
    assert tid.shape == widx.shape, f"{path.name}: token_id/window_idx shape mismatch"
    out = []
    for w in np.unique(widx):
        sel = widx == w
        out.append((int(w), tid[sel].astype(np.int64).tolist()))
    assert out, f"{path.name}: no windows"
    return out


def harvest_model(model_id, repo, kind, out, atlas, args, cache):
    """Per-coordinate median |h_i| for one model, over every corpus, at the three Phase-B sites."""
    import numpy as np
    import torch

    model, _proc, layers, sha, _mm = A.load_model(model_id, repo, kind, device="cuda")
    n_layers = len(layers)
    per_corpus = {}
    for corpus in args.corpora:
        meta = fetch_b1_meta(cache, model_id, corpus)
        wins = windows_from_meta(meta)
        if args.smoke:
            wins = wins[:2]
        sampler = SiteSample(layers, "cuda", seed=H.cell_seed(args.seed, model_id, corpus))
        with sampler, torch.no_grad():
            for _w, ids in wins:
                t = torch.tensor([ids], dtype=torch.long, device="cuda")
                model(input_ids=t)
        med = {}
        for (L, site), rows in sampler.buf.items():
            assert rows.shape[0] > 0, f"{model_id}/{corpus} L{L} {site}: empty reservoir"
            med[(L, site)] = np.median(rows.astype(np.float64), axis=0)
        per_corpus[corpus] = {"median": med, "n_tokens": dict(sampler.seen),
                              "n_sampled": {k: int(v.shape[0]) for k, v in sampler.buf.items()},
                              "n_windows": len(wins)}
        print(f"[W0-1b] {model_id}/{corpus}: {len(wins)} windows, "
              f"{max(sampler.seen.values())} tokens", flush=True)

        for (L, site), m in med.items():
            sel = DW.threshold_coords(m, DW.OUTLIER_MULT)
            fl = {"site": site, "corpus": corpus, "n_tokens_seen": int(sampler.seen[(L, site)]),
                  "n_tokens_sampled": int(sampler.buf[(L, site)].shape[0]),
                  "reservoir_max": MAX_TOKENS,
                  "seed": H.cell_seed(args.seed, model_id, corpus),
                  "rule": f"median|h_i| > {DW.OUTLIER_MULT}x global median",
                  "is_criterion_corpus": corpus == args.criterion_corpus}
            atlas.add("activation_median_abs",
                      vector_ref=atlas.vector(f"W0-1b__medabs__{model_id}__{corpus}__L{L}__{site}", m),
                      layer=L, model_id=model_id, model_sha=sha, corpus_id=corpus,
                      n=int(m.size), flags=fl)
            atlas.add("outlier_coord_count", value=float(sel["n_selected"]), layer=L,
                      model_id=model_id, model_sha=sha, corpus_id=corpus,
                      flags={**fl, **{k: sel[k] for k in
                                      ("threshold", "median", "fraction_selected")}})
            atlas.add("outlier_coords",
                      vector_ref=atlas.vector(f"W0-1b__outl__{model_id}__{corpus}__L{L}__{site}",
                                              sel["coords"], dtype="int64"),
                      layer=L, model_id=model_id, model_sha=sha, corpus_id=corpus,
                      n=sel["n_selected"], flags=fl)
            atlas.add("activation_participation_ratio", value=H.participation_ratio(m), layer=L,
                      model_id=model_id, model_sha=sha, corpus_id=corpus,
                      flags={**fl, "flat_reference": int(m.size)})
        D.s3_sync(out, args.s3_prefix)

    del model
    torch.cuda.empty_cache()
    return {"model_sha": sha, "n_layers": n_layers,
            "corpora": {c: {"n_windows": v["n_windows"]} for c, v in per_corpus.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W0-1b")
    ap.add_argument("--run-id", default="darwin_w0_1b")
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--criterion-corpus", default=CRITERION_CORPUS)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    a.corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    roster = [r for r in A.ROSTER if not want or r[0] in want]
    assert roster, f"no roster entry matches {want}"
    if a.smoke:
        roster, a.corpora = roster[:1], a.corpora[:1]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        exp.log(a.arm,
                refuter="twin overlap bands indistinguishable",
                direction="a small set of coordinates per layer exceeds 5x the global median "
                          "|h_i|, reproducing the known massive-activation phenomenon",
                utc=utc)
    ref = exp.require(a.arm)

    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "roster": [r[0] for r in roster], "corpora": a.corpora,
                "criterion_corpus": a.criterion_corpus, "sites": list(H.SITES_PHASE_B),
                "max_tokens": MAX_TOKENS, "outlier_mult": DW.OUTLIER_MULT, "seed": a.seed,
                "expected_ref": ref, "started_utc": utc, "b1_source": B1_S3}
    D.write_json(out, "manifest.json", manifest)

    cache = H.cache_dir(out, "b1meta")
    results = {}
    for model_id, repo, kind in roster:
        results[model_id] = harvest_model(model_id, repo, kind, out, atlas, a, cache)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, "w0_1b_summary.json",
                 {"manifest": manifest, "atlas": rep, "results": results,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_W0_1B_OK rows={rep['n_rows']} vectors={rep['n_vectors']} "
          f"models={json.dumps(list(results))}")


if __name__ == "__main__":
    main()
