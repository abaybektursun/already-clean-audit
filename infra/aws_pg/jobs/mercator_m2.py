"""M-2 — ontogeny replication across sizes and families (queue item 2). Wave 1.

W0-9 measured, on ONE suite at ONE size, that the consensus bottom-32 subspace crystallizes at an
identifiable training step. A finding from one training run is a fact about that run.

    REFUTER of "forms late" (pre-logged): wild variance across families — if the crystallization
    fraction scatters across suites and sizes with no common structure, the ontogeny claim is about
    OLMo-2-1B and not about training.

FOUR SUITES, TWO FAMILIES, TWO SIZES EACH. Small on purpose: four cheap suites answer the
replication question better than one expensive one, and the comparison this arm needs is BETWEEN
suites rather than inside any of them.

THE PYTHIA BLOCKER, AND WHY IT IS NOT ONE. `darwin_census.resolve_key` locates a decoder matrix by
the suffix `layers.N.self_attn.T.weight`, which OLMo uses and GPT-NeoX does not: Pythia FUSES q, k
and v into a single `query_key_value` tensor, and W0-9 chose OLMo for exactly that reason. But
every bank in this program is INPUT-side, and the three projections share their input axis, so

    dW_fused^T dW_fused  =  dW_q^T dW_q + dW_k^T dW_k + dW_v^T dW_v

exactly. The fused tensor's input Gram IS the sum of the three unfused input Grams, so no
un-fusing convention is needed and none is invented. The cell is named `qkv` rather than `q_proj`
and every row says so, because a fused cell is a sum of three cells and must not be compared to a
single one without that being visible.

ONE ESTIMAND, FOUR SUITES. The windowed consensus Gram, its normalisation, the bottom-32
extraction and the crystallization rule are `darwin_w0_ext`'s, imported — so a difference between
suites is a difference between suites.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-2"
# (repo, family, key template, layer count to sample). The template's `{L}` is the layer index;
# it is written out per family so the key convention is auditable rather than inferred.
# ORDERED CHEAPEST FIRST, deliberately: a suite is only worth its download if the ones before it
# have already produced a fraction, and if the run is cut short the loss is the most expensive
# suite rather than the whole replication.
SUITES = (
    ("EleutherAI/pythia-410m", "pythia",
     "gpt_neox.layers.{L}.attention.query_key_value.weight", "qkv"),
    ("EleutherAI/pythia-1b", "pythia",
     "gpt_neox.layers.{L}.attention.query_key_value.weight", "qkv"),
    ("allenai/OLMo-2-0425-1B", "olmo2", "model.layers.{L}.self_attn.q_proj.weight", "q_proj"),
    ("allenai/OLMo-2-1124-7B", "olmo2", "model.layers.{L}.self_attn.q_proj.weight", "q_proj"),
)
STEP_RE = re.compile(r"step(\d+)")
CRYSTALLIZE_COS2 = 0.9        # W0-9's frozen threshold, reused so the two runs mean the same thing
WINDOW = 3


def load_matrices(repo: str, revision: str, template: str, layers) -> dict:
    """{layer: fp32 tensor} straight from the safetensors shards at one revision.

    Reads shards rather than instantiating the model, for `darwin_census.load_decoder_matrices`'
    reason: a checkpoint whose vocabulary differs from the config aborts `from_pretrained` even
    though every decoder matrix is well-formed. Fails loud on a missing key.
    """
    import pathlib as _pl
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    d = snapshot_download(repo, revision=revision,
                          allow_patterns=["*.safetensors", "*.safetensors.index.json"])
    shards = sorted(_pl.Path(d).glob("*.safetensors"))
    assert shards, f"{repo}@{revision}: no safetensors shards under {d}"
    index = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                index[k] = shard
    out = {}
    for L in layers:
        key = template.format(L=L)
        assert key in index, (
            f"{repo}@{revision}: key {key!r} absent. Present sample: "
            f"{sorted(k for k in index if f'layers.{L}.' in k)[:6]}")
        with safe_open(str(index[key]), framework="pt") as f:
            out[L] = f.get_tensor(key).to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="M-2 — ontogeny replication across suites")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="", help="comma list of repo ids; default is all four")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3, help="evenly spaced layers per suite")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in a.suites.split(",") if s.strip()]
    suites = [s for s in SUITES if not wanted or s[0] in wanted]
    assert suites, f"no suite matched {wanted}"
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    from infra.pg import darwin_w0 as DW
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.invariant_core_func import _bottom_eigvecs
    from infra.aws_pg.jobs.darwin_w0_ext import checkpoint_revisions, pick_frames

    api = HfApi()
    results = {}
    for repo, family, template, cell_name in suites:
        revs = checkpoint_revisions(api, repo)
        if len(revs) < a.window + 1:
            results[repo] = {"family": family, "outcome": M.COVERAGE_IMPOSSIBLE,
                             "n_step_revisions": len(revs),
                             "why": f"{len(revs)} step revisions, need >= {a.window + 1}"}
            at.add("ontogeny_suite_revisions", value=float(len(revs)), model_id=repo,
                   flags={**results[repo], "outcome": M.COVERAGE_IMPOSSIBLE})
            print(f"[M-2] {repo}: {len(revs)} revisions — suite-absent", flush=True)
            continue
        frames = pick_frames(revs, a.n_frames)
        cfg = AutoConfig.from_pretrained(repo, revision=frames[-1][1])
        n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
        assert n_layers > 0, f"{repo}: cannot read a layer count from its config"
        layers = sorted({round(i * (n_layers - 1) / (a.n_layers - 1))
                         for i in range(a.n_layers)}) if a.n_layers > 1 else [n_layers // 2]
        print(f"[M-2] {repo} ({family}): {len(revs)} revisions, {len(frames)} frames "
              f"{[s for s, _ in frames]}, layers {layers} of {n_layers}", flush=True)

        # Consecutive-checkpoint delta terms, normalised exactly as `census_grams` normalises a
        # fine-tune cloud, so a window sum here is the same estimand as a census Gram elsewhere.
        prev, terms = None, []
        for i, (step, branch) in enumerate(frames):
            mats = load_matrices(repo, branch, template, layers)
            if prev is not None:
                term = {}
                for L in layers:
                    A = (mats[L].to(torch.float64) - prev[1][L].to(torch.float64)).numpy()
                    fro2 = float((A ** 2).sum())
                    assert fro2 > 0, f"{repo}: zero delta at layer {L} between {prev[0]} and {step}"
                    term[L] = (A.T @ A) / fro2
                terms.append({"from_step": prev[0], "to_step": step, "term": term})
            prev = (step, mats)
            gc.collect()
            _delete_cache(repo)
            print(f"    frame {i+1}/{len(frames)} step {step}", flush=True)
        del prev
        gc.collect()

        wins = []
        for s in range(0, len(terms) - a.window + 1):
            chunk = terms[s:s + a.window]
            banks = {}
            for L in layers:
                G = sum(c["term"][L] for c in chunk) / len(chunk)
                U, _w = _bottom_eigvecs(G, H.BANK_K)
                banks[L] = U
            wins.append({"lo": chunk[0]["from_step"], "hi": chunk[-1]["to_step"], "banks": banks})
        final = wins[-1]
        curve = {L: [] for L in layers}
        for w in wins:
            mid = (w["lo"] + w["hi"]) / 2.0
            is_reference = w is final
            for L in layers:
                agree = DW.subspace_agreement(w["banks"][L], final["banks"][L])["mean_cos2"]
                # THE REFERENCE WINDOW IS NOT A MEASUREMENT: agreement is computed against it, so
                # its own value is 1.0 by definition and it clears any threshold. Leaving it in the
                # curve would make "the threshold was reached" true for every suite regardless of
                # what happened during training (defect M-D26).
                if not is_reference:
                    curve[L].append((mid, float(agree)))
                at.add("ontogeny_subspace_agreement", value=float(agree), layer=L, dose=float(mid),
                       bank_id="bottom32_window", incumbent_id="final_window", model_id=repo,
                       model_sha=cfg._commit_hash if hasattr(cfg, "_commit_hash") else None,
                       flags={"family": family, "cell": cell_name, "lo_step": w["lo"],
                              "hi_step": w["hi"], "window": a.window, "n_layers": n_layers,
                              "fused_qkv": cell_name == "qkv",
                              "is_reference_window": is_reference,
                              "in_threshold_curve": not is_reference,
                              "note": ("a fused qkv cell's input Gram is exactly the sum of the "
                                       "three unfused input Grams; it is not one projection")})
        # CRYSTALLIZATION FRACTION: where in the suite's OWN training the threshold is first met,
        # as a fraction of its final step. Suites differ by orders of magnitude in total steps, so
        # an absolute step number is not comparable across them and a fraction is.
        last_step = float(frames[-1][0])
        per_layer = {}
        for L in layers:
            pts = curve[L]
            hit = [s for s, v in pts if v >= CRYSTALLIZE_COS2]
            frac = (min(hit) / last_step) if hit else None
            per_layer[L] = {"crystallization_step": (min(hit) if hit else None),
                            "crystallization_fraction": frac,
                            "final_agreement": pts[-1][1], "curve": pts}
            at.add("crystallization_fraction",
                   value=float(frac) if frac is not None else 1.0, layer=L, model_id=repo,
                   flags={"family": family, "cell": cell_name, "threshold": CRYSTALLIZE_COS2,
                          "reached": frac is not None, "last_step": last_step,
                          "note": ("value 1.0 with reached=False means the threshold was never "
                                   "met inside the swept range, not that it was met at the end")})
        fr = [v["crystallization_fraction"] for v in per_layer.values()
              if v["crystallization_fraction"] is not None]
        results[repo] = {"family": family, "cell": cell_name, "outcome": None,
                         "n_step_revisions": len(revs), "n_layers": n_layers,
                         "layers": layers, "frames": [s for s, _ in frames],
                         "last_step": last_step, "per_layer": per_layer,
                         "mean_crystallization_fraction": (float(np.mean(fr)) if fr else None),
                         "n_layers_reaching_threshold": len(fr)}
        print(f"[M-2] {repo}: crystallization fraction "
              f"{results[repo]['mean_crystallization_fraction']} "
              f"({len(fr)}/{len(layers)} layers reached {CRYSTALLIZE_COS2})", flush=True)
        M.s3_sync_incremental(out)

    # ---- the refuter --------------------------------------------------------------------------
    got = {k: v for k, v in results.items() if v.get("outcome") is None
           and v.get("mean_crystallization_fraction") is not None}
    fams = sorted({v["family"] for v in got.values()})
    fracs = {k: v["mean_crystallization_fraction"] for k, v in got.items()}
    enough = bool(len(got) >= 2 and len(fams) >= 2)
    gates.record("m2_replication_is_possible", enough, n_suites=len(got), families=fams,
                 fractions=fracs, outcome=None if enough else M.COVERAGE_IMPOSSIBLE,
                 note=("the refuter is about variance ACROSS families, so it needs at least two "
                       "families to have produced a fraction at all"))
    spread = (float(max(fracs.values()) - min(fracs.values())) if len(fracs) >= 2 else None)
    # "Wild variance" is given a number rather than a word: the refuter fires when the suites'
    # crystallization fractions span more than half of training between them.
    WILD = 0.5
    wild = bool(spread is not None and spread > WILD)
    gates.record("m2_wild_variance_across_families", wild, spread=spread, threshold=WILD,
                 fractions=fracs, families=fams,
                 note=("this gate PASSING is the refuter firing: the ontogeny claim would be "
                       "about one suite rather than about training"))
    at.add("crystallization_fraction_spread", value=float(spread) if spread is not None else 0.0,
           n=len(fracs), flags={"fractions": fracs, "families": fams, "threshold": WILD,
                                "refuter_fires": wild})

    info = at.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "suites": [s[0] for s in suites], "window": a.window,
                  "n_frames": a.n_frames, "threshold": CRYSTALLIZE_COS2,
                  "results": results, "fractions": fracs, "families": fams,
                  "spread": spread, "wild_variance": wild,
                  "replication_possible": enough})
    print("M2_OK " + json.dumps({"rows": info["n_rows"], "fractions": fracs,
                                 "spread": spread, "wild": wild}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
