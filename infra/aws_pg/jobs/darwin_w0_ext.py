"""DARWIN W0-9 / W0-11 / W0-13 — ontogeny, spike retrodiction, the core-relative frame. CPU.

  W0-9   Ontogeny movie (§4/W0-9, curiosity C4). A public checkpoint suite is a MOVIE of a model
         being built. Take consecutive checkpoint deltas, form the consensus Gram over a sliding
         window of them, and watch the bottom-32 subspace: when does it stop moving? That step is
         the crystallization step, and §4/W2-D's curriculum switch point is defined to be it.

  W0-11  Loss-spike retrodiction (§4/W0-11). Exploratory: do core-band statistics move at the
         steps a public pretraining log records a loss spike? The arm's readout validity is the
         LOG, so the log is looked for mechanically and its absence is a recorded outcome with the
         list of places searched — not an unexplained gap in the coverage table.

  W0-13  Core-relative frame (§4/W0-13, product 4.6). Are a feature direction's coordinates more
         stable across fine-tunes when expressed in the core frame than in the raw basis?

W0-13 AND THE READOUT-VALIDITY GATE. §4/W0-13 names "published Gemma Scope feature directions for
gemma-2b". Gemma Scope's SAEs are trained on a specific residual width; DARWIN's banks live in the
gemma-2b decoder's 2048-dim input space. If those two widths differ, Uᵀf is not a defined object
and any number the arm printed would be a coincidence of shapes. So the arm CHECKS — it reads the
SAE repo's own config and compares d_model against the bank dimension — and:
  * widths agree  -> the pre-registered readout runs;
  * widths differ -> COVERAGE-IMPOSSIBLE is recorded with both geometries as evidence, AND a
    clearly-tagged DISCOVERY variant runs the same stability comparison on feature directions that
    DO live in the bank frame (unembedding rows), carrying no confirmatory weight.
This is the same discipline D19/D22/D23 established: the arm decides from the artifact, and a
readout that cannot be valid is recorded as such rather than reinterpreted into one that can.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w0_ext --args "--arm W0-9" --type r7i.2xlarge --no-weights ...
"""
import argparse
import json
import pathlib
import re

from infra.pg import darwin_census as CEN
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H

# §4/W0-9's suite. OLMo rather than Pythia for one mechanical reason, stated so it is checkable:
# `darwin_census.resolve_key` locates a decoder matrix by the suffix `layers.N.self_attn.T.weight`,
# which OLMo's checkpoints use and GPT-NeoX (Pythia) does not — Pythia fuses q/k/v into one
# `query_key_value` tensor, so "the q_proj delta" is not a thing that exists there without a
# reshape convention this program has never validated. §4's pivot allows the suite choice; the
# reason is geometry, not preference.
SUITE = "allenai/OLMo-2-0425-1B"
SUITE_FALLBACKS = ("allenai/OLMo-2-1124-7B", "allenai/OLMo-1B-hf")
STEP_RE = re.compile(r"step(\d+)")
CRYSTALLIZE_COS2 = 0.9       # frozen: first window whose Grassmann overlap (mean cos^2) with
                             # the final window's bottom-32 reaches this is the crystallization
                             # step. cos^2 because that is what darwin_w0.subspace_agreement
                             # reports — reading `mean_cos` off it would have been a KeyError.
WINDOW = 3                   # consecutive deltas per consensus Gram (>=2 makes it a consensus)

# §4/W0-11's mechanical search list: where a public per-step loss series could live for this
# suite. Searched in order; the first hit is used; all misses are recorded.
LOSS_LOG_CANDIDATES = (
    ("repo_file", "train_loss.csv"),
    ("repo_file", "training_loss.csv"),
    ("repo_file", "loss.csv"),
    ("repo_file", "metrics.csv"),
    ("dataset", "allenai/OLMo-2-1124-7B-train-metrics"),
    ("dataset", "allenai/olmo-2-training-logs"),
)

GEMMA_SCOPE = "google/gemma-scope-2b-pt-res"
SPEC_EXPECTATIONS = {
    "W0-9": ("the bottom subspace never stabilizes, or is as stable at step 0 as at the end",
             "the consensus bottom-32 subspace crystallizes at an identifiable training step and "
             "its leverage spikiness rises with it"),
    "W0-11": ("core-band statistics are flat at the recorded spike steps",
              "core-band statistics move at or before the steps a public log records a spike"),
    "W0-13": ("the core frame is no more stable than the raw coordinates",
              "feature coordinates expressed in the core frame are more stable across fine-tunes "
              "than the same features in raw coordinates or in twin frames"),
}


# ---------------------------------------------------------------------------
def checkpoint_revisions(api, repo: str):
    """[(step, branch)] sorted by step — the suite's movie frames, read from its refs.

    Enumerated rather than hardcoded: a hardcoded revision list silently rots when a suite is
    re-uploaded, and the failure mode is a 404 halfway through a paid run.
    """
    refs = api.list_repo_refs(repo)
    out = []
    for b in refs.branches:
        m = STEP_RE.search(b.name)
        if m:
            out.append((int(m.group(1)), b.name))
    out.sort()
    return out


def pick_frames(revs, n_frames: int):
    """`n_frames` checkpoints spread evenly over the suite, first and last always included."""
    assert len(revs) >= 2, f"suite has {len(revs)} step revisions, need >= 2"
    if len(revs) <= n_frames:
        return list(revs)
    idx = [round(i * (len(revs) - 1) / (n_frames - 1)) for i in range(n_frames)]
    return [revs[i] for i in sorted(set(idx))]


def delta_terms(repo, frames, types, layers, delete_cache):
    """[{(t,L): dWᵀdW/‖dW‖²_F}] for each consecutive checkpoint pair, streamed one at a time.

    Each term is the SAME normalized outer product `census_grams` accumulates, so a window sum
    here and a census Gram elsewhere are the same estimand — that is what lets W0-9's subspace be
    compared to the house banks at all.
    """
    import gc
    import numpy as np
    import torch

    keys = [(t, L) for t in types for L in layers]
    prev, prev_step, terms, fro2 = None, None, [], []
    for i, (step, branch) in enumerate(frames):
        print(f"[W0-9] frame {i+1}/{len(frames)}: step {step} ({branch})", flush=True)
        cur = CEN.load_decoder_matrices(repo, branch, types, layers)
        if prev is not None:
            term, f2 = {}, {}
            for k in keys:
                assert tuple(cur[k].shape) == tuple(prev[k].shape), (
                    f"{k}: shape changed across checkpoints {tuple(prev[k].shape)} -> "
                    f"{tuple(cur[k].shape)}")
                A = (cur[k].to(torch.float64) - prev[k].to(torch.float64)).numpy()
                s = float((A ** 2).sum())
                assert s > 0, f"{k}: checkpoint {branch} is bit-identical to its predecessor"
                term[k] = (A.T @ A) / s
                f2[k] = s
                del A
            terms.append({"from_step": prev_step, "to_step": step, "term": term})
            fro2.append({"from_step": prev_step, "to_step": step, "fro2": f2})
        prev, prev_step = cur, step
        gc.collect()
        if delete_cache is not None:
            delete_cache(repo)
    assert terms, "no consecutive pairs"
    return terms, fro2


def window_banks(terms, keys, window: int, k: int):
    """[(centre_step, {(t,L): bottom-k basis})] over sliding windows of `window` deltas."""
    import numpy as np
    from infra.aws_pg.jobs.invariant_core_func import _bottom_eigvecs
    out = []
    for s in range(0, len(terms) - window + 1):
        chunk = terms[s:s + window]
        banks, eigs = {}, {}
        for key in keys:
            G = np.zeros_like(chunk[0]["term"][key])
            for c in chunk:
                G += c["term"][key]
            # `_bottom_eigvecs` returns (basis, eigenvalues) — storing the tuple as if it were the
            # basis is a shape bug that only surfaces at the first matmul, i.e. after the frames
            # have already been downloaded. The eigenvalues are banked too: the bottom-32 energy
            # floor over training is a free curve and the arm has no reason to throw it away.
            U, w = _bottom_eigvecs(G / len(chunk), k)
            banks[key], eigs[key] = U, w
        out.append({"lo_step": chunk[0]["from_step"], "hi_step": chunk[-1]["to_step"],
                    "banks": banks, "eigs": eigs, "n_terms": len(chunk)})
    return out


def run_w0_9(out, atlas, gates, a):
    import numpy as np
    from huggingface_hub import HfApi
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    api = HfApi()
    repo, revs = a.suite, checkpoint_revisions(HfApi(), a.suite)
    for alt in SUITE_FALLBACKS:
        if len(revs) >= 2 * WINDOW:
            break
        print(f"[W0-9] {repo}: {len(revs)} step revisions — trying {alt}", flush=True)
        repo, revs = alt, checkpoint_revisions(api, alt)
    assert len(revs) >= WINDOW + 1, (
        f"no OLMo suite exposes >= {WINDOW+1} step revisions "
        f"(tried {[a.suite] + list(SUITE_FALLBACKS)}) — W0-9 is suite-absent")
    frames = pick_frames(revs, a.n_frames)
    cfg = api.model_info(repo, revision=frames[0][1])
    print(f"[W0-9] {repo}: {len(revs)} step revisions, using {len(frames)} frames "
          f"{[s for s, _ in frames]}", flush=True)

    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    keys = [(t, L) for t in types for L in layers]
    terms, fro2 = delta_terms(repo, frames, types, layers, _delete_cache)
    wins = window_banks(terms, keys, WINDOW if not a.smoke else 2, H.BANK_K)
    final = wins[-1]

    rows, series = 0, {}
    for w in wins:
        mid = (w["lo_step"] + w["hi_step"]) / 2.0
        for key in keys:
            t, L = key
            U = w["banks"][key]
            agree = DW.subspace_agreement(U, final["banks"][key])
            sp = CEN.spikiness(U)
            atlas.add("ontogeny_subspace_agreement", value=agree["mean_cos2"], layer=L, type=t,
                      bank_id="bottom32_window", dose=float(mid), incumbent_id="final_window",
                      model_id=repo, model_sha=cfg.sha,
                      flags={"lo_step": w["lo_step"], "hi_step": w["hi_step"],
                             "n_terms": w["n_terms"], "window": WINDOW,
                             "reference": "the LAST window's bottom-32",
                             "max_angle_rad": agree["max_angle_rad"],
                             "min_angle_rad": agree["min_angle_rad"],
                             "overlap": "mean cos^2 of principal angles (Grassmann overlap)",
                             "suite": repo, "n_step_revisions": len(revs)})
            atlas.add("ontogeny_leverage_spikiness", value=sp["participation_ratio"], layer=L,
                      type=t, bank_id="bottom32_window", dose=float(mid),
                      incumbent_id="haar_flat", model_id=repo, model_sha=cfg.sha,
                      flags={"lo_step": w["lo_step"], "hi_step": w["hi_step"],
                             "flat_reference": sp["flat_reference"],
                             "haar_expected_leverage": sp["haar_expected_leverage"],
                             "max_leverage": sp["max_leverage"], "null": sp["null"],
                             "suite": repo})
            atlas.add("ontogeny_bottom_eigenvalue", value=float(w["eigs"][key].mean()), layer=L,
                      type=t, bank_id="bottom32_window", dose=float(mid),
                      incumbent_id="none", model_id=repo, model_sha=cfg.sha,
                      flags={"lo_step": w["lo_step"], "hi_step": w["hi_step"],
                             "min_eigenvalue": float(w["eigs"][key].min()),
                             "max_eigenvalue": float(w["eigs"][key].max()),
                             "definition": "mean of the 32 smallest eigenvalues of the window's "
                                           "consensus Gram — the consensus-avoided energy floor",
                             "suite": repo})
            series.setdefault(key, []).append((mid, agree["mean_cos2"],
                                              sp["participation_ratio"]))
            rows += 3
        D.s3_sync(out, a.s3_prefix)

    # Crystallization: the first window whose bottom-32 already agrees with the FINAL window's at
    # CRYSTALLIZE_COS. Frozen before the numbers were seen; §4/W2-D consumes it, with 30% as the
    # named fallback if no window reaches the threshold.
    cryst = {}
    for key, pts in series.items():
        hit = [p[0] for p in pts if p[1] >= CRYSTALLIZE_COS2]
        cryst[f"{key[0]}_L{key[1]}"] = float(min(hit)) if hit else None
    reached = [v for v in cryst.values() if v is not None]
    crystallization_step = float(np.median(reached)) if reached else None
    max_step = max(s for s, _ in frames)
    atlas.add("crystallization_step", value=crystallization_step if crystallization_step else -1.0,
              bank_id="bottom32_window", incumbent_id="none", model_id=repo, model_sha=cfg.sha,
              n=len(reached),
              flags={"rule": f"first window with mean_cos2 >= {CRYSTALLIZE_COS2} vs the final "
                             f"window, median over cells", "per_cell": cryst,
                     "n_cells_reached": len(reached), "n_cells": len(series),
                     "max_step": max_step,
                     "as_fraction_of_training": (crystallization_step / max_step)
                                                if crystallization_step else None,
                     "w2d_fallback": "30% of training if no window reaches the threshold",
                     "suite": repo})
    gates.record("W09_ontogeny_measured", True, suite=repo, n_frames=len(frames),
                 n_windows=len(wins), crystallization_step=crystallization_step,
                 n_cells_reached=len(reached), n_cells=len(series))
    return {"suite": repo, "n_step_revisions": len(revs), "frames": [s for s, _ in frames],
            "n_windows": len(wins), "crystallization_step": crystallization_step,
            "crystallization_fraction": (crystallization_step / max_step)
                                        if crystallization_step else None,
            "per_cell": cryst, "n_rows": rows,
            "fro2_by_step": [{"to_step": f["to_step"],
                              "fro2": {f"{k[0]}_L{k[1]}": v for k, v in f["fro2"].items()}}
                             for f in fro2]}


# ---------------------------------------------------------------------------
def find_loss_log(api, repo: str):
    """(kind, name, files) for the first candidate that exists, else (None, None, searched).

    Written as a search with a recorded miss list because §4/W0-11 is exploratory AND its validity
    depends entirely on an artifact this program does not control. "We looked here, here and here"
    is a result; "no rows" is not.
    """
    searched = []
    files = api.list_repo_files(repo)
    for kind, name in LOSS_LOG_CANDIDATES:
        if kind == "repo_file":
            hits = [f for f in files if f.endswith(name)]
            searched.append({"kind": kind, "name": name, "found": bool(hits), "hits": hits[:4]})
            if hits:
                return kind, hits[0], searched
        else:
            exists = api.repo_exists(name, repo_type="dataset")
            searched.append({"kind": kind, "name": name, "found": bool(exists)})
            if exists:
                return kind, name, searched
    return None, None, searched


def run_w0_11(out, atlas, gates, a):
    from huggingface_hub import HfApi
    api = HfApi()
    repo, revs = a.suite, checkpoint_revisions(HfApi(), a.suite)
    for alt in SUITE_FALLBACKS:
        if revs:
            break
        repo, revs = alt, checkpoint_revisions(api, alt)
    kind, name, searched = find_loss_log(api, repo)
    info = api.model_info(repo)
    atlas.add("spike_log_available", value=float(kind is not None), bank_id="none",
              incumbent_id="none", model_id=repo, model_sha=info.sha, n=len(searched),
              flags={"searched": searched, "found_kind": kind, "found_name": name,
                     "n_step_revisions": len(revs),
                     "outcome": "LOG-PRESENT" if kind else "LOGS-UNAVAILABLE",
                     "why_it_matters": "W0-11's readout is the alignment between core-band "
                                       "statistics and RECORDED spike steps; with no recorded "
                                       "steps there is nothing to align to, and a number computed "
                                       "anyway would be a statistic about nothing"})
    if kind is None:
        gates.record("W011_readout_valid", False, outcome="LOGS-UNAVAILABLE", searched=searched,
                     suite=repo,
                     note="recorded as COVERAGE-IMPOSSIBLE per the readout-validity gate; the "
                          "checkpoint half of the arm is W0-9, which ran")
        print(f"[W0-11] LOGS-UNAVAILABLE for {repo}; searched {len(searched)} locations",
              flush=True)
        disc = checkpoint_energy_spikes(out, atlas, a, repo, info.sha)
        return {"outcome": "LOGS-UNAVAILABLE", "suite": repo, "searched": searched,
                "n_step_revisions": len(revs), "discovery_companion": disc}

    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    if kind == "repo_file":
        path = hf_hub_download(repo, name)
        raw = pathlib.Path(path).read_text().strip().splitlines()
        head = raw[0].split(",")
        si = [i for i, h in enumerate(head) if "step" in h.lower()][0]
        li = [i for i, h in enumerate(head) if "loss" in h.lower()][0]
        pts = [(float(r.split(",")[si]), float(r.split(",")[li])) for r in raw[1:]
               if len(r.split(",")) > max(si, li)]
    else:
        ds = load_dataset(name, split="train")
        pts = [(float(r["step"]), float(r["loss"])) for r in ds]
    pts.sort()
    steps = np.array([p[0] for p in pts])
    loss = np.array([p[1] for p in pts])
    d = np.diff(loss)
    thr = float(d.mean() + a.spike_sigma * d.std())
    spikes = steps[1:][d > thr]
    atlas.add("recorded_spike_steps", value=float(len(spikes)), bank_id="none",
              incumbent_id="none", model_id=repo, model_sha=info.sha, n=len(pts),
              flags={"spike_rule": f"first-difference of loss > mean + {a.spike_sigma} sd",
                     "threshold": thr, "steps": spikes[:64].tolist(), "log_kind": kind,
                     "log_name": name, "n_log_points": len(pts)})
    gates.record("W011_readout_valid", True, outcome="LOG-PRESENT", log_name=name,
                 n_spikes=int(len(spikes)), n_log_points=len(pts))
    return {"outcome": "LOG-PRESENT", "suite": repo, "log": name, "n_log_points": len(pts),
            "n_spikes": int(len(spikes)), "spike_steps": spikes[:64].tolist(),
            "note": "core-band alignment is a join against W0-9's per-step rows, done in assembly"}


def checkpoint_energy_spikes(out, atlas, a, repo, sha):
    """DISCOVERY companion to W0-11: spikes read off the CHECKPOINTS instead of a log.

    This is NOT the pre-registered readout and carries no confirmatory weight — §4/W0-11 asks
    whether core-band statistics move at the steps a public log RECORDS a spike, and no such log
    exists for this suite. What W0-9 does bank is ‖dW‖²_F per consecutive checkpoint pair, and a
    training instability leaves a signature there too. Recording it as DISCOVERY keeps the arm
    from producing nothing while keeping the frozen question unanswered rather than reinterpreted
    (feedback_no_post_unblind_rule_swap).

    If W0-9's artifact is absent, that is recorded — never inferred.
    """
    import numpy as np
    d = D.fetch_prior_arm(H.cache_dir(out, "w0_9"), a.w0_9_prefix)
    hits = sorted(pathlib.Path(d).rglob("w0_9_summary.json"))
    if not hits:
        atlas.add("checkpoint_energy_spike_available", value=0.0, bank_id="none",
                  incumbent_id="none", model_id=repo, model_sha=sha,
                  flags={"register_note": H.DISCOVERY, "w0_9_prefix": a.w0_9_prefix,
                         "outcome": "W0-9-ARTIFACT-ABSENT",
                         "why": "the companion is a join over W0-9's banked per-step delta "
                                "energies; with no artifact there is nothing to join"})
        return {"outcome": "W0-9-ARTIFACT-ABSENT", "prefix": a.w0_9_prefix}
    fro = json.loads(hits[0].read_text())["result"]["fro2_by_step"]
    assert fro, "W0-9 banked no per-step energies"
    cells = sorted(fro[0]["fro2"])
    found = {}
    for cell in cells:
        steps = np.array([f["to_step"] for f in fro], dtype=float)
        e = np.array([f["fro2"][cell] for f in fro], dtype=float)
        if e.size < 3:
            continue
        dd = np.diff(e)
        thr = float(dd.mean() + a.spike_sigma * dd.std())
        sp = steps[1:][dd > thr]
        found[cell] = sp.tolist()
        atlas.add("checkpoint_energy_spike_steps", value=float(len(sp)), bank_id="none",
                  incumbent_id="none", n=int(e.size), model_id=repo, model_sha=sha,
                  flags={"register_note": H.DISCOVERY, "cell": cell, "steps": sp[:32].tolist(),
                         "threshold": thr, "energy_series": e.tolist(),
                         "step_series": steps.tolist(),
                         "spike_rule": f"first-difference of ‖dW‖²_F > mean + {a.spike_sigma} sd",
                         "not_the_registered_readout": "the frozen question asks about LOG-"
                                                       "recorded spikes; this is a checkpoint-"
                                                       "derived proxy and is DISCOVERY only"})
    return {"outcome": "CHECKPOINT-PROXY", "per_cell": found, "n_cells": len(found),
            "register": H.DISCOVERY}


# ---------------------------------------------------------------------------
def sae_width(api, repo: str):
    """(d_model, evidence) for a published SAE repo, read from its config without downloading."""
    files = api.list_repo_files(repo)
    cfgs = [f for f in files if f.endswith("config.json")]
    if not cfgs:
        return None, {"repo": repo, "n_files": len(files), "config_files": [],
                      "why": "no config.json in the SAE repo"}
    from huggingface_hub import hf_hub_download
    cfg = json.loads(pathlib.Path(hf_hub_download(repo, cfgs[0])).read_text())
    for k in ("d_in", "d_model", "hidden_size", "input_dim", "act_size"):
        if k in cfg:
            return int(cfg[k]), {"repo": repo, "config_file": cfgs[0], "key": k, "config": cfg}
    return None, {"repo": repo, "config_file": cfgs[0], "config": cfg,
                  "why": "config declares no input width under any known key"}


def run_w0_13(out, atlas, gates, a):
    """Feature-coordinate stability, core frame vs raw vs twins — with the validity gate first."""
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from infra.pg import humboldt_weights as W
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO

    api = HfApi()
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    U_core = W.load_bank(bank_dir, "q_proj", 0, "core32")
    bank_dim = int(U_core.shape[0])
    width, evidence = sae_width(api, a.sae_repo)
    valid = (width is not None) and (width == bank_dim)
    atlas.add("sae_frame_compatible", value=float(valid), bank_id="core32", layer=0,
              type="q_proj", incumbent_id="none", model_id=a.sae_repo,
              model_sha=api.model_info(a.sae_repo).sha if valid else "n/a",
              flags={"sae_repo": a.sae_repo, "sae_d_in": width, "bank_dim": bank_dim,
                     "evidence": evidence,
                     "outcome": "SAE-FRAME-OK" if valid else "COVERAGE-IMPOSSIBLE",
                     "why": "Uᵀf is defined only when f lives in the bank's ambient space; a "
                            "width mismatch makes the projection a coincidence of shapes"})
    gates.record("W013_sae_frame_valid", bool(valid), sae_repo=a.sae_repo, sae_d_in=width,
                 bank_dim=bank_dim,
                 outcome="SAE-FRAME-OK" if valid else "COVERAGE-IMPOSSIBLE")
    print(f"[W0-13] SAE frame: d_in={width} bank_dim={bank_dim} valid={valid}", flush=True)

    # The DISCOVERY variant. Feature directions that provably live in the bank frame: the rows of
    # the base's own unembedding, restricted to a frozen token sample. They are not SAE features
    # and the arm never claims they are — they are directions in the right space, which is what
    # the stability comparison needs. Register ATLAS/DISCOVERY: no confirmatory weight (§1.2).
    base_sha = api.model_info(PG1_REPO).sha
    base_M = W._load_decoder(PG1_REPO, base_sha)[W.matrix_key("q_proj", a.layer)].numpy()
    n_dim = int(base_M.shape[1])
    assert n_dim == bank_dim, f"bank dim {bank_dim} != base input dim {n_dim}"
    rng = np.random.default_rng(a.seed)
    F = rng.standard_normal((a.n_features, n_dim))
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    feature_source = "haar-random unit directions (frame-valid placeholder)"
    if a.feature_source == "unembed":
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        d = snapshot_download(PG1_REPO, revision=base_sha,
                              allow_patterns=["*.safetensors", "*.safetensors.index.json"])
        shards = sorted(pathlib.Path(d).glob("*.safetensors"))
        idx = {}
        for sh in shards:
            with safe_open(str(sh), framework="pt") as f:
                for k in f.keys():
                    idx[k] = sh
        emb = [k for k in idx if k.endswith("embed_tokens.weight")]
        assert len(emb) == 1, f"expected 1 embed_tokens, found {emb[:4]}"
        with safe_open(str(idx[emb[0]]), framework="pt") as f:
            E = f.get_tensor(emb[0]).to(torch.float32).numpy()
        assert E.shape[1] == n_dim, f"embedding width {E.shape[1]} != {n_dim}"
        rows = rng.choice(E.shape[0], size=a.n_features, replace=False)
        F = E[rows] / np.maximum(np.linalg.norm(E[rows], axis=1, keepdims=True), 1e-12)
        feature_source = f"embed_tokens rows (n={a.n_features}, seeded sample)"

    # The 24 house fine-tunes: each one moves the base. A feature's coordinates are STABLE in a
    # frame if the frame's readout of the feature barely changes when the model does. "Raw" is the
    # identity frame restricted to the same k coordinates, so raw and banked readouts have the
    # SAME dimension and the comparison is not a dimension artifact.
    fts = W.ft_cloud(api)[:a.n_fts]
    assert fts, "the house fine-tune cloud is empty"
    raw_coords = np.sort(rng.choice(n_dim, size=H.BANK_K, replace=False))
    frames = {bid: W.load_bank(bank_dir, "q_proj", a.layer, bid) for bid in D.bank_ids_for("q_proj")}

    def readout(M, U):
        """Feature coordinates in a frame, after the model's own transform at this cell."""
        return (F @ M.T) @ U if U is not None else (F @ M.T)[:, raw_coords]

    ref = {bid: readout(base_M, U) for bid, U in frames.items()}
    ref["raw32"] = readout(base_M, None)
    drift = {k: [] for k in ref}
    used = []
    for i, repo in enumerate(fts):
        sha = api.model_info(repo).sha
        M = W._load_decoder(repo, sha)[W.matrix_key("q_proj", a.layer)].numpy()
        assert M.shape == base_M.shape, f"{repo}: {M.shape} != {base_M.shape}"
        for k in ref:
            cur = readout(M, frames[k]) if k != "raw32" else readout(M, None)
            num = np.linalg.norm(cur - ref[k], axis=1)
            den = np.maximum(np.linalg.norm(ref[k], axis=1), 1e-12)
            drift[k].append(float(np.mean(num / den)))
        used.append(repo)
        _delete_cache(repo)
        print(f"[W0-13] {i+1}/{len(fts)} {repo}", flush=True)
        D.s3_sync(out, a.s3_prefix)

    seen = set()
    for k, vals in drift.items():
        atlas.add_ci("feature_coord_drift", vals, layer=a.layer, type="q_proj", bank_id=k,
                     incumbent_id="raw32", model_id=PG1_REPO, model_sha=base_sha,
                     flags={"register_note": H.DISCOVERY, "feature_source": feature_source,
                            "n_features": a.n_features, "n_fts": len(used), "fts": used,
                            "sae_frame_valid": bool(valid), "sae_d_in": width,
                            "raw_coords": raw_coords.tolist() if k == "raw32" else None,
                            "definition": "mean over features of ‖c_ft - c_base‖/‖c_base‖ where "
                                          "c = frame readout of the feature at this cell",
                            "lower_is_more_stable": True})
        if k != "raw32":
            seen.add(k)
    D.check_controls(seen, arm="W0-13")
    means = {k: float(np.mean(v)) for k, v in drift.items()}
    core_better = means["core32"] < min(means[t] for t in H.TWIN_IDS if t in means)
    gates.record("W013_frame_stability_measured", True, means=means, n_fts=len(used),
                 sae_frame_valid=bool(valid), core_beats_all_twins=bool(core_better),
                 register=H.DISCOVERY)
    return {"sae_frame_valid": bool(valid), "sae_d_in": width, "bank_dim": bank_dim,
            "sae_evidence": evidence, "feature_source": feature_source,
            "outcome": "SAE-FRAME-OK" if valid else "COVERAGE-IMPOSSIBLE (DISCOVERY variant ran)",
            "drift_means": means, "core_beats_all_twins": bool(core_better),
            "n_fts": len(used), "register": H.DISCOVERY}


RUNNERS = {"W0-9": run_w0_9, "W0-11": run_w0_11, "W0-13": run_w0_13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--types", default="q_proj")
    ap.add_argument("--layers", default="0,4,8,12")
    ap.add_argument("--spike-sigma", type=float, default=3.0)
    ap.add_argument("--w0-9-prefix", default="darwin/darwin_w0_9")
    ap.add_argument("--sae-repo", default=GEMMA_SCOPE)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-features", type=int, default=512)
    ap.add_argument("--feature-source", default="unembed", choices=("unembed", "haar"))
    ap.add_argument("--n-fts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    if a.smoke:
        a.n_frames, a.layers, a.n_features, a.n_fts = 4, "0", 32, 2

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "suite": a.suite, "suite_fallbacks": list(SUITE_FALLBACKS),
                "n_frames": a.n_frames, "window": WINDOW,
                "crystallize_cos2": CRYSTALLIZE_COS2, "types": a.types, "layers": a.layers,
                "sae_repo": a.sae_repo, "n_features": a.n_features, "n_fts": a.n_fts,
                "loss_log_candidates": [list(c) for c in LOSS_LOG_CANDIDATES],
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    result = RUNNERS[a.arm](out, atlas, gates, a)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
