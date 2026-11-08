"""DARWIN W0-6 / W0-10 / W0-12 — family tree, recipe forensics, sonification (§4 Wave 0). CPU.

  W0-10  Recipe forensics. Label gemma-2b derivatives SFT / DPO / RLHF by a frozen metadata rule,
         then ask whether a simple classifier on their BANK SIGNATURES separates the classes on a
         held-out half. REFUTER: held-out accuracy <= the majority-class baseline. This arm is
         what distinguishes "the core is an architecture-independent property" from "the core is
         an artifact of how people fine-tune" — the confound the W0-3 census cannot exclude.
  W0-6   The gemma-2b family tree. Per accepted derivative, the A4-style bank-projection PROFILE
         (dW energy on core32/anti32/twins, per type x layer), then a profile-correlation matrix
         over derivatives and duplicate-checkpoint candidates (r > 0.999).
  W0-12  Sonification (curiosity wing, §4 Wave C). Per-layer core energy over B1 documents mapped
         to audio and banked as artifacts. "No statistics are claimed by this arm" is its own
         REFUTER line — so it emits WAV files and the rows needed to locate them, and nothing
         that looks like a measurement. §1.2 still requires it to emit atlas rows.

COST HONESTY (§12: "never silent shrinkage"). §4/W0-6 caps the tree at "top-400 by downloads, log
the cutoff". 400 derivatives is ~2 TB of checkpoint traffic, which does not fit this arm's CPU
budget. The arm therefore walks the SAME frozen ordering and stops at `--n-accept`, recording
`cutoff_rank`, `n_examined` and `n_candidates_total` on every row and in the manifest, so the
tree's depth is a stated parameter rather than an unexplained number.

  pg run infra.aws_pg.jobs.darwin_w0_misc --args "--arm W0-12" --type r7i.2xlarge ...
"""
import argparse
import json
import pathlib
import struct
import wave

from infra.pg import darwin_census as CEN
from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

GEMMA2B = "google/gemma-2b"
B1_PREFIX = "humboldt/humboldt_b1"
DUPLICATE_R = 0.999                 # §4/W0-6's duplicate-checkpoint criterion
SAMPLE_RATE = 22050
NOTE_SECONDS = 0.08                 # one token -> one short tone

# §4/W0-10's mechanical recipe-class rule. A derivative's class is read from its HF tags and its
# id, in this priority order — the first family whose keyword appears wins, so a repo carrying
# both "sft" and "dpo" is DPO (the later stage in the standard pipeline). Repos matching none are
# rejected from the arm and counted, never silently binned as SFT.
#
# MATCHING IS ON WHOLE TOKENS, and that is not a stylistic preference (defect D16). A raw
# substring test labelled 14 of 14 derivatives DPO, because the near-universal HF tag
# `endpoints_compatible` contains "dpo" — en-DPO-ints_compatible. Splitting on non-alphanumerics
# and comparing whole tokens makes that impossible. Keywords are chosen to be safe as tokens:
# "it" is deliberately NOT one, since it is also the ISO code for Italian and appears as a
# language tag; the "-it" instruct-suffix convention is handled separately against the model id's
# last path component only.
import re as _re

MIN_PER_CLASS = 4          # see the roster gate in run_w0_10 for why this is not 2
MIN_MINORITY_HELD = 3      # held-out minority examples needed before a null is claimable

RECIPE_KEYWORDS = (
    ("rlhf", ("rlhf", "ppo", "rloo")),
    ("dpo", ("dpo", "orpo", "simpo", "kto", "ipo")),
    ("sft", ("sft", "instruct", "chat", "alpaca", "tulu", "zephyr")),
)


def metadata_tokens(model_id: str, tags) -> set:
    """Whole-token vocabulary of a repo's id and tags, lowercased."""
    hay = " ".join([str(model_id).lower()] + [str(t).lower() for t in (tags or [])])
    return {t for t in _re.split(r"[^a-z0-9]+", hay) if t}


def recipe_class(model_id: str, tags) -> str:
    """The frozen classifier from metadata alone. Returns a class or "unknown"."""
    toks = metadata_tokens(model_id, tags)
    for cls, keys in RECIPE_KEYWORDS:
        if toks & set(keys):
            return cls
    # The "-it" instruct suffix, checked ONLY against the id's last path component, because the
    # bare token "it" is the Italian language tag and would fire on unrelated multilingual repos.
    if str(model_id).lower().rsplit("/", 1)[-1].endswith("-it"):
        return "sft"
    return "unknown"


SPEC_EXPECTATIONS = {
    "W0-10": ("held-out accuracy <= chance + CI",
              "a frozen simple classifier on bank/spectral signatures separates recipe classes "
              "above chance on held-out checkpoints"),
    "W0-6": ("profile correlation carries no family structure (dendrogram indistinguishable from "
             "noise)",
             "derivative profiles cluster by lineage, and profile drift grows with training-time "
             "gap"),
    "W0-12": ("no statistics are claimed by this arm (joy arm)",
              "audio artifacts are banked for every B1 document"),
}


# ---------------------------------------------------------------------------
# W0-12 — sonification.
# ---------------------------------------------------------------------------
def write_wav(path, samples, rate: int = SAMPLE_RATE):
    """16-bit mono PCM, via the stdlib `wave` module so the arm has no audio dependency."""
    import numpy as np
    a = np.asarray(samples, dtype=np.float64)
    peak = float(np.abs(a).max())
    a = a / peak if peak > 0 else a
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack(f"<{pcm.size}h", *pcm.tolist()))
    return path


def sonify(series, *, f_lo: float = 180.0, f_hi: float = 1800.0, rate: int = SAMPLE_RATE,
           note_seconds: float = NOTE_SECONDS):
    """Map a per-token series to a tone sequence: value -> log-spaced pitch, fixed duration.

    Log-spaced because pitch perception is logarithmic; a linear map would compress the whole
    interesting range of e_U into a few semitones. Each note is envelope-shaped so the sequence
    does not click at every boundary.
    """
    import numpy as np
    v = np.asarray(series, dtype=np.float64).ravel()
    assert v.size > 0, "sonify on an empty series"
    lo, hi = float(v.min()), float(v.max())
    norm = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    freqs = f_lo * (f_hi / f_lo) ** norm
    n = max(2, int(rate * note_seconds))
    t = np.arange(n) / rate
    ramp = max(1, int(0.005 * rate))
    env = np.ones(n)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    out = np.concatenate([np.sin(2 * np.pi * f * t) * env for f in freqs])
    return out, {"n_notes": int(v.size), "f_lo": f_lo, "f_hi": f_hi, "value_min": lo,
                 "value_max": hi, "note_seconds": note_seconds, "sample_rate": rate,
                 "duration_s": float(out.size / rate)}


def run_w0_12(out, atlas, gates, args):
    """Bank one WAV per (model, corpus, layer) from B1's core32 e_U series."""
    b1 = D.fetch_prior_arm(H.cache_dir(out, "b1"), args.b1_prefix)
    df = D.load_atlas(b1)
    ser = df[(df["statistic"] == "e_U_series") & (df["bank_id"] == "core32")
             & (df["type"] == "q_proj")]
    assert len(ser), "no core32 q_proj e_U series in the B1 atlas"
    audio_dir = pathlib.Path(out) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    models = sorted(ser["model_id"].dropna().unique())
    for model in models:
        sm = ser[ser["model_id"] == model]
        for corpus in sorted(sm["corpus_id"].dropna().unique()):
            sc = sm[sm["corpus_id"] == corpus]
            for L in sorted(sc["layer"].dropna().unique())[::args.layer_stride]:
                row = sc[sc["layer"] == L].iloc[0]
                v = D.vector_of(b1, row)[:args.max_notes]
                wavs, meta = sonify(v)
                name = f"W0-12__{model}__{corpus}__L{int(L)}.wav".replace("/", "-")
                write_wav(audio_dir / name, wavs)
                atlas.add("sonification_artifact", value=float(meta["duration_s"]),
                          layer=int(L), type="q_proj", bank_id="core32", model_id=model,
                          corpus_id=corpus,
                          flags={**meta, "artifact": f"audio/{name}",
                                 "note": "curiosity-wing artifact; no statistic is claimed"})
                made += 1
            D.s3_sync(out, args.s3_prefix)
        print(f"[W0-12] {model}: {made} audio artifacts so far", flush=True)
    gates.record("W0_12_audio_banked", made > 0, n_artifacts=made, n_models=len(models),
                 note="joy arm: the deliverable is the artifact set, not a statistic")
    return {"n_artifacts": made, "n_models": len(models), "audio_dir": "audio/"}


# ---------------------------------------------------------------------------
# W0-6 — the family tree.
# ---------------------------------------------------------------------------
def profile_vector(dW_by_cell, bank_dir, types, layers, bank_ids):
    """The derivative's PROFILE: dW energy FRACTION on each bank, flattened over cells.

    Fraction, not raw energy: ‖dW U‖ scales with how hard the derivative was trained, so a
    correlation over raw values would cluster by training intensity rather than by lineage.
    """
    import numpy as np
    vals, keys = [], []
    for t in types:
        for L in layers:
            A = dW_by_cell[(t, L)]
            den = float(np.linalg.norm(A) ** 2)
            for bid in bank_ids:
                U = W.load_bank(bank_dir, t, L, bid)
                vals.append(float(np.linalg.norm(A @ U) ** 2 / den) if den > 0 else 0.0)
                keys.append(f"{t}/L{L}/{bid}")
    return np.asarray(vals, dtype=np.float64), keys


def run_w0_6(out, atlas, gates, args):
    import numpy as np
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    api = HfApi()
    base_sha = api.model_info(GEMMA2B).sha
    cfg = AutoConfig.from_pretrained(GEMMA2B, revision=base_sha)
    geom = CEN.base_geometry(cfg)
    accepted, log = CEN.select_derivatives(api, GEMMA2B, geom, n_accept=args.n_accept,
                                           n_examine=args.n_examine, out=out)
    cutoff = {"cutoff_rank": args.n_accept, "n_examined": args.n_examine,
              "n_candidates_total": log["n_candidates"], "n_accepted": len(accepted),
              "spec_cap": 400,
              "reason": "400 derivatives is ~2 TB of checkpoint traffic, past this arm's CPU "
                        "budget; the frozen ordering is unchanged and the cutoff is recorded"}
    D.write_json(out, "w0_6_cutoff.json", cutoff)
    gates.record("W0_6_tree_depth", True, **cutoff)
    assert len(accepted) >= 3, f"family tree needs >=3 derivatives, got {len(accepted)}"

    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    types = list(args.types or ("q_proj", "o_proj", "gate_proj"))
    layers = list(range(geom["num_hidden_layers"]))[:args.max_layers]
    bank_ids = ["core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2"]

    base = CEN.load_decoder_matrices(GEMMA2B, base_sha, types, layers)
    profiles, names = [], []
    for rec in accepted:
        repo = rec["id"]
        sha = api.model_info(repo).sha
        ft = CEN.load_decoder_matrices(repo, sha, types, layers)
        dW = {k: (ft[k].double() - base[k].double()).numpy() for k in base}
        p, keys = profile_vector(dW, bank_dir, types, layers, bank_ids)
        profiles.append(p)
        names.append(repo)
        atlas.add("family_profile",
                  vector_ref=atlas.vector(f"W0-6__profile__{repo.replace('/', '_')}", p),
                  model_id=repo, model_sha=sha, n=int(p.size),
                  flags={**cutoff, "downloads": rec["downloads"], "is_merge": rec["is_merge"],
                         "profile_keys": keys[:8], "n_cells": len(keys),
                         "ckpt_dtype": rec["ckpt_dtype"]})
        del ft, dW
        _delete_cache(repo)
        D.s3_sync(out, args.s3_prefix)
        print(f"[W0-6] profiled {repo} ({len(profiles)}/{len(accepted)})", flush=True)

    P = np.vstack(profiles)
    Cm = np.corrcoef(P)
    atlas.add("family_profile_correlation",
              vector_ref=atlas.vector("W0-6__corr", Cm), n=int(Cm.size),
              flags={"models": names, "shape": list(Cm.shape), **cutoff})
    dupes = [{"a": names[i], "b": names[j], "r": float(Cm[i, j])}
             for i in range(len(names)) for j in range(i + 1, len(names))
             if Cm[i, j] > DUPLICATE_R]
    for d in dupes:
        atlas.add("duplicate_checkpoint_candidate", value=d["r"], model_id=d["a"],
                  flags={**d, "criterion": f"profile r > {DUPLICATE_R}"})
    off = Cm[np.triu_indices_from(Cm, k=1)]
    atlas.add_ci("family_profile_correlation_offdiag", off.tolist(),
                 flags={"n_pairs": int(off.size), "n_models": len(names), **cutoff})
    gates.record("W0_6_duplicates_found", True, n_duplicate_pairs=len(dupes),
                 criterion=DUPLICATE_R, pairs=dupes[:20])
    D.write_json(out, "w0_6_tree.json",
                 {"models": names, "correlation": Cm.tolist(), "duplicates": dupes,
                  "cutoff": cutoff})
    return {"n_models": len(names), "n_duplicate_pairs": len(dupes),
            "mean_offdiag_r": float(off.mean()), "cutoff": cutoff}



# ---------------------------------------------------------------------------
# W0-10 — recipe forensics.
# ---------------------------------------------------------------------------
def base_sigma_banks(base_matrices, k=32, seed=1337):
    """Per-cell BASE-NATIVE spectral banks: the base matrix's own top-k and bottom-k right
    singular subspaces, plus matched Haar twins.

    W0-10 must be portable across bases, and the published core32/anti32 banks are not: they live
    in PG1's 2048-dim input space, so projecting a Mistral delta (4096) onto them is a shape error
    (defect D18) — and would be meaningless even if the shapes matched, since they encode a
    different model's consensus. §4/W0-10 asks for "bank/SPECTRAL signatures", and a base's own
    singular subspaces are exactly that: defined by the base alone, computed once per base, and
    cheap per derivative thereafter (matmuls, no further SVD).
    """
    from infra.pg import humboldt_core as _H
    banks = {}
    for key, W0 in base_matrices.items():
        A = W0.numpy().astype("float64") if hasattr(W0, "numpy") else W0
        bot, top, _s = _H.sigma_subspaces(A, k)
        n = A.shape[1]
        b = {"base_topsigma": top, "base_bottomsigma": bot}
        for sd in _H.HAAR_SEEDS:
            b[f"haar32_s{sd}"] = _H.haar_subspace(n, k, _H.cell_seed(seed, str(key), sd))
        banks[key] = b
    return banks


def spectral_profile(dW_by_cell, banks_by_cell):
    """dW energy FRACTION on each base-native spectral bank, flattened over cells."""
    import numpy as np
    vals, keys = [], []
    for key in sorted(dW_by_cell, key=lambda k: (str(k[0]), int(k[1]))):
        A = dW_by_cell[key]
        den = float(np.linalg.norm(A) ** 2)
        for bid in sorted(banks_by_cell[key]):
            U = banks_by_cell[key][bid]
            vals.append(float(np.linalg.norm(A @ U) ** 2 / den) if den > 0 else 0.0)
            keys.append(f"{key[0]}/L{key[1]}/{bid}")
    return np.asarray(vals, dtype=np.float64), keys


def select_stratified(api, base_repo, geom, per_class, n_examine, out):
    """Walk the frozen download ordering and accept up to `per_class` derivatives OF EACH RECIPE
    CLASS.

    A single global quota does not work here: the ordering is by downloads, the popular prefix is
    overwhelmingly SFT, and a 40-wide global quota on gemma-2b returned 13 SFT / 2 DPO / 1 RLHF —
    a roster too unbalanced for a held-out claim, and unbalanced for a reason that has nothing to
    do with the science. Stratifying keeps the ordering frozen and the rule mechanical while
    letting the minority classes fill.
    """
    from transformers import AutoConfig
    cands, seen = [], set()
    for form in CEN.TAG_FORMS:
        for m in api.list_models(filter=form.format(base_repo), sort="downloads", limit=100):
            if m.id in seen:
                continue
            seen.add(m.id)
            cands.append({"id": m.id, "downloads": int(m.downloads or 0),
                          "tag_form": form.format(base_repo),
                          "recipe": recipe_class(m.id, getattr(m, "tags", []))})
    cands.sort(key=lambda c: -c["downloads"])

    log, accepted, per = [], [], {c: 0 for c, _ in RECIPE_KEYWORDS}
    for c in cands[:n_examine]:
        cls = c["recipe"]
        if cls == "unknown":
            log.append({**c, "decision": "reject", "reason": "no recipe keyword in id or tags"})
            continue
        if per[cls] >= per_class:
            log.append({**c, "decision": "not_examined",
                        "reason": f"class quota of {per_class} already filled for {cls}"})
            continue
        try:
            files = {x.rfilename for x in api.model_info(c["id"], files_metadata=False).siblings}
            if "adapter_config.json" in files:
                log.append({**c, "decision": "reject", "reason": "adapter-only repo"})
                continue
            if not CEN.loadable_entrypoint(files):
                log.append({**c, "decision": "reject", "reason": "no safetensors data"})
                continue
            cfg = AutoConfig.from_pretrained(c["id"])
        except Exception as e:      # noqa: BLE001 — enumeration boundary
            log.append({**c, "decision": "reject",
                        "reason": f"unreadable: {type(e).__name__}: {str(e)[:120]}"})
            continue
        ok, why = CEN.cfg_prefilter(cfg, geom)
        if not ok:
            log.append({**c, "decision": "reject", "reason": why})
            continue
        log.append({**c, "decision": "accept", "reason": why})
        accepted.append(c)
        per[cls] += 1
    D.write_json(out, f"w0_10_selection__{base_repo.replace('/', '_')}.json",
                 {"rule": {"per_class": per_class, "n_examine": n_examine,
                           "sort": "downloads desc", "stratified": True},
                  "n_candidates": len(cands), "accepted_per_class": per, "log": log})
    return accepted, per, len(cands)


def run_w0_10(out, atlas, gates, args):
    """Can a frozen simple classifier tell SFT from DPO from RLHF by bank signature alone?

    Fit on half, test on the other half, with the split made BEFORE any fitting and by a hash of
    the model id so it cannot drift with roster order. The classifier is a nearest-centroid rule
    on the standardised profile — deliberately the simplest thing that could work, because the
    question is whether the SIGNATURE carries recipe information, not how good a classifier can
    be built.

    Chance is not 1/n_classes here: the classes are unbalanced, so the honest baseline is the
    MAJORITY-CLASS rate on the held-out half, and that is what the REFUTER is scored against.
    """
    import hashlib
    import numpy as np
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    api = HfApi()
    base_repo = args.base
    base_sha = api.model_info(base_repo).sha
    geom = CEN.base_geometry(AutoConfig.from_pretrained(base_repo, revision=base_sha))
    labelled, counts, n_cands = select_stratified(api, base_repo, geom, args.per_class,
                                                  args.n_examine, out)
    print(f"[W0-10] base {base_repo}: labelled {len(labelled)} of {n_cands} candidates {counts}",
          flush=True)
    atlas.add("recipe_roster_size", value=float(len(labelled)), model_id=base_repo,
              model_sha=base_sha, flags={"counts": counts, "n_candidates": n_cands,
                                         "per_class_quota": args.per_class,
                                         "base_selection": args.base_selection})
    # MIN_PER_CLASS is 4, not 2, and the first run is why. With 2 minority examples the roster
    # passed a ">=2 each" gate, and the classifier then scored held-out accuracy 0.8333 against a
    # majority baseline of 0.8333 — EXACTLY equal. That outcome is indistinguishable between "the
    # bank signature carries no recipe information" and "one or two held-out minority examples is
    # not enough to show that it does", so the arm could not have reported a null honestly. Four
    # per class is still small; the arm therefore also records an explicit power verdict below
    # rather than letting accuracy-at-baseline read as a refutation.
    n_classes_ok = sum(1 for v in counts.values() if v >= MIN_PER_CLASS)
    if len(labelled) < 2 * MIN_PER_CLASS or n_classes_ok < 2:
        gates.record("W0_10_roster_sufficient", False, n_labelled=len(labelled), counts=counts,
                     min_per_class=MIN_PER_CLASS,
                     note=f"need >={2 * MIN_PER_CLASS} labelled spanning >=2 classes with "
                          f">={MIN_PER_CLASS} each; a thinner roster cannot separate 'no signal' "
                          f"from 'no power'")
        return {"status": "VOID-underpowered-roster", "counts": counts,
                "n_labelled": len(labelled), "min_per_class": MIN_PER_CLASS}
    gates.record("W0_10_roster_sufficient", True, n_labelled=len(labelled), counts=counts,
                 min_per_class=MIN_PER_CLASS)

    types = list(args.types or ("q_proj", "o_proj", "gate_proj"))
    layers = list(range(geom["num_hidden_layers"]))[:args.max_layers]
    base = CEN.load_decoder_matrices(base_repo, base_sha, types, layers)
    # Base-NATIVE spectral banks, computed once. The published core32/anti32 are PG1-shaped and
    # cannot be applied to a foreign base (defect D18).
    sig_banks = base_sigma_banks(base, k=H.BANK_K, seed=args.seed)
    bank_ids = sorted(next(iter(sig_banks.values())))
    print(f"[W0-10] base-native spectral banks: {bank_ids}", flush=True)

    X, y, ids = [], [], []
    for rec in labelled:
        sha = api.model_info(rec["id"]).sha
        ok, why = CEN.validate_candidate(rec["id"], sha, {k: tuple(v.shape) for k, v in base.items()},
                                         types, layers)
        if not ok:
            print(f"[W0-10] DROP {rec['id']}: {why[:90]}", flush=True)
            continue
        ft = CEN.load_decoder_matrices(rec["id"], sha, types, layers)
        dW = {k: (ft[k].double() - base[k].double()).numpy() for k in base}
        prof, keys = spectral_profile(dW, sig_banks)
        X.append(prof)
        y.append(rec["recipe"])
        ids.append(rec["id"])
        atlas.add("recipe_profile",
                  vector_ref=atlas.vector(f"W0-10__profile__{rec['id'].replace('/', '_')}", prof),
                  model_id=rec["id"], model_sha=sha, n=int(prof.size),
                  flags={"recipe": rec["recipe"], "n_cells": len(keys)})
        del ft, dW
        _delete_cache(rec["id"])
        D.s3_sync(out, args.s3_prefix)

    X = np.vstack(X)
    y = np.asarray(y)
    # Split BEFORE fitting, by a hash of the id: reproducible and independent of roster order.
    held = np.array([int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 == 0 for i in ids])
    if held.all() or (~held).all() or len(set(y[~held])) < 2:
        gates.record("W0_10_split_usable", False, n_fit=int((~held).sum()), n_held=int(held.sum()),
                     note="hash split left a degenerate half; no held-out claim is possible")
        return {"status": "split-degenerate", "n": len(ids)}
    gates.record("W0_10_split_usable", True, n_fit=int((~held).sum()), n_held=int(held.sum()))

    mu, sd = X[~held].mean(0), X[~held].std(0) + 1e-12       # standardise on the FIT half only
    Z = (X - mu) / sd
    classes = sorted(set(y[~held]))
    cent = {c: Z[~held][y[~held] == c].mean(0) for c in classes}
    pred = np.array([min(classes, key=lambda c: float(np.linalg.norm(z - cent[c]))) for z in Z])
    acc = float((pred[held] == y[held]).mean())
    vals, cnts = np.unique(y[held], return_counts=True)
    majority = float(cnts.max() / cnts.sum())
    atlas.add("recipe_classifier_heldout_accuracy", value=acc, model_id=base_repo,
              model_sha=base_sha, n=int(held.sum()),
              flags={"majority_baseline": majority, "classes": classes,
                     "n_fit": int((~held).sum()), "n_held": int(held.sum()),
                     "classifier": "nearest centroid on standardised bank profile",
                     "held_ids": [i for i, h in zip(ids, held) if h]})
    # §3 power branch. A held-out half whose minority class is tiny cannot distinguish "no recipe
    # signal" from "no power to see one", so the arm says which it is instead of implying the
    # former. n_minority is the count in the HELD-OUT half, since that is what the accuracy is
    # computed on.
    n_minority_held = int(cnts.min())
    powered = bool(n_minority_held >= MIN_MINORITY_HELD)
    verdict = ("above-majority" if (powered and acc > majority)
               else "REFUTER-fires" if powered else "VOID-underpowered")
    gates.record("W0_10_above_majority", bool(acc > majority), heldout_accuracy=acc,
                 majority_baseline=majority, n_held=int(held.sum()),
                 n_minority_held=n_minority_held, powered=powered, verdict=verdict,
                 min_minority_held=MIN_MINORITY_HELD,
                 note="RECORDED, not blocking. accuracy <= majority is W0-10's named REFUTER, but "
                      "ONLY when the held-out half has enough minority examples to have shown "
                      "otherwise; below that it is VOID-underpowered, which is a different "
                      "outcome and is labelled as one")
    atlas.add("recipe_classifier_verdict", value=float(acc - majority), model_id=base_repo,
              model_sha=base_sha,
              flags={"verdict": verdict, "heldout_accuracy": acc, "majority_baseline": majority,
                     "n_minority_held": n_minority_held, "powered": powered})
    return {"n_models": len(ids), "heldout_accuracy": acc, "majority_baseline": majority,
            "classes": classes, "counts": counts, "verdict": verdict,
            "n_minority_held": n_minority_held}


RUNNERS = {"W0-6": run_w0_6, "W0-10": run_w0_10, "W0-12": run_w0_12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--b1-prefix", default=B1_PREFIX)
    ap.add_argument("--types", default="")
    ap.add_argument("--n-accept", type=int, default=16)
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1",
                    help="W0-10's base. Chosen mechanically by recipe-class depth; see "
                         "--base-selection for the evidence.")
    ap.add_argument("--base-selection", default=(
        "metadata probe over 6 candidate bases (no downloads): Mistral-7B-v0.1 sft=102 dpo=37 "
        "rlhf=9 is the only base with all three classes >= MIN_PER_CLASS. gemma-2b sft=42 dpo=4 "
        "rlhf=2; Llama-3.2-3B sft=50 dpo=16 rlhf=1; Qwen2.5-3B sft=49 dpo=1 rlhf=1; "
        "Llama-3.1-8B sft=102 dpo=1 rlhf=0; Mistral-7B-v0.3 sft=51 dpo=12 rlhf=2."))
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--n-examine", type=int, default=CEN.N_EXAMINE_DEFAULT)
    ap.add_argument("--max-layers", type=int, default=6)
    ap.add_argument("--layer-stride", type=int, default=6)
    ap.add_argument("--max-notes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    a.types = [t.strip() for t in a.types.split(",") if t.strip()] or None
    if a.smoke:
        a.n_accept, a.max_layers, a.max_notes, a.layer_stride = 3, 2, 60, 12

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
                "base": (a.base if a.arm == "W0-10" else GEMMA2B),
                "base_selection": (a.base_selection if a.arm == "W0-10" else
                                   "W0-6/W0-12 are gemma-2b arms by definition"),
                "per_class": a.per_class, "min_per_class": MIN_PER_CLASS,
                "min_minority_held": MIN_MINORITY_HELD,
                "n_accept": a.n_accept, "max_layers": a.max_layers,
                "duplicate_r": DUPLICATE_R, "expected_ref": ref, "started_utc": utc,
                "seed": a.seed}
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
