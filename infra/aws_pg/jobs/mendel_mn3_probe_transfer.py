"""MENDEL MN-3 — the frozen-probe transfer matrix (addendum item 32, the flagship's third leg).

THE QUESTION. Train a linear probe on the BASE model's representation INSIDE the sink subspace,
freeze it, and evaluate it on descendants and down-lineage. If the sink is a transported structure
rather than a per-model accident, a probe fitted in PG1's core subspace should still decode in
π0's — more than one fitted in an equally-sized subspace chosen for activation energy.

KILL, as amended at freeze (A9): **transfer no better than the HIGHWAY-PROJECTION control.** The
spec's own text listed highway-projection, haar-32 and full-stream as the controls and then wrote
the kill against haar; keying a kill on the weakest control in the table is the direction that
manufactures effects, and §5's standing rule is never-haar-alone.

THE FOUR ARMS, all the same probe on the same tokens, differing only in which 32 directions the
representation is read through:

    core       the banked `core32` sink subspace                     TREATMENT
    highway    the top-32 ACTIVATION-ENERGY directions (G4b's       STRUCTURED — the kill control
               activation-Gram eigenbasis); the directions the
               stream actually travels on
    haar32     a random 32-dim subspace                              HAAR — the chance floor
    full       the unprojected 2048-dim stream                       the CEILING (M-G's
                                                                     ceiling-ladder lesson)

THE PROBE TARGET is model-independent by construction: one-hot over the K most frequent token ids
in the frozen batch. A target derived from any model's own outputs would change between models and
make "the same probe" meaningless.

THE TRANSPORT IS OF THE SUBSPACE, NOT THE FIT. The projection matrix and the probe weights are both
fitted on PG1 and then held fixed; a descendant's activations are read through PG1's directions
with PG1's probe. Re-fitting either on the descendant would answer a different and much easier
question.

READOUT-VALIDITY GATE (COMWRITE, reused): a cell whose FULL-stream probe cannot decode the target on
the base model at all is measuring the instrument, not the model, and is VOID-readout.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import inherit_core as IC
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mercator_core as MerC

ARM = "MN-3"
K_SUB = H.BANK_K
K_TOKENS = 32
N_SEQ = 24
ALPHA = 1.0
TRAIN_FRAC = 0.5
MIN_BASE_R2 = 0.02            # the full-stream probe must decode SOMETHING on the base
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
G4B_PREFIX = f"{MC.RESULTS_S3}/mendel_g4b_v1/out"
SITE_TYPE = "q_proj"


def _windows(processor, n_seq: int, window: int = 512):
    """THE frozen batch — byte-identical to G4's and G4b's."""
    from datasets import load_dataset
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    out = []
    for start in range(0, ids.size(0), window):
        chunk = ids[start:start + window]
        if chunk.size(0) < 8:
            break
        out.append(tok.decode(chunk.tolist()))
        if len(out) >= n_seq:
            break
    assert len(out) == n_seq, f"only {len(out)} windows available, needed {n_seq}"
    return out


def harvest_codes(model, processor, device, texts, blank, projections, layers):
    """Per-layer 32-dim codes Z = h @ U for every arm, plus the token ids, in one forward pass.

    Projecting inside the hook rather than storing h is what keeps this affordable: the full
    activations would be 24 x 512 x 2048 x 18 layers, while the codes are 32-dimensional. The
    token ids are captured alongside so the probe target is aligned to the same rows.
    """
    import torch
    from infra.pg.entities import resolve_submodule_paths
    lm = resolve_submodule_paths(model)["language_model"]
    mods = dict(model.named_modules())
    sites = {}
    for L in layers:
        suffix = f".layers.{L}.self_attn.{SITE_TYPE}"
        hits = [n for n in mods if n.endswith(suffix) and n.startswith(lm)]
        assert len(hits) == 1, f"{suffix} under {lm!r}: {len(hits)} matches"
        sites[L] = hits[0]

    codes = {(L, arm): [] for L in layers for arm in projections[layers[0]]}
    full = {L: [] for L in layers}

    def hook(L):
        def f(_m, inputs, _o):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float64)
            for arm, U in projections[L].items():
                codes[(L, arm)].append((x @ torch.as_tensor(U, dtype=torch.float64,
                                                            device=x.device)).cpu().numpy())
            full[L].append(x.cpu().numpy())
        return f

    handles = [mods[sites[L]].register_forward_hook(hook(L)) for L in layers]
    tok_ids = []
    with torch.no_grad():
        for txt in texts:
            inp = processor(text=[txt], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
            tok_ids.append(inp["input_ids"][0].detach().cpu().numpy())
    for h in handles:
        h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in codes.items()},
            {L: np.concatenate(v, axis=0) for L, v in full.items()},
            np.concatenate(tok_ids, axis=0))


def token_targets(tok_ids, n_rows: int, k: int):
    """One-hot over the k most frequent token ids — a MODEL-INDEPENDENT probe target.

    The activation rows include the image-prefix positions PaliGemma prepends, so there are more
    rows than text token ids. The text ids are aligned to the LAST rows of each window's block,
    which is where the text sits; the count is asserted rather than assumed.
    """
    ids = np.asarray(tok_ids).ravel()
    assert n_rows >= ids.size, f"{n_rows} activation rows for {ids.size} token ids"
    pad = n_rows - ids.size
    ids_full = np.concatenate([np.full(pad, -1, dtype=ids.dtype), ids])
    vals, counts = np.unique(ids[ids >= 0], return_counts=True)
    top = vals[np.argsort(-counts)][:k]
    Y = np.stack([(ids_full == t).astype(np.float64) for t in top], axis=1)
    keep = ids_full >= 0
    assert Y[keep].sum(0).min() > 5, "a probe target appears fewer than 6 times"
    return Y, keep, [int(t) for t in top], int(pad)


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-3 — frozen-probe transfer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="4,9,14")
    ap.add_argument("--targets", default="pi0,pi05",
                    help="lineage members to transfer to, beyond the sampled fine-tunes")
    ap.add_argument("--n-ft", type=int, default=6, help="descendants sampled from the 24")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn3", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn3"), out)
    layers = [int(x) for x in a.layers.split(",") if x.strip()]

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1
    from infra.aws_pg.jobs.mendel_mn1_transport import _lineage_decoder
    from infra.pg.entities import resolve_submodule_paths

    assert torch.cuda.is_available(), "no CUDA — MN-3 runs forward passes"
    device = "cuda"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha

    banks = WT.fetch_banks(H.cache_dir(out, "banks"))
    g4b_vec = man.require("g4b_vectors", f"{G4B_PREFIX}/vectors", recursive=True)
    projections = {}
    rng = np.random.default_rng(a.seed)
    for L in layers:
        U_core = WT.load_bank(banks, SITE_TYPE, L, "core32")
        evecs = np.load(g4b_vec / f"act_gram_evecs__{SITE_TYPE}__L{L}.npy").astype(np.float64)
        projections[L] = {
            "core": U_core,
            "highway": np.ascontiguousarray(evecs[:, :K_SUB]),      # top activation-energy dirs
            "haar32": np.linalg.qr(rng.standard_normal((H.IN_DIM[SITE_TYPE], K_SUB)))[0],
        }

    # ---------- the BASE: fit the probes ----------
    model = _load_full_pg1(base_sha, device)
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    texts = _windows(processor, N_SEQ)
    blank = _blank_image(128)
    codes, full, tok_ids = harvest_codes(model, processor, device, texts, blank,
                                         projections, layers)
    n_rows = next(iter(full.values())).shape[0]
    Y, keep, top_tokens, n_pad = token_targets(tok_ids, n_rows, K_TOKENS)
    idx = np.where(keep)[0]
    rng2 = np.random.default_rng(a.seed + 1)
    perm = rng2.permutation(idx.size)
    n_tr = int(TRAIN_FRAC * idx.size)
    tr, te = idx[perm[:n_tr]], idx[perm[n_tr:]]
    print(f"[mn3] {n_rows} rows ({n_pad} image-prefix), {idx.size} text rows, "
          f"{n_tr} train / {idx.size - n_tr} test, {K_TOKENS} targets", flush=True)

    base_r2, probes = {}, {}
    for L in layers:
        for arm in ("core", "highway", "haar32"):
            Z = codes[(L, arm)]
            r2, _ = IC.ridge_r2(Z[tr], Y[tr], Z[te], Y[te], alpha=ALPHA)
            base_r2[(L, arm)] = r2
            probes[(L, arm)] = (Z[tr], Y[tr])
        r2f, _ = IC.ridge_r2(full[L][tr], Y[tr], full[L][te], Y[te], alpha=ALPHA)
        base_r2[(L, "full")] = r2f
        probes[(L, "full")] = (full[L][tr], Y[tr])
        v = MerC.readout_validity(ft_score=r2f, base_score=MIN_BASE_R2, higher_is_better=True,
                                  model_id=WT.PG1_REPO, task=f"top{K_TOKENS}-token")
        gates.record(f"mn3_readout_valid_L{L}", v["passed"], **{k: x for k, x in v.items()
                                                                if k != "gate"})
        print(f"[mn3] base L{L}: core {base_r2[(L, 'core')]:.4f} "
              f"highway {base_r2[(L, 'highway')]:.4f} haar {base_r2[(L, 'haar32')]:.4f} "
              f"full {r2f:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---------- the TARGETS: evaluate the FROZEN probes ----------
    repos = {}
    for name in [x.strip() for x in a.targets.split(",") if x.strip()]:
        repos[name] = name
    for r in WT.ft_cloud(api)[: a.n_ft]:
        repos["ft-" + r.split("paligemma-3b-ft-")[1].rsplit("-224", 1)[0]] = r

    rows = []
    for unit, repo in repos.items():
        if unit in ("pi0", "pi05"):
            print(f"[mn3] {unit}: lineage member — decoder swap into the PG1 shell", flush=True)
            sd, sha = _lineage_decoder(unit)
            m = _load_full_pg1(base_sha, device)
            lm = m.get_submodule(resolve_submodule_paths(m)["language_model"])
            inner = lm.model if hasattr(lm, "model") else lm
            missing = inner.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()},
                                            strict=False)
            assert not missing.unexpected_keys, f"{unit}: unexpected keys {missing.unexpected_keys[:3]}"
        else:
            from transformers import PaliGemmaForConditionalGeneration
            sha = api.model_info(repo).sha
            m = PaliGemmaForConditionalGeneration.from_pretrained(
                repo, revision=sha, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="eager").eval()
        c, f, _ = harvest_codes(m, processor, device, texts, blank, projections, layers)
        for L in layers:
            for arm in ("core", "highway", "haar32", "full"):
                Ztr, Ytr = probes[(L, arm)]
                Zte = (f[L] if arm == "full" else c[(L, arm)])[te]
                r2, _ = IC.ridge_r2(Ztr, Ytr, Zte, Y[te], alpha=ALPHA)
                rows.append({"unit": unit, "layer": L, "arm": arm, "r2": r2,
                             "base_r2": base_r2[(L, arm)],
                             "retention": r2 / base_r2[(L, arm)] if base_r2[(L, arm)] > 0
                             else float("nan")})
                cid = {"core": MC.TREATMENT, "highway": MC.STRUCTURED,
                       "haar32": MC.HAAR, "full": MC.TREATMENT}[arm]
                at.add("frozen_probe_transfer_r2", value=r2, layer=L, type=SITE_TYPE,
                       bank_id=arm, item_id=unit, control_id=cid, model_id=repo,
                       model_sha=sha, corpus_id="wikitext",
                       flags={"base_r2": base_r2[(L, arm)], "k_sub": K_SUB,
                              "k_targets": K_TOKENS})
        print(f"[mn3] {unit}: " + " ".join(
            f"{arm}={np.mean([r['r2'] for r in rows if r['unit'] == unit and r['arm'] == arm]):.4f}"
            for arm in ("core", "highway", "haar32", "full")), flush=True)
        del m
        torch.cuda.empty_cache()
        if unit.startswith("ft-"):
            from infra.aws_pg.jobs.invariant_core import _delete_cache
            _delete_cache(repo)

    # ---------- the frozen rule ----------
    core = [r["r2"] for r in rows if r["arm"] == "core"]
    highway = [r["r2"] for r in rows if r["arm"] == "highway"]
    haar = [r["r2"] for r in rows if r["arm"] == "haar32"]
    ci = MC.paired_delta_ci(core, highway, seed=a.seed)
    ci_haar = MC.paired_delta_ci(core, haar, seed=a.seed)
    at.add_ci("frozen_probe_transfer_r2", np.asarray(core) - np.asarray(highway),
              bank_id="core_minus_highway", control_id=MC.SHUFFLE, seed=a.seed,
              flags={"rule": "KILL (freeze A9) if core does not beat the HIGHWAY control"})
    verdict = MP.check_verdict(ARM, "TRANSFERS" if ci["clears_zero"] else
                               ("NO-TRANSFER" if ci["ci_hi"] < 0 else "AMBIGUOUS"))
    gates.record("mn3_core_beats_highway", ci["clears_zero"], **ci,
                 note="freeze A9: the KILL is written against the STRUCTURED control, not haar")
    at.check_control_trinity(["frozen_probe_transfer_r2"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "site": SITE_TYPE, "layers": layers, "k_subspace": K_SUB, "k_targets": K_TOKENS,
        "top_tokens": top_tokens, "n_rows": int(n_rows), "n_image_prefix_rows": n_pad,
        "n_train": int(n_tr), "n_test": int(idx.size - n_tr),
        "base_r2": {f"L{L}/{arm}": base_r2[(L, arm)] for L in layers
                    for arm in ("core", "highway", "haar32", "full")},
        "core_vs_highway": ci, "core_vs_haar": ci_haar,
        "n_units": len(repos), "units": sorted(repos),
        "rows": rows, "atlas": info,
        "kill_control": "highway-projection (freeze A9), NOT haar"})
    print("MN3_OK " + json.dumps({"verdict": verdict, "core_minus_highway":
                                  round(ci["delta_mean"], 5), "n_units": len(repos)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
