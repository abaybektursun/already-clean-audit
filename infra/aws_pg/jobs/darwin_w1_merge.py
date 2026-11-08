"""DARWIN W1-K / W1-P — watermarking and highway-aware merging. GPU.

  W1-K  Watermark (§4/W1-K). *** VOID-TAUTOLOGICAL AS IMPLEMENTED — DO NOT READ ITS ROWS AS A
        RESULT (defect D21). *** The arm stamps the base with a rank-1 in-core perturbation and
        then ADDS a fine-tune's delta arithmetically:

            W_eff = W0 + amp*outer(v,u) + dW,     recovery measured on  W_eff - W0 - dW

        which is identically amp*outer(v,u). Recovery is therefore EXACTLY 1.0 for every bank at
        every amplitude — measured: 0.9958-1.0004 across all 8 banks x 4 amplitudes, twins
        included. The statistic cannot distinguish core from Haar because it cannot distinguish
        anything; it is arithmetic, not an experiment.

        §4/W1-K says "one full FT recipe from the 24 on top", i.e. actually TRAIN on the stamped
        model. Only gradient steps can move the stamp, which is what makes recovery a question.
        The fix is to route this arm through darwin_train (stamp -> ExactAdam fine-tune -> recover),
        and until then its rows stay under their own statistic name with this flag and are not a
        watermark result.

  W1-P  Merging (§4/W1-P). 2/3/4-way merges of the official fine-tunes: highway-aware merging
        (collisions resolved in the <=32-dim highway coordinates) against naive averaging and
        TIES. REFUTER: no gain over TIES at any arity.

Both use the swap-and-evaluate path W1-N proved and G2 anchored, so their perplexities are
commensurable with every other number in this program.

  pg run infra.aws_pg.jobs.darwin_w1_merge --args "--arm W1-P" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W
from infra.aws_pg.jobs.darwin_w1_surgery import highway_basis

SPEC_EXPECTATIONS = {
    "W1-K": ("recovery <= twin recovery after FT, or battery harm above CI at all amplitudes",
             "an in-core rank-1 stamp is recoverable after a full fine-tune at an amplitude that "
             "does not harm the battery"),
    "W1-P": ("no gain over TIES at any arity",
             "highway-aware merging beats naive averaging and TIES at some arity"),
}
AMPLITUDES = (0.001, 0.01, 0.1, 1.0)     # >=4 log-spaced doses (§3)
ARITIES = (2, 3, 4)


def ties_merge(deltas, density: float = 0.2):
    """TIES-merging: trim to the top-`density` magnitude entries, elect a sign by mass, then
    average only the entries agreeing with the elected sign. The published reference recipe."""
    import numpy as np
    stack = np.stack([np.asarray(d, dtype=np.float64) for d in deltas])
    flat = np.abs(stack).reshape(stack.shape[0], -1)
    k = max(1, int(density * flat.shape[1]))
    trimmed = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        idx = np.argpartition(flat[i], flat.shape[1] - k)[flat.shape[1] - k:]
        f = trimmed[i].reshape(-1)
        s = stack[i].reshape(-1)
        f[idx] = s[idx]
    sign = np.sign(trimmed.sum(axis=0))
    agree = (np.sign(trimmed) == sign) & (trimmed != 0)
    num = (trimmed * agree).sum(axis=0)
    den = agree.sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def highway_aware_merge(deltas, U):
    """Merge with collisions resolved in the highway coordinates.

    Each delta is split into its highway component (dW U Uᵀ) and its remainder. Remainders are
    averaged — they are close to orthogonal across fine-tunes, so averaging is near-lossless. The
    HIGHWAY coefficients are where the fine-tunes actually collide (W0-8: a rank-32 highway holds
    27.6% of write energy against a 1.6% random reference), so there they are SUMMED rather than
    averaged: averaging n colliding updates in a shared <=32-dim subspace shrinks every one of
    them by n, which is precisely the interference naive averaging suffers from.
    """
    import numpy as np
    Us = np.asarray(U, dtype=np.float64)
    hi, rest = [], []
    for d in deltas:
        A = np.asarray(d, dtype=np.float64)
        h = (A @ Us) @ Us.T
        hi.append(h)
        rest.append(A - h)
    return np.stack(hi).sum(axis=0) + np.stack(rest).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(SPEC_EXPECTATIONS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="q_proj,o_proj,gate_proj,up_proj,v_proj,k_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--n-ft", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        types, layers, a.n_seq, a.n_ft = types[:1], layers[:2], 4, 2

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers,
                                                       _load_full_pg1, _ppl_blankimg)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    have_wide = len(list(pathlib.Path(bank_dir).glob("*__highway_top256.npy"))) > 0
    repos = W.ft_cloud(api)[:a.n_ft]
    shas = {r: api.model_info(r).sha for r in repos}

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "fine_tunes": repos,
                "fine_tune_shas": shas, "types": types, "layers": layers, "rank": a.rank,
                "amplitudes": list(AMPLITUDES), "arities": list(ARITIES), "n_seq": a.n_seq,
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    dl = _decoder_layers(model)
    mods = {}
    for t in types:
        for L in layers:
            parent = dl[L].self_attn if t.startswith(("q", "k", "v", "o")) else dl[L].mlp
            mods[(t, L)] = getattr(parent, t)
    originals = {k: m.weight.detach().clone() for k, m in mods.items()}
    base_np = {k: v.to(torch.float64).cpu().numpy() for k, v in originals.items()}

    def swap(fn):
        for key, mod in mods.items():
            mod.weight.data.copy_(torch.as_tensor(fn(key), dtype=mod.weight.dtype,
                                                  device=mod.weight.device))

    def restore():
        for k, mod in mods.items():
            mod.weight.data.copy_(originals[k])

    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    atlas.add("base_ppl", value=base_ppl, model_id=PG1_REPO, model_sha=base_sha,
              flags={"n_seq": a.n_seq})
    print(f"[{a.arm}] base ppl {base_ppl:.5f}", flush=True)

    if a.arm == "W1-K":
        ft = repos[0]
        w_ft = W._load_decoder(ft, shas[ft])
        seen = set()
        for bid in D.bank_ids_for("q_proj"):
            for amp in AMPLITUDES:
                # Rank-1 stamp u vᵀ with u INSIDE the bank: the recoverable object is the
                # coefficient of that direction, and the twin banks give the matched null.
                stamps = {}
                for key in mods:
                    U = W.load_bank(bank_dir, key[0], key[1], bid)
                    rng = np.random.default_rng(H.cell_seed(a.seed, bid, key[0], key[1]))
                    z = rng.standard_normal(U.shape[1]); z /= np.linalg.norm(z)
                    u = U @ z
                    v = rng.standard_normal(base_np[key].shape[0]); v /= np.linalg.norm(v)
                    stamps[key] = (u, v)
                # Stamp, then apply the fine-tune's delta on top.
                def eff(key, _amp=amp, _st=stamps):
                    u, v = _st[key]
                    dW = w_ft[W.matrix_key(key[0], key[1])].to(torch.float64).numpy() - base_np[key]
                    return base_np[key] + _amp * np.outer(v, u) + dW
                swap(eff)
                ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                # Recovery: correlation between the planted direction and the residual's rank-1
                # component along it, averaged over cells.
                recs = []
                for key in mods:
                    u, v = stamps[key]
                    Wn = mods[key].weight.detach().to(torch.float64).cpu().numpy()
                    dW = w_ft[W.matrix_key(key[0], key[1])].to(torch.float64).numpy() - base_np[key]
                    resid = Wn - base_np[key] - dW
                    recs.append(float(v @ resid @ u / (amp * (np.linalg.norm(v) ** 2)
                                                       * (np.linalg.norm(u) ** 2) + 1e-30)))
                restore()
                atlas.add_ci("watermark_recovery_TAUTOLOGICAL", recs, bank_id=bid,
                             dose=float(amp), incumbent_id="twin_band", model_id=ft,
                             model_sha=shas[ft],
                             flags={"amplitude": amp, "ppl": ppl, "base_ppl": base_ppl,
                                    "battery_harm_ppl": ppl - base_ppl, "n_cells": len(recs),
                                    "VOID": "recovery is 1.0 by construction (defect D21): the "
                                            "delta is ADDED arithmetically, never trained, so the "
                                            "residual IS the stamp. Not a watermark result."})
                atlas.add("watermark_battery_harm", value=ppl - base_ppl, bank_id=bid,
                          dose=float(amp), incumbent_id="twin_band", model_id=ft,
                          model_sha=shas[ft], flags={"amplitude": amp, "ppl": ppl})
                seen.add(bid)
                print(f"[W1-K] {bid} amp={amp}: recovery={np.mean(recs):.4f} "
                      f"harm={ppl-base_ppl:+.4f}", flush=True)
            D.s3_sync(out, a.s3_prefix)
        D.check_controls(seen, arm="W1-K")
        gates.record("W1K_completed", True, n_rows=atlas.n_rows, banks=sorted(seen))
    else:
        deltas = {}
        for r in repos:
            w = W._load_decoder(r, shas[r])
            deltas[r] = {k: w[W.matrix_key(k[0], k[1])].to(torch.float64).numpy() - base_np[k]
                         for k in mods}
            del w
        for arity in ARITIES:
            if arity > len(repos):
                continue
            group = repos[:arity]
            for scheme in ("naive_average", "ties", "highway_aware"):
                def eff(key, _g=group, _s=scheme):
                    ds = [deltas[r][key] for r in _g]
                    if _s == "naive_average":
                        m = np.stack(ds).mean(axis=0)
                    elif _s == "ties":
                        m = ties_merge(ds)
                    else:
                        m = highway_aware_merge(
                            ds, highway_basis(bank_dir, key[0], key[1], a.rank,
                                              have_wide=have_wide))
                    return base_np[key] + m
                swap(eff)
                ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                restore()
                atlas.add("merge_ppl", value=ppl, bank_id=scheme, dose=float(arity),
                          incumbent_id="ties", model_id=PG1_REPO, model_sha=base_sha,
                          flags={"arity": arity, "members": group, "base_ppl": base_ppl,
                                 "delta_ppl": ppl - base_ppl, "rank": a.rank})
                print(f"[W1-P] arity={arity} {scheme}: ppl={ppl:.5f}", flush=True)
            D.s3_sync(out, a.s3_prefix)
        gates.record("W1P_completed", True, n_rows=atlas.n_rows, arities=list(ARITIES))

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "base_ppl": base_ppl, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} base_ppl={base_ppl:.5f}")


if __name__ == "__main__":
    main()
