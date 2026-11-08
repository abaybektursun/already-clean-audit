"""NOETHER sweep — aligned bidirectional grafts on the HELMHOLTZ-1 pair: gauge drift vs damage.

Spec: studies/lineage/cross-lineage/research/2026-09-02_noether_spec.md. ONE server-side job:
minute-1 preflight -> baselines once (+ BROCA reproduction) -> streamed affine-map statistics over
FIT-A at every sweep layer (both directions from one pass) -> identity-map reproduction of BROCA's D
-> CONTROLS BEFORE CONTRAST at the band (random-matched, shuffled-fit, wrong-item-through-map, mean,
diag) -> null band + T frozen -> full-map sweep -> rank ladder at the frozen L_band -> FIT-B off-task
map -> exploratory aligned restoration -> verdict (noether_core). Fail-loud; metrics.json incremental.

  pg run infra.aws_pg.jobs.noether_sweep --type g6e.2xlarge,g6e.xlarge,g6e.4xlarge,g6e.8xlarge \
      --s3-prefix results/noether_sweep --budget-min 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

BROCA_ARTIFACT = pathlib.Path(__file__).resolve().parents[3] / \
    "studies/lineage/cross-lineage/results/broca_metrics.json"   # committed copy of the S3 artifact


class MultiHarvest:
    """Forward hooks on several decoder layers at once; captured[L] = that layer's output residual."""

    def __init__(self, layers, idxs):
        self.layers, self.idxs = layers, list(idxs)
        self.captured = {}
        self._h = []

    def __enter__(self):
        for L in self.idxs:
            def hook(m, a, out, L=L):
                self.captured[L] = out[0] if isinstance(out, tuple) else out
            self._h.append(self.layers[L].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-subject", type=int, default=5)
    ap.add_argument("--fit-per-subject", type=int, default=6)
    ap.add_argument("--sweep-stride", type=int, default=2)
    ap.add_argument("--band", default="10,12,14")
    ap.add_argument("--ident-check", default="10,14,20")
    ap.add_argument("--ranks", default="1,8,64,512")
    ap.add_argument("--fitb-chars", type=int, default=240_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip-fitb", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False       # mistake-ledger 3: parity of the fit stats
    torch.backends.cudnn.allow_tf32 = False
    from PIL import Image
    from datasets import load_dataset
    from infra.pg import noether_core as NC
    from infra.pg import text_eval as TE
    from infra.pg.diff_harvest import build_before_after
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.aws_pg.jobs.broca_sweep import PatchedLM

    assert torch.cuda.is_available()
    device = "cuda"
    band = [int(x) for x in a.band.split(",")]
    ident_layers = [int(x) for x in a.ident_check.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    broca = json.loads(BROCA_ARTIFACT.read_text())
    rep = {"object": "NOETHER sweep (aligned bidirectional grafts)", "n_per_subject": a.n_per_subject,
           "seed": a.seed, "band": band,
           "broca_artifact_sha1": hashlib.sha1(BROCA_ARTIFACT.read_bytes()).hexdigest()[:12]}
    per_item = {}
    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        (out / "per_item.json").write_text(json.dumps(per_item))
    _w()

    # ---- pair (validated HELMHOLTZ-1 construction; VLMs scored via blankimg) ----
    pg_after, pg_before, proc, paths, _i, _g = build_before_after(device, pg_key="PG", attn_impl="eager")
    lm_b = resolve_gemma2_layers(pg_before.get_submodule(paths["language_model"]))
    lm_a = resolve_gemma2_layers(pg_after.get_submodule(paths["language_model"]))
    n_layers = len(lm_b); d = lm_b[0].mlp.down_proj.out_features
    rep["bind"] = {"n_layers": n_layers, "d_model": d}
    assert n_layers == 26 and d == 2304, rep["bind"]
    sweep = sorted(set(list(range(0, n_layers, a.sweep_stride)) + [n_layers - 1]))
    for L in band + ident_layers:
        assert L in sweep, (L, sweep)
    rep["sweep_layers"] = sweep

    def acc(model, n=None):
        r = TE.mmlu_blankimg(model, proc, device, n_per_subject=n or a.n_per_subject)
        return r["acc"], r

    # ---- minute-1 preflight ----
    a_self, _ = acc(PatchedLM(pg_before, pg_before, lm_b, lm_b, 12), n=1)
    a_base, _ = acc(pg_before, n=1)
    assert abs(a_self - a_base) < 1e-9, f"preflight self-patch broke eval: {a_self} vs {a_base}"
    print(f"[NOETHER][preflight] self-patch identity OK (acc {a_base:.4f})", flush=True)

    # ---- baselines ONCE + BROCA reproduction (the known number) ----
    acc_before, rb = acc(pg_before); acc_after, ra = acc(pg_after)
    assert rb["per_item_hash"] == ra["per_item_hash"], "item pairing broke"
    eval_hashes = set(rb["per_item_hash"])
    per_item["before"] = rb["per_item"]; per_item["after"] = ra["per_item"]
    rep["baselines"] = {"acc_before": acc_before, "acc_after": acc_after, "n": rb["n"]}
    bb = broca["baselines"]
    assert abs(acc_before - bb["acc_before"]) < 1e-9 and abs(acc_after - bb["acc_after"]) < 1e-9, \
        f"pair baselines do not reproduce BROCA: {acc_before},{acc_after} vs {bb}"
    print(f"[NOETHER] baselines reproduce BROCA: before {acc_before:.4f} after {acc_after:.4f}", flush=True)
    _w()

    def D_of(wrapper, key):
        ap_, r = acc(wrapper)
        per_item[key] = r["per_item"]
        return NC.damage_fraction(ap_, acc_after, acc_before), ap_

    # ---- FIT-A corpus: MMLU rows[5:5+k] per subject, eval-format prompts, labels never read ----
    ds = load_dataset(TE.MMLU_REPO, "all")
    from collections import defaultdict
    shots_by, test_by = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        shots_by[r["subject"]].append(r)
    for r in ds["test"]:
        test_by[r["subject"]].append(r)
    fit_prompts, fit_held = [], []
    for subj in sorted(test_by):
        prefix = TE._subject_header(subj) + "".join(
            TE._q_block(s["question"], s["choices"], s["answer"]) for s in shots_by[subj][:5])
        for r in test_by[subj][a.n_per_subject:a.n_per_subject + a.fit_per_subject]:
            h = hashlib.sha1(r["question"][:64].encode()).hexdigest()
            assert h[:12] not in eval_hashes, "FIT-A item collides with the eval set"
            fit_prompts.append(prefix + TE._q_block(r["question"], r["choices"]))
            fit_held.append(int(h[0], 16) < 4)
    rep["fit_a"] = {"n_prompts": len(fit_prompts), "n_held": int(sum(fit_held)),
                    "sha1": hashlib.sha1("\n".join(fit_prompts).encode()).hexdigest()[:12]}
    gray = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    tok = proc.tokenizer; tok.padding_side = "left"
    img_id = _i if isinstance(_i, int) else int(_i)

    def stream_stats(prompts, held_flags, layers, micro=4, with_shuffle=True, with_class=True):
        """One pass over prompts, both models, all `layers`: streamed RidgeStats per layer.
        X = BROKEN (after) residual, Y = HEALTHY (before) residual, real positions only."""
        S = {L: {"fit": NC.RidgeStats(d), "held": NC.RidgeStats(d)} for L in layers}
        if with_shuffle:
            for L in layers:
                S[L]["fit_shuf"] = NC.RidgeStats(d)
        if with_class:
            for L in layers:
                S[L]["held_img"] = NC.RidgeStats(d); S[L]["held_txt"] = NC.RidgeStats(d)
        n_tok = 0
        for s in range(0, len(prompts), micro):
            chunk = prompts[s:s + micro]; hf = held_flags[s:s + micro]
            inp = proc(text=chunk, images=[gray] * len(chunk), return_tensors="pt",
                       padding="longest").to(device)
            pv = inp["pixel_values"].to(torch.bfloat16)
            with MultiHarvest(lm_a, layers) as ha, torch.inference_mode():
                pg_after(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=pv, logits_to_keep=1)
                cap_a = {L: ha.captured[L] for L in layers}
            with MultiHarvest(lm_b, layers) as hb, torch.inference_mode():
                pg_before(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                          pixel_values=pv, logits_to_keep=1)
                cap_b = {L: hb.captured[L] for L in layers}
            am = inp["attention_mask"].bool()
            is_img = (inp["input_ids"] == img_id) & am
            held_m = torch.tensor(hf, device=device, dtype=torch.bool)[:, None].expand_as(am)
            for L in layers:
                for split, m in (("fit", am & ~held_m), ("held", am & held_m),
                                 ("held_img", am & held_m & is_img), ("held_txt", am & held_m & ~is_img)):
                    if split not in S[L]:
                        continue
                    X = cap_a[L][m].float(); Y = cap_b[L][m].float()
                    if X.shape[0] == 0:
                        continue
                    S[L][split].add_products(X.shape[0], X.sum(0).cpu().numpy(), Y.sum(0).cpu().numpy(),
                                             (X.T @ X).cpu().numpy(), (X.T @ Y).cpu().numpy(),
                                             (Y.T @ Y).cpu().numpy())
                    if split == "fit" and "fit_shuf" in S[L]:
                        Xs = torch.roll(X, 997, 0)          # correspondence destroyed, all-real tokens
                        S[L]["fit_shuf"].add_products(X.shape[0], Xs.sum(0).cpu().numpy(),
                                                      Y.sum(0).cpu().numpy(), (Xs.T @ Xs).cpu().numpy(),
                                                      (Xs.T @ Y).cpu().numpy(), (Y.T @ Y).cpu().numpy())
            n_tok += int(am.sum())
            del cap_a, cap_b
        return S, n_tok

    S, n_tok = stream_stats(fit_prompts, fit_held, sweep)
    rep["fit_a"]["n_tokens"] = n_tok
    print(f"[NOETHER] FIT-A streamed: {n_tok} tokens over {len(sweep)} layers", flush=True)

    maps = {}      # maps[L] = {"W": np, "b": np, "mx", "my", "lam", ...}   (broken -> healthy)
    maps_rev = {}  # healthy -> broken (for the exploratory restoration rows)
    for L in sweep:
        sel = NC.select_lambda(S[L]["fit"], S[L]["held"])
        mx, my, *_ = S[L]["fit"].centered()
        base = NC.baseline_mse(S[L]["held"])
        cls = {}
        for c in ("held_img", "held_txt"):
            if S[L][c].n > d:
                cls[c] = NC.held_mse(sel["W"], sel["b"], S[L][c])
        spec = NC.delta_spectrum(sel["W"])
        maps[L] = {"W": sel["W"], "b": sel["b"], "mx": mx, "my": my, "lam_rel": sel["lam_rel"],
                   "held_mse": sel["held_mse"], "baseline": base, "class_mse": cls,
                   "delta_top_sv": spec[:64].tolist(), "delta_fro": float(np.linalg.norm(spec))}
        np.savez(out / "maps" / f"L{L}_broken_to_healthy.npz", W=sel["W"].astype(np.float32),
                 b=sel["b"].astype(np.float32), lam_rel=sel["lam_rel"])
        print(f"[NOETHER][map] L{L}: lam={sel['lam_rel']} held_mse={sel['held_mse']:.4g} "
              f"(identity {base['identity']:.4g}, mean {base['mean_only']:.4g}) class={cls}", flush=True)
    rep["maps"] = {str(L): {k: v for k, v in m.items() if k not in ("W", "b", "mx", "my")}
                   for L, m in maps.items()}
    _w()

    def as_transform(W, b):
        Wt = torch.tensor(W, dtype=torch.float32, device=device)
        bt = torch.tensor(b, dtype=torch.float32, device=device)
        def f(cap):
            return (cap.float() @ Wt + bt).to(cap.dtype)
        return f

    def graft_D(L, transform=None, mismatch=False, key=""):
        # broken donor -> healthy recipient (the ablation direction), optional map
        return D_of(PatchedLM(pg_after, pg_before, lm_a, lm_b, L, mismatch=mismatch,
                              transform=transform), key)

    # ---- identity-map reproduction of BROCA's D (VOID gate) ----
    ident = {}
    for L in ident_layers:
        D_I, _ = graft_D(L, key=f"I_L{L}")
        ref = broca["curve"][str(L)]["ablation_D"]
        ident[str(L)] = {"D_I": D_I, "broca_D": ref, "ok": bool(abs(D_I - ref) < 0.02)}
        print(f"[NOETHER][ident] L{L}: D_I={D_I:.4f} vs BROCA {ref:.4f} ok={ident[str(L)]['ok']}", flush=True)
    rep["identity_check"] = ident
    assert all(v["ok"] for v in ident.values()), "identity graft does not reproduce BROCA — wiring VOID"
    _w()

    # ---- CONTROLS BEFORE CONTRAST at the band ----
    D_I, D_nulls, D_wrong, D_triv = {}, {}, {}, {}
    for L in band:
        if str(L) in ident:
            D_I[L] = ident[str(L)]["D_I"]
        else:
            D_I[L], _ = graft_D(L, key=f"I_L{L}")
        m = maps[L]
        Wr, br = NC.random_matched_delta(m["W"], m["mx"], m["my"], d, seed=a.seed)
        Ws, bs = NC.ridge_solve(S[L]["fit_shuf"], m["lam_rel"])
        Wm, bm = NC.mean_map(S[L]["fit"]); Wd, bd = NC.diag_map(S[L]["fit"])
        D_nulls[L] = {"random_matched": graft_D(L, as_transform(Wr, br), key=f"rand_L{L}")[0],
                      "shuffled_fit": graft_D(L, as_transform(Ws, bs), key=f"shuf_L{L}")[0]}
        D_wrong[L] = graft_D(L, as_transform(m["W"], m["b"]), mismatch=True, key=f"wrong_L{L}")[0]
        D_triv[L] = {"mean": graft_D(L, as_transform(Wm, bm), key=f"mean_L{L}")[0],
                     "diag": graft_D(L, as_transform(Wd, bd), key=f"diag_L{L}")[0]}
        print(f"[NOETHER][ctrl] L{L}: D_I={D_I[L]:.3f} nulls={D_nulls[L]} wrong={D_wrong[L]:.3f} "
              f"triv={D_triv[L]}", flush=True)
        rep["controls"] = {"D_I": D_I, "D_nulls": D_nulls, "D_wrong_item_through_map": D_wrong,
                           "D_trivial_maps": D_triv}
        _w()
    band_w = NC.null_recovery_band(D_I, D_nulls)
    T = NC.threshold(band_w)
    rep["null_band"] = band_w; rep["threshold"] = T
    print(f"[NOETHER] null recovery band={band_w:.4f} -> T={T:.4f} (FROZEN before contrast)", flush=True)
    _w()

    # ---- full-map sweep (the contrast) ----
    curve = {}
    for L in sweep:
        m = maps[L]
        D_full, acc_p = graft_D(L, as_transform(m["W"], m["b"]), key=f"full_L{L}")
        curve[L] = {"D_full": D_full, "acc_patched": acc_p,
                    "D_identity_broca": broca["curve"].get(str(L), {}).get("ablation_D")}
        if L in D_I:
            curve[L]["D_identity"] = D_I[L]; curve[L]["G"] = NC.gauge_fraction(D_I[L], D_full)
        print(f"[NOETHER] L{L}: D_full={D_full:.4f}" + (f" G={curve[L]['G']:.3f}" if L in D_I else ""),
              flush=True)
        rep["curve"] = {str(k): v for k, v in sorted(curve.items())}
        _w()
    D_fullb = {L: curve[L]["D_full"] for L in band}
    for L in band:
        rep["curve"][str(L)]["bootstrap"] = NC.bootstrap_gauge(
            per_item["before"], per_item["after"], per_item[f"I_L{L}"], per_item[f"full_L{L}"],
            seed=a.seed)
    Ls = sweep; rep["L_loss"] = NC.persistence_depth(Ls, [curve[L]["D_full"] for L in Ls])
    _w()

    # ---- verdict (frozen §6) ----
    verdict = NC.noether_verdict(band, D_I, D_fullb, D_nulls, D_wrong, T)
    verdict["trivial_modifier"] = {str(L): {"G_diag_over_G_full":
                                            (NC.gauge_fraction(D_I[L], D_triv[L]["diag"]) /
                                             max(verdict["G"][L], 1e-6))} for L in band}
    rep["verdict"] = verdict
    print(f"[NOETHER] VERDICT: {json.dumps(verdict, default=str)}", flush=True)
    _w()

    # ---- rank ladder at the frozen L_band = argmax G over the band ----
    L_band = max(band, key=lambda L: verdict["G"][L])
    m = maps[L_band]; G_full = verdict["G"][L_band]
    ladder = {"L_band": L_band, "G_full": G_full, "fit": {}, "random": {}}
    for k in ranks:
        Wk, bk = NC.truncate_delta(m["W"], m["mx"], m["my"], k)
        Dk, _ = graft_D(L_band, as_transform(Wk, bk), key=f"rank{k}_L{L_band}")
        Wr, br = NC.random_matched_delta(m["W"], m["mx"], m["my"], k, seed=a.seed + k)
        Dr, _ = graft_D(L_band, as_transform(Wr, br), key=f"rank{k}rand_L{L_band}")
        ladder["fit"][str(k)] = {"D": Dk, "G": NC.gauge_fraction(D_I[L_band], Dk)}
        ladder["random"][str(k)] = {"D": Dr, "G": NC.gauge_fraction(D_I[L_band], Dr)}
        print(f"[NOETHER][ladder] L{L_band} k={k}: G_fit={ladder['fit'][str(k)]['G']:.3f} "
              f"G_rand={ladder['random'][str(k)]['G']:.3f}", flush=True)
        rep["ladder"] = ladder; _w()
    ks = sorted(ranks); k_star = NC.rank_knee(ks, [ladder["fit"][str(k)]["G"] for k in ks], G_full)
    ladder["k_star"] = k_star
    ladder["rank_claimed"] = bool(k_star is not None and
                                  ladder["random"][str(k_star)]["G"] < 0.5 * G_full)
    rep["ladder"] = ladder; _w()

    # ---- FIT-B off-task map (wikitext under the blank-image forward) at two band layers ----
    if not a.skip_fitb:
        wt = load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG, split="test")
        text = "\n\n".join(t for t in wt["text"] if t.strip())[:a.fitb_chars]
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        Lb = [L for L in (10, 14) if L in sweep]
        Sb, nb = stream_stats(chunks, [False] * len(chunks), Lb, with_shuffle=False, with_class=False)
        fitb = {"n_tokens": nb, "n_chunks": len(chunks)}
        for L in Lb:
            Wb, bb_ = NC.ridge_solve(Sb[L]["fit"], maps[L]["lam_rel"])
            Db, _ = graft_D(L, as_transform(Wb, bb_), key=f"fitb_L{L}")
            fitb[str(L)] = {"D": Db, "G": NC.gauge_fraction(D_I[L], Db) if L in D_I else None,
                            "held_mse_on_fitA": NC.held_mse(Wb, bb_, S[L]["held"])}
            print(f"[NOETHER][fitB] L{L}: D={Db:.4f}", flush=True)
        rep["fit_b"] = fitb; _w()

    # ---- exploratory: aligned RESTORATION (healthy state mapped INTO the broken format) ----
    restor = {}
    for L in (10, 14):
        sel = NC.select_lambda(S[L]["fit"].swapped(), S[L]["held"].swapped())
        w = PatchedLM(pg_before, pg_after, lm_b, lm_a, L, transform=as_transform(sel["W"], sel["b"]))
        ap_, r = acc(w); per_item[f"restore_L{L}"] = r["per_item"]
        from infra.pg.broca_core import restoration_fraction
        restor[str(L)] = {"R_aligned": restoration_fraction(ap_, acc_after, acc_before),
                          "R_raw_broca": broca["curve"][str(L)]["R"], "lam_rel": sel["lam_rel"]}
        print(f"[NOETHER][explore] aligned restoration L{L}: R={restor[str(L)]['R_aligned']:.4f}", flush=True)
    rep["exploratory_aligned_restoration"] = restor

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("NOETHER_OK " + json.dumps({"verdict": verdict["verdict"], "T": T, "L_band": L_band,
                                      "k_star": k_star, "L_loss": rep["L_loss"]}), flush=True)


if __name__ == "__main__":
    main()
