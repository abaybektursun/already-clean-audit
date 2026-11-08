"""W9 §9.4 UNBLIND (run ONCE, after the freeze tag + controls gate).

Computes the two pre-registered readouts on the REAL Δa = a(PG2-LM) − a(base-gemma2), applying
the frozen §6 rules + the N3 floor measured by w9_controls (read from S3, NOT recomputed):

  ρ_novel(L, width, seed): fraction of Diff-SAE decoder directions with out-of-span energy
    (method i, vs the frozen N3 floor) AND surviving the §2.2 direct-activation cross-check.
    Method (ii) max-cosine-to-base reported alongside. Bootstrap CI + across-seed variance.

  R_KL (the crux): behavioural KL recovered by patching a reconstruction back through the
    after-model at L, for three reconstructions — Diff-SAE recon, crosscoder recon, crosscoder
    ERROR term — vs the before-patch baseline gap. A residual write-hook substitutes the kept
    positions at layer L; KL is accumulated per micro-batch (vocab too large to store).

    python -m infra.aws_pg.jobs.w9_unblind --out <dir> \
        --controls-s3 s3://.../results/w9_controls/out/controls.json \
        [--widths 16,32] [--seeds 0,1,2] [--rkl-examples 200] [--no-rkl] [--no-base-dict]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


class ResidualPatch:
    """Forward hook that REPLACES the after-model layer-L output residual at selected (flattened
    B*T) positions with provided values. Set `.keep` (bool, B*T) and `.vals` ((n_keep, d)) before
    each forward; the hook writes vals into output[0][keep]. None on either ⇒ pass-through."""
    def __init__(self, layers, layer_idx):
        self.layer = layers[layer_idx]
        self.keep = None
        self.vals = None
        self._handle = None

    def _hook(self, module, args, output):
        if self.keep is None or self.vals is None:
            return output
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        shp = hs.shape
        flat = hs.reshape(-1, shp[-1])
        flat = flat.clone()
        flat[self.keep] = self.vals.to(flat.dtype)
        hs2 = flat.reshape(shp)
        if isinstance(output, tuple):
            return (hs2,) + tuple(output[1:])
        return hs2

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove(); self._handle = None
        return False


def _rkl_pass(pg_after, mm, layer_idx, patch_buffers, device, max_examples, micro=4):
    """One R_KL pass. patch_buffers: dict name -> (N_kept, d) float tensor aligned to the
    harvest keep order (image|text positions), or None for the 'true' (no-patch) run. KL is
    measured at kept TEXT positions only (the generation-relevant readout) between the TRUE
    after-forward and each patched variant. Returns {name: mean_kl}."""
    import torch
    from infra.pg.gemma_scope import resolve_gemma2_layers
    a_layers = resolve_gemma2_layers(pg_after.model.language_model)

    ii = mm["input_ids"]; am = mm["attention_mask"]; pv = mm["pixel_values"]
    is_image = mm["is_image"]; is_text_keep = mm["is_text_keep"]
    keep_all = (is_image | is_text_keep)                       # the SAE-trained domain
    B_total = ii.shape[0]
    n_ex = min(max_examples, B_total)

    names = list(patch_buffers.keys())
    per_pos = {n: [] for n in names}                           # per-text-position KL, for bootstrap CIs
    ptr = 0                                                    # running index into keep-order buffers

    patch = ResidualPatch(a_layers, layer_idx)
    with patch:
        for s in range(0, n_ex, micro):
            ids = ii[s:s+micro]; a_ = am[s:s+micro]; p_ = pv[s:s+micro]
            km_all = keep_all[s:s+micro].reshape(-1)           # (b*t,) bool — SAE domain
            tk = is_text_keep[s:s+micro].reshape(-1)           # (b*t,) bool — KL readout positions
            n_keep_chunk = int(km_all.sum().item())

            # TRUE (no patch).
            patch.keep = None; patch.vals = None
            with torch.inference_mode():
                logits_true = pg_after(input_ids=ids, attention_mask=a_, pixel_values=p_).logits
            lt = logits_true.reshape(-1, logits_true.shape[-1])[tk].float()
            logp_true = torch.log_softmax(lt, dim=-1)
            p_true = logp_true.exp()

            # each variant: patch the chunk's keep positions with its buffer slice.
            for n in names:
                buf = patch_buffers[n]
                vals = buf[ptr:ptr+n_keep_chunk].to(device)
                patch.keep = km_all; patch.vals = vals
                with torch.inference_mode():
                    logits_v = pg_after(input_ids=ids, attention_mask=a_, pixel_values=p_).logits
                lv = logits_v.reshape(-1, logits_v.shape[-1])[tk].float()
                logq = torch.log_softmax(lv, dim=-1)
                kl_pos = (p_true * (logp_true - logq)).sum(-1)   # (n_text_pos_chunk,)
                per_pos[n].append(kl_pos.cpu())
            ptr += n_keep_chunk

    import torch as _t
    per_pos = {n: _t.cat(v).numpy() if v else __import__("numpy").zeros(0) for n, v in per_pos.items()}
    n_pos_total = int(sum(len(v) for v in [per_pos[names[0]]]))
    means = {n: float(per_pos[n].mean()) if per_pos[n].size else float("nan") for n in names}
    return means, per_pos, n_pos_total, ptr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True, help="S3 URI of the frozen controls.json")
    ap.add_argument("--widths", default="16,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rkl-examples", type=int, default=200)
    ap.add_argument("--no-rkl", action="store_true")
    ap.add_argument("--no-base-dict", action="store_true")
    ap.add_argument("--layers", default="", help="comma list to restrict layers (per-layer unblind); default all")
    ap.add_argument("--freeze-sha", default="", help="recorded prereg commit SHA")
    ap.add_argument("--pair", default="base-pt", choices=["base-pt", "pt-mix"],
                    help="before/after axis: base-pt (W9 Gemma→PT) or pt-mix (W6 PT→Mix)")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import diff_harvest as H
    from infra.pg import diff_sae as DS
    from infra.pg import diff_metrics as DM
    H.hf_online_setup()                    # HF_TOKEN + online (gated models + dataset streaming)
    H.ensure_wandb()                       # dictionary_learning.training imports wandb at top

    # frozen controls (N3 floors + rules) — read, never recompute.
    ctl_local = out / "controls_frozen.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctl_local), "--region", "us-east-1"], check=True)
    controls = json.loads(ctl_local.read_text())

    device = "cuda"
    print(f"[W9-UNB] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"widths={widths} seeds={seeds} rkl={'off' if args.no_rkl else args.rkl_examples}")

    layers_run = [int(x) for x in args.layers.split(",")] if args.layers else list(H.LAYERS)
    print(f"[W9-UNB] pair={args.pair} layers={layers_run}")

    pg_after, pg_before, proc, paths, img_tok, gate = H.build_pair(args.pair, device)
    mm = H.build_mm_batch(proc, img_tok, device)

    result = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "pair": args.pair, "freeze_sha": args.freeze_sha, "controls_s3": args.controls_s3,
        "widths": widths, "seeds": seeds, "layers_run": layers_run, "swap_gate": gate,
        "frozen_rules": controls.get("frozen_rules"),
        "layers": {},
    }

    for L in layers_run:
        print(f"\n[W9-UNB] ===== LAYER {L} =====")
        acts_before, acts_after, is_img, held = H.harvest_paired_layer(pg_after, pg_before, L, mm, device)
        N, d = acts_before.shape
        delta = (acts_after - acts_before)
        # Pre-registered discovery↔confirmation firewall (§4): TRAIN the dictionary + fit the
        # before-PCA span on the ~75% discovery half; run the binding §2.2 cross-check on the
        # SEALED ~25% confirmation half so ρ_novel is OUT-OF-SAMPLE (closes the in-sample leak).
        # Routed through the single tested DM.enforce_split (asserts both halves non-empty).
        split = DM.enforce_split(held, name=f"L{L}", text=(~is_img.numpy().astype(bool)))
        disc_t = torch.from_numpy(split["disc_mask"])
        conf_idx = split["conf_idx"]; conf_txt_idx = split["conf_text_idx"]
        delta_disc = delta[disc_t]
        print(f"[W9-UNB][L{L}] real Δa N={N} d={d} img_frac={float(is_img.float().mean()):.3f} "
              f"disc={split['n_disc']} conf={conf_idx.size} conf_text={conf_txt_idx.size}")

        U, pca_info = DM.before_pca_basis(acts_before[disc_t], var_threshold=0.95, seed=1337)

        base_dirs = None; base_info = None
        if not args.no_base_dict:
            from infra.aws_pg.jobs.w9_controls import _pick_layer_sae
            from infra.pg.gemma_scope import JumpReLUSAE
            choice, is_exact = _pick_layer_sae(L)
            base_dirs = JumpReLUSAE.from_repo(choice).W_dec.detach().cpu().float()
            base_info = {"rel_path": choice.rel_path, "exact_layer": bool(is_exact)}

        # crosscoder (for R_KL crosscoder-recon / error terms) — one seed, frozen regime.
        cc_recon_after = None
        if not args.no_rkl:
            ae_cc, _ = H.train_crosscoder(acts_before, acts_after, L, seed=0, device=device)
            ae_cc = ae_cc.to(device).eval()
            with torch.inference_mode():
                recon = []
                for s in range(0, N, 8192):
                    b = acts_before[s:s+8192].to(device); a = acts_after[s:s+8192].to(device)
                    st = torch.stack([b, a], dim=1)
                    rec = ae_cc.decode(ae_cc.encode(st))[:, 1]      # after-stream recon
                    recon.append(rec.cpu())
            cc_recon_after = torch.cat(recon, 0)                    # (N, d)
            ae_cc = ae_cc.to("cpu"); del ae_cc; torch.cuda.empty_cache()

        layer_rec = {"n_tokens": int(N), "d_model": int(d),
                     "image_fraction": float(is_img.float().mean()),
                     "n_disc": int(split["n_disc"]), "n_conf": int(conf_idx.size),
                     "n_conf_text": int(conf_txt_idx.size),
                     "pca": pca_info, "base_dict": base_info, "grid": {}}

        # confirmation-half cross-check token sets (sealed), subsampled for permutation cost.
        rng_cc = np.random.default_rng(1337)
        conf_sel = conf_idx[rng_cc.choice(conf_idx.size, min(20000, conf_idx.size), replace=False)]
        ab_conf = acts_before.numpy()[conf_sel]; aa_conf = acts_after.numpy()[conf_sel]
        if conf_txt_idx.size >= 500:
            ct_sel = conf_txt_idx[rng_cc.choice(conf_txt_idx.size, min(20000, conf_txt_idx.size), replace=False)]
            ab_ct = acts_before.numpy()[ct_sel]; aa_ct = acts_after.numpy()[ct_sel]
        else:
            ab_ct = None

        def _crosscheck(dirs_np, cand_idx, ab, aa):
            survive_full = np.zeros(dirs_np.shape[0], dtype=bool)
            summ = {"n_candidates": int(cand_idx.size), "n_survive": 0}
            if cand_idx.size > 0:
                cc = DM.crosscheck_directions(dirs_np[cand_idx], ab, aa,
                                              n_perm=1000, seed=1337, fdr_q=0.05, max_candidates=256)
                survived = cc["survives_crosscheck"]
                survive_full[cand_idx[:survived.size]] = survived
                summ = {"n_candidates": int(cc["n_candidates"]), "n_survive": int(cc["n_survive"]),
                        "rand_ratio_median": float(cc["rand_ratio_median"])}
            return survive_full, summ

        for expansion in widths:
            floor95 = float(controls["layers"][str(L)]["widths"][str(expansion)]["n3_out_of_span_floor_pct95"])
            rho_by_seed = []
            for seed in seeds:
                ae, cfg = DS.train_diff_sae(delta_disc, L, seed=seed, device=device, expansion=expansion)
                q = DS.diff_sae_quality(ae, delta_disc, device)
                dirs = DS.decoder_directions(ae)                    # (dict, d) unit
                dirs_np = dirs.numpy()
                e_real = DM.out_of_span_energy(dirs, U).numpy()     # method i (disc-PCA)
                dict_size = e_real.shape[0]
                cand_idx = np.where(e_real > floor95)[0]
                cand_idx = cand_idx[np.argsort(-e_real[cand_idx])][:256]

                # PRIMARY: cross-check on the SEALED confirmation half → out-of-sample ρ_novel.
                survive_full, cc_summary = _crosscheck(dirs_np, cand_idx, ab_conf, aa_conf)
                mask, n_novel, rho = DM.novel_flags(e_real, floor95, survive_full)
                rho_by_seed.append(rho)
                # STRATUM: text-only confirmation tokens (the behaviourally-relevant population).
                cc_text = None
                if ab_ct is not None:
                    _, cc_text = _crosscheck(dirs_np, cand_idx, ab_ct, aa_ct)

                maxcos = None
                if base_dirs is not None:
                    maxcos = DM.max_cosine_to_base(dirs, base_dirs).numpy()
                np.savez_compressed(
                    out / f"features_L{L}_x{expansion}_s{seed}.npz",
                    out_of_span_energy=e_real, novel_flag=mask,
                    max_cos_base=(maxcos if maxcos is not None else np.zeros(0)))
                layer_rec["grid"][f"x{expansion}_s{seed}"] = {
                    "rho_novel": rho, "n_novel": n_novel, "dict_size": int(dict_size),
                    "fvu": q["fvu"], "dead_latent_frac": q["dead_latent_frac"], "mean_l0": q["mean_l0"],
                    "out_of_span_floor_pct95": floor95,
                    "out_of_span_mean_ci": DM.bootstrap_ci(e_real, n_boot=1000, seed=1337),
                    "crosscheck_conf": cc_summary, "crosscheck_conf_text": cc_text,
                    "method_ii_maxcos_mean": (float(np.mean(maxcos)) if maxcos is not None else None),
                }
                print(f"[W9-UNB][L{L} x{expansion} s{seed}] rho_novel(conf)={rho:.5f} n_novel={n_novel} "
                      f"cc_conf={cc_summary['n_survive']}/{cc_summary['n_candidates']} "
                      f"cc_text={(cc_text or {}).get('n_survive')}/{(cc_text or {}).get('n_candidates')} fvu={q['fvu']:.3f}")
                del ae; torch.cuda.empty_cache()

            rs = np.array(rho_by_seed)
            layer_rec["grid"][f"x{expansion}_rho_seed_mean"] = float(rs.mean())
            layer_rec["grid"][f"x{expansion}_rho_seed_std"] = float(rs.std())
            layer_rec["grid"][f"x{expansion}_rho_seed_ci"] = DM.bootstrap_ci(rs, n_boot=1000, seed=1337)

        # ---- R_KL (the crux): bootstrap CI + Diff-SAE at BOTH widths (disc-trained) ----
        if not args.no_rkl:
            buffers = {
                "baseline_before": acts_before,                              # a_before
                "ceiling_after": acts_after,                                 # a_after (KL→0 sanity)
                "crosscoder_recon": cc_recon_after,                          # recon of a_after
                "crosscoder_error": acts_before + (acts_after - cc_recon_after),
            }
            for expansion in widths:                                          # Diff-SAE recon per width
                ae_d, _ = DS.train_diff_sae(delta_disc, L, seed=0, device=device, expansion=expansion)
                ae_d = ae_d.to(device).eval()
                with torch.inference_mode():
                    dhat = [ae_d.decode(ae_d.encode(delta[s:s+8192].to(device), use_threshold=True)).cpu()
                            for s in range(0, N, 8192)]
                buffers[f"diffsae_x{expansion}"] = acts_before + torch.cat(dhat, 0)
                ae_d = ae_d.to("cpu"); del ae_d; torch.cuda.empty_cache()

            means, per_pos, n_pos, _ = _rkl_pass(pg_after, mm, L, buffers, device, args.rkl_examples)
            base = per_pos["baseline_before"]
            rkl = {"n_text_positions": int(n_pos), "rkl_examples": args.rkl_examples,
                   "kl_baseline_gap": means["baseline_before"], "kl_ceiling_after": means["ceiling_after"],
                   "R_KL": {}, "raw_kl_means": means}
            for n in buffers:
                if n in ("baseline_before", "ceiling_after"):
                    continue
                rkl["R_KL"][n] = {"point": DM.r_kl(means["baseline_before"], means[n]),
                                  "ci": DM.r_kl_bootstrap_ci(base, per_pos[n], n_boot=1000, seed=1337)}
            layer_rec["R_KL"] = rkl
            dd = rkl["R_KL"].get(f"diffsae_x{widths[0]}", {}).get("point")
            ccr = rkl["R_KL"]["crosscoder_recon"]["point"]; cce = rkl["R_KL"]["crosscoder_error"]["point"]
            print(f"[W9-UNB][L{L}] R_KL diffsae_x{widths[0]}={dd:.3f} cc_recon={ccr:.3f} cc_error={cce:.3f} "
                  f"(ceiling={means['ceiling_after']:.4g} base_gap={means['baseline_before']:.4g} npos={n_pos})")

        result["layers"][str(L)] = layer_rec
        del acts_before, acts_after, delta; torch.cuda.empty_cache()

    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print("\n[W9-UNB] DONE — metrics.json written.")
    for L, lr in result["layers"].items():
        rho = lr["grid"].get(f"x{widths[0]}_rho_seed_mean")
        rk = lr.get("R_KL", {}).get("R_KL", {})
        dd = rk.get(f"diffsae_x{widths[0]}", {}).get("point")
        ccr = rk.get("crosscoder_recon", {}).get("point"); cce = rk.get("crosscoder_error", {}).get("point")
        msg = f"  L{L}: rho_novel(conf,x{widths[0]})={rho:.5f}"
        if dd is not None:
            msg += f"  R_KL[diffsae/cc_recon/cc_err]={dd:.3f}/{ccr:.3f}/{cce:.3f}"
        print(msg)


if __name__ == "__main__":
    main()
