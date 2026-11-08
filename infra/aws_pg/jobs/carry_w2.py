"""CARRY Arm W2 (GPU) — sink/gauge controls (spec §3.1 Arm W2, §6 R-W2, §5). Three stages:

  sink_triple    mean-substitution (S17's exact null) vs zeroing vs RMS-preserving ablation
                 of the massive-activation sink coordinates, with a construction-matched
                 seed band (mean-sub of matched non-sink coords).
  random_token   input-agnosticity (S16): the sink structure must persist on nonsense input.
  rotation       the Hadamard gauge certificate (A4; the D-suite proved the (1+γ)/fold math
                 on a tiny block). Fold Gemma (1+γ) RMSNorm scales into adjacent weights, then
                 rotate the residual stream by a Hadamard H; a banked statistic must be
                 invariant. Two statistics: the u_dc spike Rayleigh quotient (residual-gauge,
                 PRIMARY) and full blank-image ppl (end-to-end corroboration).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_w2 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_w2 --budget-min 300
"""
import argparse
import contextlib
import json
import os
import pathlib

N_SINK = 8


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--n-vqa", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC
    from infra.pg.inherit_core import _split_hidden
    from infra.pg.projector_ablation import build_tasks

    freeze = CS.assert_frozen("w2")
    CS.install_pg()
    rep = {"object": "carry_w2_v1", "freeze": freeze}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_w2")
    wd.mkdir(parents=True, exist_ok=True)
    g0 = json.loads(CS.fetch(f"{CS.CARRY_GATE0}/gate0.json", wd / "gate0.json").read_text())
    assert g0.get("green"), "carry gate0 not green"
    anat = CS.load_roster(str(wd))

    # sink layer + coords: the gate0-banked layer with the largest max/median activation ratio
    gsink = g0["gamma_sink_readout"]
    sink_layer = max(gsink, key=lambda L: gsink[L]["max_abs_activation"][0]
                     / max(gsink[L]["median_abs_activation"], 1e-9))
    L = int(sink_layer[1:])
    coords = gsink[sink_layer]["coords"][:N_SINK]
    rep["sink"] = {"layer": L, "coords": coords}

    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev, d = "cuda", IP.D
        tasks = {t.name: t for t in build_tasks()}
        hashes, n_stream = CS.load_gate0_items(wd, "vqav2", "disc", n=a.n_vqa)
        vqa_rows = CS.rows_for_banked(tasks["vqav2"], hashes, n_stream)
        # harvest hbar at the sink layer for the mean-substitution values
        cap = IP.harvest(layers, lambda b: model(**b).logits,
                         [_ppl_batch(proc, wdict, model) for wdict in CS.wikitext_windows(proc, 8)],
                         [L], x_cap=4096)
        hbar_L = cap[L][0].float().cpu()             # (d,) native mean
        Xrows = cap[L][2].float().cpu()              # (n, d) captured rows
        med_abs = Xrows.abs().median(0).values
        rep["base_ppl"] = IP.ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)["ppl"]
        base_vqa = IP.eval_pg1(model, proc, tasks["vqav2"], vqa_rows, dev)["score"]
        rep["base_vqa"] = base_vqa
    _w()

    # S17's exact null operates on the MASSIVE ACTIVATIONS — the huge (coord, position) entries,
    # not the coord across all positions. Per sink coord, the massive site is where |x| exceeds
    # half its max; the massive VALUE is the mean over the harvest at those sites (≈ constant, so
    # mean-substitution should be ≈ identity — S17's finding). Content positions are untouched.
    thresh = {c: 0.5 * float(Xrows[:, c].abs().max()) for c in coords}
    hbar_massive = {c: float(Xrows[Xrows[:, c].abs() > thresh[c], c].mean()) for c in coords}

    def massive_hook(mode):
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            x = hs.float().clone()
            for c in coords:
                site = x[..., c].abs() > thresh[c]           # the massive positions
                if mode == "meansub":
                    x[..., c] = torch.where(site, torch.full_like(x[..., c], hbar_massive[c]),
                                            x[..., c])
                else:                                        # zero / rms both zero the massive
                    if mode == "rms":
                        nrm0 = x.norm(dim=-1, keepdim=True)
                    x[..., c] = torch.where(site, torch.zeros_like(x[..., c]), x[..., c])
                    if mode == "rms":
                        x = x * (nrm0 / x.norm(dim=-1, keepdim=True).clamp_min(1e-9))
            return rebuild(x.to(hs.dtype))
        return hook

    def band_hook(pick):
        # matched control: mean-substitute NON-sink (small) coords at ALL positions with their
        # global mean — the generic-substitution floor (small coords ⇒ small change)
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            x = hs.float().clone()
            for c in pick:
                x[..., c] = float(hbar_L[c])
            return rebuild(x.to(hs.dtype))
        return hook

    def damage_at(hook):
        with _ctx(layers, L, hook):
            pg = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
            vq = IP.eval_pg1(model, proc, tasks["vqav2"], vqa_rows, dev)
        return {"log1p_dppl": CS.safe_log1p_dppl(pg["mean_nll"], rep["base_ppl"]),
                "dvqa": vq["score"] - base_vqa}

    # ---- sink triple + the seed band ----------------------------------------------------
    with tm.stage("sink_triple"):
        mean_sub = damage_at(massive_hook("meansub"))       # S17's exact null
        zero = damage_at(massive_hook("zero"))
        rms_pres = damage_at(massive_hook("rms"))
        rep["sink_massive"] = {"thresh": {str(c): thresh[c] for c in coords},
                               "hbar_massive": {str(c): hbar_massive[c] for c in coords}}
        # seed band: mean-sub of matched non-sink coords (|activation| below median), 3 draws
        low_coords = [int(i) for i in (med_abs < med_abs.median()).nonzero().flatten()
                      if int(i) not in coords]
        band_draws = []
        for s in range(3):
            gen = np.random.default_rng(s)
            pick = list(gen.choice(low_coords, size=len(coords), replace=False))
            band_draws.append(damage_at(band_hook(pick)))
        band_vals = [abs(bd["log1p_dppl"]) for bd in band_draws]
        seed_band = max(max(band_vals), 1e-6)
        rep["sink_triple"] = {"mean_sub": mean_sub, "zero": zero, "rms_preserving": rms_pres,
                              "seed_band_draws": band_draws, "seed_band": seed_band}
        rep["rw2_meansub"] = CC.adjudicate_rw2_meansub(
            d_meansub=mean_sub["log1p_dppl"], d_zero=zero["log1p_dppl"], seed_band=seed_band)
        rep["rw2_rms_vs_zero_gap"] = rms_pres["log1p_dppl"] - zero["log1p_dppl"]
        print(f"[carry-w2] mean_sub={mean_sub['log1p_dppl']:+.3f} zero={zero['log1p_dppl']:+.3f} "
              f"band={seed_band:.4f} -> {rep['rw2_meansub']['verdict']}", flush=True)
    _w()

    # ---- random-token persistence (S16) -------------------------------------------------
    with tm.stage("random_token"):
        vocab = model.config.text_config.vocab_size
        real_mag, rand_mag = [], []
        wins = CS.wikitext_windows(proc, 8)
        for wi, wdict in enumerate(wins):
            b = _ppl_batch(proc, wdict, model)
            capr = IP.harvest(layers, lambda bb: model(**bb).logits, [b], [L], x_cap=4096)
            real_mag.append(float(capr[L][2].abs()[:, coords].max()))
            gen = torch.Generator(device="cpu").manual_seed(123 + wi)
            rb = dict(b)
            # preserve the <image> placeholder tokens (else features/tokens mismatch); only
            # the TEXT tokens are randomized — the S16 input-agnosticity test is on the text
            img_id = model.config.image_token_index
            ids = b["input_ids"].clone()
            text_mask = ids[0] != img_id
            n_txt = int(text_mask.sum())
            ids[0, text_mask] = torch.randint(0, vocab, (n_txt,), generator=gen).to(dev)
            rb["input_ids"] = ids
            caprr = IP.harvest(layers, lambda bb: model(**bb).logits, [rb], [L], x_cap=4096)
            rand_mag.append(float(caprr[L][2].abs()[:, coords].max()))
        rep["random_token"] = {
            "real_sink_mag_mean": float(np.mean(real_mag)),
            "random_sink_mag_mean": float(np.mean(rand_mag)),
            "persistence_ratio": float(np.mean(rand_mag) / max(np.mean(real_mag), 1e-9))}
    _w()

    # ---- rotation gauge certificate (A4) ------------------------------------------------
    # BOTH statistics measured on the SAME fp32 model (unrotated then rotated in place) so the
    # only difference is the gauge — a bf16-vs-fp32 comparison would confound precision (~1e-2)
    # with a rotation-wiring error and could falsely HALT.
    with tm.stage("rotation"):
        lib = torch.load(CS.fetch(f"{CS.SPECTER_O}/library_bank.pt", wd / "library_bank.pt"),
                         map_location="cpu", weights_only=False)
        u_dc = lib["u_dc"].float()
        del model
        torch.cuda.empty_cache()
        model2, proc2 = _load_fp32(anat["base_sha"])
        layers2 = IP.decoder_layers(model2)

        def _q_and_ppl(m, p, lyrs, probe):
            inst, hold = SC.make_residual_injector(lyrs, 0)
            its = [CS.build_ppl_item(m, p, dev, wdict, inst, hold)
                   for wdict in CS.wikitext_windows(p, 6)]
            q = float(SC.mode_quadform(its, probe.to(dev), "dc"))
            ppl = CS.safe_ppl_gate(m, p, dev, n_seq=40)["mean_nll"]
            return q, ppl

        q_unrot, ppl_unrot = _q_and_ppl(model2, proc2, layers2, u_dc)
        H = CC.hadamard_matrix(d).float()
        _rotate_model(model2, H, proc2)
        q_rot, ppl_rot = _q_and_ppl(model2, proc2, layers2, u_dc @ H)

        rel_q = abs(q_rot - q_unrot) / max(abs(q_unrot), 1e-30)
        rel_ppl = abs(ppl_rot - ppl_unrot) / max(abs(ppl_unrot), 1e-30)
        rep["rotation"] = {"q_unrot": q_unrot, "q_rot": q_rot, "rel_q": rel_q,
                           "ppl_unrot": ppl_unrot, "ppl_rot": ppl_rot, "rel_ppl": rel_ppl}
        rep["rw2_rotation"] = CC.adjudicate_rw2_rotation(max_rel_err=rel_q)
        rep["rw2_rotation"]["rel_ppl_corroboration"] = rel_ppl
        print(f"[carry-w2] rotation rel_q={rel_q:.2e} rel_ppl={rel_ppl:.2e} -> "
              f"{rep['rw2_rotation']['verdict']}", flush=True)
    _w()

    CS.assert_gpu_seconds(rep)
    print("CARRY_W2_OK", json.dumps({"meansub_verdict": rep["rw2_meansub"]["verdict"],
                                     "rotation_verdict": rep["rw2_rotation"]["verdict"]}))
    rep["green"] = True
    _w()


@contextlib.contextmanager
def _ctx(layers, L, hook):
    # bispec_core.hook_at is already a @contextmanager — delegate to it (do NOT treat its
    # return as a removable handle)
    from infra.pg.bispec_core import hook_at
    with hook_at(layers, L, hook):
        yield


def _ppl_batch(proc, wdict, model):
    import torch
    inp = proc(text=[wdict["text"]], images=[wdict["blank"]], return_tensors="pt",
               padding=False).to(next(model.parameters()).device)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    return dict(inp)


def _load_fp32(sha):
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224", revision=sha, torch_dtype=torch.float32,
        device_map="cuda", attn_implementation="eager").eval()
    p = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", revision=sha)
    return m, p


def _rotate_model(model, H, proc):
    """Fold Gemma (1+γ) RMSNorm into adjacent weights, then rotate the residual stream by H
    (orthonormal). Residual-reading matrices W ← W·H; residual-writing matrices W ← Hᵀ·W and
    bias ← bias·H; residual-vector rows (embeddings) ← rows·H. RMSNorm weights zeroed after
    the fold (Gemma computes x̂·(1+w); w=0 ⇒ scale 1, and the scale-free norm commutes with H).
    fp32 in place. Fails loud if any expected submodule is absent."""
    import torch
    import torch.nn as nn
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths

    Hd = H.to("cuda", torch.float32)
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    layers = IP.decoder_layers(model)

    def fold(norm_mod, reader_mods):
        w = norm_mod.weight.data.float()
        for m in reader_mods:
            assert m.weight.shape[1] == w.shape[0], "fold side mismatch"
            m.weight.data = (m.weight.data.float() * (1.0 + w).unsqueeze(0))
        norm_mod.weight.data.zero_()

    def rot_read(m):
        m.weight.data = (m.weight.data.float() @ Hd)

    def rot_write(m):
        m.weight.data = (Hd.t() @ m.weight.data.float())
        if getattr(m, "bias", None) is not None:
            m.bias.data = (m.bias.data.float() @ Hd)

    # embeddings (rows are residual vectors) — HF standard APIs (version-robust)
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    assert emb is not None and head is not None, "missing input/output embeddings"
    # decouple a tied lm_head BEFORE any fold touches it
    tied = head.weight.data_ptr() == emb.weight.data_ptr()
    if tied:
        head.weight = nn.Parameter(emb.weight.data.float().clone())

    # per-layer folds + rotations
    for lyr in layers:
        fold(lyr.input_layernorm, [lyr.self_attn.q_proj, lyr.self_attn.k_proj,
                                   lyr.self_attn.v_proj])
        fold(lyr.post_attention_layernorm, [lyr.mlp.gate_proj, lyr.mlp.up_proj])
        for m in (lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj,
                  lyr.mlp.gate_proj, lyr.mlp.up_proj):
            rot_read(m)
        rot_write(lyr.self_attn.o_proj)
        rot_write(lyr.mlp.down_proj)

    # final norm folds into the (now-decoupled) lm_head, then rotate the head's read side.
    # Find the top-level decoder norm robustly: an RMSNorm of the same type as a layer's
    # input_layernorm whose module name ends in ".norm" (not "*_layernorm").
    ln_type = type(layers[0].input_layernorm)
    finals = [(n, mod) for n, mod in model.named_modules()
              if isinstance(mod, ln_type) and n.split(".")[-1] == "norm"]
    assert len(finals) == 1, f"expected 1 final norm, found {[n for n, _ in finals]}"
    final_norm = finals[0][1]
    fold(final_norm, [head])
    rot_read(head)

    # embeddings + projector write the INITIAL residual → rotate rows / write-side
    emb.weight.data = (emb.weight.data.float() @ Hd)
    proj = model.get_submodule(paths["multi_modal_projector"])
    linears = [m for m in proj.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 1, f"expected 1 projector Linear, found {len(linears)}"
    rot_write(linears[0])


def _has(mod, name):
    return name in dict(mod.named_modules())


if __name__ == "__main__":
    main()
