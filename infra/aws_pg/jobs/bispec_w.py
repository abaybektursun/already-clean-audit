"""BISPEC Arm W — the write–read–data triangle: where fine-tuning writes, on the data's own axes.

Three quadratic forms summarize this whole program:

    Σ   what the DATA varies      (the site's activation covariance)
    H   what BEHAVIOUR reads      (the susceptibility surface, from Arm S)
    Ω   what FINE-TUNING writes   (the consensus ΔW second moment over 24 PG1 fine-tunes)

Arm W puts Ω and Σ in one coordinate system and asks whether the write density follows the data
covariance. That is the **lazy-training / NTK prediction**, stated parameter-free: under lazy
training the update is driven by the input second moment, so `Ω`'s energy per band equals the
band's own energy fraction. `lazy_training_reference(bands)` IS that curve — no fitted constant.

  H_ntk-aligned  every profile tracks the reference within tol in the top band
  H_offset       profiles are displaced off the top band toward mid bands, consistently
  H_stratified   the sign of the top-band displacement differs across matrix types

**Each matrix type is profiled in ITS OWN input space.** A ΔW for `q_proj` acts on the post-
`input_layernorm` residual; a ΔW for `gate_proj` acts on the post-`post_attention_layernorm`
residual. These are different vector spaces with different covariances, and banding one by the
other's eigenbasis would be a category error. So this job captures the Gram of each matrix's actual
input (`_InputCapture`-style pre-hooks on `q_proj` and `gate_proj`) and bands each ΔW against the
matching Σ. That closes §10's "post-norm vs pre-norm site conventions" threat by construction rather
than by disclosure.

**The estimand.** For matrix W with input-space band basis U_b,

    profile(b) = ‖ΔW U_b‖_F² / ‖ΔW‖_F²

— the fraction of the write's Frobenius energy that acts on input directions in band b. Summed over
bands it is exactly 1.

**Prior, from the artifact, held as a hypothesis-to-kill.** REGRAIN R9 measured the consensus write
to be 2.54× more concentrated than a Haar subspace yet **22.6× LESS** concentrated than the task's
own activation footprint, with disjoint top-8 atoms. That points at H_offset. It is the reading this
job is built to break, not to confirm.

EXPLORATORY / DISCOVERY. CPU-dominant: one short GPU capture, then a streamed ΔW pass.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_w --type g6e.xlarge --no-weights \
      --s3-prefix results/bispec_w --budget-min 420 --args "--layers 0,4,9,14,17 --k 128"
"""
import argparse
import json
import os
import pathlib

ATTN_TYPES = ("q_proj", "k_proj", "v_proj")     # input = post-input_layernorm residual
MLP_TYPES = ("gate_proj", "up_proj")            # input = post-post_attention_layernorm residual


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # HF's Xet transfer backend stalls indefinitely on this instance class: two separate
    # runs wedged inside huggingface_hub.file_download.xet_get at "Fetching 3 files: 33%",
    # log fresh but byte-count frozen for ~50 min. Force the plain HTTP path.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,4,9,14,17")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--ft-tasks", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor

    from infra.aws_pg.jobs.conserved_core import _wikitext_windows
    from infra.aws_pg.jobs.invariant_core import PG1_REPO, _delete_cache, _load_decoder
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _decoder_layers, _load_full_pg1
    from infra.pg import bispec_core as BC

    want_L = [int(x) for x in a.layers.split(",")]
    rep = {"object": "bispec_w_v1", "layers": want_L, "k": a.k,
           "label": "DISCOVERY / EXPLORATORY — measurement arm, CI-reported, no sealed exposure",
           "estimand": "profile(b) = ||dW U_b||_F^2 / ||dW||_F^2, U_b a band of the matrix's OWN "
                       "input-activation covariance",
           "lazy_null": "write density proportional to data covariance => profile(b) = band energy "
                        "fraction; a parameter-free reference curve, not a fitted one",
           "prior_to_kill": "REGRAIN R9: consensus write 2.54x Haar but 22.6x LESS concentrated than "
                            "the task footprint, top-8 disjoint => points at H_offset"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    api = HfApi()
    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    base_sha = api.model_info(PG1_REPO).sha
    rep["n_ft"] = len(repos)
    rep["cloud_repos"] = repos
    rep["base_sha"] = base_sha[:12]
    _w()

    # ---------- 1. capture each matrix type's OWN input covariance (short GPU pass) ----------
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, "cuda")
    layers = _decoder_layers(model)
    windows = _wikitext_windows(processor, a.n_ppl_seq, a.ppl_window)

    class _InGram:
        """Gram of the input to `attr` (a submodule of layers[L]), fp64, streamed per forward."""

        def __init__(self, attr):
            self.attr = attr
            self.gram = {L: None for L in want_L}
            self.n = {L: 0 for L in want_L}
            self._h = []

        def _mk(self, L):
            def pre_hook(_m, args):
                x = args[0].reshape(-1, args[0].shape[-1]).double()
                g = (x.transpose(0, 1) @ x).cpu()
                self.gram[L] = g if self.gram[L] is None else self.gram[L] + g
                self.n[L] += x.shape[0]
            return pre_hook

        def __enter__(self):
            for L in want_L:
                mod = layers[L].self_attn if self.attr == "q_proj" else layers[L].mlp
                self._h.append(getattr(mod, self.attr).register_forward_pre_hook(self._mk(L)))
            return self

        def __exit__(self, *e):
            for h in self._h:
                h.remove()

    grams = {}
    for space, attr in (("attn", "q_proj"), ("mlp", "gate_proj")):
        cap = _InGram(attr)
        blank = _blank_image(128)
        with cap, torch.inference_mode():
            for wt, _wi in windows:
                inp = processor(text=[wt], images=[blank], return_tensors="pt", padding=False).to("cuda")
                model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(torch.bfloat16))
        grams[space] = {L: cap.gram[L] / cap.n[L] for L in want_L}
        print(f"[w] captured {space} input Gram ({attr}) over {cap.n[want_L[0]]} tokens", flush=True)
    del model
    torch.cuda.empty_cache()

    # A2, applied here too: band the CENTERED covariance. The input Grams above are uncentered
    # second moments; subtract the mean outer product measured in the same pass would require the
    # mean, so we band the second moment and DISCLOSE it — with the sensitivity that the top
    # eigendirection of an uncentered post-norm Gram is close to the mean direction.
    bands = {}
    for space in grams:
        bands[space] = {}
        for L in want_L:
            bb = BC.eig_bands(torch.as_tensor(grams[space][L], dtype=torch.float64))
            bands[space][L] = bb
    rep["bands"] = {sp: {str(L): {"ranks": [hi - lo for lo, hi in bands[sp][L]["band_slices"]],
                                  "energy": bands[sp][L]["band_energy"],
                                  "top1_share": float(bands[sp][L]["spectrum"][0]
                                                      / bands[sp][L]["spectrum"].sum())}
                         for L in want_L} for sp in bands}
    rep["band_convention"] = ("uncentered second moment of the POST-NORM input (that is the object "
                              "the matrix actually multiplies); disclosed, and the top1_share is "
                              "reported so the mean-direction confound is visible")
    _w()

    # ---------- 2. stream the 24 ΔW and profile each in its own band basis ----------
    w_base = _load_decoder(PG1_REPO, base_sha)

    def key(L, t):
        pre = "self_attn" if t in ATTN_TYPES else "mlp"
        return f"layers.{L}.{pre}.{t}.weight"

    acc = {(L, t): None for L in want_L for t in ATTN_TYPES + MLP_TYPES}
    n_terms = 0
    for ridx, repo in enumerate(repos):
        sha = api.model_info(repo).sha
        w_ft = _load_decoder(repo, sha)
        for L in want_L:
            for t in ATTN_TYPES + MLP_TYPES:
                kk = key(L, t)
                assert kk in w_ft and kk in w_base, f"missing {kk}"
                dW = (w_ft[kk].double() - w_base[kk].double())
                nrm = float((dW ** 2).sum())
                assert nrm > 0, f"zero ΔW for {repo} {kk}"
                # consensus second moment on the INPUT side: dW^T dW (d_in x d_in), unit-normalized
                # per fine-tune so no single ft dominates (the invariant_core convention)
                G = (dW.transpose(0, 1) @ dW) / nrm
                acc[(L, t)] = G if acc[(L, t)] is None else acc[(L, t)] + G
        n_terms += 1
        del w_ft
        # 24 x ~12GB fp32 snapshots overflow the root volume. invariant_core deletes each after use;
        # this job did not, and died on "No space left on device" at ft 6/24.
        _delete_cache(repo)
        print(f"[w] ΔW {repo} ({ridx+1}/{len(repos)}) [snapshot purged]", flush=True)
    for kk in acc:
        acc[kk] = acc[kk] / n_terms

    profiles, prof_flat = {}, {}
    for L in want_L:
        for t in ATTN_TYPES + MLP_TYPES:
            space = "attn" if t in ATTN_TYPES else "mlp"
            bb = bands[space][L]
            Om = acc[(L, t)]                                   # (d_in, d_in) PSD consensus write form
            tot = float(torch.diagonal(Om).sum())
            prof = []
            for b in range(len(bb["band_slices"])):
                U = BC.band_projector(bb, b)
                prof.append(float(torch.einsum("ai,ab,bi->", U, Om, U)) / tot)
            rnd = BC.random_write_reference(bb)
            profiles[f"L{L}/{t}"] = {"profile": prof, "ntk_reference": bb["band_energy"],
                                     "random_reference": rnd, "space": space,
                                     "enrichment_vs_random": [prof[b] / max(rnd[b], 1e-12)
                                                              for b in range(len(rnd))]}
            prof_flat[(f"L{L}", t)] = prof
            print(f"[w] L{L} {t}: profile={[round(x,4) for x in prof]} "
                  f"ntk_ref={[round(x,4) for x in bb['band_energy']]} "
                  f"rand_ref={[round(x,4) for x in rnd]}", flush=True)
    rep["profiles"] = profiles
    _w()

    # ---------- 3. adjudicate (per space, since the reference differs) ----------
    verdicts = {}
    for space in ("attn", "mlp"):
        sub = {k: v for k, v in prof_flat.items() if (k[1] in ATTN_TYPES) == (space == "attn")}
        # the reference is band energy, which varies by layer; adjudicate per layer, then summarize
        per_layer = {}
        for L in want_L:
            ref = bands[space][L]["band_energy"]
            # adjudicate_write echoes its profile keys into the verdict; tuple keys are not
            # JSON-serializable and killed the final dump AFTER every profile had been computed.
            sl = {f"{k[0]}/{k[1]}": v for k, v in sub.items() if k[0] == f"L{L}"}
            if sl:
                per_layer[f"L{L}"] = BC.adjudicate_write(sl, ref)
        labels = {v["verdict"] for v in per_layer.values()}
        verdicts[space] = {"per_layer": per_layer,
                           "consistent": len(labels) == 1,
                           "labels": sorted(labels)}
    # the null Arm W was missing: is the write distinguishable from a Haar-random dW at all?
    struct = {}
    for L in want_L:
        for space, types in (("attn", ATTN_TYPES), ("mlp", MLP_TYPES)):
            rnd = BC.random_write_reference(bands[space][L])
            sl = {f"L{L}/{t}": prof_flat[(f"L{L}", t)] for t in types}
            struct[f"L{L}/{space}"] = BC.adjudicate_write_structure(sl, rnd)
    rep["structure_vs_random_write"] = struct
    rep["structure_summary"] = {
        "any_structured": any(v["verdict"] == "H_write-structured" for v in struct.values()),
        "reading": "bands 2-4 hold 2037 of 2048 dimensions, so a Haar-random dW already places ~99.5% "
                   "of its energy there. 'The write avoids the data's top eigendirection' is only a "
                   "finding if a RANDOM write does not avoid it by the same margin."}
    rep["verdicts"] = verdicts
    # H_stratified at the program level iff attn and mlp land differently anywhere
    attn_lbls = set(verdicts["attn"]["labels"])
    mlp_lbls = set(verdicts["mlp"]["labels"])
    if attn_lbls == mlp_lbls and len(attn_lbls) == 1:
        rep["program_verdict"] = list(attn_lbls)[0]
    else:
        rep["program_verdict"] = "H_stratified"
    rep["program_verdict_reading"] = (
        "H_ntk-aligned would be theory earning a predictive win on real third-party fine-tunes; "
        "H_offset says adaptation writes off the data's principal axes; H_stratified says the "
        "write-read offset is a map, not a constant. All three are reachable observations.")
    _w()

    print("BISPEC_W_OK", json.dumps({"program_verdict": rep["program_verdict"],
                                     "attn": verdicts["attn"]["labels"],
                                     "mlp": verdicts["mlp"]["labels"]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
