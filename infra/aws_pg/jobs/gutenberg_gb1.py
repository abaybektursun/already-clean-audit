"""GUTENBERG GB-1 — the inversion on the paper's primary object (PG1, 24-FT invariant core).

Spec §3 GB-1 + AM-0.5.2/6/8. AM-1's 98-window paired ablation protocol
(infra/vast/cells/v1_ablation.py), ported to PG1/gemma-2b with:
  - core32 = the BANKED 24-FT invariant core (copernicus_a1_banks core/32/L, hash-verified) —
    the object the paper's §1-2 are about, not a rebuild;
  - forward = FUNC's critic-passed blank-image in-distribution path, eager attention
    (invariant_core_func helpers reused verbatim);
  - corpus = AM-1's frozen recipe on PG1's tokenizer (wikitext-103-raw-v1 test @ the
    prepare_corpus pinned revision, 98 x 512-token windows);
  - arms at site q (AM-1's exact instrument: q_proj-in only): none, core32, anti32(rider),
    haar32 x3, bottomsigma32, topsigma32, topenergy32 (in-run positive control),
    energymatched32 x3, specmatched32 x3;
  - rider at site qk (FUNC's instrument, AM-0.5.6 bridge): core32, energymatched32 x3, haar32 x3.
Frozen verdict rule in gutenberg_core.inversion_verdict; gates per AM-0.5.8. FAIL LOUD.

    pg run infra.aws_pg.jobs.gutenberg_gb1 --no-weights --type g6e.2xlarge,g6.2xlarge,g5.2xlarge \
        --s3-prefix results/gutenberg_gb1 --budget-min 100
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G
from infra.pg.conserved_core_probes import energy_matched_subspace
from infra.pg.humboldt_core import hf_setup

A1_NPZ = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks.npz"
A1_MANIFEST = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks_manifest.json"
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"   # prepare_corpus.py pin
N_LAYERS, K, N_IN = 18, 32, 2048
PPL_NONE_BAND = (18.0, 28.0)          # AM-0.5.8(ii); sdpa failure mode reads ~36.8
TOPE_MIN_RATIO = 10.0                 # AM-0.5.8(iii)


def _fetch(s3_uri, dst):
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def build_windows(tok, n_windows, window):
    """AM-1's corpus recipe on PG1's tokenizer, pre-validated against the processor's
    trailing-newline merge: windows whose exact token block cannot be located get their final
    token dropped (deterministic, arm-independent, counted). Returns list of window dicts."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test",
                      revision=WIKITEXT_REV)
    text = "\n".join(r["text"] for r in ds)
    ids = tok(text, return_tensors="np", add_special_tokens=False).input_ids[0]
    need = n_windows * window
    assert len(ids) >= need, (len(ids), need)
    return [tok.decode(ids[i * window:(i + 1) * window].tolist()) for i in range(n_windows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-act-windows", type=int, default=32)
    ap.add_argument("--n-abl-windows", type=int, default=98)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--n-extra-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_act_windows, a.n_abl_windows, a.n_extra_seeds = 4, 6, 1
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb1")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()
    import torch
    from infra.aws_pg.jobs.invariant_core_func import (
        _blank_image, _decoder_layers, _load_full_pg1, _locate_text_block)
    from transformers import PaliGemmaProcessor

    # ---- banks (the paper's object, hash-verified) ----
    man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    base_sha = man["base_sha"]
    want = {(b["bank"], b["layer"]): b["sha256"] for b in man["banks"] if b["k"] == K}
    npz = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))
    U = {}
    rng = np.random.default_rng(1337)              # v1_ablation's haar sequence convention
    for L in range(N_LAYERS):
        for bid in ("core", "anti"):
            Q = np.asarray(npz[f"{bid}/{K}/{L}"], dtype=np.float64)
            assert C.bank_hash(Q) == want[(bid, L)], f"{bid}/32/{L} hash mismatch"
            C.orthonormal_or_die(Q, f"{bid}32 L{L}")
            U[(f"{bid}32", L)] = Q
        for s in range(3):
            U[(f"haar32_s{s}", L)] = np.linalg.qr(rng.standard_normal((N_IN, K)))[0]

    # ---- model + corpus ----
    dev = "cuda"
    assert torch.cuda.is_available()
    model = _load_full_pg1(base_sha, dev)
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224",
                                                   revision=base_sha)
    tok = processor.tokenizer
    layers = _decoder_layers(model)
    blank = _blank_image()

    window_texts = build_windows(tok, a.n_abl_windows, a.window)
    units, n_trimmed = [], 0
    for w_text in window_texts:
        for attempt in range(2):
            win_ids = tok(w_text, return_tensors="pt",
                          add_special_tokens=False).input_ids[0]
            inp = processor(text=[w_text], images=[blank], return_tensors="pt")
            try:
                pos = _locate_text_block(inp["input_ids"][0], win_ids)
                break
            except AssertionError:
                assert attempt == 0, "text block unlocatable even after trailing-token trim"
                w_text = tok.decode(win_ids[:-1].tolist())     # AM-0.5.8 pre-registered response
                n_trimmed += 1
        units.append({"text": w_text, "n_text": int(win_ids.shape[0]), "pos": pos,
                      "input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                      "pixel_values": inp["pixel_values"]})
    print(f"[gb1] corpus ready: {len(units)} windows, {n_trimmed} trimmed", flush=True)

    # ---- sigma banks from the base q_proj (v1_ablation convention, on device) ----
    for L in range(N_LAYERS):
        W = layers[L].self_attn.q_proj.weight.detach().to(torch.float32)
        assert W.shape == (N_IN, N_IN), W.shape
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        U[("bottomsigma32", L)] = Vh[-K:, :].T.cpu().numpy().astype(np.float64)
        U[("topsigma32", L)] = Vh[:K, :].T.cpu().numpy().astype(np.float64)
        del Vh

    Ut = {}   # fp32 device tensors for hooks

    def _to_dev(key_layer):
        if key_layer not in Ut:
            Ut[key_layer] = torch.as_tensor(np.ascontiguousarray(U[key_layer]),
                                            dtype=torch.float32, device=dev)
        return Ut[key_layer]

    # ---- one forward pass over units with optional projection / accumulation hooks ----
    def run_pass(n_windows, project_arm=None, sites=("q",), act_accumulate=None,
                 act_arms=(), gram=None):
        handles = []
        if project_arm is not None or act_accumulate is not None:
            for L in range(N_LAYERS):
                def mk(L):
                    def hook(_m, args):
                        h = args[0]
                        h32 = h.to(torch.float32)
                        if act_accumulate is not None:
                            flat = h32.reshape(-1, h32.shape[-1])
                            act_accumulate[("_tot", L)] = act_accumulate.get(("_tot", L), 0.0) \
                                + float(flat.pow(2).sum())
                            for bid in act_arms:
                                p = flat @ _to_dev((bid, L))
                                act_accumulate[(bid, L)] = act_accumulate.get((bid, L), 0.0) \
                                    + float(p.pow(2).sum())
                            if gram is not None:
                                gram[L] += flat.T @ flat
                            return None
                        Q = _to_dev((project_arm, L))
                        h32 = h32 - (h32 @ Q) @ Q.T
                        return (h32.to(h.dtype),) + tuple(args[1:])
                    return hook
                mods = []
                if act_accumulate is not None or "q" in sites:
                    mods.append(layers[L].self_attn.q_proj)
                if project_arm is not None and "k" in sites:
                    mods.append(layers[L].self_attn.k_proj)
                for m in mods:
                    handles.append(m.register_forward_pre_hook(mk(L)))
        nlls, ntoks = [], []
        with torch.inference_mode():
            for u in units[:n_windows]:
                logits = model(input_ids=u["input_ids"].to(dev),
                               attention_mask=u["attention_mask"].to(dev),
                               pixel_values=u["pixel_values"].to(dev).to(torch.bfloat16)
                               ).logits[0]
                n_text, pos = u["n_text"], u["pos"]
                win_ids = tok(u["text"], return_tensors="pt",
                              add_special_tokens=False).input_ids[0]
                text_logits = logits[pos:pos + n_text - 1, :].float()
                targets = win_ids[1:].to(dev)
                assert text_logits.shape[0] == targets.shape[0]
                loss = torch.nn.functional.cross_entropy(text_logits, targets,
                                                         reduction="sum")
                nlls.append(float(loss.item()))
                ntoks.append(int(targets.numel()))
                del logits
        for h in handles:
            h.remove()
        nll_a, ntok_a = np.asarray(nlls), np.asarray(ntoks, dtype=np.int64)
        return float(np.exp(nll_a.sum() / ntok_a.sum())), nll_a, ntok_a

    # ---- anchor gate FIRST (fail-fast preflight: the known number) ----
    ppl_none, nll_none, ntok_none = run_pass(a.n_abl_windows)
    print(f"[gb1] ppl(none) = {ppl_none:.4f}", flush=True)
    assert PPL_NONE_BAND[0] <= ppl_none <= PPL_NONE_BAND[1], \
        f"anchor gate FAILED: ppl(none)={ppl_none} outside {PPL_NONE_BAND}"

    # ---- ACT pass: gate + Gram + coordinate energies ----
    gate_arms = ("core32", "haar32_s0", "haar32_s1", "haar32_s2")
    acc = {}
    gram = {L: torch.zeros(N_IN, N_IN, dtype=torch.float32, device=dev)
            for L in range(N_LAYERS)}
    run_pass(a.n_act_windows, act_accumulate=acc, act_arms=gate_arms, gram=gram)
    shares = {bid: [acc[(bid, L)] / acc[("_tot", L)] for L in range(N_LAYERS)]
              for bid in gate_arms}
    med_core = float(np.median(shares["core32"]))
    med_haar = float(np.median(shares["haar32_s0"] + shares["haar32_s1"]
                               + shares["haar32_s2"]))
    act = {"median_share_core32": med_core, "median_share_haar": med_haar,
           "per_layer_core": shares["core32"],
           "gate": "LIVE-candidate" if med_core > med_haar else "INERT-STOP"}
    (out / "gb1_act_gate.json").write_text(json.dumps(act, indent=2))
    if med_core <= med_haar:
        print("GUTENBERG_GB1_OK verdict=INERT-STOP (pre-registered stop)", flush=True)
        return

    # topenergy32 from THIS run's coordinate energies (AM-0.5.8iii)
    tope_idx = {}
    for L in range(N_LAYERS):
        diag = gram[L].diagonal().cpu().numpy()
        idx = np.argsort(diag)[::-1][:K].copy()
        tope_idx[L] = idx
        U[("topenergy32", L)] = G.coordinate_bank(N_IN, idx)

    # energymatched / specmatched from THIS run's Gram (v1_ablation's construction + seeds)
    random_floor = K / N_IN
    ematch_diag, spec_diag = [], []
    for L in range(N_LAYERS):
        mu, V = torch.linalg.eigh(gram[L])
        mu = mu.flip(0).clamp_min(0.0).to(torch.float64).cpu().numpy()
        V = V.flip(1).to(torch.float64).cpu().numpy()
        e_target = shares["core32"][L]
        for s in range(a.n_extra_seeds):
            Q, e_real = energy_matched_subspace(
                V, mu, e_target, K, seed=70_000 + 1000 * s + L,
                tol=max(0.02 * e_target, 1e-6), max_resample=200)
            U[(f"energymatched32_s{s}", L)] = np.ascontiguousarray(Q)
            A = V.T @ U[("core32", L)]
            srng = np.random.default_rng(60_000 + 1000 * s + L)
            Qs, _ = np.linalg.qr(V @ (A * srng.choice([-1.0, 1.0], size=A.shape)))
            U[(f"specmatched32_s{s}", L)] = np.ascontiguousarray(Qs[:, :K])
            if s == 0:
                ematch_diag.append({"layer": L, "e_target": float(e_target),
                                    "e_realized": float(e_real),
                                    "below_random_floor": bool(e_target < random_floor)})
            spec_diag.append({"layer": L, "seed": s,
                              "profile_l1_rel_change": float(
                                  np.abs((V.T @ Qs[:, :K]) ** 2 - A ** 2).sum()
                                  / (A ** 2).sum())})
        gram[L] = None
        del mu, V
    del gram
    torch.cuda.empty_cache()
    for r in ematch_diag:
        if not r["below_random_floor"]:
            assert abs(r["e_realized"] - r["e_target"]) <= 0.1 * r["e_target"], \
                f"energy match failed at L{r['layer']}: {r}"

    # ---- two-directional share verification (v1_ablation's acc2 pass) ----
    ext = [f"{n}_s{s}" for n in ("energymatched32", "specmatched32")
           for s in range(a.n_extra_seeds)]
    verify_arms = tuple(["core32", "anti32", "bottomsigma32", "topsigma32",
                         "topenergy32"] + ext)
    acc2 = {}
    run_pass(a.n_act_windows, act_accumulate=acc2, act_arms=verify_arms)
    arm_shares = {bid: [acc2[(bid, L)] / acc2[("_tot", L)] for L in range(N_LAYERS)]
                  for bid in verify_arms}
    (out / "gb1_arm_shares.json").write_text(json.dumps(
        {"per_layer_share": arm_shares,
         "median_share": {b: float(np.median(v)) for b, v in arm_shares.items()},
         "energy_match_per_layer": ematch_diag, "spectral_match_per_layer": spec_diag,
         "random_floor_k_over_d": random_floor,
         "haar_share_median": med_haar}, indent=2))

    # ---- ablation passes ----
    arms_q = ["core32", "anti32", "haar32_s0", "haar32_s1", "haar32_s2",
              "bottomsigma32", "topsigma32", "topenergy32"] + ext
    arms_qk = ["core32"] + [f"energymatched32_s{s}" for s in range(a.n_extra_seeds)] \
        + ["haar32_s0", "haar32_s1", "haar32_s2"]
    rows_by_site = {"q": {"none": ppl_none}, "qk": {"none": ppl_none}}
    wnll_by_site = {"q": {"none": {"nll": nll_none.tolist(), "ntok": ntok_none.tolist()}},
                    "qk": {"none": {"nll": nll_none.tolist(), "ntok": ntok_none.tolist()}}}
    for site, arms in (("q", arms_q), ("qk", arms_qk)):
        sites = ("q",) if site == "q" else ("q", "k")
        for arm in arms:
            p, nll, ntok = run_pass(a.n_abl_windows, project_arm=arm, sites=sites)
            rows_by_site[site][arm] = p
            wnll_by_site[site][arm] = {"nll": nll.tolist(), "ntok": ntok.tolist()}
            print(f"[gb1] site={site} {arm}: ppl={p:.4f}", flush=True)

    # ---- gates + aggregation ----
    d_tope = rows_by_site["q"]["topenergy32"] - ppl_none
    d_haar_max = max(rows_by_site["q"][f"haar32_s{s}"] - ppl_none for s in range(3))
    assert d_tope >= TOPE_MIN_RATIO * max(d_haar_max, 1e-9), \
        f"positive control FAILED: topenergy32 dppl {d_tope} < {TOPE_MIN_RATIO}x haar max {d_haar_max}"
    gates = {"ppl_none": ppl_none, "ppl_none_band": PPL_NONE_BAND, "eager": True,
             "topenergy32_dppl": d_tope, "haar_max_dppl": d_haar_max,
             "tope_over_haar": d_tope / max(d_haar_max, 1e-9),
             "act_gate": act["gate"], "n_windows_trimmed": n_trimmed,
             "corpus": f"wikitext-103-raw-v1:test@{WIKITEXT_REV} "
                       f"{a.n_abl_windows}x{a.window} (prepare_corpus recipe)"}
    metrics = G.build_gb1_metrics(rows_by_site, wnll_by_site, gates, ematch_diag,
                                  arm_shares, seed=a.seed)
    metrics["provenance"] = {"a1_banks": A1_NPZ, "a1_manifest_sha256": man["manifest_sha256"],
                             "base_repo": "google/paligemma-3b-pt-224",
                             "base_sha": base_sha, "code": G.provenance(),
                             "utc": C.utc_now()}
    (out / "gutenberg_gb1_verdict.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float))
    (out / "gb1_window_nll.json").write_text(json.dumps(
        {"per_window_by_site": wnll_by_site,
         "alignment_key": "window index in the frozen corpus order",
         "n_abl_windows": a.n_abl_windows}, indent=2))
    banks_out = {}
    for (bid, L), Q in U.items():
        if bid.startswith(("energymatched32", "specmatched32")):
            banks_out[f"{bid}/{L}"] = np.asarray(Q, dtype=np.float32)
    np.savez_compressed(out / "gb1_banks.npz",
                        **banks_out,
                        **{f"topenergy32_idx/{L}": tope_idx[L] for L in tope_idx})
    print(f"GUTENBERG_GB1_OK verdict={metrics['verdict']} "
          f"E_over_C={metrics['sites']['q']['E_over_C'][0]:.3f} "
          f"ppl_none={ppl_none:.3f}", flush=True)


if __name__ == "__main__":
    main()
