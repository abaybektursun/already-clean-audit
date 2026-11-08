"""V1(a) — FUNC-style ablation at Qwen2.5-14B (pre-registered in expectations).

Two phases, mirroring the invariant_core_func pipeline on the text-only model:

  ACT  one hooked forward over the frozen corpus: per-layer energy share
       e_U = ||U U^T h||^2 / ||h||^2 at q_proj_in for core32 vs haar32.
       Frozen gate (pre-reg): if median share(core32) <= median share(haar)
       the band is gradient-starved INERT -> STOP, bank the verdict, no
       ablation (the FUNC pipeline's cheap decisive gate).

  ABL  per arm, project the q_proj input onto the complement (x' = x - QQ^T x)
       at EVERY layer; readout = wikitext ppl on the frozen windows. Arms:
       none / core32 / anti32 / haar32 x3 / bottomsigma32 / topsigma32 (from
       base-W SVD, computed in-cell) / topenergy32+128 coordinate sets (banked
       from E7 sums, §5's mandatory structured control).

AMUNDSEN AM-1 adds, without touching any banked arm's construction:

  energymatched32_s{0,1,2}  a RANDOM subspace whose removed activation-energy
       fraction matches core32's, PER LAYER (conserved_core_probes.
       energy_matched_subspace, reused unmodified). This is the arm that
       decides whether "the band is LIVE at 14B" is a core statement or an
       activation-energy statement. The activation Gram it needs is
       accumulated in the same ACT pass and eigendecomposed on device.
  specmatched32_s{0,1,2}    (AM-1b, opt-in via `spectral_control`) the control the
       AM-1 result asks for: energy-matched is not SPECTRUM-matched, because the
       energy-matched draw takes its energy from the leading eigendirections
       while core32 may spread the same energy differently. This arm writes
       core32 in the activation Gram's eigenbasis (A = V^T Q), randomises the
       SIGN of every entry — which leaves each eigendirection's contribution
       |A_ij| and therefore the whole removed-energy profile untouched — and
       re-orthonormalises. Same energy, same spectral placement, different
       directions.
  magmatched32_s{0,1,2}     the planted/magnitude control (standing rule, queue
       item 91): core32's own matrix with every ENTRY sign-randomized, then
       re-orthonormalised — identical per-coordinate magnitude profile, no
       shared direction. If it fires like core32, the coordinate-localization
       reading is dead on arrival.
  per-window NLL for every arm (v1abl_window_nll.json) so the contrast is a
       PAIRED bootstrap over windows rather than a difference of two scalars.
  realized per-arm activation-energy shares (v1abl_arm_shares.json) — the
       energy match is verified by MEASUREMENT, in both directions, never
       assumed from the construction.

Banks arrive from the census S3 prefix at cell start (small, seconds).
Projections computed in fp32 on the bf16 stream (flagged). FAIL LOUD.
"""
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from infra.pg.conserved_core_probes import energy_matched_subspace


def _orth_t(A):
    q, _ = np.linalg.qr(A)
    return torch.as_tensor(np.ascontiguousarray(q), dtype=torch.float32)


def run(cell, out_dir):
    model_dir = cell["model_dir"]
    banks_s3 = cell["banks_s3"]
    topenergy_s3 = cell["topenergy_s3"]
    corpus = np.load(cell.get("corpus_path",
                              os.path.join(model_dir, "shackleton_corpus.npy")))
    n_act = int(cell.get("n_act_windows", 32))
    n_abl = int(cell.get("n_abl_windows", 49))
    n_extra_seeds = int(cell.get("n_extra_seeds", 3))   # ematch / magmatch draws
    spectral = bool(cell.get("spectral_control", False))  # AM-1b's spectrum-matched arm
    # GUTENBERG GB-3 additions (spec AM-0.5.5) — every default preserves the banked behavior:
    # seed_offset shifts the ematch/spec/mag seed SLOTS (fresh independent draws, named by their
    # absolute slot); qk_rider re-runs core/ematch/haar with the projection at q_proj AND k_proj
    # (FUNC's instrument — the AM-0.5.6 bridge); extended_shares measures realized energy shares
    # for EVERY banked arm; bank_vectors uploads the constructed ematch/spec/mag bases + the
    # activation-Gram eigenvalues (GB-2/GB-4 inputs, verified missing 2026-08-19).
    seed_offset = int(cell.get("seed_offset", 0))
    qk_rider = bool(cell.get("qk_rider", False))
    extended_shares = bool(cell.get("extended_shares", False))
    bank_vectors = bool(cell.get("bank_vectors", False))
    K = 32

    bdir = "/workspace/banks"
    subprocess.run(["aws", "s3", "cp", "--recursive", banks_s3, bdir], check=True,
                   capture_output=True)
    subprocess.run(["aws", "s3", "cp", topenergy_s3, "/workspace/topenergy.npz"],
                   check=True, capture_output=True)
    tope = np.load("/workspace/topenergy.npz")

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=dev,
        attn_implementation=cell.get("attn", "sdpa"))
    model.eval()
    layers = model.model.layers
    NL = len(layers)
    d = layers[0].self_attn.q_proj.in_features

    U = {}
    rng = np.random.default_rng(1337)
    for L in range(NL):
        core = np.load(f"{bdir}/q_proj__L{L}__core32.npy")
        anti = np.load(f"{bdir}/q_proj__L{L}__anti32.npy")
        assert core.shape == (d, K), core.shape
        U[("core32", L)] = _orth_t(core)
        U[("anti32", L)] = _orth_t(anti)
        for s in range(3):
            g = rng.standard_normal((d, K))
            U[(f"haar32_s{s}", L)] = _orth_t(g)
        # AM-1 planted control: same |entries|, randomized signs, re-orthonormalised.
        # Seeded per (layer, draw) so the arm is reproducible from the manifest alone.
        for s in range(n_extra_seeds):
            slot = seed_offset + s
            srng = np.random.default_rng(90_000 + 1000 * slot + L)
            signs = srng.choice([-1.0, 1.0], size=core.shape)
            U[(f"magmatched32_s{slot}", L)] = _orth_t(core * signs)
        # GPU SVD: 48 CPU SVDs of 5120^2 cost ~2h billed; on-device ~5 min
        W = layers[L].self_attn.q_proj.weight.detach().to(torch.float32)
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        U[("bottomsigma32", L)] = Vh[-K:, :].T.contiguous().cpu()
        U[("topsigma32", L)] = Vh[:K, :].T.contiguous().cpu()
        del Vh
        for k in (32, 128):
            idx = tope[f"L{L:02d}_top{k}"]
            E = np.zeros((d, k)); E[idx, np.arange(k)] = 1.0
            U[(f"topenergy{k}", L)] = torch.as_tensor(E, dtype=torch.float32)
    for key in U:
        U[key] = U[key].to(dev)

    def ppl_pass(windows, project_arm=None, act_accumulate=None, act_arms=(),
                 gram=None, sites=("q",)):
        """Returns (ppl, per-window nll, per-window ntok).

        act_accumulate/gram are the measurement side-channels: the same forward
        that produces the readout also produces the per-layer energy shares and
        (optionally) the per-layer activation Gram the energy-matched arm is
        built from — one pass, no second load.

        sites: projection hook sites — ("q",) is the banked AM-1 instrument
        (q_proj input only); ("q", "k") is FUNC's instrument (the GB-3 qk rider).
        ACT accumulation always reads at q_proj (the shared post-LN residual).
        """
        handles = []
        if project_arm is not None or act_accumulate is not None:
            for L in range(NL):
                def mk(L):
                    def hook(module, args):
                        h = args[0]
                        h32 = h.to(torch.float32)
                        if act_accumulate is not None:
                            flat = h32.reshape(-1, h32.shape[-1])
                            tot = float(flat.pow(2).sum())
                            for bid in act_arms:
                                p = flat @ U[(bid, L)]
                                act_accumulate[(bid, L)] = act_accumulate.get((bid, L), 0.0) \
                                    + float(p.pow(2).sum())
                            act_accumulate[("_tot", L)] = act_accumulate.get(("_tot", L), 0.0) + tot
                            if gram is not None:
                                gram[L] += flat.T @ flat
                            return None
                        Q = U[(project_arm, L)]
                        h32 = h32 - (h32 @ Q) @ Q.T
                        return (h32.to(h.dtype),) + args[1:]
                    return hook
                mods = [layers[L].self_attn.q_proj]
                if project_arm is not None and "k" in sites:
                    mods.append(layers[L].self_attn.k_proj)
                for mod in mods:
                    handles.append(mod.register_forward_pre_hook(mk(L)))
        nlls, ntoks = [], []
        with torch.no_grad():
            for i in range(0, len(windows), 4):
                ids = torch.as_tensor(windows[i:i + 4], device=dev)
                out = model(input_ids=ids)
                logits = out.logits
                for r in range(ids.shape[0]):
                    lp = torch.log_softmax(logits[r, :-1].to(torch.float32), dim=-1)
                    tgt = ids[r, 1:]
                    nlls.append(float(-lp.gather(-1, tgt[:, None]).sum()))
                    ntoks.append(int(tgt.numel()))
                del out, logits
        for h in handles:
            h.remove()
        nll_a = np.asarray(nlls, dtype=np.float64)
        ntok_a = np.asarray(ntoks, dtype=np.int64)
        return float(np.exp(nll_a.sum() / ntok_a.sum())), nll_a, ntok_a

    # ---- ACT gate (+ the activation Gram the energy-matched arm needs) ----
    gate_arms = ("core32", "haar32_s0", "haar32_s1", "haar32_s2")
    acc = {}
    gram = {L: torch.zeros(d, d, dtype=torch.float32, device=dev) for L in range(NL)}
    ppl_pass(corpus[:n_act], act_accumulate=acc, act_arms=gate_arms, gram=gram)
    shares = {}
    for bid in gate_arms:
        shares[bid] = [acc[(bid, L)] / acc[("_tot", L)] for L in range(NL)]
    med_core = float(np.median(shares["core32"]))
    med_haar = float(np.median(shares["haar32_s0"] + shares["haar32_s1"]
                               + shares["haar32_s2"]))
    act = {"median_share_core32": med_core, "median_share_haar": med_haar,
           "per_layer_core": shares["core32"], "gate": "LIVE-candidate"
           if med_core > med_haar else "INERT-STOP"}
    with open(os.path.join(out_dir, "v1abl_act_gate.json"), "w") as f:
        json.dump(act, f, indent=2)
    if med_core <= med_haar:
        return  # pre-registered stop: gradient-starved INERT, no ablation

    # ---- AM-1: the energy-matched arm, per layer, from THIS run's Gram ----
    # random_floor = k/d is the smallest removed-energy fraction a random
    # k-subspace can have; where core32 sits BELOW it no energy match exists
    # and the construction degenerates to Haar. That is a disclosure, not a
    # failure — it is banked per layer and the layer is flagged.
    random_floor = K / d
    ematch_diag, spec_diag = [], []
    gram_eigvals = {}
    for L in range(NL):
        # fp32 eigh: the construction uses only the top-m eigenspace and the
        # eigenvalue PROPORTIONS, and the realized energy is re-measured below
        # by the same accumulator that measures core32 — fp64 here would cost
        # ~15 min of billed time on an L40S for no decision-relevant precision.
        mu, V = torch.linalg.eigh(gram[L])               # ascending
        mu = mu.flip(0).clamp_min(0.0).to(torch.float64).cpu().numpy()
        V = V.flip(1).to(torch.float64).cpu().numpy()
        if bank_vectors:
            gram_eigvals[L] = mu.copy()
        e_target = shares["core32"][L]
        for s in range(n_extra_seeds):
            slot = seed_offset + s
            Q, e_real = energy_matched_subspace(
                V, mu, e_target, K, seed=70_000 + 1000 * slot + L,
                tol=max(0.02 * e_target, 1e-6), max_resample=200)
            U[(f"energymatched32_s{slot}", L)] = torch.as_tensor(
                np.ascontiguousarray(Q), dtype=torch.float32, device=dev)
            if spectral:
                # A = V^T Q_core: row i is core32's footprint on eigendirection i.
                # Sign-randomising the ENTRIES preserves every |A_ij|, so the
                # per-eigendirection energy profile — and hence the removed
                # energy — is preserved by construction; the QR that follows
                # perturbs it slightly and the realized values are measured.
                A = V.T @ U[("core32", L)].cpu().numpy().astype(np.float64)
                srng = np.random.default_rng(60_000 + 1000 * slot + L)
                As = A * srng.choice([-1.0, 1.0], size=A.shape)
                Qs, _ = np.linalg.qr(V @ As)
                U[(f"specmatched32_s{slot}", L)] = torch.as_tensor(
                    np.ascontiguousarray(Qs[:, :K]), dtype=torch.float32, device=dev)
                spec_diag.append({"layer": L, "seed": slot,
                                  "profile_l1_rel_change": float(
                                      np.abs((V.T @ Qs[:, :K]) ** 2 - A ** 2).sum()
                                      / (A ** 2).sum())})
            if s == 0:
                ematch_diag.append({"layer": L, "e_target": float(e_target),
                                    "e_realized": float(e_real),
                                    "below_random_floor":
                                        bool(e_target < random_floor)})
        del mu, V
        gram[L] = None
    del gram
    torch.cuda.empty_cache()
    matched = [r for r in ematch_diag if not r["below_random_floor"]]
    for r in matched:
        assert abs(r["e_realized"] - r["e_target"]) <= 0.1 * r["e_target"], \
            f"energy match failed at L{r['layer']}: {r}"

    # ---- realized shares for EVERY arm (the two-directional match check) ----
    names = ("energymatched32", "magmatched32") + (("specmatched32",) if spectral else ())
    ext_arms = tuple(f"{n}_s{seed_offset + s}" for n in names for s in range(n_extra_seeds))
    verify_arms = ("core32", "topenergy32") + ext_arms
    if extended_shares:
        # GB-4 needs (cost, energy, gain) triples for EVERY costed bank; the energies for
        # anti/bottomsigma/topsigma/topenergy128/haar existed nowhere (verified 2026-08-19).
        verify_arms = verify_arms + ("anti32", "bottomsigma32", "topsigma32", "topenergy128",
                                     "haar32_s0", "haar32_s1", "haar32_s2")
    acc2 = {}
    ppl_pass(corpus[:n_act], act_accumulate=acc2, act_arms=verify_arms)
    arm_shares = {bid: [acc2[(bid, L)] / acc2[("_tot", L)] for L in range(NL)]
                  for bid in verify_arms}
    with open(os.path.join(out_dir, "v1abl_arm_shares.json"), "w") as f:
        json.dump({"per_layer_share": arm_shares,
                   "median_share": {b: float(np.median(v))
                                    for b, v in arm_shares.items()},
                   "energy_match_per_layer": ematch_diag,
                   "spectral_match_per_layer": spec_diag,
                   "random_floor_k_over_d": random_floor,
                   "n_layers_below_random_floor":
                       int(sum(r["below_random_floor"] for r in ematch_diag))},
                  f, indent=2)

    # ---- ABL contrast ----
    arms = ["none", "core32", "anti32", "haar32_s0", "haar32_s1", "haar32_s2",
            "bottomsigma32", "topsigma32", "topenergy32", "topenergy128"] \
        + list(ext_arms)
    # §9.1 fail-fast: the banked arms must reproduce the BANKED ARTIFACT (fetched
    # here, never a number typed into the manifest) before any new arm bills GPU
    # time. `none` and `core32` come first in the roster, so this costs nothing.
    expect, tol = {}, float(cell.get("expect_tol", 1e-3))
    if cell.get("expect_rows_s3"):
        subprocess.run(["aws", "s3", "cp", cell["expect_rows_s3"],
                        "/workspace/expect_rows.json"], check=True, capture_output=True)
        exp_rows = json.load(open("/workspace/expect_rows.json"))
        expect = {r["arm"]: r["ppl"] for r in exp_rows["arms"]}
        assert exp_rows["n_abl_windows"] == n_abl, \
            f"reproduction gate needs the same window count: {exp_rows['n_abl_windows']} vs {n_abl}"

    rows, wnll = [], {}
    for arm in arms:
        p, nll, ntok = ppl_pass(corpus[:n_abl],
                                project_arm=None if arm == "none" else arm)
        if arm in expect:
            rel = abs(p - expect[arm]) / expect[arm]
            assert rel <= tol, (f"reproduction gate FAILED for {arm}: {p} vs banked "
                                f"{expect[arm]} (rel {rel:.2e} > {tol})")
            print(f"[v1abl] {arm}: reproduces banked {expect[arm]:.6f} "
                  f"(rel {rel:.2e})", flush=True)
        rows.append({"arm": arm, "ppl": p})
        wnll[arm] = {"nll": nll.tolist(), "ntok": ntok.tolist()}
        print(f"[v1abl] {arm}: ppl={p:.4f}", flush=True)
    with open(os.path.join(out_dir, "v1abl_rows.json"), "w") as f:
        json.dump({"arms": rows, "n_abl_windows": n_abl, "k": K,
                   "projection_dtype": "fp32 on bf16 stream",
                   "banks_s3": banks_s3,
                   "reproduction_gate": {"source": cell.get("expect_rows_s3"),
                                         "tol": tol, "checked": sorted(expect)}},
                  f, indent=2)
    with open(os.path.join(out_dir, "v1abl_window_nll.json"), "w") as f:
        json.dump({"per_window": wnll, "n_abl_windows": n_abl,
                   "alignment_key": "window index in the frozen corpus order",
                   "readout": "sum NLL per window (nats), ntok = window-1"},
                  f, indent=2)

    # ---- GB-3 qk rider (AM-0.5.6): the same contrast under FUNC's instrument ----
    if qk_rider:
        qk_arms = ["core32"] \
            + [f"energymatched32_s{seed_offset + s}" for s in range(n_extra_seeds)] \
            + ["haar32_s0", "haar32_s1", "haar32_s2"]
        rows_qk, wnll_qk = [], {"none": wnll["none"]}
        for arm in qk_arms:
            p, nll, ntok = ppl_pass(corpus[:n_abl], project_arm=arm, sites=("q", "k"))
            rows_qk.append({"arm": arm, "ppl": p})
            wnll_qk[arm] = {"nll": nll.tolist(), "ntok": ntok.tolist()}
            print(f"[v1abl:qk] {arm}: ppl={p:.4f}", flush=True)
        with open(os.path.join(out_dir, "v1abl_rows_qk.json"), "w") as f:
            json.dump({"arms": rows_qk, "n_abl_windows": n_abl, "k": K,
                       "sites": "q_proj AND k_proj inputs (FUNC's instrument)",
                       "none_shared_with_q_site": True}, f, indent=2)
        with open(os.path.join(out_dir, "v1abl_window_nll_qk.json"), "w") as f:
            json.dump({"per_window": wnll_qk, "n_abl_windows": n_abl,
                       "alignment_key": "window index in the frozen corpus order"},
                      f, indent=2)

    # ---- GB-2/GB-4 inputs: the constructed bases + Gram eigenvalues (AM-0.5.5) ----
    if bank_vectors:
        banked = {}
        for (bid, L), Q in U.items():
            if bid.startswith(("energymatched32", "specmatched32", "magmatched32")):
                banked[f"{bid}/{L}"] = Q.detach().cpu().numpy().astype(np.float32)
        for L, mu in gram_eigvals.items():
            banked[f"gram_eigvals_desc/{L}"] = mu.astype(np.float64)
        np.savez_compressed(os.path.join(out_dir, "v1abl_gb_banks.npz"), **banked)
        print(f"[v1abl] banked {len(banked)} vector/eigenvalue arrays", flush=True)
