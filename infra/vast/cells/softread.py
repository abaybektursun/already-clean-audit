"""AM-3 — the soft-read dose ladder: does the model READ the sink coordinates?

Ablation asks "is the subspace load-bearing when deleted". That is a
write/erasure question. This cell asks the READ question: multiply the
EFFECTIVE norm gain on the subspace's directions by s and watch the readout
move. s = 1 is the identity; s < 1 attenuates the sink component of the stream
the readers see; s > 1 amplifies it.

  SITE. gemma's `input_layernorm` OUTPUT — the tensor q_proj/k_proj/v_proj all
  read, i.e. the effective-gain surface itself. For an axis-aligned U the hook
  below is EXACTLY equivalent to rewriting the RMSNorm weight so the effective
  gain (1+w) becomes s*(1+w) on those coordinates. THE GEMMA TRAP: gemma's
  RMSNorm gain is (1+w), never w; scaling w by s is a different (and wrong)
  intervention. dryrun_amundsen.py asserts the equivalence on tiny tensors and
  asserts the wrong form FAILS it.

  DOSE MATCHING. Twins are matched on REALIZED activation-perturbation norm,
  not on nominal s (standing guard; REFEREE's Haar under-dose VOID is the
  precedent). Delta_h = (s-1) P_U h, so ||Delta_h||^2 = (s-1)^2 * share_U *
  ||h||^2 per layer: the matched twin scale is
      s_twin(L) = 1 + sign(s-1) * |s-1| * sqrt(share_core(L)/share_twin(L)).
  A twin with far less energy than core therefore needs a LARGER excursion —
  including, at strong core attenuation, a negative effective gain. That is
  what matched motion means here; every realized scale is banked, rungs with
  |s_twin-1| > 3 are flagged, and the NOMINAL (unmatched) ladder is banked
  alongside as the second reading.

  ARMS. core32 (the object) · energymatched32 (twin, equal removed energy by
  construction) · haar32 (twin, matched motion only) · topenergy32 (POSITIVE
  CONTROL: the ladder must move ppl monotonically on it, else the instrument
  has no range and the arm is VOID-INSTRUMENT).

Pure inference, no training. Every subspace is built in-cell from this run's
own activation Gram (accumulated in the calibration pass) plus the banked
census core32. FAIL LOUD.
"""
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from infra.pg.conserved_core_probes import energy_matched_subspace

LADDER = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _orth(A):
    q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    return q


def run(cell, out_dir):
    model_dir = cell["model_dir"]
    corpus = np.load(cell.get("corpus_path",
                              os.path.join(model_dir, "shackleton_corpus.npy")))
    n_cal = int(cell.get("n_cal_windows", 32))
    n_ppl = int(cell.get("n_ppl_windows", 98))
    ladder = tuple(float(x) for x in cell.get("ladder", LADDER))
    bs = int(cell.get("batch", 4))    # gemma vocab is 256k: keep logits small
    K = 32

    bdir = "/workspace/g2b_banks"
    subprocess.run(["aws", "s3", "cp", "--recursive", cell["banks_s3"], bdir],
                   check=True, capture_output=True)

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=dev,
        attn_implementation=cell.get("attn", "eager"))   # gemma: eager (house)
    model.eval()
    layers = model.model.layers
    NL = len(layers)
    d = layers[0].self_attn.q_proj.in_features
    norms = [layers[L].input_layernorm for L in range(NL)]

    # ---------- pass 0: the activation Gram at the effective-gain surface ----
    gram = {L: torch.zeros(d, d, dtype=torch.float32, device=dev) for L in range(NL)}

    def gram_hook(L):
        def hook(module, args, output):
            flat = output.to(torch.float32).reshape(-1, d)
            gram[L] += flat.T @ flat
        return hook

    handles = [norms[L].register_forward_hook(gram_hook(L)) for L in range(NL)]
    with torch.no_grad():
        for i in range(0, n_cal, bs):
            model(input_ids=torch.as_tensor(corpus[i:i + bs], device=dev))
    for h in handles:
        h.remove()

    # ---------- the four subspaces, per layer ----------
    U, share = {}, {}
    diag_rows = []
    for L in range(NL):
        G = gram[L]
        tot = float(torch.diagonal(G).sum())
        core = _orth(np.load(f"{bdir}/q_proj__L{L}__core32.npy"))
        assert core.shape == (d, K), core.shape
        # top-energy coordinate set from THIS run's Gram diagonal (= sum h_j^2)
        idx = torch.topk(torch.diagonal(G), K).indices.cpu().numpy()
        tope = np.zeros((d, K)); tope[idx, np.arange(K)] = 1.0
        rng = np.random.default_rng(4200 + L)
        haar = _orth(rng.standard_normal((d, K)))

        Gn = G.cpu().numpy().astype(np.float64)
        e_core = float(np.trace(core.T @ Gn @ core) / tot)
        mu, V = torch.linalg.eigh(G)
        mu = mu.flip(0).clamp_min(0.0).to(torch.float64).cpu().numpy()
        V = V.flip(1).to(torch.float64).cpu().numpy()
        emat, e_emat = energy_matched_subspace(V, mu, e_core, K, seed=8100 + L,
                                               tol=max(0.02 * e_core, 1e-6),
                                               max_resample=200)
        for name, Q in (("core32", core), ("energymatched32", emat),
                        ("haar32", haar), ("topenergy32", tope)):
            U[(name, L)] = torch.as_tensor(np.ascontiguousarray(Q),
                                           dtype=torch.float32, device=dev)
            share[(name, L)] = float(np.trace(Q.T @ Gn @ Q) / tot)
        diag_rows.append({"layer": L, "e_core": e_core, "e_ematch_realized": e_emat,
                          "shares": {n: share[(n, L)] for n in
                                     ("core32", "energymatched32", "haar32",
                                      "topenergy32")},
                          "below_random_floor": bool(e_core < K / d)})
        del G, Gn, mu, V
        gram[L] = None
    del gram
    torch.cuda.empty_cache()

    # ---------- the dose hook ----------
    state = {"arm": None, "scale": {}, "dh2": 0.0, "h2": 0.0}

    def dose_hook(L):
        def hook(module, args, output):
            s = state["scale"].get(L)
            if s is None or state["arm"] is None:
                return None
            Q = U[(state["arm"], L)]
            o32 = output.to(torch.float32)
            delta = (s - 1.0) * ((o32 @ Q) @ Q.T)
            state["dh2"] += float(delta.pow(2).sum())
            state["h2"] += float(o32.pow(2).sum())
            return (o32 + delta).to(output.dtype)
        return hook

    hooks = [norms[L].register_forward_hook(dose_hook(L)) for L in range(NL)]

    def ppl_pass(arm, scale_by_layer):
        state.update({"arm": arm, "scale": scale_by_layer, "dh2": 0.0, "h2": 0.0})
        nlls, ntoks = [], []
        with torch.no_grad():
            for i in range(0, n_ppl, bs):
                ids = torch.as_tensor(corpus[i:i + bs], device=dev)
                logits = model(input_ids=ids).logits
                for r in range(ids.shape[0]):
                    lp = torch.log_softmax(logits[r, :-1].to(torch.float32), dim=-1)
                    tgt = ids[r, 1:]
                    nlls.append(float(-lp.gather(-1, tgt[:, None]).sum()))
                    ntoks.append(int(tgt.numel()))
                del logits
        nll = np.asarray(nlls, dtype=np.float64)
        ntok = np.asarray(ntoks, dtype=np.int64)
        realized = float(np.sqrt(state["dh2"] / state["h2"])) if state["h2"] else 0.0
        state["arm"] = None
        return {"ppl": float(np.exp(nll.sum() / ntok.sum())), "nll": nll.tolist(),
                "ntok": ntok.tolist(), "realized_rel_dh": realized}

    rows = []
    base = ppl_pass(None, {})
    rows.append({"arm": "none", "s": 1.0, "mode": "identity", **base})
    print(f"[softread] baseline ppl={base['ppl']:.4f}", flush=True)

    for s in ladder:
        for arm in ("core32", "energymatched32", "haar32", "topenergy32"):
            modes = {"nominal": {L: s for L in range(NL)}}
            if arm != "core32":
                modes["matched"] = {
                    L: 1.0 + np.sign(s - 1.0) * abs(s - 1.0)
                    * float(np.sqrt(share[("core32", L)] / share[(arm, L)]))
                    for L in range(NL)}
            for mode, scales in modes.items():
                r = ppl_pass(arm, scales)
                far = max(abs(v - 1.0) for v in scales.values())
                rows.append({"arm": arm, "s": s, "mode": mode, **r,
                             "scale_min": min(scales.values()),
                             "scale_max": max(scales.values()),
                             "far_dose_flag": bool(far > 3.0)})
                print(f"[softread] {arm} s={s} ({mode}): ppl={r['ppl']:.4f} "
                      f"rel_dh={r['realized_rel_dh']:.4f}", flush=True)

    for h in hooks:
        h.remove()

    ident = [r for r in rows if r["s"] == 1.0 and r["mode"] == "nominal"]
    ident_ok = all(abs(r["ppl"] - base["ppl"]) / base["ppl"] < 1e-3 for r in ident)
    with open(os.path.join(out_dir, "softread_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "softread_config.json"), "w") as f:
        json.dump({"ladder": list(ladder), "n_cal_windows": n_cal,
                   "n_ppl_windows": n_ppl, "k": K, "d": d, "n_layers": NL,
                   "site": "input_layernorm output (effective-gain surface)",
                   "gain_convention": "gemma (1+w); dose scales the EFFECTIVE gain",
                   "banks_s3": cell["banks_s3"], "identity_control_ok": ident_ok,
                   "per_layer": diag_rows}, f, indent=2)
    assert ident_ok, "identity control failed: s=1.0 did not reproduce baseline ppl"
