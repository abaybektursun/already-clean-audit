"""MENDEL MN-6 — the LoRA column (queue item 18).

THE QUESTION. The corpus behind "fine-tuning avoids the sink" is entirely FULL fine-tunes. Adam's
divisor acts on LoRA's A/B factor coordinates, not on W's, so the pinning mechanism may simply not
bind for the adaptation method customers actually use. Do ΔW = BA deltas respect the core?

KILL, verbatim: "LoRA deltas deliver into the core at matched-null rates → the storage threat model
re-scopes to full-FT attackers only, and every 'all fine-tunes avoid it' sentence takes a full-FT
qualifier."

THE INPUT THE QUEUE NAMES DOES NOT EXIST, and that is worth stating plainly. Queue item 18 says
"one CPU pass over cached adapter deltas (banked LoRA corpora; X1a survivors post-dedup)". Checked
at the artifact: `copernicus_x1a2/out/copernicus_x1a.json` is a census of **full** fine-tunes of
`lerobot/smolvla_base`, and it explicitly REJECTED LoRA repos — `reject_reasons` contains
"adapter/LoRA repo (delta would measure the adapter's rank)": 2. No S3 prefix in the bucket
contains a LoRA/PEFT/adapter corpus. So this arm BUILDS the corpus rather than reading one, from
HF adapters whose declared base is PG1 — the base MENDEL already has core32 banks for, which is
what makes the comparison to the full-FT result apples-to-apples.

OWNER-DEDUP (the queue asks for it, and the census shows why): in the SmolVLA census one owner
contributed 41 of 331 repos. One adapter per owner, so a single prolific uploader's near-duplicate
runs cannot carry the verdict.

THE CONTRAST is MN-2's, reused verbatim: write fraction ‖ΔW·U‖_F/‖ΔW‖_F into the sink bank, against
an ENERGY-MATCHED subspace of the same dimension and a haar band. Never haar alone (§5).

FROZEN RULE. LORA-RESPECTS-CORE iff the sink write fraction is BELOW the energy-matched band on
>= 2 types at >= 2 layers (the same shape as MN-2's rule); LORA-DELIVERS-INTO-CORE iff the sink
fraction is at or above the energy-matched band as often as not; else AMBIGUOUS.

CPU only. Downloads adapters (a few MB each), not base models.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_transport as MT

ARM = "MN-6"
BASE = "google/paligemma-3b-pt-224"
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
TYPES = H.TYPES_RESID
K = H.BANK_K
N_HAAR = 8
E_TOL = 0.03


def census(api, *, limit: int, max_per_owner: int = 1) -> dict:
    """HF adapters declaring PG1 as their base, deduped by owner.

    Returns the roster AND the rejects, because a census that reports only survivors hides the
    denominator.
    """
    found = list(api.list_models(filter=(f"base_model:adapter:{BASE}",), limit=limit))
    by_owner, kept, rejected = collections.Counter(), [], []
    for m in found:
        owner = m.id.split("/")[0]
        if by_owner[owner] >= max_per_owner:
            rejected.append({"repo": m.id, "reason": f"owner cap {max_per_owner}"})
            continue
        by_owner[owner] += 1
        kept.append(m.id)
    return {"n_found": len(found), "n_owners": len(by_owner), "kept": kept,
            "rejected": rejected, "max_per_owner": max_per_owner}


def lora_deltas(repo: str, types, layers):
    """{(type, layer): ΔW} from one PEFT adapter, or ({}, reason) if it carries none we can use.

    ΔW = (alpha / r) · B @ A, with A of shape (r, in) and B of shape (out, r). The scaling is read
    from `adapter_config.json` — a delta computed without it is wrong by a constant that varies
    across repos, which would silently make the corpus incomparable.

    `use_rslora` rescales by alpha/sqrt(r) instead; honoured because ignoring it is the same class
    of error.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = json.loads(pathlib.Path(hf_hub_download(repo, "adapter_config.json")).read_text())
    r = int(cfg.get("r", 0))
    alpha = float(cfg.get("lora_alpha", 0))
    if r <= 0 or alpha <= 0:
        return {}, f"unusable rank/alpha (r={r}, alpha={alpha})", cfg
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    sd = load_file(hf_hub_download(repo, "adapter_model.safetensors"))
    a_keys = {k: v for k, v in sd.items() if "lora_A" in k}
    out = {}
    for t in types:
        sub = "self_attn" if t[0] in "qkvo" else "mlp"
        for L in layers:
            needle = f"layers.{L}.{sub}.{t}."
            ak = [k for k in a_keys if needle in k and "vision_tower" not in k]
            if len(ak) != 1:
                continue
            A = sd[ak[0]]
            bk = ak[0].replace("lora_A", "lora_B")
            if bk not in sd:
                continue
            B = sd[bk]
            if A.shape[1] != H.IN_DIM[t]:
                continue                       # a vision-tower or resized module, not our axis
            out[(t, L)] = (B.to(torch.float64) @ A.to(torch.float64) * scale).numpy()
    return out, (None if out else "no LM-decoder LoRA pairs on our coordinate axis"), cfg


def energy_matched(act, k: int, e_target: float, seed: int):
    """A k-subspace matched on REMOVED ACTIVATION ENERGY, built in the coordinate basis.

    Via `mendel_core.energy_matched_control`, which wraps the reused
    `conserved_core_probes.energy_matched_subspace` and returns HOW WELL it matched. The wrapper
    exists because the underlying routine takes a `tol` that reads like a guarantee and is not: it
    returns its closest attempt silently, and on this substrate it missed by up to 350x. A control
    that captures 350x the target energy is not a control, so the match quality is carried into the
    metrics and gated rather than assumed.

    The Gram it is handed is the DIAGONAL per-coordinate activation energy from G4 rather than a
    full activation Gram — stated rather than glossed, because it makes the control
    coordinate-aligned. That is the right control for this contrast: the question is whether ΔW's
    energy lands in the sink more than in an equally-energetic subspace, and the budget is
    per-coordinate.
    """
    return MC.energy_matched_control(act, k, e_target, seed, tol=E_TOL)


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-6 — do LoRA deltas respect the core?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400, help="HF candidates to census")
    ap.add_argument("--max-adapters", type=int, default=40)
    ap.add_argument("--max-per-owner", type=int, default=1)
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn6", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn6"), out)

    vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    banks = WT.fetch_banks(H.cache_dir(out, "banks"))
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))

    from huggingface_hub import HfApi
    api = HfApi()
    roster = census(api, limit=a.limit, max_per_owner=a.max_per_owner)
    print(f"[mn6] census: {roster['n_found']} found, {roster['n_owners']} owners, "
          f"{len(roster['kept'])} after owner-dedup", flush=True)

    # ---- the planted positive control (§5): the instrument must detect a KNOWN co-location ----
    dW_p, U_p = MT.planted_colocation(n=H.IN_DIM["q_proj"], k=K, out_dim=64, seed=a.seed,
                                      strength=0.81)
    from infra.pg.endow_writefrac import write_fraction
    void = MT.instrument_is_void(float(write_fraction(dW_p, U_p)), 0.9)
    gates.record("mn6_instrument_detects_a_planted_colocation", not void["void"], **void)

    used, skipped, cells, matches = [], [], [], []
    for repo in roster["kept"]:
        if len(used) >= a.max_adapters:
            break
        deltas, reason, cfg = lora_deltas(repo, TYPES, layers)
        if not deltas:
            skipped.append({"repo": repo, "reason": reason})
            print(f"[mn6] skip {repo}: {reason}", flush=True)
            continue
        used.append({"repo": repo, "n_cells": len(deltas), "r": cfg.get("r"),
                     "alpha": cfg.get("lora_alpha"), "rslora": bool(cfg.get("use_rslora"))})
        for (t, L), dW in sorted(deltas.items()):
            U = WT.load_bank(banks, t, L, "core32")
            act = np.load(vec / f"act_rms__{t}__L{L}.npy")
            e_sink = float(((act ** 2)[:, None] * U ** 2).sum() / (act ** 2).sum())
            seed = H.cell_seed(a.seed, ARM, repo, t, L)
            Ue, match = energy_matched(act, K, e_sink, seed)
            matches.append(match)
            rng = np.random.default_rng(seed + 7)
            haar = [np.linalg.qr(rng.standard_normal((H.IN_DIM[t], K)))[0] for _ in range(N_HAAR)]
            c = MT.colocation_cell(dW, U_sink=U, U_energy=Ue, U_haar_list=haar)
            band = MC.exceeds_band(c["sink"], min(c["energy"], c["haar_lo"]),
                                   max(c["energy"], c["haar_hi"]))
            cells.append({"repo": repo, "type": t, "layer": L, **{k: c[k] for k in
                          ("sink", "energy", "haar_mean", "haar_lo", "haar_hi",
                           "delta_vs_energy")},
                          "e_sink_target": e_sink, "e_matched_actual": match["e_achieved"],
                          "energy_match_rel_error": match["rel_error"],
                          "energy_matched_ok": match["matched"],
                          "above_band": band["above_band"], "below_band": band["below_band"]})
            for name, val, cid in (("sink", c["sink"], MC.TREATMENT),
                                   ("energy_matched", c["energy"], MC.STRUCTURED),
                                   ("haar", c["haar_mean"], MC.HAAR)):
                at.add("lora_write_fraction", value=val, layer=L, type=t, bank_id=name,
                       item_id=repo, control_id=cid, model_id=repo,
                       flags={"k": K, "e_sink_target": e_sink,
                              "e_matched_actual": match["e_achieved"],
                              "energy_match_rel_error": match["rel_error"],
                              "energy_matched_ok": match["matched"],
                              "lora_r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha")})
        print(f"[mn6] {len(used)}/{a.max_adapters} {repo}: {len(deltas)} cells", flush=True)

    assert cells, "no LoRA adapter yielded a usable decoder delta on our coordinate axis"
    # ---- the shuffle member of the trinity: repo-label permutation on the sink fraction ----
    sink = [c["sink"] for c in cells]
    energy = [c["energy"] for c in cells]
    delta = MC.paired_delta_ci(sink, energy, seed=a.seed)

    # THE FROZEN RULE IS COMPUTED ON ALL CELLS ABOVE AND IS NOT SWAPPED (§6/e). What follows is the
    # robustness pair it needs, because a cell whose energy-matched control missed its target is a
    # broken measurement of that rule rather than a datum for it:
    #   (a) the same contrast restricted to cells where the match actually held, and
    #   (b) the contrast against the HAAR band, which has no energy-matching dependency at all.
    # Both are recorded whatever they say (§6/b banks both readings).
    match_report = MC.energy_match_report(matches)
    ok = [c for c in cells if c["energy_matched_ok"]]
    delta_matched = (MC.paired_delta_ci([c["sink"] for c in ok], [c["energy"] for c in ok],
                                        seed=a.seed) if len(ok) >= 8 else None)
    delta_haar = MC.paired_delta_ci(sink, [c["haar_mean"] for c in cells], seed=a.seed)
    repos = sorted({c["repo"] for c in cells})
    if len(repos) >= 2:
        half = set(repos[: len(repos) // 2])
        perm = MC.permutation_pvalue(sink, ["a" if c["repo"] in half else "b" for c in cells],
                                     seed=a.seed)
        at.add_ci("lora_write_fraction", [c["sink"] for c in cells if c["repo"] in half],
                  bank_id="sink", control_id=MC.SHUFFLE, seed=a.seed,
                  flags={"p_repo_label_permutation": perm["p_value"], "method": perm["method"]})
    else:
        perm = None

    below = collections.Counter((c["type"], c["layer"]) for c in cells if c["below_band"])
    above = collections.Counter((c["type"], c["layer"]) for c in cells if c["above_band"])
    n_below_types = len({t for t, _ in below})
    n_below_layers = len({L for _, L in below})
    respects = n_below_types >= 2 and n_below_layers >= 2 and delta["ci_hi"] < 0
    delivers = len(above) >= len(below)
    verdict = MP.check_verdict(ARM, "LORA-RESPECTS-CORE" if respects else
                               ("LORA-DELIVERS-INTO-CORE" if delivers else "AMBIGUOUS"))
    gates.record("mn6_sink_below_energy_matched", bool(delta["ci_hi"] < 0), **delta)
    gates.record("mn6_energy_controls_actually_matched",
                 match_report["fraction_matched"] >= 0.9, **match_report)
    gates.record("mn6_survives_on_matched_cells_only",
                 bool(delta_matched is not None and delta_matched["ci_hi"] < 0),
                 n_cells=len(ok), **(delta_matched or {}))
    gates.record("mn6_survives_against_the_haar_band", bool(delta_haar["ci_hi"] < 0), **delta_haar)
    at.check_control_trinity(["lora_write_fraction"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"], "base": BASE,
        "census": {k: v for k, v in roster.items() if k != "kept"},
        "n_adapters_used": len(used), "adapters": used, "n_skipped": len(skipped),
        "skipped": skipped[:20], "n_cells": len(cells),
        "sink_vs_energy": delta, "repo_shuffle": perm,
        "energy_match_report": match_report,
        "sink_vs_energy_matched_cells_only": delta_matched,
        "sink_vs_haar_band": delta_haar,
        "energy_match_note": (
            "`conserved_core_probes.energy_matched_subspace` returns its closest attempt whether or "
            "not it met tol. Where the sink's energy share is structurally unreachable by any "
            "k-subspace (q_proj L0 targets 4.5e-5 of 2048 dims) the 'matched' control captures "
            "orders of magnitude more, which biases TOWARD this arm's conclusion. The frozen rule "
            "is still scored on all cells; the two robustness contrasts beside it say whether the "
            "unmatched cells were load-bearing."),
        "n_cells_below_band": sum(below.values()), "n_cells_above_band": sum(above.values()),
        "below_types": n_below_types, "below_layers": n_below_layers,
        "cells": cells, "atlas": info,
        "input_note": ("queue item 18 names 'X1a survivors' as the corpus; X1a is a census of FULL "
                       "fine-tunes of lerobot/smolvla_base that explicitly rejected LoRA repos, and "
                       "no LoRA corpus is banked in the bucket. This arm builds one from HF "
                       "adapters declaring PG1 as base — the base MENDEL has core32 banks for.")})
    print("MN6_OK " + json.dumps({"verdict": verdict, "n_adapters": len(used),
                                  "n_cells": len(cells),
                                  "sink_minus_energy": round(delta["delta_mean"], 5)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
