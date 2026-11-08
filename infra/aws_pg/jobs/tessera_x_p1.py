"""TESSERA P1 — the crosscoder PORT preflight, on a synthetic planted-feature pair. Spec §5 P1.

This is the second of the two freeze-BLOCKING checklist items (§0). It answers one question: does
the pg port of `modal_pg_w4_crosscoder`'s harvest/train/stats pipeline recover a KNOWN emergent
fraction? If it does not, every Arm X number is a number about the port.

THE PLANT. Both sides are built from one shared dictionary of `n_shared` random unit directions.
The BEFORE stream is a sparse mixture of shared atoms only. The AFTER stream is the same mixture —
same coefficients, same rows — PLUS `n_new` atoms that fire only on the after side. So the pair is
row-aligned by construction and the ground truth is exact:

    planted emergent fraction = n_new / (n_shared + n_new)

Nothing about the plant is tuned to the crosscoder: the atoms are Haar-random, the sparsity is the
port's own BatchTopK budget, and the coefficients are drawn once and shared between sides.

THE PASS RULE (amendment A8), frozen here before the run and TWO-SIDED:

    recovered ∈ [planted / 2, planted × 2]   AND   fvu_after < 0.5

The bracket is two-sided on purpose. A port that OVER-reports emergent features fails exactly as a
port that under-reports does — a one-sided "recovered > 0" rule would pass a port that labels
everything emergent, which is the failure mode a shared-dictionary method actually has. The FVU leg
is what makes an "emergent ≈ 0" verdict meaningful anywhere in Arm X: a degenerate crosscoder reads
FVU ≈ 1 and reports no emergent features because it reconstructs nothing.

THE NEGATIVE CONTROL. A second pair is built with `n_new = 0` — the after stream is the before
stream exactly. Its recovered emergent fraction is the instrument's own manufactured floor, and it
must come back BELOW the planted pair's. This is the synthetic analogue of Arm X's identity pair,
and without it a port that reports 0.5 emergent on everything would pass the bracket by accident
whenever the plant happened to sit near 0.5.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_x_p1 --no-weights \
      --type g6e.2xlarge,p5.4xlarge --s3-prefix results/tessera_x_p1 --budget-min 120

`--no-weights` matters here: the plant is synthetic, no model is loaded, and the weight sync costs
~6 minutes of GPU for nothing. The first two launches omitted it.

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

D_MODEL = 2048
N_SHARED = 512
N_NEW = 64
N_TOKENS = 220_000          # above tessera_crosscoder.MIN_PAIRED_TOKENS
ACTIVE_PER_TOKEN = 16       # sparse mixture density
SEED = 1337


# ---- PRE-REGISTERED 2026-08-01, BEFORE the noise run (the P1 attribution test) --------------
# The width sweep refuted over-completeness (emergent exactly 0 at dict 1024, 1.8x the true atom
# count). What it could NOT separate is that the plant is NOISELESS and exactly linear:
# fvu_after = 0.0000 at every width means an EXACT factorization was found each time, and in a
# noiseless system nothing selects the cleanly-after-only factorization because the objective is
# already at its floor. Adding per-token noise removes the exact solution.
#
# FROZEN RULE, registered here before the run:
#   DEGENERATE-PLANT  emergent >= planted/2 at ANY width, with fvu_after > 0.01
#                     => the noiseless plant was the problem; the port reproduces the estimator.
#   PORT              emergent < planted/2 at EVERY width, with fvu_after > 0.01 everywhere
#                     => the port genuinely does not reproduce the estimator; Arm X stays blocked.
#   VOID-NOISE        fvu_after <= 0.01 anywhere => the noise did not take, the factorization is
#                     still ~exact, and this run cannot adjudicate. Re-run at higher noise.
# The VOID branch is checked FIRST so a noise level that failed to bite can never read as a finding.
NOISE_VOID_FVU = 0.01


def make_pair(n_new, seed, device, noise=0.0):
    """(acts_before, acts_after) — a row-aligned sparse-dictionary pair with `n_new` after-only
    atoms. Returns fp32 CPU tensors, the same dtype/placement the real harvest produces."""
    import torch
    g = torch.Generator().manual_seed(seed)
    D_sh = torch.randn(N_SHARED, D_MODEL, generator=g)
    D_sh = D_sh / D_sh.norm(dim=1, keepdim=True)
    D_nw = torch.randn(max(n_new, 1), D_MODEL, generator=g)
    D_nw = D_nw / D_nw.norm(dim=1, keepdim=True)

    before = torch.zeros(N_TOKENS, D_MODEL)
    after = torch.zeros(N_TOKENS, D_MODEL)
    CH = 20_000
    for s in range(0, N_TOKENS, CH):
        n = min(CH, N_TOKENS - s)
        idx = torch.randint(0, N_SHARED, (n, ACTIVE_PER_TOKEN), generator=g)
        amp = torch.rand(n, ACTIVE_PER_TOKEN, generator=g) + 0.5
        blk = torch.zeros(n, D_MODEL)
        for j in range(ACTIVE_PER_TOKEN):
            blk.index_add_(0, torch.arange(n), amp[:, j:j + 1] * D_sh[idx[:, j]])
        before[s:s + n] = blk
        if n_new > 0:
            # the after-only atoms: same rows, additional structure the before stream never has
            jdx = torch.randint(0, n_new, (n, 2), generator=g)
            jamp = torch.rand(n, 2, generator=g) + 0.5
            extra = torch.zeros(n, D_MODEL)
            for j in range(2):
                extra.index_add_(0, torch.arange(n), jamp[:, j:j + 1] * D_nw[jdx[:, j]])
            after[s:s + n] = blk + extra
        else:
            after[s:s + n] = blk
    if noise > 0:
        # independent per-token noise on BOTH sides, scaled to the streams' own RMS, so no exact
        # linear factorization exists and the objective has a real floor above zero
        rms = float(before.pow(2).mean().sqrt())
        gn = torch.Generator().manual_seed(seed + 991)
        before = before + noise * rms * torch.randn(before.shape, generator=gn)
        after = after + noise * rms * torch.randn(after.shape, generator=gn)
    return before, after


def main():
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=0,
                    help="A14 setting for the REGISTERED P1 run (0 = A9 default)")
    ap.add_argument("--lr", type=float, default=0.0,
                    help="A14 setting for the REGISTERED P1 run (0 = A9 default)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="per-token noise as a fraction of stream RMS (the PRE-REGISTERED "
                         "attribution test; see the frozen rule above make_pair)")
    ap.add_argument("--budget-sweep", default="",
                    help="PRE-REGISTERED: comma steps:lr pairs, e.g. 4000:1e-4,20000:1e-4,"
                         "4000:1e-3. Tests whether the r~0.5 collapse is a TRAINING-BUDGET "
                         "artifact. FROZEN RULE: emergent >= planted/2 at ANY setting with "
                         "fvu_after > 0.01 => SEPARABLE-WITH-BUDGET (the collapse is budget, Arm X "
                         "may proceed at that setting and the deviation from the A9-frozen "
                         "defaults is recorded as an amendment); otherwise NOT-BUDGET (structural "
                         "to this trainer on this data, Arm X stays blocked and the W4 question "
                         "stands). VOID-NOISE checked first as before.")
    ap.add_argument("--dict-sizes", default="",
                    help="DISCRIMINATOR: comma dict widths to sweep on the planted pair. The "
                         "port-vs-plant question is whether 576 true atoms in a 32768-wide "
                         "dictionary leaves allocation uncontested; a sweep answers it directly. "
                         "Empty = the single port-default run (the registered P1).")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import tessera_crosscoder as XC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    planted = N_NEW / (N_SHARED + N_NEW)
    rep = {"object": "TESSERA P1 — crosscoder port preflight on a planted pair",
           "device": device, "d_model": D_MODEL, "n_shared": N_SHARED, "n_new": N_NEW,
           "n_tokens": N_TOKENS, "active_per_token": ACTIVE_PER_TOKEN,
           "planted_emergent_fraction": planted,
           "train_setting": {"steps": a.steps or XC.CC_TRAIN_STEPS,
                             "lr": a.lr or XC.CC_LR, "noise": a.noise,
                             "amendment": "A14 when steps/lr differ from A9's port defaults"},
           "pass_rule": "recovered ∈ [planted/2, planted×2] AND fvu_after < 0.5 (amendment A8, "
                        "two-sided: an over-reporting port fails exactly as an under-reporting "
                        "one does)",
           "arms": {}}

    def _w():
        (out / "controls.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    if a.budget_sweep:
        rep["budget_sweep"] = {"noise": a.noise, "arms": {}}
        for spec in a.budget_sweep.split(","):
            st, lrv = spec.split(":")
            ab, aa = make_pair(N_NEW, a.seed, device, noise=a.noise)
            ae, cfg = XC.train_crosscoder(ab, aa, 0, a.seed, device,
                                          steps_override=int(st), lr_override=float(lrv))
            fr = XC.pair_fractions(ae, ab, aa, device)
            fv = XC.fvu(ae, ab, aa, device)
            rep["budget_sweep"]["arms"][spec] = {"steps": int(st), "lr": float(lrv), **fr, **fv}
            print(f"[P1-budget] steps={st:<6s} lr={lrv:<7s} emergent={fr['emergent_fraction']:.4f} "
                  f"fvu_after={fv['fvu_after']:.4f} (planted {planted:.4f})", flush=True)
            del ae, ab, aa
            if device == "cuda":
                torch.cuda.empty_cache()
            _w()
        av = rep["budget_sweep"]["arms"].values()
        ems, fvs = [v["emergent_fraction"] for v in av], [v["fvu_after"] for v in av]
        if a.noise > 0 and min(fvs) <= NOISE_VOID_FVU:
            rep["budget_sweep"]["verdict"] = f"VOID-NOISE — fvu fell to {min(fvs):.4g}"
        elif max(ems) >= planted / 2.0:
            rep["budget_sweep"]["verdict"] = (
                "SEPARABLE-WITH-BUDGET — the r~0.5 collapse is a training-budget artifact; Arm X "
                "may proceed at the winning setting, recorded as an amendment to A9")
        else:
            rep["budget_sweep"]["verdict"] = (
                "NOT-BUDGET — the collapse is structural to this trainer on this data at every "
                "budget tried; Arm X stays blocked and the W4 question stands")
        print("[P1-budget] " + rep["budget_sweep"]["verdict"], flush=True)
        _w()
        return

    # ---- the port-vs-plant discriminator (labelled, not the registered P1) --------------------
    if a.dict_sizes:
        rep["discriminator"] = {
            "why": "P1's registered run put 576 true atoms in a 32768-wide dictionary at k=100 — "
                   "57x over-complete, FVU 0.0000, every latent live. With allocation that "
                   "uncontested a latent has no pressure to become after-ONLY, and the r >= 0.90 "
                   "emergent cut cannot fire. Narrowing the dictionary contests allocation. If "
                   "emergent recovers as the width falls, the registered failure was the PLANT; "
                   "if it stays 0 at every width, it is the PORT.",
            "arms": {}}
        for ds in [int(x) for x in a.dict_sizes.split(",")]:
            ab, aa = make_pair(N_NEW, a.seed, device, noise=a.noise)
            ae, cfg = XC.train_crosscoder(ab, aa, 0, a.seed, device, dict_size_override=ds)
            fr = XC.pair_fractions(ae, ab, aa, device)
            fv = XC.fvu(ae, ab, aa, device)
            rep["discriminator"]["arms"][str(ds)] = {"dict_size": ds, **fr, **fv}
            print(f"[P1-disc] dict={ds:<6d} emergent={fr['emergent_fraction']:.4f} "
                  f"n_live={fr['n_live']:<6d} fvu_after={fv['fvu_after']:.4f} "
                  f"(planted {planted:.4f})", flush=True)
            del ae, ab, aa
            if device == "cuda":
                torch.cuda.empty_cache()
            _w()
        arms_ = rep["discriminator"]["arms"].values()
        ems = [v["emergent_fraction"] for v in arms_]
        fvus = [v["fvu_after"] for v in arms_]
        rep["discriminator"]["noise"] = a.noise
        if a.noise > 0 and min(fvus) <= NOISE_VOID_FVU:
            # VOID checked FIRST: a noise level that did not bite must never read as a finding
            rep["discriminator"]["verdict"] = (
                f"VOID-NOISE — fvu_after fell to {min(fvus):.4g} <= {NOISE_VOID_FVU}, so the "
                f"factorization is still ~exact and this run cannot adjudicate. Re-run higher.")
        elif max(ems) >= planted / 2.0:
            rep["discriminator"]["verdict"] = (
                "DEGENERATE-PLANT — emergent recovers, so the noiseless/over-complete plant was "
                "the problem and the port reproduces the estimator")
        else:
            rep["discriminator"]["verdict"] = (
                "PORT — emergent stays below planted/2 at every width with a live FVU, so the "
                "port does not reproduce the estimator and Arm X stays blocked")
        print("[P1-disc] " + rep["discriminator"]["verdict"], flush=True)
        _w()
        return

    for name, n_new in (("planted", N_NEW), ("negative_control", 0)):
        ab, aa = make_pair(n_new, a.seed, device, noise=a.noise)
        assert ab.shape == aa.shape == (N_TOKENS, D_MODEL), f"{name}: bad pair shape"
        ae, cfg = XC.train_crosscoder(ab, aa, 0, a.seed, device,
                                      steps_override=(a.steps or None),
                                      lr_override=(a.lr or None))
        fr = XC.pair_fractions(ae, ab, aa, device)
        fv = XC.fvu(ae, ab, aa, device)
        rep["arms"][name] = {"n_new": n_new, "cfg": cfg, **fr, **fv}
        print(f"[P1] {name:<17s} emergent={fr['emergent_fraction']:.4f} "
              f"n_live={fr['n_live']} fvu_after={fv['fvu_after']:.4f}", flush=True)
        del ae, ab, aa
        if device == "cuda":
            torch.cuda.empty_cache()
        _w()

    rec = rep["arms"]["planted"]["emergent_fraction"]
    fvu_after = rep["arms"]["planted"]["fvu_after"]
    neg = rep["arms"]["negative_control"]["emergent_fraction"]
    in_bracket = (planted / 2.0) <= rec <= (planted * 2.0)
    fvu_ok = fvu_after < 0.5
    beats_neg = rec > neg
    rep["P1"] = {
        "planted": planted, "recovered": rec, "bracket": [planted / 2.0, planted * 2.0],
        "in_bracket": bool(in_bracket), "fvu_after": fvu_after, "fvu_ok": bool(fvu_ok),
        "negative_control_emergent": neg, "beats_negative_control": bool(beats_neg),
        "verdict": ("PASS" if (in_bracket and fvu_ok and beats_neg) else "FAIL"),
        "why": ("the port recovers the planted emergent fraction inside a two-sided 2× bracket, "
                "reconstructs well below the degenerate FVU, and separates the planted pair from "
                "an identical-sides pair" if (in_bracket and fvu_ok and beats_neg) else
                "the port does not reproduce the planted fraction under the frozen rule — Arm X "
                "is BLOCKED (§5 P1: port drift → halt-fix), and no Arm X number may be read")}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TESSERA_P1 " + json.dumps(rep["P1"], indent=1), flush=True)
    assert rep["P1"]["verdict"] == "PASS", (
        f"P1 FAILED: recovered {rec:.4f} vs planted {planted:.4f} "
        f"(bracket {planted/2:.4f}-{planted*2:.4f}), fvu_after {fvu_after:.4f}, "
        f"negative control {neg:.4f} — Arm X is blocked and the freeze does not clear")


if __name__ == "__main__":
    main()
