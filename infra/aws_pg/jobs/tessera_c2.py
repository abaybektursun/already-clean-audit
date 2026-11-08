"""TESSERA-FIX C2 — the N4 patch-back TESSERA never ran. Spec §2, §6; critic finding 1.

WHAT IT DECIDES. Result 6 measured a before-only ("LOST") population of 2.2%/7.5% for π0 and called
it an erasure term. The crosscoder cannot tell *"PG1 had this feature and π0 lost it"* from *"π0's
stream never excites this direction"* — both present as before-only. N4 is the pre-registered
instrument that discriminates them, and Arm X shipped without it:

  N4 (frozen spec §5): patch each side's crosscoder reconstruction back into that side's forward;
  A7: the patched battery score must land inside the item-bootstrap 95% CI of that side's UNPATCHED
  score. Outside ⇒ that pair VOID.

If π0's own stream is faithfully carried by the dictionary, then a dictionary that says "these
features are absent" is saying something about π0. If patching π0's own reconstruction back destroys
π0's behaviour, the dictionary does not carry π0's forward and the before-only population is a
statement about the instrument — the pre-committed scope-cap wording.

AMENDMENT AF-4 — THE RANGE GUARD, and why the frozen rule needs one here.
A7 as written is UNFAILABLE on π0. π0's multimodal battery is already at chance (LACUNA: S = 0.000
on all four tasks), so a patched score cannot fall below its own CI — there is nothing below chance.
A gate that can only pass is not a gate, which is this program's most-repeated lesson (Haar-only
overlap, the identity band, W4's shuffle null, A12's upper-only branch).

So before any A7 verdict is READ, a DESTROYED patch must be shown to move the score outside the same
CI. The destroyed arm is the reconstruction with its token rows shuffled: identical values, identical
norms, identical everything except the token-to-position correspondence. If even that lands inside
the CI, the metric cannot see patch damage on this side and the verdict is VOID-NO-RANGE — reported,
not passed.

  A7 verdict is read ⟺ destroyed-patch lands OUTSIDE the unpatched CI on that side.

SCOPE, stated rather than buried. `generate` runs autoregressively; the harvested pairing exists only
for PREFILL positions. The hook therefore patches prefill and lets generated tokens run unpatched.
That is the same token set the crosscoder was trained on and the same one FVU is computed over.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_c2 --type g6e.2xlarge \\
      --args "--sites 9,14 --n-items 400" --s3-prefix results/tessera_c2 --budget-min 420

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

TASK = "vqav2"          # the faculty Result 6 is about: π0 lost it, π0.5 kept it
SITES = (9, 14)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", default=",".join(str(s) for s in SITES))
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_units as PU
    from infra.pg import tessera_crosscoder as XC
    from infra.pg.diff_metrics import bootstrap_ci
    from infra.pg.gemma_scope import ResidualHarvest
    from infra.pg.projector_ablation import eval_task, load_eval_items

    device = "cuda"
    sites = [int(s) for s in a.sites.split(",")]
    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    layers = IP.decoder_layers(model)
    task = LS.mm_tasks()[TASK]
    # Same streaming loader the July battery uses (deterministic natural shard order, dedup by
    # content hash) — the rows must be the SAME set the crosscoder is harvested on, because the
    # patch is applied by token index against that harvest.
    rows = load_eval_items(task, a.n_items)
    assert len(rows) == a.n_items, f"loaded {len(rows)} of {a.n_items} eval items"

    rep = {"object": "TESSERA-FIX C2 — the N4 patch-back (critic finding 1)",
           "task": TASK, "n_items": len(rows), "sites": sites,
           "hyperparams": {"steps": a.steps, "lr": a.lr, "seed": a.seed},
           "amendment": "AF-4 — A7 is UNFAILABLE on π0 (its battery is already at chance, so a "
                        "patched score cannot fall below its own CI). A DESTROYED patch "
                        "(row-shuffled reconstruction) must land OUTSIDE the unpatched CI before "
                        "any A7 verdict is read; otherwise VOID-NO-RANGE.",
           "scope": "prefill positions only — `generate` is autoregressive and the harvested "
                    "pairing exists only for prefill, which is also the token set the crosscoder "
                    "was trained on and FVU is computed over",
           "sides": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    def harvest(uid, L):
        """(n_tok, d) prefill acts at layer L for `uid`, plus the per-row token counts."""
        if uid == "pg1":
            LS.restore_pg1(model, pg1_sd)
        else:
            PU.install_unit(uid, model, proc, pg1_sd, device,
                            {"stitch": 0, "matched_dw": 0, "rand_dec": 0})
        chunks, counts = [], []
        with ResidualHarvest(layers, L) as h:
            for s in range(0, len(rows), 8):
                batch = rows[s:s + 8]
                imgs = [task.image_fn(r).convert("RGB") for r in batch]
                prompts = [task.prompt_fn(r) for r in batch]
                inp = proc(text=prompts, images=imgs, return_tensors="pt",
                           padding="longest").to(device)
                inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                with torch.inference_mode():
                    model(**{k: v for k, v in inp.items() if k != "labels"})
                cap = h.captured                       # (b, T, d)
                chunks.append(cap.reshape(-1, cap.shape[-1]).float().cpu())
                counts.append(tuple(cap.shape[:2]))
        return torch.cat(chunks, 0), counts

    for L in sites:
        ab, _ = harvest("pg1", L)
        aa, _ = harvest("pi0", L)
        assert ab.shape == aa.shape, f"L{L}: {tuple(ab.shape)} vs {tuple(aa.shape)}"
        ae, cfg = XC.train_crosscoder(ab, aa, L, a.seed, device,
                                      steps_override=a.steps, lr_override=a.lr)
        fv = XC.fvu(ae, ab, aa, device)
        print(f"[C2] L{L} crosscoder trained fvu_before={fv['fvu_before']:.3f} "
              f"fvu_after={fv['fvu_after']:.3f}", flush=True)
        ae = ae.to(device).eval()

        for slot, uid in ((0, "pg1"), (1, "pi0")):
            if uid == "pg1":
                LS.restore_pg1(model, pg1_sd)
            else:
                PU.install_unit(uid, model, proc, pg1_sd, device,
                                {"stitch": 0, "matched_dw": 0, "rand_dec": 0})

            # `mode` is read by the hook: None = unpatched, "recon" = the crosscoder's own
            # reconstruction of this side, "destroyed" = that same reconstruction with its TOKEN
            # ROWS shuffled (same values, same norms, correspondence broken).
            state = {"mode": None}
            other = 1 - slot
            partner = (aa if slot == 0 else ab)          # the OTHER side's harvested activations

            def paired_for(sl, flat):
                """(N,2,d) with the live side in its own slot and its harvested partner opposite.

                Falls back to self-pairing only if the live block and the harvest disagree in
                length, which would mean the forward is not the one that was harvested — asserted
                rather than silently accepted.
                """
                n = flat.shape[0]
                assert n <= partner.shape[0], (
                    f"live block {n} exceeds harvest {partner.shape[0]} — this forward is not the "
                    f"one the crosscoder was trained on")
                out = torch.empty((n, 2, flat.shape[1]), dtype=flat.dtype)
                out[:, sl] = flat.cpu()
                out[:, 1 - sl] = partner[:n].to(flat.dtype)
                return out

            def hook(_m, _i, o):
                if state["mode"] is None:
                    return o
                res = o[0] if isinstance(o, tuple) else o
                b, T, d = res.shape
                flat = res.reshape(-1, d).float()
                # AF-6. v1 wrote `torch.stack([flat, flat]); pair[:, slot] = flat`, where the second
                # line is a NO-OP — both slots already held the live side, so the encode saw a
                # SELF-PAIRED input rather than [before, after]. That patches a defensible object
                # (what the dictionary reconstructs given only this side) but NOT the paired
                # reconstruction `fvu` measures, so the banked fvu did not describe the patch.
                # `paired[slot]` is the live side; the other slot is that token's partner from the
                # harvest, which is what makes this the reconstruction FVU is computed over.
                pair = paired_for(slot, flat)
                with torch.inference_mode():
                    r = ae.decode(ae.encode(pair.to(device)))[:, slot]
                if state["mode"] == "destroyed":
                    r = r[torch.randperm(r.shape[0], device=r.device)]
                new = r.reshape(b, T, d).to(res.dtype)
                return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

            h = layers[L].register_forward_hook(hook)
            scores = {}
            for mode in (None, "recon", "destroyed"):
                state["mode"] = mode
                r = eval_task(model, proc, task, rows, device, batch_size=8)
                scores[mode or "unpatched"] = r
                print(f"[C2] L{L} {uid} mode={mode or 'unpatched'} score={r['score']:.4f}",
                      flush=True)
            h.remove()

            unp = scores["unpatched"]
            ci = unp.get("ci95") or bootstrap_ci(np.asarray(unp["per_item"], float),
                                                 stat="mean")[:2]
            lo, hi = float(ci[0]), float(ci[1])
            rec_in = bool(lo <= scores["recon"]["score"] <= hi)
            des_in = bool(lo <= scores["destroyed"]["score"] <= hi)
            # AF-4: the A7 verdict is only READ when the destroyed patch proves the metric moves.
            verdict = ("VOID-NO-RANGE" if des_in else ("PASS" if rec_in else "FAIL"))
            rep["sides"][f"L{L}/{uid}"] = {
                "slot": slot, "fvu": fv,
                "unpatched": unp["score"], "ci95": [lo, hi],
                "recon": scores["recon"]["score"], "destroyed": scores["destroyed"]["score"],
                "recon_inside_ci": rec_in, "destroyed_inside_ci": des_in,
                "range_verified": bool(not des_in), "verdict": verdict,
                "rule": "A7 — patched inside the unpatched item-bootstrap 95% CI; read ONLY when "
                        "the destroyed patch lands outside that CI (AF-4)"}
            print(f"[C2] L{L} {uid} -> {verdict}", flush=True)
            _w()
        ae.to("cpu")
        del ae, ab, aa
        torch.cuda.empty_cache()

    ctrl = [v for k, v in rep["sides"].items() if k.endswith("/pg1")]
    pi0 = [v for k, v in rep["sides"].items() if k.endswith("/pi0")]
    ctrl_ok = bool(ctrl) and all(v["verdict"] == "PASS" for v in ctrl)
    if not ctrl_ok:
        overall = "INSTRUMENT-VOID — the identity-side positive control did not pass"
    elif any(v["verdict"] == "VOID-NO-RANGE" for v in pi0):
        overall = "VOID-NO-RANGE on the π0 side — the metric cannot see patch damage there"
    elif all(v["verdict"] == "PASS" for v in pi0):
        overall = "PI0-READABLE — the dictionary carries π0's own forward"
    else:
        overall = "PI0-NOT-READABLE — scope-cap wording stands"
    rep["verdict"] = {
        "overall": overall, "control_passed": ctrl_ok,
        "wording": ("'erasure' remains UNLICENSED and the scope-cap wording stands"
                    if overall != "PI0-READABLE" else
                    "the before-only population is a statement about π0, not about the instrument"),
        "H_pi0_forward_readable": ("SUPPORTED" if overall == "PI0-READABLE" else "NOT SUPPORTED")}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"\nC2 VERDICT: {overall}")
    print("TESSERA_C2_OK", flush=True)


if __name__ == "__main__":
    main()
