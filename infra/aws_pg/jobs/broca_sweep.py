"""BROCA sweep — causal lesion localization on the HELMHOLTZ-1 pair by cross-model grafting.

Spec: studies/lineage/cross-lineage/research/2026-09-02_broca_spec.md. One job runs the whole
runbook server-side: minute-1 preflight (self-patch identity + 1-batch eval smoke) -> unpatched
baselines ONCE -> controls BEFORE contrast (mismatched-item nulls, self-patches) -> pass-1
coarse sweep -> frozen-rule pass-2 refinement -> verdict (broca_core) -> gated Arm 2 (component
patch at L*) and Arm 3 (instrument re-audit at the locus). Fail-loud; metrics.json incremental.

  pg run infra.aws_pg.jobs.broca_sweep --type g6e.2xlarge,g6e.xlarge,g6e.4xlarge,g6e.8xlarge \
      --s3-prefix results/broca_sweep --budget-min 420
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


class PatchedLM:
    """Callable model wrapper: donor forward (no-grad) on IDENTICAL inputs captures the decoder
    layer-L residual output; recipient forward runs with that output REPLACING its own layer-L
    output. mismatch='roll' shifts donor activations by one item within the batch (the
    mismatched-item null). Works for any kwargs mmlu_blankimg/mmlu_cloze pass."""

    def __init__(self, donor, recipient, donor_layers, recip_layers, layer_idx, mismatch=False,
                 transform=None):
        self.donor, self.recipient = donor, recipient
        self.dl, self.rl = donor_layers[layer_idx], recip_layers[layer_idx]
        self.mismatch = mismatch
        self.transform = transform   # NOETHER: optional per-token map on the captured donor residual
        self._cap = None

    def __call__(self, **kw):
        import torch

        def cap_hook(m, a, out):
            self._cap = out[0] if isinstance(out, tuple) else out

        h1 = self.dl.register_forward_hook(cap_hook)
        try:
            with torch.inference_mode():
                self.donor(**kw)
        finally:
            h1.remove()
        cap = self._cap
        assert cap is not None
        if self.mismatch:
            cap = torch.roll(cap, shifts=1, dims=0)
        if self.transform is not None:
            cap = self.transform(cap)

        def rep_hook(m, a, out):
            assert (out[0] if isinstance(out, tuple) else out).shape == cap.shape, \
                f"patch shape mismatch {cap.shape}"
            if isinstance(out, tuple):
                return (cap,) + tuple(out[1:])
            return cap

        h2 = self.rl.register_forward_hook(rep_hook)
        try:
            with torch.inference_mode():
                return self.recipient(**kw)
        finally:
            h2.remove()
            self._cap = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-subject", type=int, default=5)
    ap.add_argument("--pass1-stride", type=int, default=2)
    ap.add_argument("--null-layers", default="2,8,14,20,25")
    ap.add_argument("--self-layers", default="4,12,22")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip-arms", action="store_true", help="MDE only (no Arm 2/3)")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import broca_core as BC
    from infra.pg import text_eval as TE
    from infra.pg.diff_harvest import build_before_after
    from infra.pg.gemma_scope import resolve_gemma2_layers

    assert torch.cuda.is_available()
    device = "cuda"
    rep = {"object": "BROCA sweep (prefix-graft localization)", "n_per_subject": a.n_per_subject,
           "seed": a.seed}
    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- pair (the validated HELMHOLTZ-1 construction; VLMs scored via blankimg) ----
    pg_after, pg_before, proc, paths, _i, _g = build_before_after(device, pg_key="PG",
                                                                  attn_impl="eager")
    lm_b = resolve_gemma2_layers(pg_before.get_submodule(paths["language_model"]))
    lm_a = resolve_gemma2_layers(pg_after.get_submodule(paths["language_model"]))
    n_layers = len(lm_b)
    d_model = lm_b[0].mlp.down_proj.out_features
    rep["bind"] = {"n_layers": n_layers, "d_model": d_model}
    assert n_layers == 26 and d_model == 2304, rep["bind"]

    def acc(model, n=None):
        r = TE.mmlu_blankimg(model, proc, device, n_per_subject=n or a.n_per_subject)
        return r["acc"], r

    # ---- minute-1 preflight: self-patch identity on ONE tiny eval + logit parity ----
    w_self = PatchedLM(pg_before, pg_before, lm_b, lm_b, 12)
    a_self, _ = acc(w_self, n=1)
    a_base, _ = acc(pg_before, n=1)
    assert abs(a_self - a_base) < 1e-9, f"preflight self-patch broke eval: {a_self} vs {a_base}"
    print(f"[BROCA][preflight] self-patch identity OK (acc {a_base:.4f})", flush=True)
    _w()

    # ---- baselines ONCE ----
    acc_before, rb = acc(pg_before)
    acc_after, ra = acc(pg_after)
    rep["baselines"] = {"acc_before": acc_before, "acc_after": acc_after,
                       "n": rb["n"]}
    assert rb["per_item_hash"] == ra["per_item_hash"], "item pairing broke"
    print(f"[BROCA] baselines: before {acc_before:.4f} after {acc_after:.4f}", flush=True)
    _w()

    def R_of(model_wrapper):
        ap_, _ = acc(model_wrapper)
        return BC.restoration_fraction(ap_, acc_after, acc_before), ap_

    # ---- controls BEFORE contrast ----
    selfs = {}
    for li in [int(x) for x in a.self_layers.split(",")]:
        s_b, _ = acc(PatchedLM(pg_before, pg_before, lm_b, lm_b, li))
        s_a, _ = acc(PatchedLM(pg_after, pg_after, lm_a, lm_a, li))
        selfs[str(li)] = {"before_self": s_b, "after_self": s_a,
                          "ok": bool(abs(s_b - acc_before) < 1e-9 and abs(s_a - acc_after) < 1e-9)}
        print(f"[BROCA][ctrl] self-patch L{li}: ok={selfs[str(li)]['ok']}", flush=True)
    rep["self_patches"] = selfs
    assert all(v["ok"] for v in selfs.values()), "self-patch identity FAILED — wiring void"
    nulls = {}
    for li in [int(x) for x in a.null_layers.split(",")]:
        r_n, acc_n = R_of(PatchedLM(pg_before, pg_after, lm_b, lm_a, li, mismatch=True))
        nulls[str(li)] = {"R_null": r_n, "acc": acc_n}
        print(f"[BROCA][ctrl] null (mismatched donor) L{li}: R={r_n:.4f}", flush=True)
        _w()
    band = BC.null_band([v["R_null"] for v in nulls.values()])
    T = BC.threshold(band)
    rep["null_band"] = band; rep["threshold"] = T; rep["nulls"] = nulls
    print(f"[BROCA] null band={band:.4f} -> T={T:.4f} (FROZEN before contrast)", flush=True)
    _w()

    # ---- pass 1: coarse prefix-graft sweep, both directions ----
    curve = {}
    layers1 = sorted(set(list(range(0, n_layers, a.pass1_stride)) + [n_layers - 1]))
    for li in layers1:
        r_rest, acc_p = R_of(PatchedLM(pg_before, pg_after, lm_b, lm_a, li))
        r_abl = float((acc_before - [acc(PatchedLM(pg_after, pg_before, lm_a, lm_b, li))[0]][0])
                      / max(acc_before - acc_after, 1e-9))
        curve[li] = {"R": r_rest, "acc_patched": acc_p, "ablation_D": r_abl}
        print(f"[BROCA] L{li}: R={r_rest:.4f} (acc {acc_p:.4f}) D={r_abl:.4f}", flush=True)
        rep["curve"] = {str(k): v for k, v in sorted(curve.items())}
        _w()

    # ---- frozen-rule pass 2: bracket the transition ----
    Ls = sorted(curve); Rs = [curve[l]["R"] for l in Ls]
    tr = BC.transition(Ls, Rs, T)
    refine = []
    if tr["first_above_null"] is not None:
        lo = max(0, tr["first_above_null"] - 2)
        hi = min(n_layers - 1, (tr["l_star"] if tr["l_star"] is not None else Ls[-1]) + 1)
        refine = [l for l in range(lo, hi + 1) if l not in curve][:8]
    rep["pass2_layers"] = refine
    for li in refine:
        r_rest, acc_p = R_of(PatchedLM(pg_before, pg_after, lm_b, lm_a, li))
        curve[li] = {"R": r_rest, "acc_patched": acc_p}
        print(f"[BROCA][p2] L{li}: R={r_rest:.4f}", flush=True)
        rep["curve"] = {str(k): v for k, v in sorted(curve.items())}
        _w()

    # ---- verdict ----
    Ls = sorted(curve); Rs = [curve[l]["R"] for l in Ls]
    verdict = BC.broca_verdict(Ls, Rs, T, n_layers)
    rep["verdict"] = verdict
    print(f"[BROCA] VERDICT: {json.dumps(verdict)}", flush=True)
    _w()

    # ---- gated arms ----
    if verdict["verdict"].startswith("H_reader") and not a.skip_arms:
        ls = verdict["l_star"]
        # Arm 2: component resolution at L* (attn-out vs mlp-out replacement)
        comp = {}
        for name, get in (("attn", lambda l: l.self_attn.o_proj), ("mlp", lambda l: l.mlp.down_proj)):
            capbox = {}
            def cap_hook(m, i, o):  # noqa: B023
                capbox["v"] = o
            def rep_hook(m, i, o):  # noqa: B023
                return capbox["v"]
            class CompPatch:
                def __call__(self, **kw):
                    h1 = get(lm_b[ls]).register_forward_hook(cap_hook)
                    try:
                        with torch.inference_mode():
                            pg_before(**kw)
                    finally:
                        h1.remove()
                    h2 = get(lm_a[ls]).register_forward_hook(rep_hook)
                    try:
                        with torch.inference_mode():
                            return pg_after(**kw)
                    finally:
                        h2.remove()
            r_c, acc_c = R_of(CompPatch())
            comp[name] = {"R": r_c, "acc": acc_c}
            print(f"[BROCA][arm2] {name}-only @L{ls}: R={r_c:.4f}", flush=True)
        rep["arm2_components"] = comp
        _w()
        # Arm 3: cloze at L* + CKA at locus band (text-only harvest on LM submodules, H-1 path)
        clz_p = TE.mmlu_cloze(PatchedLM(pg_before, pg_after, lm_b, lm_a, ls), proc.tokenizer,
                              device, processor=proc, n_per_subject=a.n_per_subject)
        clz_b = TE.mmlu_cloze(pg_before, proc.tokenizer, device, processor=proc,
                              n_per_subject=a.n_per_subject)
        clz_a = TE.mmlu_cloze(pg_after, proc.tokenizer, device, processor=proc,
                              n_per_subject=a.n_per_subject)
        rep["arm3_cloze_at_lstar"] = {"patched": clz_p["acc"], "before": clz_b["acc"],
                                      "after": clz_a["acc"]}
        from infra.aws_pg.jobs.helmholtz_confirm import _harvest_texts, harvest_paired_text
        from infra.pg import helmholtz_core as H1C
        m_b = pg_before.get_submodule(paths["language_model"]).eval()
        m_a = pg_after.get_submodule(paths["language_model"]).eval()
        texts = _harvest_texts(2)
        ck = {}
        for li in [max(0, ls - 2), ls, min(n_layers - 1, ls + 2)]:
            ab, aa = harvest_paired_text(m_b, m_a, li, texts, proc.tokenizer, device)
            ck[str(li)] = H1C.linear_cka(ab.numpy(), aa.numpy())
            print(f"[BROCA][arm3] CKA L{li}={ck[str(li)]:.4f}", flush=True)
        rep["arm3_cka_locus"] = ck
        _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("BROCA_OK " + json.dumps({"verdict": verdict["verdict"], "T": T,
                                    "l_star": verdict.get("l_star")}), flush=True)


if __name__ == "__main__":
    main()
