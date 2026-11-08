"""CO-B6 (item 61) — the newborn-tissue question: is the output shadow already there at step 0?

Spec §3 Tier B / §6.3. Rides CO-B2's machinery exactly as §3 said it would: the same bottom
right-singular "shadow" subspace, the same Haar floor, plus the self-band ceiling §6.3 adds.

THE QUESTION. CO-B2 found that the directions L17's writes leave unlit are the directions the
effective output map is least sensitive to. If that shadow is a LEARNED feature it should differ
between a step-0 checkpoint and a fully trained one; if it is newborn tissue — geometry the model
has at initialisation and never revisits — it should not.

THE THREE-WAY READ (§6.3), and it is genuinely three-way rather than a threshold:
  SUBSTANTIALLY-UNCHANGED  cos²(step0, final) inside the final-vs-final SELF-BAND
  CHANGED                  at or below the Haar floor
  GRADED                   between them — reported as such, not rounded to either

THE SELF-BAND is what makes "unchanged" mean something. Two estimates of the SAME shadow do not
agree perfectly, because the map is a finite matrix and its small singular directions are the
least determined; bootstrapping the output map's rows gives the ceiling that a genuinely unchanged
shadow could reach. Comparing step-0-vs-final against 1.0 instead would call every real shadow
"changed".

    pg run infra.aws_pg.jobs.copernicus_b6 \
        --args "--suite allenai/OLMo-2-0425-1B" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_b6 --budget-min 90
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.aws_pg.jobs.copernicus_b2b4 import bottom_right_singular
from infra.aws_pg.jobs.darwin_w0_ext import checkpoint_revisions

ARM = "CO-B6"
QUEUE_ITEM = "61"
WAVE = "WF"
K = 32
N_HAAR = 12
N_SELF = 12          # bootstrap draws for the self-band ceiling


def effective_output_map(model):
    """lm_head · diag(1 + final_norm.weight), the map the residual stream actually sees. Resolved
    off the loaded model because the decoder-state-dict helper drops lm_head (CO-B2's finding)."""
    import torch
    head = model.lm_head.weight.detach().to(torch.float64).numpy()
    inner = model.model if hasattr(model, "model") else model
    assert hasattr(inner, "norm"), "final norm not found"
    w = inner.norm.weight.detach().to(torch.float64).numpy()
    return head * (1.0 + w)[None, :]


def self_band(Wmat, k: int, n_draws: int, seed: int):
    """Bootstrap the output map's ROWS (vocabulary) and re-read the bottom-k. The spread of
    cos²(resampled shadow, full shadow) is how well a shadow agrees with ITSELF under estimation
    noise — the ceiling an unchanged shadow can be asked to clear."""
    A = np.asarray(Wmat, dtype=np.float64)
    full, _ = bottom_right_singular(A, k)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        idx = rng.integers(0, A.shape[0], A.shape[0])
        V, _ = bottom_right_singular(A[idx], k)
        out.append(C.mean_cos2(V, full))
    return out, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM

    api = HfApi()
    revs = checkpoint_revisions(api, a.suite)
    assert len(revs) >= 2, f"{a.suite} exposes {len(revs)} step revisions, need >= 2"
    step0, branch0 = revs[0]
    stepN, branchN = revs[-1]
    print(f"[b6] {a.suite}: {len(revs)} revisions; earliest step {step0} ({branch0}), "
          f"latest step {stepN} ({branchN})", flush=True)

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "b6_shadow_substantially_unchanged",
            "the shadow content is SUBSTANTIALLY UNCHANGED from step 0 to final — newborn tissue",
            confidence="low-mid", basis="spec §11 row CO-B6", family="boundary")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_b6", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM))
    gates = C.Gates(out)

    maps = {}
    for tag, branch in (("step0", branch0), ("final", branchN)):
        m = AutoModelForCausalLM.from_pretrained(a.suite, revision=branch,
                                                 torch_dtype=torch.float32, device_map="cpu")
        maps[tag] = effective_output_map(m)
        print(f"[b6] {tag} ({branch}) output map {maps[tag].shape}", flush=True)
        del m

    assert maps["step0"].shape == maps["final"].shape, (
        f"step0 {maps['step0'].shape} and final {maps['final'].shape} differ in shape")
    d = maps["final"].shape[1]

    self_vals, shadow_final = self_band(maps["final"], K, N_SELF, a.seed)
    sb = C.band(self_vals)
    shadow_0, _ = bottom_right_singular(maps["step0"], K)
    cos2 = C.mean_cos2(shadow_0, shadow_final)
    haar = [C.mean_cos2(H.haar_subspace(d, K, a.seed + 7 * i), shadow_final) for i in range(N_HAAR)]
    hb = C.band(haar)

    unchanged = bool(cos2 >= sb["lo"])
    changed = bool(cos2 <= hb["hi"])
    verdict = ("SUBSTANTIALLY-UNCHANGED" if unchanged else
               ("CHANGED" if changed else "GRADED"))
    gates.record("b6_shadow_substantially_unchanged", unchanged,
                 cos2_step0_vs_final=cos2, self_band=[sb["lo"], sb["hi"]],
                 haar_band=[hb["lo"], hb["hi"]], three_way_verdict=verdict,
                 n_self=N_SELF, n_haar=N_HAAR, step0=int(step0), final=int(stepN),
                 band="SUBSTANTIALLY-UNCHANGED iff inside the final-vs-final self-band; CHANGED iff "
                      "at/below the Haar floor; between -> GRADED, reported as such (§6.3)",
                 prediction="shadow content substantially unchanged step-0 -> final (§11, low-mid)")
    at.add("b6_shadow_persistence", value=cos2, n=K,
           flags={"self_band": [sb["lo"], sb["hi"]], "haar_band": [hb["lo"], hb["hi"]],
                  "verdict": verdict, "suite": a.suite, "step0": int(step0), "final": int(stepN)})

    rep = {"object": "copernicus_b6_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "suite": a.suite, "k": K,
           "step0": {"step": int(step0), "branch": branch0},
           "final": {"step": int(stepN), "branch": branchN},
           "n_revisions": len(revs), "residual_dim": int(d),
           "cos2_step0_vs_final": cos2, "self_band": sb, "haar_band": hb,
           "verdict": verdict,
           "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_b6.json").write_text(json.dumps(rep, indent=2, default=float))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_B6_OK verdict={verdict} cos2={cos2:.5f} "
          f"self_band=[{sb['lo']:.5f},{sb['hi']:.5f}] haar=[{hb['lo']:.5f},{hb['hi']:.5f}]",
          flush=True)


if __name__ == "__main__":
    main()
