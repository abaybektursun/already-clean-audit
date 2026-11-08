"""TRAINSCAFFOLD Arm-0 gradient diagnostic (prereg §3.4, DISCOVERY-only; ~1 GPU-min, run FIRST).

On the ABLATED backbone, one forward+backward of the MMLU letter-CE loss; measure the fraction of the
full-weight gradient energy in the REMOVED rank-k diagonal dyads (loaded from the banked carrier), and
the SAME fraction for the emrand (energy-matched) dyads. Report the emrand-NORMALIZED contrast — a raw
large fraction is consistent with both capacity and severance, so only the contrast vs emrand is
informative (it sets the REACHABILITY prior; never a verdict). Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_graddiag --out <dir> \
        --carrier s3://.../latmech-topsig/out/topsig/carrier.pt --n-batch 16
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--carrier", required=True, help="topsig carrier.pt (local or s3://)")
    ap.add_argument("--n-batch", type=int, default=16, help="MMLU items for the one gradient")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from infra.pg import relearn as Rl
    from infra.pg import text_eval as TE
    from infra.pg import w14_localize as W14
    from infra.pg import trainscaffold as TS
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15

    assert torch.cuda.is_available(), "trainscaffold_graddiag needs GPU."
    device = "cuda"
    dkeys = W14.decoder_keys()
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    cpath = args.carrier
    if cpath.startswith("s3://"):
        import subprocess
        cpath = str(out / "carrier.pt")
        subprocess.run(["aws", "s3", "cp", args.carrier, cpath, "--region", "us-east-1"], check=True)
    factors = torch.load(cpath, map_location="cpu")

    # ---- build the ablated substrate ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    del pg_before
    model = pg_after
    lm = model.get_submodule(paths["language_model"])
    base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
    S15.apply_carrier(lm, base_sd, dkeys, factors, "carrier", device)
    torch.cuda.empty_cache()

    # ---- one MMLU letter-CE forward+backward; decoder 2-D weights require grad ----
    for p in model.parameters():
        p.requires_grad_(False)
    params = {k: lm.get_parameter(k) for k in dkeys}
    for prm in params.values():
        prm.requires_grad_(True)

    ds = load_dataset(TE.MMLU_REPO, "all")
    rows = [dict(r) for r in ds["test"]][:args.n_batch]
    letters = Rl.letter_token_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prompts = [Rl.format_qa(r["question"], r["choices"], r["answer"])["prompt"] for r in rows]
    tgt = torch.tensor([letters[int(r["answer"])] for r in rows], device=device)
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
    pos = enc["attention_mask"].long().cumsum(-1) - 1
    pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
    model.train()
    last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(last.float(), tgt)
    assert torch.isfinite(loss), "non-finite MMLU loss — HALT"
    loss.backward()

    # ---- project each matrix's grad onto the removed (carrier) and emrand dyads ----
    carrier_fracs, emrand_fracs = [], []
    for k in dkeys:
        g = params[k].grad
        if g is None:
            continue
        Uc, _Sc, Vc = factors[k]["carrier"]      # removed top-k dyads (already sliced columns)
        Ue, _Se, Ve = factors[k]["emrand"]
        kc = Uc.shape[1]
        ke = Ue.shape[1]
        carrier_fracs.append(TS.grad_in_removed_dyad_fraction(g.detach().float(), Uc.float(), Vc.float(),
                                                              list(range(kc))))
        emrand_fracs.append(TS.grad_in_removed_dyad_fraction(g.detach().float(), Ue.float(), Ve.float(),
                                                             list(range(ke))))
    import numpy as np
    cf, ef = float(np.mean(carrier_fracs)), float(np.mean(emrand_fracs))
    rep = {"object": "trainscaffold_graddiag_v1", "n_batch": args.n_batch, "n_matrices": len(carrier_fracs),
           "carrier_frac_pooled": cf, "emrand_frac_pooled": ef, "contrast": cf - ef,
           "ratio": (cf / ef if ef > 0 else float("inf")), "loss": float(loss.item()),
           "prior": ("capacity/route-around (loss routes through removed > emrand)" if cf - ef > 0.0
                     else "severance/irrelevance (no special routing)")}
    (out / "graddiag.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[TS-GRADDIAG] carrier_frac={cf:.4f} emrand_frac={ef:.4f} contrast={cf - ef:+.4f} "
          f"ratio={rep['ratio']:.2f} → prior: {rep['prior']}")
    print(f"TS_GRADDIAG_OK contrast={cf - ef:+.4f}")


if __name__ == "__main__":
    main()
