"""TRAINSCAFFOLD collateral — general-LM integrity on a substrate (prereg §2.3, eval-only, no relearn).

wikitext-2 perplexity + held-out-subject MMLU cloze on a substrate (ABLATED / DELETED / HEALTHY). For the
REACHABILITY-centric result: if the ABLATED backbone's general LM is INTACT (ppl ≈ healthy) yet MMLU
re-entry is severed, the severance is capability-specific (not a total forward-pass break); if the LM is
degraded, the ablation broke the model generally. Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_collateral --out <dir> --substrate ablated \
        --carrier s3://.../topsig/carrier.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--substrate", required=True, choices=["ablated", "deleted", "healthy"])
    ap.add_argument("--carrier", default="")
    ap.add_argument("--emrand", action="store_true",
                    help="ablated: apply the emrand (energy-matched random) factors — the top-σ SPECIFICITY "
                         "control (prereg §6 step-3): if emrand keeps recognition INTACT while top-σ floors "
                         "it, the forward-pass break is top-σ-SPECIFIC, not generic energy removal")
    ap.add_argument("--ppl-window", type=int, default=2048)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from transformers import AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import w14_localize as W14
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15

    assert torch.cuda.is_available(), "trainscaffold_collateral needs GPU."
    device = "cuda"
    dkeys = W14.decoder_keys()
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    if args.substrate == "healthy":
        model = pg_before
        del pg_after
    elif args.substrate == "deleted":
        model = pg_after
        del pg_before
    else:  # ablated
        assert args.carrier, "ablated requires --carrier"
        cpath = args.carrier
        if cpath.startswith("s3://"):
            import subprocess
            cpath = str(out / "carrier.pt")
            subprocess.run(["aws", "s3", "cp", args.carrier, cpath, "--region", "us-east-1"], check=True)
        factors = torch.load(cpath, map_location="cpu")
        model = pg_after
        lm = model.get_submodule(paths["language_model"])
        base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
        which = "emrand" if args.emrand else "carrier"
        S15.apply_carrier(lm, base_sd, dkeys, factors, which, device)
        del pg_before
    torch.cuda.empty_cache()
    model.eval()

    ppl = TE.clean_text_perplexity(model, tok, device, window=args.ppl_window)
    # held-out-subject MMLU cloze (general recognition not deleted): cap subjects via n_per_subject.
    clz = TE.mmlu_cloze(model, tok, device, processor=proc, n_per_subject=4)
    tag = args.substrate + ("_emrand" if args.emrand else "")
    rep = {"object": "trainscaffold_collateral_v1", "substrate": args.substrate, "emrand": args.emrand,
           "wikitext_ppl": ppl if isinstance(ppl, (int, float)) else ppl.get("ppl", ppl),
           "mmlu_cloze_acc": clz["acc_pct"], "cloze_n": clz["n"], "torch": torch.__version__}
    (out / f"collateral_{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[TS-COLLATERAL] {tag}: wikitext_ppl={rep['wikitext_ppl']} "
          f"mmlu_cloze={clz['acc_pct']:.2f} (in-distribution recognition; chance≈25)")
    print(f"TS_COLLATERAL_OK {tag}")


if __name__ == "__main__":
    main()
