"""ENDOW-R R1 transplant ladder (spec §4.3, gain-of-function) — names the reachability carrier.
Eight hybrid inits: the named structure copied from base gemma-2b, everything else raw-random
(design-B: base embed/norms via the swap shell; only 2-D matrices vary). Same FROZEN SAV relearn
probe as R0; reference = raw-random. G(cell) = S_recover(cell) − S_recover(raw-random).

  RAND (anchor) · ATTN · MLP · EARLY · MID · LATE · SPEC · UNT   (EMB dropped under design-B).

H_statistical iff SPEC clears & ≥½ ceiling; H_component iff a structural cell clears & SPEC doesn't;
H_holistic iff R0 positive & no cell clears. Held-out S_recover only; train-CE reported (no absolute
can-it-learn claim — the TRAINSCAFFOLD lesson).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_r_ladder --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/endow_r_ladder --budget-min 300 \
      --args "--controls-s3 s3://.../sav_controls.json --n-train 1000 --cells RAND,ATTN,MLP,EARLY,MID,LATE,SPEC,UNT"
"""
import argparse
import json
import pathlib

from infra.aws_pg.jobs._inherit_shared import INVCORE_SCRATCH, N_FT


def _all_layer_uunt(k, workdir):
    """U_unt(L) at ALL 18 layers from the cached q_proj ΔWs (consensus Gram → bottom-k eigvecs) —
    the q/k-input span the UNT cell restricts base weights to. No existence null needed here."""
    import subprocess
    import numpy as np
    import torch
    from infra.pg.inherit_core import bottom_eigvecs, consensus_gram
    subprocess.run(["aws", "s3", "cp", f"{INVCORE_SCRATCH}/", str(workdir), "--recursive",
                    "--exclude", "*", "--include", "q_proj__*", "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    U = {}
    for L in range(18):
        deltas = [torch.as_tensor(np.load(f"{workdir}/q_proj__L{L}__{i}.npy").astype(np.float64))
                  for i in range(N_FT)]
        G = consensus_gram(deltas)
        U[L], _ = bottom_eigvecs(torch.as_tensor(G), k)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cells", default="RAND,ATTN,MLP,EARLY,MID,LATE,SPEC,UNT")
    ap.add_argument("--uunt-k", type=int, default=128)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import subprocess
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import endow_transplant as ET
    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import boot_s_text, s_text
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder

    assert torch.cuda.is_available()
    device = "cuda"
    cl = out / "sav_controls.json"
    subprocess.run(["aws", "s3", "cp", a.controls_s3, str(cl), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    C = json.loads(cl.read_text())
    assert C["CONTROLS_OK"] and C["recipe"]["FROZEN"]
    fr = C["recipe"]
    hashes = C["item_hashes"]
    conf_hashes = W.mask_by(hashes, hashes, W.is_confirmation)
    b_conf = W.mask_by(C["BASE"]["per_item"], hashes, W.is_confirmation)
    f_conf = W.mask_by(C["FLOOR"]["per_item"], hashes, W.is_confirmation)

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager").eval()
    base_lm_sd = {k: v.clone() for k, v in base.model.state_dict().items()}
    del base

    cells = a.cells.split(",")
    U_unt = _all_layer_uunt(a.uunt_k, str(pathlib.Path("/tmp/uunt"))) if "UNT" in cells else None

    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    examples = [R.format_qa(r["question"], r["choices"], r["answer"])
                for r in R.ladder_subset(ded["kept"], a.n_train)]

    rep = {"recipe": fr, "n_train": len(examples), "cells": cells, "results": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def probe(dec):
        model, _si = SB.load_swapback_model(dec, device)
        R.attach_adapters(model.model, rank=fr["rank_primary"], seed=a.seed,
                          alpha=float(fr["rank_primary"]) * fr["alpha_mult"])
        model.to(device)
        tr = R.train_adapter_qa(model, tok, device, examples, lr=fr["lr"], epochs=fr["epochs"],
                                batch_size=fr["batch"], warmup_ratio=fr["warmup"], seed=a.seed,
                                micro_batch=2)
        ev = TE.mmlu_choice_logits(model, tok, device, item_filter=W.is_confirmation)
        assert ev["per_item_hash"] == conf_hashes, "confirmation identity drift — HALT"
        del model
        torch.cuda.empty_cache()
        return {"S_recover": s_text(ev["per_item"], b_conf, f_conf), "acc_pct": ev["acc_pct"],
                "train_ce_last": tr["final_loss"], "per_item": ev["per_item"],
                "S_boot": boot_s_text(ev["per_item"], b_conf, f_conf)}

    for cell in cells:
        dec = _bare_decoder(ET.build_cell(cell, base_lm_sd, seed=a.seed,
                                          U_unt=(U_unt if cell == "UNT" else None)))
        rep["results"][cell] = probe(dec)
        r = rep["results"][cell]
        print(f"[LADDER] {cell}: S={r['S_recover']:.4f} acc={r['acc_pct']:.2f} CE={r['train_ce_last']:.3f}",
              flush=True)
        _w()

    def gap_boot(a_per, b_per, seed=1234):
        rng = np.random.default_rng(seed)
        bc, fc = np.asarray(b_conf, float), np.asarray(f_conf, float)
        idx = rng.integers(0, len(b_conf), size=(2000, len(b_conf)))

        def s_of(per):
            p = np.asarray(per, float)
            return (p[idx] - fc[idx]).mean(1) / (bc[idx] - fc[idx]).mean(1)
        d = s_of(a_per) - s_of(b_per)
        return {"mean": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
                "ci_hi": float(np.percentile(d, 97.5))}

    ref = rep["results"].get("RAND")
    if ref:
        rep["G_vs_random"] = {c: gap_boot(rep["results"][c]["per_item"], ref["per_item"])
                              for c in cells if c != "RAND"}
    rep["green"] = True
    _w()
    print("LADDER_OK", json.dumps(rep.get("G_vs_random", {}), default=str))


if __name__ == "__main__":
    main()
