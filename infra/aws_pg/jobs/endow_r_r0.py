"""ENDOW-R R0 + instrument gate (spec §4.3, the R-MDE) — reachability-without-content IN LINEAGE.
Runs the FROZEN SAV relearn probe (MMLU recovery, r512, the sav_controls recipe) over four inits and
reports held-out best-checkpoint savings G = S_recover(init) − S_recover(reference):

  base-gemma-2b  — the ceiling (pretraining's MMLU capability).      [instrument-gate numerator]
  raw-random     — swap_back.random_decoder_state_dict (design-B: base embed/norms, random matrices).
  pi0-backbone   — pi0_port.remap_backbone_lm (the deployed, text-dead policy backbone).
  scratch-fair   — identical to raw-random under the identical probe (the "you handicapped scratch"
                   objection is void by construction; the bounded recipe-search upgrade is a separate job).

Instrument gate (before any verdict): discovery G(base vs raw-random) CI-lo > δ_R, else INSTRUMENT-NULL.
R0 verdict: H_endowment iff sealed G(pi0 vs scratch-fair) CI-lo > δ_R. Identification (TRAINSCAFFOLD):
held-out S_recover only; train-CE reported to expose memorization; no absolute can-it-learn claim.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_r_r0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/endow_r_r0 --budget-min 300 \
      --args "--controls-s3 s3://.../savb_exec_a/out/unit_rsvqa-lr/sav_controls.json --n-train 16000"
"""
import argparse
import json
import pathlib


def _extract_decoder(name):
    """(bare decoder sd, source_sha) for an init — pi0 (robot policy) or pg1 (pre-robot VL backbone).
    Both normalized to bare-decoder keys (the PG1-vs-pi0 disambiguation of the MMLU-scope confound)."""
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder as vlat_extract
    key = {"pi0": "pi0", "pg1": "PG1"}[name]
    dec, sha = vlat_extract(key, "cpu", None)
    return _bare_decoder(dec), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--inits", default="base,random,pi0,scratch")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg import swap_back as SW
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import boot_s_text, s_text
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder

    assert torch.cuda.is_available()
    device = "cuda"
    import subprocess
    cl = out / "sav_controls.json"
    subprocess.run(["aws", "s3", "cp", a.controls_s3, str(cl), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    C = json.loads(cl.read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    fr = C["recipe"]
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]
    conf_hashes = W.mask_by(hashes, hashes, W.is_confirmation)
    b_conf = W.mask_by(b_pi, hashes, W.is_confirmation)
    f_conf = W.mask_by(f_pi, hashes, W.is_confirmation)

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    base_lm_sd = {k: v.cpu() for k, v in base.model.state_dict().items()}
    dec_base = _bare_decoder(base_lm_sd)
    del base
    torch.cuda.empty_cache()

    # the four init decoders (design-B: base embed/norms via the swap shell; only 2-D matrices vary)
    inits = {}
    want = a.inits.split(",")
    if "base" in want:
        inits["base"] = dec_base
    if "random" in want or "scratch" in want:
        inits["random"] = _bare_decoder(SW.random_decoder_state_dict(base_lm_sd, a.seed))
    if "pg1" in want:
        inits["pg1"], _sha = _extract_decoder("pg1")
    if "pi0" in want:
        inits["pi0"], _sha = _extract_decoder("pi0")

    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    sub = R.ladder_subset(ded["kept"], a.n_train)
    examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]

    rep = {"recipe": fr, "n_train": len(examples), "seed": a.seed, "inits": list(inits),
           "results": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def run_probe(name, dec):
        if name == "base":                               # the ceiling IS base — load it directly
            model = AutoModelForCausalLM.from_pretrained(  # (swap-shell asserts decoder≠base → no-op guard)
                ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
                device_map=device, attn_implementation="eager")
            swap_info = {"base_direct": True}
        else:
            model, swap_info = SB.load_swapback_model(dec, device)
        R.attach_adapters(model.model, rank=fr["rank_primary"], seed=a.seed,
                          alpha=float(fr["rank_primary"]) * fr["alpha_mult"])
        model.to(device)
        train_rep = R.train_adapter_qa(model, tok, device, examples, lr=fr["lr"],
                                       epochs=fr["epochs"], batch_size=fr["batch"],
                                       warmup_ratio=fr["warmup"], seed=a.seed, micro_batch=2)
        ev = TE.mmlu_choice_logits(model, tok, device, item_filter=W.is_confirmation)
        assert ev["per_item_hash"] == conf_hashes, "confirmation identity drift — HALT"
        S = s_text(ev["per_item"], b_conf, f_conf)
        Sb = boot_s_text(ev["per_item"], b_conf, f_conf)
        del model
        torch.cuda.empty_cache()
        return {"S_recover": S, "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                "acc_pct": ev["acc_pct"], "train_ce_last": train_rep["final_loss"],
                "per_item": ev["per_item"], "swap": {k: v for k, v in swap_info.items()}}

    for name, dec in inits.items():
        rep["results"][name] = run_probe(name, dec)
        r = rep["results"][name]
        print(f"[R0] {name}: S={r['S_recover']:.4f} [{r['S_boot']['lo95']:.4f},{r['S_boot']['hi95']:.4f}] "
              f"acc={r['acc_pct']:.2f} CE={r['train_ce_last']:.3f}", flush=True)
        _w()

    # savings G against references (paired per-item bootstrap over the confirmation half)
    def gap_boot(a_per, b_per, n_boot=2000, seed=1234):
        # paired per-item bootstrap of ΔS = S(a) − S(b) over shared resampled confirmation items
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(b_conf), size=(n_boot, len(b_conf)))
        bc, fc = np.asarray(b_conf, float), np.asarray(f_conf, float)
        def s_of(per):
            p = np.asarray(per, float)
            num = (p[idx] - fc[idx]).mean(1)
            den = (bc[idx] - fc[idx]).mean(1)
            return num / den
        d = s_of(a_per) - s_of(b_per)
        return {"mean": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
                "ci_hi": float(np.percentile(d, 97.5))}

    res = rep["results"]
    if "base" in res and "random" in res:
        rep["instrument_gate"] = gap_boot(res["base"]["per_item"], res["random"]["per_item"])
    if "pi0" in res and "random" in res:
        rep["R0_pi0_vs_scratch"] = gap_boot(res["pi0"]["per_item"], res["random"]["per_item"])
    if "pg1" in res and "random" in res:                 # disambiguate the MMLU-scope confound
        rep["pg1_vs_scratch"] = gap_boot(res["pg1"]["per_item"], res["random"]["per_item"])
    if "pi0" in res and "pg1" in res:
        rep["pi0_vs_pg1"] = gap_boot(res["pi0"]["per_item"], res["pg1"]["per_item"])
    rep["green"] = True
    _w()
    print("R0_OK", json.dumps({"instrument_gate": rep.get("instrument_gate"),
                               "R0": rep.get("R0_pi0_vs_scratch")}, default=str))


if __name__ == "__main__":
    main()
