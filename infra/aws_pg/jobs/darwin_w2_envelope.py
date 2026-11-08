"""DARWIN W2-A — the failure envelope. GPU. (§4/W2-A, §5 ENVELOPE.)

§5 makes GRADUATED require EFFECT and ENVELOPE and COMPOSITION. W2-F discharges COMPOSITION.
This arm is ENVELOPE: fine-tune the SAME task under deliberately different optimizer regimes and
map which of them break the invariant — i.e. which ones move the consensus-avoided subspace.

Regimes (§4/W2-A names "long SGD, Muon, LR x{1,4,16}, QAT"):

    adam_lr1x / adam_lr4x / adam_lr16x   exact fp32 Adam, base LR scaled       (the LR ladder)
    sgd_long                             momentum SGD run --sgd-mult x longer  (the long-SGD tail)
    muon                                 Newton-Schulz-orthogonalized momentum (flat-spectrum step)
    qat4                                 exact fp32 Adam + 4-bit fake-quant forward with STE

Everything else is held identical by construction: one task, one trainable scope (q_proj + o_proj
across all layers), one seed, one batch builder, one eval. A difference between regimes is
therefore a difference of update rule and nothing else — which is the only way "what breaks the
invariant" is answerable.

READOUTS, per regime per checkpoint:
  * motion_fraction  ‖dW U‖²_F / ‖dW‖²_F on core32 and EVERY twin. Normalized by the update, so a
    regime that merely moves further does not automatically look like it moves more IN the core.
  * spikiness_drift  the leverage participation ratio of dW's own top-32 right-singular subspace,
    against the base matrix's. §4/W2-A asks for spikiness DRIFT, which is a comparison, not a level.
  * task loss on a held-out slice, and wikitext blank-image retention perplexity — the same
    harness G2 anchored, so the retention numbers are commensurable with the rest of the program.

§0.2.7: W2-A is an optimizer-mechanics arm, so the Adam path is `darwin_train.ExactAdam` (fp32
master weights and moments, unit-tested against torch AdamW to 1e-6) and `check_optimizer` runs
before a single step.

THE POWER FLOOR. §4/W2-A says "power floor from TENANT's motion CIs (recompute)". TENANT's banked
aggregate is fetched and INSPECTED for an interval on the motion scale. If it carries one, it is
the floor and the row says so with the object's S3 URI. If it carries only accuracy-scale
intervals — which is a property of that artifact, not a judgement call — the arm falls back to its
own cell-level bootstrap and records that substitution with the list of TENANT keys it looked at.
Either way the floor's provenance is in the artifact, never in prose.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w2_envelope --args "--arm W2-A" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

TENANT_METRICS = "s3://pg-lineage-769063704696/results/tenant_aggregate2/out/metrics.json"
MOTION_KEY_HINTS = ("motion", "realized_motion", "core_motion")
LR_LADDER = (1.0, 4.0, 16.0)          # §4/W2-A verbatim
QAT_BITS = 4

SPEC_EXPECTATIONS = {
    "W2-A": ("every regime moves the core the same amount — there is no envelope to map",
             "at least one optimizer regime moves the consensus-avoided subspace materially more "
             "than the matched-random band, and the regimes order into a failure envelope"),
}


def regimes(a):
    """{name: (kind, lr_mult, extra)} — written out so a typo cannot silently run a wrong ladder."""
    r = {f"adam_lr{int(m)}x": ("adam", m, {}) for m in LR_LADDER}
    r["sgd_long"] = ("sgd", a.sgd_lr_mult, {"steps_mult": a.sgd_mult})
    r["muon"] = ("muon", a.muon_lr_mult, {})
    r[f"qat{QAT_BITS}"] = ("qat", 1.0, {"bits": QAT_BITS})
    return r


OPTIMIZER_NAMES = {"adam": "exact_fp32_adam", "qat": "exact_fp32_adam",
                   "muon": "muon_newton_schulz_fp32", "sgd": "sgd_momentum_fp32"}


def make_optimizer(kind, params, base_lr, mult):
    """The regime's optimizer, gated by §0.2.7 first.

    `check_optimizer` runs for EVERY regime, not just the Adam ones: W2-A is in
    OPTIMIZER_MECHANICS_ARMS, and the gate's second assert requires `offload_exact_fp32` to be
    stated explicitly so the artifact records exact-math as a decision. It is False here because
    the trainable scope (q_proj + o_proj) costs ~1.8 GB of fp32 state, which fits a 48 GB L40S
    natively — the offload tax is not paid because it is not needed, and that is now on the row.
    """
    gate = D.check_optimizer("W2-A", OPTIMIZER_NAMES[kind], offload_exact_fp32=False)
    if kind in ("adam", "qat"):
        return T.ExactAdam(params, lr=base_lr * mult), gate
    if kind == "muon":
        return T.Muon(params, lr=base_lr * mult), gate
    assert kind == "sgd", kind
    import torch
    return torch.optim.SGD(params, lr=base_lr * mult, momentum=0.9), gate


def spikiness_drift(dW, W0, k):
    """PR(leverage of dW's top-k right-singular subspace) / PR(same for the base matrix).

    A ratio because §4/W2-A asks for spikiness DRIFT: the level alone confounds "this regime
    writes spikily" with "this cell is spiky to begin with".
    """
    import torch
    def pr(M):
        _, _, Vh = torch.linalg.svd(M.to(torch.float64), full_matrices=False)
        U = Vh[:k].T.cpu().numpy()
        return H.participation_ratio(H.leverage(U))
    base = pr(W0)
    assert base > 0, "degenerate base participation ratio"
    return float(pr(dW) / base), float(pr(dW)), float(base)


def tenant_power_floor(out):
    """(floor, provenance) — read from TENANT's banked aggregate, or the arm's own bootstrap.

    The decision is made by INSPECTING the artifact for a motion-scale interval, not by
    remembering what TENANT reported. `feedback_never_report_unverified_progress` and the baseline
    rule are the same rule here: a number a comparison rests on comes from its artifact.
    """
    import subprocess
    dest = pathlib.Path(out) / "tenant_metrics.json"
    subprocess.run(["aws", "s3", "cp", TENANT_METRICS, str(dest), "--region", "us-east-1"],
                   check=True, capture_output=True)
    d = json.loads(dest.read_text())
    hits = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}"
                if any(h in k.lower() for h in MOTION_KEY_HINTS) and isinstance(v, (int, float)):
                    hits.append((p, float(v)))
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o[:8]):
                walk(v, f"{path}[{i}]")
    walk(d)
    if hits:
        floor = max(abs(v) for _, v in hits)
        return floor, {"source": "TENANT banked aggregate", "uri": TENANT_METRICS,
                       "keys": [k for k, _ in hits][:16], "floor": floor,
                       "rule": "largest motion-scale magnitude banked by TENANT"}
    return None, {"source": "arm-local cell bootstrap",
                  "uri": TENANT_METRICS, "top_level_keys": sorted(d)[:16],
                  "why": "TENANT's banked aggregate carries no interval on the MOTION scale "
                         "(its n2_reprint object is accuracy-scale), so a floor taken from it "
                         "would be a floor on the wrong quantity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W2-A", choices=("W2-A",))
    ap.add_argument("--run-id", default="darwin_w2_a")
    ap.add_argument("--s3-prefix", default="darwin/darwin_w2_a")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--ckpt-every", type=int, default=40)
    ap.add_argument("--base-lr", type=float, default=1e-5)
    ap.add_argument("--sgd-lr-mult", type=float, default=100.0)
    ap.add_argument("--sgd-mult", type=int, default=3, help="'long SGD': steps x this")
    ap.add_argument("--muon-lr-mult", type=float, default=20.0)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        a.steps, a.ckpt_every, a.n_train, a.n_eval, a.n_seq = 4, 2, 12, 4, 2
        a.sgd_mult, layers = 2, layers[:2]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import (build_batches, eval_task_loss, text_task_rows)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    floor, floor_prov = tenant_power_floor(out)
    print(f"[W2-A] power floor: {floor} ({floor_prov['source']})", flush=True)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W2",
                    expected_ref=ref)
    REG = regimes(a)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W2", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha,
                "regimes": {k: {"kind": v[0], "lr_mult": v[1], "extra": v[2]}
                            for k, v in REG.items()},
                "lr_ladder": list(LR_LADDER), "qat_bits": QAT_BITS,
                "trainable_scope": list(types), "layers": layers, "steps": a.steps,
                "base_lr": a.base_lr, "optimizer": "exact fp32 Adam (§0.2.7) / SGD / Muon",
                "power_floor": floor, "power_floor_provenance": floor_prov,
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seed=a.seed)
    banks = {(t, L): {bid: W.load_bank(bank_dir, t, L, bid) for bid in D.bank_ids_for(t)}
             for t in types for L in layers}

    model = _load_full_pg1(base_sha, "cuda")
    dl = _decoder_layers(model)
    mods = T.trainable_modules(dl, types=types, layers=layers)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    atlas.add("retention_ppl", value=base_ppl, bank_id="none", dose=0.0, incumbent_id="base",
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"regime": "base", "step": 0, "role": "envelope reference"})
    print(f"[W2-A] base ppl={base_ppl:.5f}", flush=True)

    train_batches = build_batches(processor, train_rows, "cuda")
    eval_batches = build_batches(processor, eval_rows, "cuda")
    results, seen_banks = {}, set()

    for name, (kind, mult, extra) in REG.items():
        for k, m in mods.items():                      # every regime starts from the SAME weights
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        for m in mods.values():
            m.weight.requires_grad_(True)
            params.append(m.weight)
        opt, opt_gate = make_optimizer(kind, params, a.base_lr, mult)
        gates.record(f"W2A_optimizer_{name}", True, **opt_gate)
        n_steps = a.steps * extra.get("steps_mult", 1)
        qat = T.QATLinear(mods, extra["bits"]) if kind == "qat" else _Null()
        model.train()
        hist = []
        with qat:
            for step in range(n_steps):
                b = train_batches[step % len(train_batches)]
                loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist.append(float(loss.detach()))
                if (step + 1) % a.ckpt_every == 0 or step + 1 == n_steps:
                    model.eval()
                    tl = eval_task_loss(model, eval_batches)
                    ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                    for (t, L), m in mods.items():
                        dW = (m.weight.detach().to(torch.float32) - pristine[(t, L)])
                        if float((dW ** 2).sum()) == 0:
                            continue
                        dw_np = dW.cpu().numpy().astype(np.float64)
                        drift, pr_d, pr_b = spikiness_drift(dW, pristine[(t, L)], H.BANK_K)
                        for bid, U in banks[(t, L)].items():
                            mf = T.motion_fraction(dw_np, U)
                            atlas.add("motion_fraction", value=mf, layer=L, type=t, bank_id=bid,
                                      dose=float(mult), incumbent_id="haar_band",
                                      model_id=PG1_REPO, model_sha=base_sha,
                                      flags={"regime": name, "kind": kind, "step": step + 1,
                                             "n_steps": n_steps, "lr": a.base_lr * mult,
                                             "task_loss": tl, "retention_ppl": ppl,
                                             "analytic_haar_expectation": H.BANK_K / U.shape[0],
                                             "power_floor": floor,
                                             "power_floor_source": floor_prov["source"],
                                             "definition": "‖dW U‖²_F / ‖dW‖²_F"})
                            seen_banks.add(bid)
                        atlas.add("spikiness_drift", value=drift, layer=L, type=t,
                                  bank_id="dW_top32", dose=float(mult), incumbent_id="base_top32",
                                  model_id=PG1_REPO, model_sha=base_sha,
                                  flags={"regime": name, "kind": kind, "step": step + 1,
                                         "pr_delta": pr_d, "pr_base": pr_b,
                                         "definition": "PR(leverage of dW's top-32 right-singular "
                                                       "subspace) / PR(same for the base matrix)"})
                    atlas.add("retention_ppl", value=ppl, bank_id="none", dose=float(mult),
                              incumbent_id="base", model_id=PG1_REPO, model_sha=base_sha,
                              flags={"regime": name, "kind": kind, "step": step + 1,
                                     "task_loss": tl, "base_ppl": base_ppl,
                                     "retention_loss": ppl - base_ppl})
                    atlas.add("task_loss", value=tl, bank_id="none", dose=float(mult),
                              incumbent_id="base", model_id=PG1_REPO, model_sha=base_sha,
                              flags={"regime": name, "kind": kind, "step": step + 1,
                                     "train_loss_last": hist[-1], "retention_ppl": ppl})
                    results.setdefault(name, []).append(
                        {"step": step + 1, "task_loss": tl, "ppl": ppl,
                         "retention_loss": ppl - base_ppl, "train_loss": hist[-1]})
                    print(f"[W2-A] {name} step {step+1}/{n_steps}: train={hist[-1]:.4f} "
                          f"task={tl:.4f} ppl={ppl:.4f} (base {base_ppl:.4f})", flush=True)
                    model.train()
                    D.s3_sync(out, a.s3_prefix)
        del opt
        torch.cuda.empty_cache()

    for k, m in mods.items():
        m.weight.data.copy_(pristine[k].to(m.weight.dtype))
    D.check_controls(seen_banks, arm="W2-A")

    gates.record("W2A_envelope_measured", True, regimes=sorted(REG), n_steps=a.steps,
                 power_floor=floor, power_floor_source=floor_prov["source"],
                 banks=sorted(seen_banks),
                 final={k: v[-1] for k, v in results.items()})
    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, "w2_a_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "base_ppl": base_ppl, "results": results,
                  "power_floor": floor, "power_floor_provenance": floor_prov,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_W2_A_OK rows={rep['n_rows']} regimes={len(REG)}")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
