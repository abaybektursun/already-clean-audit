"""DARWIN W1-X / W1-H — the telemetry harness and attention puppetry. GPU.

  W1-X  Telemetry harness (§4/W1-X). Package the B1 projector as a streaming callback (32 floats
        per layer per token) and MEASURE its overhead. This is an engineering arm: its deliverable
        is the tool plus the overhead number, not a scientific statistic. The overhead is measured
        as paired hooked/unhooked forwards over the SAME windows in the same process, alternating,
        so drift in GPU clocks cannot masquerade as instrumentation cost.

  W1-H  Attention puppetry (§4/W1-H, extends B4). eps-perturbations along core vs twins, with the
        PAIRED readout B4 lacked: attention-map JS divergence AND next-token KL from the same
        forward, so "moved attention" can be separated from "moved the output". A perturbation
        that moves attention only because it moved everything is not puppetry.

Both reuse HUMBOLDT's already-validated instruments (`humboldt_act.SiteEnergy`, `BankInject`,
`js_divergence`) rather than reimplementing them — §13 REUSED, not NEW.

  pg run infra.aws_pg.jobs.darwin_w1_probe --args "--arm W1-X" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib
import time

from infra.pg import darwin_core as D
from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H

SPEC_EXPECTATIONS = {
    "W1-I": ("no statistics are claimed by this arm (joy arm)",
             "greedy search finds token sequences with core energy above the twin-search controls"),
    "W1-X": ("overhead exceeds the budget that would make streaming telemetry usable",
             "the streaming projector adds small measured overhead and produces a usable demo log"),
    "W1-H": ("attention-map JS response to core perturbation within the twin band",
             "core-direction perturbations move attention maps more than twin perturbations at "
             "matched next-token KL"),
}
EPS_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)      # >=4 doses, log-spaced ratio 2 (§3)


def run_w1_x(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    """Paired hooked/unhooked timing over the same windows, alternating."""
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _ppl_blankimg

    res = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
    n_tok = int(res["n_tokens"])

    def timed(hooked):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if hooked:
            with A.SiteEnergy(layers, bank_dir, "cuda"):
                _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
        else:
            _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # ALTERNATING pairs: a block of hooked runs followed by a block of unhooked runs would
    # attribute any clock drift or thermal ramp to the instrumentation.
    pairs = []
    for i in range(a.reps):
        off = timed(False)
        on = timed(True)
        pairs.append((off, on))
        print(f"[W1-X] rep {i+1}/{a.reps}: unhooked {off:.3f}s hooked {on:.3f}s "
              f"({100*(on/off-1):+.1f}%)", flush=True)
    offs = np.array([p[0] for p in pairs])
    ons = np.array([p[1] for p in pairs])
    ratio = ons / offs
    fl = {"n_reps": a.reps, "n_seq": a.n_seq, "n_tokens": n_tok,
          "floats_per_layer_per_token": H.BANK_K,
          "n_layers": len(layers), "sites": list(H.SITES_PHASE_B),
          "design": "alternating paired unhooked/hooked, same windows, same process"}
    atlas.add_ci("telemetry_overhead_ratio", ratio.tolist(), model_id=A.PG1_REPO,
                 model_sha=base_sha, flags=fl)
    atlas.add_ci("telemetry_unhooked_seconds", offs.tolist(), model_id=A.PG1_REPO,
                 model_sha=base_sha, flags=fl)
    atlas.add_ci("telemetry_hooked_seconds", ons.tolist(), model_id=A.PG1_REPO,
                 model_sha=base_sha, flags=fl)
    mean_ratio = float(ratio.mean())
    # The demo log the arm is supposed to deliver: one real streaming capture, banked.
    with A.SiteEnergy(layers, bank_dir, "cuda") as se:
        _ppl_blankimg(model, processor, "cuda", n_seq=min(4, a.n_seq))
        cap = se.collect()
    demo = {f"L{L}_{site}": v.shape for (L, site), v in sorted(cap.items())[:6]}
    n_floats = sum(int(v.size) for v in cap.values())
    atlas.add("telemetry_demo_floats", value=float(n_floats), model_id=A.PG1_REPO,
              model_sha=base_sha, flags={**fl, "sample_shapes": {k: list(v) for k, v in demo.items()}})
    gates.record("W1X_overhead_measured", True, mean_overhead_ratio=mean_ratio,
                 pct=100 * (mean_ratio - 1), n_reps=a.reps, demo_floats=n_floats)
    print(f"[W1-X] mean overhead ratio {mean_ratio:.4f} ({100*(mean_ratio-1):+.2f}%)", flush=True)
    return {"mean_overhead_ratio": mean_ratio, "pct": 100 * (mean_ratio - 1),
            "n_reps": a.reps, "demo_floats": n_floats}


def run_w1_h(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    """Paired attention-JS / next-token-KL response to an eps-perturbation, core vs every twin."""
    import numpy as np
    import torch
    from infra.pg import humboldt_weights as W

    D.check_dose_grid(list(EPS_LADDER), arm="W1-H", min_doses=2 if a.smoke else 4)
    tok = processor.tokenizer
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt").input_ids.to("cuda")

    def forward(inject=None):
        ctx = inject if inject is not None else _Null()
        with ctx, torch.no_grad():
            o = model(input_ids=ids, output_attentions=True)
        att = o.attentions[-1][0].float().cpu().numpy()          # last layer, all heads
        logp = torch.log_softmax(o.logits[0, -1].float(), dim=-1).cpu().numpy()
        return att, logp

    base_att, base_logp = forward()
    seen, n_rows = set(), 0
    target_layers = layers if not a.smoke else layers[:2]
    for L in range(0, len(target_layers), a.layer_stride):
        for bid in D.bank_ids_for("q_proj"):
            U = W.load_bank(bank_dir, "q_proj", L, bid)
            z = np.random.default_rng(H.cell_seed(a.seed, bid, L)).standard_normal(H.BANK_K)
            z = z / np.linalg.norm(z)
            for eps in EPS_LADDER:
                inj = A.BankInject(layers[L], "q_proj_in", U, z, eps, "cuda")
                att, logp = forward(inj)
                js = A.js_divergence(att, base_att)
                kl = float(np.sum(np.exp(base_logp) * (base_logp - logp)))
                atlas.add("attention_js", value=js, layer=L, type="q_proj", bank_id=bid,
                          dose=float(eps), incumbent_id="twin_band", model_id=A.PG1_REPO,
                          model_sha=base_sha,
                          flags={"eps": eps, "next_token_kl": kl, "site": "q_proj_in",
                                 "paired": "JS and KL from the SAME forward"})
                atlas.add("next_token_kl", value=kl, layer=L, type="q_proj", bank_id=bid,
                          dose=float(eps), incumbent_id="twin_band", model_id=A.PG1_REPO,
                          model_sha=base_sha, flags={"eps": eps, "attention_js": js})
                # The ratio is the arm's actual object: attention movement PER unit of output
                # movement. A perturbation that moves attention only because it moved everything
                # is not puppetry, and the raw JS cannot tell those apart.
                atlas.add("js_per_kl", value=js / kl if kl > 0 else 0.0, layer=L, type="q_proj",
                          bank_id=bid, dose=float(eps), incumbent_id="twin_band",
                          model_id=A.PG1_REPO, model_sha=base_sha,
                          flags={"eps": eps, "attention_js": js, "next_token_kl": kl})
                seen.add(bid)
                n_rows += 3
        D.s3_sync(out, a.s3_prefix)
        print(f"[W1-H] layer {L}: {n_rows} rows", flush=True)
    D.check_controls(seen, arm="W1-H")
    gates.record("W1H_paired_readout_recorded", True, n_rows=n_rows, banks=sorted(seen),
                 eps_ladder=list(EPS_LADDER))
    return {"n_rows": n_rows, "banks": sorted(seen)}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False



def run_w1_i(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    """W1-I — core-resonant text: greedy token search maximising mean core energy.

    The SAME search is run against every bank, so the control is not a random baseline but the
    identical optimiser pointed at a twin. A search that finds high energy for core32 AND for
    haar32 has found a property of greedy search, not of the core.

    Budget is frozen: `search_steps` extensions x `cand` candidates, scored on mean e_U at the
    q_proj_in site. Recorded as artifacts; §4 calls this a joy arm and no statistic is claimed.
    """
    import numpy as np
    import torch

    tok = processor.tokenizer
    vocab = np.random.default_rng(a.seed).choice(min(32000, len(tok)), size=a.cand, replace=False)
    # SiteEnergy stacks the banks in humboldt_core.BANK_IDS, which predates DARWIN's derived
    # topsigma32 (journal amendment A1), so the projector has no column for it and the lookup ran
    # off the end (defect D24). Iterate the banks the projector ACTUALLY provides and record the
    # missing structured control as a coverage gap rather than crashing or silently dropping it.
    available = [b for b in D.bank_ids_for("q_proj") if b in H.BANK_IDS]
    missing = sorted(set(D.bank_ids_for("q_proj")) - set(available))
    if missing:
        atlas.add("resonant_search_bank_unavailable", value=float(len(missing)),
                  type="q_proj", model_id=A.PG1_REPO, model_sha=base_sha,
                  flags={"missing": missing,
                         "reason": "humboldt_act.SiteEnergy stacks humboldt_core.BANK_IDS, which "
                                   "does not include DARWIN's derived topsigma32; the projector "
                                   "has no column for it at this site"})
        print(f"[W1-I] coverage gap: {missing} unavailable at the projector site", flush=True)
    results = {}
    for bid in available:
        ids = tok("The", return_tensors="pt").input_ids.to("cuda")
        best_hist = []
        for _step in range(a.search_steps):
            best_e, best_t = -1.0, None
            for t in vocab:
                cand = torch.cat([ids, torch.tensor([[int(t)]], device="cuda")], dim=1)
                with A.SiteEnergy(layers, bank_dir, "cuda", sites=("q_proj_in",)) as se, \
                        torch.no_grad():
                    model(input_ids=cand)
                    cap = se.collect()
                order = se.order["q_proj_in"]
                col = [i for i, (ty, b) in enumerate(order) if ty == "q_proj" and b == bid][0]
                e = float(np.mean([v[:, col].mean() for v in cap.values()]))
                if e > best_e:
                    best_e, best_t = e, int(t)
            ids = torch.cat([ids, torch.tensor([[best_t]], device="cuda")], dim=1)
            best_hist.append(best_e)
        text = tok.decode(ids[0].tolist())
        results[bid] = {"final_energy": best_hist[-1], "text": text,
                        "trajectory": best_hist}
        atlas.add("resonant_search_energy", value=best_hist[-1], bank_id=bid, type="q_proj",
                  model_id=A.PG1_REPO, model_sha=base_sha, n=len(best_hist),
                  flags={"text": text, "trajectory": best_hist, "cand": a.cand,
                         "search_steps": a.search_steps,
                         "note": "joy arm; identical search pointed at each bank, so the twin "
                                 "IS the control"})
        print(f"[W1-I] {bid}: e={best_hist[-1]:.5f}  {text[:70]!r}", flush=True)
        D.s3_sync(out, a.s3_prefix)
    # Structured-control completeness is checked against what the projector can express; the
    # gap above is recorded, not waived silently.
    D.check_controls(set(results) | set(missing), arm="W1-I")
    D.write_json(out, "w1_i_texts.json", results)
    gates.record("W1I_search_completed", True, n_banks=len(results),
                 best_bank=max(results, key=lambda k: results[k]["final_energy"]))
    return {"n_banks": len(results),
            "energies": {k: v["final_energy"] for k, v in results.items()}}


RUNNERS = {"W1-X": run_w1_x, "W1-H": run_w1_h, "W1-I": run_w1_i}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layer-stride", type=int, default=6)
    ap.add_argument("--search-steps", type=int, default=8)
    ap.add_argument("--cand", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    if a.smoke:
        a.n_seq, a.reps, a.layer_stride = 2, 2, 12

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _decoder_layers, _load_full_pg1

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    layers = _decoder_layers(model)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id,
                    register="ATLAS", wave="W1", expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "n_layers": len(layers),
                "n_seq": a.n_seq, "reps": a.reps, "eps_ladder": list(EPS_LADDER),
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    result = RUNNERS[a.arm](model, processor, layers, bank_dir, out, atlas, gates, a, base_sha)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(result, default=str)[:220]}")


if __name__ == "__main__":
    main()
