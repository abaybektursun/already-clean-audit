"""W13 — low-rank arm behavioral graft + §9-1d cliff gate (CEILING/FLOOR + rank-8 capability).

The no-retrain GraftLM graft of the rank-r truncated ΔW, scored on RefCOCO (acc@0.5 / <loc>-flip)
and TextVQA (ANLS) → R_cap(r). This is:
  • the §9-1d preflight gate: reproduce CEILING (full 2-D ΔW graft) + FLOOR (base) and the rank-8
    cliff (rank-8 closes most KL yet installs ~0 RefCOCO capability) at the real N (not n=64); and
  • the LOW-RANK ARM (prereg §3.1) — an explicit replication of the already-confirmed §3.5-2
    (capability needs ~eff-rank); it carries NO confirmatory weight. The SOLE confirmatory arm is
    the sparse-dictionary arm (separate job).

Faithfulness is BEHAVIORAL (R_cap), never Frobenius (§6). Graft = base gemma-2-2b decoder +
rank-r ΔW on the 182 decoder matrices, norms/embed/lm_head/connector frozen (the e4 construction).
Runs at the deployed bf16 graft precision; the SVD factors come from a generous-q randomized SVD
(discovery/gate run — the frozen confirmation run uses full fp64 linalg.svd per §3.1).

    pg run infra.aws_pg.jobs.w13_lowrank --type g6e.xlarge \
        --args="--n=300 --ranks=8,32,128,512,1024" --s3-prefix results/w13_lowrank --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEED = 1337
N_BOOT = 2000
TAU_STAR = 0.9          # frozen behavioral-faithfulness target (prereg §6)
DYN_RANGE_MIN = 0.15    # a task carries a verdict only if CEILING-FLOOR >= this (§6A)

from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, _per_item_scores, _boot_ci
from infra.aws_pg.jobs.e4_compress_curve import _set_decoder, _decoder_keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="eval examples per task per graft")
    ap.add_argument("--ranks", default="8,32,128,512,1024", help="comma rank ladder")
    ap.add_argument("--q", type=int, default=1536, help="randomized-SVD rank budget (covers eff-rank tail)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    from infra.pg import battery_tasks as B

    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "behavioral graft needs a GPU (generation)"
    n = 16 if a.smoke else a.n
    ranks = [8, 512] if a.smoke else [int(x) for x in a.ranks.split(",")]
    q = 256 if a.smoke else a.q
    print(f"[W13-LR] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} n={n} ranks={ranks} q={q}", flush=True)

    # CEILING/FLOOR scaffold (bit-identical to E4): pg_after=PG2-pt, pg_before=PG2 vision/proj +
    # base gemma-2-2b decoder. We mutate pg_before's LM per graft.
    pg_after, pg_before, proc, paths, _img_tok, gate = H.build_pair("base-pt", device)
    print(f"[W13-LR] swap_gate={json.dumps(gate, default=str)}", flush=True)
    lm_edit = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    dkeys = _decoder_keys()
    for k in dkeys:
        assert k in base_sd and k in after_sd, f"missing decoder key {k}"
    del pg_after
    torch.cuda.empty_cache()

    tasks = {"refcoco": B.REFCOCO_TASK, "textvqa": B.TEXTVQA_TASK}
    rows = {name: _eval_rows(t, n, SEED) for name, t in tasks.items()}
    for name in tasks:
        print(f"[W13-LR] {name}: {len(rows[name])} eval rows", flush=True)

    def score():
        s = {}
        for name, t in tasks.items():
            sc = _per_item_scores(t, rows[name], _generate(pg_before, proc, t, rows[name], device, a.batch))
            s[name] = _boot_ci(sc, N_BOOT, SEED)
        return s

    # FLOOR (base decoder, no install)
    _set_decoder(lm_edit, base_sd, {}, device)
    floor = score()
    print(f"[W13-LR] FLOOR  refcoco={floor['refcoco']['mean']:.4f} textvqa={floor['textvqa']['mean']:.4f}", flush=True)

    # CEILING = full 2-D ΔW graft (achieved full-reconstruction ceiling, norms at base; §2.4d)
    _set_decoder(lm_edit, base_sd, {k: after_sd[k].to(torch.bfloat16) for k in dkeys}, device)
    ceiling = score()
    print(f"[W13-LR] CEILING refcoco={ceiling['refcoco']['mean']:.4f} textvqa={ceiling['textvqa']['mean']:.4f}", flush=True)

    dyn_range = {name: ceiling[name]["mean"] - floor[name]["mean"] for name in tasks}
    for name in tasks:
        print(f"[W13-LR] dynamic_range[{name}]={dyn_range[name]:.4f} "
              f"({'OK' if dyn_range[name] >= DYN_RANGE_MIN else 'BELOW 0.15 → inconclusive'})", flush=True)

    # ---- truncated SVD of dW per matrix (randomized, generous q) ----
    print(f"[W13-LR] randomized SVD (q={q}) of {len(dkeys)} decoder dW ...", flush=True)
    svd = {}
    for k in dkeys:
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)
        qq = min(q + 256, min(dW.shape))
        U, S, V = torch.svd_lowrank(dW, q=qq, niter=8)
        svd[k] = (U[:, :q].cpu(), S[:q].cpu(), V[:, :q].cpu())
        del dW, U, S, V
        torch.cuda.empty_cache()

    def rankk(r):
        d = {}
        for k in dkeys:
            U, S, V = svd[k]
            kk = min(r, S.shape[0])
            dWk = (U[:, :kk].to(device) * S[:kk].to(device)) @ V[:, :kk].to(device).T
            d[k] = (base_sd[k].float().to(device) + dWk).to(torch.bfloat16)
        return d

    def rcap(sc):
        return {name: {"score": sc[name]["mean"], "ci": [sc[name]["lo"], sc[name]["hi"]],
                       "R_cap": (sc[name]["mean"] - floor[name]["mean"]) /
                                max(ceiling[name]["mean"] - floor[name]["mean"], 1e-9),
                       "R_cap_winsor": min(1.0, (sc[name]["mean"] - floor[name]["mean"]) /
                                           max(ceiling[name]["mean"] - floor[name]["mean"], 1e-9))}
                for name in tasks}

    curve = {}
    for r in ranks:
        _set_decoder(lm_edit, base_sd, rankk(r), device)
        c = rcap(score())
        curve[r] = c
        print(f"[W13-LR] rank={r:5d}  refcoco R_cap={c['refcoco']['R_cap']:.3f} "
              f"(acc={c['refcoco']['score']:.3f})  textvqa R_cap={c['textvqa']['R_cap']:.3f} "
              f"(anls={c['textvqa']['score']:.3f})", flush=True)

    # N_behav per task = smallest rank with winsorized R_cap >= tau*
    n_behav = {}
    for name in tasks:
        hit = [r for r in ranks if curve[r][name]["R_cap_winsor"] >= TAU_STAR]
        n_behav[name] = min(hit) if hit else None

    cliff_rank = min(ranks)
    results = {
        "experiment": "w13_lowrank", "arm": "low-rank (replication / no confirmatory weight)",
        "status": "discovery", "n_per_task": n, "ranks": ranks, "svd_q": q,
        "tau_star": TAU_STAR, "seed": SEED, "swap_gate": gate,
        "floor": floor, "ceiling": ceiling, "dynamic_range": dyn_range,
        "curve": {str(r): curve[r] for r in ranks},
        "N_behav": n_behav,
        "cliff_gate_9_1d": {
            "rank": cliff_rank,
            "refcoco_R_cap": curve[cliff_rank]["refcoco"]["R_cap"],
            "textvqa_R_cap": curve[cliff_rank]["textvqa"]["R_cap"],
            "note": "§9-1d: low rank should close little CAPABILITY (the cliff). RefCOCO R_cap≈0 expected.",
        },
    }
    (out / "w13_lowrank.json").write_text(json.dumps(results, indent=2))
    print(f"[W13-LR] N_behav(τ*={TAU_STAR}): refcoco={n_behav['refcoco']} textvqa={n_behav['textvqa']}", flush=True)
    print(f"[W13-LR] cliff rank-{cliff_rank}: refcoco R_cap={results['cliff_gate_9_1d']['refcoco_R_cap']:.3f} "
          f"textvqa R_cap={results['cliff_gate_9_1d']['textvqa_R_cap']:.3f}", flush=True)
    print("[W13-LR] done → w13_lowrank.json", flush=True)


if __name__ == "__main__":
    main()
