"""COMWRITE Arm B — the four-cell decomposition, eval-only (comwrite spec §5). One job = one
donor × one item-half × a list of cells. Surgery is comwrite_surgery.build_decoder on the PG1
language_model state-dict INSIDE the full multimodal model (the XGEN decoder-replacement
precedent); vision tower / projector / embeddings never touched (asserted).

LOO discipline (§2): the donor's cells use ITS LEAVE-ONE-OUT basis from the Arm-A bank
(loo__{type}__L{layer}__{donor_idx}.npy at the anatomy S3 prefix) — never a basis its own ΔW
entered. k = k* from the anatomy artifact (asserted against --k). Excluded (L,t) pairs (failed
A1 existence) carry no commons (§8).

Readouts per cell: donor-task score, cross-task scores, blank-image wikitext ppl (the damage
instrument, gated once against the invcore-FUNC artifact on the UNMODIFIED model, |Δ| ≤ 0.5).
Identity cell must be BITWISE base (harness inertness) — asserted before any contrast.
Adjudication (§5.3) happens downstream on the banked metrics, not here.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_b_cells --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/comwrite_b_vqav2_disc --budget-min 300 --args "\
      --donor vqav2 --half disc --cells identity,shared,residual,ft,em_random@0,em_random@1,em_random@2,random@0,shuffle@0 \
      --anatomy-s3 s3://pg-lineage-769063704696/results/comwrite_a_anatomy/out \
      --gate0-s3 s3://pg-lineage-769063704696/results/comwrite_gate0/out --k 16"
"""
import argparse
import json
import pathlib

from infra.aws_pg.jobs.comwrite_gate0 import DONORS, PPL_TOL


def _parse_cell(spec):
    """'shared_k8@1' → ('shared', 8, 1); 'em_random@2' → ('em_random', None, 2);
    'shared' → ('shared', None, 0). A _k<k> suffix overrides k* for that cell — the §5.1
    vqav2 rank ladder (base+shared(k) for k in the ladder)."""
    name, _, draw = spec.partition("@")
    k = None
    if "_k" in name:
        name, _, ks = name.rpartition("_k")
        k = int(ks)
    return name, k, int(draw or 0)


def gold_fns():
    """Per-donor gold-answer accessors for the NLL instrument (dataset schemas verified
    against the wired task builders: vqav2 = transfer_a1 rows with `answers`; textvqa =
    battery majority target_fn; cococap = the first COCO reference caption)."""
    from collections import Counter
    from infra.pg import battery_tasks as BT
    return {"vqav2": lambda r: Counter(x["answer"] for x in r["answers"]).most_common(1)[0][0],
            "textvqa": BT.TEXTVQA_TASK.target_fn,
            "cococap": lambda r: BT._coco_caps(r)[0]}


def nll_eval(model, proc, task, rows, gold_fn, device, batch_size=16):
    """Teacher-forced mean per-token NLL (nats) of the GOLD answer given (image, prompt) —
    the §5.3 cross-task NLL-texture instrument (the C7 H_interface signature adjudicates on
    it; scores alone cannot see a format re-pricing). Uses the processor's suffix path (the
    intended PaliGemma teacher-forcing API); labels are popped so no internal loss runs."""
    import numpy as np
    import torch
    from infra.pg.diff_metrics import bootstrap_ci
    per_item = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = [task.image_fn(r).convert("RGB") for r in batch]
        prompts = [task.prompt_fn(r) for r in batch]
        suffixes = [str(gold_fn(r)) for r in batch]
        assert all(s.strip() for s in suffixes), "empty gold answer — NLL undefined"
        inp = proc(text=prompts, images=images, suffix=suffixes, return_tensors="pt",
                   padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        labels = inp.pop("labels")
        with torch.inference_mode():
            logits = model(**inp).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        tok_nll = -lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        for i in range(len(batch)):
            assert bool(mask[i].any()), "no suffix tokens in labels — processor drift"
            per_item.append(float(tok_nll[i][mask[i]].mean()))
    arr = np.asarray(per_item, dtype=float)
    ci = bootstrap_ci(arr, stat="mean")
    return {"nll": float(arr.mean()), "ci95": [ci["lo"], ci["hi"]], "n": len(per_item),
            "per_item": per_item}


def rows_for(task, wanted_hashes, n_stream):
    """Re-stream the task's items and align to the Gate-0 banked half by content hash — the
    item-level alignment key (§3). n_stream must cover Gate-0's full stream window (the half's
    hashes are spread across it). Fails loud if the stream no longer covers the bank."""
    from infra.pg.projector_ablation import _content_hash, load_eval_items
    wanted = set(wanted_hashes)
    rows = load_eval_items(task, n_stream)
    by_hash = {}
    for r in rows:
        h = _content_hash(task, r)
        if h in wanted and h not in by_hash:
            by_hash[h] = r
    missing = len(wanted) - len(by_hash)
    assert missing == 0, f"{missing}/{len(wanted)} banked items not re-streamed"
    return [by_hash[h] for h in wanted_hashes]                # bank order (stable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor", required=True, choices=sorted(DONORS))
    ap.add_argument("--half", required=True, choices=("disc", "sealed"))
    ap.add_argument("--cells", required=True, help="comma list; draws as name@seed")
    ap.add_argument("--anatomy-s3", required=True, help=".../comwrite_a_anatomy/out")
    ap.add_argument("--gate0-s3", required=True, help=".../comwrite_gate0/out")
    ap.add_argument("--k", type=int, required=True, help="k*; asserted == anatomy k_star")
    ap.add_argument("--n-items", type=int, default=800, help="items per readout task")
    ap.add_argument("--n-cross", type=int, default=400)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--roll", type=int, default=7)
    ap.add_argument("--surgery-types", default="",
                    help="§5.1 location sub-arm: comma list of matrix types to KEEP in the "
                         "surgery; all other (L,t) pairs are moved to `excluded` (no commons "
                         "there: shared→0, residual→ΔW). Empty ⇒ full 18×7.")
    ap.add_argument("--offline", action="store_true",
                    help="the §10 offline gate: full pipeline minus model-forward, on CPU — "
                         "artifact fetches, item alignment, every cell BUILT + strict-loaded "
                         "+ leak-asserted + booked; ppl gate and evals skipped")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [_parse_cell(c) for c in a.cells.split(",")]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import comwrite_surgery as CS
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import build_tasks

    SH.install_pg()                 # pycocoevalcap etc. — NOT on the pg-env AMI (CIDEr path)

    # --- the Arm-A artifact: roster, k*, excluded pairs, this donor's LOO bases. ---
    anat = json.loads(SH.fetch(f"{a.anatomy_s3}/metrics.json",
                               out / "anatomy_metrics.json").read_text())
    assert int(anat["k_star"]) == a.k, f"--k {a.k} != anatomy k*={anat['k_star']}"
    roster = anat["roster"]
    donor_repo = f"google/paligemma-3b-ft-{a.donor}-224"
    donor_idx = roster.index(donor_repo)
    excluded = {(int(L), t) for L, t in anat["excluded_lt"]}
    grid = [(m["layer"], m["type"]) for m in anat["per_matrix"]]
    if a.surgery_types:
        keep = {t.strip() for t in a.surgery_types.split(",")}
        assert keep <= {t for _L, t in grid}, f"unknown type in --surgery-types {keep}"
        excluded |= {(L, t) for L, t in grid if t not in keep}
    k_max = max([a.k] + [k for _n, k, _d in cells if k is not None])
    bases_full = {}
    for L, t in grid:
        if (L, t) in excluded:
            continue
        f = SH.fetch(f"{a.anatomy_s3}/scratch/loo__{t}__L{L}__{donor_idx}.npy",
                     out / f"bank/loo__{t}__L{L}.npy")
        U = np.load(f).astype(np.float32)
        assert U.shape[1] >= k_max, f"bank basis {U.shape} < k_max={k_max}"
        bases_full[(L, t)] = torch.from_numpy(U[:, :k_max].copy())
    bases = {lt: U[:, : a.k] for lt, U in bases_full.items()}      # the k* default grid
    rep = {"object": "comwrite_b_cells_v1", "donor": a.donor, "donor_idx": donor_idx,
           "half": a.half, "k": a.k, "cells": a.cells, "excluded_lt": sorted(excluded),
           "n_bases": len(bases), "anatomy_s3": a.anatomy_s3, "base_sha": anat["base_sha"],
           "readouts": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- items: the Gate-0 banked halves (FRESH comwrite seals), aligned by content hash.
    # Rows are model-independent → streamed ONCE here, reused by every cell. The re-stream
    # window is the bank's own length (the window Gate-0 verifiably streamed). ---
    tasks = {t.name: t for t in build_tasks()}
    halves, rows_cache, rep["realized_n"] = {}, {}, {}
    for slug, tname in DONORS.items():
        it = json.loads(SH.fetch(f"{a.gate0_s3}/items_{slug}.json",
                                 out / f"items_{slug}.json").read_text())
        sealed = set(it["sealed_idx"])
        pick = [h for i, h in enumerate(it["hashes"])
                if (i in sealed) == (a.half == "sealed")]
        n = a.n_items if slug == a.donor else a.n_cross
        assert len(pick) >= n, \
            f"{slug} {a.half} half has {len(pick)} items < requested {n} — re-gate or lower n"
        halves[slug] = pick[:n]
        rows_cache[slug] = rows_for(tasks[tname], halves[slug], len(it["hashes"]))
        rep["realized_n"][slug] = len(rows_cache[slug])
    donor_task = tasks[DONORS[a.donor]]
    cross_slugs = [s for s in DONORS if s != a.donor]
    gold = gold_fns()

    # --- models: PG1 (base, eager, pinned sha) + the donor FT; decoders to CPU fp32-safe sds. ---
    device = "cpu" if a.offline else "cuda"
    rep["offline"] = bool(a.offline)
    model, proc = IP.load_pg1(device=device, sha=anat["base_sha"])
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    base_lm_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
    vt_orig = {k: v.detach().clone().cpu()
               for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
    ft = PaliGemmaForConditionalGeneration.from_pretrained(
        donor_repo, revision=anat["shas"][donor_repo],   # the EXACT sha the LOO bases used
        torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager").eval()
    ft_lm_sd = {k: v.detach().clone()
                for k, v in ft.get_submodule(resolve_submodule_paths(ft)["language_model"])
                .state_dict().items()}
    del ft

    # --- instrument gates BEFORE any contrast (§2): base ppl vs artifact; identity bitwise. ---
    func = json.loads(SH.fetch(SH.INVCORE_FUNC_JSON,
                               "/tmp/comwrite/invcore_func.json").read_text())
    assert anat["base_sha"] == func["base_sha"], \
        (f"provenance fork: anatomy base sha {anat['base_sha'][:8]} != invcore-FUNC "
         f"{func['base_sha'][:8]} — the ppl reference was measured on different weights")
    if a.offline:
        rep["ppl_gate"] = "OFFLINE_SKIPPED (GPU instrument; gated in the GPU run)"
    else:
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": func["capability_baseline"]["ppl"]}
        assert abs(pp["ppl"] - float(func["capability_baseline"]["ppl"])) <= PPL_TOL, \
            f"base ppl gate RED: {pp['ppl']:.3f} vs {func['capability_baseline']['ppl']:.3f}"
    ident_sd, _ = CS.build_decoder("identity", base_lm_sd, ft_lm_sd, bases,
                                   excluded=excluded, roll=a.roll)
    assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), \
        "identity cell not bitwise base — surgery pipeline NOT inert (§5.2 VOID)"
    rep["identity_gate"] = "bitwise_base_ok"
    _w()

    # --- the cells. Each: build → strict-load → leak asserts → three readout families. ---
    dev = next(model.parameters()).device
    for name, k_ov, draw in cells:
        k_cell = k_ov if k_ov is not None else a.k
        tag = f"{name}@{draw}" if k_ov is None else f"{name}_k{k_cell}@{draw}"
        bases_c = bases if k_ov is None else {lt: U[:, :k_cell]
                                              for lt, U in bases_full.items()}
        cell_sd, book = CS.build_decoder(name, base_lm_sd, ft_lm_sd, bases_c,
                                         excluded=excluded, seed=1337 + draw, roll=a.roll)
        missing, unexpected = lm.load_state_dict(
            {k: v.to(dev) for k, v in cell_sd.items()}, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt_now = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt_now[k].detach().cpu(), vt_orig[k]) for k in vt_orig), \
            f"{tag}: LEAK — vision tower changed"
        r = {"book_totals": {k: v for k, v in book.items() if k != "per_key"}}
        if a.offline:
            rep["readouts"][tag] = r
            print(f"[comwrite-b][offline] {tag}: built+loaded+leak-asserted "
                  f"inj={r['book_totals']['total_injected_fro2']:.3e}", flush=True)
            _w()
            continue
        r["donor"] = IP.eval_pg1(model, proc, donor_task, rows_cache[a.donor], "cuda")
        for slug in cross_slugs:
            r[f"cross_{slug}"] = IP.eval_pg1(model, proc, tasks[DONORS[slug]],
                                             rows_cache[slug], "cuda")
            r[f"nll_{slug}"] = nll_eval(model, proc, tasks[DONORS[slug]],
                                        rows_cache[slug], gold[slug], "cuda")
        r["ppl"] = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["readouts"][tag] = r
        print(f"[comwrite-b] {tag}: donor={r['donor']['score']:.4f} "
              f"ppl={r['ppl']['ppl']:.2f} inj={r['book_totals']['total_injected_fro2']:.3e}",
              flush=True)
        _w()

    # restore base (leave the instance's model in a known state for any tail check).
    lm.load_state_dict({k: v.to(dev) for k, v in base_lm_sd.items()}, strict=True)
    rep["green"] = True
    _w()
    if a.offline:
        print("BCELLS_OFFLINE_OK", json.dumps(
            {t: v["book_totals"]["total_injected_fro2"] for t, v in rep["readouts"].items()}))
    else:
        print("BCELLS_OK", json.dumps({t: {"donor": v["donor"]["score"],
                                           "ppl": v["ppl"]["ppl"]}
                                       for t, v in rep["readouts"].items()}))


if __name__ == "__main__":
    main()
