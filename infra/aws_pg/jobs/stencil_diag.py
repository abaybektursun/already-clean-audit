"""STENCIL discovery diagnostic (GPU) — the VOID-fork investigation (journal Entry 7). The Phase-A
range probe found STRONG forgetting range but MARGINAL task-gain on professional_law. Before banking
VOID-no-range, this job explores whether the gain axis is dead by SUBJECT and by RECIPE, and — the
decisive readout — measures **training-set accuracy vs held-out accuracy** (memorize-but-don't-
generalize vs no-learning-at-all). free-LoRA only (no banks needed; fits the L40S-48GB pool).

For each (subject, recipe): base held-out acc, base train acc, adapted held-out acc (gain), adapted
train acc (did it learn the training questions?), composite forgetting. A gain axis EXISTS iff some
cell reaches adapted held-out gain > δ_gain clearly above the ~1.6pt eval SE.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_diag \
      --args "--subjects professional_law,elementary_mathematics,high_school_psychology --recipes 2e-4:6:150,5e-5:10:150 --n-vqa 100 --n-coco 32 --n-ppl 100" \
      --type g6e.xlarge,g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,p5.4xlarge --s3-prefix results/stencil_diag --budget-min 300
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="professional_law,elementary_mathematics,high_school_psychology")
    ap.add_argument("--recipes", default="2e-4:6:150,5e-5:10:150", help="lr:epochs:rank comma-list")
    ap.add_argument("--n-vqa", type=int, default=100)
    ap.add_argument("--n-coco", type=int, default=32)
    ap.add_argument("--n-ppl", type=int, default=100)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import stencil_core as SC
    from infra.pg.relearn import qa_item_hash

    SS.install_pg()
    device = "cuda"
    subjects = [s.strip() for s in a.subjects.split(",")]
    recipes = []
    for spec in a.recipes.split(","):
        lr, ep, rk = spec.split(":")
        recipes.append({"lr": float(lr), "epochs": int(ep), "rank": int(rk),
                        "batch_size": 16, "warmup_ratio": 0.1, "micro_batch": 8})
    rep = {"object": "stencil_diag_v1", "subjects": subjects, "recipes": a.recipes, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/stencil")
        mmlu = SS.mmlu_by_subject()
        roster = SS.build_roster(mmlu)
        bctx = SS.make_battery_ctx(proc, roster, n_vqa=a.n_vqa, n_coco=a.n_coco, n_ppl=a.n_ppl)
        snap = SS.snapshot_base(model)
        base_battery = SS.battery_probe(model, proc, device, bctx)
    weights = {c: 1.0 for c in SS.BATTERY_COMPONENTS}
    rep["base_battery"] = base_battery
    _w()

    # base subject references (held-out + training questions), measured once
    subj_data = {}
    with tm.stage("base_subject_refs"):
        for s in subjects:
            tr_rows, ev_rows, ev_hashes = SS.subject_train_eval(mmlu[s]["test"])
            ev_hashes = ev_hashes[:300]                          # bound acc-eval cost (SE≈2pt @ n=300)
            tr_hashes = sorted({qa_item_hash(r["question"]) for r in tr_rows})[:300]
            base_eval = SS.subject_accuracy(model, proc, device, ev_hashes)["acc"]
            base_train = SS.subject_accuracy(model, proc, device, tr_hashes)["acc"]
            subj_data[s] = {"tr_rows": tr_rows, "ev_hashes": ev_hashes, "tr_hashes": tr_hashes,
                            "base_eval": base_eval, "base_train": base_train, "n_eval": len(ev_hashes)}
            rep.setdefault("base_subject", {})[s] = {"base_eval_acc": base_eval, "base_train_acc": base_train,
                                                     "n_eval": len(ev_hashes), "n_train": len(tr_rows)}
    _w()

    cells = []
    rep["cells"] = cells
    for s in subjects:
        d = subj_data[s]
        for rc in recipes:
            SS.reset_model(model, snap)
            examples, dedup = SS.train_examples(d["tr_rows"], d["ev_hashes"])
            SC.attach_free_lora(model, rank=rc["rank"], alpha=rc["rank"], seed=SS.SEED)
            n_batches = max(1, (len(examples) + rc["batch_size"] - 1) // rc["batch_size"])
            total = rc["epochs"] * n_batches

            def _eval(m, step):
                ev = SS.subject_accuracy(m, proc, device, d["ev_hashes"])["acc"]
                trn = SS.subject_accuracy(m, proc, device, d["tr_hashes"])["acc"]
                batt = SS.battery_probe(m, proc, device, bctx)
                comp = SC.composite_forgetting(base_battery, batt, weights)
                # multimodal-collateral composite = the capabilities ORTHOGONAL to text-MMLU adaptation
                # (VQA/COCO); broad-MMLU transfers + text-ppl improves, so they mask real forgetting.
                mm = SC.composite_forgetting(base_battery, batt, {"vqav2": 1.0, "coco_cider": 1.0})
                return {"eval_acc": ev, "train_acc": trn, "forgetting": comp["composite"],
                        "forget_mm": mm["composite"], "battery": batt,
                        "per_component": comp["per_component"]}

            with tm.stage(f"{s}/lr{rc['lr']}"):
                stats = SC.train_with_trajectory(
                    model, proc, device, examples, lr=rc["lr"], epochs=rc["epochs"],
                    batch_size=rc["batch_size"], warmup_ratio=rc["warmup_ratio"], seed=SS.SEED,
                    max_steps=total, micro_batch=rc["micro_batch"],
                    checkpoint_steps=[max(1, total // 3), max(2, 2 * total // 3), total], eval_fn=_eval)
            fin = stats["trajectory"][-1] if stats["trajectory"] else {}
            cell = {"subject": s, "lr": rc["lr"], "epochs": rc["epochs"], "rank": rc["rank"],
                    "base_eval_acc": d["base_eval"], "base_train_acc": d["base_train"],
                    "adapted_eval_acc": fin.get("eval_acc"), "adapted_train_acc": fin.get("train_acc"),
                    "eval_gain": (fin.get("eval_acc", 0) - d["base_eval"]),
                    "train_gain": (fin.get("train_acc", 0) - d["base_train"]),
                    "forgetting": fin.get("forgetting"), "forget_mm": fin.get("forget_mm"),
                    "adapted_battery": fin.get("battery"), "per_component": fin.get("per_component"),
                    "trajectory": [{k: t.get(k) for k in ("step", "eval_acc", "forgetting", "forget_mm", "per_component")}
                                   for t in stats["trajectory"]],
                    "final_loss": stats["final_loss"], "n_eval": d["n_eval"], "steps": stats["steps"]}
            cells.append(cell)
            print(f"[diag] {s} lr{rc['lr']} ep{rc['epochs']}: base_eval={d['base_eval']:.3f} "
                  f"eval_gain={cell['eval_gain']:+.4f} train_gain={cell['train_gain']:+.4f} "
                  f"forget_composite={cell['forgetting']:+.4f} forget_MM(vqa+coco)={cell['forget_mm']:+.4f} "
                  f"per_comp={cell['per_component']}", flush=True)
            _w()

    # verdict: is there a regime with BOTH generalization gain AND multimodal collateral forgetting?
    best_eval_gain = max((c["eval_gain"] for c in cells), default=0.0)
    best_train_gain = max((c["train_gain"] for c in cells), default=0.0)
    # a cell "usable for STENCIL" needs gain>δ AND positive multimodal forgetting at some checkpoint
    usable = [c for c in cells if c["eval_gain"] > 0.03
              and max((t.get("forget_mm") or -9 for t in c["trajectory"]), default=-9) > 0.02]
    best_mm = max((c.get("forget_mm") or -9 for c in cells), default=-9)
    rep["diagnosis"] = {
        "best_eval_gain": best_eval_gain, "best_train_gain": best_train_gain, "best_forget_mm_final": best_mm,
        "gain_axis_alive": bool(best_eval_gain > 0.03),
        "n_usable_cells": len(usable),
        "usable_cells": [{"subject": c["subject"], "lr": c["lr"], "gain": c["eval_gain"],
                          "forget_mm_final": c["forget_mm"]} for c in usable],
        "interpretation": (
            f"PROCEED: {len(usable)} cell(s) show gain>δ AND multimodal collateral forgetting — the "
            "map has a real tradeoff to arbitrate; pick the best for discovery" if usable else
            ("gain exists but NO multimodal collateral forgetting at a gain regime → the tradeoff has "
             "no forgetting axis (VOID on collateral) — a real boundary" if best_eval_gain > 0.03 else
             "no generalization gain axis → VOID"))}
    _w()
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("STENCIL_DIAG_OK", json.dumps(rep["diagnosis"], default=str), flush=True)


if __name__ == "__main__":
    main()
