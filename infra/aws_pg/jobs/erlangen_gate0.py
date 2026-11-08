"""ERLANGEN Gate-0 — freeze preflight (CPU).

Runbook step 1–2. Banks `erlangen_freeze.json`:
  * the spec sha-compare result (computed on the control plane, carried in as an arg — the code
    tarball ships `infra/` only, so the .md cannot be hashed here; the launcher does it and this
    job records WHICH sha it was frozen against, so the two can never silently diverge)
  * every model repo's resolved sha (PG1, gemma-2b, SigLIP, π0, π0.5, the 5 M-task FT roster)
  * the reuse-ledger surface check (every REUSED function actually exists at the pinned path)
  * dataset binding checks (COCO / VQAv2 / wikitext / MMLU all load and yield usable rows)
  * the frozen item manifest + its content hash
  * ft-max / ft-min resolved FROM WEIGHTS (amendment A8), not from any prior narrative
  * the coverage enumeration: every (probe family × unit) cell marked
    measured / impossible / VOID-expected / pending (spec §0 governance)

    python -m infra.aws_pg.jobs.erlangen_gate0 --out DIR --spec-sha <sha256> [--skip-ftnorm]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib

# The 5 frozen M tasks and their FT start points (amendment A8), with the pre-registered fallback
# order if a dataset binding fails.
M_TASKS = (
    ("cococap", "google/paligemma-3b-ft-cococap-224"),
    ("refcoco", "google/paligemma-3b-ft-refcoco-seg-224"),
    ("textvqa", "google/paligemma-3b-ft-textvqa-224"),
    ("vqav2", "google/paligemma-3b-ft-vqav2-224"),
    ("okvqa", "google/paligemma-3b-ft-okvqa-224"),
)
FALLBACK_5TH = ("okvqa", "gqa", "ai2d", "science-qa")

LEDGER = (
    ("inherit_pg1.load_pg1", "infra/pg/inherit_pg1.py", "def load_pg1("),
    ("inherit_pg1.harvest", "infra/pg/inherit_pg1.py", "def harvest("),
    ("inherit_pg1.ablate_at", "infra/pg/inherit_pg1.py", "def ablate_at("),
    ("lacuna_shell.canon_map", "infra/pg/lacuna_shell.py", "def canon_map("),
    ("lacuna_shell.install_canon", "infra/pg/lacuna_shell.py", "def install_canon("),
    ("lacuna_shell.build_row_policy", "infra/pg/lacuna_shell.py", "def build_row_policy("),
    ("lacuna_shell.resolve_keys", "infra/pg/lacuna_shell.py", "def resolve_keys("),
    ("projector_ablation.load_eval_items", "infra/pg/projector_ablation.py", "def load_eval_items("),
    ("projector_ablation.build_tasks", "infra/pg/projector_ablation.py", "def build_tasks("),
    ("battery_tasks.COCO_TASK", "infra/pg/battery_tasks.py", "COCO_TASK = TaskSpec("),
    ("invariant_core_func._blank_image", "infra/aws_pg/jobs/invariant_core_func.py", "def _blank_image("),
    ("invariant_core_func._ppl_blankimg", "infra/aws_pg/jobs/invariant_core_func.py", "def _ppl_blankimg("),
    ("specter_core", "infra/pg/specter_core.py", "def "),
    ("relearn", "infra/pg/relearn.py", "def "),
    ("erlangen_core.p1_verdict", "infra/pg/erlangen_core.py", "def p1_verdict("),
    ("erlangen_core.axiom_bootstrap", "infra/pg/erlangen_core.py", "def axiom_bootstrap("),
    ("erlangen_core.ddmin", "infra/pg/erlangen_core.py", "def ddmin("),
    ("erlangen_probe.score_pack", "infra/pg/erlangen_probe.py", "def score_pack("),
    ("erlangen_probe.apply_intervention", "infra/pg/erlangen_probe.py", "def apply_intervention("),
)

# Coverage enumeration (spec §0): probe family × unit. Filled with the measured/impossible/pending
# status so an under-scoped probe renders as a VISIBLE cell instead of silently not existing.
UNITS = ("pg1", "pi05", "ft_max", "ft_min", "gemma1")
FAMILIES = ("T_fingerprint", "A_algebra", "X_transplant", "M_cost", "B_rider", "D_witness",
            "G_grammar")


def coverage_table():
    """Every (family, unit) cell, pre-declared. `impossible` cells carry their reason."""
    cov = {}
    for f in FAMILIES:
        for u in UNITS:
            status, why = "pending", ""
            if f == "T_fingerprint":
                status = "pending"
            elif f == "A_algebra":
                status, why = ("pending", "") if u == "pg1" else \
                    ("out-of-scope", "the algebra is measured on the reference unit only (spec §2-A "
                                     "fixes 8 loci × 3 kinds on ONE model; other units are the "
                                     "transfer question, funded from the curiosity reserve)")
            elif f == "X_transplant":
                status = "pending"
            elif f == "M_cost":
                status, why = ("pending", "") if u in ("ft_max", "ft_min", "pg1") else \
                    ("impossible", "M is defined over the 5 M-task FT start points, not over this unit")
            elif f == "B_rider":
                status, why = ("pending", "") if u in ("ft_max", "ft_min", "pg1") else \
                    ("impossible", "B rides M's training runs only")
            elif f == "D_witness":
                status, why = ("pending", "") if u == "pi05" else \
                    ("out-of-scope", "D searches the PG1→π0 diff (spec §2-D fixes the hop)")
            elif f == "G_grammar":
                status = "pending"
            cov[f"{f}|{u}"] = {"family": f, "unit": u, "status": status, "reason": why}
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-sha", required=True,
                    help="sha256 of the executor spec copy, computed + compared on the control plane")
    ap.add_argument("--amend-sha", default="")
    ap.add_argument("--n-items", type=int, default=32)
    ap.add_argument("--skip-ftnorm", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")     # reference_hf_xet_wedge
    from infra.pg import diff_harvest as H
    H.hf_online_setup()                                   # FIRST, before transformers/datasets

    rep = {"object": "erlangen_freeze_v1", "spec_sha256": a.spec_sha,
           "amendments_sha256": a.amend_sha, "n_items": a.n_items}
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))

    # ---- 1. reuse-ledger surfaces ------------------------------------------------------------
    root = pathlib.Path(__file__).resolve().parents[3]
    ledger = []
    for name, path, pat in LEDGER:
        f = root / path
        ok = f.exists() and pat in f.read_text()
        ledger.append({"surface": name, "file": path, "pattern": pat, "verified": bool(ok)})
    rep["ledger"] = ledger
    rep["ledger_ok"] = all(x["verified"] for x in ledger)
    missing = [x["surface"] for x in ledger if not x["verified"]]
    assert not missing, f"reuse ledger BROKEN — absent surfaces: {missing}"
    print(f"[gate0] ledger {len(ledger)}/{len(ledger)} surfaces verified", flush=True)

    # ---- 2. repo shas -------------------------------------------------------------------------
    from huggingface_hub import HfApi
    api = HfApi()
    repos = {"pg1": "google/paligemma-3b-pt-224", "gemma1": "google/gemma-2b",
             "siglip": "google/siglip-so400m-patch14-224",
             "pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}
    repos.update({f"ft_{t}": r for t, r in M_TASKS})
    rep["repos"] = repos
    rep["shas"] = {k: api.model_info(v, revision="main").sha for k, v in repos.items()}
    print(f"[gate0] resolved {len(rep['shas'])} repo shas", flush=True)
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))

    # ---- 3. dataset bindings + the frozen item manifest ---------------------------------------
    from infra.pg import erlangen_probe as P
    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    rep["readouts"] = list(P.READOUT_NAMES)
    rep["interventions"] = list(P.INTERVENTIONS)
    rep["item_examples"] = {k: {"prompt": v[0]["prompt"][:160], "target": v[0]["target"][:160],
                                "n": len(v)} for k, v in packs.items()}
    print(f"[gate0] item manifest {rep['items_sha256'][:12]} over {len(packs)} channels", flush=True)
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))

    # ---- 4. M-task dataset availability (the A8 fallback decision, made HERE, pre-data) -------
    from datasets import get_dataset_split_names
    # (dataset, split, config) — a config is only needed where the repo registers several
    # (lmms-lab/GQA does; asking without one raises).
    m_data = {"cococap": ("lmms-lab/COCO-Caption", "val", None),
              "refcoco": ("lmms-lab/RefCOCO", "testA", None),
              "textvqa": ("lmms-lab/textvqa", "validation", None),
              "vqav2": ("lmms-lab/VQAv2", "validation", None),
              "okvqa": ("lmms-lab/OK-VQA", "val2014", None),
              "gqa": ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev_balanced_instructions"),
              "ai2d": ("lmms-lab/ai2d", "test", None),
              "science-qa": ("derek-thomas/ScienceQA", "test", None)}
    avail = {}
    rep["m_task_datasets"] = avail

    def _check(t):
        ds, sp, cfg = m_data[t]
        names = list(get_dataset_split_names(ds, config_name=cfg) if cfg
                     else get_dataset_split_names(ds))
        avail[t] = {"dataset": ds, "want_split": sp, "config": cfg, "splits": names[:12],
                    "ok": bool(sp in names)}
        print(f"[gate0] dataset {t}: {ds} split={sp} ok={avail[t]['ok']}", flush=True)
        return avail[t]["ok"]

    for t in ("cococap", "refcoco", "textvqa", "vqav2"):
        assert _check(t), f"M task {t} dataset binding FAILED — it has no pre-registered fallback"
    # the 5th slot walks the frozen fallback order and STOPS at the first that resolves, so the
    # choice can never be made after seeing data (and an unreachable later fallback — GQA's config
    # requirement, which failed the first run — never blocks a working earlier one).
    fifth = next((t for t in FALLBACK_5TH if _check(t)), None)
    assert fifth is not None, f"no usable 5th M task in the frozen fallback order {FALLBACK_5TH}"
    rep["m_tasks_frozen"] = [t for t, _ in M_TASKS[:4]] + [fifth]
    rep["m_task_fts"] = {t: r for t, r in M_TASKS}
    rep["fifth_task_chosen"] = fifth
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))

    # ---- 5. coverage enumeration ---------------------------------------------------------------
    rep["coverage"] = coverage_table()
    rep["coverage_counts"] = {}
    for c in rep["coverage"].values():
        rep["coverage_counts"][c["status"]] = rep["coverage_counts"].get(c["status"], 0) + 1
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))

    # ---- 6. ft-max / ft-min FROM WEIGHTS (amendment A8) ---------------------------------------
    if not a.skip_ftnorm:
        import torch
        from transformers import PaliGemmaForConditionalGeneration

        from infra.pg.lacuna_shell import canon_map

        def _lm(repo, sha):
            m = PaliGemmaForConditionalGeneration.from_pretrained(
                repo, revision=sha, torch_dtype=torch.float32, device_map="cpu",
                attn_implementation="eager")
            sd = canon_map({k: v.detach().cpu() for k, v in m.state_dict().items()})
            del m
            return {k: v for k, v in sd.items()
                    if k.startswith("language_model.layers.") and v.ndim == 2}

        base = _lm(repos["pg1"], rep["shas"]["pg1"])
        norms = {}
        rep["ft_frobenius"] = norms          # bound BEFORE the loop (feedback_fleet_incremental_save)
        for t, r in M_TASKS:
            sd = _lm(r, rep["shas"][f"ft_{t}"])
            tot = 0.0
            for k, v in base.items():
                assert k in sd, f"{r}: missing {k}"
                tot += float(((sd[k] - v) ** 2).sum())
            norms[t] = tot ** 0.5
            del sd
            print(f"[gate0] ‖ΔW_LM‖_F {t} = {norms[t]:.2f}", flush=True)
            (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))
        rep["ft_max"] = max(norms, key=lambda k: norms[k])
        rep["ft_min"] = min(norms, key=lambda k: norms[k])
        print(f"[gate0] ft_max={rep['ft_max']} ft_min={rep['ft_min']}", flush=True)

    rep["G0_GREEN"] = bool(rep["ledger_ok"] and rep.get("items_sha256")
                           and rep.get("m_tasks_frozen"))
    (out / "erlangen_freeze.json").write_text(json.dumps(rep, indent=2))
    print(f"[gate0] DONE G0_GREEN={rep['G0_GREEN']}", flush=True)


if __name__ == "__main__":
    main()
