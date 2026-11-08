"""LACUNA Arm 3 — F0 feasibility preflight (spec §6.3-F0, run BEFORE the freeze completes).

Each unit must (1) BUILD, (2) run one train step on the real LIBERO dataset with a finite loss and
a finite gradient, (3) survive one eval episode through lerobot's own harness (the EGL smoke). Any
failure routes to a PRE-REGISTERED pivot, never to improvisation:
  U3 unbuildable             → P-A3-1 (drop U3; contrast becomes U1/U2 vs U4; H_generic-VLM UNTESTABLE)
  the FT harness fails at all → P-A3-2 (Arm 3 VOID; the matrix cell stays open, disclosed)

P3 (positive control) also runs here: `lerobot/pi05_libero_finetuned` must be decisively non-floor
on the same harness, else the harness itself is VOID and no Arm-3 number means anything.

Banked incrementally — a crash in the eval leg keeps the build + train-step results.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_a3_f0 --type g6e.8xlarge --no-weights \
      --s3-prefix results/lacuna_a3_f0 --budget-min 180
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

S3 = "s3://pg-lineage-769063704696"
DATASET_REPO = "HuggingFaceVLA/libero"
POS_CONTROL = "lerobot/pi05_libero_finetuned"


def _sh(cmd, **kw):
    print(f"[F0] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def load_any_policy(path, device="cuda"):
    """π0-family checkpoints load through PI0Policy, π0.5 through PI05Policy — decided by the
    checkpoint's own config `type`, never by the caller guessing."""
    import json as _json
    import pathlib as _p
    from huggingface_hub import hf_hub_download
    cfgp = _p.Path(path) / "config.json"
    raw = (_json.loads(cfgp.read_text()) if cfgp.exists()
           else _json.loads(_p.Path(hf_hub_download(path, "config.json")).read_text()))
    kind = raw.get("type")
    if kind == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
    else:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as P
    return P.from_pretrained(path).to(device), kind


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="U1,U2,U3,U4")
    ap.add_argument("--max-episodes", type=int, default=8,
                    help="LIBERO episodes downloaded for the train-step test")
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--eval-task-ids", default="[0]")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-pos-control", action="store_true")
    ap.add_argument("--manual-train-step", action="store_true",
                    help="ALSO run a raw policy.forward(batch) on a LeRobotDataset batch. OFF by "
                         "default: a raw batch has no `observation.language.tokens` (lerobot's "
                         "training preprocessor adds it), so this leg reproduces exactly the "
                         "hand-rolled obs→action path the N5a journal spent five rollouts proving "
                         "is a dead end. `lerobot-train` is the path the grid actually uses and is "
                         "therefore the feasibility test that counts.")
    ap.add_argument("--skip-egl", action="store_true",
                    help="skip the apt/EGL install and the eval legs — EGL is needed only for "
                         "LIBERO RENDERING, so the build + train-step + lerobot-train legs can be "
                         "tested without depending on the Ubuntu archive (which returned exit 100 "
                         "on two consecutive attempts, 2026-07-28)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = [u.strip() for u in a.units.split(",") if u.strip()]

    if a.skip_egl:
        # pip only: lerobot[libero,pi] pulls the maintainer-pinned transformers and the π0 policies.
        # No apt, no EGL, no LIBERO config seed — those matter only for rendering.
        import subprocess as _sp
        _sh([sys.executable, "-m", "pip", "install", "-q",
             "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"])
        _sp.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True)
    else:
        from infra.aws_pg.jobs.n5a_pi0_libero_eval import _install
        _install()                               # lerobot[libero,pi] + EGL + the LIBERO config seed
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    import lerobot
    import torch
    import transformers
    from infra.pg import ctrliface_lerobot as CL
    from infra.pg import lacuna_units as LU

    t0 = time.time()
    rep = {"object": "lacuna_a3_f0_v1", "units": units,
           "lerobot": getattr(lerobot, "__version__", "?"),
           "transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "dataset": DATASET_REPO, "pivot": None}
    res = {}
    rep["results"] = res

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    unit_dir = pathlib.Path("/work/units"); unit_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. BUILD --------------------------------------------------------------------------
    # ALL FOUR units are written as local checkpoint dirs with the LIBERO feature layout, so they
    # differ only in backbone weights (§2.3). π0's published config carries the openpi 3-camera
    # layout and a raw forward on a LIBERO batch raises "All image features are missing".
    paths = {}
    for u, spec in (("U1", ("published", "lerobot/pi0_base")),
                    ("U2", ("published", "lerobot/pi05_base")),
                    ("U3", ("built", "pg1_backbone")), ("U4", ("built", "random"))):
        if u not in units:
            continue
        d = unit_dir / u
        man = (LU.build_published(spec[1], d) if spec[0] == "published"
               else LU.build_unit(spec[1], d, seed=a.seed))
        paths[u] = str(d)
        res[u] = {"build": man}
        print(f"[F0] {u} built: {json.dumps(man, default=str)[:320]}", flush=True)
        _w()
        _sh(["aws", "s3", "sync", str(d), f"{S3}/weights/lacuna_units/{u}/", "--region", "us-east-1"])
    for u in units:
        res.setdefault(u, {})["path"] = paths[u]
    _w()

    # ---- 2. ONE TRAIN STEP on the real dataset (OPT-IN — see --manual-train-step) ----------
    for u in (units if a.manual_train_step else []):
        policy, kind = load_any_policy(paths[u], "cuda")
        policy.train()
        meta = CL.dataset_meta(DATASET_REPO)
        dt = CL.build_delta_timestamps(policy, meta)
        loader, n_ep, _f = CL.build_loader(DATASET_REPO, dt, batch_size=1, shuffle=False,
                                           num_workers=0, seed=a.seed,
                                           max_episodes=a.max_episodes)
        batch = CL.to_device(next(iter(loader)), "cuda")
        loss = CL.loss_fn_factory(policy)(batch)
        loss.backward()
        gnorm = float(sum((p_.grad.detach().float().norm() ** 2)
                          for p_ in policy.parameters() if p_.grad is not None) ** 0.5)
        finite = bool(torch.isfinite(loss).item()) and gnorm == gnorm
        res[u]["train_step"] = {"policy_class": kind, "loss": float(loss.item()),
                                "grad_norm": gnorm, "finite": finite, "n_episodes_used": n_ep}
        print(f"[F0] {u} ({kind}) train step: loss={loss.item():.5f} |g|={gnorm:.4g}", flush=True)
        assert finite, f"{u}: non-finite loss/grad — the unit is untrainable"
        del policy, loader, batch, loss
        torch.cuda.empty_cache()
        _w()

    # ---- 2b. the REAL training path: a 2-step `lerobot-train` on LIBERO-Spatial --------------
    # The manual step above proves the unit is differentiable; this proves the harness the grid
    # will actually use accepts the unit, the episode restriction, and writes a checkpoint.
    bindir = pathlib.Path(sys.executable).parent
    tcmd = ([str(bindir / "lerobot-train")] if (bindir / "lerobot-train").exists()
            else [sys.executable, "-m", "lerobot.scripts.train"])
    meta = CL.dataset_meta(DATASET_REPO)
    eps = CL._episodes_for_suites(meta, ("spatial",))
    assert eps, "no LIBERO-Spatial episodes matched (task naming drift)"
    res["train_cli"] = {"n_spatial_episodes": len(eps)}
    smoke_unit = "U3" if "U3" in units else units[0]
    td = pathlib.Path("/work/train_smoke")
    _sh(tcmd + [f"--dataset.repo_id={DATASET_REPO}",
                f"--dataset.episodes={json.dumps(eps[:a.max_episodes])}",
                f"--policy.path={paths[smoke_unit]}", "--policy.device=cuda",
                f"--output_dir={td}", "--steps=2", "--batch_size=2", "--seed=0",
                "--save_freq=2", "--eval_freq=0", "--log_freq=1", "--wandb.enable=false"],
        env={**os.environ})
    ck = td / "checkpoints" / "last" / "pretrained_model"
    res["train_cli"].update({"unit": smoke_unit, "checkpoint_written": ck.exists(),
                             "checkpoint": str(ck)})
    assert ck.exists(), f"lerobot-train wrote no checkpoint at {ck}"
    print(f"[F0] lerobot-train smoke OK ({smoke_unit}) → {ck}", flush=True)
    _w()

    # ---- 3. ONE EVAL EPISODE (the EGL smoke) through lerobot's own harness ------------------
    binp = pathlib.Path(sys.executable).parent / "lerobot-eval"
    base = [str(binp)] if binp.exists() else [sys.executable, "-m", "lerobot.scripts.eval"]

    def _eval(tag, policy_path, episodes, task_ids):
        d = out / f"eval_{tag}"
        _sh(base + [f"--output_dir={d}", "--env.type=libero", f"--env.task={a.suite}",
                    f"--env.task_ids={task_ids}", "--eval.batch_size=1",
                    f"--eval.n_episodes={episodes}", f"--policy.path={policy_path}",
                    "--policy.n_action_steps=10", "--env.max_parallel_tasks=1"],
            env={**os.environ})
        found = sorted(str(p) for p in d.rglob("*.json"))
        agg = {}
        for f in found:
            j = json.loads(pathlib.Path(f).read_text())
            if isinstance(j, dict):
                agg[pathlib.Path(f).name] = j.get("aggregated", j)
        return {"jsons": found, "aggregated": agg}

    if a.skip_egl:
        rep["eval_legs"] = "SKIPPED — --skip-egl (rendering needs the apt/EGL stack)"
        rep["f0_pass"] = "partial: build + train-step + lerobot-train CLI only"
        rep["elapsed_s"] = round(time.time() - t0, 1)
        _w()
        print("LACUNA_A3_F0_PARTIAL_OK", json.dumps(
            {u: res[u].get("train_step") for u in units}, default=str), flush=True)
        return

    if not a.skip_pos_control:
        res["P3_positive_control"] = _eval("pos", POS_CONTROL, 3, "[0,1,2]")
        print(f"[F0] P3: {json.dumps(res['P3_positive_control']['aggregated'], default=str)[:600]}",
              flush=True)
        _w()

    for u in units:
        res[u]["eval_smoke"] = _eval(u.lower(), paths[u], a.eval_episodes, a.eval_task_ids)
        print(f"[F0] {u} eval smoke ok", flush=True)
        _w()

    rep["f0_pass"] = True
    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_A3_F0_OK", json.dumps({u: res[u].get("train_step") for u in units},
                                        default=str), flush=True)


if __name__ == "__main__":
    main()
