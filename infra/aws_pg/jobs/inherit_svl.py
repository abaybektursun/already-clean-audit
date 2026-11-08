"""INHERIT Arm G-reverse (secondary, corroborative; spec §3, §9 step 7 — budget-gated). Estimate
S_vl NATIVELY on PaliGemma-1 (the VL next-token-CE gradient Gram at the residual entering layer L),
then ablate its transported span in π0.5 ROLLOUTS vs an energy-matched random null: does robot
control depend on directions that were VL-load-bearing in PG1 BEFORE robot training? A positive
makes the inheritance claim bidirectional. No BACKWARD transport is needed (S_vl is estimated on
PG1 and applied in π0.5; both are the same-index 2048-d residual by lineage).

Two models, sequenced to fit one GPU: PG1 Gram first (freed), then π0.5 for the rollout.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_svl --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/inherit_svl --budget-min 240 \
      --args "--n-grad-items 256 --tasks-per-suite 2 --n-episodes 6"
"""
import argparse
import contextlib
import json
import pathlib
import subprocess
import sys

SEALED_SUITE_ENVS = {"goal": "libero_goal", "long": "libero_10"}
K_SVL = 16
DELTA_FLOOR = 0.15


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
                    "libxext6 libxrender1 libgomp1 git python3-dev build-essential"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)
    import os
    os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    subprocess.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=14, help="S_vl estimation+ablation layer (14=pi05 bite; 0=fwd-inherit locus)")
    ap.add_argument("--n-grad-items", type=int, default=256)
    ap.add_argument("--tasks-per-suite", type=int, default=2)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import ctrliface_core as CC
    from infra.pg import inherit_core as IC
    from infra.pg import inherit_pg1 as IP

    L_SVL = int(a.layer)
    ctrl = json.loads(SH.fetch(SH.CTRLIFACE_CONTROLS, "/tmp/inherit/ctrliface_controls.json").read_text())
    floor = float(ctrl["floor"])
    rep = {"L": L_SVL, "k": K_SVL, "sealed_suites": list(SEALED_SUITE_ENVS),
           "note": "L=14 is the CTRLIFACE pi05 bite; L=0 is the forward-inheritance locus (disc)"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # ---- Stage 1: S_vl on PG1 (VL next-token CE gradient Gram at the input-residual of layer L) ----
    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    _vqa, coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(coco, a.n_grad_items * 2)
    split = IC.sha1_split(hashes, name="inherit_coco")             # DISCOVERY half only (no test leak)
    disc = [rows[i] for i in split["disc_idx"]][: a.n_grad_items]

    # build teacher-forced VL batches: image + "caption en\n" prefix, caption as suffix labels
    def _batches():
        for s in range(0, len(disc), a.batch_size):
            chunk = disc[s:s + a.batch_size]
            images = [coco.image_fn(r).convert("RGB") for r in chunk]
            prefix = [coco.prompt_fn(r) for r in chunk]
            suffix = [coco.refs_fn(r)[0] for r in chunk]          # one reference caption as the target
            inp = proc(text=prefix, images=images, suffix=suffix, return_tensors="pt",
                       padding="longest").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            yield inp

    def loss_fn(inp):
        out = model(**{k: v for k, v in inp.items()})            # processor built labels via suffix
        assert out.loss is not None, "no VL loss (suffix labels missing)"
        return out.loss

    batches = list(_batches())
    G, npos = IP.accumulate_gram_pre(model, layers, loss_fn, batches, [L_SVL], d=IP.D)
    Svl, evals = CC.topk_eigvecs(G[L_SVL].cpu(), 128)
    rep["svl_positions"] = npos
    rep["svl_top_evals"] = [float(x) for x in evals[:16]]
    torch.save({"Svl": Svl.to(torch.float32), "evals": evals}, out / "svl_bases.pt")
    # PG1-native h̄/trace at L for the transported em-null (host value belongs to π0.5 at rollout —
    # but the em-null RANK is set from the removed energy; we set it on π0.5 activations below).
    del model, proc, layers
    import gc
    gc.collect(); torch.cuda.empty_cache()
    _w()

    # ---- Stage 2: ablate S_vl(transported) in π0.5 rollouts vs energy-matched random ----
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_pi05_hook as PH
    from infra.pg import ctrliface_rollout_lerobot as RL
    from infra.aws_pg.jobs.inherit_rollout_op import _roll
    policy = LR.load_policy()
    _bk, play, _d = LR.get_backbone(policy)
    meta = LR.dataset_meta()
    pre, post = RL.load_processors(policy, dataset_stats=meta.stats)
    bank = torch.load(SH.fetch(SH.CTRLIFACE_BANK, "/tmp/inherit/bases.pt"),
                      map_location="cpu", weights_only=False)
    hbar_L = torch.as_tensor(bank["hbar"][L_SVL], dtype=torch.float64)   # π0.5-native replacement
    trace_L = float(bank["trace"][L_SVL])
    tasks = [(env, tid) for env in SEALED_SUITE_ENVS.values() for tid in range(a.tasks_per_suite)]

    # capture π0.5 activations at L to set E_target of S_vl(k) (energy-matched null on the HOST)
    dt = LR.build_delta_timestamps(policy, meta)
    loader, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=2, seed=1,
                                   max_episodes=40)
    cap = [pre(b) for b in __import__("itertools").islice(loader, 6)]
    X = PH.capture_activations(policy, lambda b: policy.forward(b), cap, IP.D, [L_SVL], n_prefix=None)[L_SVL]
    Svl_k = Svl[:, :K_SVL].double()
    E_target, tr = SH.em_rank_inputs(X, hbar_L, Svl_k)
    Qem, r_e = CC.energy_matched_random(IP.D, E_target, tr, a.seed, "cpu", torch.float64)
    rep["em_rank"] = int(r_e)

    intact = _roll(policy, pre, post, L_SVL, None, None, lambda: contextlib.nullcontext(),
                   a.seed, tasks, a.n_episodes)
    bt_svl = _roll(policy, pre, post, L_SVL, Svl_k, hbar_L,
                   lambda: PH.ablate_residual(policy, L_SVL, Svl_k, hbar_L, None),
                   a.seed, tasks, a.n_episodes)
    bt_em = _roll(policy, pre, post, L_SVL, Qem, hbar_L,
                  lambda: PH.ablate_residual(policy, L_SVL, Qem, hbar_L, None),
                  a.seed, tasks, a.n_episodes)
    gap = CC.gap_cluster_bootstrap(bt_svl, bt_em, intact, floor, seed=a.seed)   # RF(em)−RF(svl)
    rep["intact_success"] = float(np.concatenate([np.asarray(v) for v in intact.values()]).mean())
    rep["rf_svl"] = CC.rf_cluster_bootstrap(bt_svl, intact, floor, seed=a.seed)
    rep["rf_em"] = CC.rf_cluster_bootstrap(bt_em, intact, floor, seed=a.seed)
    rep["gap_em_minus_svl"] = gap
    rep["bidirectional_inherit"] = bool(gap["ci_lo"] > DELTA_FLOOR)
    _w()
    print("SVL_OK", json.dumps({"rf_svl": rep["rf_svl"]["rf"], "rf_em": rep["rf_em"]["rf"],
                                "gap_ci_lo": gap["ci_lo"],
                                "bidirectional": rep["bidirectional_inherit"]}, default=str))


if __name__ == "__main__":
    main()
