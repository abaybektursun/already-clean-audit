"""INHERIT Arm C (content; descriptive; spec §3, §9 step 8). Linear probes for scene variables on
S_action-projected activations vs equal-k random projections, held-out-EPISODE split, in π0.5 AND
(transported) in PG1. Exploratory throughout (#7); no confirmatory weight.

Scene variables (pinned at run): the LIBERO observation.state (8-dim: arm joints + gripper) is the
guaranteed target (present in every demo batch). Object pose is included IFF the demo batch exposes
it; else descoped + disclosed.

Mechanism: capture the π0.5 backbone residual entering L14 during demo forwards (capture-only hook),
mean-pool per episode-frame, project onto (i) S_action top-k, (ii) equal-k random; ridge-probe the
scene variable, held-out-episode R². The PG1 twin repeats with the TRANSPORTED S_action on PG1
activations over the same visual inputs (does the same content pre-exist in the same directions?).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_probes --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/inherit_probes --budget-min 120 --args "--n-episodes 80"
"""
import argparse
import json
import pathlib
import subprocess
import sys

L_PROBE = 14
K_PROBE = 16


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
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
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_pi05_hook as PH
    from infra.pg import ctrliface_rollout_lerobot as RL
    from infra.pg import inherit_core as IC

    bank = torch.load(SH.fetch(SH.CTRLIFACE_BANK, "/tmp/inherit/bases.pt"),
                      map_location="cpu", weights_only=False)
    d = int(bank["d"])
    Sa = torch.as_tensor(bank["bases"][L_PROBE], dtype=torch.float64)[:, :K_PROBE]
    Rk = IC.random_basis(d, K_PROBE, a.seed, "cpu", torch.float64)
    rep = {"L": L_PROBE, "k": K_PROBE}

    policy = LR.load_policy()
    _bk, layers, _d = LR.get_backbone(policy)
    meta = LR.dataset_meta()
    dt = LR.build_delta_timestamps(policy, meta)
    loader, n_ep, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=a.batch_size,
                                      seed=a.seed, max_episodes=a.n_episodes, shuffle=False)
    pre, _post = RL.load_processors(policy, dataset_stats=meta.stats)

    # capture per-frame pooled residual at L + the scene target (observation.state) per batch item
    Z, Y, ep_ids = [], [], []
    it = iter(loader)
    n_batches = max(1, a.n_episodes // a.batch_size)
    for bi in range(n_batches):
        raw = next(it)
        b = pre(raw)
        cap = PH.capture_activations(policy, lambda bb: policy.forward(bb), [b], d, [L_PROBE],
                                     n_prefix=None)[L_PROBE]
        n_item = raw["observation.state"].shape[0] if torch.is_tensor(raw.get("observation.state")) else \
            b["observation.state"].shape[0]
        pooled = cap.reshape(n_item, -1, d).mean(1)               # (n_item, d)
        state = raw.get("observation.state")
        state = state if torch.is_tensor(state) else b["observation.state"]
        Z.append(pooled.double())
        Y.append(state.detach().to(torch.float64).cpu().reshape(n_item, -1))
        ep_ids.extend([bi] * n_item)                             # batch index as the episode proxy
    Z = torch.cat(Z, 0)
    Y = torch.cat(Y, 0)
    ep_ids = np.asarray(ep_ids)
    rep["n_items"], rep["target_dim"] = int(Z.shape[0]), int(Y.shape[1])

    # held-out-EPISODE split (no batch on both sides)
    uniq = np.unique(ep_ids)
    rng = np.random.default_rng(a.seed)
    held = set(rng.permutation(uniq)[: max(1, len(uniq) // 3)].tolist())
    te = np.array([e in held for e in ep_ids])
    tr = ~te
    assert tr.sum() > 0 and te.sum() > 0

    def probe(basis):
        Zt = (Z - Z.mean(0)) @ basis                             # (n, k) projected features
        r2, per = IC.ridge_r2(Zt[tr], Y[tr], Zt[te], Y[te], alpha=1.0)
        return {"r2_mean": r2, "r2_per_target": per}

    rep["pi05"] = {"saction": probe(Sa), "random_k": probe(Rk),
                   "full": probe(torch.eye(d, dtype=torch.float64))}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- PG1 twin: transported S_action on PG1 activations over the SAME robot images ---
    # (descriptive corroboration: does the same scene content pre-exist in the same directions in PG1?)
    from PIL import Image
    from infra.pg import inherit_pg1 as IP
    del policy
    import gc
    gc.collect(); torch.cuda.empty_cache()
    model, proc = IP.load_pg1()
    pg1_layers = IP.decoder_layers(model)
    # replay the same demo images through PG1 (task text as prompt), capture L residual
    loader2, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=a.batch_size,
                                    seed=a.seed, max_episodes=a.n_episodes, shuffle=False)
    it2 = iter(loader2)
    Zp, Yp, epp = [], [], []
    img_key = None

    def _pil(t):
        arr = t.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        return Image.fromarray(arr.clip(0, 255).astype("uint8")).convert("RGB")

    for bi in range(n_batches):
        raw = next(it2)
        if img_key is None:
            img_key = sorted(k for k in raw if k.startswith("observation.images.")
                             and "empty" not in k)[0]
        ims = [_pil(raw[img_key][j]) for j in range(raw[img_key].shape[0])]
        txt = [str(t) for t in raw["task"]]
        inp = proc(text=txt, images=ims, return_tensors="pt", padding="longest").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        h = IP.harvest(pg1_layers, lambda bb: model(**bb), [inp], [L_PROBE], x_cap=10 ** 9)
        X = h[L_PROBE][2]
        pooled = X.reshape(len(ims), -1, d).mean(1)
        state = raw.get("observation.state")
        Zp.append(pooled.double())
        Yp.append(state.detach().to(torch.float64).cpu().reshape(len(ims), -1))
        epp.extend([bi] * len(ims))
    Zp = torch.cat(Zp, 0); Yp = torch.cat(Yp, 0); epp = np.asarray(epp)
    tep = np.array([e in held for e in epp]); trp = ~tep

    def probe_pg1(basis):
        Zt = (Zp - Zp.mean(0)) @ basis
        r2, per = IC.ridge_r2(Zt[trp], Yp[trp], Zt[tep], Yp[tep], alpha=1.0)
        return {"r2_mean": r2, "r2_per_target": per}

    rep["pg1_twin"] = {"saction": probe_pg1(Sa), "random_k": probe_pg1(Rk)}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print("PROBES_OK", json.dumps({"pi05_saction_r2": rep["pi05"]["saction"]["r2_mean"],
                                   "pi05_random_r2": rep["pi05"]["random_k"]["r2_mean"],
                                   "pg1_saction_r2": rep["pg1_twin"]["saction"]["r2_mean"]},
                                  default=str))


if __name__ == "__main__":
    main()
