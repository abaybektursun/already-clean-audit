"""CTRLIFACE rollout binding — reuse lerobot 0.5.2's OWN rollout/env stack with a HOOK-instrumented
policy (the only way to ablate the backbone DURING a LIBERO rollout; the N5a console-script path is
black-box). Correct obs/action/normalization by construction (N5a lesson). Heavy imports inside
functions. Pinned at Gate-0c: `lerobot.scripts.lerobot_eval.rollout(env, policy, env_pre, env_post,
pre, post, seeds)` returns {success:(n,seq)bool, done, reward, action}; `lerobot.envs.factory`
`make_env_config(env_type, **kw)` / `make_env(cfg, n_envs)` / `make_env_pre_post_processors(env_cfg,
policy_cfg)`; `lerobot.policies.factory.make_pre_post_processors(config, pretrained_path, dataset_stats)`.
"""
from __future__ import annotations

import contextlib

from infra.pg.ctrliface_rollout import build_Q_hbar
from infra.pg import ctrliface_pi05_hook as PH
from infra.pg.ctrliface_lerobot import POLICY_REPO, DATASET_REPO


def load_processors(policy, dataset_stats=None):
    """Policy pre/post processors (normalize + tokenize `task` → observation.language.tokens)."""
    from lerobot.policies.factory import make_pre_post_processors
    return make_pre_post_processors(policy.config, pretrained_path=POLICY_REPO,
                                    dataset_stats=dataset_stats)


def dataset_stats(repo=DATASET_REPO):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    return LeRobotDatasetMetadata(repo).stats


def _single_vecenv(envs):
    """Extract the one gym VectorEnv from make_env's dict[str, dict[int, VectorEnv]]."""
    for _name, taskmap in envs.items():
        for _tid, venv in taskmap.items():
            return venv
    raise AssertionError("make_env returned no environment")


def make_libero_vecenv(suite, task_id, n_envs):
    """A LIBERO gym VectorEnv of `n_envs` parallel copies of one task (suite, task_id). Returns
    (venv, env_cfg). `suite` e.g. 'libero_spatial'; task_id int (matches N5a --env.task/--env.task_ids)."""
    from lerobot.envs.factory import make_env, make_env_config
    cfg = make_env_config("libero", task=suite, task_ids=[task_id])
    envs = make_env(cfg, n_envs=n_envs)
    return _single_vecenv(envs), cfg


def _ablation_ctx(policy, spec, bases, hbars, d, seed, n_prefix, E_target, trace_C, n_layers):
    """The ablation as a context manager over the π0.5 monkeypatch (NOT forward hooks — the fused
    backbone forward bypasses layer modules). identity → no-op; positive → full-read mean-replace at
    L; else → mean-ablate span(Q) at L (Q from build_Q_hbar)."""
    kind, L, k = spec["kind"], spec["L"], spec["k"]
    if kind == "identity":
        return contextlib.nullcontext()
    if kind == "positive":
        return PH.mean_replace_at(policy, L, hbars[L], n_prefix)
    Q, hbar = build_Q_hbar(kind, L, k, bases, hbars, d, seed, E_target=E_target,
                           trace_C=trace_C, n_layers=n_layers)
    return PH.ablate_residual(policy, L, Q, hbar, n_prefix)


def eval_task(policy, pre, post, layers, d, suite, task_id, n_episodes, spec, bases, hbars, seed,
              n_prefix=None, E_target=None, trace_C=None):
    """Roll out one LIBERO task `n_episodes` times under an ablation `spec` and return the per-episode
    success list (1.0/0.0). `spec` = {"kind":..,"L":..,"k":..}. The ablation is injected via the π0.5
    compute_layer_complete monkeypatch for the whole rollout. Reuses lerobot's `rollout` (correct
    obs/action/normalization). success is only True at termination ⇒ per-episode = success.any(seq)."""
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    n_layers = len(layers)
    venv, cfg = make_libero_vecenv(suite, task_id, n_envs=n_episodes)
    env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
    if hasattr(policy, "reset"):
        policy.reset()
    torch.manual_seed(seed)                                 # fix the flow-sampler noise across arms
    with _ablation_ctx(policy, spec, bases, hbars, d, seed, n_prefix, E_target, trace_C, n_layers):
        r = rollout(venv, policy, env_pre, env_post, pre, post, seeds=[seed + i for i in range(n_episodes)])
    venv.close()
    return [float(x) for x in r["success"].any(dim=1).float().tolist()]
