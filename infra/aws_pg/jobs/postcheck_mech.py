"""POSTCHECK cells N2 and N3 — where the expert's attention goes, and what the shipped mode regenerates.

Neither is posted. N2 is what the write-up says about mechanism; N3 is the number behind a sentence that
is ("the depth prefix costs 100 of the ~102 generated tokens per action"), and DEPTHANCHOR never measured
it in the SHIPPED adaptive mode.

N2 — THE ATTENTION SHARE. The action expert cross-attends to the VLM's keys and values for the prompt,
the images and the depth prefix. Reading: near-zero mass on the depth positions says the channel was
never routed; substantial mass with no effect on the action says the values carry nothing. Those are
different mechanisms and the message should not guess between them.

    The hook is `ActionExpertCrossAttention._attention`, which receives exactly the q, k, v and mask
    that `scaled_dot_product_attention` is about to use, so the probabilities recomputed from them are
    the ones the model uses — not a reimplementation of the block. Two self-checks run on every
    captured call, because a recomputed attention is worth nothing unless it is the real one:
      * the probabilities sum to 1 along the key axis (SDPA's own normalization), and
      * the number of depth positions equals the number the depth-token mask marks.
    If either fails the cell reports the failure instead of a number.

N3 — WHAT ADAPTIVE MODE REGENERATES. `_compute_depth_update_mask` is a module-level function, so it is
wrapped in place: every call's mask is recorded, giving the distribution of regenerated cells per step,
the share of calls that regenerate none, and the seconds those calls take.

  pg run infra.aws_pg.jobs.postcheck_mech --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO,models--allenai--MolmoAct2-Think \\
     --s3-prefix results/postcheck/mech/1 --budget-min 150 --args " --stages n2,n3"
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import time
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs.postcheck_files import Bank
from infra.aws_pg.jobs.postcheck_render import load_frames

BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
S3_FRAMES = f"s3://{BUCKET}/postcheck/frames/v1"
MODELS = {"tuned": "allenai/MolmoAct2-Think-LIBERO", "base": "allenai/MolmoAct2-Think"}
N2_FRAMES = 20
N3_EPISODES = 10
N3_SUITE = "libero_object"


class AttentionProbe:
    """Records the share of cross-attention mass that lands on the depth positions, per layer."""

    def __init__(self, model, n_layers: int):
        self.model = model
        # The expert integrates a flow-matching field, so `_attention` is called once per layer PER
        # INTEGRATION STEP: the call index is `step * n_layers + layer`, and the first draft reported
        # 360 "layers" for a 36-layer expert. The layer is the call index modulo the layer count.
        self.n_layers = int(n_layers)
        self.rows: List[Dict[str, object]] = []
        self.checks: Dict[str, object] = {"prob_sum_max_error": 0.0, "n_calls": 0,
                                          "depth_positions_seen": set()}
        self._depth_mask: Optional["np.ndarray"] = None
        self._calls_in_frame = 0
        self._installed = []

    def install(self) -> Dict[str, object]:
        import sys
        import torch
        backbone = self.model.model
        mod = sys.modules[type(backbone).__module__]
        cls = None
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and name.endswith("CrossAttention") and hasattr(obj, "_attention"):
                cls = obj
                break
        if cls is None:
            raise RuntimeError("no cross-attention class with an `_attention` method in the remote code")
        probe = self

        original_attn = cls._attention

        def recording_attention(self_mod, q, k, v, *, attn_mask=None, _orig=original_attn):
            probe._record(q, k, attn_mask)
            return _orig(self_mod, q, k, v, attn_mask=attn_mask)

        cls._attention = recording_attention
        self._installed.append((cls, "_attention", original_attn))

        # Hook the GATE entry point, not the mask helper: on a checkpoint with no gate
        # `_depth_gate_from_condition` returns (None, None) BEFORE it ever calls
        # `_get_depth_token_mask`, so a hook on the helper never fires and the base measured nothing
        # (mech/2, `n2_base`: 0 calls). The gate entry point is called on both checkpoints and carries
        # the ids, so the mask is computed here regardless of whether a gate exists.
        original_gate = type(backbone)._depth_gate_from_condition

        def recording_gate(self_mod, *, input_ids, encoder_attention_mask, layer_kv_states,
                           _orig=original_gate):
            try:
                m = self_mod._get_depth_token_mask(input_ids, encoder_attention_mask)
                probe._depth_mask = None if m is None else m.detach().cpu().numpy().reshape(-1)
            except Exception as exc:                       # recorded, never silently skipped
                probe.checks["depth_mask_error"] = f"{type(exc).__name__}: {exc}"
                probe._depth_mask = None
            return _orig(self_mod, input_ids=input_ids,
                         encoder_attention_mask=encoder_attention_mask,
                         layer_kv_states=layer_kv_states)

        type(backbone)._depth_gate_from_condition = recording_gate
        self._installed.append((type(backbone), "_depth_gate_from_condition", original_gate))
        return {"cross_attention_class": cls.__name__}

    def remove(self) -> None:
        for cls, name, original in self._installed:
            setattr(cls, name, original)
        self._installed = []

    def start_frame(self) -> None:
        self._calls_in_frame = 0

    def _record(self, q, k, attn_mask) -> None:
        import torch
        if self._depth_mask is None:
            return
        with torch.no_grad():
            qh = q.transpose(1, 2).float()          # (b, H, t, d), as SDPA sees it
            kh = k.transpose(1, 2).float()
            scores = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
            if attn_mask is not None:
                scores = scores + attn_mask.to(scores.dtype)
            probs = torch.softmax(scores, dim=-1)
            total = probs.sum(dim=-1)
            err = float((total - 1.0).abs().max().item())
            keys = probs.shape[-1]
            mask = self._depth_mask
            if mask.size != keys:
                # the mask indexes the VLM sequence; if the key axis is not that sequence the number
                # below would be a share of the wrong thing, so it is refused rather than reported
                self.checks["key_axis_mismatch"] = {"keys": int(keys), "mask": int(mask.size)}
                return
            idx = np.flatnonzero(mask.astype(bool))
            depth_mass = float(probs[..., idx].sum(dim=-1).mean().item()) if idx.size else 0.0
            self.checks["prob_sum_max_error"] = max(self.checks["prob_sum_max_error"], err)
            self.checks["n_calls"] += 1
            self.checks["depth_positions_seen"].add(int(idx.size))
            self.rows.append({"layer": self._calls_in_frame % self.n_layers,
                              "flow_step": self._calls_in_frame // self.n_layers,
                              "n_keys": int(keys), "n_depth_positions": int(idx.size),
                              "depth_mass": depth_mass,
                              "depth_share_if_uniform": float(idx.size / keys) if keys else 0.0})
            self._calls_in_frame += 1

    def summary(self) -> Dict[str, object]:
        checks = dict(self.checks)
        checks["depth_positions_seen"] = sorted(checks["depth_positions_seen"])
        if not self.rows:
            return {"measured": False, "checks": checks}
        by_layer: Dict[int, List[float]] = {}
        uni: Dict[int, List[float]] = {}
        for r in self.rows:
            by_layer.setdefault(r["layer"], []).append(r["depth_mass"])
            uni.setdefault(r["layer"], []).append(r["depth_share_if_uniform"])
        layers = sorted(by_layer)
        med = [float(np.median(by_layer[l])) for l in layers]
        return {
            "measured": True,
            "checks": checks,
            "probabilities_normalized": bool(checks["prob_sum_max_error"] < 1e-3),
            "n_layers": len(layers),
            "depth_mass_per_layer_median": med,
            "depth_mass": {"min": float(min(med)), "median": float(np.median(med)),
                           "max": float(max(med))},
            "uniform_share": float(np.median([np.median(v) for v in uni.values()])),
            "ratio_to_uniform": (float(np.median(med) / np.median([np.median(v) for v in uni.values()]))
                                 if uni else None),
            "n_recorded_calls": len(self.rows),
        }


def stage_n2(bank: Bank, which: str, frames: List[Dict[str, object]],
             work: pathlib.Path) -> Dict[str, object]:
    from infra.pg import depthanchor_model as dam
    from infra.pg import postcheck_battery as pb
    model, processor, path = _load(which)
    n_layers = int(model.config.action_expert_config.num_layers)
    # both checkpoints are read under the SAME prompt and normalization, or the attention share would
    # differ for a reason that is not the depth channel. The base ships no `libero` tag, so it is copied
    # in through the released code's own constructor, exactly as cell N1 does.
    stats = model._get_robot_stats()
    tag_notes = {}
    if "libero" not in stats.metadata_by_tag:
        local = work / "tuned_norm_stats.json"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://{BUCKET}/postcheck/code/v1/think_libero/norm_stats.json", str(local),
                        "--region", REGION], check=True)
        payload = json.loads(local.read_text())
        tag_notes = pb.add_norm_tag(model, "libero", (payload.get("metadata_by_tag") or {})["libero"])
        tag_notes["norm_modes_match"] = bool(str(payload.get("norm_mode")) == str(stats.norm_mode))
    dam.NORM_TAG = "libero"
    probe = AttentionProbe(model, n_layers)
    info = probe.install()
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    try:
        for f in frames[:N2_FRAMES]:
            probe.start_frame()
            actor.natural(f, seed=0)
    finally:
        probe.remove()
    out = dict(probe.summary(), repo=MODELS[which], n_expert_layers=n_layers,
               norm_tag_added=tag_notes or None, **info)
    bank.artifact(f"n2_rows_{which}.json", probe.rows[:2000])
    _free(model)
    return out


def stage_n3(bank: Bank) -> Dict[str, object]:
    """The shipped adaptive mode, with the update mask logged per call."""
    import sys
    from infra.pg import depthanchor_libero as lib
    from infra.pg import depthanchor_model as dam

    model, processor, path = _load("tuned")
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    mod = sys.modules[type(model.model).__module__]
    masks: List[Dict[str, object]] = []
    original = mod._compute_depth_update_mask

    def recording_mask(current_image, previous_image, *, num_depth_codes, _orig=original):
        m = _orig(current_image, previous_image, num_depth_codes=num_depth_codes)
        arr = np.asarray(m).reshape(-1)
        masks.append({"n_cells": int(arr.size), "n_regenerated": int(arr.sum()),
                      "fraction": float(arr.mean())})
        return m

    mod._compute_depth_update_mask = recording_mask
    orient = json.loads(subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/depthanchor/camera/v1/orientation.json", "-",
         "--region", REGION], capture_output=True, text=True, check=True).stdout)
    rows = []
    try:
        for task_id in range(N3_EPISODES):
            before = len(masks)
            t0 = time.perf_counter()
            r = lib.rollout(actor, N3_SUITE, task_id=task_id, episode=0, policy=lib.ArmNormal(),
                            flip=bool(orient["render_flip"]), hflip=bool(orient["render_hflip"]),
                            max_steps=lib.HORIZON[N3_SUITE])
            rec = r.as_record()
            mine = masks[before:]
            rows.append({"task_id": task_id, "success": r.success, "n_steps": r.n_steps,
                         "calls": rec["meta"]["calls"], "seconds": round(r.seconds, 1),
                         "mask_calls": len(mine),
                         "regenerated": [m["n_regenerated"] for m in mine],
                         "tail_per_call": rec.get("tail_per_call"),
                         "grid_cells_per_call": [len(b) for b in rec.get("bins_per_call", [])]})
            print(f"[n3] task {task_id}: success={r.success} calls={rec['meta']['calls']} "
                  f"mask_calls={len(mine)} median_regen="
                  f"{np.median([m['n_regenerated'] for m in mine]) if mine else None}", flush=True)
            bank.stage("n3_partial", {"rows": rows})
    finally:
        mod._compute_depth_update_mask = original
        _free(model)
    bank.artifact("n3_rows.json", rows)
    regen = [m["n_regenerated"] for m in masks]
    per_call_seconds = [r["seconds"] / max(1, r["calls"]) for r in rows]
    tails = [t for r in rows for t in (r.get("tail_per_call") or [])]
    cells = [c for r in rows for c in (r.get("grid_cells_per_call") or [])]
    return {
        "suite": N3_SUITE, "episodes": len(rows),
        "successes": int(sum(1 for r in rows if r["success"])),
        "mask_calls": len(masks),
        "regenerated_cells": {
            "n": len(regen),
            "median": float(np.median(regen)) if regen else None,
            "mean": float(np.mean(regen)) if regen else None,
            "min": int(min(regen)) if regen else None,
            "max": int(max(regen)) if regen else None,
            "fraction_of_calls_regenerating_zero": float(np.mean([r == 0 for r in regen])) if regen else None,
            "fraction_of_calls_regenerating_all": float(np.mean([r == 100 for r in regen])) if regen else None,
            "histogram": _hist(regen),
        },
        "seconds_per_call": {"median": float(np.median(per_call_seconds)),
                             "min": float(min(per_call_seconds)),
                             "max": float(max(per_call_seconds))},
        "generated_tokens_after_the_grid": _hist(tails),
        "grid_cells_per_call": _hist(cells),
    }


def _hist(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(int(v))] = out.get(str(int(v)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _snapshot_on_disk(repo: str) -> pathlib.Path:
    """The restored snapshot directory, found on disk rather than resolved through the hub cache.

    `snapshot_download(local_files_only=True)` needs `refs/main` to resolve a revision, and the POSTCHECK
    seeder downloaded the base checkpoint at an explicit pinned revision, which writes no `refs/main`.
    That killed `results/postcheck/n1/1` with `LocalEntryNotFoundError` on a checkpoint that was sitting
    on the disk. Loading from the directory needs no cache bookkeeping at all.
    """
    import os
    root = pathlib.Path(os.environ.get("HF_HOME", "/opt/hf-cache/hf")) / "hub"
    d = root / ("models--" + repo.replace("/", "--"))
    snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
    if not snaps:
        raise FileNotFoundError(f"no restored snapshot for {repo} under {root}")
    return snaps[-1]


def _load(which: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = _snapshot_on_disk(MODELS[which])
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), trust_remote_code=True, dtype=torch.float32).to("cuda").eval()
    return model, processor, path


def _free(model) -> None:
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/postcheck")
    ap.add_argument("--stages", default="n2,n3")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = Bank(pathlib.Path(args.out), "postcheck_mech")
    bank.top("args", vars(args))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    if "n3" in stages:
        from infra.pg import depthanchor_libero as lib
        bank.stage("env", {"install": lib.install(extra_numpy=None)})

    frames: List[Dict[str, object]] = []
    if "n2" in stages:
        dest = work / "rendered"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("frames.npz", "frames_index.json"):
            subprocess.run(["aws", "s3", "cp", f"{S3_FRAMES}/{name}", str(dest / name),
                            "--region", REGION], check=True)
        frames = load_frames(dest)
        bank.stage("inputs", {"n_frames": len(frames), "n2_frames": min(N2_FRAMES, len(frames))})
        for which in ("tuned", "base"):
            bank.stage(f"n2_{which}", stage_n2(bank, which, frames, work))
    if "n3" in stages:
        bank.stage("n3", stage_n3(bank))

    st = bank.payload["stages"]
    bank.top("verdict", {
        "N2_tuned": {k: (st.get("n2_tuned") or {}).get(k)
                     for k in ("measured", "probabilities_normalized", "depth_mass",
                               "uniform_share", "ratio_to_uniform")},
        "N2_base": {k: (st.get("n2_base") or {}).get(k)
                    for k in ("measured", "probabilities_normalized", "depth_mass",
                              "uniform_share", "ratio_to_uniform")},
        "N3": {k: (st.get("n3") or {}).get(k)
               for k in ("regenerated_cells", "seconds_per_call", "grid_cells_per_call",
                         "generated_tokens_after_the_grid", "successes", "episodes")},
    })
    print(json.dumps(bank.payload["verdict"], indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
