"""POSTCHECK offline gate — the whole pipeline minus the model forward, on the control plane.

Nothing here touches a GPU, a checkpoint or the network. It runs:

  1. the P1 weights reader's self-test (sigmoid at a known bias; a moved bias must fail the claim)
  2. the P4 scanner's self-test (a synthetic three-file dataset with one planted file_index mismatch
     and one planted absent episode; and the clean copy, which must find neither)
  3. the edit battery and its aggregate under TWO planted physics — a model that reads its depth
     prefix and one that does not — in BOTH remove modes, because `reads_the_prefix` deciding the
     message must be shown to fire in both directions before it is allowed to decide anything
  4. the battery's own refusals: an unknown remove mode, and a frame whose grid is short

`python -m infra.aws_pg.jobs.postcheck_dryrun` must print GREEN before a GPU cell is launched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

from infra.pg import depthanchor_core as core
from infra.pg import postcheck_battery as pb
from infra.pg import postcheck_dataset as pcd
from infra.pg import postcheck_weights as pcw

GRID = core.N_CELLS


class StubResult:
    def __init__(self, a_norm, bins, gate=None, n_tail=1, seconds=0.01, meta=None):
        self.a_norm = a_norm
        self.a_robot = np.clip(a_norm, -1.0, 1.0) * 0.9375
        self.bins = np.asarray(bins, dtype=np.int64)
        self.token_ids = np.arange(self.bins.size + 2)
        self.n_grid_tokens = int(self.bins.size)
        self.n_tail_tokens = n_tail
        self.seconds = seconds
        self.gate = gate
        self.meta = meta or {}


class StubActor:
    """A deterministic stand-in for the released model.

    `mode="reads"`   the action depends on the grid, so shuffling it moves the action well above the
                     seed floor — the outcome under which N1 would change the message.
    `mode="inert"`   the action depends on the image and the seed only, and the grid contributes an
                     amount far below the sampler noise — DEPTHANCHOR's measured outcome.
    Either way the RGB swap moves the action a lot, because the image is what the policy reads.
    """

    def __init__(self, mode: str, short_grid_on: Optional[str] = None):
        self.mode = mode
        self.calls = 0
        self.short_grid_on = short_grid_on
        self._captured: Dict[str, object] = {}
        self._forced: Optional[float] = None

    # the pieces of the battery's interface
    def force_gate(self, value):
        self._forced = None if value is None else float(value)

    def _bins_for(self, frame) -> np.ndarray:
        rng = core.rng_for("stub-grid", frame["spec"]["frame_id"])
        n = GRID - 1 if frame["spec"]["frame_id"] == self.short_grid_on else GRID
        return rng.integers(0, core.N_CODES, size=n)

    def _action(self, frame, bins, seed: int, image_key: str, depth_scale: float) -> np.ndarray:
        img = core.rng_for("stub-img", image_key).normal(size=(core.CHUNK, core.ACTION_DIM))
        noise = core.rng_for("stub-seed", frame["spec"]["frame_id"], seed).normal(
            size=(core.CHUNK, core.ACTION_DIM)) * 0.02
        b = np.asarray(bins, dtype=np.float64)
        depth = np.zeros((core.CHUNK, core.ACTION_DIM))
        if b.size:
            # a term that depends on WHERE each code sits, so a permutation changes it
            w = np.cos(np.arange(b.size) * 0.37)
            depth[:, 0] = float(np.dot(b - b.mean(), w) / max(1.0, b.size))
            depth[:, 1] = float(b[: b.size // 2].mean() - b[b.size // 2:].mean())
        return img + noise + depth_scale * depth

    @staticmethod
    def _image_key(frame) -> str:
        """Keyed on the IMAGE, not on the frame id: the ceiling cell swaps the image and keeps the
        spec, so a stub that keys on the id has a ceiling of exactly zero and every ratio divides by
        it. That is how this gate caught its own stub."""
        import hashlib
        return hashlib.sha256(np.ascontiguousarray(frame["agentview"]).tobytes()).hexdigest()[:16]

    def natural(self, frame, seed: int = 0):
        self.calls += 1
        bins = self._bins_for(frame)
        return StubResult(self._action(frame, bins, seed, self._image_key(frame), self._scale()),
                          bins, gate=self._gate())

    def inject(self, frame, bins, seed: int = 0):
        want = np.asarray(bins, dtype=np.int64).reshape(-1)
        if want.shape != (GRID,):
            raise ValueError(f"inject needs {GRID} bins, got {want.shape}")
        self.calls += 1
        scale = 0.0 if self._forced == 0.0 else self._scale()
        return StubResult(self._action(frame, want, seed, self._image_key(frame), scale),
                          want, gate=self._gate())

    def _scale(self) -> float:
        # "reads" is planted an order of magnitude ABOVE the sampler noise and "inert" three orders
        # below it, so `reads_the_prefix` has to separate two cases that are not close to the margin.
        # The first draft put "reads" at 0.02 against a seed floor of the same size and the clause
        # came out False on a model that reads its prefix — a planted physics too weak to be planted.
        return 0.5 if self.mode == "reads" else 2e-5

    def _gate(self):
        return None if self.mode == "base" else np.full(4, 0.018)

    def _scale_str(self) -> str:
        return f"{self._scale():g}"


def _frames(n_per_task: int = 3, n_tasks: int = 4) -> List[Dict[str, object]]:
    out = []
    for t in range(n_tasks):
        for k in range(n_per_task):
            fid = f"render:suite{t % 2}/{t}:s{k * 20}"
            rng = core.rng_for("stub-frame", fid)
            out.append({
                "agentview": rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
                "state": rng.normal(size=8).astype(np.float32),
                "task": f"pretend task {t}",
                "spec": {"frame_id": fid, "suite": f"suite{t % 2}", "task_key": f"suite{t % 2}/{t}",
                         "step": k * 20},
            })
    return out


def _ladder() -> core.CodeLadder:
    d = np.sort(core.rng_for("stub", "ladder").normal(size=core.N_CODES))
    return core.CodeLadder(d_primary=d, d_local=d + 1e-3)


def stage_battery() -> Dict[str, object]:
    ladder = _ladder()
    frames = _frames()
    results = {}
    for mode, remove, expect_reads in (("reads", "gate0", True), ("inert", "gate0", False),
                                       ("reads", "no_depth", True), ("inert", "no_depth", False)):
        actor = StubActor(mode)
        if remove == "no_depth":
            # the battery calls `call_no_depth`, which needs the released model; the stub stands in
            pb_call = pb.call_no_depth
            pb.call_no_depth = lambda a, f, seed=0: StubResult(
                a._action(f, np.zeros(0), seed, a._image_key(f), 0.0), np.zeros(0), gate=None)
        try:
            bat = pb.run_battery(actor, frames, ladder, remove_mode=remove)
            agg = pb.aggregate(bat)
        finally:
            if remove == "no_depth":
                pb.call_no_depth = pb_call
        key = f"{mode}:{remove}"
        assert agg["g0_passes_1e-4"], (key, agg["g0_max"])
        assert agg["tail_mismatches"] == 0, key
        assert agg["reads_the_prefix"] is expect_reads, (
            key, agg["shuffle_over_ceiling"], agg["m"])
        # the remove cell must exist and must say which mode it ran
        removed = [r for r in bat["records"] if r["cell"] == "remove"]
        assert len(removed) == len(frames), key
        assert all(r["remove_mode"] == remove for r in removed), key
        results[key] = {
            "m": agg["m"], "shuffle_over_ceiling": agg["shuffle_over_ceiling"]["point"],
            "reads_the_prefix": agg["reads_the_prefix"],
            "remove_over_ceiling": agg["ratio_to_ceiling"]["remove"]["point"],
            "n_records": len(bat["records"]), "calls": actor.calls,
        }
        print(f"[dryrun] {key:14s} m={agg['m']:.4f} shuffle/C={agg['shuffle_over_ceiling']['point']:.3e} "
              f"reads={agg['reads_the_prefix']}", flush=True)
    return results


def stage_refusals() -> Dict[str, object]:
    ladder = _ladder()
    frames = _frames()
    out = {}
    try:
        pb.run_battery(StubActor("inert"), frames, ladder, remove_mode="whatever")
        raise AssertionError("an unknown remove mode must raise")
    except ValueError as exc:
        out["unknown_remove_mode_raises"] = str(exc)[:120]
    # a frame whose grid is short carries no records and is counted, never padded
    short = frames[0]["spec"]["frame_id"]
    bat = pb.run_battery(StubActor("inert", short_grid_on=short), frames, ladder, remove_mode="gate0")
    assert [s["frame_id"] for s in bat["frames_skipped_short_grid"]] == [short], bat["frames_skipped_short_grid"]
    assert all(r["frame_id"] != short for r in bat["records"]), "a short-grid frame produced records"
    out["short_grid_frame_skipped_and_counted"] = True
    out["records_without_it"] = len(bat["records"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/postcheck_dryrun")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[dryrun] 1/4 P1 weights reader", flush=True)
    pcw.selftest()
    print("[dryrun] 2/4 P4 dataset scanner", flush=True)
    p4 = pcd.selftest()
    print("[dryrun] 3/4 the edit battery under two planted physics, both remove modes", flush=True)
    battery = stage_battery()
    print("[dryrun] 4/4 the battery's refusals", flush=True)
    refusals = stage_refusals()

    report = {"p1_selftest": "ok",
              "p4_selftest": {"planted_file_index_mismatches": p4["planted"]["file_index_mismatches"],
                              "planted_absent": p4["planted"]["episodes_listed_but_absent"],
                              "clean_mismatches": p4["clean"]["file_index_mismatches"]},
              "battery": battery, "refusals": refusals}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v["reads_the_prefix"] for k, v in battery.items()}, indent=1))
    print("[dryrun] GREEN", flush=True)


if __name__ == "__main__":
    main()
