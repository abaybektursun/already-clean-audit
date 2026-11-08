"""Resumable stage ledger for the 72-H100-hour PARALLAX truth sprint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json
from .launch_packet import validate_launch_packet


TERMINAL = {"succeeded", "void"}
VALID_STATES = {"pending", "running", "succeeded", "failed", "void", "blocked"}


@dataclass(frozen=True)
class Stage:
    index: int
    stage_id: str
    start_hour: float
    end_hour: float

    @property
    def budget_hours(self) -> float:
        return self.end_hour - self.start_hour


class SprintDAG:
    def __init__(self, matrix_path: Path, state_path: Path):
        self.matrix_path = matrix_path
        self.state_path = state_path
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        rows = matrix["truth_sprint_72h"]["stages"]
        self.stages = [
            Stage(index, row["stage"], float(row["hours"][0]), float(row["hours"][1]))
            for index, row in enumerate(rows)
        ]
        self._validate_definition()

    def _validate_definition(self) -> None:
        if not self.stages or self.stages[0].start_hour != 0:
            raise ValueError("truth sprint must start at hour 0")
        for previous, current in zip(self.stages, self.stages[1:]):
            if previous.end_hour != current.start_hour:
                raise ValueError(f"non-contiguous stages: {previous.stage_id} -> {current.stage_id}")
        if self.stages[-1].end_hour != 72:
            raise ValueError("truth sprint must end at hour 72")
        if any(stage.budget_hours <= 0 or stage.budget_hours > 6 for stage in self.stages):
            raise ValueError("every microstage must have a budget in (0, 6] H100 hours")
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage IDs must be unique")

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "matrix": str(self.matrix_path),
            "cumulative_h100_hours": 0.0,
            "stages": {
                stage.stage_id: {
                    "index": stage.index,
                    "status": "pending",
                    "budget_hours": stage.budget_hours,
                    "attempts": 0,
                    "updated_utc": None,
                    "artifact_manifest": None,
                    "note": None,
                }
                for stage in self.stages
            },
        }

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._initial_state()
            atomic_json(self.state_path, state)
            return state
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _stage(self, stage_id: str) -> Stage:
        try:
            return next(stage for stage in self.stages if stage.stage_id == stage_id)
        except StopIteration as error:
            raise KeyError(f"unknown stage: {stage_id}") from error

    def start(
        self,
        stage_id: str,
        *,
        launch_packet: Path | None = None,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        stage = self._stage(stage_id)
        state = self.load()
        row = state["stages"][stage_id]
        if row["status"] in TERMINAL:
            raise ValueError(f"stage already terminal: {stage_id}={row['status']}")
        if stage.index:
            previous = self.stages[stage.index - 1]
            previous_status = state["stages"][previous.stage_id]["status"]
            if previous_status not in TERMINAL:
                raise ValueError(
                    f"previous stage is not terminal: {previous.stage_id}={previous_status}"
                )

        # The first two microstages are a six-hour preflight envelope. Crossing
        # that cumulative threshold is a big run even though each stage is small.
        if stage.start_hour >= 6:
            if launch_packet is None:
                raise ValueError("launch packet required before cumulative H100 hour 6")
            validate_launch_packet(launch_packet, repo_root=repo_root)

        row["status"] = "running"
        row["attempts"] += 1
        row["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.state_path, state)
        return state

    def finish(
        self,
        stage_id: str,
        *,
        status: str,
        used_h100_hours: float,
        artifact_manifest: Path,
        note: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"succeeded", "failed", "void", "blocked"}:
            raise ValueError(f"invalid finish status: {status}")
        if used_h100_hours < 0:
            raise ValueError("used H100 hours cannot be negative")
        stage = self._stage(stage_id)
        if used_h100_hours > stage.budget_hours * 1.25:
            raise ValueError("stage exceeded the mandatory 1.25x overrun pause")
        if not artifact_manifest.is_file():
            raise FileNotFoundError(artifact_manifest)
        state = self.load()
        row = state["stages"][stage_id]
        if row["status"] != "running":
            raise ValueError(f"stage is not running: {stage_id}={row['status']}")
        row.update(
            {
                "status": status,
                "used_h100_hours": used_h100_hours,
                "artifact_manifest": str(artifact_manifest),
                "note": note,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        state["cumulative_h100_hours"] = round(
            float(state["cumulative_h100_hours"]) + used_h100_hours, 6
        )
        atomic_json(self.state_path, state)
        return state
