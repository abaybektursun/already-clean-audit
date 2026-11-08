"""Render a deterministic SVG stub from a frozen PARALLAX pilot result."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .training_common import TrainingGateError


def _points(values: list[float], *, x: float, y: float, width: float, height: float) -> str:
    if not values:
        raise TrainingGateError("figure series is empty")
    low, high = min(values), max(values)
    span = high - low if high > low else 1.0
    denominator = max(1, len(values) - 1)
    return " ".join(
        f"{x + width * index / denominator:.2f},{y + height * (high - value) / span:.2f}"
        for index, value in enumerate(values)
    )


def render(result_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pilot figure: {output}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("label") != "PILOT_MECHANICAL_NOT_PROMOTION_ELIGIBLE":
        raise TrainingGateError("figure input is not a frozen mechanical pilot")
    task_id = result["mechanical_adjudication"]["thresholds"].get("task_id")
    if task_id is None:
        task_id = next(iter(result["snapshots"][0]["evaluation"]["summary"]))
    steps = [int(row["step"]) for row in result["snapshots"]]
    objectives = [float(row["objective"]) for row in result["snapshots"]]
    probabilities = [
        float(row["evaluation"]["summary"][task_id]["mean_target_probability"])
        for row in result["snapshots"]
    ]
    verdict = str(result["mechanical_adjudication"]["verdict"])
    source_sha = sha256_file(result_path)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">
  <rect width="960" height="480" fill="#0f172a"/>
  <text x="48" y="48" fill="#f8fafc" font-family="monospace" font-size="22">PARALLAX PG1 mechanical pilot</text>
  <text x="48" y="76" fill="#94a3b8" font-family="monospace" font-size="13">NOT PROMOTION ELIGIBLE · {html.escape(verdict)}</text>
  <rect x="48" y="112" width="408" height="260" fill="#111827" stroke="#334155"/>
  <rect x="504" y="112" width="408" height="260" fill="#111827" stroke="#334155"/>
  <text x="64" y="140" fill="#cbd5e1" font-family="monospace" font-size="14">fixed-anchor objective</text>
  <text x="520" y="140" fill="#cbd5e1" font-family="monospace" font-size="14">held-out target probability</text>
  <polyline points="{_points(objectives, x=72, y=164, width=360, height=172)}" fill="none" stroke="#fb7185" stroke-width="3"/>
  <polyline points="{_points(probabilities, x=528, y=164, width=360, height=172)}" fill="none" stroke="#38bdf8" stroke-width="3"/>
  <text x="72" y="356" fill="#64748b" font-family="monospace" font-size="12">steps {html.escape(str(steps))}</text>
  <text x="528" y="356" fill="#64748b" font-family="monospace" font-size="12">steps {html.escape(str(steps))}</text>
  <text x="48" y="420" fill="#64748b" font-family="monospace" font-size="11">source sha256 {source_sha}</text>
  <text x="48" y="444" fill="#64748b" font-family="monospace" font-size="11">mechanical rehearsal only · confirmation reads 0</text>
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return {
        "status": "GREEN",
        "source_sha256": source_sha,
        "output": str(output),
        "steps": steps,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.result.resolve(), args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
