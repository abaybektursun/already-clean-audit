"""Render the PARALLAX label-length confound audit as a deterministic teaching SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

from .artifacts import sha256_file
from .training_common import TrainingGateError


def _points(
    values: Sequence[float], *, x: float, y: float, width: float, height: float,
    low: float = -0.02, high: float = 0.11,
) -> str:
    if not values or high <= low:
        raise TrainingGateError("invalid figure series")
    denominator = max(1, len(values) - 1)
    return " ".join(
        f"{x + width * index / denominator:.2f},{y + height * (high - value) / (high - low):.2f}"
        for index, value in enumerate(values)
    )


def _delta_curve(lineage: dict[str, Any], score: str) -> list[float]:
    values = [
        float(row["mean_target_probability"])
        for row in lineage[score]["curve"]
    ]
    return [value - values[0] for value in values]


def _class_lines(lineage: dict[str, Any], score: str) -> list[str]:
    first = lineage[score]["curve"][0]["by_target"]
    last = lineage[score]["curve"][-1]["by_target"]
    result = []
    for target in ("angular_leaf_spot", "bean_rust", "healthy"):
        start = float(first[target]["mean_target_probability"])
        end = float(last[target]["mean_target_probability"])
        result.append(f"{target}: {start:.3f} → {end:.3f} ({end - start:+.3f})")
    return result


def _lineage_panel(
    *, lineage: dict[str, Any], title: str, x: int, color: str,
) -> str:
    summed = _delta_curve(lineage, "registered_sum_score")
    meaned = _delta_curve(lineage, "diagnostic_mean_score")
    sum_gain = summed[-1]
    mean_gain = meaned[-1]
    sum_gate = "PASS" if lineage["registered_sum_score"]["clears_threshold"] else "NO RANGE"
    mean_gate = (
        "PASS" if lineage["diagnostic_mean_score"]["would_clear_registered_threshold"]
        else "NO RANGE"
    )
    return f'''<g>
      <rect x="{x}" y="154" width="522" height="304" rx="16" fill="#111827" stroke="#334155"/>
      <text x="{x + 24}" y="188" class="panel-title">{html.escape(title)}</text>
      <text x="{x + 24}" y="214" class="small">change in mean target probability from step 0</text>
      <line x1="{x + 54}" y1="370.92" x2="{x + 490}" y2="370.92" stroke="#475569"/>
      <line x1="{x + 54}" y1="359.38" x2="{x + 490}" y2="359.38" stroke="#fbbf24" stroke-dasharray="5 5"/>
      <text x="{x + 365}" y="350" class="threshold">frozen +0.01 gate</text>
      <polyline points="{_points(summed, x=x + 54, y=244, width=436, height=150)}" fill="none" stroke="{color}" stroke-width="4"/>
      <polyline points="{_points(meaned, x=x + 54, y=244, width=436, height=150)}" fill="none" stroke="#a78bfa" stroke-width="4" stroke-dasharray="8 6"/>
      <text x="{x + 24}" y="423" class="metric" fill="{color}">REGISTERED sum  {sum_gain:+.4f}  · {sum_gate}</text>
      <text x="{x + 24}" y="447" class="metric" fill="#c4b5fd">DIAGNOSTIC mean {mean_gain:+.4f}  · {mean_gate}</text>
    </g>'''


def render(audit_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("interpretive_verdict") != "VOID_SHARED_RANGE_LABEL_LENGTH_CONFOUND":
        raise TrainingGateError("figure input is not the frozen label-length adjudication")
    pg1 = audit["lineages"]["pg1_pi"]
    pg2 = audit["lineages"]["pg2"]
    pg2_sum_classes = _class_lines(pg2, "registered_sum_score")
    pg2_mean_classes = _class_lines(pg2, "diagnostic_mean_score")
    token_counts = audit["confound"]["candidate_suffix_tokens"]
    source_sha = sha256_file(audit_path)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    .eyebrow {{ fill:#fca5a5; font-size:14px; font-weight:700; letter-spacing:1.5px; }}
    .title {{ fill:#f8fafc; font-size:30px; font-weight:750; }}
    .subtitle {{ fill:#94a3b8; font-size:16px; }}
    .panel-title {{ fill:#f8fafc; font-size:19px; font-weight:700; }}
    .small {{ fill:#94a3b8; font-size:13px; }}
    .threshold {{ fill:#fbbf24; font-size:11px; }}
    .metric {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size:14px; font-weight:700; }}
    .body {{ fill:#cbd5e1; font-size:14px; }}
    .mono {{ fill:#cbd5e1; font-family: ui-monospace, SFMono-Regular, monospace; font-size:13px; }}
    .gate {{ fill:#fecaca; font-size:17px; font-weight:750; }}
  </style>
  <rect width="1200" height="800" fill="#07111f"/>
  <text x="52" y="45" class="eyebrow">PARALLAX · CONFOUND GATE FIRED</text>
  <text x="52" y="82" class="title">The same seed tells two stories until answer length is controlled</text>
  <text x="52" y="112" class="subtitle">Registered evidence is preserved. Shared-range promotion is not licensed.</text>
  {_lineage_panel(lineage=pg1, title="PRIMARY · PaliGemma 1", x=52, color="#38bdf8")}
  {_lineage_panel(lineage=pg2, title="SUPPORTING · PaliGemma 2", x=626, color="#34d399")}

  <rect x="52" y="486" width="522" height="232" rx="16" fill="#111827" stroke="#334155"/>
  <text x="76" y="522" class="panel-title">Why the interface is confounded</text>
  <text x="76" y="554" class="mono">angular_leaf_spot  {token_counts['angular_leaf_spot']} suffix tokens</text>
  <text x="76" y="580" class="mono">bean_rust          {token_counts['bean_rust']} suffix tokens</text>
  <text x="76" y="606" class="mono">healthy            {token_counts['healthy']} suffix tokens</text>
  <text x="76" y="642" class="body">TRAIN · mean CE over every active suffix token</text>
  <text x="76" y="667" class="body">EVAL · softmax over summed candidate log-likelihood</text>
  <text x="76" y="696" class="small">Long targets weigh more in training; short answers are favored in evaluation.</text>

  <rect x="626" y="486" width="522" height="232" rx="16" fill="#111827" stroke="#334155"/>
  <text x="650" y="522" class="panel-title">PG2 reveals the failure mode by class</text>
  <text x="650" y="550" class="small">REGISTERED sum score</text>
  <text x="650" y="574" class="mono">{html.escape(pg2_sum_classes[0])}</text>
  <text x="650" y="596" class="mono">{html.escape(pg2_sum_classes[1])}</text>
  <text x="650" y="618" class="mono">{html.escape(pg2_sum_classes[2])}</text>
  <text x="650" y="648" class="small">DIAGNOSTIC mean score</text>
  <text x="650" y="672" class="mono">{html.escape(pg2_mean_classes[0])}</text>
  <text x="650" y="694" class="mono">{html.escape(pg2_mean_classes[1])}</text>
  <text x="650" y="716" class="mono">{html.escape(pg2_mean_classes[2])}</text>

  <rect x="52" y="742" width="1096" height="38" rx="10" fill="#3f151c" stroke="#ef4444"/>
  <text x="72" y="767" class="gate">PAID ACTION OFF · SEED 29 BLOCKED · G3 NOT GREEN · NEXT: tiny equal-token interface control</text>
  <text x="52" y="794" fill="#475569" font-family="ui-monospace, monospace" font-size="8">audit sha256 {source_sha}</text>
</svg>'''
    svg += "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return {
        "status": "GREEN_RENDERED",
        "audit_sha256": source_sha,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.audit.resolve(), args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
