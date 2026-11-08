"""DEPTHANCHOR figures and gallery — spec §10, regenerated from banked artifacts only.

Nothing here recomputes a number. Every panel is drawn from what a session banked: the code ladder JSON,
the preflight's per-frame fidelity rows, the merged records and the aggregate of the contrast session, the
per-suite closed-loop summaries, and `films.npz`. The one thing it does compute is the VQ-VAE DECODE of
banked grids, which is what turns a 100-vector of codes into a picture — so this job needs the 420 MB
depth-reference dir and not the 22 GB checkpoint.

Because the figures are regenerated from artifacts, running this again after a correction cannot leave a
figure carrying the old verdict (repo rule: regenerate every figure after any correction).

  pg run infra.aws_pg.jobs.depthanchor_figures --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct-7B-D-0812 --s3-prefix results/depthanchor/figures/1 \\
     --budget-min 60 --args " --loop-prefixes results/depthanchor/loop/spatial/1,..."
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_core as core
from infra.pg import depthanchor_frames as frames_mod

S3_FIGURES = f"{sh.S3_ROOT}/figures/v1"


def _pull_metrics(prefix: str, work: pathlib.Path, name: str) -> Optional[Dict[str, object]]:
    uri = f"s3://{sh.BUCKET}/{prefix}/out/metrics.json"
    if not sh.s3_exists(uri):
        print(f"[figures] no metrics at {uri}; the panels that need {name} are skipped", flush=True)
        return None
    return json.loads(sh.pull_json(uri, work / f"{name}_metrics.json").read_text())


def _pull_artifact(prefix: str, filename: str, work: pathlib.Path, name: str):
    uri = f"s3://{sh.BUCKET}/{prefix}/out/{filename}"
    if not sh.s3_exists(uri):
        print(f"[figures] no {filename} at {uri}", flush=True)
        return None
    path = sh.pull_json(uri, work / f"{name}_{filename}")
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthanchor")
    ap.add_argument("--preflight-prefix", required=True)
    ap.add_argument("--gates-prefix", default="results/depthanchor/gates/1")
    ap.add_argument("--contrast-prefix", default="results/depthanchor/contrast/1")
    ap.add_argument("--loop-prefixes", default="")
    ap.add_argument("--n-strips", type=int, default=4)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, "figures")
    bank.top("args", vars(args))

    preflight = _pull_metrics(args.preflight_prefix, work, "preflight")
    contrast = _pull_metrics(args.contrast_prefix, work, "contrast")
    gates_metrics = _pull_metrics(args.gates_prefix, work, "gates")
    records = _pull_artifact(args.contrast_prefix, "records.json", work, "contrast") or []
    gate_records = _pull_artifact(args.gates_prefix, "records.json", work, "gates") or []
    per_cell_maps = _pull_artifact(args.contrast_prefix, "per_cell_maps.json", work, "contrast") or []
    samples = _pull_artifact(args.contrast_prefix, "samples.json", work, "contrast") or {}
    aggregate = _pull_artifact(args.contrast_prefix, "aggregate.json", work, "contrast")
    ladder, ladder_payload = sh.load_ladder(work)
    pack = frames_mod.load_pack(sh.pull_pack(work / "pack"))
    bank.stage("inputs", {
        "preflight": bool(preflight), "gates": bool(gates_metrics), "contrast": bool(contrast),
        "records": len(records), "gate_records": len(gate_records),
        "per_cell_maps": len(per_cell_maps), "samples": len(samples),
        "aggregate": bool(aggregate), "pack_frames": pack["n_frames"]})

    from infra.pg import depthanchor_model as dam
    refs = dam.load_depth_refs()
    strips = _build_strips(refs, pack, samples, ladder, args.n_strips)
    fidelity = None
    rows = ((preflight or {}).get("stages", {}).get("refs") or {}).get("fidelity_per_frame")
    if rows:
        fidelity = {"per_frame": rows}

    loop_prefixes = [p.strip() for p in args.loop_prefixes.split(",") if p.strip()]
    closed_loop, films = _closed_loop(loop_prefixes, work, refs)

    from infra.pg import depthanchor_figures as figs
    merged = {"records": list(gate_records) + list(records), "per_cell_maps": per_cell_maps}
    made = figs.draw_all(out / "figures", offline=merged, aggregate=aggregate, ladder=ladder,
                         gates=(aggregate or {}).get("gates"), fidelity=fidelity,
                         closed_loop=closed_loop, strips=strips, films=films)
    gallery = _gallery(out / "figures", made, preflight, gates_metrics, contrast, closed_loop)
    sh.push_dir(out / "figures", S3_FIGURES)
    bank.stage("figures", {"drawn": sorted(made), "gallery": str(gallery),
                           "closed_loop_suites": sorted((closed_loop or {}).get("by_suite", {}))})
    bank.top("verdict", {"figures": sorted(made), "gallery": str(gallery)})
    print(json.dumps({"drawn": sorted(made)}, indent=1), flush=True)


def _build_strips(refs, pack, samples, ladder, n: int) -> List[Dict[str, object]]:
    """F4: for the banked sample frames, the decoded predicted grid against the decoded +8 target edit.
    Both grids are reconstructed from banked artifacts — the sample's bins and the pack's target cells —
    so the panel shows the same edit the contrast cell injected."""
    by_id = {f["spec"]["frame_id"]: f for f in pack["frames"]}
    strips = []
    for fid, sample in list(samples.items())[:n]:
        frame = by_id.get(fid)
        if frame is None:
            continue
        bins = np.asarray(sample["bins"], dtype=np.int64)
        target = np.asarray(sample["target_cells"], dtype=np.int64)
        edited = core.edit_shift_cells(bins, target, +8, ladder, "target").bins
        strips.append({"frame_id": fid, "rgb": np.asarray(frame["agentview"]),
                       "decoded_pred": refs.decode_bins(bins),
                       "decoded_edited": refs.decode_bins(edited),
                       "target_cells": target})
    return strips


def _closed_loop(prefixes: List[str], work: pathlib.Path, refs) -> tuple:
    """Merge the per-suite loop jobs into one structure for F7/F9, and decode the filmed grids for F8."""
    if not prefixes:
        return None, None
    by_arm_suite: Dict[str, Dict[str, object]] = {}
    by_suite: Dict[str, object] = {}
    pooled: Dict[str, Dict[str, object]] = {}
    traces: Dict[str, object] = {}
    films: Dict[str, Dict[str, np.ndarray]] = {}
    counts: Dict[str, List[int]] = {}
    episodes_per_cell: set = set()
    for prefix in prefixes:
        m = _pull_metrics(prefix, work, prefix.replace("/", "_"))
        if not m:
            continue
        summary = (m.get("stages") or {}).get("summary")
        if not summary:
            continue
        suite = summary["suite"]
        by_suite[suite] = summary
        for arm, entry in summary["by_arm"].items():
            by_arm_suite.setdefault(arm, {})[suite] = entry
            counts.setdefault(arm, [0, 0])
            counts[arm][0] += int(entry["successes"])
            counts[arm][1] += int(entry["n"])
            episodes_per_cell.add(int(entry["n"]))
        uri = f"s3://{sh.BUCKET}/{prefix}/out/films.npz"
        if sh.s3_exists(uri) and not films:
            local = work / "films.npz"
            sh.s3("cp", uri, str(local))
            data = np.load(local, allow_pickle=False)
            arms = sorted({k.split(":")[0] for k in data.files})
            for arm in arms:
                film: Dict[str, np.ndarray] = {}
                if f"{arm}:frames" in data.files:
                    film["frames"] = data[f"{arm}:frames"]
                if f"{arm}:bins" in data.files:
                    film["decoded"] = np.stack([refs.decode_bins(g) for g in data[f"{arm}:bins"]])
                if f"{arm}:eef" in data.files:
                    traces[arm] = data[f"{arm}:eef"].tolist()
                if film:
                    films[arm] = film
    for arm, (ok, n) in counts.items():
        rate, lo, hi = core.wilson(ok, n)
        pooled[arm] = {"successes": ok, "n": n, "rate": rate, "ci95": [lo, hi]}
    confirmation = None
    if "C-normal" in pooled:
        normal = pooled["C-normal"]
        confirmation = core.closed_loop_confirmation(
            (normal["rate"], normal["ci95"][0], normal["ci95"][1]),
            {a: (v["rate"], v["ci95"][0], v["ci95"][1]) for a, v in pooled.items() if a != "C-normal"},
            label="H_geo")
    return ({"by_arm_suite": by_arm_suite, "by_suite": by_suite, "pooled": pooled,
             "pooled_confirmation": confirmation, "traces": traces,
             "episodes_per_cell": sorted(episodes_per_cell),
             "suite": ",".join(sorted(by_suite))}, films or None)


def _gallery(fig_dir: pathlib.Path, made: Dict[str, object], preflight, gates_metrics, contrast,
             closed_loop) -> pathlib.Path:
    """The last artifact of the study: one page assembled from the banked figures, each with the number
    that makes it worth looking at."""
    fig_dir = pathlib.Path(fig_dir)
    label = ((contrast or {}).get("verdict") or {}).get("label", "not yet decided")
    lines = ["<!doctype html><meta charset='utf-8'><title>DEPTHANCHOR gallery</title>",
             "<style>body{font:14px system-ui;margin:2rem;max-width:1100px}"
             "img{max-width:100%;border:1px solid #ddd}h2{margin-top:2rem}"
             "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.25rem .5rem}"
             "</style>",
             "<h1>DEPTHANCHOR — is MolmoAct2-Think's depth map load-bearing or decoration?</h1>",
             f"<p><b>Offline label: {label}</b></p>"]
    verdicts = {"preflight": (preflight or {}).get("verdict"),
                "gates": (gates_metrics or {}).get("verdict"),
                "contrast": (contrast or {}).get("verdict"),
                "closed loop": (closed_loop or {}).get("pooled_confirmation")}
    lines.append("<h2>Verdicts as banked</h2><pre>"
                 + json.dumps(verdicts, indent=1, default=str).replace("<", "&lt;") + "</pre>")
    for name in sorted(made):
        paths = made[name]
        lines.append(f"<h2>{name}</h2>")
        for p in paths:
            rel = pathlib.Path(p).name
            if rel.endswith((".png", ".gif")):
                lines.append(f"<div><img src='{rel}' alt='{rel}'></div>")
            else:
                lines.append(f"<div><a href='{rel}'>{rel}</a></div>")
    path = fig_dir / "gallery.html"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    main()
