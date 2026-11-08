"""DEPTHOPEN offline gate — every instrument that does not need the model, on planted data.

House rule: the full pipeline minus the model forward runs green before any GPU. What this covers so far:

  G-subset-1  a planted 6-file dataset is rebuilt, and the rebuilt meta locates every episode's rows
  G-subset-2  the rebuild does not touch image bytes or row order (hash equality against the source)
  G-subset-3  the source's own off-by-one `data/file_index` does NOT reach the subset
  G-subset-4  the verifier RAISES on a planted defect: meta pointing at the wrong file, a reordered
              episode, and an episode listed but absent. An instrument that cannot fail is not a gate.
  G-subset-5  stratification: whole files are picked per suite, straddling files are never picked, and
              the episode counts land on the target
  G-enum      the file enumerator raises on a gap instead of scanning a truncated list, which is the
              defect that cost POSTCHECK a claim
  G-rules     every frozen rule in SPEC §6 fires in BOTH directions on planted numbers, including the
              "no room" branch, the H_need branch, the H_harm branch and the noise-floor branch. A rule
              with no range cannot fail, so each one is shown failing
  G-report    the report generator is itself an instrument: planted `metrics.json` files go through it
              and the rows, the rule verdicts and the prediction grid come out with the planted values

Run locally: .pgvenv/bin/python -m infra.aws_pg.jobs.depthopen_dryrun
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
from typing import Dict, List

import numpy as np

from infra.pg import depthopen_subset as ds

IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")
STAT_KEYS = IMAGE_KEYS + ("observation.state", "action", "timestamp", "frame_index",
                          "episode_index", "index", "task_index")


def _episode_stats(key: str, n: int) -> Dict[str, object]:
    if key in IMAGE_KEYS:
        return {f"stats/{key}/min": [[[0.0]], [[0.0]], [[0.0]]],
                f"stats/{key}/max": [[[1.0]], [[1.0]], [[1.0]]],
                f"stats/{key}/mean": [[[0.3]], [[0.3]], [[0.3]]],
                f"stats/{key}/std": [[[0.1]], [[0.1]], [[0.1]]],
                f"stats/{key}/count": [n]}
    width = {"observation.state": 8, "action": 7}.get(key, 1)
    return {f"stats/{key}/min": [0.0] * width, f"stats/{key}/max": [1.0] * width,
            f"stats/{key}/mean": [0.5] * width, f"stats/{key}/std": [0.1] * width,
            f"stats/{key}/count": [n]}


def plant_dataset(root: pathlib.Path, *, episodes_per_file: int = 3, n_files: int = 8,
                  rows_per_episode: int = 5, break_file_index: bool = True) -> Dict[str, object]:
    """A dataset with the release's shape: images as bytes in the parquet, 4 suites of tasks, and — when
    asked — the release's own off-by-one in `meta/episodes` `data/file_index`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(7)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    suite_map = ds.load_suite_map()
    by_suite: Dict[str, List[str]] = {}
    for lang, suite in suite_map.items():
        by_suite.setdefault(suite, []).append(lang)
    suites = sorted(by_suite)
    tasks_rows = []
    task_suite = {}
    ti = 0
    for suite in suites:
        for lang in sorted(by_suite[suite])[:2]:                    # two tasks per suite is enough
            tasks_rows.append({"task_index": ti, "task": lang})
            task_suite[ti] = suite
            ti += 1
    files_per_suite = n_files // len(suites)

    ep_rows, index = [], 0
    ep_id = 0
    for f in range(n_files):
        suite = suites[min(len(suites) - 1, f // files_per_suite)]
        task_index = next(t for t, s in task_suite.items() if s == suite)
        cols: Dict[str, list] = {k: [] for k in
                                 ("observation.images.image", "observation.images.wrist_image",
                                  "observation.state", "action", "timestamp", "frame_index",
                                  "episode_index", "index", "task_index")}
        first_index = index
        for k in range(episodes_per_file):
            for r in range(rows_per_episode):
                cols["observation.images.image"].append(bytes([ep_id % 251, r, 7]) * 4)
                cols["observation.images.wrist_image"].append(bytes([ep_id % 251, r, 9]) * 4)
                cols["observation.state"].append(rng.normal(size=8).astype(np.float32).tolist())
                cols["action"].append(rng.normal(size=7).astype(np.float32).tolist())
                cols["timestamp"].append(float(r) / 10.0)
                cols["frame_index"].append(r)
                cols["episode_index"].append(ep_id)
                cols["index"].append(index)
                cols["task_index"].append(task_index)
                index += 1
            ep_rows.append({
                "episode_index": ep_id,
                "data/chunk_index": 0,
                # the release's defect: the listed file is one LESS than the file that holds the rows
                "data/file_index": max(0, f - 1) if break_file_index else f,
                "dataset_from_index": first_index + k * rows_per_episode,
                "dataset_to_index": first_index + (k + 1) * rows_per_episode,
                "tasks": [tasks_rows[task_index]["task"]],
                "length": rows_per_episode,
                **{kk: vv for key in STAT_KEYS for kk, vv in _episode_stats(key, rows_per_episode).items()},
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            })
            ep_id += 1
        pq.write_table(pa.Table.from_pydict(cols), root / ds.DATA_TEMPLATE.format(i=f))

    pq.write_table(pa.Table.from_pylist(ep_rows), root / ds.META_EPISODES)
    pq.write_table(pa.Table.from_pylist(tasks_rows), root / ds.META_TASKS)
    info = {"codebase_version": "v3.0", "fps": 10, "features": {},
            "total_episodes": ep_id, "total_frames": index, "total_tasks": len(tasks_rows),
            "chunks_size": 1000, "data_files_size_in_mb": 100,
            "data_path": ds.DATA_TEMPLATE.replace("{i:03d}", "{file_index:03d}"),
            "video_path": None, "robot_type": "panda", "splits": {"train": f"0:{ep_id}"}}
    (root / ds.META_INFO).write_text(json.dumps(info, indent=4))
    return {"info": info, "episodes": ep_rows, "tasks": tasks_rows, "task_suite": task_suite,
            "n_files": n_files, "n_episodes": ep_id, "n_frames": index}


def _scan(root: pathlib.Path, n_files: int):
    from infra.pg import postcheck_dataset as pcd
    opener = lambda i: (lambda: open(root / ds.DATA_TEMPLATE.format(i=i), "rb"))
    return pcd.scan_data_files(opener, n_files)


def gate_subset(out: pathlib.Path) -> Dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = out / "src"
    planted = plant_dataset(src)
    scan = _scan(src, planted["n_files"])
    episodes_in_file = ds.episodes_by_file(scan)
    episode_task: Dict[int, int] = {}
    tasks = {int(r["task_index"]): r["task"] for r in planted["tasks"]}
    task_suite = ds.suite_by_task_index(tasks, ds.load_suite_map())
    for f in range(planted["n_files"]):
        t = pq.read_table(src / ds.DATA_TEMPLATE.format(i=f), columns=["episode_index", "task_index"])
        for e, ti in zip(t.column("episode_index").to_pylist(), t.column("task_index").to_pylist()):
            episode_task[int(e)] = int(ti)

    episode_suite = ds.episode_suites(planted["episodes"], ds.load_suite_map())
    f_suite = ds.file_suites(episodes_in_file, episode_suite)
    if any(v is None for v in f_suite.values()):
        raise AssertionError("planted files should each sit in one suite")
    selection = ds.select_files(f_suite, episodes_in_file, per_suite_episodes=6, seed=20260909)

    # G-subset-5: whole files per suite, target met, nothing borrowed across suites
    for suite, entry in selection["per_suite"].items():
        if entry["n_episodes"] < 6:
            raise AssertionError(f"{suite}: {entry['n_episodes']} episodes, wanted 6")
        for f in entry["files"]:
            if f_suite[f] != suite:
                raise AssertionError(f"file {f} is {f_suite[f]}, selected for {suite}")

    # G-subset-1/2/3: build, then verify against meta and against the source bytes
    dst = out / "subset"
    built = ds.build_subset(src, dst, files=selection["files"], info=planted["info"],
                            episodes_meta=planted["episodes"], tasks_rows=planted["tasks"])
    verified = ds.verify_subset(dst, src_root=src, mapping=built["mapping"], n_byte_checks=4)

    # G-subset-3 explicitly: the source's listed file was wrong for most episodes; the subset's is right
    src_ep = {int(r["episode_index"]): int(r["data/file_index"]) for r in planted["episodes"]}
    actual_file = {int(e): int(v["files"][0]) for e, v in scan["by_episode"].items()}
    src_wrong = sum(1 for e, listed in src_ep.items() if listed != actual_file[e])
    dst_ep = pq.read_table(dst / ds.META_EPISODES).to_pylist()
    dst_wrong = 0
    for row in dst_ep:
        m = next(m for m in built["mapping"] if m["subset_episode"] == int(row["episode_index"]))
        if int(row["data/file_index"]) != int(m["subset_file"]):
            dst_wrong += 1
    if src_wrong == 0:
        raise AssertionError("the planted source was supposed to carry the off-by-one")
    if dst_wrong != 0:
        raise AssertionError(f"{dst_wrong} subset episodes still mislocated")

    # stats: quantiles present for the two keys training normalizes with
    stats = json.loads((dst / ds.META_STATS).read_text())
    for key in ds.QUANTILE_KEYS:
        for q in ds.QUANTILES:
            if q not in stats[key]:
                raise AssertionError(f"{key} stats missing {q}")

    # G-subset-4: three planted defects, each must RAISE
    fails = {}
    for name, mutate in (("wrong_file_index", _mutate_meta_file_index),
                         ("reordered_rows", _mutate_row_order),
                         ("episode_absent", _mutate_drop_episode)):
        broken = out / f"broken_{name}"
        if broken.exists():
            shutil.rmtree(broken)
        shutil.copytree(dst, broken)
        mutate(broken)
        try:
            ds.verify_subset(broken)
            fails[name] = "DID NOT RAISE"
        except RuntimeError as exc:
            fails[name] = f"raised: {str(exc)[:70]}"
    if any(v == "DID NOT RAISE" for v in fails.values()):
        raise AssertionError(f"the verifier missed a planted defect: {fails}")

    # G-enum: a gap in the listing raises rather than scanning a truncated list
    class _Sib:
        def __init__(self, name):
            self.rfilename = name

    class _Api:
        def __init__(self, names):
            self._names = names

        def dataset_info(self, repo, revision=None):
            return type("I", (), {"siblings": [_Sib(n) for n in self._names]})()

    complete = [ds.DATA_TEMPLATE.format(i=i) for i in range(5)]
    got_ok = ds.enumerate_data_files("x/y", "sha", api=_Api(complete + ["meta/info.json"]))
    enum_raised = None
    try:
        ds.enumerate_data_files("x/y", "sha", api=_Api(complete[:3] + [complete[4]]))
    except RuntimeError as exc:
        enum_raised = str(exc)[:80]
    if enum_raised is None:
        raise AssertionError("the enumerator accepted a gapped listing")

    return {"planted": {k: planted[k] for k in ("n_files", "n_episodes", "n_frames")},
            "selection": {k: selection[k] for k in ("n_files", "n_episodes", "seed")},
            "per_suite": {s: {"n_files": v["n_files"], "n_episodes": v["n_episodes"]}
                          for s, v in selection["per_suite"].items()},
            "built": {k: built[k] for k in ("n_files", "n_episodes", "n_frames")},
            "verified": verified,
            "source_mislocated_episodes": src_wrong,
            "subset_mislocated_episodes": dst_wrong,
            "planted_defects": fails,
            "enumerator_ok_len": len(got_ok),
            "enumerator_raised_on_gap": enum_raised}


def _mutate_meta_file_index(root: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(root / ds.META_EPISODES).to_pylist()
    t[0]["data/file_index"] = int(t[0]["data/file_index"]) + 1
    pq.write_table(pa.Table.from_pylist(t), root / ds.META_EPISODES)


def _mutate_row_order(root: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = root / ds.DATA_TEMPLATE.format(i=0)
    t = pq.read_table(p)
    idx = list(range(t.num_rows))
    idx[0], idx[-1] = idx[-1], idx[0]
    pq.write_table(t.take(pa.array(idx)), p)


def _mutate_drop_episode(root: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = pq.read_table(root / ds.META_EPISODES).to_pylist()
    extra = dict(rows[-1])
    extra["episode_index"] = int(rows[-1]["episode_index"]) + 1
    pq.write_table(pa.Table.from_pylist(rows + [extra]), root / ds.META_EPISODES)


def gate_rules() -> Dict[str, object]:
    """Every §6 rule, on planted numbers, in both directions."""
    from infra.pg import depthopen_rules as rules

    # rule 1: reads / does not read / no room, and a cancelled readout
    reads_case = rules.rule1_reads(shuffle_over_ceiling=0.40, shuffle_ci95=[0.30, 0.50], m_arm=0.20,
                                   removal_over_seed_floor=9.0, arm="planted-reads")
    quiet_case = rules.rule1_reads(shuffle_over_ceiling=0.002, shuffle_ci95=[0.001, 0.004], m_arm=0.19,
                                   removal_over_seed_floor=8.0, arm="planted-quiet")
    no_room = rules.rule1_reads(shuffle_over_ceiling=0.40, shuffle_ci95=[0.30, 0.50], m_arm=0.20,
                                removal_over_seed_floor=1.4, arm="planted-no-room")
    touching = rules.rule1_reads(shuffle_over_ceiling=0.25, shuffle_ci95=[0.15, 0.35], m_arm=0.20,
                                 removal_over_seed_floor=9.0, arm="planted-interval-touches")
    cancelled = rules.rule1_reads(shuffle_over_ceiling=None, shuffle_ci95=None, m_arm=0.2,
                                  removal_over_seed_floor=9.0, arm="planted-missing")
    if reads_case["verdict"] != "reads":
        raise AssertionError(f"rule 1 missed a reading arm: {reads_case}")
    if quiet_case["verdict"] != "does not read":
        raise AssertionError(f"rule 1 called an inert arm reading: {quiet_case}")
    if no_room["verdict"] != "no room":
        raise AssertionError(f"rule 1 skipped the no-room branch: {no_room}")
    if touching["verdict"] != "does not read":
        raise AssertionError(f"rule 1 accepted an interval that spans the margin: {touching}")
    if cancelled["verdict"] != "CANCELLED":
        raise AssertionError(f"rule 1 scored a readout with a missing number: {cancelled}")

    # rule 2: H_open, H_need, and A0 also reading
    h_open = rules.rule2_hypothesis({"A0": "does not read", "A3": "reads", "A4": "reads"})
    h_need = rules.rule2_hypothesis({"A0": "does not read", "A3": "does not read",
                                     "A4": "does not read"})
    a0_reads = rules.rule2_hypothesis({"A0": "reads", "A3": "reads", "A4": "no room"})
    all_no_room = rules.rule2_hypothesis({"A0": "no room", "A3": "no room", "A4": "no room"})
    if all_no_room["verdict"] != "undecidable: no arm has room":
        raise AssertionError(f"rule 2 read every arm's no-room as H_need: {all_no_room}")
    mixed = rules.rule2_hypothesis({"A0": "no room", "A3": "does not read"})
    if mixed["verdict"] != "H_need":
        raise AssertionError(f"rule 2 should read H_need from the arm WITH room: {mixed}")
    if (h_open["verdict"], h_need["verdict"]) != ("H_open", "H_need"):
        raise AssertionError(f"rule 2 branches: {h_open['verdict']}, {h_need['verdict']}")
    if a0_reads["verdict"] != "H_open_with_A0_reading":
        raise AssertionError(f"rule 2 missed the A0-reads branch: {a0_reads}")

    # rule 3: better, harm, no difference, and inside the noise floor
    better = rules.rule3_better(arm="A3", camera_arm={"k": 90, "n": 200},
                                camera_control={"k": 50, "n": 200},
                                libero_arm={"k": 180, "n": 200}, libero_control={"k": 184, "n": 200},
                                banked_flip_rate=0.05)
    harm = rules.rule3_better(arm="A3", camera_arm={"k": 40, "n": 200},
                              camera_control={"k": 90, "n": 200},
                              libero_arm={"k": 180, "n": 200}, libero_control={"k": 184, "n": 200},
                              banked_flip_rate=0.05)
    same = rules.rule3_better(arm="A3", camera_arm={"k": 52, "n": 200},
                              camera_control={"k": 50, "n": 200},
                              libero_arm={"k": 180, "n": 200}, libero_control={"k": 184, "n": 200},
                              banked_flip_rate=0.05)
    noisy = rules.rule3_better(arm="A3", camera_arm={"k": 90, "n": 200},
                               camera_control={"k": 50, "n": 200},
                               libero_arm={"k": 180, "n": 200}, libero_control={"k": 184, "n": 200},
                               banked_flip_rate=0.90)
    band_fail = rules.rule3_better(arm="A3", camera_arm={"k": 90, "n": 200},
                                   camera_control={"k": 50, "n": 200},
                                   libero_arm={"k": 100, "n": 200}, libero_control={"k": 184, "n": 200},
                                   banked_flip_rate=0.05)
    if better["verdict"] != "H_better":
        raise AssertionError(f"rule 3 missed H_better: {better}")
    if harm["verdict"] != "H_harm":
        raise AssertionError(f"rule 3 missed H_harm: {harm}")
    if not same["verdict"].startswith("no difference"):
        raise AssertionError(f"rule 3 read a difference that is not there: {same}")
    if noisy["verdict"] != "inside the banked run-to-run noise":
        raise AssertionError(f"rule 3 ignored the noise floor: {noisy}")
    if band_fail["verdict"] != "camera difference excludes zero but the LIBERO band clause fails":
        raise AssertionError(f"rule 3 ignored the LIBERO band: {band_fail}")

    flips = rules.flip_rate([True, False, True, True], [True, True, True, False])
    if flips["n_flipped"] != 2 or abs(flips["rate_difference_abs"]) > 1e-12:
        raise AssertionError(f"flip rate wrong: {flips}")

    return {"rule1": {k: v["verdict"] for k, v in
                      {"reads": reads_case, "quiet": quiet_case, "no_room": no_room,
                       "interval_touches": touching, "missing": cancelled}.items()},
            "rule2": {"h_open": h_open["verdict"], "h_need": h_need["verdict"],
                      "a0_reads": a0_reads["verdict"], "all_no_room": all_no_room["verdict"],
                      "mixed_room": mixed["verdict"]},
            "rule3": {"better": better["verdict"], "harm": harm["verdict"], "same": same["verdict"],
                      "noisy": noisy["verdict"], "band_fail": band_fail["verdict"],
                      "better_interval": [round(better["camera_difference"]["lo"], 4),
                                          round(better["camera_difference"]["hi"], 4)]},
            "flip_rate": flips}


def _plant_metrics(root: pathlib.Path, cell: str, payload: Dict[str, object]) -> None:
    d = root / cell / "1" / "out"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload, indent=1))


def gate_report(out: pathlib.Path) -> Dict[str, object]:
    """Planted numbers through the real report generator, with A3 reading and A0 not."""
    import subprocess
    results = out / "planted_results"
    _plant_metrics(results, "data", {"cell": "data", "verdict": {
        "n_data_files_enumerated": 379,
        "p4_recheck": {"episodes_listed": 1693, "episodes_present": 1693, "rows_present": 273465,
                       "file_index_mismatches": 373, "file_index_offset_histogram": {"1": 373}},
        "subset": {"n_files": 91, "n_episodes": 395, "n_frames": 65000,
                   "per_suite": {"libero_10": 103}},
        "companions": {"rows": 65000, "frames_per_s": 90.0}}})
    _plant_metrics(results, "smoke", {"cell": "smoke", "verdict": {
        "g_smoke": {"peak_reserved_gb": 41.2, "cap_gb": 46.0, "seconds_per_step": 2.1,
                    "loss_decreasing": True, "merged_checkpoint_written": True,
                    "reloads_and_acts": True, "passes": True},
        "g_grad": {"n_gate_trainable": 72, "expected_trainable": 72,
                   "optimizer_groups_holding_the_gate": ["action_expert"],
                   "grad_norm_min": 1e-5, "grad_norm_max": 3e-3, "passes": True}}})

    def arm_payload(arm, *, shuffle, ci, m, removal, sig, share):
        return {"cell": f"depthopen_arm:{arm}", "stages": {
            "train": {"steps": 2000, "seconds_per_step": 2.1, "loss_decreasing": True,
                      "peak_reserved_gb": 41.2, "n_gate_trainable": 72,
                      "gate_grad_norm_max": 3e-3, "device_batch_size": 2, "global_batch_size": 2},
            "r1": {"present": True, "n_layers": 36, "bias_min": -0.1, "bias_max": 0.2,
                   "sigmoid_min": sig - 0.02, "sigmoid_median": sig, "sigmoid_max": sig + 0.02,
                   "max_abs_bias_deviation": 0.2, "verdict": "moved"},
            "r2": {"m": m, "remove_mode": "gate0", "n_frames_usable": 40, "g0_max": 1e-6,
                   "realized_gate_on_frames": {"n": 40, "min": sig - 0.03, "median": sig,
                                               "max": sig + 0.03},
                   "shuffle_over_ceiling": {"point": shuffle, "ci95": ci},
                   "ratio_to_ceiling": {"const": {"point": shuffle * 0.8},
                                        "remove": {"point": shuffle * 1.5}},
                   "ratio_to_seed_floor": {"remove": {"point": removal}},
                   "reads_the_prefix": ci[0] > m},
            "r3": {"measured": True, "n_layers": 36,
                   "depth_mass": {"min": share / 3, "median": share, "max": share * 2},
                   "uniform_share": 0.17, "ratio_to_uniform": share / 0.17,
                   "probabilities_normalized": True}}}

    _plant_metrics(results / "arm", "A0", arm_payload("A0", shuffle=0.002, ci=[0.001, 0.004], m=0.19,
                                                     removal=8.0, sig=0.018, share=0.096))
    _plant_metrics(results / "arm", "A3", arm_payload("A3", shuffle=0.42, ci=[0.33, 0.51], m=0.20,
                                                      removal=11.0, sig=0.51, share=0.24))
    for arm, k in (("A0", 184), ("A3", 180)):
        _plant_metrics(results / "r4", arm, {"cell": f"depthopen_r4:{arm}", "verdict": {
            "arm": arm, "pooled": rules_wilson(k, 200),
            "by_suite": {s: rules_wilson(k // 4, 50) for s in
                         ("libero_spatial", "libero_object", "libero_goal", "libero_10")},
            "g_known": {"k": 9, "n": 10, "passes": True}, "n_rollouts": 200}})
    for model, dim, k in (("released", "camera", 60), ("released", "layout", 120),
                          ("A0", "camera", 50), ("A3", "camera", 90),
                          ("A0", "layout", 100), ("A3", "layout", 104)):
        _plant_metrics(results / "r5" / model, dim, {"cell": f"r5:{model}:{dim}", "verdict": {
            "model": model, "dimension": dim, "overall": rules_wilson(k, 200),
            "by_level": {str(l): rules_wilson(k // 5, 40) for l in (1, 2, 3, 4, 5)}}})
    # a second released camera run, so the flip rate exists
    d = results / "r5" / "released" / "camera" / "2" / "out"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"cell": "r5:released:camera", "verdict": {
        "model": "released", "dimension": "camera", "overall": rules_wilson(56, 200),
        "by_level": {str(l): rules_wilson(11, 40) for l in (1, 2, 3, 4, 5)}}}, indent=1))

    proc = subprocess.run([sys.executable, "scripts/depthopen_report.py", "--results", str(results)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"the report generator failed on planted numbers:\n{proc.stderr[-2000:]}")
    text = (results / "RESULTS.md").read_text()
    verdict = json.loads((results / "verdict.json").read_text())
    decided = verdict["decided"]
    checks = {
        "A3 reads": decided["reads"]["A3"]["verdict"] == "reads",
        "A0 does not read": decided["reads"]["A0"]["verdict"] == "does not read",
        "rule2 H_open": decided["rule2"]["verdict"] == "H_open",
        "rule3 present": "A3" in (decided.get("rule3") or {}),
        "flip rate banked": decided.get("banked_camera_flip_rate") is not None,
        "grid filled": len(decided["prediction_grid"]) == 7,
        "379 in the document": "379" in text,
        "373 in the document": "373" in text,
        "G-smoke row": "G-smoke" in text,
        "LIBERO-plus row": "LIBERO-plus" in text,
    }
    if not all(checks.values()):
        raise AssertionError(f"the report dropped something: {checks}")
    return {"checks": checks, "rule3_A3": decided["rule3"]["A3"]["verdict"],
            "banked_flip_rate": decided["banked_camera_flip_rate"],
            "results_lines": len(text.splitlines())}


def rules_wilson(k: int, n: int) -> Dict[str, float]:
    from infra.pg import depthopen_rules as rules
    return rules.wilson(k, n)


def main() -> None:
    out = pathlib.Path(tempfile.mkdtemp(prefix="depthopen_dryrun_"))
    report = {"out": str(out), "gates": {}}
    report["gates"]["rules"] = gate_rules()
    report["gates"]["report"] = gate_report(out)
    report["gates"]["subset"] = gate_subset(out)
    print(json.dumps(report, indent=1, default=str))
    print("\nOFFLINE GATE GREEN")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
