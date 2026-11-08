"""DEPTHOPEN §4 — build the training subset as a fresh, self-consistent LeRobot v3 dataset.

WHY A REBUILD RATHER THAN A POINTER. `allenai/MolmoAct2-LIBERO-Dataset` carries 273,465 frames in 379
parquet files with the images inside the parquet (there are no video files), and its `meta/episodes`
`data/file_index` is wrong for 372 episodes — POSTCHECK measured that, and this study must not train on
rows located by a field that is off by one. Rebuilding from whole source files gives a dataset whose meta
is derived from the files' own `episode_index` / `index` columns, so the loader and the depth-annotation
generator agree with each other by construction, and the exact bytes trained on are banked.

WHY WHOLE FILES. Depth companions are generated per data file (the generator copies files and adds three
columns), so selecting whole files keeps the companion dataset aligned with the training dataset with no
re-chunking. 4-5 episodes live in each file, so whole-file selection at ~22 files per suite lands on the
SPEC's 400 episodes, and only a quarter of the release has to be downloaded.

WHY THE FILE LIST IS ENUMERATED, NEVER COUNTED. DEPTHANCHOR passed a hand-typed `n_files=378` and
POSTCHECK inherited it; `file-378.parquet` exists (47.3 MB, the 382 rows of episodes 1690-1692) and
neither study ever opened it, which cost POSTCHECK a false claim about three absent episodes. Here the
list comes from the Hub listing and is checked against `file-000 … file-(n-1)`; a gap raises.
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Dict, List, Optional, Sequence

DATA_TEMPLATE = "data/chunk-000/file-{i:03d}.parquet"
META_EPISODES = "meta/episodes/chunk-000/file-000.parquet"
META_INFO = "meta/info.json"
META_TASKS = "meta/tasks.parquet"
META_STATS = "meta/stats.json"
SUITES_JSON = pathlib.Path(__file__).with_name("depthopen_suites.json")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
QUANTILE_KEYS = ("action", "observation.state")


def _stable_key(text: str) -> int:
    """A process-independent integer for a label, for seeding. `hash()` is salted per process."""
    import hashlib
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ enumeration
def enumerate_data_files(repo: str, revision: str, api=None) -> List[str]:
    """Every data parquet path in the repository at this revision, in file order.

    Fails loud if the names are not exactly `file-000 … file-(n-1)`: a hand-typed or inferred count is
    the defect this function exists to remove.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.dataset_info(repo, revision=revision)
    names = sorted(s.rfilename for s in (info.siblings or [])
                   if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet"))
    expected = [DATA_TEMPLATE.format(i=i) for i in range(len(names))]
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise RuntimeError(f"data file names are not contiguous: {len(names)} files, "
                           f"missing={missing[:5]}, unexpected={extra[:5]}")
    return names


# ------------------------------------------------------------------ suites
def normalize_language(text: str) -> str:
    """DEPTHANCHOR's normalizer, so the keys of `depthopen_suites.json` match."""
    return " ".join(str(text).replace("_", " ").lower().split()).strip(" .")


def load_suite_map(path: pathlib.Path = SUITES_JSON) -> Dict[str, str]:
    payload = json.loads(pathlib.Path(path).read_text())
    return {k: v.split("/")[0] for k, v in payload["language_to_suite_task"].items()}


def suite_by_task_index(tasks: Dict[int, str], suite_map: Dict[str, str]) -> Dict[int, str]:
    """task_index -> suite, by matching the dataset's own task strings against the LIBERO task lists.

    Every task must match: an unmatched string means the suite map and the dataset disagree, which would
    silently unbalance the strata.
    """
    out: Dict[int, str] = {}
    unmatched = []
    for ti, text in tasks.items():
        key = normalize_language(text)
        if key not in suite_map:
            unmatched.append((ti, text))
            continue
        out[int(ti)] = suite_map[key]
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} dataset tasks have no suite: {unmatched[:3]}")
    return out


# ------------------------------------------------------------------ selection
def episode_suites(meta_rows: Sequence[Dict[str, object]], suite_map: Dict[str, str]
                   ) -> Dict[int, str]:
    """episode -> suite, from `meta/episodes`' own `tasks` language string.

    Deliberately not from `task_index` joined through the scan: the scan reads index columns only, and
    the language string is the field POSTCHECK's P5 already showed to be the reliable join key here.
    """
    out: Dict[int, str] = {}
    unmatched = []
    for row in meta_rows:
        tasks = list(row.get("tasks") or [])
        key = normalize_language(tasks[0]) if tasks else ""
        if key not in suite_map:
            unmatched.append({"episode_index": int(row["episode_index"]), "tasks": tasks})
            continue
        out[int(row["episode_index"])] = suite_map[key]
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} episodes have no suite: {unmatched[:3]}")
    return out


def file_suites(episodes_in_file: Dict[int, List[int]], episode_suite: Dict[int, str]
                ) -> Dict[int, Optional[str]]:
    """A file's suite, or None when its episodes straddle two suites (those files are not selected, so a
    stratum never borrows episodes from its neighbour)."""
    out: Dict[int, Optional[str]] = {}
    for i, episodes in episodes_in_file.items():
        suites = {episode_suite[e] for e in episodes}
        out[int(i)] = suites.pop() if len(suites) == 1 else None
    return out


def select_files(f_suite: Dict[int, Optional[str]], episodes_in_file: Dict[int, List[int]], *,
                 per_suite_episodes: int, seed: int) -> Dict[str, object]:
    """Whole files, systematically spread across each suite's own file range, until the episode target.

    Systematic (every k-th file from a seeded offset) rather than uniform-random, so the sample covers
    each suite's whole range instead of clumping; the seed only picks the phase.
    """
    import numpy as np
    chosen: List[int] = []
    per_suite: Dict[str, Dict[str, object]] = {}
    for suite in sorted({s for s in f_suite.values() if s}):
        files = [i for i in sorted(f_suite) if f_suite[i] == suite]
        counts = [len(episodes_in_file[i]) for i in files]
        mean_per_file = float(np.mean(counts)) if counts else 0.0
        want_files = max(1, int(round(per_suite_episodes / max(mean_per_file, 1e-9))))
        want_files = min(want_files, len(files))
        step = len(files) / want_files
        # a STABLE per-suite offset: Python's str hash is salted per process, so `hash(suite)` made this
        # "seeded" draw pick a different 91 files on every run — measured, two runs of the same job
        phase = float(np.random.default_rng([seed, _stable_key(suite)]).random())
        picked = sorted({files[min(len(files) - 1, int((k + phase) * step))] for k in range(want_files)})
        # a collision from rounding can shrink the pick; top up with the nearest unused files
        for i in files:
            if len(picked) >= want_files:
                break
            if i not in picked:
                picked = sorted(set(picked) | {i})
        n_ep = sum(len(episodes_in_file[i]) for i in picked)
        per_suite[suite] = {"n_files_available": len(files), "n_files": len(picked),
                            "n_episodes": n_ep, "mean_episodes_per_file": round(mean_per_file, 3),
                            "files": picked}
        chosen.extend(picked)
    return {"seed": seed, "per_suite_episodes_target": per_suite_episodes,
            "files": sorted(chosen), "per_suite": per_suite,
            "n_files": len(chosen),
            "n_episodes": sum(v["n_episodes"] for v in per_suite.values())}


# ------------------------------------------------------------------ build
def _quantiles(values, qs=QUANTILES) -> Dict[str, List[float]]:
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    out = {}
    for name, q in qs.items():
        out[name] = np.quantile(a, q, axis=0).tolist()
    return out


def _agg_image_stats(entries: Sequence[Dict[str, object]], key: str) -> Dict[str, object]:
    """Count-weighted aggregate of the per-episode image stats the source meta already carries."""
    import numpy as np
    tot = 0.0
    mean = None
    m2 = None
    lo = hi = None
    for e in entries:
        n = float(np.asarray(e[f"stats/{key}/count"]).ravel()[0])
        mu = np.asarray(e[f"stats/{key}/mean"], dtype=np.float64)
        sd = np.asarray(e[f"stats/{key}/std"], dtype=np.float64)
        mn = np.asarray(e[f"stats/{key}/min"], dtype=np.float64)
        mx = np.asarray(e[f"stats/{key}/max"], dtype=np.float64)
        if mean is None:
            mean, m2, lo, hi, tot = mu * n, (sd ** 2 + mu ** 2) * n, mn, mx, n
        else:
            mean = mean + mu * n
            m2 = m2 + (sd ** 2 + mu ** 2) * n
            lo = np.minimum(lo, mn)
            hi = np.maximum(hi, mx)
            tot += n
    mean = mean / tot
    var = np.maximum(m2 / tot - mean ** 2, 0.0)
    return {"min": lo.tolist(), "max": hi.tolist(), "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(), "count": [int(tot)]}


def build_subset(src_root: pathlib.Path, dst_root: pathlib.Path, *, files: Sequence[int],
                 info: Dict[str, object], episodes_meta: Sequence[Dict[str, object]],
                 tasks_rows: Sequence[Dict[str, object]],
                 on_progress: Optional[Callable[[Dict[str, object]], None]] = None
                 ) -> Dict[str, object]:
    """Copy the selected files, renumber `episode_index` and `index` contiguously, and write meta.

    Row order inside a file is preserved exactly; only the two index columns change, so a frame's images,
    state and action are the released bytes. `meta/episodes` is written from what the copied files
    actually contain, and the global stats' quantiles are computed over the subset's own rows because
    training normalizes with `--norm_mode q01_q99` read from this file.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_root, dst_root = pathlib.Path(src_root), pathlib.Path(dst_root)
    (dst_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    meta_by_episode = {int(r["episode_index"]): r for r in episodes_meta}

    mapping: List[Dict[str, object]] = []
    ep_rows: List[Dict[str, object]] = []
    next_ep = 0
    next_index = 0
    action_all: List[np.ndarray] = []
    state_all: List[np.ndarray] = []
    scalar_cols: Dict[str, List[float]] = {}

    for out_i, src_i in enumerate(sorted(files)):
        table = pq.read_table(src_root / DATA_TEMPLATE.format(i=src_i))
        src_ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        order = list(dict.fromkeys(src_ep.tolist()))          # first-seen order, rows stay put
        remap = {e: next_ep + k for k, e in enumerate(order)}
        new_ep = np.asarray([remap[int(e)] for e in src_ep], dtype=np.int64)
        new_index = np.arange(next_index, next_index + table.num_rows, dtype=np.int64)

        out = table.set_column(table.schema.get_field_index("episode_index"),
                               "episode_index", pa.array(new_ep))
        out = out.set_column(out.schema.get_field_index("index"), "index", pa.array(new_index))
        pq.write_table(out, dst_root / DATA_TEMPLATE.format(i=out_i))

        for e in order:
            rows = np.flatnonzero(src_ep == e)
            src_meta = meta_by_episode[int(e)]
            ep_rows.append({
                "episode_index": remap[int(e)],
                "data/chunk_index": 0,
                "data/file_index": out_i,
                "dataset_from_index": int(new_index[rows[0]]),
                "dataset_to_index": int(new_index[rows[-1]]) + 1,
                "tasks": list(src_meta["tasks"]),
                "length": int(len(rows)),
                **{k: src_meta[k] for k in src_meta if k.startswith("stats/")},
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            })
            mapping.append({"subset_episode": remap[int(e)], "source_episode": int(e),
                            "source_file": int(src_i), "subset_file": out_i,
                            "length": int(len(rows))})
        next_ep += len(order)
        next_index += table.num_rows

        action_all.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
        state_all.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64))
        for c in ("timestamp", "frame_index", "task_index"):
            scalar_cols.setdefault(c, []).extend(
                float(np.asarray(v).ravel()[0]) for v in table.column(c).to_pylist())
        if on_progress:
            on_progress({"files_done": out_i + 1, "n_files": len(files),
                         "episodes": next_ep, "rows": next_index})

    # meta/episodes, from what the copied files contain
    ep_table = pa.Table.from_pylist(ep_rows)
    pq.write_table(ep_table, dst_root / META_EPISODES)

    # meta/tasks — all source tasks, indices unchanged so `task_index` needs no remap
    pq.write_table(pa.Table.from_pylist([dict(r) for r in tasks_rows]), dst_root / META_TASKS)

    # meta/info.json
    new_info = dict(info)
    new_info["total_episodes"] = next_ep
    new_info["total_frames"] = next_index
    new_info["splits"] = {"train": f"0:{next_ep}"}
    (dst_root / META_INFO).write_text(json.dumps(new_info, indent=4) + "\n")

    # meta/stats.json — quantiles over the subset's own rows, images aggregated from episode stats
    stats: Dict[str, object] = {}
    for key, values in (("action", np.concatenate(action_all)),
                        ("observation.state", np.concatenate(state_all))):
        stats[key] = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
                      "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
                      "count": [int(values.shape[0])], **_quantiles(values)}
    for key in ("observation.images.image", "observation.images.wrist_image"):
        stats[key] = _agg_image_stats(ep_rows, key)
    for c, vals in scalar_cols.items():
        a = np.asarray(vals, dtype=np.float64)
        stats[c] = {"min": [a.min()], "max": [a.max()], "mean": [a.mean()], "std": [a.std()],
                    "count": [int(a.size)]}
    for c, a in (("episode_index", np.arange(next_ep, dtype=np.float64)),
                 ("index", np.arange(next_index, dtype=np.float64))):
        stats[c] = {"min": [a.min()], "max": [a.max()], "mean": [a.mean()], "std": [a.std()],
                    "count": [int(a.size)]}
    (dst_root / META_STATS).write_text(json.dumps(stats, indent=4) + "\n")

    return {"n_files": len(files), "n_episodes": next_ep, "n_frames": next_index,
            "mapping": mapping, "source_files": sorted(int(f) for f in files),
            "stats_keys": sorted(stats)}


# ------------------------------------------------------------------ verification
def verify_subset(dst_root: pathlib.Path, src_root: Optional[pathlib.Path] = None,
                  mapping: Optional[Sequence[Dict[str, object]]] = None,
                  n_byte_checks: int = 8) -> Dict[str, object]:
    """Re-read the built dataset and check it against its own meta, then against the source bytes.

    Three things can go wrong and each has a check here: meta that does not locate rows (the released
    dataset's own bug), a renumbering that reorders rows, and a copy that re-encodes images.
    """
    import numpy as np
    import pyarrow.parquet as pq

    dst_root = pathlib.Path(dst_root)
    info = json.loads((dst_root / META_INFO).read_text())
    ep = pq.read_table(dst_root / META_EPISODES).to_pylist()
    files = sorted(p for p in (dst_root / "data" / "chunk-000").glob("file-*.parquet"))
    rows_seen = 0
    where: Dict[int, Dict[str, int]] = {}
    for i, path in enumerate(files):
        cols = pq.read_table(path, columns=["episode_index", "index", "frame_index"])
        e = np.asarray(cols.column("episode_index").to_pylist(), dtype=np.int64)
        g = np.asarray(cols.column("index").to_pylist(), dtype=np.int64)
        f = np.asarray(cols.column("frame_index").to_pylist(), dtype=np.int64)
        if not np.array_equal(g, np.arange(g[0], g[0] + g.size)):
            raise RuntimeError(f"{path.name}: global index is not contiguous")
        for eid in dict.fromkeys(e.tolist()):
            sel = np.flatnonzero(e == eid)
            if not np.array_equal(f[sel], np.arange(f[sel].size)):
                raise RuntimeError(f"{path.name}: episode {eid} frame_index is not 0..n-1")
            where[int(eid)] = {"file": i, "from": int(g[sel[0]]), "to": int(g[sel[-1]]) + 1}
        rows_seen += e.size

    problems = []
    for row in ep:
        eid = int(row["episode_index"])
        got = where.get(eid)
        if got is None:
            problems.append({"episode": eid, "why": "listed but absent"})
            continue
        if (got["file"] != int(row["data/file_index"])
                or got["from"] != int(row["dataset_from_index"])
                or got["to"] != int(row["dataset_to_index"])
                or (got["to"] - got["from"]) != int(row["length"])):
            problems.append({"episode": eid, "meta": row["data/file_index"], "actual": got})
    if problems:
        raise RuntimeError(f"{len(problems)} episodes whose meta does not locate their rows: {problems[:3]}")
    if rows_seen != int(info["total_frames"]) or len(ep) != int(info["total_episodes"]):
        raise RuntimeError(f"info.json says {info['total_episodes']}/{info['total_frames']}, "
                           f"files hold {len(ep)}/{rows_seen}")

    byte_checks = []
    if src_root is not None and mapping:
        import hashlib
        src_root = pathlib.Path(src_root)
        step = max(1, len(mapping) // max(1, n_byte_checks))
        for entry in list(mapping)[::step][:n_byte_checks]:
            src = pq.read_table(src_root / DATA_TEMPLATE.format(i=int(entry["source_file"])),
                                columns=["episode_index", "observation.images.image", "action"])
            dst = pq.read_table(dst_root / DATA_TEMPLATE.format(i=int(entry["subset_file"])),
                                columns=["episode_index", "observation.images.image", "action"])
            se = np.asarray(src.column("episode_index").to_pylist(), dtype=np.int64)
            de = np.asarray(dst.column("episode_index").to_pylist(), dtype=np.int64)
            si = np.flatnonzero(se == int(entry["source_episode"]))
            di = np.flatnonzero(de == int(entry["subset_episode"]))
            if si.size != di.size:
                raise RuntimeError(f"episode {entry} has {si.size} source rows and {di.size} copied")
            a = src.column("observation.images.image").to_pylist()
            b = dst.column("observation.images.image").to_pylist()
            h1 = hashlib.sha256(b"".join(_img_bytes(a[k]) for k in si)).hexdigest()
            h2 = hashlib.sha256(b"".join(_img_bytes(b[k]) for k in di)).hexdigest()
            if h1 != h2:
                raise RuntimeError(f"image bytes differ for {entry}")
            byte_checks.append({"subset_episode": int(entry["subset_episode"]),
                                "source_episode": int(entry["source_episode"]),
                                "n_rows": int(si.size), "sha256_16": h1[:16]})

    return {"n_files": len(files), "n_episodes": len(ep), "n_frames": rows_seen,
            "meta_locates_every_episode": True, "image_byte_checks": byte_checks}


def _img_bytes(cell) -> bytes:
    """One parquet image cell as bytes, whichever of LeRobot's two encodings it uses."""
    if isinstance(cell, dict):
        return bytes(cell.get("bytes") or b"")
    return bytes(cell)


def episodes_by_file(scan: Dict[str, object]) -> Dict[int, List[int]]:
    """episode ids per data file, from the scan's own per-episode record.

    `scan["per_file"]["episodes"]` is truncated to four ids for readability, so membership is read from
    `by_episode`, and an episode split across files raises: the subset copies whole files, and a split
    episode would make one of its halves unreachable.
    """
    out: Dict[int, List[int]] = {}
    for eid, entry in scan["by_episode"].items():
        files = sorted({int(f) for f in entry["files"]})
        if len(files) != 1:
            raise RuntimeError(f"episode {eid} spans files {files}")
        out.setdefault(files[0], []).append(int(eid))
    return {k: sorted(v) for k, v in sorted(out.items())}
