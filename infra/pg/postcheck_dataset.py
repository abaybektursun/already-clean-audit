"""POSTCHECK P4 — does `MolmoAct2-LIBERO-Dataset`'s episode metadata locate its rows?

Independent of DEPTHANCHOR by construction: this reads the released parquet files directly and never
opens `infra/pg/depthanchor_rowindex.json`, which is the artifact whose claim is under test. It also
never downloads the images. `meta/episodes` says, for every episode, which data file holds it
(`data/file_index`), where its rows start in the global index (`dataset_from_index`) and how many there
are (`length`); this scans the data files' own `episode_index` and `index` columns and checks all three.

WHY IT IS CHEAP. The data files are 23.8 GB because they carry two 256x256 image columns per row. The
four columns this needs — `episode_index`, `frame_index`, `index`, `task_index` — are int64, about 8 MB
in total across the whole dataset. Parquet stores columns separately, so reading only those four over
HTTP range requests touches a few hundred megabytes instead of the whole repository, and the scan is a
fresh read of the released files at a pinned revision either way.

THE FOUR COUNTS the spec asks for: episodes listed, episodes present, `file_index` mismatches, and
`from_index`/`length` mismatches — plus the identities of any episode listed and absent.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Sequence

COLUMNS = ("episode_index", "frame_index", "index", "task_index")
DATA_GLOB = "data/chunk-000/file-{i:03d}.parquet"
META_EPISODES = "meta/episodes/chunk-000/file-000.parquet"
META_INFO = "meta/info.json"


def _read_columns(open_file, columns: Sequence[str] = COLUMNS) -> Dict[str, List[int]]:
    """The named columns of one parquet file, as python lists. `open_file` is a callable returning a
    binary file object, so the caller decides between a local path and an HTTP range reader."""
    import pyarrow.parquet as pq
    with open_file() as fh:
        table = pq.read_table(fh, columns=list(columns))
    return {c: table.column(c).to_pylist() for c in columns}


def scan_data_files(opener, n_files: int, columns: Sequence[str] = COLUMNS,
                    on_progress=None) -> Dict[str, object]:
    """Walk every data file and record, per episode, which file holds its rows and what its global
    `index` span is. `opener(i)` returns a callable opening data file `i`."""
    by_episode: Dict[int, Dict[str, object]] = {}
    per_file = []
    for i in range(n_files):
        cols = _read_columns(opener(i), columns)
        ep = cols["episode_index"]
        idx = cols["index"]
        seen: Dict[int, List[int]] = {}
        for e, g in zip(ep, idx):
            seen.setdefault(int(e), []).append(int(g))
        for e, spans in seen.items():
            entry = by_episode.setdefault(int(e), {"episode_index": int(e), "files": [],
                                                   "n_rows": 0, "index_min": None, "index_max": None})
            entry["files"].append(i)
            entry["n_rows"] += len(spans)
            lo, hi = min(spans), max(spans)
            entry["index_min"] = lo if entry["index_min"] is None else min(entry["index_min"], lo)
            entry["index_max"] = hi if entry["index_max"] is None else max(entry["index_max"], hi)
        per_file.append({"file_index": i, "n_rows": len(ep),
                         "n_episodes": len(seen), "episodes": sorted(seen)[:4],
                         "task_indices": sorted({int(t) for t in cols["task_index"]})[:6]})
        if on_progress is not None:
            on_progress({"files_done": i + 1, "n_files": n_files,
                         "episodes_seen": len(by_episode),
                         "rows_seen": sum(v["n_rows"] for v in by_episode.values())})
    return {"by_episode": by_episode, "per_file": per_file,
            "n_files": n_files,
            "n_episodes_present": len(by_episode),
            "n_rows_present": sum(v["n_rows"] for v in by_episode.values())}


def _histogram(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(int(v))] = out.get(str(int(v)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def compare(meta_rows: List[Dict[str, object]], scan: Dict[str, object],
            info: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """`meta/episodes` against the scan. Every mismatch is kept with both values, because a count
    without an example is not checkable by the person receiving it."""
    present = scan["by_episode"]
    file_mismatch, span_mismatch, absent, multi_file = [], [], [], []
    listed_file_index_range = [None, None]
    for row in meta_rows:
        e = int(row["episode_index"])
        listed_file = row.get("data/file_index")
        if listed_file is not None:
            lf = int(listed_file)
            listed_file_index_range[0] = lf if listed_file_index_range[0] is None else min(listed_file_index_range[0], lf)
            listed_file_index_range[1] = lf if listed_file_index_range[1] is None else max(listed_file_index_range[1], lf)
        if e not in present:
            absent.append({"episode_index": e, "listed_file_index": listed_file,
                           "listed_length": row.get("length"),
                           "listed_from_index": row.get("dataset_from_index")})
            continue
        got = present[e]
        if len(got["files"]) > 1:
            multi_file.append({"episode_index": e, "files": got["files"]})
        actual_file = got["files"][0]
        if listed_file is not None and int(listed_file) != int(actual_file):
            file_mismatch.append({"episode_index": e, "listed": int(listed_file),
                                  "actual": int(actual_file)})
        want_from = row.get("dataset_from_index")
        want_len = row.get("length")
        got_from, got_len = got["index_min"], got["n_rows"]
        if (want_from is not None and int(want_from) != int(got_from)) or \
           (want_len is not None and int(want_len) != int(got_len)):
            span_mismatch.append({"episode_index": e,
                                  "listed_from_index": None if want_from is None else int(want_from),
                                  "actual_index_min": int(got_from),
                                  "listed_length": None if want_len is None else int(want_len),
                                  "actual_n_rows": int(got_len)})
    out = {
        "episodes_listed": len(meta_rows),
        "episodes_present": scan["n_episodes_present"],
        "rows_present": scan["n_rows_present"],
        "file_index_mismatches": len(file_mismatch),
        "from_index_or_length_mismatches": len(span_mismatch),
        "episodes_listed_but_absent": len(absent),
        "episodes_spanning_more_than_one_file": len(multi_file),
        "listed_file_index_range": listed_file_index_range,
        "n_data_files": scan["n_files"],
        "absent_episodes": absent,
        # the FULL list, not a sample: 372 rows of three integers is 12 kB, and "the listed index is
        # one less than the actual" is a bug someone can fix in a minute where "372 are wrong" is a bug
        # someone has to investigate. The offset histogram is the sentence itself.
        "file_index_offset_histogram": _histogram(m["actual"] - m["listed"] for m in file_mismatch),
        "file_index_mismatches_all": file_mismatch,
        "first_mismatching_episode": (min(m["episode_index"] for m in file_mismatch)
                                      if file_mismatch else None),
        "file_index_mismatch_examples": file_mismatch[:10],
        "span_mismatch_examples": span_mismatch[:10],
        "multi_file_examples": multi_file[:10],
    }
    if info:
        out["info_json"] = {k: info.get(k) for k in
                            ("total_episodes", "total_frames", "total_tasks", "chunks_size",
                             "data_files_size_in_mb", "codebase_version") if k in info}
        out["info_vs_scan"] = {
            "episodes_listed_in_info": info.get("total_episodes"),
            "episodes_present": scan["n_episodes_present"],
            "frames_listed_in_info": info.get("total_frames"),
            "rows_present": scan["n_rows_present"],
            "episodes_match": info.get("total_episodes") == scan["n_episodes_present"],
            "frames_match": info.get("total_frames") == scan["n_rows_present"],
        }
    return out


# ------------------------------------------------------------------ the registered failing input
def selftest(tmp: Optional[pathlib.Path] = None) -> Dict[str, object]:
    """Spec §4: the scanner is checked on a synthetic three-file dataset with ONE planted file_index
    mismatch and ONE planted absent episode before it touches the real files. It must find exactly
    those, and no others."""
    import shutil
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = pathlib.Path(tmp or tempfile.mkdtemp(prefix="postcheck_p4_"))
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    # three files, two episodes each, six rows per episode
    rows, meta, g = [], [], 0
    for f in range(3):
        ep_idx, fr_idx, idx, task = [], [], [], []
        for e in (f * 2, f * 2 + 1):
            start = g
            for k in range(6):
                ep_idx.append(e); fr_idx.append(k); idx.append(g); task.append(e % 3)
                g += 1
            meta.append({"episode_index": e, "data/file_index": f,
                         "dataset_from_index": start, "length": 6})
        pq.write_table(pa.table({"episode_index": ep_idx, "frame_index": fr_idx,
                                 "index": idx, "task_index": task}),
                       root / "data" / "chunk-000" / f"file-{f:03d}.parquet")
        rows.append(len(ep_idx))
    # PLANT ONE: episode 3 is listed in the wrong file. PLANT TWO: episode 99 is listed and absent.
    for m in meta:
        if m["episode_index"] == 3:
            m["data/file_index"] = 0
    meta.append({"episode_index": 99, "data/file_index": 2, "dataset_from_index": 999, "length": 6})

    scan = scan_data_files(lambda i: (lambda: open(root / "data" / "chunk-000" / f"file-{i:03d}.parquet", "rb")), 3)
    got = compare(meta, scan, info={"total_episodes": 7, "total_frames": 42})
    assert got["episodes_listed"] == 7, got
    assert got["episodes_present"] == 6, got
    assert got["rows_present"] == 36, got
    assert got["file_index_mismatches"] == 1, got["file_index_mismatch_examples"]
    assert got["file_index_mismatch_examples"][0] == {"episode_index": 3, "listed": 0, "actual": 1}, got
    assert got["episodes_listed_but_absent"] == 1, got
    assert got["absent_episodes"][0]["episode_index"] == 99, got
    assert got["from_index_or_length_mismatches"] == 0, got["span_mismatch_examples"]
    assert got["info_vs_scan"]["episodes_match"] is False and got["info_vs_scan"]["frames_match"] is False
    # and the negative control: with the plants removed, the scanner must find nothing
    clean = [m for m in meta if m["episode_index"] != 99]
    for m in clean:
        if m["episode_index"] == 3:
            m["data/file_index"] = 1
    ok = compare(clean, scan, info={"total_episodes": 6, "total_frames": 36})
    assert ok["file_index_mismatches"] == 0 and ok["episodes_listed_but_absent"] == 0, ok
    assert ok["from_index_or_length_mismatches"] == 0, ok
    assert ok["info_vs_scan"]["episodes_match"] is True and ok["info_vs_scan"]["frames_match"] is True
    if tmp is None:
        shutil.rmtree(root, ignore_errors=True)
    print("[postcheck_dataset] SELFTEST OK — planted 1 file_index mismatch and 1 absent episode, "
          "found exactly those; clean copy finds none")
    return {"planted": got, "clean": ok}


if __name__ == "__main__":
    selftest()
