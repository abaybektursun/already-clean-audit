"""Mechanically separate train/discovery raw lines from the sealed confirmation manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .parallax_tasks import DISCOVERY_SPLITS


def run(source: Path, output: Path, receipt: Path) -> dict[str, object]:
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite discovery manifest or receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    counts: Counter[str] = Counter()
    total_source_rows = 0
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            for raw_line in incoming:
                total_source_rows += 1
                # The preparation step parses only the routing field and emits no
                # item content or target values to logs/results.
                split = str(json.loads(raw_line)["split"])
                counts[split] += 1
                if split in DISCOVERY_SPLITS:
                    outgoing.write(raw_line)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if set(counts) != {"train", "discovery_eval", "confirmation_eval"}:
        raise ValueError(f"unexpected source splits: {sorted(counts)}")
    discovery_rows = counts["train"] + counts["discovery_eval"]
    if sum(1 for _ in output.open("rb")) != discovery_rows:
        raise ValueError("discovery manifest row count mismatch")
    packet = {
        "schema_version": 1,
        "status": "GREEN",
        "source_manifest_sha256": sha256_file(source),
        "discovery_manifest_sha256": sha256_file(output),
        "source_rows": total_source_rows,
        "discovery_rows": discovery_rows,
        "routed_counts": dict(sorted(counts.items())),
        "confirmation_values_emitted": 0,
        "policy": "training_runtimes_must_open_only_items.discovery.jsonl",
    }
    atomic_json(receipt, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.source.resolve(), args.output.resolve(), args.receipt.resolve())
    print(json.dumps({"status": packet["status"], "discovery_rows": packet["discovery_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
