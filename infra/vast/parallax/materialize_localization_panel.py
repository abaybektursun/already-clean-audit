"""Materialize the independently seeded PG2 localization panel after design freeze."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from .materialize_tasks import Bank, shape_scene


SPATIAL_CASES = (
    (0, 1, "directly left of"),
    (1, 0, "directly right of"),
    (0, 2, "directly above"),
    (2, 0, "directly below"),
    (2, 3, "directly left of"),
    (3, 2, "directly right of"),
    (1, 3, "directly above"),
    (3, 1, "directly below"),
)


def materialize(config_path: Path, root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite localization panel: {root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_new_panel_materialization":
        raise ValueError("localization design is not frozen before panel materialization")
    panel = config["new_panel"]
    if panel["task_id"] != "rel_a_spatial_selection_localization_v1":
        raise ValueError("unexpected localization task ID")
    per_split = {
        "discovery_eval": int(panel["discovery_items"]),
        "confirmation_eval": int(panel["confirmation_items"]),
    }
    if any(count % len(SPATIAL_CASES) for count in per_split.values()):
        raise ValueError("panel counts must be multiples of the balanced eight-case grammar")
    bank = Bank(root, {"image_mode": "RGB", "image_size": [224, 224]})
    for split, count in per_split.items():
        seed_domain = str(panel["generator_seed_domains"][split])
        rng = random.Random(seed_domain)
        for index in range(count):
            image, objects = shape_scene(rng)
            target_index, reference_index, relation = SPATIAL_CASES[index % len(SPATIAL_CASES)]
            reference = objects[reference_index]
            answer = objects[target_index]["label"]
            bank.add(
                task_id=panel["task_id"],
                split=split,
                source_identity={
                    "generator": "parallax_localization_shapes_v1",
                    "seed_domain": seed_domain,
                    "index": index,
                },
                prompt=(
                    f"Which labeled shape is {relation} the {reference['color']} "
                    f"{reference['shape']}? Answer A, B, C, or D."
                ),
                target=answer,
                candidates=list("ABCD"),
                image=image,
                extra={
                    "scene": objects,
                    "relation": relation,
                    "reference": reference["label"],
                },
            )
    return bank.finish(config_path, config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.config, args.root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_items": result["total_items"],
                "confirmation_seal_sha256": result["confirmation_seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
