"""Materialize the fresh eight-item PG2 scalar-operator panel after design commit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from .materialize_localization_panel import SPATIAL_CASES
from .materialize_tasks import Bank, shape_scene


def materialize(config_path: Path, root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite scalar operator panel: {root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_operator_panel_materialization":
        raise ValueError("scalar operator design is not frozen")
    panel = config["panel"]
    count = int(panel["confirmation_items"])
    if count != len(SPATIAL_CASES):
        raise ValueError("operator panel must contain one item per spatial case")
    rng = random.Random(str(panel["generator_seed_domain"]))
    bank = Bank(root, {"image_mode": "RGB", "image_size": [224, 224]})
    for index, (target_index, reference_index, relation) in enumerate(SPATIAL_CASES):
        image, objects = shape_scene(rng)
        reference = objects[reference_index]
        bank.add(
            task_id=panel["task_id"],
            split="confirmation_eval",
            source_identity={
                "generator": "parallax_scalar_operator_shapes_v1",
                "seed_domain": panel["generator_seed_domain"],
                "index": index,
            },
            prompt=(
                f"Which labeled shape is {relation} the {reference['color']} "
                f"{reference['shape']}? Answer A, B, C, or D."
            ),
            target=objects[target_index]["label"],
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
    print(json.dumps({"status": result["status"], "total_items": result["total_items"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
