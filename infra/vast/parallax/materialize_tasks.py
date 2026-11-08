"""Materialize the frozen PARALLAX truth-sprint task bank before GPU use."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from datasets import load_dataset
from PIL import Image, ImageDraw

from .artifacts import atomic_json, sha256_file


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Bank:
    def __init__(self, root: Path, global_config: dict[str, Any]):
        self.root = root
        self.image_root = root / "images"
        self.manifest_root = root / "manifests"
        self.global_config = global_config
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        *,
        task_id: str,
        split: str,
        source_identity: dict[str, Any],
        prompt: str,
        target: str,
        candidates: list[str] | None = None,
        image: Image.Image | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        semantic = {
            "task_id": task_id,
            "split": split,
            "source_identity": source_identity,
            "prompt": prompt,
            "target": target,
            "candidates": candidates,
            "extra": extra or {},
        }
        item_id = canonical_hash(semantic)
        row: dict[str, Any] = {
            "item_id": item_id,
            "task_id": task_id,
            "split": split,
            "prompt": prompt,
            "target": target,
            "candidates": candidates,
            "source_identity": source_identity,
            **(extra or {}),
        }
        if image is not None:
            normalized = image.convert(self.global_config["image_mode"]).resize(
                tuple(self.global_config["image_size"]), Image.Resampling.BICUBIC
            )
            relative = Path("images") / task_id / split / f"{item_id}.png"
            target_path = self.root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(target_path, format="PNG", optimize=True)
            row["image"] = relative.as_posix()
            row["image_sha256"] = sha256_file(target_path)
            row["image_bytes"] = target_path.stat().st_size
        else:
            row["image"] = None
        self.items.append(row)

    def finish(self, config_path: Path, task_path: Path) -> dict[str, Any]:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.items.sort(key=lambda row: (row["task_id"], row["split"], row["item_id"]))
        item_path = self.manifest_root / "items.jsonl"
        with item_path.open("w", encoding="utf-8") as handle:
            for row in self.items:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in self.items:
            counts[row["task_id"]][row["split"]] += 1
        split_summary = {
            "schema_version": 1,
            "task_counts": {task: dict(sorted(counter.items())) for task, counter in sorted(counts.items())},
            "total_items": len(self.items),
        }
        atomic_json(self.manifest_root / "splits.json", split_summary)

        confirmation_ids = sorted(
            row["item_id"] for row in self.items if row["split"] == "confirmation_eval"
        )
        confirmation_seal = hashlib.sha256("\n".join(confirmation_ids).encode()).hexdigest()
        (self.manifest_root / "confirmation.sha256").write_text(confirmation_seal + "\n")

        image_rows = [row for row in self.items if row["image"] is not None]
        content = {
            "schema_version": 1,
            "status": "GREEN_MATERIALIZED",
            "data_config_sha256": sha256_file(config_path),
            "task_config_sha256": sha256_file(task_path),
            "items_jsonl_sha256": sha256_file(item_path),
            "confirmation_seal_sha256": confirmation_seal,
            "total_items": len(self.items),
            "total_images": len(image_rows),
            "total_image_bytes": sum(int(row["image_bytes"]) for row in image_rows),
            "unique_item_ids": len({row["item_id"] for row in self.items}),
            "unique_image_hashes": len({row["image_sha256"] for row in image_rows}),
            "split_summary": split_summary["task_counts"],
            "payload_policy": "local_payload_ignored_manifest_tracked",
        }
        if content["unique_item_ids"] != content["total_items"]:
            raise AssertionError("duplicate item IDs")
        atomic_json(self.manifest_root / "content.json", content)
        return content


def _take_per_class(
    dataset: Iterable[dict[str, Any]],
    *,
    label_key: str,
    counts: dict[int, int],
) -> list[tuple[int, dict[str, Any]]]:
    found: Counter[int] = Counter()
    selected: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in enumerate(dataset):
        label = int(row[label_key])
        if label not in counts or found[label] >= counts[label]:
            continue
        selected.append((source_index, row))
        found[label] += 1
        if all(found[label] >= wanted for label, wanted in counts.items()):
            break
    if any(found[label] < wanted for label, wanted in counts.items()):
        raise RuntimeError(f"source exhausted before balanced sample: {found} vs {counts}")
    return selected


def materialize_external(bank: Bank, data: dict[str, Any]) -> None:
    beans = data["sources"]["hf_beans_pinned"]
    bean_labels = beans["labels"]
    for split, source_split in beans["source_splits"].items():
        per_class = int(beans["per_class"][split])
        dataset = load_dataset(
            beans.get("load_repo", beans["repo"]),
            split=source_split,
            streaming=True,
            revision=beans["revision"],
        )
        selected = _take_per_class(
            dataset, label_key="labels", counts={index: per_class for index in range(len(bean_labels))}
        )
        for source_index, row in selected:
            label = bean_labels[int(row["labels"])]
            bank.add(
                task_id="vr_a_beans_balanced",
                split=split,
                source_identity={
                    "repo": beans["repo"],
                    "revision": beans["revision"],
                    "source_split": source_split,
                    "source_index": source_index,
                },
                prompt="Identify the condition of the bean leaf. Answer with one label: " + ", ".join(bean_labels) + ".",
                target=label,
                candidates=bean_labels,
                image=row["image"],
            )

    fashion = data["sources"]["hf_fashion_mnist_pinned"]
    fashion_labels = fashion["labels"]
    # Source-train items are consumed in stable order: first train allocation,
    # then a disjoint discovery allocation. Official test is confirmation only.
    train_count = int(fashion["per_class"]["train"])
    discovery_count = int(fashion["per_class"]["discovery_eval"])
    source_train = load_dataset(
        fashion.get("load_repo", fashion["repo"]),
        split="train", streaming=True, revision=fashion["revision"]
    )
    selected = _take_per_class(
        source_train,
        label_key="label",
        counts={index: train_count + discovery_count for index in range(len(fashion_labels))},
    )
    seen: Counter[int] = Counter()
    for source_index, row in selected:
        label_index = int(row["label"])
        split = "train" if seen[label_index] < train_count else "discovery_eval"
        seen[label_index] += 1
        bank.add(
            task_id="vr_b_fashion_category",
            split=split,
            source_identity={
                "repo": fashion["repo"],
                "revision": fashion["revision"],
                "source_split": "train",
                "source_index": source_index,
            },
            prompt="Identify the clothing category. Answer with one label: " + ", ".join(fashion_labels) + ".",
            target=fashion_labels[label_index],
            candidates=fashion_labels,
            image=row["image"],
        )
    confirmation_count = int(fashion["per_class"]["confirmation_eval"])
    source_test = load_dataset(
        fashion.get("load_repo", fashion["repo"]),
        split="test", streaming=True, revision=fashion["revision"]
    )
    for source_index, row in _take_per_class(
        source_test,
        label_key="label",
        counts={index: confirmation_count for index in range(len(fashion_labels))},
    ):
        label_index = int(row["label"])
        bank.add(
            task_id="vr_b_fashion_category",
            split="confirmation_eval",
            source_identity={
                "repo": fashion["repo"],
                "revision": fashion["revision"],
                "source_split": "test",
                "source_index": source_index,
            },
            prompt="Identify the clothing category. Answer with one label: " + ", ".join(fashion_labels) + ".",
            target=fashion_labels[label_index],
            candidates=fashion_labels,
            image=row["image"],
        )


SHAPES = ("circle", "square", "triangle", "diamond")
COLORS = ("red", "blue", "green", "orange")
POSITIONS = ((38, 38), (150, 38), (38, 150), (150, 150))


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: str, center: tuple[int, int], label: str) -> None:
    x, y = center
    box = (x - 22, y - 22, x + 22, y + 22)
    if shape == "circle":
        draw.ellipse(box, fill=color, outline="black", width=2)
    elif shape == "square":
        draw.rectangle(box, fill=color, outline="black", width=2)
    elif shape == "triangle":
        draw.polygon([(x, y - 26), (x - 25, y + 22), (x + 25, y + 22)], fill=color, outline="black")
    else:
        draw.polygon([(x, y - 27), (x - 25, y), (x, y + 27), (x + 25, y)], fill=color, outline="black")
    draw.text((x - 4, y - 7), label, fill="white", stroke_width=2, stroke_fill="black")


def shape_scene(rng: random.Random) -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (224, 224), "#f3f1e8")
    draw = ImageDraw.Draw(image)
    shapes = list(SHAPES)
    colors = list(COLORS)
    rng.shuffle(shapes)
    rng.shuffle(colors)
    # Preserve exact rows/columns for unambiguous "directly" relations while
    # making the pixels item-specific enough to prevent cross-split duplicates.
    xs = (38 + rng.randrange(-8, 9), 150 + rng.randrange(-8, 9))
    ys = (38 + rng.randrange(-8, 9), 150 + rng.randrange(-8, 9))
    positions = ((xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1]))
    objects = []
    for index, (position, shape, color) in enumerate(zip(positions, shapes, colors)):
        label = "ABCD"[index]
        _draw_shape(draw, shape, color, position, label)
        objects.append({"label": label, "shape": shape, "color": color, "x": position[0], "y": position[1]})
    return image, objects


def materialize_synthetic(bank: Bank, data: dict[str, Any]) -> None:
    split_counts = data["sources"]["synthetic_shapes_v1"]["counts_per_candidate"]
    relations = ("directly left of", "directly right of", "directly above", "directly below")
    # (target, reference) pairs are adjacent on the 2x2 grid, so each query has
    # exactly one answer rather than a same-row/same-column ambiguity.
    spatial_cases = (
        (0, 1, "directly left of"),
        (1, 0, "directly right of"),
        (0, 2, "directly above"),
        (2, 0, "directly below"),
        (2, 3, "directly left of"),
        (3, 2, "directly right of"),
        (1, 3, "directly above"),
        (3, 1, "directly below"),
    )
    for split, count in split_counts.items():
        rng = random.Random(f"parallax-shapes-v1:{split}")
        for index in range(int(count)):
            image, objects = shape_scene(rng)
            target_index, reference_index, relation = spatial_cases[index % len(spatial_cases)]
            reference = objects[reference_index]
            answer = objects[target_index]["label"]
            bank.add(
                task_id="rel_a_spatial_selection",
                split=split,
                source_identity={"generator": "synthetic_shapes_v1", "split": split, "index": index},
                prompt=f"Which labeled shape is {relation} the {reference['color']} {reference['shape']}? Answer A, B, C, or D.",
                target=answer,
                candidates=list("ABCD"),
                image=image,
                extra={"scene": objects, "relation": relation, "reference": reference["label"]},
            )

            first, second = objects[0], objects[3]
            true_relation = "left of" if first["x"] < second["x"] else "right of"
            false_relation = "right of" if true_relation == "left of" else "left of"
            true_value = index % 2 == 0
            relation2 = true_relation if true_value else false_relation
            bank.add(
                task_id="rel_b_relation_comparison",
                split=split,
                source_identity={"generator": "synthetic_shapes_v1", "split": split, "index": index},
                prompt=f"Is shape {first['label']} {relation2} shape {second['label']}? Answer yes or no.",
                target="yes" if true_value else "no",
                candidates=["yes", "no"],
                image=image,
                extra={"scene": objects, "relation": relation2, "first": first["label"], "second": second["label"]},
            )

    nonce_counts = data["sources"]["synthetic_nonce_facts_v1"]["counts"]
    syllables = ("zor", "vek", "tal", "mip", "lun", "daf", "kir", "bos")
    properties = ("amber", "silent", "northern", "striped", "cobalt", "warm", "hollow", "swift")
    used_nonces: set[str] = set()
    used_image_nonces: set[str] = set()
    for split, count in nonce_counts.items():
        rng = random.Random(f"parallax-nonce-v1:{split}")
        for index in range(int(count)):
            nonce = rng.choice(syllables) + rng.choice(syllables) + str(index)
            while nonce in used_nonces:
                nonce = rng.choice(syllables) + rng.choice(syllables) + str(index) + rng.choice(syllables)
            used_nonces.add(nonce)
            prop = properties[index % len(properties)]
            bank.add(
                task_id="new_a_nonce_fact_binding",
                split=split,
                source_identity={"generator": "synthetic_nonce_facts_v1", "split": split, "index": index},
                prompt=f"A {nonce} is always described as {prop}.\nQuestion: How is a {nonce} described?\nAnswer:",
                target=prop,
                candidates=list(properties),
                extra={"nonce": nonce, "template_id": index % 4},
            )

            symbol = Image.new("RGB", (224, 224), "white")
            draw = ImageDraw.Draw(symbol)
            points = [(rng.randrange(25, 199), rng.randrange(25, 199)) for _ in range(6)]
            draw.line(points + [points[0]], fill=COLORS[index % len(COLORS)], width=9, joint="curve")
            image_nonce = f"sig{rng.choice(syllables)}{index}"
            while image_nonce in used_image_nonces:
                image_nonce = f"sig{rng.choice(syllables)}{index}{rng.choice(syllables)}"
            used_image_nonces.add(image_nonce)
            bank.add(
                task_id="new_b_image_nonce_binding",
                split=split,
                source_identity={"generator": "synthetic_nonce_images_v1", "split": split, "index": index},
                prompt="What is this symbol called?",
                target=image_nonce,
                image=symbol,
                extra={"training_statement": f"This symbol is called {image_nonce}. Remember it.", "nonce": image_nonce},
            )

    interface_counts = data["sources"]["synthetic_interface_v1"]["counts_per_candidate"]
    codes = {"yes": "zorp", "no": "miv"}
    used_pairs: set[tuple[int, int]] = set()
    for split, count in interface_counts.items():
        rng = random.Random(f"parallax-interface-v1:{split}")
        for index in range(int(count)):
            first_number = rng.randrange(1, 100)
            second_number = rng.randrange(1, 100)
            while first_number == second_number:
                second_number = rng.randrange(1, 100)
            truth = index % 2 == 0
            left, right = (
                (max(first_number, second_number), min(first_number, second_number))
                if truth
                else (min(first_number, second_number), max(first_number, second_number))
            )
            while (left, right) in used_pairs:
                first_number = rng.randrange(1, 100)
                second_number = rng.randrange(1, 100)
                if first_number == second_number:
                    continue
                left, right = (
                    (max(first_number, second_number), min(first_number, second_number))
                    if truth
                    else (min(first_number, second_number), max(first_number, second_number))
                )
            used_pairs.add((left, right))
            correct = left + right
            options = [correct, correct + 1, correct - 1, correct + 2]
            rng.shuffle(options)
            letter = "ABCD"[options.index(correct)]
            rendered = "\n".join(f"{label}. {value}" for label, value in zip("ABCD", options))
            bank.add(
                task_id="iface_a_permuted_abcd",
                split=split,
                source_identity={"generator": "synthetic_interface_v1", "split": split, "index": index},
                prompt=f"What is {left} + {right}?\n{rendered}\nAnswer with the letter.",
                target=letter,
                candidates=list("ABCD"),
                extra={"semantic_answer": str(correct), "option_values": options},
            )
            bank.add(
                task_id="iface_b_nonce_output_codes",
                split=split,
                source_identity={"generator": "synthetic_interface_v1", "split": split, "index": index},
                prompt=f"Is {left} greater than {right}? Answer only with the registered code.",
                target=codes["yes" if truth else "no"],
                candidates=list(codes.values()),
                extra={"semantic_answer": "yes" if truth else "no", "codebook": codes},
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--identity-data-config", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.root.exists():
        if not args.replace:
            raise SystemExit(f"refusing to overwrite existing bank: {args.root}")
        shutil.rmtree(args.root)
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    bank = Bank(args.root, data["global"])
    materialize_external(bank, data)
    materialize_synthetic(bank, data)
    content = bank.finish(args.identity_data_config or args.data_config, args.task_config)
    print(json.dumps(content, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
