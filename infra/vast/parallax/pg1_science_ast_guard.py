"""Prove PG1 v3 edits did not alter registered scientific computation ASTs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


BASELINE_COMMIT = "8bba643"
BASELINE_PATH = Path(
    "infra/vast/parallax/tests/fixtures/pg1_science_ast_baseline_8bba643.json"
)
CELL_PATH = Path("infra/vast/parallax/training_interface_control_cell.py")
PRIMARY_PATH = Path("infra/vast/parallax/training_interface_control_primary_cell.py")
CRITIC_PATH = Path("infra/vast/parallax/audit_primary_interface_independent.py")
NATIVE_CELL_PATH = Path("infra/vast/parallax/pg1_native_cell.py")
NATIVE_CONTROLLER_PATH = Path("infra/vast/parallax/pg1_native_controller.py")
NATIVE_WORKER_PATH = Path("infra/vast/parallax/pg1_native_worker.py")
PROTECTED_FUNCTIONS = {
    str(CELL_PATH): (
        "_objective",
        "_mean_score_items",
        "_cell_adjudication",
        "_aggregate_adjudication",
    ),
    str(PRIMARY_PATH): (
        "_validate_primary_config",
        "_metric_inputs",
        "adjudicate_primary_packet",
    ),
    str(CRITIC_PATH): (
        "_void",
        "_mapping",
        "_parse_rows",
        "_matrix",
        "_mapped_interaction",
        "_compare",
        "_percentile",
        "_sampled_gain",
        "recompute",
        "_close",
        "_assert_producer_agreement",
        "build_audit",
    ),
}
PROTECTED_CONSTANTS = {
    str(CELL_PATH): ("ALLOWED",),
    str(PRIMARY_PATH): ("LINEAGE", "ROLE", "DESIGN_ID"),
    str(CRITIC_PATH): (
        "SEMANTIC_LABELS",
        "CODES",
        "CODEBOOKS",
        "EXPECTED_STEPS",
        "METRIC_ID",
        "BOOTSTRAP_SEED",
        "BOOTSTRAP_REPLICATES",
        "PERMUTATION_SEED",
        "PERMUTATION_REPLICATES",
        "MAX_P",
    ),
}


class ScienceAstError(RuntimeError):
    pass


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assignments(tree: ast.Module) -> dict[str, ast.Assign | ast.AnnAssign]:
    rows: dict[str, ast.Assign | ast.AnnAssign] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    rows[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            rows[node.target.id] = node
    return rows


def _statement_index(function: ast.FunctionDef | ast.AsyncFunctionDef, target: str) -> int:
    for index, statement in enumerate(function.body):
        if isinstance(statement, ast.Assign) and any(
            isinstance(candidate, ast.Name) and candidate.id == target
            for candidate in statement.targets
        ):
            return index
    raise ScienceAstError(f"protected run anchor is absent: {target}")


def fingerprint_sources(sources: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "pg1-scientific-ast-baseline/v1",
        "baseline_commit": BASELINE_COMMIT,
        "files": {},
    }
    for path in (str(CELL_PATH), str(PRIMARY_PATH), str(CRITIC_PATH)):
        tree = _tree(sources[path])
        functions = _functions(tree)
        assignments = _assignments(tree)
        function_hashes = {
            name: _hash(ast.dump(functions[name], include_attributes=False))
            for name in PROTECTED_FUNCTIONS[path]
        }
        constant_hashes = {
            name: _hash(ast.dump(assignments[name], include_attributes=False))
            for name in PROTECTED_CONSTANTS[path]
        }
        slices: dict[str, str] = {}
        if path == str(CELL_PATH):
            run = functions["run"]
            anchor = _statement_index(run, "lineage_rows")
            slices["run_from_lineage_rows"] = _hash(
                [ast.dump(node, include_attributes=False) for node in run.body[anchor:]]
            )
        elif path == str(PRIMARY_PATH):
            run = functions["run"]
            anchor = _statement_index(run, "result")
            slices["run_from_primary_adjudication"] = _hash(
                [ast.dump(node, include_attributes=False) for node in run.body[anchor:]]
            )
        result["files"][path] = {
            "functions": function_hashes,
            "constants": constant_hashes,
            "protected_slices": slices,
        }
    return result


def current_fingerprint(root: Path) -> dict[str, Any]:
    paths = (CELL_PATH, PRIMARY_PATH, CRITIC_PATH)
    return fingerprint_sources(
        {str(path): (root / path).read_text(encoding="utf-8") for path in paths}
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _filtered_call_trace(
    function: ast.FunctionDef | ast.AsyncFunctionDef, allowed: set[str]
) -> list[str]:
    calls = [
        (node.lineno, _call_name(node))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) in allowed
    ]
    return [name for _, name in sorted(calls) if name is not None]


def verify_native_equivalence(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    native_tree = _tree((root / NATIVE_CELL_PATH).read_text(encoding="utf-8"))
    native_functions = _functions(native_tree)
    legacy_cell = baseline["files"][str(CELL_PATH)]
    legacy_primary = baseline["files"][str(PRIMARY_PATH)]
    for name in PROTECTED_FUNCTIONS[str(CELL_PATH)]:
        observed = _hash(ast.dump(native_functions[name], include_attributes=False))
        if observed != legacy_cell["functions"][name]:
            raise ScienceAstError(f"native copied scientific function changed: {name}")
    science = native_functions["_run_science"]
    science_anchor = _statement_index(science, "lineage_rows")
    science_slice = _hash(
        [ast.dump(node, include_attributes=False) for node in science.body[science_anchor:]]
    )
    if science_slice != legacy_cell["protected_slices"]["run_from_lineage_rows"]:
        raise ScienceAstError("native scientific run slice differs from 8bba643")
    primary = native_functions["run"]
    primary_anchor = _statement_index(primary, "result")
    primary_slice = _hash(
        [ast.dump(node, include_attributes=False) for node in primary.body[primary_anchor:]]
    )
    if primary_slice != legacy_primary["protected_slices"]["run_from_primary_adjudication"]:
        raise ScienceAstError("native primary adjudication slice differs from 8bba643")

    controller = _functions(
        _tree((root / NATIVE_CONTROLLER_PATH).read_text(encoding="utf-8"))
    )["execute"]
    controller_trace = _filtered_call_trace(
        controller,
        {
            "load_issued",
            "consume",
            "capsule_preflight_graph",
            "execute_job",
            "stop_preserve_scratch",
            "finalize_attempt",
        },
    )
    expected_controller = [
        "load_issued",
        "consume",
        "capsule_preflight_graph",
        "execute_job",
        "finalize_attempt",
        "stop_preserve_scratch",
    ]
    if controller_trace != expected_controller:
        raise ScienceAstError("native controller authorization/execution trace changed")
    worker = _functions(_tree((root / NATIVE_WORKER_PATH).read_text(encoding="utf-8")))[
        "finalize_attempt"
    ]
    worker_trace = _filtered_call_trace(
        worker,
        {
            "durable_sink",
            "verify_durability",
            "run_critic",
            "request_destroy",
            "stop_preserve_scratch",
        },
    )
    expected_worker = [
        "durable_sink",
        "verify_durability",
        "run_critic",
        "durable_sink",
        "verify_durability",
        "durable_sink",
        "verify_durability",
        "durable_sink",
        "verify_durability",
        "request_destroy",
        "stop_preserve_scratch",
    ]
    if worker_trace != expected_worker:
        raise ScienceAstError("native critic/durability/destroy trace changed")
    return {
        "native_cell": str(NATIVE_CELL_PATH),
        "copied_scientific_functions": len(PROTECTED_FUNCTIONS[str(CELL_PATH)]),
        "scientific_slice_sha256": science_slice,
        "primary_slice_sha256": primary_slice,
        "controller_call_trace": controller_trace,
        "worker_call_trace": worker_trace,
    }


def verify(root: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = current_fingerprint(root)
    if current != baseline:
        raise ScienceAstError("scientific AST/constants differ from baseline commit 8bba643")
    native = verify_native_equivalence(root, baseline)
    return {
        "status": "GREEN_SCIENTIFIC_AST_IDENTICAL_TO_8BBA643",
        "baseline": str(baseline_path),
        "baseline_commit": BASELINE_COMMIT,
        "protected_files": len(current["files"]),
        "protected_functions": sum(
            len(row["functions"]) for row in current["files"].values()
        ),
        "protected_constants": sum(
            len(row["constants"]) for row in current["files"].values()
        ),
        "protected_run_slices": sum(
            len(row["protected_slices"]) for row in current["files"].values()
        ),
        "native_equivalence": native,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else args.root / args.baseline
    print(json.dumps(verify(args.root.resolve(), baseline), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
