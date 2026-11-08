"""Structural invariants over the REGRAIN job mains — no model, no GPU, pure AST.

These pin two failure families that actually bit this workstream (2026-07-09):

  1. `rep["green"] = True` was written BEFORE the summary `print(...)`. Two jobs
     (regrain_offline, regrain_c_disc) then crashed on a stale key inside that print and
     exited 1 while leaving a GREEN artifact behind. The drivers gate on `green`, so a
     green-but-dead job is a status signal that cannot fail. Green must be the LAST
     statement its block executes.

  2. A job's summary print reaches into `rep` with string keys that silently drift when the
     producing code renames them (`rho_is_binding` -> `rho_stopped_before_cap`,
     `mdl["0.2"]` -> `mdl["0.20"]`). Nothing catches that until the job dies at the end of a
     multi-hour GPU run. We cannot type-check dict keys here, but we CAN require that the
     crash-prone print never runs after the artifact has been declared complete.
"""
import ast
import pathlib

import pytest

JOBS = sorted((pathlib.Path(__file__).parents[2] / "aws_pg" / "jobs").glob("regrain_*.py"))
assert JOBS, "no regrain job modules found"


def _green_stmt(s):
    return (isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Subscript)
            and getattr(s.targets[0].value, "id", None) == "rep"
            and getattr(s.targets[0].slice, "value", None) == "green"
            and getattr(s.value, "value", None) is True)


def _is_writer(s):
    return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", None) == "_w")


def _is_return(s):
    return isinstance(s, ast.Return)


@pytest.mark.parametrize("path", JOBS, ids=lambda p: p.name)
def test_green_is_the_last_thing_a_job_does(path):
    """After `rep["green"] = True` only `_w()` and an immediate `return` may follow.

    Anything else -- most dangerously a summary print that indexes `rep` -- can raise and
    strand a green artifact on a job that exited nonzero.
    """
    tree = ast.parse(path.read_text())
    seen = False
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if not _green_stmt(stmt):
                continue
            seen = True
            tail = body[i + 1:]
            assert tail and _is_writer(tail[0]), \
                f"{path.name}: `green = True` must be followed immediately by `_w()`"
            extra = [s for s in tail[1:] if not _is_return(s)]
            assert not extra, (
                f"{path.name}: {len(extra)} statement(s) run AFTER the artifact is declared "
                f"green (first at line {extra[0].lineno}). A crash there leaves a green "
                f"artifact on a dead job. Move them above `green = True`.")
    if path.name not in ("regrain_dryrun.py", "regrain_figures.py"):
        assert seen, f"{path.name}: no `rep['green'] = True` -- every job must declare completion"


@pytest.mark.parametrize("path", JOBS, ids=lambda p: p.name)
def test_no_try_except_in_job_science(path):
    """House rule: fail and trace, never fail and hide."""
    bad = [n.lineno for n in ast.walk(ast.parse(path.read_text()))
           if isinstance(n, (ast.Try, ast.ExceptHandler))]
    assert not bad, f"{path.name}: try/except at line(s) {bad}"
