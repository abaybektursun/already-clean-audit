"""LATMECH weight-free dry-run gate (the prereg §9.1 gate, packaged as a pg job module).

Runs the canonical pure-logic suite `infra/pg/tests/test_latmech.py` on a CPU instance
(torch+numpy, NO model, NO GPU, NO datasets). The gate that MUST pass before any GPU minute —
it proves the LOGIC (the runtime preflight, §9.3, separately proves the WIRING on real weights).

The `pg-env` AMI venv has no pytest, so when pytest is absent we inject a MINIMAL shim (only
`approx` + `raises`, the two surfaces the suite uses) and run every `test_*` function directly.
Fail-loud: a failing assertion propagates with the test name on the stack → step rc≠0 → STATUS.

    python -m infra.aws_pg.jobs.latmech_drytest --out <dir>
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import pathlib
import sys
import types


def _install_pytest_shim():
    """Inject a minimal `pytest` (approx + raises) when the real one is absent — no AMI rebuild,
    no test duplication. `raises` asserts the expected exception fired (unexpected ones propagate
    — fail-loud, not error-hiding)."""
    if importlib.util.find_spec("pytest") is not None:
        return
    shim = types.ModuleType("pytest")

    class _Approx:
        def __init__(self, v, rel=1e-6, abs=0.0):
            self.v, self.rel, self.abs = float(v), rel, abs

        def __eq__(self, other):
            return math.isclose(float(other), self.v, rel_tol=self.rel, abs_tol=self.abs)

        def __repr__(self):
            return f"approx({self.v}, rel={self.rel}, abs={self.abs})"

    @contextlib.contextmanager
    def _raises(exc):
        ok = False
        try:
            yield
        except exc:
            ok = True
        assert ok, f"expected {getattr(exc, '__name__', exc)} to be raised"

    shim.approx = lambda v, rel=1e-6, abs=0.0: _Approx(v, rel=rel, abs=abs)
    shim.raises = _raises
    sys.modules["pytest"] = shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _install_pytest_shim()
    tpath = pathlib.Path(__file__).resolve().parents[2] / "pg" / "tests" / "test_latmech.py"
    assert tpath.exists(), f"test suite not found at {tpath}"
    spec = importlib.util.spec_from_file_location("test_latmech", tpath)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    tests = sorted(n for n in dir(T) if n.startswith("test_") and callable(getattr(T, n)))
    assert tests, "no test_* functions found"
    for name in tests:
        getattr(T, name)()                 # fail-loud: any AssertionError propagates with its name
    (out / "drytest.json").write_text(json.dumps({"n_tests": len(tests), "passed": True,
                                                   "tests": tests}, indent=1))
    print(f"[LATMECH-DRYTEST] OK — {len(tests)} weight-free tests passed")


if __name__ == "__main__":
    main()
