"""MENDELEEV cell shim (N11): one Vast cell = one AWS job module, unchanged.

run_cells.py calls run(cell, out_dir); this imports cell["job"] — an infra.aws_pg.jobs module
exposing main(argv) — and calls main([*cell["args"].split(), "--out", out_dir]). It is the SAME
module the AWS lane runs under run_job.sh, so a cell's placement (AWS L40S vs Vast 80 GB) never
changes its code path. ("module" is run_cells' key for THIS shim; "job" names the science module.)

Env contract on the box:
  HF_HUB_DISABLE_XET=1        Xet wedges silently while looking healthy to every watchdog.
  online hub                  the controller ships HF_TOKEN at rent time; lifecycle.sh clears the
                              image's offline ENV when the manifest sets HF_ONLINE=1; and
                              diff_harvest.hf_online_setup() is called here BEFORE the job imports
                              transformers/datasets (the hub freezes HF_HUB_OFFLINE at import, D17).
  HF_HOME=/workspace/hf       downloads land on the rental's sized disk, not the container overlay.

Manifest cell: {"id": "rec_full", "module": "infra.vast.cells.mdl_job",
                "job": "infra.aws_pg.jobs.matrix_recovery", "args": "--row 2 --recipe full --arm rec"}
"""
import importlib
import os
import shlex
import sys


def run(cell, out_dir):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — controller.py ships it at rent time"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    argv = [*shlex.split(cell["args"]), "--out", out_dir]
    print(f"[mdl_job] {cell['id']}: python -m {cell['job']} {' '.join(argv)}", flush=True)
    importlib.import_module(cell["job"]).main(argv)


def _dryrun():
    """Weight-free: a stub job module registered in sys.modules must receive --out and its args."""
    import tempfile
    import types
    seen = {}
    stub = types.ModuleType("mdl_dryrun_stub")
    stub.main = lambda argv: seen.setdefault("argv", argv)
    sys.modules["mdl_dryrun_stub"] = stub
    os.environ.setdefault("HF_TOKEN", "hf_dryrun")
    out = tempfile.mkdtemp()
    run({"id": "dry", "module": "infra.vast.cells.mdl_job", "job": "mdl_dryrun_stub",
         "args": "--row 2 --recipe full --arm rec --smoke"}, out)
    assert seen["argv"] == ["--row", "2", "--recipe", "full", "--arm", "rec", "--smoke", "--out", out]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1" and os.environ["HF_HUB_OFFLINE"] == "0"
    print("mdl_job dry-run OK:", seen["argv"])


if __name__ == "__main__":
    _dryrun()
