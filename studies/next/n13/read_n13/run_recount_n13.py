"""Run the registered recount (studies/next/n13/scripts/recount_n13.py) under python -I: puts the worktree on sys.path and runs the
module as __main__ with the given arguments.

    python -I run_recount_n13.py <worktree> --banks <banks dir> --verdict <verdict_n13.json>
"""
import runpy
import sys

wt = sys.argv[1]
sys.path.insert(0, wt)
sys.argv = ["recount_n13"] + sys.argv[2:]
runpy.run_module("studies.next.n13.scripts.recount_n13", run_name="__main__")
