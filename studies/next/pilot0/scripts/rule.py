"""J8 frozen rule (09_PLAN.md, Head's decisions item 4), one step at a time.

S: pad-excluded mean CE of the target batch (the batch trainer step s consumed), W: W&B train/CE loss at step s,
N: pad-excluded mean CE of the neighbour batch (the batch of step s-1), both under the same checkpoint.
Reading of "does not count if the neighbour batch is also within 0.0124": the target and the neighbour are both
within 0.0124 of W. A target outside 0.0124 is the fail branch whatever the neighbour does.
"""
import json
import sys

BAND = 0.0124  # half of the per-step sd 0.0248 of train/CE loss
MIN_ROWS = 400


def branch(S, N, W, rows_S, rows_N):
    dS, dN = abs(S - W), abs(N - W)
    if rows_S < MIN_ROWS or rows_N < MIN_ROWS:
        return "does not count branch (fewer than 400 rows scored)", dS, dN
    if dS <= BAND and dN <= BAND:
        return "does not count branch (neighbour also within 0.0124)", dS, dN
    if dS <= BAND and dN > dS:
        return "pass branch", dS, dN
    return "fail branch", dS, dN


def read(path):
    r = json.load(open(path))
    return r["pad"]["mean_ce_excluding_pad_targets"], r["rows_done"], r["complete"]


if __name__ == "__main__":
    target, neighbour, W = sys.argv[1], sys.argv[2], float(sys.argv[3])
    S, rs, cs = read(target)
    N, rn, cn = read(neighbour)
    b, dS, dN = branch(S, N, W, rs, rn)
    print(f"target {target}: S {S:.7f} rows {rs} complete {cs}")
    print(f"neighbour {neighbour}: N {N:.7f} rows {rn} complete {cn}")
    print(f"W {W!r}  |S-W| {dS:.7f}  |N-W| {dN:.7f}  ->  {b}")
