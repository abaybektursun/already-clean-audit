"""Refill SPEC_D1 §11's table from bank/w5/verdict/grid.md and say which cells moved.

usage: python3 -I fill_spec11.py <SPEC_D1.md> <grid.md> [--write]
The instrument writes grid.md with the same six columns as §11. The eight data rows of §11 (between the header separator and
the blank line before "## 12.") are replaced verbatim by grid.md's rows, matched by their readout cell; every readout must
match in order. Without --write nothing is changed: the differing cells are printed (row, column, old -> new) so the §11 note
can say what moved. With --write the rows are replaced and the same report printed.
"""
import re
import sys
from pathlib import Path

COLS = ["readout", "H_RG predicts", "H_SP predicts", "H_VEST predicts", "observed", "label"]


def rows(text):
    out = []
    for line in text.split("\n"):
        if line.startswith("| ") and not line.startswith("| readout") and not line.startswith("|---"):
            out.append([c.strip() for c in line.strip().strip("|").split(" | ")])
    return out


def main(spec_path, grid_path, write=False):
    spec, grid = Path(spec_path), Path(grid_path)
    text = spec.read_text()
    start = text.index("## 11. Prediction grid")
    end = text.index("## 12.", start)
    section = text[start:end]
    new_rows = rows(grid.read_text())
    old_rows = rows(section)
    assert len(new_rows) == len(old_rows) == 8, (len(new_rows), len(old_rows))
    assert [r[0] for r in new_rows] == [r[0] for r in old_rows], "readout order differs"
    moved = []
    for o, n in zip(old_rows, new_rows):
        assert len(o) == len(n) == 6, (o[0], len(o), len(n))
        for k, (a, b) in enumerate(zip(o, n)):
            if a != b:
                moved.append((o[0], COLS[k], a, b))
    for row, col, a, b in moved:
        print(f"MOVED {row} [{col}]:\n   old: {a}\n   new: {b}")
    print(f"{len(moved)} cell(s) differ across {len({m[0] for m in moved})} row(s)")
    if write and moved:
        grid_lines = [l for l in grid.read_text().split("\n") if l.startswith("| ") and not l.startswith("| readout") and not l.startswith("|---")]
        sec_lines = section.split("\n")
        idx = [i for i, l in enumerate(sec_lines) if l.startswith("| ") and not l.startswith("| readout") and not l.startswith("|---")]
        assert len(idx) == 8
        for i, l in zip(idx, grid_lines):
            sec_lines[i] = l
        spec.write_text(text[:start] + "\n".join(sec_lines) + text[end:])
        print("SPEC §11 rows replaced")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--write" in sys.argv[3:])
