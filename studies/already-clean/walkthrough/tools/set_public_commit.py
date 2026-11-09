"""Point every data link at one commit of the public repository (BUILD.md §2).

Each picture's data.js names, in meta.commit, the commit of abaybektursun/already-clean-audit that holds its source
files; site.js builds the "data" link from it. The data scripts wrote the commit of the repository they ran in, which is
not the public one, so before a publish the head runs this once with the public commit that holds the current files:

    python3 tools/set_public_commit.py <40-hex sha>

It rewrites meta.commit in every data.js under the walkthrough (the first "commit" field after "meta", and nothing
else in the file, so the data values stay byte for byte), then every hand-written link of the form
.../already-clean-audit/blob/<sha>/... in the html pages. It re-reads each data.js as JSON and asserts the new value.
It prints each file it changed and exits 1 on any file it could not handle.
"""
import json
import re
import sys
from pathlib import Path

WALKTHROUGH = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(already-clean-audit/blob/)([0-9a-f]{7,40})(/)")


def set_data_js(path, sha):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"\s*window\.\w+\s*=\s*", text)
    assert m and text.rstrip().endswith(";"), f"{path}: expected `window.NAME = {{...}};`"
    i = text.find('"meta"')
    assert i >= 0, f"{path}: no meta object"
    c = re.compile(r'("commit"\s*:\s*")([0-9a-f]{7,40})(")')
    mm = c.search(text, i)
    assert mm, f"{path}: no meta.commit"
    if mm.group(2) == sha:
        return False
    text = text[:mm.start(2)] + sha + text[mm.end(2):]
    data = json.loads(text[m.end():].rstrip()[:-1])
    assert data["meta"]["commit"] == sha, f"{path}: meta.commit did not take"
    path.write_text(text, encoding="utf-8")
    return True


def set_html(path, sha):
    text = path.read_text(encoding="utf-8")
    new, n = LINK.subn(lambda mm: mm.group(1) + sha + mm.group(3) if mm.group(2) != sha else mm.group(0), text)
    if new == text:
        return 0
    path.write_text(new, encoding="utf-8")
    return n


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]):
        print(__doc__)
        sys.exit(2)
    sha = sys.argv[1]
    changed = 0
    for p in sorted(WALKTHROUGH.rglob("data.js")):
        if set_data_js(p, sha):
            print(f"meta.commit -> {sha[:9]}  {p.relative_to(WALKTHROUGH)}")
            changed += 1
    for p in sorted(WALKTHROUGH.rglob("*.html")):
        n = set_html(p, sha)
        if n:
            print(f"{n} link(s) -> {sha[:9]}  {p.relative_to(WALKTHROUGH)}")
            changed += 1
    print(f"{changed} files changed")


if __name__ == "__main__":
    main()
