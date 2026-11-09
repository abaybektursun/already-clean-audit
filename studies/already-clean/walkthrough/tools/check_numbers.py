"""List every number in the walkthrough's prose and captions that no data file holds (BUILD.md §4).

Reads all text of index.html and the .figure-caption of every other page. Each number is looked up in the files
named by the data-source attribute of its nearest enclosing element; with no data-source in scope, in every data.js
under the walkthrough. A data-source value is one or more paths separated by spaces, each resolved against the html
file's folder and then against the repository root (--repo).

A number written with d decimals matches a value that rounds to it at d decimals; a percentage also matches the
fraction (59.3% matches 0.593). Signs are ignored. In a data.js or .json file the values are its numbers and the
length of each array and object; in any other file, every number in its text.

Not numbers: dates and times ("24 October", "20 Oct 18:40Z", "November 2026"), digits joined to letters ("7B",
"L03N6372+", "GSM8K"), digits after a letter and a hyphen ("OLMo-3", "top-50"), digits after § or #, the text of
script, style, code, pre, kbd and title, and any element with data-nocheck.

Prints every number found nowhere and exits 1 if there is any. The union of every data.js holds hundreds of
thousands of values, so a short number almost always matches it: the summary counts how many numbers were checked
only against that union. A data-source on the chapter or the figure makes the check strict.

    python3 tools/check_numbers.py [directory] [--repo REPO]
    python3 tools/check_numbers.py --self-test     the registered inputs that must and must not fail
"""
import argparse
import json
import math
import re
import subprocess
import sys
from bisect import bisect_left
from html.parser import HTMLParser
from pathlib import Path

WALKTHROUGH = Path(__file__).resolve().parent.parent

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
DATES = re.compile(
    rf"\b\d{{1,2}}\s+(?:{MONTHS})\b\.?(?:,?\s+\d{{4}}\b)?"     # 24 October, 4 Nov 2026
    rf"|\b(?:{MONTHS})\.?\s+\d{{1,2}}\b(?:,\s*\d{{4}}\b)?"    # October 24, 2026
    rf"|\b(?:{MONTHS})\s+\d{{4}}\b"                          # November 2026
    r"|\b\d{1,2}:\d{2}(?::\d{2})?Z?\b")                       # 18:40Z
NUMBER = re.compile(r"(?<![\w.,§#])(?<![A-Za-z]-)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\.?\w)(\s?%)?")
EMPTY_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP = {"script", "style", "code", "pre", "kbd", "title", "template"}


# ---------- values ----------
def json_values(obj, out):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(abs(float(obj)))
    elif isinstance(obj, list):
        out.append(float(len(obj)))
        for x in obj:
            json_values(x, out)
    elif isinstance(obj, dict):
        out.append(float(len(obj)))
        for x in obj.values():
            json_values(x, out)


def text_values(text):
    return [abs(float(m.group(1).replace(",", "") + (m.group(2) or ""))) for m in NUMBER.finditer(text)]


def file_values(path):
    """Sorted absolute values held by one data file."""
    text = path.read_text(encoding="utf-8")
    out = []
    if path.name == "data.js":
        m = re.match(r"\s*window\.\w+\s*=\s*", text)
        assert m and text.rstrip().endswith(";"), f"{path}: expected `window.NAME = {{...}};`"
        json_values(json.loads(text[m.end():].rstrip()[:-1]), out)
    elif path.suffix == ".json":
        json_values(json.loads(text), out)
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                json_values(json.loads(line), out)
    else:
        out = text_values(text)
    return sorted(out)


def found_in(number, frac, pct, values):
    n = float(number.replace(",", "") + (frac or ""))
    half = 0.5 * 10 ** -(len(frac) - 1 if frac else 0) * (1 + 1e-9)
    for target, h in [(n, half)] + ([(n / 100, half / 100)] if pct else []):
        i = bisect_left(values, target - h)
        if i < len(values) and values[i] <= target + h:
            return True
    return False


# ---------- html ----------
class Numbers(HTMLParser):
    """(line, number text, decimals, percent, data-source names or None, excerpt) for each number in scope."""

    def __init__(self, whole):
        super().__init__(convert_charrefs=True)
        self.whole, self.stack, self.found = whole, [], []

    def handle_starttag(self, tag, attrs):
        if tag not in EMPTY_TAGS:
            self.stack.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        for k in range(len(self.stack) - 1, -1, -1):
            if self.stack[k][0] == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        if any(t in SKIP or "data-nocheck" in a for t, a in self.stack):
            return
        if not self.whole and not any("figure-caption" in (a.get("class") or "").split() for _, a in self.stack):
            return
        sources = next((a["data-source"].split() for _, a in reversed(self.stack) if a.get("data-source")), None)
        line0 = self.getpos()[0]
        clean = DATES.sub(lambda m: " " * len(m.group()), data)
        for m in NUMBER.finditer(clean):
            excerpt = " ".join(data[max(0, m.start() - 40):m.end() + 30].split())
            self.found.append((line0 + data.count("\n", 0, m.start()), m.group(1) + (m.group(2) or ""),
                               m.group(1), m.group(2), bool(m.group(3)), sources, excerpt))


def numbers_in(html, whole):
    p = Numbers(whole)
    p.feed(html)
    p.close()
    return p.found


def check(html, whole, values_for, union):
    """[(line, number, scope, matched, excerpt)]; values_for(names) returns sorted values for data-source names."""
    out = []
    for line, shown, number, frac, pct, sources, excerpt in numbers_in(html, whole):
        values = union if sources is None else values_for(tuple(sources))
        scope = "union" if sources is None else " ".join(sources)
        out.append((line, shown + ("%" if pct else ""), scope, found_in(number, frac, pct, values), excerpt))
    return out


def self_test():
    union = []
    json_values({"share": 0.2654, "lo": 0.2112, "hi": 0.3207, "n": 20560068, "neg": -0.3, "steps": [0] * 16}, union)
    union.sort()
    named = {("x.json",): [7.25]}
    cases = [  # (whole, html, numbers expected to be found nowhere)
        (True, "<p>We scored 20,560,068 tokens at 16 points; the share was 0.265 [0.211, 0.321], or 26.5%, on "
               "24 October 2026 at 18:40Z, with OLMo-3 7B (L03N6372+, §3.1); 0.27 rounds to it and 0.28 does not; "
               "−0.30 is the sign case, and 99,999 strangers.</p>", ["0.28", "99,999"]),
        (False, '<p>Not a caption: 4,321.</p><figure class="figure" data-source="x.json"><div id="p"></div>'
                '<figcaption class="figure-caption">7.25 of 0.265 <code>123</code> <span data-nocheck>Opus 5.5'
                '</span></figcaption></figure>', ["0.265"]),
        (True, "<style>p { margin: 0 0 1rem; }</style><title>3 things</title><p>Sentence ends at 0.246.</p>",
         ["0.246"]),
    ]
    bad = 0
    for whole, html, want in cases:
        got = [num for _, num, _, ok, _ in check(html, whole, lambda names: named[names], union) if not ok]
        if got != want:
            bad += 1
            print(f"self-test FAILED: {html[:60]!r}...: found nowhere {got}, expected {want}")
    print(f"self-test: {len(cases) - bad} of {len(cases)} cases as registered")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", type=Path, default=WALKTHROUGH)
    ap.add_argument("--repo", type=Path, help="repository root for data-source paths (default: git toplevel)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    root = a.root.resolve()
    assert root.is_dir(), f"{root} is not a directory"
    repo = a.repo.resolve() if a.repo else None
    if repo is None:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        repo = Path(r.stdout.strip()) if r.returncode == 0 else None

    data_files = sorted(root.rglob("data.js"))
    union = sorted(v for p in data_files for v in file_values(p))
    cache = {}

    def values_for(page):
        def get(names):
            vals = []
            for name in names:
                cands = [page.parent / name] + ([repo / name] if repo else [])
                path = next((c.resolve() for c in cands if c.is_file()), None)
                assert path, f"{page.relative_to(root)}: data-source {name} not found (looked in " \
                             f"{', '.join(str(c) for c in cands)})"
                if path not in cache:
                    cache[path] = file_values(path)
                vals += cache[path]
            return sorted(vals)
        return get

    pages = [root / "index.html"] if (root / "index.html").is_file() else []
    pages += [p for p in sorted(root.rglob("*.html")) if p != root / "index.html"]
    n_named = n_union = 0
    missing = []
    for page in pages:
        for line, number, scope, ok, excerpt in check(page.read_text(encoding="utf-8"), page == root / "index.html",
                                                       values_for(page), union):
            if not ok:
                missing.append(f"{page.relative_to(root)}:{line}: {number}  [{scope}]  | {excerpt}")
            elif scope == "union":
                n_union += 1
            else:
                n_named += 1
    for m in missing:
        print(m)
    total = n_named + n_union + len(missing)
    print(f"\n{total} numbers in {len(pages)} pages: {n_named} found in a named data-source, {n_union} found only in "
          f"the union of {len(data_files)} data.js files ({len(union):,} values; no data-source in scope), "
          f"{len(missing)} found nowhere")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
