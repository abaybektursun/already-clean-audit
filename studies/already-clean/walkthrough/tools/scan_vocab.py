"""Fail on internal words in the walkthrough's reader-facing text (PLAN_PUBLIC_2026-11-08.md §2, BUILD.md §3).

Reads every .html and .js file under the walkthrough (or under the directory given), except data.js. In html it
reads text nodes, the attributes a reader sees (title, alt, aria-label, placeholder, a meta description) and the
string literals of inline scripts. In js it reads string literals: a literal holding html markup is read as html;
otherwise a literal is read when it has a space or starts with a capital letter, so class names, event names and
keys are skipped. Comments are skipped. Matching is case-insensitive on word boundaries, plurals included.

A line holding `vocab-ok` (as `<!-- vocab-ok -->` or `// vocab-ok`) is exempt: use it for the one sentence that
defines a word. Prints file, line and word for every hit, and exits 1 if there is any.

    python3 tools/scan_vocab.py [directory]
    python3 tools/scan_vocab.py --self-test     the registered inputs that must and must not fail
"""
import argparse
import hashlib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

WALKTHROUGH = Path(__file__).resolve().parent.parent

# The plan's §2 left column, then the head's list for this build. A phrase that contains a word listed alone
# ("null baseline", "zero ablation", "freeze half", "Wilson interval") is covered by that word.
WORDS = [
    # VOCAB_2026-11-10.md: internal words and the second edition's made-up replacements for standard terms.
    "null", "does not count", "test half",
    "card", "lane", "executor", "worker", "arm", "S3", "S4", "k_rec", "H_local", "amendment",
    "readout", "verdict", "transplant", "top-50 slot", "slot", "favorite",
    "freeze", "LFS", "sha256",
    "stranger", "twin", "instrument", "known number", "half precision", "full precision",
    "exemplar", "competitor", "probe", "identity pair", "failing input", "dated alike",
    "echo", "grid pole", "discovery set", "sorting model", "fake neuron",
    "pseudo-birth", "distinct-token twin",
    "snapshot", "training point", "carrier", "shared vector", "switch off", "switched off", "switching off",
    "switch-off", "switched-off", "rank agreement", "rule-setting", "chance share", "competition problem",
    "mathematics model", "fifty contexts", "16-bit", "32-bit", "birth", "born", "positive side", "negative side",
    "resampling band", "gate matrix", "up matrix", "down matrix", "weight-change statistic",
]
# "head" is matched only as a role word; "attention head(s)" is the model's own term and is allowed.
PATTERNS_HEAD = [("head (role)", re.compile(r"(?<!attention )(?<!attention-)(?<!\w)heads?(?!\w)", re.IGNORECASE))]
# Hypothesis codes (H_age, H_MIG, ...) and the internal pass/fail labels, which are upper case.
PATTERNS = [
    # "gate" as a pass/fail check; gate_proj and "gate row" of gate_proj are tensor names.
    ("gate (a check)", re.compile(r"(?<![\w_])gates?(?![\w_])(?!_proj)(?! row)", re.IGNORECASE)),
    # "unit" for a neuron; "unit vector", "in units of" and "natural units" are ordinary uses.
    ("unit (say neuron)", re.compile(r"(?<!in )(?<!natural )(?<!\w)units?(?!\w)(?! vector)(?! of (?:the|one|nats|its))", re.IGNORECASE)),
    # "side" for a pole.
    ("side (say pole)", re.compile(r"(?<!\w)(?:(?:a|the|each|its|one|own|another|control|early|late|older|other) side(?:’s|'s)?|sides)(?!\w)", re.IGNORECASE)),
    ("hypothesis code", re.compile(r"(?<![\w])H_[A-Za-z0-9]+(?![\w])")),
    ("pass/fail label", re.compile(r"(?<![\w])(?:PASS|FAIL|PASSED|FAILED)(?![\w])")),
]
# Two words are matched by their sha256 so that this file does not spell them: the internal label for a
# cancelled run (plan §2) and a word the site's owner bars from every text.
HASHED = {
    "445be54d48a2e6294369c84c61cd0929209d4e1084536159bdf7002bddfe094b": "the internal label for a cancelled run",
    "fe14010b4fe83303852f0467c919ef9a7ca089b91e96e3aad7d426dd87079297": "a word barred from every text",
}
VISIBLE_ATTRS = {"title", "alt", "aria-label", "placeholder"}
ESCAPE = "vocab-ok"


def word_regex(w):
    tail = r"(?:s|es)?" if re.search(r"\w$", w) else ""
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, w.split())) + tail + r"(?!\w)", re.IGNORECASE)


REGEXES = [(w, word_regex(w)) for w in WORDS] + PATTERNS + PATTERNS_HEAD


def hits_in(text):
    """(offset, word) for every internal word in one piece of reader-facing text."""
    found = []
    for name, rx in REGEXES:
        found += [(m.start(), name) for m in rx.finditer(text)]
    for m in re.finditer(r"[^\W\d_]+", text):
        w = m.group().lower()
        for stem in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
            if stem and hashlib.sha256(stem.encode()).hexdigest() in HASHED:
                found.append((m.start(), "<" + HASHED[hashlib.sha256(stem.encode()).hexdigest()] + ">"))
                break
    return found


# ---------- javascript: string literals ----------
REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")
PLACEHOLDER = "\ue000"  # stands for a ${...} expression; neither a word character nor a space
REGEX_KEYWORDS = {"return", "typeof", "case", "in", "of", "delete", "throw", "new", "else", "do"}


class JSLiterals:
    """Collects (line, text) for every string literal and template chunk of a script."""

    def __init__(self, src, line0=1):
        self.src, self.line, self.out = src, line0, []
        self.prev, self.prev_word = "", ""
        self.code(0, top=True)

    def code(self, i, top):
        s, n, depth = self.src, len(self.src), 0
        while i < n:
            c = s[i]
            if c == "\n":
                self.line += 1
                i += 1
            elif c in " \t\r":
                i += 1
            elif s.startswith("//", i):
                j = s.find("\n", i)
                i = n if j < 0 else j
            elif s.startswith("/*", i):
                j = s.find("*/", i + 2)
                assert j >= 0, f"unterminated comment at line {self.line}"
                self.line += s.count("\n", i, j)
                i = j + 2
            elif c in "'\"":
                j = i + 1
                while s[j] != c:
                    assert s[j] != "\n", f"unterminated string at line {self.line}"
                    j += 2 if s[j] == "\\" else 1
                self.out.append((self.line, s[i + 1:j]))
                i, self.prev, self.prev_word = j + 1, c, ""
            elif c == "`":
                i = self.template(i + 1)
                self.prev, self.prev_word = "`", ""
            elif c == "/" and (self.prev in REGEX_BEFORE or self.prev == "" or self.prev_word in REGEX_KEYWORDS):
                j, in_class = i + 1, False
                while s[j] != "/" or in_class:
                    assert s[j] != "\n", f"unterminated regular expression at line {self.line}"
                    if s[j] == "\\":
                        j += 1
                    elif s[j] == "[":
                        in_class = True
                    elif s[j] == "]":
                        in_class = False
                    j += 1
                i = j + 1
                while i < n and s[i].isalpha():
                    i += 1
                self.prev, self.prev_word = "/", ""
            elif c.isalpha() or c in "_$":
                m = re.compile(r"[\w$]+").match(s, i)
                self.prev, self.prev_word = "a", m.group()
                i = m.end()
            else:
                if c == "{":
                    depth += 1
                elif c == "}":
                    if not top and depth == 0:
                        return i + 1
                    depth -= 1
                self.prev, self.prev_word = c, ""
                i += 1
        assert top, "unterminated ${ in a template literal"
        return i

    def template(self, i):
        """One piece for the whole template, each ${...} replaced by a placeholder (plus its newlines), so that
        markup cut by an expression still parses as markup. Literals inside an expression are collected too."""
        s, buf, start = self.src, [], self.line
        while True:
            c = s[i]
            if c == "\\":
                buf.append(s[i:i + 2])
                i += 2
            elif c == "`":
                self.out.append((start, "".join(buf)))
                return i + 1
            elif s.startswith("${", i):
                before = self.line
                i = self.code(i + 2, top=False)
                buf.append(PLACEHOLDER + "\n" * (self.line - before))
            else:
                if c == "\n":
                    self.line += 1
                buf.append(c)
                i += 1


def js_pieces(src, line0=1):
    """(line, text) for the reader-facing text of a script's string literals."""
    pieces = []
    for line, lit in JSLiterals(src, line0).out:
        if re.search(r"<[A-Za-z/!][^>]*>", lit):
            pieces += html_pieces(lit, line)
        elif re.search(r"\s", lit) or re.match(r"[A-Z]", lit):
            pieces.append((line, lit))
    return pieces


# ---------- html: text nodes and visible attributes ----------
class HTMLText(HTMLParser):
    # Text inside <code>, <pre> and <a class="data-link"> is a path or an identifier, not prose; it is not scanned.
    SKIP_TAGS = ("code", "pre")

    def __init__(self, line0):
        super().__init__(convert_charrefs=True)
        self.line0, self.pieces, self.raw = line0, [], None
        self.skip = []

    def at(self):
        return self.line0 + self.getpos()[0] - 1

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for k, v in attrs:
            if v and (k in VISIBLE_ATTRS or (tag == "meta" and k == "content" and a.get("name") == "description")):
                self.pieces.append((self.at(), v))
        if tag in ("script", "style"):
            self.raw = tag
        if tag in self.SKIP_TAGS or (tag == "a" and "data-link" in (a.get("class") or "").split()):
            self.skip.append(tag)

    def handle_endtag(self, tag):
        if tag == self.raw:
            self.raw = None
        if self.skip and self.skip[-1] == tag:
            self.skip.pop()

    def handle_data(self, data):
        if self.skip:
            return
        if self.raw == "script":
            self.pieces += js_pieces(data, self.at())
        elif self.raw is None:
            self.pieces.append((self.at(), data))


def html_pieces(src, line0=1):
    p = HTMLText(line0)
    p.feed(src)
    p.close()
    return p.pieces


# ---------- the scan ----------
def scan_text(src, kind):
    """[(line, word, excerpt)] for one file's source; kind is 'html' or 'js'."""
    lines = src.splitlines()
    out = []
    for line, text in (html_pieces(src) if kind == "html" else js_pieces(src)):
        for off, word in hits_in(text):
            at = line + text.count("\n", 0, off)
            if ESCAPE in lines[at - 1]:
                continue
            excerpt = " ".join(text[max(0, off - 30):off + 40].split())
            out.append((at, word, excerpt))
    return sorted(set(out))


def files(root):
    # notebook/ holds the first edition's text verbatim, dated; its words are its own and are not scanned.
    return sorted(p for p in root.rglob("*") if p.suffix in (".html", ".js") and p.name != "data.js" and p.is_file()
                  and "notebook" not in p.parts)


def self_test():
    cancelled = "".join(map(chr, [118, 111, 105, 100]))
    cases = [  # (kind, source, expected [(line, word)])
        ("html", "<p>The verdict was clear.</p>", [(1, "verdict")]),
        ("html", "<p>ok</p>\n<p>Each snapshot has fifty.</p>", [(2, "snapshot")]),
        ("html", "<p>Two snapshots.</p>", [(1, "snapshot")]),
        ("html", "<p>The verdict, defined once. <!-- vocab-ok --></p>", []),
        ("html", '<div class="label readout"><label for="x">Neuron</label></div>', []),
        ("html", '<span title="the gate result">x</span>', [(1, "gate (a check)")]),
        ("html", '<p>See <code>verdict_n13.json</code> and <a class="data-link" href="x">studies/next/n6/verdict_n6.json</a>.</p>', []),
        ("html", '<p>A twin, an instrument, the sorting model.</p>', [(1, "instrument"), (1, "sorting model"), (1, "twin")]),
        ("html", '<p>32 attention heads; the head decided.</p>', [(1, "head (role)")]),
        ("html", "<script>const s = 'a readout here'; // a verdict in a comment\n</script>", [(1, "readout")]),
        ("html", f"<p>It was marked {cancelled.upper()}.</p>", [(1, "<the internal label for a cancelled run>")]),
        ("html", "<p>H_age held; the check reads PASS.</p>", [(1, "hypothesis code"), (1, "pass/fail label")]),
        ("js", "const re = /'/g;\nconst s = 'the null result';", [(2, "null")]),
        ("js", "el.className = 'label';\nb.setAttribute('aria-label', x);", []),
        ("js", "root.innerHTML = `\n<div class=\"label\">Training step</div>\n<h3>The readout</h3>`;", [(3, "readout")]),
        ("js", "const t = `Step ${n.map((x) => 'one arm').join()} done`;", [(1, "arm")]),
        ("js", "h = `<div class=\"jd-head\" id=\"jd-head${i}\">\n${a +\n b}<p>The readout</p></div>`;", [(3, "readout")]),
        ("js", "const x = a / b; const s = 'Carrier rank';", [(1, "carrier")]),
    ]
    bad = 0
    for kind, src, want in cases:
        got = [(line, word) for line, word, _ in scan_text(src, kind)]
        if got != want:
            bad += 1
            print(f"self-test FAILED: {kind} {src!r}: got {got}, expected {want}")
    print(f"self-test: {len(cases) - bad} of {len(cases)} cases as registered")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", type=Path, default=WALKTHROUGH)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    root = a.root.resolve()
    assert root.is_dir(), f"{root} is not a directory"
    total, per_word = 0, {}
    scanned = files(root)
    for p in scanned:
        for line, word, excerpt in scan_text(p.read_text(encoding="utf-8"), "html" if p.suffix == ".html" else "js"):
            print(f"{p.relative_to(root)}:{line}: {word}  | {excerpt}")
            total += 1
            per_word[word] = per_word.get(word, 0) + 1
    summary = ", ".join(f"{w} {n}" for w, n in sorted(per_word.items(), key=lambda kv: -kv[1]))
    print(f"\n{len(scanned)} files scanned under {root}; {total} hits" + (f": {summary}" if total else ""))
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
