# Building a picture for the walkthrough

The contract is `BUILD.md`. This note says how to follow it. `neuron_across_training/` is the worked example.

## What is shared

- `site.css`: the tokens, the type scale, the page column (prose 68ch, a picture up to 960px, 16px gutters on a
  phone), chapters, the chapter list, the picture frame (`.figure`, `.figure-caption`, `.data-link`), the value box,
  controls (`.button`, `.toggle`, `input.scrub`; each at least 40px tall), tables (`.table-scroll` and `.data-table`)
  and footnotes. Each block of `site.css` starts with a comment that shows its markup.
- `site.js`: `WALK.register(id, {data, mount})` (data, the name of the global data.js sets, is optional),
  `WALK.mountAll()` (it runs by itself once the page has loaded),
  `WALK.fmt` (`int`, `fixed`, `percent`, `interval`; thousands separators and a true minus sign; it throws on NaN),
  `WALK.valueBox` and `WALK.hover` (hover, tap and keyboard focus), `WALK.dataLink`, and the theme switch
  (`<button class="button" data-theme-toggle>`).

## Colours

Use only the tokens. The same meaning keeps the same colour on every page.

| token | meaning | light | dark |
|---|---|---|---|
| `--ink` | the thing measured; text | `#18211f` | `#e3e9e6` |
| `--claim` | the claim or the carrier | `#0792b1` | `#0388a5` |
| `--control` | the control or the baseline | `#c9690c` | `#cb7229` |
| `--muted` | everything else: secondary text, axes, other marks | `#56625e` | `#9aa7a3` |
| `--bar`, `--grid`, `--rule` | filled marks that are none of the above, gridlines, hairlines | | |
| `--accent` | links and the focus ring only | `#0a6b5d` | `#4cc3ae` |

The dataviz validator (`validate_palette.js`, two slots, all pairs, surface `--bg`) passes both themes: light, worst
colour-blind ΔE 19.4 and normal-vision ΔE 25.9, contrast 3.36 and 3.51 to 1; dark, 16.8 and 25.0, contrast 4.45 and
5.25 to 1. Measured the same way, `--claim` and `--control` stay at ΔE 30.1 or more from `--ink` and 12.2 or more from
`--muted`, in both themes, for normal, protan and deutan vision. `--accent` is not a chart colour: it fails the
validator (chroma 0.084 in light, lightness 0.744 in dark). Never put text in a series colour; label a coloured mark in
`--ink` or `--muted`, and name it in a legend or a direct label as well, so colour is never the only cue.

## A new picture folder

1. `extract.py` reads the banked files, writes `data.js`, and prints `path sha256` for every input. The same pairs go
   into `meta.sources`. A path is repository-relative when the file is in the repository, an `s3://` URI when it is
   only on S3, or an absolute path for a file only on the laptop. `meta.commit` is the commit of the public repository
   (`abaybektursun/already-clean-audit`) that holds the files; the data link opens the first repository-relative
   source, or `meta.link` when no source is in the repository.
2. `<id>.js` registers a mount. It reads nothing but its data object and `WALK`; no framework, no network; nothing
   moves unless the reader acts. Scope every css rule under `#<id>` in `<id>.css`.
3. `index.html` is the standalone page, below. The caption is one sentence, the one in the plan's §4. Every number in
   it must be in `data.js`; `data-source` on the figure makes `tools/check_numbers.py` check it there.
4. Check at 360px and at 1,280px, in both themes, with the fonts blocked; then run the tools.

```html
<!-- <id>/index.html. Doctype, charset, viewport, title and the optional font stylesheet: as in neuron_across_training/index.html -->
<link rel="stylesheet" href="../site.css"><link rel="stylesheet" href="<id>.css">
<main class="page">
  <header class="page-head"><h1>Title</h1>
    <nav aria-label="Site"><a href="../index.html">The walkthrough</a>
      <button class="button" type="button" data-theme-toggle>Dark theme</button></nav></header>
  <figure class="figure" data-source="data.js">
    <div id="<id>"></div>
    <figcaption class="figure-caption">The sentence from the plan. <a class="data-link">data</a></figcaption>
  </figure>
</main>
<script src="../site.js"></script><script src="data.js"></script><script src="<id>.js"></script>
```

```js
// <id>/<id>.js
(() => {
  'use strict';
  function mount(root, D) {
    for (const r of D.rows) {
      const bar = document.createElement('div');
      bar.className = 'x-bar ' + r.kind; // in <id>.css: .x-bar.claim { background: var(--claim) }
      bar.style.width = (100 * r.value) / D.max + '%';
      WALK.hover(bar, { title: r.name, rows: [['value', WALK.fmt.fixed(r.value, 3), r.kind]] });
      root.appendChild(bar);
    }
  }
  WALK.register('<id>', { data: 'X_DATA', mount });
})();
```

```js
// <id>/data.js, written by extract.py
window.X_DATA = {"rows": [...], "max": ...,
  "meta": {"commit": "<public repository sha>", "sources": [{"path": "studies/next/m1/verdict_m1.json", "sha256": "..."}]}};
```

For a dense picture (hundreds of marks), do not make each mark focusable: attach one pointer listener to the plot,
find the mark under the pointer, call `WALK.valueBox.show(content, x, y)`, and move through the marks with the arrow
keys. For a time axis, use `<input class="scrub" type="range">`; it is keyboard-ready.

## Checks

- `python3 tools/scan_vocab.py` fails on any word from the plan's §2 left column in html and js text. A line that
  defines a word carries `<!-- vocab-ok -->` or `// vocab-ok`.
- `python3 tools/check_numbers.py` lists every number in `index.html` and in each `.figure-caption` that no data
  file holds. Numbers outside any `data-source` are checked against all data.js files together, which almost always
  match; put `data-source` on each figure and chapter. Wrap a name that holds digits ("Opus 5.5") in `data-nocheck`.
- `tools/build_pages.sh <empty directory>` copies the html, css, js and images, runs both tools' self-tests, then both
  tools on the copy, and stops at the first failure.
- `python3 tools/set_public_commit.py <sha>` points every `meta.commit` and hand-written data link at one commit of the
  public repository; the head runs it once before a publish (BUILD.md §2).
- On the article page a mount carries `data-load="<id>/<id>.css <id>/data.js <id>/<id>.js"` and nothing is loaded until
  the mount nears the viewport; a standalone page keeps its script tags.
