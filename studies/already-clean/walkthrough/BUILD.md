# Build contract for the public walkthrough, second edition

Written 2026-11-08 by the head. Every builder, writer and checker follows this file. The plan is
`PLAN_PUBLIC_2026-11-08.md` (chapters, mechanisms, pictures, vocabulary). The first edition's conventions are in
`VISUALS_PLAN.md` and in the seven existing picture folders; this edition keeps them and adds the contracts below.

## 1. Layout

```
studies/already-clean/walkthrough/
  index.html            the article: eleven chapters, pictures mounted inline, one reading column
  site.css              the one stylesheet (tokens, type, layout, picture frame, caption, controls)
  site.js               shared helpers: mount registry, number formatting, hover value box, theme
  notebook/index.html   the first edition's chronology, moved verbatim, dated
  <picture>/            one folder per picture (existing seven; new m01 ... m13, alwayson, f08 ... f13)
    extract.py          reads the banked files, writes data.js; prints each input's sha256
    data.js             window.<ID>_DATA = {...}; includes meta.sources = [{path, sha256}], meta.commit
    <picture>.js        builds the picture inside the element with id "<picture>"; reads window.<ID>_DATA
    index.html          the standalone page: site.css, the caption, the data link, the same mount
  tools/
    scan_vocab.py       fails on any word from the plan's §2 left column in html/js text
    check_numbers.py    lists every number in index.html captions and finds it in a data.js or a named file
    build_pages.sh      copies the site to a clean gh-pages tree (no .py, no .md, no plans)
    set_public_commit.py points every data link at one commit of the public repository, before a publish
```

Existing pages keep their folder names and ids. Each existing page is retrofitted to `site.css` by replacing its own
tokens block with a link to `site.css`; its layout rules stay in its own css file only where they are specific to it.

## 2. The picture contract

- One picture, one folder, one `extract.py`, one `data.js`, one script, one standalone page.
- `extract.py` takes no network and no GPU. It reads only files in this repository (or a file the plan names on S3,
  fetched once by the head and committed). It writes `data.js` and prints `path sha256` for every input. The same
  shas go into `data.js` under `meta.sources`, with `meta.commit` = the commit of the public repository
  (`abaybektursun/already-clean-audit`) that holds the source files, so the data link resolves; a source that is not
  in the repository is named by its `s3://` URI, and `meta.link` names the page the data link opens instead.
  The data scripts cannot know the public commit, so before a publish the head pushes `main` to the public
  repository and runs `tools/set_public_commit.py <sha>` once, which rewrites every `meta.commit` and every
  hand-written `blob/<sha>/` link to that commit without touching any data value.
- The script builds into `document.getElementById('<picture>')`. If `data.js` is missing it writes "The data file did
  not load" and the data path into the mount, and throws; it never tells a reader to run a script. It uses no framework
  and no network. It reads nothing but its data object and the shared helpers.
- On the article page a mount names its files, `<div id="m01" data-load="m01/m01.css m01/data.js m01/m01.js">`, and
  `site.js` loads them in that order when the mount comes within one screen of the viewport, not at page load (the
  first edition loaded 5.33 MB up front); then it mounts every registered picture on the page. A file that fails to
  load writes its path into the mount and throws. The standalone page loads its own files with script tags.
- Fonts are the one network request allowed (Google Fonts, as in the first edition); every page must read correctly in
  the system fallback.
- Interaction: hover or tap shows the exact value and the item's name; a drag moves through snapshots where there is
  a time axis; one toggle switches one condition where the plan names one. Nothing animates on its own.
- Every number drawn is a datum from `data.js`. Nothing is illustrative.
- The caption is one sentence that states what the picture shows (the sentence from the plan's §4), then a "data"
  link to the source file's path at `meta.commit` in the public repository.
- Phone first: the picture is legible at 360 px wide; controls are at least 40 px tall; text is at least 14 px, except
  axis tick labels, which may be 12 px (the exception the second edition's pictures use; the seven first-edition pictures
  keep their own sizes and controls).
- Colours come from `site.css` tokens only: `--ink` for the thing measured, `--claim` for the claim or the carrier,
  `--control` for the control or the baseline, `--muted` for everything else; `--accent` is for links and the focus
  ring only (it fails the chart-colour validator). The same meaning keeps the same colour on every page. Never put
  text in a series colour; colour is never the only cue. `README_BUILD.md` has the token table and the worked example.
- Light and dark both work; the existing pages' token names are kept.

## 3. The article contract

- One reading column, 68 characters wide for prose; a picture may widen to 960 px.
- Chapters 1 to 11 in the order of the plan's §3, each with the same five parts under one `h2`: the question, what we
  did, the picture with its caption, what we decided and the record's reason, what it left open. The part names are not
  shown as headings; the prose carries them. Chapter 3 states once that the November studies were chosen together;
  the five causal transitions the plan's §1 names are the only places the prose says one result led to the next.
- Mechanism pictures (M1 to M13) sit in a chapter 0, "What the operations are", and are linked from the first use
  of each term. A reader who knows the operation can skip chapter 0.
- Each chapter carries the definition boxes of the plan's §3c as `<aside class="def" data-source="<files>">`: one
  formula line in `<code>` and one or two sentences; the sentence's numbers are checked against the named files.
- A first-edition picture keeps its own text ("neuron", "training point"); its page says so in a
  `<p class="figure-note">` under the caption, and the article says so once in its header.
- Chapter 11 ends with the open list of the plan's §3b, each item with its date and file.
- A number appears in prose only when it is also in a picture or a table on the page.
- Every date is written once per chapter as "24 October".
- Internal words never appear (plan §2). `tools/scan_vocab.py` runs before every publish and fails the build. It scans
  prose text nodes and the visible attributes (`title`, `alt`, `aria-label`, `placeholder`) of html, and the string
  literals of js. It skips `code` and `pre` elements, the text of `a.data-link`, other attribute values, and the
  `notebook/` folder (the first edition's text, dated, verbatim). A line that defines a word once carries
  `<!-- vocab-ok -->` or `// vocab-ok`. "head" is flagged only as a role word; "attention heads" passes. "label" is
  on the list; an axis label is markup, not the word.
- `tools/check_numbers.py` checks every number in `index.html` and in each `.figure-caption` against the data files;
  a figure or chapter with `data-source` is checked against that file alone, which is the only strict form (the union
  of all data.js files holds over 130,000 values). Wrap a name that holds digits ("Opus 5.5") in `data-nocheck`.
- Each chapter's "we decided because" sentence has a `data-journal` attribute naming the dated journal entry it
  rests on, so the checker can find it.

## 4. Provenance and checks

- `tools/check_numbers.py` extracts every number from captions and prose and finds it in a `data.js` or in a file
  named by a `data-source` attribute; a number found nowhere fails the build.
- The fresh reader's check (plan §5 step 4) is run on the built site, not on the source: every number against its
  file, every caption against its data, every decision against its journal entry, the vocabulary scan, and a read at
  phone width.
- The gh-pages tree holds only what the browser needs: html, css, js, data.js, images. No python, no markdown, no
  plans, no audits.

## 5. Order of work

1. `site.css`, `site.js`, the retrofit of one existing page as the example, and `tools/` (one worker).
2. `extract.py` and `data.js` for every new picture, from the files the plan names (one worker; fetch from S3 only
   what the plan names, once).
3. The new pictures, two workers at a time, each in its own worktree, each one folder.
4. The retrofit of the remaining existing pages to `site.css`, with the plan's §4 caption corrections: "600 sides of
   300 units" on the births and one-job pages, the chance level and the 5 November retag on the births page and the hero.
5. The prose (the head), then `index.html` assembled with the mounts.
6. The checks; then push `main` to the public repository, run `tools/set_public_commit.py` with that commit, commit
   and push again; then `tools/build_pages.sh` and the publish of the built tree to `gh-pages` of both repositories.

Each worker commits named files on its own branch and reports the commit; the head merges. One writer per file.
