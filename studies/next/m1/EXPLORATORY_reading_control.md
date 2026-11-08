Dated exploratory analysis, 2026-11-05, on the closed and unblinded M1 result (`verdict_m1.json`, `READ_head.md`). No rule here decides a headline.

# M1 reading control: does the reading of L04N3542− pick out that pole?

## The question

`READ_head.md` reads pole L04N3542− (layer 4, neuron 3542, negative sign) from its top contexts in `raw_examples_m1.md`:
its strongest positions are the tokens `erg`, `gent`, `-div`, `.prod`, `.Cross` in "divergence", "convergent", "cross
product", `numpy.prod`, and it is "a token-level feature that occurs in web, PDF and code documents alike". A reading of
top contexts can describe what any strongly active pole looks like (`12_SHAPING_NOTE.md` §3, §4 item 3). This control
reads five random layer-4 poles of matched strength beside it, blind to which label is which.

## Selection (from `studies/next/m1/banks/start_a/samples/contexts.jsonl`)

The file holds one row per grid pole (640 rows, 20 per layer), with `pole`, `max_abs` (the largest |a| over the
corpus for that sign), quantiles, and `topk`: the 50 strongest positions with `before`, `token`, `after`, `a`.

- Target: L04N3542−, `max_abs` 2.5264. Window, a factor of 2: [1.2632, 5.0529].
- The other 19 layer-4 grid poles, with `max_abs`: L04N802+ 2.2761, L04N802− 1.2708, L04N2801+ 2.3223, L04N2801− 1.5039,
  L04N3542+ 2.7241, L04N3936+ 1.0515 (outside), L04N3936− 1.9778, L04N5845+ 1.0244 (outside), L04N5845− 2.4270,
  L04N6289+ 3.0024, L04N6289− 1.7270, L04N6542+ 1.0265 (outside), L04N6542− 3.6057, L04N7402+ 1.1077 (outside),
  L04N7402− 1.8217, L04N7919+ 1.9642, L04N7919− 2.3451, L04N10274+ 1.3159, L04N10274− 1.8760. Fifteen qualify; the
  factor did not need widening.
- `default_rng(0).choice(15, 5, replace=False)` over the fifteen in file order gave indices 7, 6, 3, 4, 9:
  **L04N6289+, L04N5845−, L04N2801−, L04N3542+, L04N6542−**. One of them, L04N3542+, is the opposite sign of the
  target's own neuron; the draw is kept as it fell.
- The six (five in draw order, then the target) were shuffled with `default_rng(1).permutation(6)` and labelled A to F.
  The script (`select_and_blind.py`, in the session scratchpad, reproduced below under Method) wrote the top-20 contexts
  of each label without pole names or activation values, and wrote the map to a separate file that was not opened until
  the six sentences below were written and committed.

**Blinding was partial.** The reader (this agent) knew the target's tokens from the brief and from `READ_head.md`
before reading. Recognizing the target among A to F is therefore not evidence of anything. The test is whether the five
other sentences are as specific as the target's.

## Six blind sentences (top-20 contexts of each label, written before the map was opened)

- **A.** The token that completes formal-methods and safety-engineering terms — " verification", " refinement",
  " validation", " specifications", "ritical" in "safety-critical", " methods" in "formal methods" — in papers and
  calls for papers on formal verification of safety-critical systems; a topic-level feature with a few outliers
  (" Pad" in "Launch Pad", "och" in "Grady Booch", " VH" in VHDL).
- **B.** The second token of consumer software and service names — " Music" in "Apple Music", "mail" in "Gmail",
  "Drive" in "OneDrive", "rello" in "Trello", "Telegram" — in web text and one Python scope string that list apps and
  cloud services; a token-level feature for app and service names.
- **C.** Tokens in cryptocurrency and blockchain text — " Coin" in CoinMarketCap, Coin Center, CoinGPT, CoinEx;
  " Silk" in Silk Road; " Exodus"; "amoto" in Nakamoto; "/state" in state-channel URLs; crypto figures' names — with
  a few code tokens ("helpers" in an import path, `_Address` in Python type stubs); a topic-level crypto feature.
- **D.** No single topic: three token families — "it" in "Prioritize/Prioritise" (and "itet" in Georgian "Prioriteti"),
  "keywords"/"stopwords"/"Keyword"/Ukrainian "слова" in "Ключові слова", and " Hot" in casino slot names — plus
  "ilight"/"lig" in misspelled "highlight", across web text, Python code and non-English text.
- **E.** Word-pieces of the div-/-ergent family — "erg" and "-div" in "divergence", "divergent", "KL-divergence",
  "φ-divergences", "neuro-divergent"; "gent" in "divergent"/"convergent" plate boundaries; "ver" in "converged" — and of
  products: ".cross(" in C# code, "numpy.prod", "Cross Product", plus " logarith" in "logarithm"; spread over math text,
  geology, code and everyday prose; a token-level feature.
- **F.** The brand name Motorola — "ola" after "Motor" and the whole token " Motorola" — in web text about mobile phones
  and two-way radios, with one outlier (" HTC"); a single-name token-level feature.

## Reveal (appended after the sentences above were committed in bd3e29da0)

Map from `studies/next/m1/reading_control/map.json` (`default_rng(1).permutation(6)` = 4, 0, 2, 1, 5, 3):

| label | pole | `max_abs` | blind reading, in brief | top-50 slots at start_a: web / PDF / code / math |
|---|---|---|---|---|
| A | L04N6542− | 3.6057 | formal-methods and safety-engineering terms; topic-level | 50 / 0 / 0 / 0 |
| B | L04N6289+ | 3.0024 | second tokens of app and service names (Apple Music, Gmail, OneDrive, Trello); token-level | 47 / 0 / 2 / 1 |
| C | L04N2801− | 1.5039 | cryptocurrency and blockchain text, a few code tokens; topic-level | 47 / 0 / 3 / 0 |
| D | L04N5845− | 2.4270 | three token families: "priorit-", "keyword(s)/stopwords", " Hot"; token-level, no topic | 36 / 6 / 7 / 1 |
| **E** | **L04N3542− (target)** | 2.5264 | div-/-ergent word-pieces and vector or array products; token-level | 35 / 4 / 8 / 3 |
| F | L04N3542+ | 2.7241 | the brand name Motorola; token-level | 48 / 2 / 0 / 0 |

The last column is from `banks/start_a/m1/counts.npz` summed by the file's `groups` (`target_g_m_output.txt`, part 3).

Two notes on the sentences, made after the reveal and not edited into them: the `.cross(` context in E is in a
`stack_edu-CSharp` document (`raw_examples_m1.md`, start_a rank 2), as the sentence guessed; the language names in D
("Georgian", "Ukrainian") are guesses from the text and were not checked. The opposite pole of the target's own neuron
(F) reads as something unrelated to the target.

## L04N3542− under G and M (`banks/{start_a,G,M}/samples/contexts.jsonl`, `banks/{start_a,G,M}/m1/counts.npz`)

Output: `studies/next/m1/reading_control/target_g_m_output.txt`, from `target_g_m.py`. Token families are counted over
each checkpoint's top-50 positions with the sets written in the script: the diverge/converge family (`erg`, `-div`,
`gent`, ` conver`, ` divergence`, `iver`, `ver`, …) and the product family (any token containing prod, cross, dot or
ultiply, except ` dotted`). Activation values are compared only as ranks or within one checkpoint, since the scale differs between models.

| checkpoint | cm_share | max \|a\| | diverge/converge family | product family | other | start's top-50 positions still in the top-50 |
|---|---|---|---|---|---|---|
| start_a | 0.22 | 2.526 | 37 / 50 | 11 / 50 | 2 (` logarith`, `.div`) | 50 |
| G (0 % code+math) | 0.42 | 1.956 | 27 / 50 | 22 / 50 | 1 (`.div`) | 24 |
| M (65 % code+math) | 0.64 | 1.604 | 6 / 50 | 40 / 50 | 4 (`.div`, `DIV`, `.Div`: division operations; ` dotted` in a music text) | 13 |

- Under G the same tokens still lead: `erg` in "KL-divergence" and "φ-divergences" holds ranks 0, 1, 3, 5, 6.
  `cross`, `.dot`, `.Cross` and ` product` take 8 of the other 15 top-20 ranks, mostly in C# vector code.
- Under M they do not. 17 of the top-20 are product, dot or cross operations (`.Cross`, `.cross(`, `numpy.prod`,
  `(dot`, `.dot`, `DotProdTest`, "cross product", `Vector3.Multiply`), mostly in C#, Unity, GLSL and Python code; the
  other three are `iver` in "Divergent Boundaries" (rank 11), `.div` (a division function) and ` dotted` (a music
  text). Six diverge/converge positions remain in the top-50; the start's rank-0 position (`erg` in "φ-divergences")
  is at rank 49.
- At fixed positions: `erg` in "φ-divergences" goes from −2.526 (rank 0) to −1.789 under G (rank 1) and −0.704 under M
  (rank 49); `.cross(` in the C# ray tracer goes from −2.078 (rank 2) to −1.661 under G (rank 2) and −1.604 under M
  (rank 0); `numpy.prod` from −1.860 (rank 4) to below G's top-50 and −1.434 under M (rank 1). The pole's response to
  the divergence word-piece is 1.22 times its response to `.cross(` at the start, 1.08 times under G and 0.44 times
  under M.
- Directory counts of the − pole (`counts.npz`, layer 4, neuron 3542, sign index 1): `stack_edu-CSharp` holds 4 slots at
  the start, 15 under G and 26 under M. The pole's code+math share rises under G as well as under M, against the band-0
  direction for G.

## Conclusion

The head's token list picks out L04N3542−. None of the five matched random poles has a div-/-ergent word-piece or a
product operation in its top-20, and each has its own reading of the same specificity: Motorola, app and service
names, cryptocurrency, formal-methods terms, and "priorit-"/"keyword"/" Hot". The head's general sentence, "a
token-level feature that occurs in web, PDF and code documents alike", does not pick it out. "Token-level" fits B, D
and F as well, and the spread over document groups fits D as closely as the target (D 36 web, 6 PDF, 7 code, 1 math
slots of 50; the target 35, 4, 8, 3); A, B, C and F hold 47 to 50 of their 50 slots in web documents. All five random poles gave a coherent one-line reading, so a readable top-20 is what a strong layer-4 pole looks
like, not evidence that this pole is special. Under G the same tokens still dominate. Under M they do not: the
diverge/converge word-pieces fall from 37 to 6 of the top-50 positions and vector and array products in code rise from
11 to 40, and at fixed positions the divergence word-piece falls from 1.22 times to 0.44 times the pole's response to
`.cross(`. So the head's clause that a mix can change which documents hold a feature's strongest instances "without
changing what the feature responds to" holds for the two token families the pole had at the start, but not for their
order: under M this pole's strongest positions are product operations in code, not divergence words. That sentence of
`READ_head.md` should say so before it is used in a write-up. Limits: one reader, who knew the target's tokens before
reading; five random poles from one layer; top-20 contexts only for the blind step.

## Method and files

- `studies/next/m1/reading_control/select_and_blind.py`: the selection, the shuffle and the blind dump. It was run once
  from the session scratchpad before the sentences were written; the committed copy differs only in its docstring and
  output file names, and its outputs `blind_top20.txt` and `map.json` are byte-identical to the originals (checked with
  `cmp`).
- `studies/next/m1/reading_control/blind_top20.txt`: the six top-20 lists as the reader saw them.
- `studies/next/m1/reading_control/map.json`: labels to poles.
- `studies/next/m1/reading_control/target_g_m.py`, `target_g_m_output.txt`: the target's contexts, token families,
  position overlap and directory counts at start_a, G and M, and the six blind poles' directory groups at start_a.
  0 GPU-hours.
