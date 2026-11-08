"use strict";
/* The D1 data explorer. Reads window.D (data/core.js eagerly; words, contexts and docs on demand) and
   renders one page of expandable sections. Pass, strong and class follow neurons_verdict exactly
   (SPEC_D1 §2.3, §6.1); θs is live, and the frozen value is the one that decides. */

const C = D.core;
const FROZEN = C.theta_frozen;
const [MIN_POS, MAX_NEAR, MAX_DIST] = C.rules.pass;
const NS = "http://www.w3.org/2000/svg";
const MODEL_NAME = {
  "anthropic/claude-opus-5-5": "Opus 5.5", "anthropic/claude-sonnet-5-5": "Sonnet 5.5",
  "deepseek-ai/DeepSeek-V4.1-Flash-1017": "DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
  "google/gemini-3.7-flash": "Gemini 3.7 Flash", "zai-org/GLM-5.3": "GLM-5.3",
};
const MODEL_LABEL = Object.fromEntries(C.models.map((m) => [m.key, m.label]));
const KIND = { pos: "Positives", near: "Near misses", distant: "Distant", max: "Blind maximum" };
const ROLE = { explainer: "Explainer", second_reader: "Second reader", writer: "Writer", writer_job: "Job writer",
  clusterer: "Clusterer", judge: "Judge" };
const CLASSES = ["ONE", "TWO-REL", "UNREL", "UNK-F", "UNK-N", "RARE"];
/* definitions of this page's own labels; everything else is defined by glossary.js and by the data (see "terms") */
const GLOSS = {
  pole: "One sign of one MLP neuron: the tokens where its activation is positive (+) or negative (−). Each pole is studied on its own.",
  theta: "The strong threshold. A text is strong when its score reaches θs. Frozen at 0.3153: the 99th percentile of shuffled-pole scores on the discovery set (SPEC_D1 §6.1).",
  ownTheta: "The θs this round set for itself: the 99th percentile of its own texts’ scores at shuffled poles. Each round’s journal pass rate uses it.",
  score: "A text’s peak activation at the pole divided by the pole’s largest activation over the 50M-token corpus. 1 means the corpus maximum.",
  pass: "Share of scored poles whose explanation passes: at least 3 of 5 positives strong, at most 2 of 5 near misses and at most 1 of 5 distant texts (SPEC_D1 §6.1). The whisker is a Wilson 95% interval.",
  pos: "Five texts the writer wrote from the explanation alone, meant to fire the pole. In a table column: the share that is strong.",
  near: "Five texts close to the explanation that should not fire the pole. In a table column: the share that is strong.",
  distant: "Five texts on the most unrelated concept the writer could find, the essay’s test for packing. In a table column: the share that is strong.",
  shuffled: "The same 15 texts scored at a random other pole of the same layer: how often an unrelated text is strong by chance.",
  max: "The pole’s largest activation magnitude over the corpus, in the model’s raw units.",
  q999: "The 99.9th percentile of |a| over the tokens of this pole.",
  ntok: "Corpus tokens where the activation has this pole’s sign.",
  density: "How often the pole is clearly active: tokens with |a| at least 1% of the pole’s max, written as 1 in N corpus tokens.",
  dn: "Share of the pole’s top 50 tokens where gate and up are both negative. The essay reads this regime as an echo of the other pole.",
  same: "Share of the top 50 tokens that are the same token string.",
  pos0: "Share of the top 50 tokens at window position 0, which the harvest excludes from scoring.",
  rounds: "One dot per calibration round, in table order. Filled when that round’s explanation passes at the live θs.",
  klass: "Computed, never taken from the explainer: its label and job list, then the test. Hover a class for its meaning.",
  ONE: "One job, and the explanation passes.",
  "TWO-REL": "Two or three related jobs, and the explanation passes.",
  UNREL: "Two or more unrelated jobs, at least two of them supported by their own five texts.",
  "UNK-F": "Not understood. An explanation was given and failed the test. Says something about the explainer, not about packing.",
  "UNK-N": "Not understood. An agent declined, so there is no explanation or no texts.",
  RARE: "Fewer than 50 corpus tokens reach 1% of the pole’s max. Not classified.",
  sgate: "40 OLMo poles that tune the clusterer and judge at W4. Open by design.",
  sealed: "Computed by the jobs but not read before unblinding at W5 (SPEC_D1 §9.3).",
  unsealed: "Read at W5, after the prediction grid was frozen. Before that only the counts and harvest statistics were open (SPEC_D1 §9.3).",
  peak: "The token where the pole’s activation peaked on this text. The score is that peak over the pole’s corpus max.",
  space: "The token starts with a space.",
  round: "One calibration round: the same 120 Qwen3-4B-Base discovery poles explained and tested again with one change of prompt or model.",
  changed: "What this round changed against the round before it.",
  explainerCol: "The model that read each pole’s 30 top contexts and wrote its explanation.",
  writerCol: "The model that wrote the test texts from the explanation alone, without seeing the contexts.",
  role: "The agent’s job in the pipeline: explainer, second reader, writer, job writer, clusterer or judge.",
  calls: "API requests with a recorded response, counted from responses/*.usage.jsonl.",
  tokIn: "Prompt tokens summed over the calls, as the provider counted them.",
  tokOut: "Completion tokens summed over the calls, as the provider counted them.",
  time: "Latency summed over the calls, in minutes. Calls run in parallel, so this is more than the wall-clock time.",
  auditPoles: "Poles drawn for this model’s Q1 audit.",
  explainedBy: "Which explainer model explained how many of this cell’s poles.",
  texts: "Test texts written for this cell and scored on the GPU.",
  scoredPoles: "Poles the GPU probe scored: the explained poles plus their shuffle partners.",
  scoredOn: "The GPU that ran this cell’s probe.",
  step: "OLMo-3 stage-1 pretraining step of the checkpoint, at 4,194,304 tokens per step.",
  revision: "Hugging Face revision (commit) of the weights the job loaded.",
  gpu: "The GPU the job ran on.",
  tps: "Corpus tokens processed per second of the harvest loop.",
  g0col: "Largest G0 ratio over the layers: max |act(g)·u − a| / max |a| on the first batch, recomputed in fp32 against the module’s own output. Passes at 0.01 or below.",
  dead: "Poles whose largest |a| over the corpus is exactly 0.",
  dir: "The job’s output directory under bank/neurons/.",
  w4dir: "A W4 explain directory under bank/w4/ in the explain worktree.",
  started: "Start time in the directory’s manifest, UTC.",
  batches: "Request batches in the manifest: batch name, request count, roles and models.",
  clusters: "Lines in clusters.jsonl: clusterer answers collected, one line per pole and sample. Not opened for the runs on S while they were sealed.",
  layersWidth: "Transformer layers × MLP neurons per layer. Each neuron has two poles.",
  act: "The MLP gate activation function of the loaded model.",
  tokensCol: "Corpus tokens the harvest scored.",
  rarePoles: "Poles with fewer than 50 corpus tokens at 1% of their max |a| or more. They are RARE and not classified.",
  nll: "Mean next-token negative log-likelihood, in nats, over the first 120k target tokens of the corpus.",
  rental: "One vast.ai GPU rental, by its label.",
  wave: "The study wave the rental served.",
  opened: "When the rental was created, UTC.",
  rentalStatus: "How the rental ended: done, or killed_infra when an infrastructure fault ended it.",
  gpuh: "Billed GPU-hours of the rental.",
  cost: "Realized cost in US dollars, from the spend record.",
  job: "The GPU job’s bank path under bank/neurons/: wave, kind, attempt.",
  jobStep: "One step of the job’s pipeline, from its pipeline_status.json.",
  jobModel: "Model key the step harvested.",
  jobTokens: "Corpus tokens the step processed.",
  state: "The step’s final state in pipeline_status.json.",
  poleId: "L<layer>N<neuron> and the sign: + for the positive pole, − for the negative pole.",
  layer: "MLP layer of the pole, counting from 0.",
  explanation: "The explainer’s one-sentence account of what the pole does.",
  sgatePass: "Filled when the pole’s W2-C explanation passes at the live θs.",
  clusterPrompt: "Clusterer prompt the directory ran: v1, or v2 (21 Oct), which states that every one of the pole’s 50 exemplars goes in exactly one cluster.",
  kept: "Per batch: answers kept against requests, counted as lines of responses/<batch>.jsonl. Rejected answers were moved out and asked again; declined ids hit the attempts cap. For the runs on S only these counts are read.",
  gateCol: "The readout that decides whether the run may go on. Judge gate: a_ctrl, the judge’s accuracy when each S_gate pole’s main cluster is paired with the pole’s own record at T14; it must reach 0.80 before S is judged (SPEC_D1 §6.3). G2: the share of T0 poles the explainer called confident, which must stay within 10% (SPEC_D1 §5).",
  dcUnit: "What one row counts: grid poles (a proxy), or the clusters of S, which is how the verdict reads presence.",
  dcN: "Poles or clusters compared between the two GPUs.",
  dcPct: "Presence by the percentile rule the verdict uses: the median a⁺ of the exemplars reaches the pole’s own q99.9. Present on GPU A / GPU B, and how many change between them.",
  dcRaw: "Presence by the raw-ratio rule: the median a⁺ of the exemplars reaches 0.3 × the pole’s max |a|. Present on GPU A / GPU B, and how many change between them.",
  q1Class: "The class computed for each audit pole at the frozen θs, from the explainer’s label and job list and the test of its texts (SPEC_D1 §6.2). UNK joins UNK-F and UNK-N.",
  q1Cell: "Share of the cell’s 600 audit poles in this class, with its Wilson 95% interval.",
  essayCol: "The essay’s own estimate under its own thresholds: 90% one job, 7% two related jobs, 2–4% packed. Not the same instrument, so not a test of it.",
  runCol: "One unblind run: the instrument’s output directory under bank/ in the explain worktree. The final run is bank/w5; each earlier one was moved aside and kept.",
  runWhen: "When the unblind wrote metrics.json, UTC.",
  runPu: "p_u with its 95% pole-bootstrap interval in that run.",
  runN: "Late-born primary clusters the run’s p_u counted.",
  runHard: "Judge accuracy on the hard decoy over all late-born clusters, on the flagged ones, and on the controls. Not measurable while the control accuracy is below 0.80.",
  runSens: "p_u with clusters born at T7 or T15 left out, the checker’s device-boundary sensitivity (checks/W1-W3.md F2 item 3).",
  runSha: "First 12 hex digits of the SHA-256 of that run’s metrics.json.",
  clName: "The clusterer’s name for one group of the pole’s 50 T15 exemplars, read from their contexts.",
  clN: "How many of the 50 T15 exemplars the clusterer put in this cluster.",
  clPresent: "Presence at T0 to T15 by the percentile rule: filled where the median a⁺ of the cluster’s exemplars reaches the pole’s own q99.9 at that checkpoint. Hover for the values.",
  clBirth: "Birth: the first checkpoint at which the cluster is present and still present at the next one (at T15, present there), by the percentile rule; in brackets by the 0.3 × max rule (SPEC_D1 §2.4).",
  clStatus: "How the verdict tabled the cluster, in this order: echo (double-negative regime, takes no part); never_born; earliest (born at the pole’s earliest birth); formed_before_step_1000 (born late at T1); readings_disagree (late by the percentile rule, not by the 0.3 × max rule); unmatched_in_sample_1 (no cluster of the second clusterer sample holds more than half its exemplars); late_primary (late by both rules and matched: the clusters p_u counts). The main cluster is the one with the largest exemplar activation at T15.",
  clJudge: "Judge accuracy on this cluster: against random decoys, and against the hard decoy. Each judgment pairs the pole’s record at t_before with a decoy record and asks which belongs to the job.",
  clR: "r(c) at birth: the cluster’s median a⁺ at its birth checkpoint over the pole’s max |a| there. Exploratory.",
  centerMax: "Score of the pole’s blind-maximum text at each checkpoint: the text’s peak activation over that checkpoint’s own max |a|.",
  centerPos: "Mean score of the pole’s five positives at each checkpoint, read the same way.",
  vcPrompt: "What the explainer’s prompt showed as this pole’s most promoted tokens: the pole’s own list, or the opposite pole’s.",
};
function term(text, key) {
  if (!(key in GLOSS)) throw new Error(`no definition for ${key}`);
  return h("span", { class: "term", "data-g": key, tabindex: "0" }, text);
}
function defined(text, d) { return h("span", { class: "term", "data-d": d }, text); }

/* ---------- small helpers ------------------------------------------------------------------- */

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (k === "style") Object.assign(e.style, v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
}
function s(tag, attrs, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null) e.append(kid.nodeType ? kid : String(kid));
  return e;
}
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`);
const int = (x) => (x == null ? "–" : Math.round(x).toLocaleString("en-US"));
const sig = (x, d = 3) => (x == null ? "–" : Math.abs(x) >= 1000 ? int(x) : Number(x.toPrecision(d)).toString());
const th4 = (x) => x.toFixed(4);
const oneIn = (n, d) => {  // "1 in N" with N kept short: 8.2, 96, 4.1k, 52M
  if (!d) return "never";
  const x = n / d;
  return `1 in ${x >= 1e6 ? `${Number((x / 1e6).toPrecision(2))}M` : x >= 1e4 ? `${Number((x / 1e3).toPrecision(2))}k` : sig(x, 2)}`;
};
const model = (id) => MODEL_NAME[id] || id || "–";
const signWord = (id) => (id.endsWith("+") ? "positive" : "negative");
const minus = (id) => id.replace(/-$/, "−");
function parsePole(id) {
  const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
  return { layer: +m[1], neuron: +m[2], sign: m[3] };
}
function wilson(k, n, z = 1.959963984540054) {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d;
  const hw = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - hw, c + hw];
}
function chevron() {
  return s("svg", { class: "chev", viewBox: "0 0 14 14", "aria-hidden": "true" }, s("path", { d: "M3 5l4 4 4-4" }));
}
function lock() {
  return s("svg", { viewBox: "0 0 14 14", "aria-hidden": "true" }, s("rect", { x: 2.5, y: 6, width: 9, height: 6.5, rx: 1.5 }),
    s("path", { d: "M4.5 6V4.5a2.5 2.5 0 015 0V6" }));
}
const loaded = {};
function load(name) {
  loaded[name] ||= new Promise((resolve, reject) => {
    const tag = h("script", { src: `data/${name}.js` });
    tag.onload = () => resolve(D[name]);
    tag.onerror = () => reject(new Error(`data/${name}.js did not load. Run python -m infra.pg.neurons_explore.`));
    document.head.append(tag);
  });
  return loaded[name];
}
function failed(err) { return h("p", { class: "loading" }, err.message); }

/* ---------- the rules, as neurons_verdict states them --------------------------------------- */

const strongN = (xs, t) => (xs || []).reduce((n, x) => n + (x >= t), 0);
function passes(rec, t) {
  if (!rec.s) return null;
  return strongN(rec.s.pos, t) >= MIN_POS && strongN(rec.s.near, t) <= MAX_NEAR && strongN(rec.s.distant, t) <= MAX_DIST;
}
function klass(rec, t) {  // assign_class + part_of
  if (rec.rare) return "RARE";
  if (!rec.s) return "UNK-N";
  const p = passes(rec, t), n = rec.related.length;
  if (rec.label === "ONE" && n === 1 && p) return "ONE";
  if (rec.label === "TWO-REL" && n >= 2 && n <= 3 && rec.related.every(Boolean) && p) return "TWO-REL";
  if (rec.label === "UNREL") {
    const un = rec.related.flatMap((r, k) => (r ? [] : [k]));
    if (un.length >= 2 && un.filter((k) => strongN(rec.s[`job_pos:${k}`], t) >= MIN_POS).length >= 2) return "UNREL";
  }
  return rec.declined || rec.label === "UNK" ? "UNK-N" : "UNK-F";
}
function passRate(recs, t) {
  const scored = recs.filter((r) => r.s && !r.rare);
  const k = scored.filter((r) => passes(r, t)).length;
  return { k, n: scored.length, rate: scored.length ? k / scored.length : NaN, ci: wilson(k, scored.length) };
}
function strongRates(recs, t) {
  const scored = recs.filter((r) => r.s && !r.rare), out = {};
  for (const kind of ["pos", "near", "distant"]) {
    const xs = scored.flatMap((r) => r.s[kind]);
    out[kind] = strongN(xs, t) / xs.length;
  }
  const sh = scored.flatMap((r) => r.sh);
  out.sh = strongN(sh, t) / sh.length;
  return out;
}
const recsOf = (round) => Object.values(round.poles);

/* ---------- θ, live ---------------------------------------------------------------------- */

const state = { theta: FROZEN };
const watchers = new Set();
let pending = false;
function onTheta(fn) { watchers.add(fn); fn(state.theta); return () => watchers.delete(fn); }
function setTheta(t) {
  state.theta = Math.min(0.6, Math.max(0.05, Math.round(t * 1e4) / 1e4));
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    for (const fn of [...watchers]) if (fn(state.theta) === false) watchers.delete(fn);
  });
}
const isFrozen = () => Math.abs(state.theta - FROZEN) < 5e-5;

function buildRail() {
  const rail = document.getElementById("rail"), handle = document.getElementById("rail-handle");
  const fill = document.getElementById("rail-fill"), ticks = document.getElementById("rail-ticks");
  const LO = 0.05, HI = 0.6, x = (t) => ((t - LO) / (HI - LO)) * 100;
  const marks = C.rounds.map((r) => ({ t: r.theta, label: `${r.id}: its own θs ${th4(r.theta)}` }));
  for (const m of marks) {
    const tick = h("div", { class: "rail-tick", style: { left: `${x(m.t)}%` } });
    tick.addEventListener("pointerenter", (e) => tip(m.label, e));
    tick.addEventListener("pointerleave", untip);
    ticks.append(tick);
  }
  ticks.append(h("div", { class: "rail-tick frozen", style: { left: `${x(FROZEN)}%` }, title: `frozen θs ${th4(FROZEN)}` }));
  const snaps = [FROZEN, ...marks.map((m) => m.t)];
  const fromEvent = (e) => {
    const r = rail.getBoundingClientRect();
    let t = LO + Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (HI - LO);
    const near = snaps.find((v) => Math.abs(v - t) < 0.004);
    return near ?? t;
  };
  rail.addEventListener("pointerdown", (e) => {
    rail.setPointerCapture(e.pointerId); rail.classList.add("dragging"); setTheta(fromEvent(e));
  });
  rail.addEventListener("pointermove", (e) => { if (rail.hasPointerCapture(e.pointerId)) setTheta(fromEvent(e)); });
  rail.addEventListener("pointerup", () => rail.classList.remove("dragging"));
  rail.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.01 : 0.001;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") setTheta(state.theta + step);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") setTheta(state.theta - step);
    else if (e.key === "Home" || e.key === "Escape") setTheta(FROZEN);
    else return;
    e.preventDefault();
  });
  const value = document.getElementById("theta-value"), st = document.getElementById("theta-state");
  onTheta((t) => {
    handle.style.left = `${x(t)}%`; fill.style.width = `${x(t)}%`;
    value.textContent = th4(t);
    rail.setAttribute("aria-valuenow", th4(t));
    st.replaceChildren(isFrozen() ? "frozen" : h("button", { onclick: () => setTheta(FROZEN) }, "Back to frozen"));
  });
}

/* ---------- tooltip ---------------------------------------------------------------------- */

const tipEl = document.getElementById("tip");
function tip(html, e) {
  tipEl.innerHTML = html;
  tipEl.classList.add("on");
  const w = tipEl.offsetWidth, ht = tipEl.offsetHeight;
  let left = e.clientX + 14, top = e.clientY + 16;
  if (left + w > innerWidth - 8) left = e.clientX - w - 14;
  if (top + ht > innerHeight - 8) top = e.clientY - ht - 12;
  tipEl.style.left = `${left}px`; tipEl.style.top = `${top}px`;
}
function untip() { tipEl.classList.remove("on"); }

/* ---------- terms --------------------------------------------------------------------------
   Every label, code word and piece of jargon explains itself on hover. Definitions come from
   glossary.js (D.gloss); pole ids, checkpoints, rounds, spec sections, commits and hashes are
   defined from the data. annotate() wraps each known surface in a .term element, and a
   MutationObserver runs it on whatever the page renders. */

const WORD_CH = /[\p{L}\p{N}_]/u;
const isWord = (ch) => ch !== undefined && WORD_CH.test(ch);
const lexicon = new Map(), heads = new Map();  // surface → definitions; first character → surfaces, longest first
function addSurface(t, e) {
  if (!lexicon.has(t)) {
    lexicon.set(t, []);
    if (!heads.has(t[0])) heads.set(t[0], []);
    heads.get(t[0]).push(t);
  }
  lexicon.get(t).push(e);
}
for (const r of C.rounds) {
  const p = passRate(recsOf(r), r.theta);
  addSurface(r.id, { d: `Calibration round on the 120 discovery poles of Qwen3-4B-Base. ${r.note} Explainer ${roleModel(r, "explainer")}, writer ${roleModel(r, "writer")}, prompt ${r.prompt}. At its own θs ${th4(r.theta)}, ${p.k} of ${p.n} scored explanations pass.`, src: `JOURNAL ${r.journal}` });
}
for (const r of C.cancelled_rounds) addSurface(r.id, { d: `Calibration round with every Q1 role on ${model(r.model)}. ${r.note}`, src: `JOURNAL ${r.journal}` });
for (const [id, label] of Object.entries(MODEL_NAME)) {  // the language models, by what they did here
  const byRole = {}, uses = [];
  for (const r of C.rounds) for (const [role, u] of Object.entries(r.usage)) if (id in u.models) (byRole[ROLE[role].toLowerCase()] ||= []).push(r.id);
  for (const [role, ids] of Object.entries(byRole)) uses.push(`${role} in ${ids.join(", ")}`);
  const cells = C.cells.filter((c) => c.explainers[id]).map((c) => `${int(c.explainers[id])} ${c.key}`);
  if (cells.length) uses.push(`W2-C explainer of ${cells.join(", ")} audit poles`);
  const w4 = [...new Set(C.w4.flatMap((d) => d.batches.filter((b) => (b.models || []).includes(id)).flatMap((b) => b.roles || [])))];
  if (w4.length) uses.push(`W4 ${w4.join(" and ")}`);
  for (const r of C.cancelled_rounds.filter((x) => x.model === id)) uses.push(`round ${r.id}, cancelled`);
  const e = { d: `Language model called over an API, id ${id}.${uses.length ? ` Here: ${uses.join("; ")}.` : ""}`, src: "explain manifests" };
  for (const t of new Set([label, id, id.split("/")[1]])) addSurface(t, e);
}
const ROLE_OF_MODEL = { Q4B: "Its 120 discovery poles calibrated the instrument at W2-D; 600 more are audited in Q1.", G4: "600 of its poles are audited in Q1.",
  O7: "600 of its poles are audited in Q1; its 16 stage-1 checkpoints form the development grid of Q2.", C0: "Harvested only to reproduce a banked wikitext NLL (gate G1)." };
for (const m of [...C.models, { ...C.c0, key: "C0" }]) {
  const e = { d: `Model studied here, ${m.repo} at revision ${m.revision.slice(0, 10)}: ${m.n_layers} layers, MLP width ${int(m.width)}, ${m.act_fn_measured} activation, ${int(m.n_poles)} poles. ${ROLE_OF_MODEL[m.key]}`, src: `${m.repo} config, harvest manifest` };
  for (const t of new Set([m.label, m.repo].filter(Boolean))) addSurface(t, e);
}
const jobsOn = {};
for (const j of C.runs.jobs) if (j.device) (jobsOn[j.device] ||= new Set()).add(j.job);
for (const [dev, jobs] of Object.entries(jobsOn)) {
  const e = { d: `GPU model. It ran ${jobs.size} banked job${jobs.size > 1 ? "s" : ""} here: ${[...jobs].join(", ")}.`, src: "pipeline_status.json" };
  const short = dev.replace(/^NVIDIA /, "");
  for (const t of new Set([dev, short, short.split(" ").slice(0, 3).join(" ")])) addSurface(t, e);
}
const glossed = D.gloss.map((g) => [g, { d: g.d, src: g.src, code: !!g.code_only, docs: g.in }]);
for (const [g, e] of glossed) for (const t of new Set([g.t, ...(g.also || [])])) addSurface(t, e);
for (const list of heads.values()) list.sort((a, b) => b.length - a.length);

/* the definition of surface t where it stands: a document's own meaning first, a code span's code meaning next;
   outside the documents the page speaks the spec's language, plus entries scoped to "page" */
function pick(t, doc, code) {
  const es = lexicon.get(t);
  if (!es) return null;
  const here = (e) => (doc ? e.docs.includes(doc) : e.docs.includes("SPEC_D1.md") || e.docs.includes("page"));
  const ok = es.filter((e) => (code || !e.code) && (!e.docs || here(e)));
  return ok.find((e) => e.docs) || (code && ok.find((e) => e.code)) || ok[0] || null;
}

const TOKENS_PER_STEP = 4194304;  // OLMo-3 stage 1, SPEC_D1 §4.1
const tokens = (n) => (n >= 1e12 ? `${sig(n / 1e12, 3)}T` : `${sig(n / 1e9, 3)}B`);
const SECTION = /§ ?(\d+[a-z]?(?:\.\d+[a-z]?)*)/y;
const GENERATED = [
  [/L(\d{1,2})N(\d{1,6})([+\-−]?)/y, (m) => poleTerm(+m[1], +m[2], m[3])],
  [/(?:T_?|t)(\d{1,2}) ?(?:…|\.\.) ?(?:T_?|t)(\d{1,2})/y, (m) => rangeTerm(+m[1], +m[2])],
  [/T_?(\d{1,2})|t(\d\d)(?:_step\d+)?/y, (m) => checkpointTerm(+(m[1] ?? m[2]))],
  [/stage1-step(\d+)/y, (m) => stepTerm(+m[1])],
  [SECTION, (m, ctx) => sectionTerm(m[1], ctx)],
  [/[0-9a-f]{7,64}/y, (m) => hashTerm(m[0])],
  [/([\w./-]+\.(?:py|md|sh|jsonl?|js)):(\d+)(?:-(\d+))?/y, (m) => ({ d: `${m[3] ? `Lines ${m[2]}–${m[3]}` : `Line ${m[2]}`} of ${m[1]}, as the file stood when this was written.` })],
];
const CODE_GENERATED = [  // only as a whole code span: in prose these shapes mean other things
  [/:(\d+)(?:-(\d+))?/y, (m) => ({ d: `${m[2] ? `Lines ${m[1]}–${m[2]}` : `Line ${m[1]}`} of the file named just before, as it stood when this was written.` })],
  [/test_\w+/y, (m) => ({ d: `Unit test (pytest). Its name states what it checks: “${m[0].slice(5).replace(/_/g, " ")}”.` })],
];
function generated(text, i, ctx, list = GENERATED) {  // the longest data-defined surface at i, with its definition
  let best = null;
  for (const [re, fn] of list) {
    re.lastIndex = i;
    const m = re.exec(text);
    if (!m || (best && m[0].length <= best.s.length) || isWord(text[i + m[0].length])) continue;
    const def = fn(m, ctx);
    if (def) best = { s: m[0], def };
  }
  return best;
}

function poleTerm(layer, neuron, sign) {
  const id = `L${String(layer).padStart(2, "0")}N${neuron}`;
  if (!sign) return { d: `Neuron ${neuron} in MLP layer ${layer} (layers count from 0). Its two poles are ${id}+ and ${id}−, studied separately.` };
  const key = `${id}${sign === "+" ? "+" : "-"}`, where = [];
  if (C.poles[key]) where.push(`${MODEL_LABEL[C.poles[key].model]} ${C.poles[key].set} pole`);
  if (C.poles[`O7:${key}`]) where.push("OLMo-3-1025-7B S_gate pole");
  const cls = (r) => (r.c_frz === "UNK" ? r.unk_frz : r.c_frz);
  for (const c of C.cells) if (c.audit[key]) where.push(C.sealed ? `${c.label} audit pole, results sealed until W5` : `${c.label} audit pole, class ${cls(c.poles[key])} at the frozen θs`);
  const cell = C.cells.find((c) => c.audit[key] || (c.key === "O7" && C.poles[`O7:${key}`]));
  const w = where.length !== 1 ? null : C.poles[key]?.set === "discovery" ? D.words?.[C.frozen_round]?.[key] : cell ? D[`words_${cell.key}`]?.[key] : null;  // an id in two models shows no explanation
  return {
    d: `The ${sign === "+" ? "positive" : "negative"} pole of neuron ${neuron} in MLP layer ${layer} (layers count from 0): the tokens where its activation is ${sign === "+" ? "above" : "below"} zero.${where.length ? ` Here: ${where.join("; ")}.` : ""}`,
    html: w?.e ? `<span class="lang">${esc(w.e)}</span>` : "",
  };
}
function checkpointTerm(k) {
  const cp = C.grid.checkpoints.find((c) => c.t === k);
  if (!cp) return null;
  return { d: `Checkpoint T${k} of the development grid: OLMo-3-1025-7B at stage-1 step ${int(cp.step)}${k ? `, about ${tokens(cp.step * TOKENS_PER_STEP)} training tokens` : ", the random initialization"}.${k === 15 ? " Each pole’s 50 exemplar positions are its strongest tokens here." : ""}`, src: cp.dir };
}
function rangeTerm(a, b) {
  const [p, q] = [a, b].map((k) => C.grid.checkpoints.find((c) => c.t === k));
  return p && q && a < b ? { d: `Checkpoints T${a} to T${b} of the development grid: OLMo-3-1025-7B stage-1 steps ${int(p.step)} to ${int(q.step)}.` } : null;
}
function stepTerm(step) {
  const k = C.grid.checkpoints.findIndex((c) => c.step === step);
  return { d: `Hugging Face revision of OLMo-3-1025-7B at stage-1 pretraining step ${int(step)}, about ${tokens(step * TOKENS_PER_STEP)} tokens.${k >= 0 ? ` Grid checkpoint T${k}.` : ""}` };
}
function hashTerm(x) {
  if (!/\d/.test(x) || !/[a-f]/.test(x)) return null;
  const pr = Object.values(C.prompts);
  const p = pr.find((q) => q.sha.startsWith(x));
  if (p) return { d: `SHA-256 of the ${ROLE[p.role].toLowerCase()} prompt${p.version ? ` ${p.version}` : ""}, first committed in ${p.commit}.`, src: `infra/pg/neurons_prompts/${p.role}.md` };
  const c = C.commits[x];
  if (c) return { d: `Git commit of ${new Date(c[0]).toISOString().slice(0, 16).replace("T", " ")}Z: ${c[1]}`, src: "git log" };
  const pc = pr.find((q) => q.commit === x);
  if (pc) return { d: `Git commit that first holds the ${ROLE[pc.role].toLowerCase()} prompt ${pc.version || pc.sha.slice(0, 8)}.`, src: "git log" };
  const m = [...C.models, C.c0].find((q) => q.revision?.startsWith(x));
  if (m) return { d: `Hugging Face revision (commit) of ${m.repo} that the harvest ran on.` };
  const cp = C.grid.checkpoints.find((q) => q.revision.startsWith(x));
  if (cp) return { d: `Hugging Face revision of OLMo-3-1025-7B checkpoint T${cp.t}, stage-1 step ${int(cp.step)}.` };
  if (C.grid.checkpoints.some((q) => q.git === x) || [...C.models, C.c0].some((q) => q.pg_git_sha?.startsWith(x))) return { d: "Git commit of this repository that the harvest job ran from." };
  const run = RES?.runs.find((r) => r.sha.startsWith(x));
  if (run) return { d: `SHA-256 of metrics.json of unblind ${runName(run)}, written ${fmtDay(run.created)}.`, src: `bank/${run.dir}/verdict/metrics.json` };
  const input = RES && inputHashes().find(([sha]) => sha.startsWith(x));
  if (input) return { d: `SHA-256 of ${input[1]} as the final unblind read it.`, src: "bank/w5/verdict/metrics.json inputs" };
  return null;
}
function inputHashes() {  // every {path, sha256} the final metrics.json records for its inputs
  if (inputHashes.list) return inputHashes.list;
  const out = [];
  const walk = (o) => {
    if (Array.isArray(o)) o.forEach(walk);
    else if (o && typeof o === "object") {
      if (typeof o.sha256 === "string" && typeof o.path === "string") out.push([o.sha256, o.path.replace(/^.*\/bank\//, "bank/")]);
      Object.values(o).forEach(walk);
    }
  };
  walk(M);
  return (inputHashes.list = out);
}
const runName = (r) => (r.dir === "w5" ? `run ${RES.runs.length}, the final one` : r.dir.replace(/^w5_run(\d+)_.*/, "run $1"));
const docSections = {};
function sectionsOf(name) {
  if (docSections[name]) return docSections[name];
  const out = new Map(), lines = (D.docs.find((d) => d.name === name)?.text || "").split("\n");
  const clean = (t) => t.replace(/`|\*\*/g, "").trim();
  lines.forEach((l, i) => {
    const m = /^#{2,4} (\d+[a-z]?(?:\.\d+[a-z]?)*)\.? (.*)$/.exec(l) || /^(\d+\.\d+[a-z]?) \*\*([^*]+)\*\*(.*)$/.exec(l);
    if (!m) return;
    const body = [m[3] || ""];
    for (let j = i + 1; j < lines.length && body.join(" ").length < 240 && !/^(#|\d+\.\d+[a-z]? \*\*)/.test(lines[j]); j++) if (lines[j].trim()) body.push(lines[j]);
    const text = clean(body.join(" ")).replace(/\s+/g, " ");
    out.set(m[1], { title: clean(m[2]).replace(/\.$/, ""), text: text.length > 230 ? `${text.slice(0, 230).replace(/\s\S*$/, "")} …` : text });
  });
  return (docSections[name] = out);
}
function sectionTerm(num, ctx) {
  if (!D.docs) return {};  // matched now, defined on hover once the documents are loaded
  const named = ctx.in === "SPEC" ? "SPEC_D1.md" : ctx.in && D.docs.find((d) => d.name.split("/").pop().startsWith(`${ctx.in}.`))?.name;
  const target = named || "SPEC_D1.md", s = sectionsOf(target).get(num);
  const own = ctx.doc && ctx.doc !== target ? sectionsOf(ctx.doc).get(num) : null;
  if (!s && !own) return null;
  const label = (n) => n.replace(/\.md$/, "");
  if (!s) return { d: `${label(ctx.doc)} §${num}, ${own.title}. ${own.text}`, src: ctx.doc };
  return { d: `${label(target)} §${num}, ${s.title}. ${s.text}${own ? ` This document’s own §${num} is “${own.title}”.` : ""}`, src: target };
}

const SKIP = ".term, .lang, .kwic, .ttext, .vocab, input, textarea, script, style, #tip, #results, [data-noterm]";
const docOf = (el) => el.closest("[data-doc]")?.dataset.doc ?? null;
const DOC_BEFORE = /(SPEC_D1|SPEC|DATA_CONTRACT|EXPLAIN|REVIEW_EXPLAIN|VERIFY|HARVEST|JOURNAL)(\.md)?`?\s*$/;
function matchAt(text, i, doc) {
  let best = null;
  for (const t of heads.get(text[i]) || []) {
    if (!text.startsWith(t, i) || (isWord(t.at(-1)) && isWord(text[i + t.length]))) continue;
    if (pick(t, doc, false)) { best = t; break; }
  }
  if (isWord(text[i - 1])) return best && !isWord(best[0]) ? { s: best } : null;
  const ctx = { doc, in: DOC_BEFORE.exec(text.slice(Math.max(0, i - 24), i))?.[1] ?? null };  // "DATA_CONTRACT §6" names its document
  const g = generated(text, i, ctx);
  if (g && (!best || g.s.length > best.length)) return { s: g.s, in: g.s.startsWith("§") ? ctx.in : null };
  return best ? { s: best } : null;
}
function wrapText(node) {
  const parent = node.parentNode;
  if (!parent) return;
  const text = node.data, doc = docOf(parent), out = [];
  let last = 0;
  for (let i = 0; i < text.length; i++) {
    if (isWord(text[i - 1]) && isWord(text[i])) continue;  // not the start of a word
    const m = matchAt(text, i, doc);
    if (!m) continue;
    out.push(text.slice(last, i), m);
    last = i + m.s.length;
    i = last - 1;
  }
  if (!out.length) return;
  out.push(text.slice(last));
  const svg = parent.namespaceURI === NS;
  const parts = out.map((x) => {
    if (typeof x === "string") return x;
    const el = svg ? document.createElementNS(NS, "tspan") : document.createElement("span");
    el.setAttribute("class", "term");
    el.dataset.k = x.s;
    if (x.in) el.dataset.in = x.in;
    el.textContent = x.s;
    return el;
  });
  // in a flex or grid box every child is laid out apart; keep the sentence one inline run there
  if (!svg && /flex|grid/.test(getComputedStyle(parent).display)) node.replaceWith(h("span", {}, parts));
  else node.replaceWith(...parts);
}
function wholeCode(code) {
  const raw = code.textContent.trim().replace(/…$/, ""), doc = docOf(code), base = raw.replace(/\/$/, "").split("/").pop();
  const path = !/[\s=]/.test(raw);  // a command or assignment is not named by the last part of its last path
  for (const c of new Set([raw, raw.replace(/\/$/, ""), ...(path ? [base, `${base}/`] : [])])) {
    const g = generated(c, 0, { doc, in: null }), gc = generated(c, 0, {}, CODE_GENERATED), e = pick(c, doc, true);
    const whole = c === raw || c === raw.replace(/\/$/, "");  // a path's last part only stands for a code name, never an English word
    if ((e && (whole || e.code || /[._\d-]/.test(c))) || (g && g.s.length === c.length) || (gc && gc.s.length === c.length)) { code.classList.add("term"); code.dataset.k = c; return; }
  }
}
function mergeSub(sub) {  // θ<sub>s</sub> and T<sub>15</sub> are one term each
  const prev = sub.previousSibling, k = sub.textContent;
  if (prev?.nodeType !== 3 || sub.closest(SKIP)) return;
  const head = /θ$/.test(prev.data) && k === "s" ? "θ" : /(^|\W)T$/.test(prev.data) && /^\d+$/.test(k) ? "T" : null;
  if (!head) return;
  prev.data = prev.data.slice(0, -1);
  const span = h("span", { class: "term", "data-k": head === "θ" ? "θs" : `T_${k}` }, head);
  sub.replaceWith(span);
  span.append(sub);
}
function annotate(node) {
  if (node.nodeType === 3) { if (node.parentElement && !node.parentElement.closest(SKIP)) wrapText(node); return; }
  if (node.nodeType !== 1 || node.closest(SKIP)) return;
  node.querySelectorAll("sub").forEach(mergeSub);
  (node.matches("code") ? [node] : node.querySelectorAll("code")).forEach((c) => c.closest(SKIP) || wholeCode(c));
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT,
    { acceptNode: (t) => (t.parentElement.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
  const texts = [];
  while (walker.nextNode()) texts.push(walker.currentNode);
  texts.forEach(wrapText);
}
const observer = new MutationObserver((records) => {
  for (const r of records) for (const n of r.addedNodes) if (n.isConnected) annotate(n);
  observer.takeRecords();  // the wrapping itself is not new content
});

async function define(el) {
  if (el.dataset.g) return { d: GLOSS[el.dataset.g] };
  if (el.dataset.d) return { d: el.dataset.d };
  const k = el.dataset.k, ctx = { doc: docOf(el), in: el.dataset.in || null };
  const e = pick(k, ctx.doc, el.tagName === "CODE");
  if (e) return e;
  if (k.startsWith("§")) await load("docs");
  const g = generated(k, 0, ctx), gc = el.tagName === "CODE" && generated(k, 0, ctx, CODE_GENERATED);
  return g && g.s.length === k.length ? g.def : gc && gc.s.length === k.length ? gc.def : null;
}
let tipFor = null, tipTimer = 0;
async function showTerm(el) {
  const g = await define(el);
  if (!g || tipFor !== el) return;
  tipEl.innerHTML = `<b class="tip-h">${esc(el.textContent.trim().replace(/\s+/g, " "))}</b>${esc(g.d)}${g.html || ""}${g.src ? `<small class="tip-src">${esc(g.src)}</small>` : ""}`;
  tipEl.classList.add("on");
  const r = el.getClientRects()[0] || el.getBoundingClientRect(), w = tipEl.offsetWidth, ht = tipEl.offsetHeight;
  let top = r.bottom + 8;
  if (top + ht > innerHeight - 8) top = r.top - ht - 8;
  tipEl.style.left = `${Math.max(8, Math.min(r.left, innerWidth - w - 8))}px`;
  tipEl.style.top = `${top}px`;
}
const termOf = (e) => e.target instanceof Element && e.target.closest(".term");
function enter(t, delay) { if (t === tipFor) return; tipFor = t; clearTimeout(tipTimer); tipTimer = setTimeout(() => showTerm(t), delay); }
function leave() { tipFor = null; clearTimeout(tipTimer); untip(); }
document.addEventListener("pointerover", (e) => { const t = termOf(e); if (t) enter(t, tipEl.classList.contains("on") ? 0 : 120); });
document.addEventListener("pointerout", (e) => { const t = termOf(e); if (t && !(e.relatedTarget instanceof Node && t.contains(e.relatedTarget))) leave(); });
document.addEventListener("focusin", (e) => { const t = termOf(e); if (t) enter(t, 0); });
document.addEventListener("focusout", (e) => { if (termOf(e)) leave(); });
addEventListener("scroll", () => { if (tipFor) leave(); }, { passive: true });

/* a table row that opens and closes like a button, by pointer or by Enter and Space */
function clickable(row, toggle) {
  Object.assign(row, { tabIndex: 0 });
  row.setAttribute("role", "button");
  row.setAttribute("aria-expanded", "false");
  const go = async () => { const open = row.getAttribute("aria-expanded") !== "true"; row.setAttribute("aria-expanded", String(open)); row.classList.toggle("open", open); await toggle(open); };
  row.addEventListener("click", go);
  row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  return (open) => { if ((row.getAttribute("aria-expanded") === "true") !== open) return go(); return Promise.resolve(); };
}

/* ---------- folds: every expandable thing ------------------------------------------------ */

function fold(render) {
  const body = h("div", { class: "fold-in" });
  const box = h("div", { class: "fold" }, body);
  let built = false;
  box.addEventListener("transitionend", (e) => {
    if (e.target === box && box.classList.contains("open")) box.classList.add("settled");
  });
  return {
    el: box,
    async set(open) {
      if (open && !built) {
        built = true;
        try { await render(body); } catch (err) { body.append(failed(err)); console.error(err); }
      }
      if (open === box.classList.contains("open")) return;
      if (open) { void box.offsetHeight; box.classList.add("open"); }
      else { box.classList.remove("settled"); requestAnimationFrame(() => box.classList.remove("open")); }
      await settle(box);
    },
    get open() { return box.classList.contains("open"); },
  };
}
/* resolves when a fold's height transition has ended, so a scroll lands on the final layout */
function settle(box) {
  const ms = parseFloat(getComputedStyle(box).transitionDuration) * 1000;
  if (!ms) return Promise.resolve();
  return new Promise((resolve) => {
    const done = (e) => { if (e.target === box) { box.removeEventListener("transitionend", done); resolve(); } };
    box.addEventListener("transitionend", done);
    setTimeout(() => { box.removeEventListener("transitionend", done); resolve(); }, ms + 80);
  });
}

function toggler(button, f) {
  button.setAttribute("aria-expanded", "false");
  const go = (open) => { button.setAttribute("aria-expanded", String(open)); return f.set(open); };
  button.addEventListener("click", () => go(!f.open));
  return go;
}

const sections = {};
function section(id, title, summary, render) {
  const sum = h("span", { class: "sum" });
  const head = h("button", { class: "sec-head", id: `h-${id}` }, h("h2", {}, title), sum, chevron());
  const f = fold(async (body) => { const inner = h("div", { class: "sec-body" }); body.append(inner); await render(inner); });
  const open = toggler(head, f);
  const el = h("section", { class: "section", id }, head, f.el);
  onTheta(() => summary(sum));
  sections[id] = { el, open: () => (f.open ? Promise.resolve() : open(true)) };
  return el;
}

/* ---------- charts ----------------------------------------------------------------------- */

function scale(d0, d1, r0, r1) { return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0); }

/* pass rate of every round as θ moves; the frozen θ dashed, the live θ in the accent */
function curvesChart(onHover) {
  const W = 520, H = 250, L = 40, R = 14, T = 20, B = 44;
  const ts = []; for (let t = 0.1; t <= 0.5001; t += 0.005) ts.push(+t.toFixed(3));
  const x = scale(0.1, 0.5, L, W - R), y = scale(0, 1, H - B, T);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Pass rate of each calibration round as θs moves from 0.1 to 0.5" },
    s("text", { x: L, y: 8 }, "share of poles passing"), s("text", { x: (L + W - R) / 2, y: H - 2, "text-anchor": "middle" }, "θs"));
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v * 100}%`));
  }
  for (const v of [0.1, 0.2, 0.3, 0.4, 0.5]) svg.append(s("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, v.toFixed(1)));
  svg.append(s("line", { class: "frozen-line", x1: x(FROZEN), x2: x(FROZEN), y1: T, y2: H - B }));
  const live = s("line", { class: "theta-line", x1: 0, x2: 0, y1: T, y2: H - B });
  svg.append(live);
  const lines = {};
  for (const r of C.rounds) {
    const recs = recsOf(r);
    const d = ts.map((t, i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(passRate(recs, t).rate).toFixed(1)}`).join("");
    const own = passRate(recs, r.theta).rate;
    const g = s("g", { class: "curve", "data-round": r.id, style: "transition: opacity 180ms" },
      s("path", { d, fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5, style: "transition: stroke 180ms, stroke-width 180ms" }),
      s("circle", { cx: x(r.theta), cy: y(own), r: 3, fill: "var(--ink-3)" }),
      s("path", { d, fill: "none", stroke: "transparent", "stroke-width": 12, class: "hot" }));
    g.addEventListener("pointerenter", (e) => { onHover(r.id); tip(`<b>${r.id}</b> ${esc(r.note)}<br>Own θs ${th4(r.theta)}: ${pct(own)}`, e); });
    g.addEventListener("pointermove", (e) => tip(tipEl.innerHTML, e));
    g.addEventListener("pointerleave", () => { onHover(null); untip(); });
    lines[r.id] = g;
    svg.append(g);
  }
  const ends = s("g");
  svg.append(ends);
  onTheta((t) => {
    live.setAttribute("x1", x(Math.min(0.5, Math.max(0.1, t)))); live.setAttribute("x2", x(Math.min(0.5, Math.max(0.1, t))));
  });
  function highlight(id) {
    for (const [rid, g] of Object.entries(lines)) {
      const on = rid === id || (id == null && rid === C.frozen_round);
      g.firstChild.setAttribute("stroke", on ? "var(--ink)" : "var(--ink-3)");
      g.firstChild.setAttribute("stroke-width", on ? 2.25 : 1.25);
      g.style.opacity = id == null || on ? 1 : 0.35;
      if (on) svg.insertBefore(g, ends);
    }
  }
  highlight(null);
  return { el: svg, highlight };
}

/* one round's strong share by text kind as θ moves: where the shuffled null ends, positives must begin */
function kindsChart() {
  const W = 520, H = 250, L = 40, R = 84, T = 20, B = 44;
  const ts = []; for (let t = 0.1; t <= 0.5001; t += 0.005) ts.push(+t.toFixed(3));
  const x = scale(0.1, 0.5, L, W - R), y = scale(0, 1, H - B, T);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Share of strong texts by kind as θs moves, for one round" },
    s("text", { x: L, y: 8 }, "share of texts strong"), s("text", { x: (L + W - R) / 2, y: H - 2, "text-anchor": "middle" }, "θs"));
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v * 100}%`));
  }
  for (const v of [0.1, 0.2, 0.3, 0.4, 0.5]) svg.append(s("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, v.toFixed(1)));
  svg.append(s("line", { class: "frozen-line", x1: x(FROZEN), x2: x(FROZEN), y1: T, y2: H - B }));
  const live = s("line", { class: "theta-line", x1: 0, x2: 0, y1: T, y2: H - B });
  const styles = { pos: ["Positives", "var(--ink)", null, 2], near: ["Near misses", "var(--ink-2)", "5 3", 1.5],
    distant: ["Distant", "var(--ink-3)", "2 3", 1.5], sh: ["Shuffled", "var(--ink-3)", null, 1] };
  const paths = {}, labels = {};
  for (const [k, [lab, col, dash, w]] of Object.entries(styles)) {
    paths[k] = s("path", { fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash, style: "transition: d 260ms var(--ease)" });
    labels[k] = s("text", { x: W - R + 8, style: `fill: ${col}` }, lab);
    svg.append(paths[k], labels[k]);
  }
  svg.append(live);
  onTheta((t) => { const xx = x(Math.min(0.5, Math.max(0.1, t))); live.setAttribute("x1", xx); live.setAttribute("x2", xx); });
  const cache = {};
  function show(id) {
    const r = C.rounds.find((q) => q.id === id), recs = recsOf(r);
    cache[id] ||= ts.map((t) => strongRates(recs, t));
    const rates = cache[id], ends = [];
    for (const k of Object.keys(styles)) {
      paths[k].setAttribute("d", rates.map((v, i) => `${i ? "L" : "M"}${x(ts[i]).toFixed(1)},${y(v[k]).toFixed(1)}`).join(""));
      ends.push([k, y(rates.at(-1)[k])]);
    }
    ends.sort((a, b) => a[1] - b[1]);
    const pos = [];
    ends.forEach(([, yy], i) => pos.push(i ? Math.max(yy, pos[i - 1] + 12) : yy));
    for (let i = pos.length - 1; i >= 0; i--) pos[i] = Math.min(pos[i], i === pos.length - 1 ? H - B - 2 : pos[i + 1] - 12);
    ends.forEach(([k], i) => labels[k].setAttribute("y", pos[i] + 4));
  }
  return { el: svg, show };
}

/* every discovery pole (column) in every round (row): passes, fails, or has no texts */
function passField(onPick) {
  const poles = Object.keys(C.rounds[0].poles).sort((a, b) => {
    const p = parsePole(a), q = parsePole(b);
    return p.layer - q.layer || p.neuron - q.neuron || a.localeCompare(b);
  });
  const CW = 7.4, RH = 18, L = 62, T = 22, Rr = 46;
  const W = L + poles.length * CW + Rr, H = T + C.rounds.length * RH + 8;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Pass or fail of every discovery pole in every calibration round" });
  const cols = s("g"), cells = s("g");
  svg.append(cols, cells);
  let prevLayer = -1, lastX = -Infinity;
  poles.forEach((p, i) => {
    const { layer } = parsePole(p);
    if (layer !== prevLayer && L + i * CW - lastX >= 18) {
      svg.append(s("text", { class: "layerlab", x: L + i * CW + 1, y: 10 }, layer));
      lastX = L + i * CW;
    }
    prevLayer = layer;
    const col = s("rect", { class: "col", x: L + i * CW, y: T - 6, width: CW, height: C.rounds.length * RH, rx: 2 });
    col.addEventListener("pointerenter", (e) => {
      col.classList.add("hover");
      const w = D.words?.[C.frozen_round]?.[p];
      tip(`<b>${minus(p)}</b> layer ${layer}, ${signWord(p)} pole<br>${C.rounds.filter((x) => passes(x.poles[p], state.theta)).length} of ${C.rounds.length} rounds pass at θs ${th4(state.theta)}${w?.e ? `<span class="lang">${esc(w.e.slice(0, 160))}${w.e.length > 160 ? "…" : ""}</span>` : ""}`, e);
    });
    col.addEventListener("pointermove", (e) => tip(tipEl.innerHTML, e));
    col.addEventListener("pointerleave", () => { col.classList.remove("hover"); untip(); });
    col.addEventListener("click", () => onPick(p));
    col.style.cursor = "pointer";
    cols.append(col);
  });
  const dots = [], counts = [];
  C.rounds.forEach((r, j) => {
    const yy = T + j * RH + 3;
    svg.append(s("text", { class: `rlab${r.id === C.frozen_round ? " frozen" : ""}`, x: 0, y: yy + 4 }, r.id));
    const cnt = s("text", { class: "rcount", x: W - 2, y: yy + 4, "text-anchor": "end" });
    counts.push([r, cnt]); svg.append(cnt);
    poles.forEach((p, i) => {
      const c = s("circle", { class: "cell", cx: L + i * CW + CW / 2, cy: yy, r: 2.6, "pointer-events": "none" });
      dots.push([r.poles[p], c]); cells.append(c);
    });
  });
  onTheta((t) => {
    for (const [rec, c] of dots) {
      const p = passes(rec, t);
      c.setAttribute("class", `cell ${rec.rare || p === null ? "n" : p ? "p" : "f"}`);
    }
    for (const [r, cnt] of counts) cnt.textContent = pct(passRate(recsOf(r), t).rate, 0);
  });
  return svg;
}

/* the scores of one pole's texts: positives, near misses, distant, blind maximum, job texts, shuffled */
function scoreStrip(rec, onHover) {
  const rows = [["pos", rec.s.pos], ["near", rec.s.near], ["distant", rec.s.distant], ["max", rec.s.max]];
  for (const k of Object.keys(rec.s).filter((k) => k.startsWith("job_pos:"))) rows.push([k, rec.s[k]]);
  rows.push(["sh", rec.sh]);
  const all = rows.flatMap((r) => r[1]);
  const lo = Math.min(-0.05, ...all), hi = Math.max(1.25, ...all);
  const W = 640, L = 112, R = 16, RH = 22, T = 6, H = T + rows.length * RH + 24;
  const x = scale(lo, hi, L, W - R);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Each test text's score at this pole against θs" });
  for (const v of [0, 0.5, 1]) {
    svg.append(s("line", { class: v === 0 ? "axis" : "grid", x1: x(v), x2: x(v), y1: T, y2: H - 22 }),
      s("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, v === 1 ? "1 = corpus max" : v));
  }
  const live = s("line", { class: "theta-line", y1: T, y2: H - 22 });
  svg.append(live);
  const marks = [], dots = {};
  svg.dots = dots;
  rows.forEach(([kind, xs], j) => {
    const yy = T + j * RH + RH / 2;
    const label = kind === "sh" ? "Shuffled pole" : kind.startsWith("job_pos:") ? `Job ${+kind.split(":")[1] + 1} texts` : KIND[kind];
    svg.append(s("text", { x: 0, y: yy + 4, style: "fill: var(--ink-2)" }, label));
    xs.forEach((v, i) => {
      const c = s("circle", { cx: x(v), cy: yy, r: kind === "sh" ? 3.2 : 5, class: "hot" });
      const key = `${kind}:${i}`;
      dots[key] = c;
      c.addEventListener("pointerenter", (e) => { onHover(key, true); tip(`${esc(label)} ${i + 1}: score <b>${v.toFixed(3)}</b>`, e); });
      c.addEventListener("pointerleave", () => { onHover(key, false); untip(); });
      marks.push([v, c]); svg.append(c);
    });
  });
  onTheta((t) => {
    live.setAttribute("x1", x(t)); live.setAttribute("x2", x(t));
    for (const [v, c] of marks) { const f = c.classList.contains("focus"); c.setAttribute("class", `hot ${v >= t ? "dot-s" : "dot-w"}${f ? " focus" : ""}`); }
  });
  return svg;
}

/* the pole's own |a| distribution over the corpus, with its percentiles and the strong line θ·max */
function histogram(f, edges) {
  const W = 560, H = 120, L = 8, R = 8, T = 18, B = 22;
  const c = f.hist.c, lo = f.hist.lo;
  if (!c.length) return h("p", { class: "muted" }, "This pole never fired in the corpus.");
  const lx = (v) => Math.log10(v);
  const x0 = lx(edges[lo]), x1 = lx(edges[lo + c.length]);
  const x = scale(x0, x1, L, W - R);
  const ymax = Math.log10(Math.max(...c) + 1), y = scale(0, ymax, H - B, T);
  let d = `M${x(x0)},${y(0)}`;
  c.forEach((n, i) => { const yy = y(Math.log10(n + 1)); d += `L${x(lx(edges[lo + i]))},${yy}L${x(lx(edges[lo + i + 1]))},${yy}`; });
  d += `L${x(x1)},${y(0)}Z`;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Histogram of the pole's activation magnitudes over the corpus, log scale" },
  s("path", { d, fill: "var(--wash)", stroke: "var(--ink-3)", "stroke-width": 1 }));
  for (let p = Math.ceil(x0); p <= Math.floor(x1); p++) svg.append(s("text", { x: x(p), y: H - 6, "text-anchor": "middle" }, `10${sup(p)}`));
  const mark = (v, label, col, low) => {
    if (!(v > 0)) return null;
    const xx = x(Math.min(x1, Math.max(x0, lx(v))));
    const g = s("g", {}, s("line", { x1: xx, x2: xx, y1: low ? T + 6 : T - 4, y2: H - B, stroke: col, "stroke-width": low ? 1.5 : 1 }),
      low ? s("text", { x: xx - 5, y: H - B - 6, "text-anchor": "end", class: "halo", style: `fill: ${col}` }, label)
        : s("text", { x: xx, y: T - 7, "text-anchor": "middle", style: `fill: ${col}` }, label));
    svg.append(g);
    return g;
  };
  mark(f.q99, "q99", "var(--ink-3)");
  mark(f.q999, "q99.9", "var(--ink-3)");
  mark(f.max, "max", "var(--ink)");
  let strong = null;
  onTheta((t) => { strong?.remove(); strong = mark(t * f.max, "θs × max", "var(--accent-ink)", true); });
  return svg;
}
function sup(n) { return String(n).split("").map((ch) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(ch)] ?? (ch === "-" ? "⁻" : ch)).join(""); }

function fmtDay(utc) { const d = new Date(utc); return `${d.getUTCDate()} ${d.toLocaleString("en-GB", { month: "short", timeZone: "UTC" })} ${utc.slice(11, 16)}Z`; }

/* ---------- hero --------------------------------------------------------------------------- */

const WAVES = [
  ["W0", "Instrument", "Built and tested at toy scale", "19 Oct", "done"],
  ["W1", "Harvests", "Three models, 50M tokens each", "20 Oct", "done"],
  ["W2-D", "Calibration", "Prompt v3 and θs frozen", "20 Oct", "done"],
  ["W2-C", "Audit", "1,840 poles explained and scored", "21 Oct", "done"],
  ["W3", "Grid", "16 OLMo checkpoints harvested", "21 Oct", "done"],
  ["W4", "Clusters and judge", "", "", "next"],
  ["W5", "Unblind", "The prediction grid, filled once", "", ""],
];
const doneWaves = (w) => {  // W4 and W5 from the data
  if (w[0] === "W5") return M ? [w[0], w[1], `p_u ${f3(M.q2.p_u.point)}: ${M.q2.p_u.rule}; b above the band. Run ${RES.runs.length}.`, fmtDay(M.created_utc).replace(/ \d\d:\d\dZ$/, ""), "done"] : w;
  return w;
};
function w4Run() {  // the latest W4 run on S: answers kept against requests, per role
  const d = C.w4.filter((x) => /^w4\/o7_s/.test(x.dir) && x.started).sort((a, b) => a.started.localeCompare(b.started)).at(-1);
  if (!d) return null;
  const by = {};
  for (const b of d.batches) for (const r of b.roles || []) { const t = (by[r] ||= { kept: 0, n: 0 }); t.kept += b.kept; t.n += b.n; }
  return { dir: d.dir, by, judgments: d.n_judgments };
}
function hero() {
  const disc = Object.keys(C.rounds[0].poles).length, audit = C.cells.reduce((n, c) => n + c.n_audit, 0);
  return h("div", { class: "hero" },
    h("h1", {}, "Does “already clean” hold, and did it grow that way?"),
    M ? finding() : null,
    h("p", { class: M ? "soft" : null }, `Every pole, prompt, text and score of study D1, compiled ${C.built.replace("T", " ")}. `, h("b", {}, disc), " discovery ", term("poles", "pole"),
      " in ", h("b", {}, C.rounds.length), " calibration rounds, ", h("b", {}, C.grid.s_gate.length), " ", term("judge-tuning poles", "sgate"), " over ",
      h("b", {}, C.grid.checkpoints.length), " checkpoints, and ", h("b", {}, int(audit)), " audit poles", C.sealed ? [" that stay ", term("sealed until W5", "sealed")] : [", ", term("open since W5", "unsealed")],
      ". Drag ", term("θs", "theta"), ` above to rescore everything; the frozen ${th4(FROZEN)} is the value that decides.`),
    h("ol", { class: "waves" }, WAVES.map(doneWaves).map(([w, name, what, when, st]) => {
      if (w === "W4" && M) {
        const run = w4Run(), c = run?.by.clusterer, j = run?.by.judge;
        what = `S clustered (${int(c.kept)} answers) and judged (${int(run.judgments)} judgments in the verdict).`;
        when = `${fmtDay(C.w4.map((d) => d.started).filter(Boolean).sort()[0]).replace(/ \d\d:\d\dZ$/, "")}–${fmtDay(M.created_utc).replace(/ \d\d:\d\dZ$/, "")}`;
        st = "done";
      } else if (w === "W4") {
        const started = C.w4.map((d) => d.started).filter(Boolean).sort()[0], run = w4Run();
        const gate = C.w4.some((d) => d.judge_gate?.gate === "pass");
        const c = run?.by.clusterer, j = run?.by.judge;
        what = [gate ? "Judge gate passed." : "Judge gate on S_gate first.",
          c ? (c.kept === c.n ? " S clustered." : ` S clustering ${int(c.kept)} of ${int(c.n)}.`) : "",
          j ? ` Judge ${int(j.kept)} of ${int(j.n)} lines.` : ""].join("");
        when = started ? `started ${fmtDay(started)}` : "next";
      }
      return h("li", { class: st }, h("b", {}, `${w} ${name}`), h("span", {}, what), h("small", {}, when));
    })));
}

function guide() {
  const steps = [
    ["Top contexts", "The 30 places in a 50M-token corpus where the pole fired hardest."],
    ["Explainer", "A language model reads them and writes one sentence: what the pole does."],
    ["Writer", "A second model sees only that sentence and writes 5 positives, 5 near misses, 5 distant texts and 1 blind maximum."],
    ["Model", "Each text runs through the studied model. Its score is the pole’s peak on it over the pole’s corpus maximum."],
    ["Rule", "A text is strong when its score reaches θs. The explanation passes with at least 3 strong positives, at most 2 near misses, at most 1 distant."],
  ];
  const defs = [["Pole", "pole"], ["θs", "theta"], ["S_gate", "sgate"], C.sealed ? ["Sealed", "sealed"] : ["Open since W5", "unsealed"]];
  const btn = h("button", { class: "guide-head" }, "How to read this page", chevron());
  const f = fold((b) => b.append(h("div", { class: "guide" },
    h("ol", { class: "flow" }, steps.map(([t, d]) => h("li", {}, h("b", {}, t), h("span", {}, d)))),
    h("div", { class: "guide-cols" },
      h("dl", { class: "kv" }, defs.flatMap(([t, k]) => [h("dt", {}, t), h("dd", {}, GLOSS[k])])),
      h("dl", { class: "kv" }, CLASSES.flatMap((c) => [h("dt", {}, c), h("dd", {}, GLOSS[c])]))),
    h("p", { class: "note" }, "Every underlined word, code name, pole id and section number explains itself on hover. Every chart and number follows the θs in the bar; the frozen 0.3153 is the value that decides results."))));
  toggler(btn, f);
  return h("div", { class: "guide-wrap" }, btn, f.el);
}

/* ---------- results (W5) ------------------------------------------------------------------- */

const RES = C.results, M = RES?.metrics;
const f3 = (x) => (x == null ? "–" : x.toFixed(3));
const ci3 = (c) => `[${f3(c[0])}, ${f3(c[1])}]`;
const tLabel = (s) => String(s).replace(/T_(\d+)/g, "T$1");
const Q1_ROWS = ["ONE", "TWO-REL", "UNREL", "UNK", "RARE"];
const ESSAY = { ONE: [0.9, 0.9], "TWO-REL": [0.07, 0.07], UNREL: [0.02, 0.04] };

function finding() {  // the answer, written from metrics.json
  const q = M.q2, pu = q.p_u, bo = q.birth_order, v = q.vestigial;
  const one = C.cells.map((c) => `${pct(M.q1[c.key].fractions.ONE.p)} (${c.key})`).join(", ");
  return h("p", { class: "finding" },
    "On OLMo’s ", int(q.n_poles), " poles of S the parent test labels ", h("b", {}, pu.rule), ": the unrelated share of late-born clusters is ",
    h("b", {}, `p_u = ${f3(pu.point)}`), ` ${ci3(pu.ci95)}, its lower bound above the ${C.rules.pu[1].toFixed(2)} rule. Birth order supports the main-job-is-oldest form of H_RG: `,
    h("b", {}, `b = ${f3(bo.b)}`), ` ${ci3(bo.ci95)}, above the permutation band (97.5th percentile ${f3(bo.null_97_5)}). Not vestigial: v = ${f3(v.v)}. `,
    "A mixed outcome under §6.7, reported as such. In Q1, under explanations that pass, ", one, " of poles have one job, against the essay’s 90%.");
}

function numberLine({ lo, hi, ticks, rules = [], band, points }) {  // one readout on a 0..1 style axis
  const W = 1000, H = 64, L = 18, R = 18;
  const x = scale(lo, hi, L, W - R);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": points.map((p) => p.label).join("; ") });
  if (band) svg.append(s("rect", { x: x(band[0]), y: 14, width: x(band[1]) - x(band[0]), height: 22, class: "band" }));
  svg.append(s("line", { class: "axis", x1: L, x2: W - R, y1: 46, y2: 46 }));
  for (const t of ticks) svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: 42, y2: 50 }), s("text", { x: x(t), y: 62, "text-anchor": "middle" }, t));
  for (const r of rules) {
    svg.append(s("line", { x1: x(r.at), x2: x(r.at), y1: 8, y2: 46, stroke: "var(--ink-2)", "stroke-dasharray": "4 3" }),
      s("text", { x: x(r.at) + (r.anchor === "end" ? -6 : 6), y: 12, "text-anchor": r.anchor || "start", style: "fill: var(--ink-2)" }, r.label));
  }
  points.forEach((p, k) => {
    const yy = 25 + k * 9 - (points.length - 1) * 4.5;
    const g = s("g", {}, s("line", { x1: x(p.ci[0]), x2: x(p.ci[1]), y1: yy, y2: yy, stroke: p.soft ? "var(--ink-3)" : "var(--accent)", "stroke-width": 2 }),
      s("circle", { cx: x(p.v), cy: yy, r: p.soft ? 3.5 : 5, fill: p.soft ? "var(--ink-3)" : "var(--accent)", stroke: "var(--bg)", "stroke-width": 2 }));
    g.addEventListener("pointerenter", (e) => tip(esc(p.label), e));
    g.addEventListener("pointerleave", untip);
    svg.append(g);
  });
  return svg;
}

function q1Table() {
  const tbody = h("tbody", {}, Q1_ROWS.map((k) => h("tr", {}, h("td", {}, term(k === "UNK" ? "UNK" : k, k === "UNK" ? "q1Class" : k)),
    ...C.cells.map((c) => {
      const f = M.q1[c.key].fractions[k];
      return h("td", { class: "num" }, h("div", { class: "q1cell" }, h("span", { class: "q1bar" }, h("i", { style: { width: `${100 * f.p}%` } }),
        h("u", { style: { left: `${100 * f.wilson95[0]}%`, width: `${100 * (f.wilson95[1] - f.wilson95[0])}%` } }),
        ESSAY[k] ? h("b", { style: { left: `${100 * ESSAY[k][0]}%`, width: `${Math.max(0.6, 100 * (ESSAY[k][1] - ESSAY[k][0]))}%` } }) : null),
        h("span", {}, `${pct(f.p)} `, h("small", { class: "soft" }, `${int(f.k)}`))));
    }),
    h("td", { class: "num soft hide-sm" }, ESSAY[k] ? (ESSAY[k][0] === ESSAY[k][1] ? pct(ESSAY[k][0], 0) : `${pct(ESSAY[k][0], 0)}–${pct(ESSAY[k][1], 0)}`) : "–"))));
  return h("div", { class: "wrap-x" }, h("table", { class: "tbl q1t" },
    h("thead", {}, h("tr", {}, [term("Class", "q1Class"), ...C.cells.map((c) => term(c.label, "q1Cell")), term("Essay", "essayCol")].map((t, i) => h("th", { class: i === 4 ? "hide-sm" : null }, t)))), tbody));
}

function q1Splits() {  // by depth tercile and by explainer model, every cell
  const row = (label, f) => h("tr", {}, h("td", {}, label), h("td", { class: "r num" }, int(f.n)),
    ...Q1_ROWS.slice(0, 4).map((k) => h("td", { class: "r num" }, `${pct(f[k].p)}`, h("small", { class: "soft" }, ` [${pct(f[k].wilson95[0], 0)}, ${pct(f[k].wilson95[1], 0)}]`))));
  const out = h("div");
  for (const c of C.cells) {
    const q = M.q1[c.key];
    const rows = [...Object.entries(q.by_depth_tercile).map(([t, f]) => row(`depth tercile ${t}`, f)),
      ...Object.entries(q.by_explainer_model).map(([m, x]) => row(`explainer ${model(m)}`, x.fractions))];
    out.append(h("h4", { class: "mini-h" }, `${c.label}`, h("small", {}, `label kappa ${sig(q.kappa_label, 2)} over ${int(q.n_rated_twice)} poles rated twice; ${int(q.declined.n)} declined`)),
      h("div", { class: "wrap-x" }, h("table", { class: "tbl" }, h("thead", {}, h("tr", {}, ["", "n", ...Q1_ROWS.slice(0, 4)].map((t, i) => h("th", { class: i ? "r" : null }, t)))), h("tbody", {}, rows))));
  }
  return out;
}

function birthsChart() {  // births per checkpoint (percentile reading), main and other clusters
  const B = M.q2.births, n = B.main.counts.length, W = 1000, H = 150, L = 40, R = 12, T = 10, Bt = 26;
  const vmax = Math.max(...B.non_main.counts, ...B.main.counts);
  const x = scale(0, n, L, W - R), y = scale(0, vmax, H - Bt, T), bw = (W - R - L) / n;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Births per checkpoint, main and other clusters" });
  for (const v of [0, 100, 200, 300].filter((v) => v <= vmax)) svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }), s("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, v));
  for (let t = 0; t < n; t++) {
    const a = B.main.counts[t], o = B.non_main.counts[t];
    const g = s("g", {}, s("rect", { x: x(t) + 3, y: y(o), width: bw / 2 - 4, height: y(0) - y(o), rx: 2, fill: "var(--ink-3)" }),
      s("rect", { x: x(t) + bw / 2 + 1, y: y(a), width: bw / 2 - 4, height: y(0) - y(a), rx: 2, fill: "var(--accent)" }),
      s("text", { x: x(t) + bw / 2, y: H - 8, "text-anchor": "middle" }, `T${t}`));
    g.addEventListener("pointerenter", (e) => tip(`<b>T${t}</b>: ${a} main clusters and ${o} others born here`, e));
    g.addEventListener("pointerleave", untip);
    svg.append(g);
  }
  return svg;
}

function centerChart() {  // §2.4 center birth curve: the median ratio curve, by class
  const cb = M.q2.center_birth_curve, n = cb.grid.length, W = 1000, H = 170, L = 40, R = 90, T = 10, B = 24;
  const x = scale(0, n - 1, L, W - R), y = scale(0, 1, H - B, T);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Median ratio of the blind maximum’s score to its T15 score, by checkpoint" });
  for (const v of [0, 0.3, 0.5, 1]) svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v), "stroke-dasharray": v === 0.3 ? "4 3" : null }), s("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, v));
  const series = [["all", cb.blind_max.median_ratio_curve, "var(--accent)", 2.2], ...Object.entries(cb.by_class).map(([k, v]) => [k, v.blind_max.median_ratio_curve, "var(--ink-3)", 1.2])];
  const ends = [];
  for (const [k, ys, col, w] of series) {
    svg.append(s("path", { d: ys.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: col, "stroke-width": w }));
    ends.push([k, y(ys.at(-2)), col]);
  }
  ends.sort((a, b) => a[1] - b[1]);
  let prev = -Infinity;
  for (const [k, yy, col] of ends) { const v = Math.max(yy, prev + 12); svg.append(s("text", { x: W - R + 8, y: v + 4, style: `fill: ${col}` }, k)); prev = v; }
  for (let i = 0; i < n; i++) svg.append(s("text", { x: x(i), y: H - 6, "text-anchor": "middle" }, `T${i}`));
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), k = Math.max(0, Math.min(n - 1, Math.round((((e.clientX - r.left) / r.width) * W - L) / ((W - R - L) / (n - 1)))));
    tip(`<b>T${k}</b>: median ratio ${series.map(([c, ys]) => `${c} ${sig(ys[k], 2)}`).join(", ")}; ${cb.blind_max.first_t_counts[`T_${k}`]} poles first reach 0.3 here`, e);
  });
  svg.addEventListener("pointerleave", untip);
  return svg;
}

function runsTable() {
  return h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Run", "runCol"), term("Unblinded", "runWhen"), term("p_u", "runPu"), term("Clusters", "runN"), "b", "H_MIG", term("Hard decoy", "runHard"), term("Without T7, T15", "runSens"), term("metrics.json", "runSha")]
      .map((t, i) => h("th", { class: [[3, 4, 5].includes(i) ? "r" : "", [4, 6, 7, 8].includes(i) ? "hide-sm" : ""].join(" ").trim() || null }, t)))),
    h("tbody", {}, RES.runs.map((r, i) => h("tr", { class: i === RES.runs.length - 1 ? "final" : null }, h("td", { class: "id" }, runName(r)),
      h("td", { class: "num soft nowrap" }, fmtDay(r.created)), h("td", { class: "num nowrap" }, `${f3(r.p_u[0])} `, h("small", { class: "soft" }, ci3(r.p_u.slice(1)))),
      h("td", { class: "r num" }, int(r.n_clusters)), h("td", { class: "r num hide-sm" }, f3(r.b)), h("td", { class: "r num" }, r.h_mig),
      h("td", { class: "num hide-sm" }, r.hard.map(f3).join(" / ")), h("td", { class: "num hide-sm" }, r.sens_p_u == null ? "–" : f3(r.sens_p_u)),
      h("td", { class: "hide-sm" }, h("span", { class: "hash" }, r.sha.slice(0, 12))))))));
}

function judgeSaw(p) {  // one parent-test judgment as the judge saw it
  const truth = p.shown_order.indexOf("true"), said = p.answer === "A" ? 0 : 1;
  return h("div", { class: "jsaw" },
    h("div", { class: "who" }, h("b", {}, p.control ? "Control" : "Late-born cluster"), ` ${p.cluster}, ${p.decoy_type} decoy ${minus(p.decoy_pole)}, record at ${tLabel(p.t_before)}. `,
      `The true record was shown as ${"AB"[truth]}; the judge answered ${p.answer}: `, h("b", {}, p.correct ? "correct" : "wrong"), said === truth ? "" : "."),
    h("pre", { class: "lang" }, p.judge_saw));
}

function vocabNote() {
  const rows = C.cells.filter((c) => c.vocab_check).map((c) => {
    const v = c.vocab_check, neg = v.prompts["-"] || {}, pos = v.prompts["+"] || {};
    return h("tr", {}, h("td", {}, c.label), h("td", { class: "r num" }, `${int(v.mirror)} of ${int(v.pairs)}`),
      h("td", { class: "r num" }, `${int(pos.own || 0)} own, ${int(pos.mirror || 0)} opposite`), h("td", { class: "r num" }, `${int(neg.own || 0)} own, ${int(neg.mirror || 0)} opposite`),
      h("td", { class: "r num" }, `${pct(v.by_sign["+"].pass / v.by_sign["+"].n)} / ${pct(v.by_sign["-"].pass / v.by_sign["-"].n)}`),
      h("td", { class: "r num hide-sm" }, `${int(v.paired["+"] || 0)} / ${int(v.paired["-"] || 0)}`));
  });
  return [h("p", {}, "The harvest’s vocab.json holds each pole in its own sign: in every neuron with both poles, the − pole’s promoted tokens are the + pole’s suppressed ones. neurons_verdict.pole_vocab swaps the − pole’s lists once more before the explainer’s prompt is written, so every negative pole’s prompt listed the opposite pole’s tokens as its most promoted and most suppressed. The second reader’s prompts carry the same swap, and the last-layer suppressor flag is computed from the swapped lists. Found by this page’s builder on 21 Oct and reported then; not in the study’s record. The counts below are read from the banked explainer requests and the frozen-θ passes; rare and declined poles count as not passing."),
    h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["Cell", "Neurons whose poles mirror", term("+ prompts show", "vcPrompt"), term("− prompts show", "vcPrompt"), "Pass, + / −", "Neuron pairs, only + / only − passes"].map((t, i) => h("th", { class: [i ? "r" : "", i === 5 ? "hide-sm" : ""].join(" ").trim() || null }, t)))),
      h("tbody", {}, rows))),
    h("p", { class: "note" }, "Within a neuron the negative pole, whose prompt was wrong, passes about as often as the positive pole, whose prompt was right, and the neurons where only one pole passed split about evenly. The two poles of a neuron are different objects, so this comparison neither bounds nor measures the effect; no run explained a negative pole with its own list. The last-layer single-token suppressor count is 0 under either view.")];
}

function resultsSection(body) {
  const q = M.q2;
  body.append(h("h3", { class: "sub-h", style: { marginTop: "8px" } }, "The prediction grid", h("small", {}, `SPEC_D1 §11, as the unblind wrote it; run ${RES.runs.length}, ${fmtDay(M.created_utc)}`)),
    h("div", { class: "doc gridmd", html: md(RES.grid_md.split("\n").filter((l) => !l.startsWith("# ")).join("\n")) }));

  body.append(h("h3", { class: "sub-h" }, "Q2: the parent test", h("small", {}, `${int(q.p_u.n_clusters)} late-born clusters on ${int(q.n_poles)} poles; hover a point`)));
  const sens = q.sensitivity_F2_excl_T7_T15;
  body.append(numberLine({ lo: 0, hi: 0.5, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], rules: [{ at: C.rules.pu[0], label: "H_RG if the upper bound is below", anchor: "end" }, { at: C.rules.pu[1], label: "H_SP if the lower bound is above" }],
    points: [{ v: q.p_u.point, ci: q.p_u.ci95, label: `p_u = ${f3(q.p_u.point)} ${ci3(q.p_u.ci95)}: ${q.p_u.rule}` },
      { v: sens.p_u.point, ci: sens.p_u.ci95, soft: true, label: `Without clusters born at T7 or T15: ${f3(sens.p_u.point)} ${ci3(sens.p_u.ci95)} (${int(sens.p_u.n_clusters)} clusters)` }] }));
  body.append(h("p", { class: "note" }, `p_u = (a_ctrl − A) / (a_ctrl − 0.5): A = ${f3(q.p_u.A)} is the judge’s accuracy on late-born clusters against random decoys, a_ctrl = ${f3(q.p_u.a_ctrl)} its accuracy on the positive controls (each pole’s main cluster against its own record at T14). `
    + `${int(q.flags.n_flagged)} late-born clusters (${pct(q.flags.share)}) are flagged. The grey point is the device-boundary sensitivity. Across ${RES.runs.length} unblind runs p_u ranged ${f3(Math.min(...RES.runs.map((r) => r.p_u[0])))}–${f3(Math.max(...RES.runs.map((r) => r.p_u[0])))}.`));

  const bo = q.birth_order;
  body.append(h("h3", { class: "sub-h" }, "Birth order", h("small", {}, `b over ${int(bo.n_poles)} poles with at least two clusters; the band is the permutation null`)));
  body.append(numberLine({ lo: 0.3, hi: 0.6, ticks: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], band: [0.3, bo.null_97_5], rules: [{ at: bo.null_mean, label: `null mean ${f3(bo.null_mean)}`, anchor: "end" }],
    points: [{ v: bo.b, ci: bo.ci95, label: `b = ${f3(bo.b)} ${ci3(bo.ci95)}; p = ${sig(bo.p, 2)}, Holm ${sig(M.holm.b, 2)}` },
      { v: sens.birth_order.b, ci: sens.birth_order.ci95, soft: true, label: `Without clusters born at T7 or T15: ${f3(sens.birth_order.b)} ${ci3(sens.birth_order.ci95)}, band to ${f3(sens.birth_order.null_97_5)}` }] }));
  body.append(h("p", { class: "note" }, `Above the band: the main cluster is born first more often than a random order gives. Drift share ${f3(bo.drift_share)}. Vestigial: v = ${f3(q.vestigial.v)} ${ci3(q.vestigial.ci95)} over ${int(q.vestigial.n_clusters)} clusters, far below the 0.5 rule. H_MIG: ${q.H_MIG.count} poles (${pct(q.H_MIG.share_of_S)} of S). `
    + `Hard decoy: accuracy ${f3(q.hard.accuracy)}, ${f3(q.hard.accuracy_on_flagged)} on flagged clusters, control ${f3(q.hard.ctrl_accuracy)}: below 0.80, so not measurable.`));

  body.append(h("h3", { class: "sub-h" }, "Births", h("small", {}, `per checkpoint, percentile reading; ${int(q.births.main.n)} main clusters (accent) and ${int(q.births.non_main.n)} others (grey)`)), birthsChart());
  const sc = q.status_counts;
  body.append(h("p", { class: "note" }, `Of ${int(q.n_clusters)} clusters: ${Object.entries(sc).filter(([, n]) => n).map(([k, n]) => `${k} ${int(n)}`).join(", ")}; ${int(q.echo_clusters)} echo clusters take no part. `
    + `Clusterer agreement between its two samples: B4 (m within 1) ${f3(q.B4_m_within_1)}, B3 late-split ${f3(q.B3_late_split_agreement)}. Open a pole of S under Poles → Audit OLMo for its clusters and judgments.`));

  const cb = q.center_birth_curve;
  body.append(h("h3", { class: "sub-h" }, "Center birth curve", h("small", {}, "exploratory; median over poles of the blind maximum’s score over its own T15 score")), centerChart());
  body.append(h("p", { class: "note" }, `The blind maximum first reaches 0.3 of its T15 score at a median ${tLabel(cb.blind_max.median_first_t)} over ${int(cb.blind_max.n_poles - cb.blind_max.n_undefined_no_T15_score)} poles (${Object.entries(cb.by_class).map(([k, v]) => `${k} ${tLabel(v.blind_max.median_first_t)}`).join(", ")}). `
    + `${int(cb.blind_max.first_t_counts.T_0)} poles are there already at T0, the untrained checkpoint. Only ${pct(cb.blind_max.share_nondecreasing)} of per-pole curves never fall by more than 0.05. A pole whose blind maximum is weak at T15 reaches 0.3 of it early, and most such poles are UNK, so the comparison by class is biased (RESULTS.md).`));

  body.append(h("h3", { class: "sub-h" }, "Q1: one job, at the frozen θs", h("small", {}, "bars are the share, the light band its Wilson 95% interval, the dark tick the essay’s estimate")), q1Table());
  const fq = fold((b) => b.append(q1Splits()));
  const bq = h("button", { class: "sec-head", style: { padding: "12px 4px" } }, h("h2", { style: { fontSize: "15px" } }, "Q1 by depth and by explainer"), h("span", { class: "sum" }, "each cell split by depth tercile and by the model that explained the pole"), chevron());
  toggler(bq, fq);
  body.append(h("div", { class: "entry" }, bq, fq.el));

  body.append(h("h3", { class: "sub-h" }, "The unblind runs", h("small", {}, "each earlier run kept beside the final one; every label is the same in all of them")), runsTable());
  body.append(h("p", { class: "note" }, `Source of every number above: bank/w5/verdict/metrics.json, sha256 ${RES.sha.slice(0, 16)}…, unblind ${M.created_utc.slice(0, 19)}Z, and the earlier runs’ own metrics.json.`));

  const parts = [
    ["The instrument’s figures", "F1–F6 as the unblind drew them", (b) => b.append(h("div", { class: "figs" }, Object.entries(RES.figures).map(([k, svg]) => h("figure", { class: "fig" }, h("div", { html: svg }), h("figcaption", {}, k.replace(/_/g, " ")))))) ],
    ["What the judge saw", `the ${RES.parent.length} parent-test samples of §10`, (b) => b.append(...RES.parent.map(judgeSaw))],
    ["Reader’s note: the − pole’s vocabulary", "every negative pole’s explainer prompt listed the opposite pole’s tokens", (b) => b.append(...vocabNote())],
  ];
  for (const [title, sum, render] of parts) {
    const f = fold(render), btn = h("button", { class: "sec-head", style: { padding: "12px 4px" } }, h("h2", { style: { fontSize: "15px" } }, title), h("span", { class: "sum" }, sum), chevron());
    toggler(btn, f);
    body.append(h("div", { class: "entry" }, btn, f.el));
  }
}

/* ---------- calibration rounds ------------------------------------------------------------- */

function roleModel(r, role) {
  const m = r.usage[role]?.models || {};
  return Object.keys(m).sort((a, b) => m[b] - m[a]).map(model).join(", ") || "–";
}
function prCell() {
  const fill = h("i", { class: "pr-fill" }), ci = h("i", { class: "pr-ci" }), val = h("span", { class: "pr-val" });
  const el = h("div", { class: "pr" }, h("div", { class: "pr-track" }, fill, ci), val);
  return { el, set(p) { fill.style.width = `${100 * p.rate}%`; ci.style.left = `${100 * p.ci[0]}%`; ci.style.width = `${100 * (p.ci[1] - p.ci[0])}%`; val.textContent = pct(p.rate); } };
}

function calibration(body) {
  body.append(h("p", { class: "lead", html: `Each round changed one thing on the same 120 Qwen3-4B-Base poles. An explanation <b>passes</b> when at least ${MIN_POS} of 5 positives are strong, at most ${MAX_NEAR} of 5 near misses and at most ${MAX_DIST} of 5 distant texts. A text is strong when its score reaches θ<sub>s</sub>. Each round set its own θ<sub>s</sub> from its own texts on shuffled poles; those are the small ticks on the rail.` }));
  let curves, follow;
  const tbody = h("tbody");
  const table = h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Round", "round"), term("What changed", "changed"), term("Explainer", "explainerCol"), term("Writer", "writerCol"), term("Own θs", "ownTheta"), term("Pass rate at the live θs", "pass"),
      term("Positives", "pos"), term("Near", "near"), term("Distant", "distant"), term("Shuffled", "shuffled")]
      .map((t, i) => h("th", { class: [i >= 4 && i !== 5 ? "r" : "", [1, 3, 4, 7, 8, 9].includes(i) ? "hide-sm" : ""].join(" ").trim() || null }, t)))), tbody);
  for (const r of C.rounds) {
    const pr = prCell(), rates = ["pos", "near", "distant", "sh"].map((k, i) => h("td", { class: `r num${i ? " hide-sm" : ""}` }));
    const row = h("tr", { class: "row" },
      h("td", { class: "id" }, r.id, r.id === C.frozen_round ? h("span", { class: "frozen-mark" }, "frozen") : null),
      h("td", { class: "soft hide-sm" }, r.note), h("td", { class: "nowrap" }, roleModel(r, "explainer")), h("td", { class: "nowrap hide-sm" }, roleModel(r, "writer")),
      h("td", { class: "r num hide-sm" }, th4(r.theta)),
      h("td", {}, pr.el, h("small", { class: "pr-own" }, `${pct(passRate(recsOf(r), r.theta).rate)} at its own θs`)), ...rates);
    row.addEventListener("pointerenter", () => follow?.(r.id));
    row.addEventListener("pointerleave", () => follow?.(null));
    const f = fold((b) => roundDetail(b, r));
    const detail = h("tr", { class: "detail" }, h("td", { colspan: 10 }, f.el));
    clickable(row, (open) => f.set(open));
    tbody.append(row, detail);
    onTheta((t) => {
      const recs = recsOf(r);
      pr.set(passRate(recs, t));
      const sr = strongRates(recs, t);
      ["pos", "near", "distant", "sh"].forEach((k, i) => { rates[i].textContent = pct(sr[k]); });
    });
  }
  for (const r of C.cancelled_rounds) {
    tbody.append(h("tr", {}, h("td", { class: "id muted" }, r.id), h("td", { class: "muted", colspan: 9 }, `${model(r.model)}. ${r.note}`)));
  }
  body.append(h("div", { class: "wrap-x" }, table));
  body.append(h("p", { class: "note", html: "Positives, near misses, distant and shuffled: the share of those texts that are strong at the live θ<sub>s</sub>. Shuffled scores each text at another random pole of the same layer; its 99th percentile is how a round sets its own θ<sub>s</sub>. Rounds d1, d2 and d2w were scored before text position 0 stopped counting; d2w-r is d2w's texts under the later rule. r13 forks d3: the same frozen Opus 5.5 explanations, with texts written again by Sonnet 5.5, the configuration of 1,344 of the 1,840 audit poles." }));

  const kinds = kindsChart(), kindTitle = h("small", {});
  follow = (id) => { curves.highlight(id); kinds.show(id || C.frozen_round); kindTitle.textContent = `round ${id || C.frozen_round}`; };
  curves = curvesChart(follow);
  body.append(h("div", { class: "pair" },
    h("div", {}, h("h3", { class: "sub-h" }, "Pass rate as θs moves", h("small", {}, "hover a round to follow it")), curves.el),
    h("div", {}, h("h3", { class: "sub-h" }, "Strong share by kind", kindTitle), kinds.el)),
    h("div", { class: "legend" }, h("span", {}, h("i", { class: "dash" }), `frozen θs ${th4(FROZEN)}`),
      h("span", {}, h("i", { class: "line" }), "live θs"), h("span", {}, "Dots: each round at its own θs")));
  follow(null);

  body.append(h("h3", { class: "sub-h" }, "Every pole in every round", h("small", {}, "columns are poles in layer order; click one to open it")));
  body.append(h("div", { class: "field" }, passField((p) => openPole(p))),
    h("div", { class: "legend" }, h("span", {}, h("i", { class: "p" }), "passes"), h("span", {}, h("i", { class: "f" }), "fails"),
      h("span", {}, h("i", { class: "n" }), "rare or declined, not scored"), h("span", {}, "Right column: pass rate")));
}

function roundDetail(body, r) {
  const panel = h("div", { class: "panel" });
  body.append(panel);
  const roles = Object.entries(r.usage);
  const from = r.forked && C.rounds.find((q) => `bank/${q.explain}/explain` === r.forked.explain);
  const carried = (role) => r.forked && !r.forked.dropped_batches.some((b) => b.endsWith(`-${role}`));
  const uTable = h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Role", "role"), "Model", term("Calls", "calls"), term("Input tokens", "tokIn"), term("Output tokens", "tokOut"), term("Time", "time")].map((t, i) => h("th", { class: i >= 2 ? "r" : null }, t)))),
    h("tbody", {}, roles.map(([role, u]) => h("tr", {}, h("td", {}, ROLE[role] || role), h("td", {}, Object.keys(u.models).map(model).join(", "),
      carried(role) ? h("small", { class: "soft" }, ` carried over from ${from ? from.id : r.forked.explain}`) : null),
      h("td", { class: "r num" }, int(u.calls)), h("td", { class: "r num" }, int(u.in)), h("td", { class: "r num" }, int(u.out)),
      h("td", { class: "r num" }, `${Math.round(u.s / 60)} min`)))));
  const classes = h("div");
  onTheta((t) => {
    const cnt = Object.fromEntries(CLASSES.map((c) => [c, 0]));
    for (const rec of recsOf(r)) cnt[klass(rec, t)]++;
    const n = recsOf(r).length;
    classes.replaceChildren(h("div", { class: "cbars" }, CLASSES.map((c) => h("div", { class: "cbar" }, term(c, c),
      h("span", { class: "cbar-track" }, h("i", { style: { width: `${(100 * cnt[c]) / n}%` } })), h("span", { class: "num" }, cnt[c])))));
  });
  const essay = Counter(recsOf(r).map((x) => x.essay).filter(Boolean));
  panel.append(
    h("div", { class: "cols" },
      h("div", {}, h("h3", { class: "sub-h", style: { marginTop: 0 } }, "Calls"), uTable),
      h("div", {}, h("h3", { class: "sub-h", style: { marginTop: 0 } }, "Classes at the live θs"), classes,
        h("h3", { class: "sub-h" }, "The essay’s reading"),
        h("p", { class: "soft", style: { margin: 0 } }, `Strong at a fixed 0.30, positives against near misses: won ${essay.win || 0}, tied ${essay.tie || 0}, lost ${essay.lose || 0}. Reported, never used to decide.`))),
    h("h3", { class: "sub-h" }, "Provenance"),
    h("dl", { class: "kv" },
      h("dt", {}, "Prompt"), h("dd", {}, `${r.prompt || "–"}: `, ...Object.entries(r.prompts).filter(([k]) => ["explainer", "writer", "writer_job"].includes(k))
        .flatMap(([k, v], i) => [i ? ", " : "", `${ROLE[k]} `, h("span", { class: "hash" }, v.slice(0, 12))])),
      h("dt", {}, "Explain directory"), h("dd", {}, `bank/${r.explain}, started ${r.started.slice(0, 16).replace("T", " ")}Z`),
      r.forked ? [h("dt", {}, "Fork"), h("dd", {}, `Of ${r.forked.explain}${from ? ` (round ${from.id})` : ""} at ${r.forked.utc.slice(0, 16).replace("T", " ")}Z, without ${r.forked.dropped}.`)] : null,
      h("dt", {}, "Probe job"), h("dd", {}, `${r.probe} on ${r.device || "–"}; ${r.probe_rule}`),
      h("dt", {}, "Own θs"), h("dd", {}, `${th4(r.theta)}, the 99th percentile of ${int(r.theta_n)} shuffled scores`),
      h("dt", {}, "Record"), h("dd", {}, `JOURNAL ${r.journal}`)));
}
function Counter(xs) { const c = {}; for (const x of xs) c[x] = (c[x] || 0) + 1; return c; }

/* ---------- poles -------------------------------------------------------------------------- */

const cellOf = (k) => C.cells.find((c) => c.key === k);
function poleRounds(key) {
  if (key.includes(":")) {
    const [ck, id] = key.split(":");
    return [{ id: "W2-C", rec: cellOf(ck).poles[id], wfile: `words_${ck}`, theta: FROZEN, frozen: true }];
  }
  return C.rounds.map((r) => ({ id: r.id, rec: r.poles[key], wfile: "words", wkey: r.id, theta: r.theta, frozen: r.id === C.frozen_round }));
}
const wordsOf = (W, r, id) => (r.wkey ? W[r.wfile]?.[r.wkey] : W[r.wfile])?.[id];
const poleId = (key) => (key.includes(":") ? key.split(":")[1] : key);
const rowsByKey = {};
let poleView;

function polesSection(body) {
  body.append(h("p", { class: "lead", html: `Every pole the study explained: the discovery set, S_gate and the ${C.sealed ? "audit poles it has not yet read" : "1,800 audit poles, open since W5; OLMo’s 600 are S, each with its clusters, births and judgments"}. Open a pole to see its activations over the corpus, the words it promotes, each round’s explanation and test texts with their scores, and the 30 contexts the explainer saw.` }));
  const tabs = [
    ["disc", `Discovery ${Object.keys(C.rounds[0].poles).length}`],
    ["sgate", `S_gate ${C.grid.s_gate.length}`],
    ...C.cells.map((c) => [`audit:${c.key}`, `Audit ${c.key} ${c.n_audit}`]),
  ];
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Pole set" });
  const filter = h("input", { class: "filter", placeholder: "Filter by id, layer or words", "aria-label": "Filter poles" });
  const list = h("div");
  body.append(h("div", { class: "toolbar" }, seg, filter), list);
  const views = {};
  function show(tab) {
    for (const b of seg.children) b.setAttribute("aria-pressed", String(b.dataset.tab === tab));
    views[tab] ||= buildList(tab);
    list.replaceChildren(views[tab].el);
    views[tab].filter(filter.value);
    poleView = tab;
  }
  for (const [tab, label] of tabs) seg.append(h("button", { "data-tab": tab, onclick: () => show(tab) }, label));
  filter.addEventListener("input", () => views[poleView].filter(filter.value));
  show("disc");
  polesSection.show = show;
}

function buildList(tab) {
  const cell = tab.startsWith("audit:") ? cellOf(tab.split(":")[1]) : null, sealedCell = C.sealed ? cell : null;
  const keys = tab === "disc" ? Object.keys(C.rounds[0].poles)
    : tab === "sgate" ? C.grid.s_gate.map((p) => `O7:${p}`)
    : Object.keys(cell.audit).map((p) => `${cell.key}:${p}`);
  keys.sort((a, b) => { const p = parsePole(poleId(a)), q = parsePole(poleId(b)); return p.layer - q.layer || p.neuron - q.neuron || a.localeCompare(b); });
  const head = tab === "disc" ? [term("Pole", "poleId"), term("Layer", "layer"), term(`Explanation in ${C.frozen_round}`, "explanation"), term("max |a|", "max"), term("Density", "density"), term("Rounds passing", "rounds"), term(`Class in ${C.frozen_round}`, "klass")]
    : !sealedCell ? [term("Pole", "poleId"), term("Layer", "layer"), term("Explanation", "explanation"), term("max |a|", "max"), term("Density", "density"), term("Passes", "sgatePass"), term("Class", "klass")]
    : [term("Pole", "poleId"), term("Layer", "layer"), "", term("max |a|", "max"), term("Density", "density"), term("Double-negative", "dn"), term("Result", "sealed")];
  const tbody = h("tbody");
  const table = h("table", { class: "tbl poles" }, h("thead", {}, h("tr", {}, head.map((t, i) => h("th", { class: [1, 3, 4].includes(i) ? "r hide-sm" : i === 2 ? "hide-sm" : null }, t)))), tbody);
  const items = [];
  for (const key of keys) {
    const id = poleId(key);
    const f = cell ? cell.audit[id] : C.poles[key];
    const nTok = cell ? cell.n_tokens : C.models.find((x) => x.key === f.model).n_tokens_scored;
    const ex = h("span", { class: "lang" }, sealedCell ? "" : "…");
    const cells = [h("td", { class: "id" }, minus(id)), h("td", { class: "r num hide-sm" }, parsePole(id).layer), h("td", { class: "ex hide-sm" }, ex),
      h("td", { class: "r num hide-sm" }, sig(f.max)), h("td", { class: "r num soft hide-sm" }, oneIn(nTok, f.density))];
    let update = null;
    if (sealedCell) {
      cells.push(h("td", { class: "num" }, sig(f.dn, 2)), h("td", {}, h("span", { class: "sealed" }, lock(), "sealed")));
    } else {
      const rounds = poleRounds(key);
      const minis = h("span", { class: "minis" }, rounds.map(() => h("i")));
      const n = h("span", { class: "num soft passn" }), cls = h("span", { class: "cls" });
      cells.push(h("td", {}, minis, n), h("td", {}, cls));
      update = (t) => {
        let k = 0;
        rounds.forEach((r, i) => { const p = passes(r.rec, t); minis.children[i].className = r.rec.rare || p === null ? "n" : p ? "p" : ""; k += !!p; });
        n.textContent = rounds.length > 1 ? `${k}/${rounds.length}` : "";
        const fr = rounds.find((r) => r.frozen);
        const c = klass(fr.rec, t);
        cls.textContent = c; cls.className = `cls term ${c.startsWith("UNK") ? "UNK" : c}`; cls.dataset.g = c; cls.tabIndex = 0;
      };
    }
    const row = h("tr", { class: "row", "data-key": key }, cells);
    const fd = fold((b) => (sealedCell ? sealedDetail(b, sealedCell, id) : poleDetail(b, key)));
    const detail = h("tr", { class: "detail" }, h("td", { colspan: head.length }, fd.el));
    const toggle = clickable(row, (open) => { setHash(open ? key : null, key); return fd.set(open); });
    tbody.append(row, detail);
    const item = { key, row, detail, update, ex, text: `${id} ${minus(id)} layer ${parsePole(id).layer}`.toLowerCase() };
    items.push(item);
    rowsByKey[key] = { toggle, row, get open() { return fd.open; } };
  }
  if (!sealedCell) {
    load(tab === "disc" ? "words" : `words_${cell ? cell.key : "O7"}`).then((w) => {
      for (const it of items) {
        const e = (tab === "disc" ? w[C.frozen_round] : w)?.[poleId(it.key)]?.e;
        it.ex.textContent = e || "declined";
        it.ex.title = e || "";
        it.text += ` ${(e || "").toLowerCase()}`;
      }
    });
  }
  let alive = true;
  onTheta((t) => { if (!alive) return false; for (const it of items) it.update?.(t); });
  return {
    el: h("div", { class: "wrap-x" }, table),
    filter(q) {
      const terms = q.toLowerCase().trim().split(/\s+/).filter(Boolean);
      for (const it of items) {
        const show = terms.every((w) => it.text.includes(w));
        it.row.style.display = show ? "" : "none"; it.detail.style.display = show ? "" : "none";
      }
    },
  };
}

function sealedDetail(body, cell, id) {
  const f = cell.audit[id];
  body.append(h("div", { class: "panel" },
    h("div", { class: "facts" }, factsOf(f, cell.n_tokens).map(([v, l]) => h("div", { class: "fact" }, h("b", {}, v), h("span", {}, l)))),
    h("div", { class: "hatch sealbox" }, h("span", { class: "sealed" }, lock(), `This pole’s explanation, texts, scores and class are sealed until W5 (SPEC_D1 §9.3). Its harvest statistics above are open; they are what drew S.`))));
}
function factsOf(f, nTok) {
  return [[sig(f.max), term("max |a|", "max")], [sig(f.q999), term("q99.9 of |a|", "q999")], [int(f.n), term("tokens on this pole", "ntok")],
    [oneIn(nTok, f.density), term("density, tokens", "density")], [sig(f.dn, 2), term("double-negative share", "dn")],
    [sig(f.same, 2), term("same-token share", "same")], [sig(f.pos0, 2), term("at window position 0", "pos0")]];
}

const SET_NOTE = { S_gate: "S_gate: tunes the clusterer and judge at W4.", discovery: "Discovery set: every prompt and model was tuned here.",
  audit: "Audit pole, explained once with the frozen prompts.", S: "Audit pole in S, the development sample: its clusters, births and judgments are below the test." };
async function poleDetail(body, key) {
  const id = poleId(key), f = C.poles[key] || (await load("audit"))[key], mm = C.models.find((m) => m.key === f.model);
  const { layer, neuron } = parsePole(id), inS = f.set === "audit" && C.grid.s.includes(id) && !C.sealed;
  const panel = h("div", { class: "panel" });
  body.append(panel);
  panel.append(
    h("p", { class: "soft", style: { margin: "0 0 14px" } }, `${mm.label}, layer ${layer} of ${mm.n_layers}, neuron ${neuron}, ${signWord(id)} pole. ${SET_NOTE[inS ? "S" : f.set]}`),
    h("div", { class: "facts" }, factsOf(f, mm.n_tokens_scored).map(([v, l]) => h("div", { class: "fact" }, h("b", {}, v), h("span", {}, l)))),
    h("div", { class: "cols" },
      h("div", {}, h("h3", { class: "sub-h", style: { marginTop: 0 } }, "|a| over the corpus", h("small", {}, "tokens per bin, log scales")), histogram(f, C.hist_edges[f.model])),
      h("div", {}, h("h3", { class: "sub-h", style: { marginTop: 0 } }, "What it writes", h("small", {}, "unembedding of its output direction")),
        h("p", { class: "muted", style: { margin: "0 0 6px", fontSize: "12.5px" } }, "Promotes"),
        h("div", { class: "vocab" }, f.up.map(([t, v]) => tokChip(t, v))),
        h("p", { class: "muted", style: { margin: "12px 0 6px", fontSize: "12.5px" } }, "Suppresses"),
        h("div", { class: "vocab" }, f.down.map(([t, v]) => tokChip(t, v))),
        id.endsWith("-") ? h("p", { class: "note" }, "This is the pole’s own view. Its explainer prompt listed the two lists the other way round: Results, reader’s note.") : null)));

  const rounds = poleRounds(key), W = {};
  for (const name of new Set(rounds.map((r) => r.wfile))) W[name] = await load(name);
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Round" });
  const roundBox = h("div");
  panel.append(h("h3", { class: "sub-h" }, rounds.length > 1 ? "Explanations and tests, by round" : "Explanation and test"),
    h("div", { class: "toolbar", style: { marginBottom: 0 } }, seg, h("span", { class: "legend", style: { margin: 0 } }, h("span", {}, h("i", { class: "p" }), "passes at the live θs"))), roundBox);
  let offRound = null;
  const pick = (r) => {
    for (const b of seg.children) b.setAttribute("aria-pressed", String(b.dataset.r === r.id));
    offRound?.();
    roundBox.replaceChildren();
    offRound = roundView(roundBox, r, wordsOf(W, r, id));
  };
  for (const r of rounds) {
    const dot = h("i", { class: "dot" });
    seg.append(h("button", { "data-r": r.id, onclick: () => pick(r) }, r.id, dot));
    onTheta((t) => { dot.className = `dot${passes(r.rec, t) ? " on" : ""}`; });
  }
  pick(rounds.find((r) => r.frozen) || rounds[0]);

  if (inS) await q2Panel(panel, id);
  panel.append(h("h3", { class: "sub-h" }, "Top 30 contexts", h("small", {}, "as the explainer saw them; click one for its full text")));
  const ctxBox = h("div", { class: "kwic" }, h("p", { class: "loading" }, "Loading contexts…"));
  panel.append(ctxBox);
  const ctx = (await load(f.set === "audit" ? `ctx_${f.model}_L${String(layer).padStart(2, "0")}` : "contexts"))[key];
  ctxBox.replaceChildren(...(ctx || []).map((c, i) => kwic(c, i)));
}

function tokChip(t, v) {
  const lead = t.startsWith(" ");
  return h("span", { class: "tok" }, lead ? h("span", { class: "sp term", "data-g": "space" }, "␣") : null, lead ? t.slice(1) : t, h("small", {}, sig(v, 2)));
}

function roundView(box, r, w) {
  const rec = r.rec, off = [];
  const listen = (fn) => off.push(onTheta(fn));
  if (!w) { box.append(h("p", { class: "muted" }, "No words banked for this pole in this round.")); return () => {}; }
  box.append(h("p", { class: "expl lang" }, w.e || "The explainer declined."),
    h("div", { class: "who" }, h("b", {}, model(rec.model)), ` labelled it ${rec.label}`, rec.conf != null ? `, confidence ${rec.conf}` : "",
      rec.declined ? `. Declined: ${rec.declined}` : ""));
  if (w.jobs.length > 1) {
    box.append(h("ul", { class: "jobs" }, w.jobs.map(([name, why, rel]) => h("li", {}, name, h("small", {}, `${rel ? "related to the others" : "unrelated to the others"}${why ? `. ${why}` : ""}`)))));
  }
  if (w.e2 || rec.declined2) {
    box.append(h("p", { class: "muted", style: { margin: "0 0 4px", fontSize: "12.5px" } }, `Second reader, ${model(rec.model2)}, labelled it ${rec.label2 || "–"}${w.conf2 != null ? `, confidence ${w.conf2}` : ""}`),
      h("p", { class: "second lang" }, w.e2 || `Declined: ${rec.declined2}`));
  }
  if (!rec.s) { box.append(h("p", { class: "muted" }, rec.rare ? "Rare pole: not scored." : "No texts: the pole is UNK-N.")); return () => off.forEach((f) => f()); }

  const verdict = h("div", { style: { margin: "20px 0 8px" } });
  const clause = (n, label, need, ok) => h("span", { class: `clause${ok ? " ok" : " no"}` }, h("i", {}, ok ? "✓" : "✕"),
    h("b", { class: "num" }, `${n}/5`), ` ${label} `, h("small", {}, need));
  listen((t) => {
    const p = passes(rec, t), c = klass(rec, t), np = strongN(rec.s.pos, t), nn = strongN(rec.s.near, t), nd = strongN(rec.s.distant, t);
    verdict.replaceChildren(
      h("div", { class: "verdict-line" }, h("span", { class: `verdict${p ? " pass" : ""}` }, h("i"), p ? "Passes" : "Fails"),
        clause(np, "positives strong", "need ≥ 3", np >= MIN_POS), clause(nn, "near misses strong", "allow ≤ 2", nn <= MAX_NEAR),
        clause(nd, "distant strong", "allow ≤ 1", nd <= MAX_DIST)),
      h("div", { class: "who", style: { margin: "6px 0 0" } }, "Class ", term(c, c), ` at θs ${th4(t)}${Math.abs(t - r.theta) < 5e-5 && !r.frozen ? ", this round’s own θs" : ""}.`));
  });
  const lines = {};
  const strip = scoreStrip(rec, (k, on) => lines[k]?.classList.toggle("hl", on));
  box.append(verdict, h("div", { style: { maxWidth: "680px" } }, strip));

  const texts = h("div", { class: "texts" });
  const groups = ["pos", "near", "distant", "max", ...Object.keys(rec.s).filter((k) => k.startsWith("job_pos:"))];
  for (const kind of groups) {
    const toks = w.t?.[kind] || [];
    const title = kind.startsWith("job_pos:") ? `Job ${+kind.split(":")[1] + 1}` : KIND[kind];
    const hint = { pos: "should fire", near: "should not fire", distant: "most unrelated concept", max: "fire as hard as possible" }[kind] || "written from the job alone";
    const g = h("div", { class: "tgroup" }, h("h5", {}, KIND[kind] && kind !== "max" ? term(title, kind) : title, h("small", {}, hint)), h("div"));
    rec.s[kind].forEach((v, i) => {
      const [tk, pk] = toks[i] || [[], -1];
      const text = h("div", { class: "ttext lang" }, tk.map((t, j) => {
        if (j !== pk) return t;
        const lead = t.match(/^\s*/)[0], word = t.slice(lead.length) || t;
        return [lead.length < t.length ? lead : "", h("span", { class: "pk term", "data-g": "peak" }, word)];
      }));
      const bar = h("i"), tick = h("u");
      const line = h("div", { class: "tline" }, text, h("div", { class: "tscore" }, h("div", { class: "tbar" }, bar, tick), h("span", { class: "tval" }, v.toFixed(2))));
      bar.style.width = `${Math.max(0, Math.min(100, (v / 1.3) * 100))}%`;
      listen((t) => { line.classList.toggle("strong", v >= t); tick.style.left = `${(t / 1.3) * 100}%`; });
      lines[`${kind}:${i}`] = line;
      line.addEventListener("pointerenter", () => strip.dots[`${kind}:${i}`]?.classList.add("focus"));
      line.addEventListener("pointerleave", () => strip.dots[`${kind}:${i}`]?.classList.remove("focus"));
      g.lastChild.append(line);
    });
    texts.append(g);
  }
  box.append(texts, h("p", { class: "note" }, `Shuffled: the same positives, near misses and distant texts scored at ${minus(rec.sp)}, a random pole of the same layer. Scores are the text’s peak activation at the pole over its corpus max; the highlighted token is where it peaked.`));
  return () => off.forEach((f) => f());
}

function kwic(c, i) {
  const [a, g, u, doc, pos, before, token, after] = c;
  const one = (t) => t.replace(/\n/g, "↵");
  const row = h("div", { class: "kw" },
    h("span", { class: "rk" }, i + 1),
    h("span", { class: "l" }, h("span", {}, one(before).slice(-220))),
    h("span", { class: "m" }, one(token)),
    h("span", { class: "r" }, one(after)),
    h("span", { class: "a" }, `${a > 0 ? "+" : ""}${sig(a, 3)}`));
  let full = null;
  row.addEventListener("click", () => {
    if (full) { full.remove(); full = null; return; }
    full = h("div", { class: "kw-full" }, before, h("span", { class: "m" }, token), after,
      h("small", {}, `a ${sig(a, 4)}, g ${sig(g, 3)}, u ${sig(u, 3)} (${g >= 0 ? "g+" : "g−"} ${u >= 0 ? "u+" : "u−"}). Document ${doc}, token ${pos}.`));
    row.append(full);
  });
  return row;
}

/* one pole of S over training: its exemplars, its own texts, its clusters and their judgments */
async function q2Panel(panel, id) {
  const tr = C.grid.traj[id], cen = C.grid.center[id], cl = (await load("clusters"))[id] || [];
  panel.append(h("h3", { class: "sub-h" }, "Over training", h("small", {}, `OLMo-3’s ${C.grid.checkpoints.length} stage-1 checkpoints, T0 to T15`)),
    h("div", { class: "cols" },
      h("div", {}, h("h4", { class: "mini-h" }, "The pole’s 50 T15 exemplars", h("small", {}, "median a⁺ over q99.9; filled dots are present")), spark(id, tr, false)),
      h("div", {}, h("h4", { class: "mini-h" }, "Its own texts", h("small", {}, "center probe: score at each checkpoint")), cen ? centerPole(cen) : h("p", { class: "muted" }, "No texts: the pole has no blind maximum and five positives."))));
  const rows = cl.map((c) => {
    const strip = h("span", { class: "strip" }, [...c.pct].map((b, t) => {
      const cellEl = h("i", { class: b === "1" ? "on" : c.raw[t] === "1" ? "raw" : null });
      cellEl.addEventListener("pointerenter", (e) => tip(`<b>T${t}</b>: median a⁺ ${sig(c.med[t])}; ${b === "1" ? "present" : "not present"} (percentile), ${c.raw[t] === "1" ? "present" : "not present"} (0.3 × max)`, e));
      cellEl.addEventListener("pointerleave", untip);
      return cellEl;
    }));
    const ctl = c.j.filter((j) => j[7]), tests = c.j.filter((j) => !j[7]);
    const judge = c.acc[0] != null ? `${sig(c.acc[0], 2)} / ${sig(c.acc[1], 2)}` : ctl.length ? `control ${ctl.filter((j) => j[6]).length}/${ctl.length}` : "–";
    const row = h("tr", { class: c.j.length ? "row" : null }, h("td", { class: "lang" }, c.name, c.main ? h("span", { class: "frozen-mark" }, "main") : null, c.echo ? h("span", { class: "frozen-mark" }, "echo") : null),
      h("td", { class: "r num hide-sm" }, c.ranks.length), h("td", {}, strip),
      h("td", { class: "num nowrap" }, c.bp == null ? "–" : `T${c.bp}`, h("small", { class: "soft" }, c.br == null ? " [–]" : ` [T${c.br}]`)),
      h("td", { class: "soft hide-sm" }, defined(c.status, GLOSS.clStatus)), h("td", { class: "num nowrap" }, judge), h("td", { class: "r num hide-sm" }, c.r == null ? "–" : sig(c.r, 2)));
    if (!c.j.length) return [row];
    const f = fold((b) => b.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["", "Decoy", "Record at", "Sample", "Shown", "Answer"].map((x) => h("th", {}, x)))),
      h("tbody", {}, [...tests, ...ctl].map(([type, dp, tb, smp, shown, ans, ok, control]) => h("tr", {}, h("td", {}, control ? "control" : "test"),
        h("td", {}, `${type} `, h("span", { class: "soft" }, minus(dp))), h("td", { class: "num" }, tLabel(tb)), h("td", { class: "num" }, smp),
        h("td", { class: "soft" }, shown.map((x, k) => `${"AB"[k]} ${x}`).join(", ")), h("td", {}, `${ans} `, h("b", {}, ok ? "correct" : "wrong")))))))));
    const detail = h("tr", { class: "detail" }, h("td", { colspan: 7 }, f.el));
    clickable(row, (open) => f.set(open));
    return [row, detail];
  });
  panel.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl clusters" },
    h("thead", {}, h("tr", {}, [term("Cluster", "clName"), term("Exemplars", "clN"), term("Present T0–T15", "clPresent"), term("Birth", "clBirth"), term("Status", "clStatus"), term("Judge", "clJudge"), term("r at birth", "clR")]
      .map((x, i) => h("th", { class: [[1, 6].includes(i) ? "r" : "", [1, 4, 6].includes(i) ? "hide-sm" : ""].join(" ").trim() || null }, x)))),
    h("tbody", {}, rows.flat()))),
    h("p", { class: "note" }, `${cl.length} clusters from the clusterer’s first sample. Click a judged cluster for its judgments: random and hard decoys at the record before its birth, and the controls on the main cluster.`));
}
function centerPole(cen) {
  const n = cen.max.length, W = 480, H = 120, L = 30, R = 8, T = 8, B = 20;
  const top = Math.max(0.05, ...cen.max, ...cen.pos);
  const x = scale(0, n - 1, L, W - R), y = scale(0, top, H - B, T);
  const end = cen.max.at(-1), first = end > 0 ? cen.max.findIndex((v) => v >= 0.3 * end) : -1;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Blind maximum and positives’ mean score at each checkpoint", "data-noterm": "" });
  for (const v of [0, top / 2, top]) svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }), s("text", { x: L - 4, y: y(v) + 4, "text-anchor": "end" }, sig(v, 2)));
  if (end > 0) svg.append(s("line", { x1: L, x2: W - R, y1: y(0.3 * end), y2: y(0.3 * end), stroke: "var(--ink-3)", "stroke-dasharray": "4 3" }));
  for (const [k, col, w] of [["pos", "var(--ink-3)", 1.4], ["max", "var(--accent)", 2]]) svg.append(s("path", { d: cen[k].map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: col, "stroke-width": w }));
  if (first >= 0) svg.append(s("circle", { cx: x(first), cy: y(cen.max[first]), r: 4, fill: "var(--accent)", stroke: "var(--bg)", "stroke-width": 1.5 }));
  for (const i of [0, 5, 10, 15]) svg.append(s("text", { x: x(i), y: H - 4, "text-anchor": "middle" }, `T${i}`));
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), k = Math.max(0, Math.min(n - 1, Math.round((((e.clientX - r.left) / r.width) * W - L) / ((W - R - L) / (n - 1)))));
    tip(`<b>T${k}</b>: blind maximum ${sig(cen.max[k], 3)}, positives’ mean ${sig(cen.pos[k], 3)}${k === first ? "; first at 0.3 of its T15 score" : ""}`, e);
  });
  svg.addEventListener("pointerleave", untip);
  return h("div", {}, svg, h("div", { class: "legend" }, h("span", {}, h("i", { class: "line" }), term("blind maximum", "centerMax")), h("span", {}, h("i", { class: "line soft" }), term("positives’ mean", "centerPos")),
    h("span", {}, h("i", { class: "dash" }), first >= 0 ? `0.3 of its T15 score, first reached at T${first}` : "no positive T15 score")));
}

function setHash(key, owner) {
  if (key) history.replaceState(null, "", `#${encodeURIComponent(key)}`);
  else if (decodeURIComponent(location.hash.slice(1)) === owner) history.replaceState(null, "", location.pathname + location.search);
}
function poleKeyKnown(key) {
  return key in C.poles || C.cells.some((c) => key.startsWith(`${c.key}:`) && key.slice(c.key.length + 1) in c.audit);
}

async function openPole(key) {
  await sections.poles.open();
  const tab = key.includes(":") ? (key.startsWith("O7:") && C.grid.s_gate.includes(poleId(key)) ? "sgate" : `audit:${key.split(":")[0]}`) : "disc";
  if (poleView !== tab) polesSection.show(tab);
  const r = rowsByKey[key];
  if (!r) return;
  await r.toggle(true);
  r.row.scrollIntoView({ behavior: "smooth", block: "start" });
  r.row.classList.remove("flash"); void r.row.offsetWidth; r.row.classList.add("flash");
}

/* ---------- prompts ------------------------------------------------------------------------ */

function usedBy(sha) {
  const out = C.rounds.filter((r) => Object.values(r.prompts).includes(sha)).map((r) => r.id);
  for (const c of C.cells) if (Object.values(c.prompts).includes(sha)) out.push(`W2-C ${c.key}`);
  return out;
}
function promptsSection(body) {
  body.append(h("p", { class: "lead", html: "Every committed version of every role’s prompt, by SHA-256. The frozen set is prompt v3 for the explainer, second reader, writer and job writer; the clusterer and judge are tuned on S_gate at W4." }));
  const byRole = {};
  for (const p of Object.values(C.prompts)) (byRole[p.role] ||= []).push(p);
  for (const [role, versions] of Object.entries(byRole)) {
    versions.sort((a, b) => a.when.localeCompare(b.when));
    const head = h("button", { class: "sec-head", style: { padding: "14px 4px" } },
      h("h2", { style: { fontSize: "15px" } }, ROLE[role]),
      h("span", { class: "sum" }, versions.map((v, i) => [i ? ", " : "", h("b", {}, v.version || v.sha.slice(0, 8)), " ", h("span", { class: "hash" }, v.sha.slice(0, 8))])),
      chevron());
    const f = fold((b) => promptDetail(b, versions));
    toggler(head, f);
    body.append(h("div", { class: "entry" }, head, f.el));
  }
}
function promptDetail(body, versions) {
  const seg = h("div", { class: "seg" }), diffBtn = h("button", { "aria-pressed": "false" }, "Show changes from the previous version");
  const meta = h("p", { class: "note" }), text = h("div", { class: "prompt" }), key = h("p", { class: "diffkey" });
  let cur = versions.length - 1, diff = false;
  const draw = () => {
    for (const [i, b] of [...seg.children].entries()) b.setAttribute("aria-pressed", String(i === cur));
    diffBtn.disabled = cur === 0;
    diffBtn.setAttribute("aria-pressed", String(diff && cur > 0));
    const v = versions[cur], used = usedBy(v.sha);
    meta.replaceChildren(`SHA-256 `, h("span", { class: "hash" }, v.sha), `. First committed in ${v.commit}, ${v.when.slice(0, 16).replace("T", " ")}. ${used.length ? `Used by ${used.join(", ")}.` : "Not used by any banked round."}`);
    const showing = diff && cur > 0;
    key.replaceChildren(...(showing ? [h("span", { class: "add" }, "new in this version"), h("span", { class: "del" }, `removed since ${versions[cur - 1].version || versions[cur - 1].sha.slice(0, 8)}`)] : []));
    text.replaceChildren(...(showing ? lineDiff(versions[cur - 1].text, v.text) : [v.text]));
  };
  versions.forEach((v, i) => seg.append(h("button", { onclick: () => { cur = i; draw(); } }, v.version || v.sha.slice(0, 8))));
  diffBtn.addEventListener("click", () => { diff = !diff; draw(); });
  body.append(h("div", { class: "panel" }, h("div", { class: "toolbar" }, seg, versions.length > 1 ? h("div", { class: "seg" }, diffBtn) : null), meta, key, text));
  draw();
}
function lineDiff(a, b) {
  const A = a.split("\n"), B = b.split("\n"), n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) { out.push(`${A[i]}\n`); i++; j++; }
    else if (j < m && (i === n || L[i][j + 1] >= L[i + 1][j])) { out.push(h("span", { class: "add" }, `${B[j]}\n`)); j++; }
    else { out.push(h("span", { class: "del" }, `${A[i]}\n`)); i++; }
  }
  return out;
}

/* ---------- audit cells -------------------------------------------------------------------- */

function auditSection(body) {
  body.append(h("p", { class: "lead", html: `Q1, the audit: 600 poles per model, explained once with the frozen prompts and scored on GPU. ${C.sealed ? "Their classes are computed but not read until W5 (SPEC_D1 §9.3). This page holds their counts and harvest statistics only." : "Their classes were read at W5; the Results section holds the fractions, and every pole is open under Poles."}` }));
  const tbody = h("tbody");
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, ["Model", term("Audit poles", "auditPoles"), term("Explained by", "explainedBy"), term("Texts", "texts"), term("Scored poles", "scoredPoles"), term("Scored on", "scoredOn"), C.sealed ? "" : term("One job", "q1Cell")].map((t, i) => h("th", { class: [1, 3, 4].includes(i) ? "r" : null }, t)))), tbody)));
  for (const c of C.cells) {
    const row = h("tr", { class: "row" }, h("td", { class: "id" }, c.label), h("td", { class: "r num" }, int(c.n_audit)),
      h("td", {}, Object.entries(c.explainers).map(([m, n]) => `${model(m)} ${int(n)}`).join(", ")),
      h("td", { class: "r num" }, int(c.n_texts)), h("td", { class: "r num" }, int(c.n_probe_poles)), h("td", {}, c.device),
      h("td", { class: "num" }, C.sealed ? h("span", { class: "sealed" }, lock(), "sealed") : pct(M.q1[c.key].fractions.ONE.p)));
    row.tabIndex = 0;
    const go = async () => { await sections.poles.open(); polesSection.show(`audit:${c.key}`); document.getElementById("poles").scrollIntoView({ behavior: "smooth" }); };
    row.addEventListener("click", go);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    tbody.append(row);
  }
  body.append(h("p", { class: "note" }, `Explained-by counts include the 40 S_gate poles in the OLMo cell. Scored poles are the explained poles plus their shuffle partners. Click a model to list its poles.`));
}

/* ---------- development grid --------------------------------------------------------------- */

function gridSection(body) {
  const cps = C.grid.checkpoints, steps = C.grid_steps;
  body.append(h("p", { class: "lead", html: `Q2, the development test, on OLMo-3-1025-7B: ${cps.length} stage-1 checkpoints harvested over one ${sig(cps[0].tokens / 1e6, 4)}M-token corpus, at the 50 strongest T<sub>15</sub> positions of ${C.grid.s_gate.length + C.grid.n_s} poles. A cluster is <b>present</b> at a checkpoint when the median of its exemplars reaches the pole’s own 99.9th percentile there.` }));
  // checkpoint rail on log tokens
  const W = 1000, H = 92, L = 34, R = 26;
  const tok = (st) => st * 4194304;  // tokens per step, SPEC_D1 §4.1
  const x = scale(Math.log10(tok(1000)), Math.log10(tok(steps.at(-1))), L + 54, W - R);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "The 16 checkpoints on a log scale of training tokens" });
  svg.append(s("line", { class: "axis", x1: L + 40, x2: W - R + 6, y1: 34, y2: 34 }));
  for (const [v, lab] of [[1e10, "10B"], [1e11, "100B"], [1e12, "1T"]]) svg.append(s("line", { class: "grid", x1: x(Math.log10(v)), x2: x(Math.log10(v)), y1: 20, y2: 48 }), s("text", { x: x(Math.log10(v)), y: 84, "text-anchor": "middle" }, lab));
  cps.forEach((c, i) => {
    const cx = i === 0 ? L : x(Math.log10(tok(c.step)));
    const dot = s("circle", { cx, cy: 34, r: 6, class: "dot-s hot" });
    dot.addEventListener("pointerenter", (e) => tip(`<b>T${i}</b> step ${int(c.step)}${i ? `, ${sig(tok(c.step) / 1e9, 3)}B tokens` : ", random init"}<br>revision ${c.revision.slice(0, 8)}<br>${esc(c.device)}, ${int(c.tps)} tokens/s<br>G0 max ratio ${sig(c.g0, 2)} ${c.g0_pass ? "(pass)" : "(FAIL)"}, dead poles ${int(c.dead)}`, e));
    dot.addEventListener("pointerleave", untip);
    svg.append(dot, s("text", { x: cx, y: 62, "text-anchor": "middle", style: "fill: var(--ink-2)" }, `T${i}`));
  });
  body.append(svg);

  const tbody = h("tbody", {}, cps.map((c, i) => h("tr", {}, h("td", { class: "id" }, `T${i}`), h("td", { class: "r num" }, int(c.step)),
    h("td", {}, h("span", { class: "hash" }, c.revision.slice(0, 12))), h("td", {}, c.device), h("td", { class: "r num" }, int(c.tps)),
    h("td", { class: "r num" }, sig(c.g0, 2)), h("td", { class: "r num" }, int(c.dead)), h("td", { class: "soft" }, c.dir))));
  const f = fold((b) => b.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, ["", term("Step", "step"), term("Revision", "revision"), term("GPU", "gpu"), term("Tokens/s", "tps"), term("G0 ratio", "g0col"), term("Dead poles", "dead"), term("Directory", "dir")].map((t, i) => h("th", { class: [1, 4, 5, 6].includes(i) ? "r" : null }, t)))), tbody))));
  const btn = h("button", { class: "sec-head", style: { padding: "12px 4px" } }, h("h2", { style: { fontSize: "15px" } }, "Checkpoint table"), h("span", { class: "sum" }, "revision, GPU, throughput and G0 per checkpoint"), chevron());
  toggler(btn, f);
  body.append(h("div", { class: "entry" }, btn, f.el));
  if (C.device_checks.length) deviceCheck(body);

  body.append(h("h3", { class: "sub-h" }, `S_gate over training`, h("small", {}, "median of the 50 T15 exemplars over the pole’s q99.9 at each checkpoint; at 1 or above the exemplars are present")));
  const grid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(168px, 1fr))", gap: "14px 18px" } });
  for (const p of C.grid.s_gate) grid.append(spark(p, C.grid.traj[p]));
  body.append(grid);
  body.append(C.sealed ? h("div", { class: "hatch sealbox", style: { marginTop: "22px" } }, h("span", { class: "sealed" }, lock(), `The other ${C.grid.n_s} poles, the development sample S, are sealed until W5: their trajectories decide births and the parent test.`))
    : h("p", { class: "note" }, `The other ${C.grid.n_s} poles, the development sample S, are OLMo’s audit poles: open one under Poles → Audit OLMo for its trajectory, clusters, births and judgments, or read the population under Results.`));

  body.append(h("h3", { class: "sub-h" }, C.sealed ? "W4 so far" : "W4", h("small", {}, `explain directories under bank/w4 at build time, ${C.built.replace("T", " ")}`)));
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Directory", "w4dir"), term("Started", "started"), term("Clusterer prompt", "clusterPrompt"), term("Answers kept per batch", "kept"), term("Clusters collected", "clusters"), term("Gate", "gateCol")]
      .map((t, i) => h("th", { class: [i === 4 ? "r" : "", [1, 2].includes(i) ? "hide-sm" : ""].join(" ").trim() || null }, t)))),
    h("tbody", {}, C.w4.map((d) => h("tr", {}, h("td", { class: "id" }, d.dir),
      h("td", { class: "num soft nowrap hide-sm" }, d.started ? fmtDay(d.started) : "no manifest yet"),
      h("td", { class: "hide-sm" }, d.batches.some((b) => (b.roles || []).includes("clusterer")) ? d.clusterer_prompt || "–" : "–"),
      h("td", {}, d.batches.length ? d.batches.map((b) => h("div", {}, `${b.batch}: `, h("b", {}, `${int(b.kept)} of ${int(b.n)}`), ` ${(b.models || []).map(model).join(", ")}`,
        h("span", { class: "soft" }, [b.rejected ? `, ${int(b.rejected)} rejected` : "", b.declined ? `, ${int(b.declined)} declined` : ""].join("")))) : "–"),
      h("td", { class: "r num" }, d.sealed ? h("span", { class: "sealed" }, lock(), "sealed") : d.n_clusters == null ? "–" : int(d.n_clusters),
        d.n_judgments ? h("small", { class: "soft" }, h("br"), `${int(d.n_judgments)} judgments`) : null),
      h("td", { class: "nowrap" }, gateText(d))))))));
}
function gateText(d) {
  if (d.judge_gate) {
    const g = d.judge_gate;
    return [h("b", {}, g.gate === "pass" ? "pass" : "fail"), `, a_ctrl ${g.a_ctrl.toFixed(3)}`];
  }
  if (d.g2) return [h("b", {}, d.g2.within_line ? "pass" : "fail"), `, G2 ${d.g2["k_confidence_gt_0.5"]} of ${d.g2.n_poles}`];
  return "–";
}
/* the same checkpoint harvested on two GPUs: does presence change with the device? */
function deviceCheck(body) {
  const poles = C.device_checks.find((r) => !r.clusters) || C.device_checks[0], cl = C.device_checks.find((r) => r.clusters)?.clusters;
  const k = C.grid.checkpoints.findIndex((c) => `stage1-step${c.step}` === poles.checks.branch[0]);
  const both = (x) => `${int(x.present_A)} / ${int(x.present_B)}, `;
  const flips = (x) => `${int(x.n_flipped)} flipped (${pct(x.flip_share, 2)})`;
  const ea = poles.exemplar_a, rd = ea.rel_diff;
  body.append(h("h3", { class: "sub-h" }, `Device check: T${k} on two GPUs`, h("small", {}, `${poles.devices.A} (job b) against ${poles.devices.B} (re-harvest); ${poles.checks.positions_identical ? "same positions and" : "DIFFERENT positions,"} ${int(poles.checks.n_tokens_scored)} tokens`)));
  const rows = [["poles", poles.n_poles, both(poles.presence.percentile) + flips(poles.presence.percentile), both(poles.presence.raw_ratio) + flips(poles.presence.raw_ratio)]];
  if (cl) rows.push([`clusters of S (${int(cl.n_poles)} poles)`, cl.n, flips(cl.percentile), flips(cl.raw_ratio)]);
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Unit", "dcUnit"), term("Compared", "dcN"), term("Percentile rule", "dcPct"), term("0.3 × max rule", "dcRaw")].map((t, i) => h("th", { class: i === 1 ? "r" : null }, t)))),
    h("tbody", {}, rows.map(([u, n, a, b]) => h("tr", {}, h("td", {}, u), h("td", { class: "r num" }, int(n)), h("td", { class: "num" }, a), h("td", { class: "num" }, b)))))));
  body.append(h("p", { class: "note" }, `Exemplar activations differ by a median ${pct(rd.median, 2)} (p90 ${pct(rd.p90, 2)}, p99 ${pct(rd.p99, 2)}); ${int(Math.round(ea.sign_differs_share * ea.n_entries))} of ${int(ea.n_entries)} change sign. `
    + `q99.9 differs by at most ${pct(poles.stats.q999_rel_diff.max, 1)} and max |a| by at most ${pct(poles.stats.max_abs_rel_diff.max, 1)}; the median pole keeps ${sig(poles.topk.overlap_median)} of its top 20 tokens. `
    + `The declared rule re-harvests job b if more than 5% flip (JOURNAL 21 Oct 19:31Z). `
    + (cl?.raw_ratio.flipped ? `The clusters that flip under the 0.3 × max rule: ${cl.raw_ratio.flipped.map(([p, k]) => `${minus(p)} cluster ${k}`).join(", ")}.` : "The list of flipped clusters of S stays sealed.")));
}
function spark(p, tr, opens = true) {  // opens: the small titled chart of the grid; else the pole view's wider one
  const W = opens ? 168 : 480, H = opens ? 64 : 120, L = 2, R = 2, T = opens ? 16 : 8, B = 6;
  const ratios = tr.med.map((m, i) => (tr.q999[i] > 0 ? m / tr.q999[i] : 0));
  const ly = (v) => Math.log10(Math.max(1e-3, v));
  const x = scale(0, ratios.length - 1, L, W - R), y = scale(-3, 1, H - B, T);
  const d = ratios.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(Math.min(1, ly(v))).toFixed(1)}`).join("");
  const svg = s("svg", { class: "chart hot", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": `${p} exemplar presence over the 16 checkpoints`, "data-noterm": "" },
    opens ? s("text", { x: 0, y: 10, style: "fill: var(--ink); font-weight: 500" }, minus(p)) : null,
    s("line", { class: "grid", x1: L, x2: W - R, y1: y(0), y2: y(0) }),
    s("path", { d, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.4 }),
    ratios.map((v, i) => s("circle", { cx: x(i), cy: y(Math.min(1, ly(v))), r: 2.4, class: tr.pct[i] ? "dot-s" : "dot-w", style: "stroke-width: 1" })));
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), i = Math.round(((e.clientX - r.left) / r.width) * (ratios.length - 1));
    const k = Math.max(0, Math.min(ratios.length - 1, i));
    tip(`<b>${minus(p)}</b> T${k}<br>median ${sig(tr.med[k])}, q99.9 ${sig(tr.q999[k])}, max ${sig(tr.max[k])}<br>ratio ${sig(ratios[k], 2)}: ${tr.pct[k] ? "present" : "not present"} (percentile rule), ${tr.raw[k] ? "present" : "not present"} (0.3 × max rule)`, e);
  });
  svg.addEventListener("pointerleave", untip);
  if (opens) svg.addEventListener("click", () => openPole(`O7:${p}`));
  return svg;
}

/* ---------- models and every neuron ------------------------------------------------------- */

function modelsSection(body) {
  body.append(h("p", { class: "lead", html: "The three audited models, harvested over the same 50M-token corpus, and the instruct model that reproduced a banked wikitext number (gate G1). Open a model to see every one of its poles: the distribution of max |a| by layer, and the G0 check at every layer next to the two inputs registered to make it fail." }));
  const tbody = h("tbody");
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, ["Model", term("Revision", "revision"), term("Layers × width", "layersWidth"), term("Activation", "act"), term("Tokens", "tokensCol"), term("Tokens/s", "tps"), term("G0 max ratio", "g0col"), term("Rare poles", "rarePoles"), term("Corpus NLL", "nll")].map((t, i) => h("th", { class: i >= 4 ? "r" : null }, t)))), tbody)));
  for (const m of [...C.models, { ...C.c0, key: "C0", label: "Qwen3-4B-Instruct-2507 (G1)" }]) {
    const row = h("tr", { class: "row" }, h("td", { class: "id" }, m.label), h("td", {}, h("span", { class: "hash" }, (m.revision || "").slice(0, 10))),
      h("td", { class: "num" }, `${m.n_layers} × ${int(m.width)}`), h("td", {}, m.act_fn_measured), h("td", { class: "r num" }, `${sig(m.n_tokens_scored / 1e6, 4)}M`),
      h("td", { class: "r num" }, int(m.tps)), h("td", { class: "r num" }, `${sig(Math.max(...m.g0), 2)}`), h("td", { class: "r num" }, m.atlas ? int(m.atlas.layers.reduce((n, l) => n + l.rare, 0)) : "–"),
      h("td", { class: "r num" }, sig(m.nll, 4)));
    const f = fold((b) => modelDetail(b, m));
    tbody.append(row, h("tr", { class: "detail" }, h("td", { colspan: 9 }, f.el)));
    clickable(row, (open) => f.set(open));
  }
}
function modelDetail(body, m) {
  const panel = h("div", { class: "panel" });
  body.append(panel);
  panel.append(h("dl", { class: "kv" }, h("dt", {}, "Repository"), h("dd", {}, `${m.repo}${m.branch ? `, ${m.branch}` : ""}`),
    h("dt", {}, "Runtime"), h("dd", {}, `transformers ${m.transformers}, torch ${m.torch}, ${m.dtype}, ${m.device_name}`),
    h("dt", {}, "Harvest"), h("dd", {}, `started ${String(m.started_utc).replace("T", " ")}, harness ${m.pg_git_sha}, ${int(m.n_poles)} poles`),
    m.g1 ? [h("dt", {}, "G1"), h("dd", {}, `wikitext mean NLL ${m.g1.wikitext_mean_nll} against the banked ${m.g1.banked}, gap ${sig(m.g1.gap, 2)}, tolerance ${m.g1.tolerance}: ${m.g1.pass ? "pass" : "FAIL"}`)] : null));
  if (m.atlas) {
    panel.append(h("h3", { class: "sub-h" }, `Every pole, by layer`, h("small", {}, `${int(m.n_poles)} poles; shade is the number of poles with that max |a|`)), atlasChart(m));
  }
  panel.append(h("h3", { class: "sub-h" }, "G0 at every layer", h("small", {}, "the recomputed MLP output against the module’s, as a share of its max; pass below the tolerance")), g0Chart(m));
}
function atlasChart(m) {
  const A = m.atlas, rows = A.layers, nb = A.max_edges.length - 1;
  const used = rows.flatMap((r) => r.max.flatMap((n, i) => (n ? [i] : [])));
  const b0 = Math.max(0, Math.min(...used) - 2), b1 = Math.min(nb, Math.max(...used) + 3);
  const W = 1000, L = 40, R = 76, T = 22, RH = 12, H = T + rows.length * RH + 40;
  const x = scale(b0, b1, L, W - R);
  const vmax = Math.max(...rows.flatMap((r) => r.max));
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Number of poles at each max |a|, per layer" });
  rows.forEach((r, j) => {
    const yy = T + j * RH;
    if (j % 4 === 0 || j === rows.length - 1) svg.append(s("text", { x: L - 8, y: yy + RH - 3, "text-anchor": "end" }, j));
    r.max.forEach((n, i) => {
      if (!n) return;
      const rect = s("rect", { x: x(i), y: yy, width: x(1) - x(0) + 0.3, height: RH - 1, fill: "var(--accent)", "fill-opacity": (0.08 + 0.92 * Math.sqrt(n / vmax)).toFixed(3) });
      rect.addEventListener("pointerenter", (e) => tip(`Layer ${j}: <b>${int(n)}</b> poles with max |a| in ${sig(10 ** A.max_edges[i], 2)} to ${sig(10 ** A.max_edges[i + 1], 2)}`, e));
      rect.addEventListener("pointerleave", untip);
      svg.append(rect);
    });
    svg.append(s("text", { x: W - R + 8, y: yy + RH - 3 }, r.rare ? `${int(r.rare)} rare` : ""));
  });
  for (let e = -3; e <= 4; e++) {
    const i = ((e - A.max_edges[0]) / (A.max_edges.at(-1) - A.max_edges[0])) * nb;
    if (i >= b0 && i <= b1) svg.append(s("text", { x: x(i), y: H - 22, "text-anchor": "middle" }, `10${sup(e)}`));
  }
  svg.append(s("text", { x: L - 8, y: 12, "text-anchor": "end" }, "layer"),
    s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }, "max |a| over the corpus"));
  return svg;
}
function g0Chart(m) {
  const series = [["G0, the model as run", m.g0, "var(--accent)"], ...(m.g0_failing || []).map((fi) => [fi.act ? `${fi.act} swapped in` : `u × ${fi.u_scale}`, fi.ratio, "var(--ink-3)"])];
  const n = m.g0.length, W = 1000, H = 160, L = 40, R = 150, T = 8, B = 24;
  const x = scale(0, n - 1, L, W - R), y = scale(-4, 0, H - B, T);
  const ly = (v) => Math.max(-4, Math.log10(Math.max(1e-4, v)));
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "G0 ratio per layer, log scale" });
  for (const e of [-4, -3, -2, -1, 0]) svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(e), y2: y(e) }), s("text", { x: L - 8, y: y(e) + 4, "text-anchor": "end" }, `10${sup(e)}`));
  const tol = Math.log10(m.g0_tol || 0.01);
  svg.append(s("line", { x1: L, x2: W - R, y1: y(tol), y2: y(tol), stroke: "var(--ink)", "stroke-dasharray": "4 3" }));
  const ends = [[`tolerance ${m.g0_tol || 0.01}`, y(tol), "var(--ink)"]];
  series.forEach(([label, ys, col], k) => {
    svg.append(s("path", { d: ys.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(ly(v))}`).join(""), fill: "none", stroke: col, "stroke-width": k ? 1.25 : 2, "stroke-dasharray": k === 2 ? "2 2" : null }));
    ends.push([label, y(ly(ys.at(-1))), k ? "var(--ink-2)" : "var(--accent-ink)"]);
  });
  ends.sort((a, b) => a[1] - b[1]);
  let prev = -Infinity;
  for (const [label, yy, col] of ends) { const v = Math.max(yy, prev + 13); svg.append(s("text", { x: W - R + 8, y: v + 4, style: `fill: ${col}` }, label)); prev = v; }
  for (let i = 0; i < n; i += 4) svg.append(s("text", { x: x(i), y: H - 6, "text-anchor": "middle" }, i));
  return svg;
}

/* ---------- runs and spend ----------------------------------------------------------------- */

const RENTAL_STATUS = { done: "The rental ran its job to the end and was destroyed.",
  killed_infra: "The rental was killed for an infrastructure reason, not because the job failed; the spend record’s host verdict for each such rental here is slow_net.",
  killed_manual: "The rental was destroyed by hand while still loading, unbilled: the respawn had landed on an H100 at $6.64/h, more than the credit left could pay for (JOURNAL 23 Oct 20:27Z)." };
const JOB_STATE = { ok: "The step finished and wrote its outputs; from the job’s pipeline_status.json." };
function stepDef(j) {  // what one pipeline step ran, from its module and arguments
  const arg = (k) => new RegExp(`--${k} (\\S+)`).exec(j.args || "")?.[1];
  if (j.module === "neurons_corpus") return `Corpus step (neurons_corpus): builds the corpus token arrays, ${arg("tokens-per-shard")} tokens per shard here.`;
  const what = `${arg("repo")}${arg("revision") ? ` at ${arg("revision")}` : ""}`;
  if (/--probe-texts/.test(j.args)) return `Probe step (neurons_harvest): runs ${what} on the writer texts and records each text’s peak activation at the listed poles.`;
  return `Harvest step (neurons_harvest) of ${what}: every pole’s top-50, statistics and output vocabulary over ${arg("max-tokens") === "all" ? "the whole corpus" : `${int(+arg("max-tokens"))} corpus tokens`}.`;
}
function runsSection(body) {
  const R = C.runs;
  body.append(h("p", { class: "lead", html: `Every GPU step banked under <span class="hash">bank/neurons/</span>, from its pipeline status and manifests, and every vast.ai rental from the spend record. AWS steps carry no dollar figure in the bank.` }));
  const usd = R.rentals.reduce((n, r) => n + (r.realized_usd || 0), 0), hrs = R.rentals.reduce((n, r) => n + (r.billing?.gpu?.hours || 0), 0);
  body.append(h("h3", { class: "sub-h" }, "vast.ai rentals", h("small", {}, `$${usd.toFixed(2)} over ${R.rentals.length} rentals, ${hrs.toFixed(1)} billed GPU-hours`)));
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Rental", "rental"), term("Wave", "wave"), term("Opened", "opened"), term("Status", "rentalStatus"), term("GPU-hours", "gpuh"), term("Cost", "cost")].map((t, i) => h("th", { class: i >= 4 ? "r" : null }, t)))),
    h("tbody", {}, R.rentals.map((r) => h("tr", {}, h("td", { class: "id" }, r.rental.replace(/^neurons_/, "")), h("td", {}, r.session || "–"),
      h("td", { class: "num soft" }, String(r.created_utc || "").replace("T", " ").replace("Z", "")), h("td", {}, defined(r.status, RENTAL_STATUS[r.status])),
      h("td", { class: "r num" }, sig(r.billing?.gpu?.hours, 3)), h("td", { class: "r num" }, `$${(r.realized_usd || 0).toFixed(2)}`)))))));
  body.append(h("p", { class: "note" }, `The spend record also holds ${R.record.n - R.rentals.length} rentals from before D1. All ${R.record.n} total $${R.record.usd.toFixed(2)} against the $36.00 vast credit, as SPEC_D1 §12 counts it.`));
  body.append(h("h3", { class: "sub-h" }, "GPU steps", h("small", {}, `${R.jobs.length} steps, AWS and vast.ai`)));
  body.append(h("div", { class: "wrap-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, [term("Job", "job"), term("Step", "jobStep"), term("Model", "jobModel"), term("GPU", "gpu"), term("Tokens", "jobTokens"), term("Tokens/s", "tps"), "Started", term("State", "state")].map((t, i) => h("th", { class: [4, 5].includes(i) ? "r" : null }, t)))),
    h("tbody", {}, R.jobs.map((j) => {
      const row = h("tr", { class: "row", title: `${j.module} ${j.args}` }, h("td", { class: "id" }, j.job), h("td", {}, /^t\d\d_step/.test(j.step) ? j.step : defined(j.step, stepDef(j))), h("td", {}, j.model || "–"), h("td", {}, j.device || "–"),
        h("td", { class: "r num" }, j.tokens ? int(j.tokens) : "–"), h("td", { class: "r num" }, j.tps ? int(j.tps) : "–"),
        h("td", { class: "num soft" }, String(j.started || "–").replace("T", " ").replace("Z", "")), h("td", {}, defined(j.state, JOB_STATE[j.state])));
      return row;
    })))));
  body.append(h("p", { class: "note" }, "Hover a step for its module and arguments."));
}

/* ---------- documents ---------------------------------------------------------------------- */

function md(text) {
  const emph = (t) => t.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
  const inline = (t) => esc(t).split(/(`[^`]+`)/).map((x, i) => (i % 2 ? `<code>${x.slice(1, -1)}</code>` : emph(x))).join("");  // no emphasis inside code
  const L = text.split("\n"), out = [];
  let i = 0;
  const block = (l) => /^(#{1,6} |```|\s*\||\s*[-*] |\s*\d+\. )/.test(l) || !l.trim();
  while (i < L.length) {
    const l = L[i];
    if (!l.trim()) { i++; continue; }
    if (l.startsWith("```")) {
      const buf = []; i++;
      while (i < L.length && !L[i].startsWith("```")) buf.push(L[i++]);
      i++; out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
    } else if (/^#{1,6} /.test(l)) {
      const n = Math.min(4, l.match(/^#+/)[0].length);
      out.push(`<h${n}>${inline(l.replace(/^#+ /, ""))}</h${n}>`); i++;
    } else if (/^\s*\|/.test(l)) {
      const rows = [];
      while (i < L.length && /^\s*\|/.test(L[i])) rows.push(L[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => inline(c.trim()));
      const body = rows.filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      out.push(`<table><thead><tr>${cells(body[0]).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body.slice(1).map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    } else if (/^\s*([-*]|\d+\.) /.test(l)) {
      const ordered = /^\s*\d+\. /.test(l), items = [];
      while (i < L.length && L[i].trim() && (/^\s*([-*]|\d+\.) /.test(L[i]) || /^\s{2,}/.test(L[i]))) {
        if (/^\s*([-*]|\d+\.) /.test(L[i]) && !/^\s{4,}/.test(L[i])) items.push(L[i].replace(/^\s*([-*]|\d+\.) /, ""));
        else items[items.length - 1] += ` ${L[i].trim()}`;
        i++;
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
    } else {
      const buf = [];
      while (i < L.length && !block(L[i])) buf.push(L[i++]);
      if (!buf.length) buf.push(L[i++]);
      out.push(`<p>${inline(buf.join(" "))}</p>`);
    }
  }
  return out.join("");
}

async function docsSection(body) {
  body.append(h("p", { class: "lead", html: "The study’s own documents, as they stand in the drafts worktree. The journal is split into its entries, newest first." }));
  const docs = await load("docs");
  for (const d of docs) {
    const head = h("button", { class: "sec-head", style: { padding: "14px 4px" } }, h("h2", { style: { fontSize: "15px" } }, d.name),
      h("span", { class: "sum" }, `${sig(d.bytes / 1000, 3)} kB, modified ${d.modified.replace("T", " ")}`), chevron());
    const f = fold((b) => {
      if (d.name !== "JOURNAL.md") { b.append(h("div", { class: "doc", style: { padding: "4px 4px 24px" }, html: md(d.text) })); return; }
      const parts = d.text.split(/\n(?=## )/);
      b.append(h("div", { class: "doc", style: { padding: "4px" }, html: md(parts[0]) }));
      for (const p of parts.slice(1).reverse()) {
        const title = p.split("\n")[0].replace(/^## /, "");
        const eh = h("button", {}, h("h4", {}, title), chevron());
        const ef = fold((bb) => bb.append(h("div", { class: "doc", html: md(p.split("\n").slice(1).join("\n")) })));
        toggler(eh, ef);
        b.append(h("div", { class: "entry", "data-entry": title }, eh, ef.el));
      }
    });
    toggler(head, f);
    body.append(h("div", { class: "entry", "data-doc": d.name }, head, f.el));
    docsSection.items = (docsSection.items || []).concat([{ d, head }]);
  }
}

/* ---------- search ------------------------------------------------------------------------- */

let index = null;
async function buildIndex() {
  const cellFiles = C.files.filter((f) => f.startsWith("words_"));
  const [words, docs, cl, ...cw] = await Promise.all([load("words"), load("docs"), C.files.includes("clusters") ? load("clusters") : {}, ...cellFiles.map(load)]);
  const items = [];
  for (const key of Object.keys(C.poles)) if (C.poles[key].set !== "shuffle partner") items.push({ g: "Poles", key, title: minus(poleId(key)), text: `${poleId(key)} ${minus(poleId(key))}`, go: () => openPole(key) });
  for (const c of C.cells) for (const p of Object.keys(c.audit)) items.push({ g: "Poles", key: `${c.key}:${p}`, title: `${minus(p)} (audit ${c.key}${C.sealed ? ", sealed" : ""})`, text: p, go: () => openPole(`${c.key}:${p}`) });
  const addWords = (key, where, p, w) => {
    if (w.e) items.push({ g: "Explanations", group: key, where, title: minus(p), text: w.e, go: () => openPole(key) });
    for (const [kind, ts] of Object.entries(w.t || {})) for (const [tk] of ts) items.push({ g: "Test texts", group: `${key}:${kind}`, where, title: `${minus(p)}, ${(KIND[kind] || "job text").toLowerCase()}`, text: tk.join(""), go: () => openPole(key) });
  };
  for (const [rid, poles] of Object.entries(words)) for (const [p, w] of Object.entries(poles)) addWords(p, rid, p, w);
  cellFiles.forEach((f, i) => { const ck = f.slice(6); for (const [p, w] of Object.entries(cw[i])) addWords(`${ck}:${p}`, `W2-C ${ck}`, p, w); });
  for (const [p, list] of Object.entries(cl)) for (const c of list) items.push({ g: "Clusters", title: `${minus(p)} cluster ${c.c}`, text: c.name, go: () => openPole(`O7:${p}`) });
  for (const p of Object.values(C.prompts)) for (const line of p.text.split("\n")) if (line.trim()) items.push({ g: "Prompts", title: `${ROLE[p.role]} ${p.version || p.sha.slice(0, 8)}`, text: line, go: () => sections.prompts.open().then(() => document.getElementById("prompts").scrollIntoView({ behavior: "smooth" })) });
  for (const d of docs) for (const line of d.text.split("\n")) if (line.trim()) items.push({ g: "Documents", title: d.name, text: line, go: () => openDoc(d.name) });
  index = items;
}
async function openDoc(name) {
  await sections.docs.open();
  const it = (docsSection.items || []).find((x) => x.d.name === name);
  if (!it) return;
  if (it.head.getAttribute("aria-expanded") !== "true") it.head.click();
  it.head.scrollIntoView({ behavior: "smooth", block: "start" });
}
function mark(text, q) {
  const i = text.toLowerCase().indexOf(q);
  if (i < 0) return esc(text.slice(0, 140));
  const a = Math.max(0, i - 50);
  return `${a ? "…" : ""}${esc(text.slice(a, i))}<mark>${esc(text.slice(i, i + q.length))}</mark>${esc(text.slice(i + q.length, i + q.length + 90))}`;
}
function buildSearch() {
  const input = document.getElementById("q"), box = document.getElementById("results");
  let timer = 0, hits = [], active = -1;
  const close = () => { box.classList.remove("open"); active = -1; };
  const run = async () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { close(); return; }
    if (!index) { box.replaceChildren(h("p", { class: "empty" }, "Loading the index…")); box.classList.add("open"); await buildIndex(); }
    const groups = {}, seen = {};
    for (const it of index) {
      if (!(it.text.toLowerCase().includes(q) || it.title.toLowerCase().includes(q))) continue;
      if (it.group && seen[it.group]) { seen[it.group].rounds.add(it.where); continue; }
      const g = (groups[it.g] ||= []);
      if (g.length >= 6) continue;
      const hit = { ...it, rounds: new Set(it.where ? [it.where] : []) };
      if (it.group) seen[it.group] = hit;
      g.push(hit);
    }
    hits = []; box.replaceChildren();
    for (const [g, list] of Object.entries(groups)) {
      box.append(h("h4", {}, g));
      for (const it of list) {
        const where = it.rounds.size ? h("span", { class: "muted" }, ` in ${[...it.rounds].join(", ")}`) : null;
        const b = h("button", { class: "hit", role: "option", onclick: () => { close(); it.go(); } }, h("b", {}, it.title), where, h("span", { class: "ctx", html: mark(it.text, q) }));
        hits.push(b); box.append(b);
      }
    }
    if (!hits.length) box.append(h("p", { class: "empty" }, `Nothing matches “${input.value.trim()}”.`));
    box.classList.add("open");
  };
  input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 110); });
  input.addEventListener("focus", () => { if (!index) buildIndex(); if (input.value.trim().length >= 2) run(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { close(); input.blur(); return; }
    if (!hits.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + hits.length) % hits.length;
      hits.forEach((b, i) => b.classList.toggle("active", i === active));
      hits[active].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && active >= 0) hits[active].click();
  });
  document.addEventListener("pointerdown", (e) => { if (!(e.target instanceof Element && e.target.closest(".search"))) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); } });
}

/* ---------- boot -------------------------------------------------------------------------- */

function boot() {
  observer.observe(document.body, { childList: true, subtree: true });
  annotate(document.getElementById("bar"));
  buildRail();
  buildSearch();
  const app = document.getElementById("app");
  const frozen = C.rounds.find((r) => r.id === C.frozen_round);
  const audit = C.cells.reduce((n, c) => n + c.n_audit, 0);
  app.append(
    hero(), guide(),
    M ? section("results", "Results", (el) => {
      el.innerHTML = `p_u <b>${f3(M.q2.p_u.point)}</b>: ${M.q2.p_u.rule}; b above the band; one job in ${C.cells.map((c) => `${pct(M.q1[c.key].fractions.ONE.p)} ${c.key}`).join(", ")} audit poles.`;
    }, resultsSection) : "",
    section("calibration", "Calibration rounds", (el) => {
      const p = passRate(recsOf(frozen), state.theta);
      el.innerHTML = `${C.rounds.length} rounds on 120 poles. The frozen round passes <b>${pct(p.rate)}</b> at θ<sub>s</sub> ${th4(state.theta)}.`;
    }, calibration),
    section("poles", "Poles", (el) => { el.innerHTML = `<b>${Object.keys(C.rounds[0].poles).length}</b> discovery and <b>${C.grid.s_gate.length}</b> S_gate poles open, <b>${int(audit)}</b> audit poles${C.sealed ? " sealed" : ""}.`; }, polesSection),
    section("prompts", "Prompts", (el) => {
      const roles = new Set(Object.values(C.prompts).map((p) => p.role));
      el.innerHTML = `<b>${Object.keys(C.prompts).length}</b> committed texts across ${roles.size} roles.`;
    }, promptsSection),
    section("audit", "Audit", (el) => { el.innerHTML = `${C.cells.map((c) => c.label).join(", ")}: <b>${int(audit)}</b> poles explained and scored${C.sealed ? ", results sealed until W5" : ""}.`; }, auditSection),
    section("grid", "Development grid", (el) => { el.innerHTML = `OLMo-3-1025-7B at <b>${C.grid.checkpoints.length}</b> checkpoints, step 0 to ${int(C.grid_steps.at(-1))}.`; }, gridSection),
    section("models", "Models and every neuron", (el) => { el.innerHTML = `${C.models.map((m) => m.label).join(", ")}: <b>${int(C.models.reduce((n, m) => n + m.n_poles, 0))}</b> poles harvested.`; }, modelsSection),
    section("runs", "Runs and spend", (el) => {
      const usd = C.runs.rentals.reduce((n, r) => n + (r.realized_usd || 0), 0);
      el.innerHTML = `<b>${C.runs.jobs.length}</b> GPU steps; <b>$${usd.toFixed(2)}</b> on vast.ai rentals.`;
    }, runsSection),
    section("docs", "Documents", (el) => { el.innerHTML = "Results, registry, index, spec, journal, data contract, directions, ideas, theories, checks, checker and closeout reports, verify notes, handoffs, the judge incident log and the runbook."; }, docsSection));
  const bar = document.getElementById("bar");
  addEventListener("scroll", () => bar.classList.toggle("scrolled", scrollY > 8), { passive: true });
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 600));
  idle(() => { load("words"); load("docs"); });
  const linked = decodeURIComponent(location.hash.slice(1));
  if (linked && poleKeyKnown(linked)) openPole(linked);
}
boot();
