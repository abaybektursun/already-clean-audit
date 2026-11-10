// "One explanation and its test texts": the picture, shared by the walkthrough page and index.html.
// It builds itself inside the element with id "xtest" and reads window.EXPLANATION_TEST_DATA, which data.js
// sets. With no #xtest on the page it does nothing. Pass or fail is computed here from the stored scores
// with the study's rule (SPEC_D1 §6.1: at least 3 of 5 texts written to fire the neuron strong, at most 2
// of 5 near misses, at most 1 of 5 distant texts); extract.py checks the same rule against the stored values.
(() => {
  'use strict';
  const root = document.getElementById('xtest');
  if (!root) return;
  const D = window.EXPLANATION_TEST_DATA;
  if (!D) {
    root.innerHTML = '<p class="xt-missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      'explanation_test to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in explanation_test');
  }

  const [MIN, MAX, STEP] = D.slider;               // the strong line's range, and the axis: 0 to 1.2
  const STOPS = D.stops;
  const STUDY = 1;                                 // STOPS[1] is the study's line
  const BLIND = 15;                                // texts 0-4 written to fire it, 5-9 near misses, 10-14 distant, 15 the blind maximum
  const GROUPS = [
    { name: 'Written to fire it', note: 'The explanation says these should fire the neuron.',
      rule: 'needs at least 3', why: 'the rule needs at least 3', ok: (k) => k >= 3, lower: 'written to fire it' },
    { name: 'Near misses', note: 'Close to those texts, but without the thing the explanation names.',
      rule: 'at most 2 allowed', why: 'the rule allows at most 2', ok: (k) => k <= 2, lower: 'near misses' },
    { name: 'Distant texts', note: 'On a subject far from the explanation.',
      rule: 'at most 1 allowed', why: 'the rule allows at most 1', ok: (k) => k <= 1, lower: 'distant texts' },
    { name: 'Written to fire it as hard as possible', note: 'Not part of the rule.', blind: true },
  ];
  const TICKS = [0, 0.25, 0.5, 0.75, 1];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');

  // ---------- the rule ----------
  const nStrong = (xs, t) => xs.reduce((k, s) => k + (s >= t ? 1 : 0), 0);
  function judge(scores, t) {
    const k = [0, 1, 2].map((g) => nStrong(scores.slice(5 * g, 5 * g + 5), t));
    return { k, passes: GROUPS.slice(0, 3).every((g, i) => g.ok(k[i])) };
  }
  const own = (n) => n.t.slice(0, 15).map((x) => x[1]);
  const rnd = (n) => n.r.t.map((x) => x[0]);
  D.neurons.forEach((n) => {
    const j = judge(own(n), STOPS[STUDY]);
    if (j.passes !== n.check.passes || j.k.join() !== n.check.tally.join()) {
      throw new Error(n.p + ': the page computes ' + j.k + ' (' + j.passes + '), extract.py found ' +
        n.check.tally + ' (' + n.check.passes + ')');
    }
  });

  const show = (p) => p.replace('-', '−');
  function describe(p) {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(p);
    return 'layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' pole';
  }
  const fmt = (t) => (STOPS.indexOf(t) >= 0 ? t.toFixed(4) : t.toFixed(3));
  const frac = (s) => Math.max(0, Math.min(1, (s - MIN) / (MAX - MIN)));

  // ---------- static DOM ----------
  root.innerHTML = `
  <div class="xt-top">
    <label for="xt-pick">Neuron</label>
    <div class="xt-selwrap"><select id="xt-pick"></select></div>
    <button type="button" class="xt-btn" id="xt-rand" aria-pressed="false"></button>
  </div>
  <p class="xt-id" id="xt-id"></p>
  <blockquote class="xt-expl" id="xt-expl"></blockquote>
  <p class="xt-by" id="xt-by"></p>
  <p class="xt-rnote" id="xt-rnote" aria-live="polite"></p>
  <div class="xt-chart" id="xt-chart">
    <div class="xt-head">
      <div class="xt-verdict" aria-live="polite"><span class="xt-where" id="xt-where"></span><b class="xt-word" id="xt-word"></b><span class="xt-why" id="xt-why"></span></div>
      <div class="xt-lane">
        <div class="xt-lanehead"><b>Strong line</b><span class="xt-note" id="xt-note"></span><span class="xt-stops" id="xt-stops"></span></div>
        <div class="xt-scale" id="xt-scale">
          <div class="xt-handle" id="xt-handle" role="slider" tabindex="0" aria-label="Strong line"
            aria-valuemin="${MIN}" aria-valuemax="${MAX}"></div>
        </div>
      </div>
      <div></div>
    </div>
    <div id="xt-groups"></div>
  </div>
  <details>
    <summary>Show the scores as a table</summary>
    <div class="xt-tablewrap"><table id="xt-table"></table></div>
  </details>
  <p class="xt-caption" id="xt-caption"></p>`;
  const $ = (id) => root.querySelector('#xt-' + id);

  // picker
  const pick = $('pick');
  D.groups.forEach((name, g) => {
    const og = document.createElement('optgroup');
    og.label = name;
    D.neurons.forEach((n, i) => {
      if (n.g !== g) return;
      const o = document.createElement('option');
      o.value = String(i);
      const tail = n.n || (n.x.length > 54 ? n.x.slice(0, 52).trimEnd() + '…' : n.x);
      o.textContent = show(n.p) + ', ' + tail;
      og.appendChild(o);
    });
    pick.appendChild(og);
  });

  // stops and ticks
  const stopBtns = STOPS.map((t, k) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'xt-stop';
    b.textContent = t.toFixed(4);
    b.setAttribute('aria-label', 'Set the strong line to ' + t.toFixed(4) + (k === STUDY ? ', the study’s line' : ''));
    b.addEventListener('click', () => { userActed(); setTheta(t); });
    $('stops').appendChild(b);
    return b;
  });
  const scale = $('scale');
  TICKS.forEach((v) => {
    const s = document.createElement('span');
    s.className = 'xt-tick';
    s.style.left = 100 * frac(v) + '%';
    s.textContent = v === 0 ? '0' : String(v);
    scale.appendChild(s);
  });
  const handle = $('handle');

  // rows: 3 groups of 5 and the blind maximum, built once and refilled per neuron
  const rows = [];
  const heads = [];
  let ri = 0;
  GROUPS.forEach((g, gi) => {
    const box = document.createElement('div');
    box.className = 'xt-group' + (g.blind ? ' xt-blind' : '');
    const head = document.createElement('div');
    head.className = 'xt-row xt-ghead';
    head.style.setProperty('--xt-i', String(ri + gi));
    const gt = document.createElement('div');
    gt.className = 'xt-gtext';
    const h = document.createElement('h3');
    h.textContent = g.name;
    const note = document.createElement('span');
    note.className = 'xt-gnote';
    note.textContent = g.note;
    gt.append(h, note);
    const hd = { g, sqs: [], count: null, verdict: null };
    if (!g.blind) {
      const tally = document.createElement('div');
      tally.className = 'xt-tally';
      const sqs = document.createElement('span');
      sqs.className = 'xt-sqs';
      sqs.setAttribute('aria-hidden', 'true');
      for (let i = 0; i < 5; i++) {
        const q = document.createElement('i');
        q.className = 'xt-sq';
        sqs.appendChild(q);
        hd.sqs.push(q);
      }
      const count = document.createElement('b');
      const rule = document.createElement('span');
      rule.className = 'xt-rule';
      rule.textContent = g.rule;
      const verdict = document.createElement('span');
      tally.append(sqs, count, rule, verdict);
      gt.appendChild(tally);
      hd.count = count;
      hd.verdict = verdict;
    }
    const gtrack = document.createElement('div');
    gtrack.className = 'xt-track';
    head.append(gt, gtrack, document.createElement('span'));
    box.appendChild(head);
    heads.push(hd);
    const n = g.blind ? 1 : 5;
    for (let k = 0; k < n; k++) {
      const row = document.createElement('div');
      row.className = 'xt-row';
      row.style.setProperty('--xt-i', String(ri + gi + 1));
      const text = document.createElement('p');
      text.className = 'xt-text';
      const track = document.createElement('div');
      track.className = 'xt-track';
      const bar = document.createElement('i');
      bar.className = 'xt-bar';
      track.appendChild(bar);
      const v = document.createElement('span');
      v.className = 'xt-v';
      row.append(text, track, v);
      box.appendChild(row);
      rows.push({ row, text, bar, v, i: ri });
      ri++;
    }
    $('groups').appendChild(box);
  });

  // ---------- state ----------
  let cur = 0;                    // index into D.neurons
  let theta = STOPS[STUDY];
  let random = false;             // showing the scores at the random neuron

  function scoresNow() {
    const n = D.neurons[cur];
    return random ? rnd(n) : own(n);
  }

  function markText(el, text, a, b) {
    el.textContent = '';
    if (a == null) { el.textContent = text; return; }
    while (b - a > 1 && /\s/.test(text[a])) a++;   // a token's leading space stays outside the mark
    const m = document.createElement('mark');
    m.className = 'xt-m';
    m.textContent = text.slice(a, b);
    el.append(text.slice(0, a), m, text.slice(b));
  }

  function drawNeuron() {
    const n = D.neurons[cur];
    const id = $('id');
    id.textContent = '';
    const b = document.createElement('b');
    b.textContent = show(n.p);
    id.append(b, describe(n.p) + ' of OLMo-3 7B' +
      (n.p.endsWith('-') ? '. On a negative pole a text’s score is how far below zero the neuron goes.' : '.'));
    $('expl').textContent = '“' + n.x + '”';
    $('by').textContent = 'Written by ' + D.explainers[n.e] + ' from the neuron’s 30 strongest contexts; another ' +
      'model, shown only this sentence, wrote the 16 texts below, each marked where the neuron peaked.';
    rows.forEach((r) => {
      const t = n.t[r.i];
      if (random && r.i < BLIND) markText(r.text, t[0], n.r.t[r.i][1], n.r.t[r.i][2]);
      else if (random) markText(r.text, t[0], null, null);
      else markText(r.text, t[0], t[2], t[3]);
    });
  }

  function drawScores() {
    const n = D.neurons[cur];
    const s = scoresNow();
    rows.forEach((r) => {
      const none = random && r.i === BLIND;
      const v = none ? null : r.i === BLIND ? n.t[BLIND][1] : s[r.i];
      r.row.classList.toggle('xt-none', none);
      r.row.classList.toggle('xt-strong', v != null && v >= theta);
      r.row.classList.toggle('xt-over', v != null && v > MAX);
      r.bar.style.setProperty('--xt-w', v == null ? '0' : String(frac(v)));
      r.v.textContent = v == null ? 'not scored' : v.toFixed(3);
    });
    const j = judge(s, theta);
    heads.forEach((h, gi) => {
      if (h.g.blind) return;
      h.sqs.forEach((q, i) => q.classList.toggle('xt-on', i < j.k[gi]));
      h.count.textContent = j.k[gi] + ' of 5 strong';
      const ok = h.g.ok(j.k[gi]);
      h.verdict.className = ok ? 'xt-ok' : 'xt-no';
      h.verdict.textContent = ok ? '✓ meets the rule' : '✗ breaks the rule';
    });
    $('where').textContent = random ? 'Scored at the random neuron ' + show(n.r.p) : '';
    $('word').textContent = j.passes ? 'Passes' : 'Fails';
    const broken = GROUPS.slice(0, 3).map((g, i) => ({ g, k: j.k[i] })).filter((x) => !x.g.ok(x.k));
    $('why').textContent = j.passes
      ? j.k[0] + ' of 5 written to fire it were strong, ' + j.k[1] + ' of 5 near misses and ' + j.k[2] + ' of 5 distant texts.'
      : broken.map((x) => (x.k === 1 ? '1 of 5 ' + x.g.lower + ' was' : x.k + ' of 5 ' + x.g.lower + ' were') +
        ' strong; ' + x.g.why).join('. ') + '.';
    drawLine();
    drawTable(n, s);
  }

  function drawLine() {
    root.style.setProperty('--xt-f', String(frac(theta)));
    const k = STOPS.indexOf(theta);
    $('note').textContent = k === STUDY ? 'the study’s line' : k === 0 ? '0.1 below the study’s line'
      : k === 2 ? '0.1 above the study’s line' : 'drag it, or pick a registered stop:';
    stopBtns.forEach((b, i) => b.classList.toggle('xt-on', i === k));
    handle.textContent = fmt(theta);
    handle.setAttribute('aria-valuenow', String(theta));
    handle.setAttribute('aria-valuetext', fmt(theta) + (k === STUDY ? ', the study’s line' : '') + '. ' +
      (random ? 'At the random neuron: ' : '') + $('word').textContent);
    // the tag is centered on the line, kept inside the bar column
    const W = scale.clientWidth;
    const w = handle.offsetWidth;
    const x = frac(theta) * W;
    handle.style.left = Math.max(0, Math.min(W - w, x - w / 2)) + 'px';
  }

  function drawTable(n, s) {
    const table = $('table');
    table.textContent = '';
    const head = table.insertRow();
    ['Kind', 'Text', random ? 'Score at the random neuron' : 'Score', 'Strong at ' + fmt(theta)].forEach((h) => {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    });
    n.t.forEach((t, i) => {
      const tr = table.insertRow();
      const g = GROUPS[Math.min(3, Math.floor(i / 5))];
      tr.insertCell().textContent = g.blind ? 'Written to fire it as hard as possible (not in the rule)' : g.name;
      const tt = tr.insertCell();
      tt.className = 'xt-ttext';
      tt.textContent = t[0];
      const v = random ? (i === BLIND ? null : s[i]) : t[1];
      const c = tr.insertCell();
      c.className = 'xt-num';
      c.textContent = v == null ? 'not scored' : v.toFixed(4);
      const st = tr.insertCell();
      st.textContent = v == null ? '' : v >= theta ? 'strong' : 'no';
    });
  }

  function drawRandomNote() {
    const n = D.neurons[cur];
    const btn = $('rand');
    btn.setAttribute('aria-pressed', String(random));
    btn.textContent = random ? 'Back to this neuron’s scores' : 'Score the same texts at a random neuron';
    $('rnote').textContent = random
      ? 'Now the bars show the same 15 texts scored at ' + show(n.r.p) + ', ' + describe(n.r.p) + ', another ' +
        'neuron of the same layer drawn at random; the marked words are where that neuron peaked. The explanation ' +
        'was not written for it, and the text written to fire it as hard as possible has no score there.'
      : '';
    root.querySelector('.xt-by').hidden = random;
  }

  function setTheta(t) {
    theta = t;
    drawScores();
  }

  // a position on the bar column to a line value: steps of 0.005, and within half a step of a stop, the stop
  function snap(v) {
    v = Math.max(MIN, Math.min(MAX, v));
    v = Math.round(v / STEP) * STEP;
    v = Math.round(v * 1000) / 1000;
    const near = STOPS.find((s) => Math.abs(v - s) < STEP / 2);
    return near === undefined ? v : near;
  }

  // ---------- dragging the line ----------
  // A press on the scale, or a mouse press on any bar's track, moves the line at once. A finger on a track
  // moves it only once it slides sideways, so that scrolling the page past the bars leaves the line alone.
  let drag = null;
  function fromPointer(e, el) {
    const r = el.getBoundingClientRect();
    return snap(MIN + ((e.clientX - r.left) / r.width) * (MAX - MIN));
  }
  function begin(e) {
    userActed();
    drag.live = true;
    root.classList.add('xt-dragging');
    root.setPointerCapture(e.pointerId);
    setTheta(fromPointer(e, drag.el));
  }
  function press(e, el, wait) {
    if (e.button !== 0) return;
    drag = { el, id: e.pointerId, x: e.clientX, y: e.clientY, live: false };
    if (!wait) { begin(e); e.preventDefault(); }
  }
  scale.addEventListener('pointerdown', (e) => press(e, scale, false));
  root.addEventListener('pointerdown', (e) => {
    const tr = e.target.closest && e.target.closest('.xt-track');
    if (tr && !tr.closest('.xt-ghead')) press(e, tr, e.pointerType === 'touch');
  });
  root.addEventListener('pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    if (!drag.live) {
      const dx = Math.abs(e.clientX - drag.x), dy = Math.abs(e.clientY - drag.y);
      if (dx > 6 && dx > dy) begin(e);
      return;
    }
    const v = fromPointer(e, drag.el);
    if (v !== theta) setTheta(v);
  });
  const endDrag = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const live = drag.live;
    drag = null;
    root.classList.remove('xt-dragging');
    if (live && e.type === 'pointerup') handle.focus({ preventScroll: true });
  };
  root.addEventListener('pointerup', endDrag);
  root.addEventListener('pointercancel', endDrag);
  handle.addEventListener('keydown', (e) => {
    const big = e.shiftKey ? 10 : 1;
    let v = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') v = theta + STEP * big;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') v = theta - STEP * big;
    else if (e.key === 'PageUp') v = theta + STEP * 10;
    else if (e.key === 'PageDown') v = theta - STEP * 10;
    else if (e.key === 'Home') v = MIN;
    else if (e.key === 'End') v = MAX;
    if (v === null) return;
    e.preventDefault();
    userActed();
    setTheta(snap(v));
  });

  // ---------- picker and random comparison ----------
  pick.addEventListener('change', () => {
    userActed();
    cur = Number(pick.value);
    random = false;
    drawRandomNote();
    drawNeuron();
    drawScores();
  });
  $('rand').addEventListener('click', () => {
    userActed();
    random = !random;
    drawRandomNote();
    drawNeuron();
    drawScores();
  });
  new ResizeObserver(() => drawLine()).observe(scale);
  document.fonts.ready.then(drawLine);              // the tag's width changes when the mono face arrives

  // ---------- the one animation on first view ----------
  // Bars grow from zero, the line drops in from its tag down through the rows, then the verdict appears.
  // Skipped under prefers-reduced-motion; any press or key inside the picture finishes it at once.
  let introPending = !reduce.matches;
  const timers = [];
  function finishIntro() {
    timers.forEach(clearTimeout);
    rows.forEach((r) => r.bar.style.removeProperty('--xt-d'));
    root.classList.remove('xt-pre-bars', 'xt-pre-line', 'xt-pre-verdict');
  }
  function userActed() {
    if (!introPending) return;
    introPending = false;
    seen.disconnect();
    root.classList.add('xt-still');
    finishIntro();
    root.getBoundingClientRect();
    root.classList.remove('xt-still');
  }
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    seen.disconnect();
    if (!introPending) return;
    introPending = false;
    rows.forEach((r) => r.bar.style.setProperty('--xt-d', r.i * 45 + 'ms'));
    root.classList.remove('xt-pre-bars');
    timers.push(setTimeout(() => root.classList.remove('xt-pre-line'), 1150));
    timers.push(setTimeout(() => { root.classList.remove('xt-pre-verdict'); rows.forEach((r) => r.bar.style.removeProperty('--xt-d')); }, 1900));
  }, { threshold: 0.9 });
  root.addEventListener('keydown', userActed, true);

  // ---------- caption ----------
  $('caption').textContent =
    'A text’s score is its peak activation divided by the neuron’s strongest activation in the corpus, and across ' +
    'the 599 OLMo-3 7B neurons tested, 1,794 of the 2,995 texts written to fire their neuron reached the study’s line, ' +
    'against 23 of 8,985 texts scored at a random neuron. A failure judges this explanation and these texts, not the ' +
    'neuron, and a neuron that barely fires anywhere can still pass, because every score is relative to its own strongest activation.';

  // ---------- first draw ----------
  pick.value = '0';
  drawRandomNote();
  drawNeuron();
  if (introPending) {
    root.classList.add('xt-still', 'xt-pre-bars', 'xt-pre-line', 'xt-pre-verdict');
    seen.observe(root.querySelector('.xt-head'));   // the chart is taller than a phone screen; its head is not
  }
  drawScores();
  root.getBoundingClientRect();
  root.classList.remove('xt-still');
})();
