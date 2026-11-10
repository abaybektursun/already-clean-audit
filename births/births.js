// "When a neuron's groups are born": the picture, shared by the walkthrough page and index.html.
// It builds itself inside the element with id "nbirths" and reads window.BIRTHS_DATA, which data.js sets.
// With no #nbirths on the page it does nothing. Arrow keys act while focus is inside the picture; a host that
// is only this picture sets data-keys="page" on #nbirths so they act anywhere on the page.
(() => {
  'use strict';
  const root = document.getElementById('nbirths');
  if (!root) return;

  const D = window.BIRTHS_DATA;
  if (!D) {
    root.innerHTML = '<p class="nb-missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      '<code>births</code> to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in this picture’s folder');
  }

  root.innerHTML = `
  <div class="nb-pick">
    <label class="nb-sr" for="nb-neuron">Neuron</label>
    <div class="nb-select"><select id="nb-neuron"></select></div>
    <button class="nb-btn" id="nb-random" type="button">Random neuron</button>
  </div>
  <p class="nb-who" id="nb-who"></p>
  <div class="nb-readout" aria-live="polite">
    <div>
      <div class="nb-count"><b id="nb-k"></b> of <span id="nb-m"></span> groups past their onset</div>
      <p class="nb-note" id="nb-note"></p>
    </div>
    <div>
      <div class="nb-label">Training step</div>
      <div class="nb-value" id="nb-step"></div>
      <div class="nb-pctbar" aria-hidden="true"><i id="nb-pctfill"></i></div>
      <div class="nb-label" id="nb-pct" style="margin-top:4px"></div>
    </div>
  </div>
  <div class="nb-chart" role="group" aria-label="The neuron's groups at all 16 checkpoints">
    <div class="nb-head">
      <h3>The neuron’s groups, in the order of their onsets</h3>
      <div class="nb-legend">
        <span><i class="nb-key is-on"></i>present, from its onset on</span>
        <span><i class="nb-key is-pre"></i>present, but not at the next checkpoint</span>
        <span><i class="nb-key is-off"></i>not present</span>
        <span><i class="nb-key is-chance"></i>chance level: random groups of the same size</span>
      </div>
    </div>
    <div class="nb-scrub" id="nb-scrub">
      <div class="nb-overlay"><div class="nb-band" id="nb-band"></div></div>
      <div id="nb-rows"></div>
      <div class="nb-line nb-axis">
        <span class="nb-axisname">Training step</span>
        <div class="nb-plot nb-track" id="nb-track" role="slider" tabindex="0" aria-label="Checkpoint"
             aria-valuemin="0" aria-valuemax="15">
          <div class="nb-rail"></div>
          <div class="nb-thumb" id="nb-thumb"></div>
        </div>
      </div>
      <div class="nb-line"><div class="nb-plot nb-ticklabels" id="nb-ticks"></div></div>
      <div class="nb-line nb-strip">
        <span class="nb-stripname" id="nb-stripname"></span>
        <div class="nb-plot"><div class="nb-bars" id="nb-bars"></div><div class="nb-base"></div></div>
      </div>
      <div class="nb-tip" id="nb-tip"></div>
    </div>
    <p class="nb-pop" id="nb-pop"></p>
    <div class="nb-controls">
      <button class="nb-btn" id="nb-play" type="button"></button>
      <span class="nb-hint">Drag across the chart, or use the left and right arrow keys.</span>
    </div>
    <details>
      <summary>Show this neuron’s groups as a table</summary>
      <div class="nb-tablewrap"><table id="nb-table"></table></div>
    </details>
  </div>
  <p class="nb-caption" id="nb-caption"></p>`;

  const $ = (id) => root.querySelector('#nb-' + id);
  const NS = D.steps.length;
  const LAST = NS - 1;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 640px)');
  const STEP_MS = 620;

  // ---------- formatting ----------
  const int = (x) => Math.round(x).toLocaleString('en-US');
  const fmtPct = (p) => (p === 0 ? '0' : p >= 100 ? '100' : String(Number(p.toPrecision(3))));
  const compact = (s) => {
    if (s < 1000) return String(s);
    if (s < 1e6) return String(Number((s / 1000).toPrecision(3))) + 'k';
    return String(Number((s / 1e6).toPrecision(3))) + 'M';
  };
  const show = (s) => s.replace(/\n/g, '↵').replace(/\t/g, '⇥');
  const one = (x) => (Math.round(x * 10) / 10).toLocaleString('en-US', { minimumFractionDigits: 1 });
  // a chance level is a checkpoint, or half-way between two when the random groups' median falls there
  const pointText = (t) => (Number.isInteger(t) ? 'step ' + int(D.steps[t].step)
    : 'between steps ' + int(D.steps[Math.floor(t)].step) + ' and ' + int(D.steps[Math.ceil(t)].step));
  const VS = {
    later: 'onset later than chance', earlier: 'onset earlier than chance', chance: 'onset at chance',
  };
  const describe = (p) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(p);
    return 'Layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' activations.';
  };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const NOTE = {
    'L03N6372+': 'None of its 50 strongest contexts at the end is ‘ lithium’, so the lithium contexts that ' +
      'filled its top 20 from step 60,000 to step 320,000 have no row here.',
  };

  // ---------- per-neuron arrays ----------
  D.neurons.forEach((n) => {
    n.g = n.groups.map(([name, size, birth, pres, ex, chance, vs]) => ({
      name, size, birth, ex, pres: [...pres].map((c) => c === '1'), chance, vs,
    }));
  });
  const pop = D.population;
  const C = D.chance;
  const popCum = pop.births.reduce((acc, x, i) => (acc.push((i ? acc[i - 1] : 0) + x), acc), []);

  // ---------- static DOM: picker ----------
  const select = $('neuron');
  [['named', 'Named in the walkthrough'], ['random', 'Drawn at random']].forEach(([kind, label]) => {
    const members = D.neurons.map((n, i) => [n, i]).filter(([n]) => n.kind === kind);
    if (!members.length) return;
    const og = document.createElement('optgroup');
    og.label = label + ' (' + members.length + ')';
    members.forEach(([n, i]) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = n.pole + '  ' + (n.label || n.g.length + (n.g.length === 1 ? ' group' : ' groups'));
      og.appendChild(o);
    });
    select.appendChild(og);
  });
  const randomIdx = D.neurons.map((n, i) => (n.kind === 'random' ? i : -1)).filter((i) => i >= 0);

  // ---------- static DOM: axis and population strip ----------
  const colPct = (i) => ((i + 0.5) / NS) * 100;
  const track = $('track');
  const tickLabels = [];
  for (let s = 0; s < NS; s++) {
    const t = document.createElement('i');
    t.className = 'nb-tick';
    t.style.left = colPct(s) + '%';
    track.insertBefore(t, $('thumb'));
    const l = document.createElement('span');
    l.className = 'nb-tl' + (s === 0 ? ' is-first' : '') + (s === LAST ? ' is-last' : '');
    l.style.left = colPct(s) + '%';
    l.textContent = compact(D.steps[s].step);
    $('ticks').appendChild(l);
    tickLabels.push(l);
  }
  const maxBirths = Math.max(...pop.births, ...pop.chance);
  const bars = pop.births.map((x, s) => {
    const col = document.createElement('span');
    col.className = 'nb-barcol';
    col.dataset.s = String(s);
    const b = document.createElement('i');
    b.className = 'nb-bar';
    b.style.height = (x / maxBirths) * 100 + '%';
    b.style.minHeight = x ? '2px' : '0';
    col.appendChild(b);
    const ch = document.createElement('i');
    ch.className = 'nb-barchance';
    ch.style.bottom = (pop.chance[s] / maxBirths) * 100 + '%';
    col.appendChild(ch);
    $('bars').appendChild(col);
    return b;
  });
  $('stripname').textContent = 'All ' + int(pop.n_groups) + ' groups of the ' + int(pop.n_neurons) + ' neurons, by the checkpoint of their onset';

  // ---------- state ----------
  let N = D.neurons[0];
  let cur = LAST;
  let rowEls = [];
  let dragging = false;
  let timer = 0;

  // ---------- rows ----------
  const rowsBox = $('rows');
  function buildRows() {
    rowsBox.textContent = '';
    rowEls = N.g.map((g, gi) => {
      const row = document.createElement('div');
      row.className = 'nb-line nb-row';
      const name = document.createElement('span');
      name.className = 'nb-name';
      name.textContent = g.name;
      name.title = g.name;
      const plot = document.createElement('div');
      plot.className = 'nb-plot nb-cells';
      const cells = [];
      for (let s = 0; s < NS; s++) {
        const c = document.createElement('i');
        c.className = 'nb-cell';
        c.dataset.g = String(gi);
        c.dataset.s = String(s);
        plot.appendChild(c);
        cells.push(c);
      }
      if (g.chance !== null) {
        const k = document.createElement('i');
        k.className = 'nb-chance';
        k.style.left = colPct(g.chance) + '%';
        plot.appendChild(k);
      }
      row.append(name, plot);
      rowsBox.appendChild(row);
      return { row, cells };
    });
  }

  function cellState(g, s) {
    if (s > cur) return '';
    if (!g.pres[s]) return 'is-off';
    return s >= g.birth ? 'is-on' : 'is-pre';
  }

  // stagger: delay per row (ms) for rows whose state changes in this update
  function paintRows(stagger) {
    let k = 0;
    N.g.forEach((g, gi) => {
      const { row, cells } = rowEls[gi];
      const born = g.birth <= cur;
      const was = row.classList.contains('is-born');
      const delay = stagger && born && !was ? (k++) * stagger : 0;
      row.classList.toggle('is-born', born);
      cells.forEach((c, s) => {
        const cls = 'nb-cell' + (cellState(g, s) ? ' ' + cellState(g, s) : '');
        if (c.className !== cls) {
          c.style.transitionDelay = delay + 'ms';
          c.className = cls;
        }
      });
      row.querySelector('.nb-name').style.transitionDelay = delay + 'ms';
    });
  }

  // ---------- per-neuron views ----------
  function rangesText(g) {
    const runs = [];
    let start = -1;
    for (let s = 0; s <= NS; s++) {
      const on = s < NS && g.pres[s];
      if (on && start < 0) start = s;
      if (!on && start >= 0) {
        const a = compact(D.steps[start].step), b = compact(D.steps[s - 1].step);
        runs.push(start === s - 1 ? a : a + '–' + b);
        start = -1;
      }
    }
    return runs.join(', ');
  }

  function chanceText() {
    if (N.g[0].chance === null) return ' Its one group holds all its contexts, so it has no chance level.';
    const k = { later: 0, earlier: 0, chance: 0 };
    N.g.forEach((g) => { k[g.vs] += 1; });
    return ' Against random groups of the same size from this neuron, ' + k.later +
      (k.later === 1 ? ' group has its' : ' groups have their') + ' onset later than chance, ' + k.earlier + ' earlier and ' +
      k.chance + ' at chance.';
  }

  function drawNeuron() {
    const who = $('who');
    who.textContent = '';
    const idb = document.createElement('b');
    idb.textContent = N.pole;
    who.append(idb, ' ' + describe(N.pole) + ' ' +
      (N.label ? cap(N.label) + '.' : 'Drawn at random from the ' + int(pop.n_neurons) + ' neurons studied.') + ' ' +
      N.g.length + (N.g.length === 1 ? ' group' : ' groups') + ' at the end of training.' + chanceText() +
      (NOTE[N.pole] ? ' ' + NOTE[N.pole] : ''));
    $('m').textContent = String(N.g.length);
    buildRows();

    const table = $('table');
    table.textContent = '';
    const head = table.insertRow();
    ['Group', 'Contexts', 'Onset at step', 'Chance level', 'Against chance', 'Present at steps',
      'Its strongest context at the end'].forEach((h) => {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    });
    N.g.forEach((g) => {
      const tr = table.insertRow();
      tr.insertCell().textContent = g.name;
      const n = tr.insertCell(); n.className = 'nb-num'; n.textContent = String(g.size);
      const b = tr.insertCell(); b.className = 'nb-num'; b.textContent = int(D.steps[g.birth].step);
      tr.insertCell().textContent = g.chance === null ? 'none (one group)' : pointText(g.chance);
      tr.insertCell().textContent = g.chance === null ? '' : VS[g.vs].replace('onset ', '');
      tr.insertCell().textContent = rangesText(g);
      const c = tr.insertCell();
      c.className = 'nb-ctxcell';
      const mark = document.createElement('span');
      mark.className = 'nb-mark';
      mark.textContent = '[' + show(g.ex[1]) + ']';
      c.append('…' + show(g.ex[0]).slice(-36), mark, show(g.ex[2]).slice(0, 16) + '…');
    });
  }

  // ---------- per-step views ----------
  function drawStep(stagger) {
    const st = D.steps[cur];
    const born = N.g.filter((g) => g.birth <= cur);
    const absent = born.filter((g) => !g.pres[cur]).length;
    $('k').textContent = String(born.length);
    $('note').textContent = born.length === 0
      ? 'No group of this neuron has had its onset yet.'
      : 'by step ' + int(st.step) + '.' +
        (absent ? ' ' + absent + ' of them ' + (absent === 1 ? 'is' : 'are') + ' not present at this checkpoint.' : '');
    $('step').textContent = int(st.step);
    $('pct').textContent = fmtPct(st.pct) + '% of training';
    $('pctfill').style.width = st.pct + '%';
    $('band').style.left = (cur / NS) * 100 + '%';
    if (!dragging) $('thumb').style.left = colPct(cur) + '%';
    tickLabels.forEach((l, s) => {
      l.classList.toggle('is-cur', s === cur);
      l.style.visibility = narrow.matches && ((s === 0 && cur === 1) || (s === LAST && cur === LAST - 1)) ? 'hidden' : '';
    });
    bars.forEach((b, s) => { b.className = 'nb-bar' + (s === cur ? ' is-cur' : s < cur ? ' is-past' : ''); });
    const pp = $('pop');
    pp.textContent = '';
    const k = document.createElement('b');
    k.textContent = int(popCum[cur]);
    pp.append('All ' + int(pop.n_neurons) + ' neurons: ', k, ' of ' + int(pop.n_groups) + ' groups (' +
      fmtPct((100 * popCum[cur]) / pop.n_groups) + '%) with their onset by step ' + int(st.step) + ', ' + int(pop.births[cur]) +
      ' of them at this checkpoint.');
    paintRows(stagger);
    track.setAttribute('aria-valuenow', String(cur));
    track.setAttribute('aria-valuetext', 'step ' + int(st.step) + ', ' + fmtPct(st.pct) + '% of training, ' +
      born.length + ' of ' + N.g.length + ' groups past their onset');
  }

  function setStep(s, stagger) {
    s = Math.max(0, Math.min(LAST, s));
    if (s === cur) return;
    cur = s;
    drawStep(stagger);
  }

  function setNeuron(i) {
    N = D.neurons[i];
    select.value = String(i);
    drawNeuron();
    drawStep(0);
    caption();
  }

  // ---------- play ----------
  const playBtn = $('play');
  const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.6 6z"/></svg>';
  const PAUSE_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';
  function setPlayLabel() {
    playBtn.innerHTML = timer ? PAUSE_ICON : PLAY_ICON;
    playBtn.append(timer ? 'Pause' : 'Play through training');
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = 0;
    root.dataset.playing = '';
    setPlayLabel();
  }
  function play(by) {
    if (cur === LAST) { cur = -1; setStep(0, 0); }
    timer = setInterval(() => {
      if (cur >= LAST) { stop(); return; }
      setStep(cur + 1, reduce.matches ? 0 : 45);
      if (cur >= LAST) stop();
    }, STEP_MS);
    root.dataset.playing = by;
    setPlayLabel();
  }
  playBtn.addEventListener('click', () => (timer ? stop() : play('button')));
  setPlayLabel();

  // ---------- autoplay: once, from the first point, when the chart is first in view ----------
  // Skipped under prefers-reduced-motion, where the last point shows. Any press inside the picture, or any key
  // press while it runs, stops it; a press or focus inside the picture before it starts cancels it.
  let autoPending = !reduce.matches;
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    seen.disconnect();
    if (!autoPending) return;
    autoPending = false;
    play('auto');
  }, { threshold: 0.35 });
  if (autoPending) seen.observe($('scrub'));
  const userActed = () => {
    if (autoPending && cur === 0 && !dragging) setStep(LAST, 0);
    autoPending = false;
    seen.disconnect();
  };
  root.addEventListener('pointerdown', (e) => {
    if (playBtn.contains(e.target)) { autoPending = false; seen.disconnect(); return; }
    if (root.dataset.playing === 'auto') stop();
    userActed();
  });
  root.addEventListener('focusin', (e) => { if (e.target !== playBtn) userActed(); });
  document.addEventListener('keydown', (e) => {
    if (root.dataset.playing === 'auto' && e.target !== playBtn) stop();
  });

  // ---------- scrub ----------
  const scrub = $('scrub');
  const tip = $('tip');
  const frac = (e) => {
    const r = track.getBoundingClientRect();
    return Math.max(0, Math.min(0.99999, (e.clientX - r.left) / r.width));
  };
  scrub.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const r = track.getBoundingClientRect();
    if (e.clientX < r.left - 8) return;          // a press on the group names does not move the point
    dragging = true;
    stop();
    scrub.classList.add('is-dragging');
    scrub.setPointerCapture(e.pointerId);
    tip.classList.remove('is-on');
    track.focus({ preventScroll: true });
    dragTo(e);
  });
  function dragTo(e) {
    const f = frac(e);
    $('thumb').style.left = Math.max(0.5 / NS, Math.min(1 - 0.5 / NS, f)) * 100 + '%';
    setStep(Math.floor(f * NS), 0);
  }
  scrub.addEventListener('pointermove', (e) => {
    if (dragging) { dragTo(e); return; }
    if (e.pointerType !== 'mouse') return;
    const c = e.target.closest('.nb-cell');
    const col = e.target.closest('.nb-barcol');
    if (c) showTip(Number(c.dataset.g), Number(c.dataset.s), c);
    else if (col) showBarTip(Number(col.dataset.s), col);
    else tip.classList.remove('is-on');
  });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    scrub.classList.remove('is-dragging');
    $('thumb').style.left = colPct(cur) + '%';
  };
  scrub.addEventListener('pointerup', endDrag);
  scrub.addEventListener('pointercancel', endDrag);
  scrub.addEventListener('pointerleave', () => tip.classList.remove('is-on'));

  function placeTip(el) {
    const sr = scrub.getBoundingClientRect();
    const cr = el.getBoundingClientRect();
    const x = cr.left + cr.width / 2 - sr.left;
    const w = tip.offsetWidth;
    tip.style.left = Math.max(w / 2, Math.min(sr.width - w / 2, x)) + 'px';
    tip.style.top = cr.top - sr.top + 'px';
    tip.classList.add('is-on');
  }

  function showBarTip(s, col) {
    const st = D.steps[s];
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = int(pop.births[s]) + (pop.births[s] === 1 ? ' group' : ' groups') + ' with their onset at this checkpoint';
    tip.append(b, '\nStep ' + int(st.step) + ' (' + fmtPct(st.pct) + '% of training), all ' + int(pop.n_neurons) +
      ' neurons\n' + int(popCum[s]) + ' of ' + int(pop.n_groups) + ' with their onset by then\nChance level: ' + one(pop.chance[s]) +
      ' random groups of the same sizes with their onset here, the mean of ' + pop.chance_draws + ' draws');
    placeTip(col);
  }

  function showTip(gi, s, cell) {
    const g = N.g[gi];
    const st = D.steps[s];
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = g.name;
    const state = !g.pres[s] ? 'not present' : s >= g.birth ? 'present' : 'present, but not at the next checkpoint';
    const ctx = document.createElement('span');
    ctx.className = 'nb-ctx';
    const mark = document.createElement('span');
    mark.className = 'nb-mark';
    mark.textContent = show(g.ex[1]);
    ctx.append('…' + show(g.ex[0]).slice(-34), mark, show(g.ex[2]).slice(0, 14) + '…');
    tip.append(b, '\nStep ' + int(st.step) + ' (' + fmtPct(st.pct) + '% of training): ' + state +
      '\nOnset at step ' + int(D.steps[g.birth].step) + '; ' + g.size + (g.size === 1 ? ' context' : ' contexts') +
      (g.chance === null ? '\nNo chance level: the neuron has one group'
        : '\nChance level: ' + pointText(g.chance) + ', the median onset of random groups of its size; ' + VS[g.vs]) +
      '\nIts strongest context at the end:', ctx);
    placeTip(cell);
  }

  // ---------- keys ----------
  (root.dataset.keys === 'page' ? document : root).addEventListener('keydown', (e) => {
    if (e.target === select || e.altKey || e.ctrlKey || e.metaKey) return;
    let s = -1;
    const onTrack = e.target === track;
    if (e.key === 'ArrowRight' || (onTrack && e.key === 'ArrowUp')) s = cur + 1;
    else if (e.key === 'ArrowLeft' || (onTrack && e.key === 'ArrowDown')) s = cur - 1;
    else if (e.key === 'Home') s = 0;
    else if (e.key === 'End') s = LAST;
    else if (e.key === ' ' && onTrack) { e.preventDefault(); timer ? stop() : play('key'); return; }
    else return;
    e.preventDefault();
    stop();
    setStep(s, 0);
  });

  // ---------- neuron controls ----------
  select.addEventListener('change', () => { stop(); setNeuron(Number(select.value)); });
  narrow.addEventListener('change', () => drawStep(0));
  $('random').addEventListener('click', () => {
    stop();
    const pool = randomIdx.filter((i) => D.neurons[i] !== N);
    setNeuron(pool[Math.floor(Math.random() * pool.length)]);
  });

  // ---------- caption ----------
  function caption() {
    const early = D.steps.filter((s) => s.pct < 1).length;
    let gap = 0, gi = 0;
    for (let s = 1; s < NS; s++) if (D.steps[s].pct - D.steps[s - 1].pct > gap) { gap = D.steps[s].pct - D.steps[s - 1].pct; gi = s; }
    $('caption').textContent =
      'Rows exist only for groups among the neuron’s 50 strongest contexts at the end of training, so a group that ' +
      'appeared and faded before the end has no row. The ' + NS + ' checkpoints are drawn evenly, but ' + early + ' fall in ' +
      'the first 1% of training and none between ' + fmtPct(D.steps[gi - 1].pct) + '% and ' + fmtPct(D.steps[gi].pct) +
      '%, so an onset, which is only the crossing of the neuron’s top-0.1% line, is dated only to the stretch since the checkpoint before. ' +
      'Ticks mark the chance level. On each row the tick is the median onset checkpoint of ' + int(C.reps) + ' random groups ' +
      'of the same size drawn from the same neuron’s 50 strongest contexts and dated by the same rule (for the strongest ' +
      'group, random groups that hold the neuron’s top context); on each bar it is the mean count of ' + pop.chance_draws +
      ' random groupings of all ' + int(pop.n_neurons) + ' neurons. Of the ' + int(C.n_groups) + ' groups on neurons with ' +
      'two or more groups, ' + C.later + ' have their onset later than ' + Math.round(100 * (1 - C.alpha)) + '% of their random ' +
      'groups and ' + C.earlier + ' earlier, where chance gives ' + one(C.later_expected) + ' and ' + one(C.earlier_expected) +
      '; the other ' + int(C.n_groups - C.later - C.earlier) + ' fall within the range of their random groups.';
  }

  setNeuron(0);
  if (autoPending) { cur = -1; setStep(0, 0); }
})();
