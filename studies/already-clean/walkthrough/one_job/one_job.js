// "How many neurons have one job": the picture, shared by the walkthrough page and index.html.
// It builds itself inside the element with id "onejob" and reads window.ONE_JOB_DATA, which data.js sets.
// With no #onejob on the page it does nothing. Each neuron's class is computed here from its stored test
// scores, with the rule of infra/pg/neurons_verdict.py (explanation_passes, job_supported, assign_class);
// extract.py checks the same rule against the stored classes.
(() => {
  'use strict';
  const root = document.getElementById('onejob');
  if (!root) return;
  const D = window.ONE_JOB_DATA;
  if (!D) {
    root.innerHTML = '<p class="oj-missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      'one_job to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in one_job');
  }

  const GROUPS = [
    { key: 'ONE', css: 'oj-c-one', name: 'One job', lower: 'one job' },
    { key: 'TWO-REL', css: 'oj-c-two', name: 'Two related jobs', lower: 'two related jobs' },
    { key: 'UNREL', css: 'oj-c-unrel', name: 'Unrelated jobs', lower: 'unrelated jobs' },
    { key: 'UNK', css: 'oj-c-unk', name: 'Cannot tell', lower: 'cannot tell' },
  ];
  const GROUP_OF = { ONE: 0, 'TWO-REL': 1, UNREL: 2, UNK: 3, RARE: 3 };   // too rare to test counts as cannot tell
  const LABEL = { ONE: 'one job', 'TWO-REL': 'two related jobs', UNREL: 'unrelated jobs' };
  const [MIN, MAX, STEP] = D.slider;
  const STOPS = D.stops;
  const STUDY = 1;                                  // STOPS[1] is the study's line
  const N = D.n;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 640px)');

  // ---------- the rule ----------
  const nStrong = (xs, t) => xs.reduce((k, s) => k + (s >= t ? 1 : 0), 0);
  // fixed: the class does not depend on the strong line (too rare, nothing tested, the explainer named no job,
  // or its label does not match its list of jobs)
  function judge(n, t) {
    if (n.f === 1) return { cls: 'RARE', fixed: true };
    if (!n.s) return { cls: 'UNK', fixed: true };
    const pos = nStrong(n.s.slice(0, 5), t), near = nStrong(n.s.slice(5, 10), t), dist = nStrong(n.s.slice(10, 15), t);
    const passes = pos >= 3 && near <= 2 && dist <= 1;
    const rel = [...n.j].map((c) => c === '1');
    const out = { pos, near, dist, passes, cls: 'UNK', fixed: false };
    if (n.l === 'ONE') out.fixed = rel.length !== 1;
    else if (n.l === 'TWO-REL') out.fixed = !(rel.length >= 2 && rel.length <= 3 && rel.every(Boolean));
    else if (n.l === 'UNREL') {
      const ks = rel.map((r, k) => (r ? -1 : k)).filter((k) => k >= 0);
      out.unrel = { n: ks.length, sup: 0 };
      out.fixed = ks.length < 2;
      if (!out.fixed) {
        out.unrel.sup = ks.filter((k) => {
          const x = n.u && n.u[k];
          if (!x) throw new Error(n.p + ': no texts for unrelated job ' + k);
          return nStrong(x, t) >= 3;
        }).length;
        if (out.unrel.sup >= 2) out.cls = 'UNREL';
      }
      return out;
    } else out.fixed = true;                         // the explainer said it cannot tell
    if (!out.fixed && passes) out.cls = n.l;
    return out;
  }

  // ---------- static DOM ----------
  root.innerHTML = `
  <div class="oj-top">
    <div class="oj-ctl">
      <div class="oj-ctl-head">
        <label for="oj-line">Strong line</label>
        <output class="oj-val" id="oj-val" for="oj-line"></output>
        <span class="oj-note" id="oj-note"></span>
      </div>
      <div class="oj-slider" id="oj-slider">
        <input type="range" id="oj-line" min="${MIN}" max="${MAX}" step="${STEP}">
      </div>
      <p class="oj-def" id="oj-def"></p>
    </div>
    <button class="oj-btn" id="oj-random" type="button">Random neuron</button>
  </div>
  <div class="oj-legend" id="oj-legend"></div>
  <div class="oj-chart">
    <div class="oj-heads" id="oj-heads"><div></div></div>
    <div class="oj-plots" id="oj-plots">
      <div class="oj-axis" aria-hidden="true">
        <span class="oj-yl" style="top:0">${N}</span>
        <span class="oj-yl oj-yl-ref" style="top:${100 * (1 - D.essay_one / N)}%">${D.essay_one}</span>
        <span class="oj-yl" style="top:100%">0</span>
      </div>
      <div class="oj-ref" style="top:${100 * (1 - D.essay_one / N)}%" aria-hidden="true"></div>
    </div>
  </div>
  <div class="oj-panel" id="oj-panel" aria-live="polite"></div>
  <details>
    <summary>Show the counts as a table</summary>
    <div class="oj-tablewrap"><table id="oj-table"></table></div>
  </details>
  <p class="oj-caption" id="oj-caption"></p>`;
  const $ = (id) => root.querySelector('#oj-' + id);

  const fmt = (t) => {
    const k = STOPS.indexOf(t);
    return k >= 0 ? t.toFixed(4) : t.toFixed(3);
  };
  $('def').textContent = 'A test text is strong when its peak activation, divided by the neuron’s strongest ' +
    'activation in the corpus, reaches the line. The study’s line, ' + fmt(STOPS[STUDY]) + ', is the score that 1 in ' +
    '100 texts reached at a random neuron during tuning.';

  const legend = $('legend');
  GROUPS.forEach((g) => {
    const s = document.createElement('span');
    const k = document.createElement('i');
    k.className = 'oj-key ' + g.css;
    s.append(k, g.name);
    legend.appendChild(s);
  });
  const ess = document.createElement('span');
  const ek = document.createElement('i');
  ek.className = 'oj-key oj-key-line';
  ess.append(ek, 'The essay’s estimate, ' + D.essay_one + ' of ' + N + ' (' + Math.round(100 * D.essay_one / N) + '%)');
  legend.appendChild(ess);

  // stops under the slider: the thumb's center runs from 3px to (100% - 3px)
  const input = $('line');
  const frac = (t) => (t - MIN) / (MAX - MIN);
  const stopBtns = STOPS.map((t, k) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'oj-stop';
    b.textContent = t.toFixed(3);
    b.style.left = 'calc(3px + (100% - 6px) * ' + frac(t) + ')';
    b.setAttribute('aria-label', 'Set the strong line to ' + t.toFixed(4) + (k === STUDY ? ', the study’s line' : ''));
    b.addEventListener('click', () => { setTheta(t); input.focus({ preventScroll: true }); });
    $('slider').appendChild(b);
    return b;
  });

  const heads = $('heads');
  const plots = $('plots');
  const M = D.models.map((model, m) => {
    const h = document.createElement('div');
    h.className = 'oj-head';
    const name = document.createElement('div');
    name.className = 'oj-name';
    name.textContent = model.name;
    const count = document.createElement('div');
    count.className = 'oj-count';
    const of = document.createElement('div');
    of.className = 'oj-of';
    of.textContent = 'of ' + N + ' have one job';
    h.append(name, count, of);
    heads.appendChild(h);
    const grid = document.createElement('div');
    grid.className = 'oj-grid';
    grid.setAttribute('role', 'img');
    grid.dataset.m = String(m);
    const dots = model.neurons.map((n, i) => {
      const d = document.createElement('i');
      d.className = 'oj-d';
      d.dataset.i = String(i);
      grid.appendChild(d);
      return d;
    });
    plots.appendChild(grid);
    // the order the dots stand in before they sort: a fixed shuffle, the same on every load
    let seed = 12345 + m;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    const shuffle = model.neurons.map((_, i) => i);
    for (let i = shuffle.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [shuffle[i], shuffle[j]] = [shuffle[j], shuffle[i]];
    }
    return { model, grid, dots, count, shuffle, res: [], tally: [0, 0, 0, 0], shown: 0 };
  });

  // ---------- state ----------
  let theta = STOPS[STUDY];
  let sorted = false;
  let sel = null;                 // [m, i]
  let countAnim = 0;
  const cols = () => parseInt(getComputedStyle(root).getPropertyValue('--oj-cols'), 10);

  function compute() {
    M.forEach((c) => {
      c.res = c.model.neurons.map((n) => judge(n, theta));
      c.tally = [0, 0, 0, 0];
      c.res.forEach((r) => { c.tally[GROUP_OF[r.cls]]++; });
    });
  }

  // intro: true for the one sort on first view (slow, rippling from the baseline up)
  function place(intro) {
    const W = cols();
    M.forEach((c) => {
      let order;
      if (sorted) {
        order = c.res.map((r, i) => i);
        order.sort((a, b) => GROUP_OF[c.res[a].cls] - GROUP_OF[c.res[b].cls] || a - b);
      } else {
        order = c.shuffle;
      }
      order.forEach((i, k) => {
        const d = c.dots[i];
        d.style.setProperty('--x', String(k % W));
        d.style.setProperty('--y', String(Math.floor(k / W)));
        d.style.setProperty('--oj-delay', intro ? Math.round((k / N) * 900) + 'ms' : '0ms');
        const css = GROUPS[GROUP_OF[c.res[i].cls]].css;
        if (d._css !== css) {
          if (d._css) d.classList.remove(d._css);
          d.classList.add(css);
          d._css = css;
        }
      });
    });
  }

  function drawCounts(animate) {
    cancelAnimationFrame(countAnim);
    if (!animate) {
      M.forEach((c) => { c.shown = c.tally[0]; c.count.textContent = String(c.tally[0]); });
      return;
    }
    const t0 = performance.now();
    const DUR = 1500;
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / DUR);
      const e = 1 - (1 - p) ** 3;
      M.forEach((c) => { c.count.textContent = String(Math.round(c.tally[0] * e)); });
      if (p < 1) countAnim = requestAnimationFrame(tick);
    };
    countAnim = requestAnimationFrame(tick);
  }

  function describe(p) {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(p);
    return 'layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' pole';
  }

  function why(n, r) {
    const ex = D.explainers[n.e];
    if (n.f === 1) {
      return n.s ? 'Too rare to test: fewer than 50 tokens of the corpus activate it at 1% of its peak or more.'
        : 'The neuron never has an activation of this sign in the corpus, so this pole was not tested.';
    }
    if (n.f === 2) return 'The explainer’s answer was rejected three times for quoting the contexts it was shown, so nothing was tested.';
    if (n.f === 3) return 'The writer of the test texts gave no usable answer in three tries, so nothing was tested.';
    if (n.l === 'UNK') return 'The explainer, ' + ex + ', said it could not tell what this neuron does.';
    if (n.l === 'UNREL') {
      const u = r.unrel;
      if (u.n < 2) return 'The explainer, ' + ex + ', called its jobs unrelated but marked fewer than two of them unrelated, so the test gives it no class.';
      return 'The explainer, ' + ex + ', said it has unrelated jobs. That class needs at least 2 of them to have 3 or more ' +
        'strong texts out of 5 written from that job alone; here ' + u.sup + ' of ' + u.n + ' do.';
    }
    if (r.fixed) return 'The explainer, ' + ex + ', called it ' + LABEL[n.l] + ', but its list of jobs does not match that name, so the test gives it no class.';
    return 'The explanation ' + (r.passes ? 'passed' : 'failed') + ': ' + r.pos + ' of 5 texts written from it alone were strong ' +
      '(at least 3 needed), ' + r.near + ' of 5 near misses (at most 2 allowed) and ' + r.dist + ' of 5 distant texts ' +
      '(at most 1). The explainer, ' + ex + ', called it ' + LABEL[n.l] + '.';
  }

  function drawPanel() {
    const panel = $('panel');
    panel.textContent = '';
    if (!sel) {
      const p = document.createElement('p');
      p.className = 'oj-hint';
      p.textContent = 'Point at a dot, or press Random neuron, to read that neuron’s explanation and how its class was decided.';
      panel.appendChild(p);
      return;
    }
    const [m, i] = sel;
    const c = M[m];
    const n = c.model.neurons[i];
    const r = c.res[i];
    const g = GROUPS[GROUP_OF[r.cls]];
    const who = document.createElement('p');
    who.className = 'oj-who';
    const id = document.createElement('b');
    id.textContent = n.p;
    who.append(id, c.model.name + ', ' + describe(n.p) + '.');
    const cls = document.createElement('p');
    cls.className = 'oj-cls';
    const k = document.createElement('i');
    k.className = 'oj-key ' + g.css;
    const b = document.createElement('b');
    b.textContent = g.name;
    cls.append(k, b, r.fixed ? ' at every strong line' : ' at this strong line');
    panel.append(who, cls);
    if (n.x) {
      const ex = document.createElement('p');
      ex.className = 'oj-expl';
      ex.textContent = '“' + n.x + '”';
      panel.appendChild(ex);
    }
    const w = document.createElement('p');
    w.className = 'oj-why';
    w.textContent = why(n, r);
    panel.appendChild(w);
  }

  function select(m, i) {
    if (sel) M[sel[0]].dots[sel[1]].classList.remove('oj-sel');
    sel = [m, i];
    M[m].dots[i].classList.add('oj-sel');
    drawPanel();
  }

  function drawTable() {
    const table = $('table');
    table.textContent = '';
    const head = table.insertRow();
    ['At strong line ' + fmt(theta), ...M.map((c) => c.model.name)].forEach((h) => {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    });
    GROUPS.forEach((g, k) => {
      const tr = table.insertRow();
      tr.insertCell().textContent = g.name;
      M.forEach((c) => { tr.insertCell().textContent = String(c.tally[k]); });
    });
    const tr = table.insertRow();
    tr.insertCell().textContent = 'Neurons tested';
    M.forEach((c) => { tr.insertCell().textContent = String(c.model.neurons.length); });
  }

  function drawLabels() {
    const k = STOPS.indexOf(theta);
    $('val').textContent = fmt(theta);
    $('note').textContent = k === STUDY ? 'the study’s line' : k === 0 ? '0.1 below the study’s line'
      : k === 2 ? '0.1 above the study’s line' : '';
    stopBtns.forEach((b, j) => b.classList.toggle('oj-on', j === k));
    input.setAttribute('aria-valuetext', fmt(theta) + (k === STUDY ? ', the study’s line' : '') + '; one job: ' +
      M.map((c) => c.model.name + ' ' + c.tally[0]).join(', '));
    M.forEach((c) => {
      c.grid.setAttribute('aria-label', c.model.name + ', at strong line ' + fmt(theta) + ': ' +
        GROUPS.map((g, j) => c.tally[j] + ' ' + g.lower).join(', ') + ', of ' + c.model.neurons.length + ' neurons.');
    });
  }

  function setTheta(t) {
    userActed();
    theta = t;
    input.value = String(t);
    compute();
    place(false);
    drawCounts(false);
    drawLabels();
    drawTable();
    if (sel) drawPanel();
  }

  // the slider's own positions, except that within half a step of a stop the line takes the stop's exact value
  input.addEventListener('input', () => {
    const v = Number(input.value);
    const near = STOPS.find((s) => Math.abs(v - s) < STEP / 2);
    setTheta(near === undefined ? v : near);
  });

  // ---------- the sort on first view ----------
  // The dots stand in a fixed shuffle until the chart is first in view, then sort into their classes once
  // while the one-job counts count up. Skipped under prefers-reduced-motion; any press or focus inside the
  // picture before then sorts it at once.
  let introPending = !reduce.matches;
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    seen.disconnect();
    if (!introPending) return;
    introPending = false;
    sorted = true;
    root.style.setProperty('--oj-dur', '900ms');
    place(true);
    drawCounts(true);
    setTimeout(() => {
      root.style.removeProperty('--oj-dur');
      M.forEach((c) => c.dots.forEach((d) => d.style.setProperty('--oj-delay', '0ms')));
    }, 1900);
  }, { threshold: 0.35 });
  function userActed() {
    if (!introPending) return;
    introPending = false;
    seen.disconnect();
    sorted = true;
    place(false);
    drawCounts(false);
  }
  root.addEventListener('pointerdown', userActed);
  root.addEventListener('focusin', userActed);

  // ---------- pointing at dots ----------
  const dotAt = (e) => {
    const d = e.target.closest && e.target.closest('.oj-d');
    return d ? [Number(d.parentNode.dataset.m), Number(d.dataset.i)] : null;
  };
  plots.addEventListener('pointerover', (e) => {
    if (e.pointerType !== 'mouse') return;
    const hit = dotAt(e);
    if (hit && (!sel || hit[0] !== sel[0] || hit[1] !== sel[1])) select(hit[0], hit[1]);
  });
  plots.addEventListener('pointerdown', (e) => {
    const hit = dotAt(e);
    if (hit) select(hit[0], hit[1]);
  });

  $('random').addEventListener('click', () => {
    let m, i;
    do {
      m = Math.floor(Math.random() * M.length);
      i = Math.floor(Math.random() * M[m].dots.length);
    } while (sel && sel[0] === m && sel[1] === i);
    select(m, i);
  });

  narrow.addEventListener('change', () => {
    root.classList.add('oj-still');
    place(false);
    root.getBoundingClientRect();
    root.classList.remove('oj-still');
  });

  // ---------- caption ----------
  function caption() {
    const said = M.map((c) => c.model.neurons.filter((n) => n.l === 'ONE').length);
    $('caption').textContent =
      'Each dot is the positive or the negative pole of one of ' + N / 2 + ' neurons drawn at random from the model, ' +
      'classed from its test scores at the strong line; the explainer itself had said one job for ' +
      said.slice(0, -1).join(', ') + ' and ' + said[said.length - 1] + ' of the ' + N + '. ' +
      'Cannot tell means the test gave no class, not that the neuron does unrelated things, and the counts depend on ' +
      'who wrote the explanations: on OLMo-3 7B, 54.7% of neurons explained by Opus 5.5 have one job, against 44.8% of ' +
      'those explained by Sonnet 5.5.';
  }

  // ---------- first draw ----------
  input.value = String(theta);
  compute();
  M.forEach((c, m) => {
    const want = c.model.check.stops[STUDY];
    if (c.tally[0] !== want) throw new Error(c.model.name + ': ' + c.tally[0] + ' one job at the study’s line, extract.py found ' + want);
  });
  root.classList.add('oj-still');
  if (introPending) {
    place(false);
    M.forEach((c) => { c.count.textContent = '0'; });
    seen.observe(plots);
  } else {
    sorted = true;
    place(false);
    drawCounts(false);
  }
  root.getBoundingClientRect();
  root.classList.remove('oj-still');
  drawLabels();
  drawTable();
  drawPanel();
  caption();
})();
