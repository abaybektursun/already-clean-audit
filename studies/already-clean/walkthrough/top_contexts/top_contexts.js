// "What a top context is": one neuron's 50 strongest contexts, placed on the stream of scored tokens.
// It builds itself inside the element with id "topctx" and reads window.TOP_CONTEXTS_DATA, which data.js
// sets. With no #topctx on the page it does nothing. No framework.
(() => {
  'use strict';
  const root = document.getElementById('topctx');
  if (!root) return;
  const D = window.TOP_CONTEXTS_DATA;
  if (!D) {
    root.innerHTML = '<p class="missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      'top_contexts to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in top_contexts');
  }

  root.innerHTML = `
  <div class="pick">
    <label class="sr" for="tc-neuron">Neuron</label>
    <div class="select-wrap"><select id="tc-neuron"></select></div>
    <button class="btn" id="tc-random" type="button">Random neuron</button>
  </div>
  <p class="who" id="tc-who"></p>
  <div class="fig" id="tc-fig">
    <div class="fig-head">
      <h3><span class="count" id="tc-count"></span> scored tokens, and the ${D.k} where this neuron fired hardest</h3>
      <div class="legend">
        <span><i class="key sel"></i>the open context</span>
        <span><i class="key line"></i>99.9th percentile: at most 1 in 1,000 scored tokens goes past it</span>
      </div>
    </div>
    <div class="plot" id="tc-plot" role="slider" tabindex="0" aria-valuemin="1" aria-valuemax="${D.k}"
         aria-label="The ${D.k} strongest contexts, by rank">
      <div class="area" id="tc-area">
        <div class="strack"></div>
        <div class="drawn" id="tc-drawn"></div>
        <i class="head-dot" id="tc-head"></i>
      </div>
    </div>
    <div class="xaxis" id="tc-xaxis"></div>
    <p class="xtitle">Position in the corpus, in millions of tokens, first to last</p>
    <div class="tip" id="tc-tip"></div>
  </div>
  <div class="readout" id="tc-readout" aria-live="polite">
    <div class="stat"><div class="label">Rank</div><div class="value" id="tc-rank"></div></div>
    <div class="stat"><div class="label">Activation</div><div class="value" id="tc-act"></div></div>
    <div class="stat"><div class="label">Position in the corpus</div><div class="value" id="tc-pos"></div></div>
    <div class="stat"><div class="label">Document</div><div class="value" id="tc-doc"></div><div class="note" id="tc-docnote"></div></div>
  </div>
  <div class="open" id="tc-open"></div>
  <div class="controls">
    <button class="btn" id="tc-prev" type="button"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M8.5 1.5 3.5 6l5 4.5z"/></svg>Stronger</button>
    <button class="btn" id="tc-next" type="button">Weaker<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3.5 1.5 8.5 6l-5 4.5z"/></svg></button>
    <span class="hint">Click a tick to open its context, or step through the ${D.k} by rank.</span>
  </div>
  <details>
    <summary>Show the ${D.k} as a table</summary>
    <div class="tablewrap"><table id="tc-table"></table></div>
  </details>
  <p class="caption" id="tc-caption"></p>`;

  const $ = (id) => root.querySelector('#tc-' + id);
  const K = D.k;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 640px)');
  const INTRO_MS = 2600;
  const MINUS = '−';

  // ---------- formatting ----------
  const int = (x) => Math.round(x).toLocaleString('en-US');
  const signed = (s, neg) => (neg ? MINUS + s : s);
  const fmt = (x) => {
    const a = Math.abs(x);
    const s = a >= 100 ? int(a) : a === 0 ? '0' : String(Number(a.toPrecision(3)));
    return x < 0 ? MINUS + s : s;
  };
  const quoted = (tok) => "'" + tok.replace(/\n/g, '\\n').replace(/\t/g, '\\t') + "'";
  const pillText = (tok) => {
    const t = tok.replace(/\n/g, '↵').replace(/\t/g, '⇥');
    return t.trim() === '' ? t.replace(/ /g, '␣') : t.trim();
  };
  const describe = (p) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(p);
    return 'Layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' +
      (m[3] === '+' ? 'positive activations.' : 'negative activations, drawn upward by size.');
  };
  const millions = (n) => (n >= 1e6 ? Number((n / 1e6).toPrecision(2)) + ' million' : int(Number(n.toPrecision(2))));
  // tokens scored among the first x positions of the stream: every position but each window's first
  const scoredBefore = (x) => x - Math.ceil(x / D.window);

  // ---------- per-neuron summary ----------
  D.neurons.forEach((n) => {
    n.neg = n.pole.endsWith('-');
    const c = new Map();
    n.ctx.forEach((e) => c.set(e[4], (c.get(e[4]) || 0) + 1));
    n.common = [...c.entries()].sort((a, b) => b[1] - a[1]);
    n.docCount = new Map();
    n.ctx.forEach((e) => n.docCount.set(e[1], (n.docCount.get(e[1]) || 0) + 1));
  });
  const LI = quoted(' lithium').replace(/ /g, '\u00a0');
  const NOTE = {
    lithium: (n) => 'The lithium neuron: for part of training it fired almost only on ' + LI + '. ' +
      (n.common.some(([t]) => t === ' lithium') ? '' : 'None of these ' + K + ' is ' + LI + '. '),
    got: () => 'The “got” neuron of the walkthrough’s first finding. ',
    random: () => 'Drawn at random from the 600 neurons studied. ',
  };

  // ---------- picker ----------
  const picker = $('neuron');
  [['named', 'Named in the text', (n) => n.group !== 'random'], ['random', 'Drawn at random', (n) => n.group === 'random']]
    .forEach(([, label, keep]) => {
      const members = D.neurons.map((n, i) => [n, i]).filter(([n]) => keep(n));
      const og = document.createElement('optgroup');
      og.label = label + ' (' + members.length + ')';
      members.forEach(([n, i]) => {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = n.pole + '  ' + (n.group === 'lithium' ? 'the lithium neuron'
          : n.group === 'got' ? 'the “got” neuron' : quoted(n.common[0][0]));
        og.appendChild(o);
      });
      picker.appendChild(og);
    });
  const randomIdx = D.neurons.map((n, i) => (n.group === 'random' ? i : -1)).filter((i) => i >= 0);

  // ---------- x axis (static) ----------
  const xPct = (tok) => (tok / D.n_stream) * 100;
  function drawXAxis() {
    const ax = $('xaxis');
    ax.textContent = '';
    const every = narrow.matches ? 10 : 5;
    for (let m = 0; m * 1e6 <= D.n_stream; m += every) {
      const t = document.createElement('i');
      t.className = 'xt';
      t.style.left = xPct(m * 1e6) + '%';
      const l = document.createElement('span');
      l.className = 'xl' + (m === 0 ? ' first' : '');
      l.style.left = xPct(m * 1e6) + '%';
      l.textContent = String(m);
      ax.append(t, l);
    }
  }

  // ---------- state ----------
  const area = $('area');
  const plot = $('plot');
  const tip = $('tip');
  let N = null;
  let spikes = [];
  let yEls = [];
  let top = 1;
  let sel = 0;
  let hover = -1;
  let phase = reduce.matches ? 'done' : 'pending';   // pending -> running -> done
  let raf = 0;

  const hPct = (e) => (Math.abs(e[2]) / top) * 100;

  function niceStep(x) {
    const p = 10 ** Math.floor(Math.log10(x));
    const f = x / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }

  function drawYAxis() {
    yEls.forEach((e) => e.remove());
    yEls = [];
    const H = area.clientHeight;
    const step = niceStep(N.max / 3);
    const qpx = (N.q999 / top) * H;
    const add = (cls, v, text) => {
      const l = document.createElement('span');
      l.className = 'yl' + (cls ? ' ' + cls : '');
      l.style.bottom = (v / top) * 100 + '%';
      l.textContent = text;
      area.appendChild(l);
      yEls.push(l);
    };
    for (let k = 0; k * step <= top; k++) {
      const v = k * step;
      if (v > 0) {
        const g = document.createElement('i');
        g.className = 'grid';
        g.style.bottom = (v / top) * 100 + '%';
        area.insertBefore(g, area.firstChild);
        yEls.push(g);
      }
      if (Math.abs((v / top) * H - qpx) >= 15) add('', v, signed(String(Number(v.toPrecision(3))), N.neg && v > 0));
    }
    const q = document.createElement('i');
    q.className = 'qline';
    q.style.bottom = (N.q999 / top) * 100 + '%';
    area.appendChild(q);
    yEls.push(q);
    add('q', N.q999, signed(N.q999.toPrecision(2), N.neg));
  }

  function buildSpikes() {
    spikes.forEach((s) => s.remove());
    spikes = N.ctx.map((e) => {
      const s = document.createElement('i');
      s.className = 'spike';
      s.style.left = xPct(e[0]) + '%';
      area.insertBefore(s, $('drawn'));
      return s;
    });
  }

  function raise(mode) {
    // mode 'now': full height at once; 'grow': rise from zero, left to right, over 500 ms
    if (mode === 'now' || reduce.matches) {
      spikes.forEach((s, i) => { s.style.transition = 'none'; s.style.height = hPct(N.ctx[i]) + '%'; });
      area.getBoundingClientRect();
      spikes.forEach((s) => { s.style.transition = ''; });
      return;
    }
    area.getBoundingClientRect();
    spikes.forEach((s, i) => {
      const delay = Math.round((N.ctx[i][0] / D.n_stream) * 500);
      s.style.transition = 'height 300ms cubic-bezier(.2, .8, .3, 1) ' + delay + 'ms, background-color 200ms ease';
      s.style.height = hPct(N.ctx[i]) + '%';
    });
  }

  function drawTable() {
    const table = $('table');
    table.textContent = '';
    const head = table.insertRow();
    ['Rank', 'Activation', 'Position in the corpus', 'Document', 'Marked token'].forEach((h) => {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    });
    N.ctx.forEach((e, i) => {
      const tr = table.insertRow();
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'rk';
      b.textContent = String(i + 1);
      b.addEventListener('click', () => {
        select(i);
        $('open').scrollIntoView({ block: 'nearest', behavior: reduce.matches ? 'auto' : 'smooth' });
      });
      tr.insertCell().appendChild(b);
      [fmt(e[2]), int(e[0]), int(e[1]), quoted(e[4])].forEach((v) => { tr.insertCell().textContent = v; });
    });
  }

  function setNeuron(i, how) {
    N = D.neurons[i];
    picker.value = String(i);
    top = N.max * 1.06;
    const who = $('who');
    who.textContent = '';
    const idb = document.createElement('b');
    idb.textContent = N.pole;
    const common = [];
    N.common.slice(0, 3).forEach(([t, c], j) => {
      const tk = document.createElement('span');
      tk.className = 'tk';
      tk.textContent = quoted(t) + ' ' + c;
      common.push(j ? ', ' : '', tk);
    });
    who.append(idb, ' ' + describe(N.pole) + ' ' + NOTE[N.group](N) +
      'Most common marked tokens among the ' + K + ': ', ...common, '.');
    drawYAxis();
    buildSpikes();
    drawTable();
    caption();
    if (how !== 'pending') raise(how);
    select(0);
  }

  // ---------- the open context ----------
  function appendText(el, text) {
    text.split('\n').forEach((p, i) => {
      if (i > 0) {
        const nl = document.createElement('span');
        nl.className = 'nl';
        nl.textContent = '↵';
        el.appendChild(nl);
      }
      if (p) el.appendChild(document.createTextNode(p.replace(/\t/g, '  ')));
    });
  }

  function contextRow(e) {
    const row = document.createElement('div');
    row.className = 'ctx';
    const b = document.createElement('span');
    b.className = 'b';
    const bi = document.createElement('span');
    appendText(bi, e[3]);
    b.appendChild(bi);
    const t = document.createElement('span');
    t.className = 't';
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = pillText(e[4]);
    pill.title = quoted(e[4]);
    t.appendChild(pill);
    const a = document.createElement('span');
    a.className = 'a';
    appendText(a, e[5]);
    const v = document.createElement('span');
    v.className = 'v';
    v.textContent = fmt(e[2]);
    row.append(b, t, a, v);
    return row;
  }

  let swap = 0;
  function select(r) {
    sel = Math.max(0, Math.min(K - 1, r));
    const e = N.ctx[sel];
    spikes.forEach((s, i) => s.classList.toggle('sel', i === sel));
    const small = (txt) => { const s = document.createElement('small'); s.textContent = txt; return s; };
    $('rank').replaceChildren(String(sel + 1), small(' of ' + K));
    $('act').textContent = fmt(e[2]);
    $('pos').replaceChildren(int(e[0]), small(' of ' + int(D.n_stream)));
    $('doc').textContent = int(e[1]);
    const nd = N.docCount.get(e[1]);
    $('docnote').textContent = nd === 1 ? 'the only one of the ' + K + ' from it' : 'holds ' + nd + ' of the ' + K;
    const open = $('open');
    const token = ++swap;
    const put = () => {
      if (token !== swap) return;
      open.replaceChildren(contextRow(e));
      open.classList.remove('out');
    };
    if (reduce.matches || !open.firstChild) put();
    else { open.classList.add('out'); setTimeout(put, 140); }
    $('prev').disabled = sel === 0;
    $('next').disabled = sel === K - 1;
    [...$('table').rows].forEach((tr, i) => tr.classList.toggle('on', i === sel + 1));
    plot.setAttribute('aria-valuenow', String(sel + 1));
    plot.setAttribute('aria-valuetext', 'rank ' + (sel + 1) + ', activation ' + fmt(e[2]) + ', token ' +
      quoted(e[4]) + ', position ' + int(e[0]));
  }

  // ---------- caption ----------
  function caption() {
    $('caption').textContent =
      'Each tick is one of the ' + K + ' at its position among the ' + int(D.n_tokens_scored) + ' tokens scored for ' +
      D.model + ' at step ' + int(D.step) + ', drawn as high as its activation. The ' + K + ' are the neuron’s ' +
      'peak, not all it does: about ' + millions(N.density) + ' scored tokens reached at least 1% of its strongest activation.';
  }

  // ---------- intro: the line draws across once, the ticks rise as it passes, the strongest opens ----------
  const drawn = $('drawn');
  const head = $('head');
  const count = $('count');
  function finishIntro() {
    if (phase === 'done') return;
    cancelAnimationFrame(raf);
    phase = 'done';
    seen.disconnect();
    drawn.style.transform = 'scaleX(1)';
    head.classList.remove('on');
    count.textContent = int(D.n_tokens_scored);
    raise('now');
    root.classList.remove('pre');
  }
  function runIntro() {
    phase = 'running';
    const ease = (t) => 0.5 - Math.cos(Math.PI * t) / 2;
    const order = N.ctx.map((e, i) => [e[0] / D.n_stream, i]).sort((a, b) => a[0] - b[0]);
    let next = 0;
    let t0 = 0;
    head.classList.add('on');
    const frame = (now) => {
      if (phase !== 'running') return;
      if (!t0) t0 = now;
      const t = Math.min(1, (now - t0) / INTRO_MS);
      const p = ease(t);
      drawn.style.transform = 'scaleX(' + p + ')';
      head.style.left = p * 100 + '%';
      count.textContent = int(scoredBefore(Math.round(p * D.n_stream)));
      while (next < order.length && order[next][0] <= p) {
        const i = order[next++][1];
        spikes[i].style.height = hPct(N.ctx[i]) + '%';
      }
      if (t < 1) { raf = requestAnimationFrame(frame); return; }
      head.classList.remove('on');
      phase = 'done';
      seen.disconnect();
      setTimeout(() => root.classList.remove('pre'), 220);
    };
    raf = requestAnimationFrame(frame);
  }
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting) || phase !== 'pending') return;
    seen.disconnect();
    runIntro();
  }, { threshold: 0.6 });

  // ---------- pointer on the plot: the nearest tick top ----------
  function nearest(cx, cy) {
    const r = area.getBoundingClientRect();
    let best = 0, bd = Infinity;
    N.ctx.forEach((e, i) => {
      const dx = cx - (r.left + (e[0] / D.n_stream) * r.width);
      const dy = cy - (r.bottom - (Math.abs(e[2]) / top) * r.height);
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
  function showTip(i) {
    if (hover >= 0) spikes[hover].classList.remove('hover');
    hover = i;
    spikes[i].classList.add('hover');
    const e = N.ctx[i];
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = 'Rank ' + (i + 1) + '   ' + fmt(e[2]);
    tip.append(b, '\n' + quoted(e[4]) + '\nposition ' + int(e[0]));
    const fr = $('fig').getBoundingClientRect();
    const ar = area.getBoundingClientRect();
    const x = ar.left - fr.left + (e[0] / D.n_stream) * ar.width;
    const y = ar.bottom - fr.top - (Math.abs(e[2]) / top) * ar.height;
    const w = tip.offsetWidth;
    tip.style.left = Math.max(w / 2, Math.min(fr.width - w / 2, x)) + 'px';
    tip.style.top = y + 'px';
    tip.classList.add('on');
  }
  function hideTip() {
    if (hover >= 0) spikes[hover].classList.remove('hover');
    hover = -1;
    tip.classList.remove('on');
  }
  plot.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse' || phase !== 'done') return;
    showTip(nearest(e.clientX, e.clientY));
  });
  plot.addEventListener('pointerleave', hideTip);
  plot.addEventListener('click', (e) => {
    finishIntro();
    select(nearest(e.clientX, e.clientY));
    if (e.pointerType !== 'mouse') hideTip();
  });
  plot.addEventListener('keydown', (e) => {
    let r = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') r = sel + 1;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') r = sel - 1;
    else if (e.key === 'Home') r = 0;
    else if (e.key === 'End') r = K - 1;
    else return;
    e.preventDefault();
    finishIntro();
    hideTip();
    select(r);
  });

  // ---------- any press or focus inside the picture ends the intro at once ----------
  root.addEventListener('pointerdown', finishIntro);
  root.addEventListener('focusin', finishIntro);

  // ---------- controls ----------
  $('prev').addEventListener('click', () => select(sel - 1));
  $('next').addEventListener('click', () => select(sel + 1));
  picker.addEventListener('change', () => { hideTip(); setNeuron(Number(picker.value), 'grow'); });
  $('random').addEventListener('click', () => {
    hideTip();
    const pool = randomIdx.filter((i) => D.neurons[i] !== N);
    setNeuron(pool[Math.floor(Math.random() * pool.length)], 'grow');
  });
  narrow.addEventListener('change', () => { drawXAxis(); drawYAxis(); });

  // ---------- start ----------
  drawXAxis();
  if (phase === 'pending') {
    root.classList.add('pre');
    count.textContent = '0';
    setNeuron(0, 'pending');
    seen.observe(plot);
  } else {
    drawn.style.transform = 'scaleX(1)';
    count.textContent = int(D.n_tokens_scored);
    setNeuron(0, 'now');
  }
})();
