// m10: the relative weight update of one late-onset cluster's neuron. Registers the mount "m10" with site.js; reads
// window.M10_DATA (m10/data.js). For one cluster: the neuron's r over each of the fifteen intervals between the sixteen
// checkpoints against its layer's median and spread, then z, and e computed from the z values shown.
// The data holds each neuron's r and z, not the layer's median and spread. For each cluster it holds two neurons of one
// layer (the cluster's neuron and its matched neuron), so at each interval the median m and spread s follow from z's
// definition, z = (ln r - ln m) / s, applied to both neurons. They are checked against the range over the 32 layers that the data
// carries separately, and the page says how they were found.
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
  const MINUS = '−';
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const svgEl = (parent, tag, attrs) => {
    const e = document.createElementNS(SVGNS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    parent.appendChild(e);
    return e;
  };
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m10: cannot read the pole ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative' };
  };
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  function mount(root, D) {
    const F = WALK.fmt;
    const steps = D.snapshot_steps;
    const J = D.step_index;
    const NJ = J.length;
    if (steps.length !== 16 || NJ !== 15 || J.some((j, i) => j !== i + 1)) throw new Error('m10: expected 16 checkpoints and intervals 1 to 15');
    const SP = D.layer_spread_over_32_layers.input_rows;
    if (SP.length !== NJ || SP.some((s, i) => s.step !== i + 1)) throw new Error('m10: the spread over the 32 layers does not cover intervals 1 to 15');

    // ---------- the numbers for one cluster ----------
    function derive(b) {
      const c = b.carrier.input_rows;
      const m = b.comparison_unit.input_rows;
      const cw = sideOf(b.carrier.side);
      const mw = sideOf(b.comparison_unit.side);
      if (cw.layer !== b.layer || mw.layer !== b.layer) throw new Error('m10: ' + b.birth + ': the two neurons are not in the cluster’s layer');
      if (c.r.length !== NJ || c.z.length !== NJ || m.r.length !== NJ || m.z.length !== NJ) throw new Error('m10: ' + b.birth + ' lacks an interval');
      const s = [];
      const med = [];
      for (let i = 0; i < NJ; i++) {
        const dz = c.z[i] - m.z[i];
        if (!(Math.abs(dz) > 1e-3)) throw new Error('m10: ' + b.birth + ' interval ' + (i + 1) + ': the two neurons have the same z; the layer’s spread cannot be found');
        const si = (Math.log(c.r[i]) - Math.log(m.r[i])) / dz;
        const lo = SP[i].one_z_in_log_r_min_over_layers;
        const hi = SP[i].one_z_in_log_r_max_over_layers;
        if (!(si >= lo * (1 - 1e-3) && si <= hi * (1 + 1e-3))) {
          throw new Error('m10: ' + b.birth + ' interval ' + (i + 1) + ': the spread found, ' + si + ', is outside the range over the 32 layers [' + lo + ', ' + hi + ']');
        }
        s.push(si);
        med.push(Math.exp(Math.log(c.r[i]) - si * c.z[i]));
      }
      const k = b.birth_snapshot;
      const win = b.rise_window_steps;
      const ref = b.reference_steps;
      if (win.join() !== [k - 2, k - 1, k].join()) throw new Error('m10: ' + b.birth + ': the three intervals are not k − 2, k − 1, k');
      if (ref.join() !== Array.from({ length: k - 4 }, (_, i) => i + 2).join()) throw new Error('m10: ' + b.birth + ': the earlier intervals are not 2 to k − 3');
      const zw = win.map((j) => c.z[j - 1]);
      const zr = ref.map((j) => c.z[j - 1]);
      const e = mean(zw) - mean(zr);
      if (Math.abs(e - c.e.banked) > 2e-5) throw new Error('m10: ' + b.birth + ': e from the shown z is ' + e + ', the data ' + c.e.banked);
      return { b, c, m, cw, mw, s, med, k, win, ref, zw, zr, meanWin: mean(zw), meanRef: mean(zr), e };
    }
    const all = D.births.map(derive);
    let cur = Math.max(0, D.births.findIndex((b) => b.birth === D.default_birth));
    let X = all[cur];

    // ---------- formatting ----------
    const sig = (x, n) => {
      const a = Math.abs(x);
      if (a === 0) return '0';
      const d = Math.max(0, n - 1 - Math.floor(Math.log10(a)));
      return F.fixed(x, d);
    };
    const z3 = (v) => F.fixed(v, 3);
    const sumText = (vals) => vals.map((v, i) => (i === 0 ? z3(v) : (v < 0 ? ' ' + MINUS + ' ' : ' + ') + F.fixed(Math.abs(v), 3))).join('');
    const sideText = (w) => 'L' + w.layer + '.N' + w.unit + (w.side === 'positive' ? '+' : '-') + ' (layer ' + w.layer + ', neuron ' + F.int(w.unit) + ', ' + w.side + ' pole)';
    const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
    // an interval j is the change from the checkpoint at steps[j - 1] to the checkpoint at steps[j]; it is named by its end
    const ends = (list) => (list.length === 1 ? 'ending at step ' + F.int(steps[list[0]]) : 'ending at steps ' + list.map((j) => F.int(steps[j])).join(', '));
    const span = (list) => (list.length === 1 ? 'ending at step ' + F.int(steps[list[0]]) : 'ending at steps ' + F.int(steps[list[0]]) + ' to ' + F.int(steps[list[list.length - 1]]));
    const roleOf = (j) => (X.win.includes(j) ? 'one of the three intervals ending at the onset' : X.ref.includes(j) ? 'an earlier interval' : 'not used (it starts at the random initialization)');
    // a compact training-step label for the axis: 0, 1k, ..., 560k, 1M, 1.41M
    const stepLabel = (s) => (s === 0 ? '0' : s >= 1e6 ? Number((s / 1e6).toFixed(2)) + 'M' : Number((s / 1e3).toFixed(1)) + 'k');

    // ---------- the frame ----------
    root.replaceChildren();
    const wrap = el('div', 'm10-wrap');
    root.appendChild(wrap);

    const pickRow = el('div', 'control-row');
    const pickLab = el('label', 'control-label', 'Neuron');
    pickLab.htmlFor = 'm10-pick';
    const pick = el('select', 'm10-select');
    pick.id = 'm10-pick';
    all.forEach((x, i) => {
      const o = el('option', '', 'Layer ' + x.cw.layer + ', neuron ' + F.int(x.cw.unit) + ', ' + x.cw.side + ' pole');
      o.value = String(i);
      pick.appendChild(o);
    });
    pickRow.append(pickLab, pick);
    wrap.appendChild(pickRow);

    const who = el('p', 'm10-who');
    wrap.appendChild(who);

    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm10-key ' + cls), text); return s; };
    const legend = el('p', 'm10-legend');
    legend.append(key('carrier', 'this neuron'), key('median', 'the layer’s median'), key('spread', 'the layer’s median ± one spread'),
      key('win', 'the three intervals ending at the onset'), key('mean', 'mean z over a span of intervals'));
    wrap.appendChild(legend);

    wrap.appendChild(el('h3', 'm10-h', 'r: the relative weight update of the neuron’s gate_proj row and up_proj row (8,192 numbers) from one checkpoint to the next'));
    const rBox = el('div', 'm10-chart');
    wrap.appendChild(rBox);
    wrap.appendChild(el('h3', 'm10-h m10-h2', 'z: r z-scored within the layer over the same interval, in units of the layer’s spread'));
    const zBox = el('div', 'm10-chart');
    wrap.appendChild(zBox);
    const scrubWrap = el('div', 'm10-scrubwrap');
    const scrub = el('input', 'scrub');
    scrub.type = 'range';
    scrub.min = '1';
    scrub.max = String(NJ);
    scrub.step = '1';
    scrub.setAttribute('aria-label', 'Checkpoint');
    scrubWrap.appendChild(scrub);
    const xName = el('p', 'm10-xname', 'training step of the checkpoint (k = thousand, M = million); each column is the change from the checkpoint before. Drag, or tap a column.');
    scrubWrap.appendChild(xName);
    wrap.appendChild(scrubWrap);

    const out = el('div', 'm10-out');
    out.setAttribute('aria-live', 'polite');
    wrap.appendChild(out);
    const eBox = el('div', 'm10-e');
    wrap.appendChild(eBox);
    const note = el('p', 'm10-note');
    wrap.appendChild(note);

    // ---------- the charts ----------
    const GL = 46; // left gutter for the y ticks
    const GR = 10;
    const geom = (box) => {
      const W = Math.max(280, Math.floor(box.clientWidth));
      const colW = (W - GL - GR) / NJ;
      return { W, colW, x: (j) => GL + (j - 0.5) * colW };
    };
    const logTicks = (lo, hi) => {
      for (const set of [[1, 2, 5], [1, 2, 3, 5], [1, 1.5, 2, 3, 4, 5, 6, 8], [1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9]]) {
        const t = [];
        for (let p = Math.floor(Math.log10(lo)) - 1; p <= Math.ceil(Math.log10(hi)); p++) {
          set.forEach((m) => { const v = m * Math.pow(10, p); if (v >= lo && v <= hi) t.push(v); });
        }
        if (t.length >= 3) return t;
      }
      throw new Error('m10: no ticks between ' + lo + ' and ' + hi);
    };
    const tickText = (v) => String(Number(v.toPrecision(2)));
    const linTicks = (lo, hi) => {
      const span = hi - lo;
      const step = span > 12 ? 4 : span > 6 ? 2 : span > 2.5 ? 1 : 0.5;
      const t = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(Math.round(v * 100) / 100);
      return t;
    };

    function frame(box, H, g) {
      box.replaceChildren();
      box.style.height = H + 'px';
      const s = svgEl(box, 'svg', { width: g.W, height: H, viewBox: '0 0 ' + g.W + ' ' + H, 'aria-hidden': 'true' });
      return s;
    }
    function shade(s, g, top, bottom) {
      const x0 = g.x(X.win[0]) - g.colW / 2;
      svgEl(s, 'rect', { x: x0, y: top, width: 3 * g.colW, height: bottom - top, class: 'm10-win' });
    }
    function xTicks(s, g, y) {
      // every checkpoint's step where the columns are wide enough; on a narrow screen every other one
      J.forEach((j) => {
        if (g.colW < 32 && j % 2 === 0) return;
        const t = svgEl(s, 'text', { x: g.x(j), y, class: 'm10-tick', 'text-anchor': 'middle' });
        t.textContent = stepLabel(steps[j]);
      });
    }
    let curRects = [];
    let dots = { r: [], z: [] };

    const RH = 230;
    const ZH = 240;
    const PT = 10; // plot top
    const PB = 24; // room for the x ticks
    function drawR() {
      const g = geom(rBox);
      const s = frame(rBox, RH, g);
      const lows = X.med.map((m, i) => m * Math.exp(-X.s[i]));
      const highs = X.med.map((m, i) => m * Math.exp(X.s[i]));
      const lo = Math.min(...X.c.r, ...lows) / 1.12;
      const hi = Math.max(...X.c.r, ...highs) * 1.12;
      const y = (v) => PT + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (RH - PT - PB);
      shade(s, g, PT, RH - PB);
      logTicks(lo, hi).forEach((v) => {
        svgEl(s, 'line', { x1: GL, x2: g.W - GR, y1: y(v), y2: y(v), class: 'm10-grid' });
        const t = svgEl(s, 'text', { x: GL - 6, y: y(v) + 4, class: 'm10-tick', 'text-anchor': 'end' });
        t.textContent = tickText(v);
      });
      svgEl(s, 'line', { x1: GL, x2: g.W - GR, y1: RH - PB, y2: RH - PB, class: 'm10-axis' });
      const area = J.map((j, i) => g.x(j) + ' ' + y(highs[i])).join(' L') + ' L' + J.slice().reverse().map((j) => g.x(j) + ' ' + y(lows[j - 1])).join(' L');
      svgEl(s, 'path', { d: 'M' + area + ' Z', class: 'm10-bandarea' });
      svgEl(s, 'path', { d: 'M' + J.map((j, i) => g.x(j) + ' ' + y(X.med[i])).join(' L'), class: 'm10-medline' });
      svgEl(s, 'path', { d: 'M' + J.map((j, i) => g.x(j) + ' ' + y(X.c.r[i])).join(' L'), class: 'm10-rline' });
      dots.r = J.map((j, i) => svgEl(s, 'circle', { cx: g.x(j), cy: y(X.c.r[i]), r: 4, class: 'm10-dot' }));
      xTicks(s, g, RH - 6);
      return { s, g, H: RH };
    }

    function drawZ() {
      const g = geom(zBox);
      const s = frame(zBox, ZH, g);
      const vals = X.c.z.concat([-1, 1, X.meanWin, X.meanRef]);
      const pad = 0.6;
      const lo = Math.min(...vals) - pad;
      const hi = Math.max(...vals) + pad;
      const y = (v) => PT + (1 - (v - lo) / (hi - lo)) * (ZH - PT - PB);
      shade(s, g, PT, ZH - PB);
      linTicks(lo, hi).forEach((v) => {
        svgEl(s, 'line', { x1: GL, x2: g.W - GR, y1: y(v), y2: y(v), class: 'm10-grid' });
        const t = svgEl(s, 'text', { x: GL - 6, y: y(v) + 4, class: 'm10-tick', 'text-anchor': 'end' });
        t.textContent = F.fixed(v, Number.isInteger(v) ? 0 : 1);
      });
      svgEl(s, 'line', { x1: GL, x2: g.W - GR, y1: ZH - PB, y2: ZH - PB, class: 'm10-axis' });
      svgEl(s, 'rect', { x: GL, y: y(1), width: g.W - GL - GR, height: y(-1) - y(1), class: 'm10-bandarea' });
      svgEl(s, 'line', { x1: GL, x2: g.W - GR, y1: y(0), y2: y(0), class: 'm10-medline' });
      // the two means and the step between them, which is e
      const xr0 = g.x(X.ref[0]) - g.colW / 2;
      const xr1 = g.x(X.ref[X.ref.length - 1]) + g.colW / 2;
      const xw1 = g.x(X.win[2]) + g.colW / 2;
      svgEl(s, 'line', { x1: xr0, x2: xr1, y1: y(X.meanRef), y2: y(X.meanRef), class: 'm10-mean' });
      svgEl(s, 'line', { x1: xr1, x2: xw1, y1: y(X.meanWin), y2: y(X.meanWin), class: 'm10-mean' });
      svgEl(s, 'line', { x1: xr1, x2: xr1, y1: y(X.meanRef), y2: y(X.meanWin), class: 'm10-ebar' });
      dots.z = J.map((j, i) => svgEl(s, 'circle', { cx: g.x(j), cy: y(X.c.z[i]), r: 4, class: 'm10-dot' }));
      const lab = svgEl(s, 'text', { x: xr1 - 6, y: (y(X.meanRef) + y(X.meanWin)) / 2 + 5, class: 'm10-elab', 'text-anchor': 'end' });
      lab.textContent = 'e = ' + z3(X.e);
      xTicks(s, g, ZH - 6);
      return { s, g, H: ZH };
    }

    function hoverCols(c) {
      J.forEach((j) => {
        const r = svgEl(c.s, 'rect', { x: c.g.x(j) - c.g.colW / 2, y: 0, width: c.g.colW, height: c.H, class: 'm10-col', tabindex: '-1' });
        WALK.hover(r, () => stepBox(j));
        r.addEventListener('click', () => { scrub.value = String(j); update(); });
      });
      const k = svgEl(c.s, 'rect', { x: 0, y: PT - 4, width: c.g.colW, height: c.H - PT - PB + 8, rx: 4, class: 'm10-cur' });
      curRects.push([k, c.g]);
    }

    function stepBox(j) {
      const i = j - 1;
      return {
        title: 'Checkpoint at step ' + F.int(steps[j]) + ' (the change from step ' + F.int(steps[j - 1]) + ')',
        rows: [['r', sig(X.c.r[i], 4), 'claim'], ['the layer’s median', sig(X.med[i], 4), 'control'], ['the layer’s spread s', sig(X.s[i], 3)],
          ['z', z3(X.c.z[i]), 'claim'], ['used for e', X.win.includes(j) ? 'the three intervals' : X.ref.includes(j) ? 'an earlier interval' : 'not used']],
      };
    }

    let lastW = 0;
    function draw() {
      curRects = [];
      const rc = drawR();
      const zc = drawZ();
      hoverCols(rc);
      hoverCols(zc);
      const g = zc.g;
      scrub.style.width = (g.W - GL - GR) * (NJ - 1) / NJ + 22 + 'px';
      scrub.style.marginLeft = GL + g.colW / 2 - 11 + 'px';
      xName.style.marginLeft = GL + 'px';
      lastW = rBox.clientWidth;
      mark();
    }

    function mark() {
      const j = Number(scrub.value);
      curRects.forEach(([r, g]) => r.setAttribute('x', g.x(j) - g.colW / 2));
      dots.r.concat(dots.z).forEach((d) => d.classList.remove('cur'));
      dots.r[j - 1].classList.add('cur');
      dots.z[j - 1].classList.add('cur');
    }

    // ---------- the text ----------
    function drawText() {
      who.replaceChildren();
      who.append(el('b', '', cap(sideText(X.cw))), '. Its cluster “' + X.b.group_name + '”, a group of the pole’s top-50 activating tokens that an LLM named, ' +
        'has its onset at the checkpoint at step ' + F.int(steps[X.k]) + ' (stage1-step' + steps[X.k] + '): the first checkpoint at which the pole’s median activation on the cluster’s tokens is in the pole’s own top 0.1 % and stays there. ' +
        'The three intervals ending at the onset are those ' + ends(X.win) + '.');

      eBox.replaceChildren();
      eBox.appendChild(el('p', 'm10-outhead', 'e for this neuron, from the z values above'));
      const tab = el('table', 'm10-tab');
      const row = (a, b2) => { const r = el('tr'); r.append(el('th', '', a), el('td', 'm10-num', b2)); tab.appendChild(r); };
      row('mean z over the three intervals ending at the onset (' + ends(X.win) + ')', '(' + sumText(X.zw) + ') / 3 = ' + z3(X.meanWin));
      row('mean z over the ' + X.ref.length + ' earlier intervals (' + span(X.ref) + ')', '(' + sumText(X.zr) + ') / ' + X.ref.length + ' = ' + z3(X.meanRef));
      row('e = the first mean minus the second', z3(X.meanWin) + ' ' + MINUS + ' ' + (X.meanRef < 0 ? '(' + z3(X.meanRef) + ')' : z3(X.meanRef)) + ' = ' + z3(X.e));
      row('e in the data file', z3(X.c.e.banked));
      eBox.appendChild(tab);
      eBox.appendChild(el('p', 'm10-note', 'The first interval, from the random initialization at step 0 to step 1,000, is not used. Below zero, the neuron’s weights changed less, relative to its layer, ' +
        'in the three intervals ending at the onset than in the earlier intervals.'));

      note.textContent = 'How we found the layer’s median and spread. The spread s is 1.4826 times the median absolute deviation of ln r over the ' +
        'layer’s neurons over that interval, and z = (ln r − ln m) / s, with m the layer’s median. The data file holds each neuron’s r and z but not m and s. For this cluster it holds ' +
        'two neurons of layer ' + X.b.layer + ': the cluster’s neuron and its matched neuron (neuron ' + F.int(X.mw.unit) + ' of the same layer). At each interval, m and s are the two numbers for which ' +
        'both neurons’ z hold: s = (ln r₁ − ln r₂) / (z₁ − z₂) and m = r₁ · e^(−s · z₁). So the z this page computes from m and s returns the z in the data. ' +
        'We checked every interval of all ' + all.length + ' neurons: each s found this way lies inside the range from the smallest to the largest spread over the 32 layers, which the data file holds separately.';
    }

    function update() {
      const j = Number(scrub.value);
      const i = j - 1;
      scrub.setAttribute('aria-valuetext', 'checkpoint at step ' + F.int(steps[j]) + ', z ' + z3(X.c.z[i]));
      mark();
      out.replaceChildren();
      out.appendChild(el('p', 'm10-outhead', 'Checkpoint at step ' + F.int(steps[j]) + ' (stage1-step' + steps[j] + '): the change from step ' + F.int(steps[j - 1])));
      const tab = el('table', 'm10-tab');
      const row = (a, b2) => { const r = el('tr'); r.append(el('th', '', a), el('td', 'm10-num', b2)); tab.appendChild(r); };
      row('r, this neuron', sig(X.c.r[i], 4));
      row('m, the layer’s median r', sig(X.med[i], 4));
      row('s, the layer’s spread of ln r', sig(X.s[i], 3));
      row('z = (ln r − ln m) / s', '(ln ' + sig(X.c.r[i], 4) + ' − ln ' + sig(X.med[i], 4) + ') / ' + sig(X.s[i], 3) + ' = ' + z3((Math.log(X.c.r[i]) - Math.log(X.med[i])) / X.s[i]));
      row('used for e as', roleOf(j));
      out.appendChild(tab);
    }

    function setBirth(i) {
      cur = i;
      X = all[i];
      pick.value = String(i);
      scrub.value = String(X.k);
      drawText();
      draw();
      update();
    }
    pick.addEventListener('change', () => setBirth(Number(pick.value)));
    scrub.addEventListener('input', update);
    new ResizeObserver(() => { if (rBox.clientWidth !== lastW) { draw(); } }).observe(wrap);
    setBirth(cur);
  }

  WALK.register('m10', { data: 'M10_DATA', mount });
})();
