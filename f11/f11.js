// f11: the carrier's weight-change statistic against a matched unit's, 67 held-out births. Registers the mount "f11"
// with site.js; reads window.F11_DATA (f11/data.js). One row per pair, sorted by the difference: on the left the two
// units' e, on the right carrier minus matched unit, with the mean of the 67 differences and its range drawn as a
// band. Hover, tap or the arrow keys give the pair and both values.
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
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
    if (!m) throw new Error('f11: cannot read the side ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative' };
  };

  function mount(root, D) {
    const F = WALK.fmt;
    const H = D.held_out_input_rows;
    const N = H.pairs.recomputed;
    const P = D.pairs.slice().sort((a, b) => b.difference_input_rows - a.difference_input_rows);
    if (P.length !== N) throw new Error('f11: ' + P.length + ' pairs, the data says ' + N);
    P.forEach((p) => {
      if (Math.abs(p.carrier_e_input_rows - p.comparison_e_input_rows - p.difference_input_rows) > 1e-12) throw new Error('f11: ' + p.birth + ': the difference is not carrier minus matched unit');
      p.c = sideOf(p.carrier_side);
      p.m = sideOf(p.comparison_side);
      if (p.c.layer !== p.layer || p.m.layer !== p.layer) throw new Error('f11: ' + p.birth + ': a unit is outside the birth’s layer');
    });
    const MEAN = H.mean_difference.recomputed;
    const LO = H.band_low.recomputed;
    const HI = H.band_high.recomputed;
    const ABOVE = H.carrier_above.recomputed;
    const BELOW = H.carrier_below.recomputed;
    const mean = P.reduce((s, p) => s + p.difference_input_rows, 0) / N;
    if (Math.abs(mean - MEAN) > 1e-12) throw new Error('f11: the mean of the drawn differences is ' + mean + ', the data ' + MEAN);
    const above = P.filter((p) => p.difference_input_rows > 0).length;
    const below = P.filter((p) => p.difference_input_rows < 0).length;
    if (above !== ABOVE || below !== BELOW) throw new Error('f11: ' + above + ' above and ' + below + ' below zero, the data ' + ABOVE + ' and ' + BELOW);
    if (!(LO < MEAN && MEAN < HI)) throw new Error('f11: the mean is outside its range');

    const f3 = (v) => F.fixed(v, 3);
    const unitText = (w) => 'unit ' + F.int(w.unit) + ', ' + w.side + ' side';

    root.replaceChildren();
    const wrap = el('div', 'f11-wrap');
    root.appendChild(wrap);

    // the three numbers of the result
    const facts = el('div', 'f11-facts');
    const fact = (big, text) => { const f = el('p', 'f11-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(f3(MEAN), 'the mean of the ' + N + ' differences, carrier minus matched unit');
    fact(f3(LO) + ' to ' + f3(HI), 'the range of that mean, drawn as the band');
    fact(ABOVE + ' and ' + BELOW, 'pairs above zero and below it');
    wrap.appendChild(facts);

    wrap.appendChild(el('p', 'f11-note', 'Each row is one held-out birth: its carrier and a matched unit of the same layer, whose side holds a random group of ' +
      'another unit’s contexts of the same size, dated to the same snapshot by the same rule and matched on the number of distinct tokens. e is the ' +
      'weight-change statistic of the gate row and up row: the mean z over the three steps ending at the birth minus the mean z over the earlier steps. ' + // vocab-ok
      'Rows are sorted by the difference.'));

    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f11-key ' + cls), text); return s; };
    const legend = el('p', 'f11-legend');
    legend.append(key('carrier', 'the carrier’s e'), key('matched', 'the matched unit’s e'), key('diff', 'carrier minus matched unit'),
      key('mean', 'mean of the ' + N + ' differences, ' + f3(MEAN)), key('band', 'its range, ' + f3(LO) + ' to ' + f3(HI)));
    wrap.appendChild(legend);

    const heads = el('div', 'f11-heads');
    const hL = el('p', 'f11-h', 'e of each unit');
    const hR = el('p', 'f11-h', 'carrier minus matched unit');
    heads.append(hL, hR);
    wrap.appendChild(heads);

    const plot = el('div', 'f11-plot');
    plot.tabIndex = 0;
    plot.setAttribute('role', 'group');
    plot.setAttribute('aria-label', N + ' pairs, one row each; the arrow keys move between pairs');
    plot.setAttribute('aria-describedby', 'walk-value-box');
    wrap.appendChild(plot);
    wrap.appendChild(el('p', 'f11-hint', 'Hover or tap a row, or focus the chart and use the up and down arrow keys.'));

    const O = D.held_out_output_column;
    const R = D.rule_setting_input_rows;
    const NU = D.natural_units_input_rows;
    const A = D.activation_size;
    const pc = (v) => F.fixed(Math.abs(100 * v), v > -0.001 && v < 0.001 ? 2 : 1);
    wrap.appendChild(el('p', 'f11-note f11-more', 'The same comparison read other ways, from the same data. For the down-matrix column instead of the input rows: mean ' +
      f3(O.mean_difference.recomputed) + ', range ' + f3(O.band_low.recomputed) + ' to ' + f3(O.band_high.recomputed) + '. On the ' + R.pairs.recomputed +
      ' pairs of the half used to set the rules: ' + f3(R.mean_difference.recomputed) + ', range ' + f3(R.band_low.recomputed) + ' to ' + f3(R.band_high.recomputed) +
      '. In r itself, over the three steps ending at the birth, the carriers’ r is ' + pc(NU.relative_change.recomputed) + ' % lower (range ' +
      pc(NU.relative_band_low.recomputed) + ' % lower to ' + pc(NU.relative_band_high.recomputed) + ' % higher). The carriers’ largest activation one snapshot ' +
      'before the birth is ' + F.fixed(A.ratio_of_max_activation, 2) + ' times the matched units’; at equal activation (a line fitted to the differences against ' +
      'the log ratio of the two activations) the mean difference is ' + f3(A.adjusted_for_activation_size.mean_difference) + ', range ' +
      f3(A.adjusted_for_activation_size.band[0]) + ' to ' + f3(A.adjusted_for_activation_size.band[1]) + '.'));
    if (!(NU.relative_change.recomputed < 0 && NU.relative_band_low.recomputed < 0 && NU.relative_band_high.recomputed > 0)) throw new Error('f11: the r change and its range do not have the signs the sentence states');

    // ---------- the chart ----------
    const GAP = 18;
    const PAD = 8;
    const TOP = 6;
    const AX = 26;
    let rowH = 8;
    let g = null;
    let at = -1;

    const ticks = (lo, hi, w) => {
      const step = (hi - lo) / Math.max(1, Math.floor(w / 46)) > 1 ? 2 : 1;
      const t = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(v);
      return t;
    };

    function draw() {
      const W = Math.max(300, Math.floor(plot.clientWidth));
      const wL = Math.round((W - GAP) * 0.45);
      const wR = W - GAP - wL;
      heads.style.gridTemplateColumns = wL + 'px ' + wR + 'px';
      heads.style.columnGap = GAP + 'px';
      const Hh = TOP + N * rowH + AX;
      plot.replaceChildren();
      plot.style.height = Hh + 'px';
      const s = svgEl(plot, 'svg', { width: W, height: Hh, viewBox: '0 0 ' + W + ' ' + Hh, 'aria-hidden': 'true' });
      const eVals = P.flatMap((p) => [p.carrier_e_input_rows, p.comparison_e_input_rows]);
      const dVals = P.map((p) => p.difference_input_rows).concat([0, LO, HI]);
      const lL = Math.floor(Math.min(...eVals) - 0.2);
      const hL = Math.ceil(Math.max(...eVals) + 0.2);
      const lR = Math.floor(Math.min(...dVals) - 0.2);
      const hR2 = Math.ceil(Math.max(...dVals) + 0.2);
      const xl = (v) => PAD + ((v - lL) / (hL - lL)) * (wL - 2 * PAD);
      const xr = (v) => wL + GAP + PAD + ((v - lR) / (hR2 - lR)) * (wR - 2 * PAD);
      const yc = (i) => TOP + (i + 0.5) * rowH;
      const yB = TOP + N * rowH;
      const axis = (x, lo, hi, w, x0, x1) => {
        ticks(lo, hi, w).forEach((v) => {
          svgEl(s, 'line', { x1: x(v), x2: x(v), y1: TOP, y2: yB, class: 'f11-grid' });
          const t = svgEl(s, 'text', { x: x(v), y: yB + 17, class: 'f11-tick', 'text-anchor': 'middle' });
          t.textContent = F.int(v);
        });
        svgEl(s, 'line', { x1: x0, x2: x1, y1: yB, y2: yB, class: 'f11-axis' });
      };
      axis(xl, lL, hL, wL, 0, wL);
      axis(xr, lR, hR2, wR, wL + GAP, W);
      const hi = svgEl(s, 'rect', { x: 0, y: 0, width: W, height: rowH, class: 'f11-row', visibility: 'hidden' });
      // the result: band, zero, mean
      svgEl(s, 'rect', { x: xr(LO), y: TOP, width: xr(HI) - xr(LO), height: N * rowH, class: 'f11-band' });
      svgEl(s, 'line', { x1: xr(0), x2: xr(0), y1: TOP, y2: yB, class: 'f11-zero' });
      svgEl(s, 'line', { x1: xr(MEAN), x2: xr(MEAN), y1: TOP, y2: yB, class: 'f11-mean' });
      const yDiv = TOP + ABOVE * rowH;
      svgEl(s, 'line', { x1: 0, x2: W, y1: yDiv, y2: yDiv, class: 'f11-divider' });
      const marks = P.map((p, i) => {
        const y = yc(i);
        svgEl(s, 'line', { x1: xl(p.comparison_e_input_rows), x2: xl(p.carrier_e_input_rows), y1: y, y2: y, class: 'f11-link' });
        svgEl(s, 'circle', { cx: xl(p.comparison_e_input_rows), cy: y, r: 3, class: 'f11-matched' });
        const c = svgEl(s, 'circle', { cx: xl(p.carrier_e_input_rows), cy: y, r: 3.5, class: 'f11-carrier' });
        svgEl(s, 'line', { x1: xr(0), x2: xr(p.difference_input_rows), y1: y, y2: y, class: 'f11-stem' });
        const d = svgEl(s, 'circle', { cx: xr(p.difference_input_rows), cy: y, r: 3.5, class: 'f11-diff' });
        return [c, d];
      });
      const lab = (x, y, anchor, text) => { const t = svgEl(s, 'text', { x, y, class: 'f11-count', 'text-anchor': anchor }); t.textContent = text; };
      lab(wL + GAP + 2, TOP + 14, 'start', ABOVE + ' above zero');
      lab(W - 2, yB - 6, 'end', BELOW + ' below zero');
      g = { hi, marks, W, yc };
      if (at >= 0) show(at, false);
    }

    function boxFor(i) {
      const p = P[i];
      return {
        title: 'Layer ' + p.layer + ', birth at snapshot ' + p.birth_snapshot,
        rows: [['carrier, ' + unitText(p.c), f3(p.carrier_e_input_rows), 'claim'],
          ['matched unit, ' + unitText(p.m), f3(p.comparison_e_input_rows), 'control'],
          ['difference', f3(p.difference_input_rows), 'ink'],
          ['row', (i + 1) + ' of ' + N]],
      };
    }
    function show(i, box, x, y) {
      at = i;
      g.hi.setAttribute('y', g.yc(i) - rowH / 2);
      g.hi.setAttribute('visibility', 'visible');
      g.marks.forEach((m, k) => m.forEach((c) => c.classList.toggle('cur', k === i)));
      if (box === 'pointer') WALK.valueBox.show(boxFor(i), x, y);
      else if (box) WALK.valueBox.showFor(g.marks[i][1], boxFor(i));
    }
    function clear() {
      at = -1;
      g.hi.setAttribute('visibility', 'hidden');
      g.marks.forEach((m) => m.forEach((c) => c.classList.remove('cur')));
      WALK.valueBox.hide();
    }
    const rowAt = (e) => {
      const r = plot.getBoundingClientRect();
      const i = Math.floor((e.clientY - r.top - TOP) / rowH);
      return i >= 0 && i < N ? i : -1;
    };
    plot.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const i = rowAt(e);
      if (i < 0) { clear(); return; }
      show(i, 'pointer', e.clientX, e.clientY);
    });
    plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') clear(); });
    plot.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const i = rowAt(e);
      if (i >= 0) show(i, 'tap');
    });
    plot.addEventListener('keydown', (e) => {
      const move = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1, Home: -N, End: N, PageDown: 10, PageUp: -10 }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      const from = at >= 0 ? at : move > 0 ? -1 : N;
      show(Math.max(0, Math.min(N - 1, from + move)), 'key');
    });
    plot.addEventListener('blur', clear);

    let lastW = 0;
    new ResizeObserver(() => {
      const w = plot.clientWidth;
      if (w === lastW) return;
      lastW = w;
      rowH = w < 560 ? 8 : 9;
      draw();
    }).observe(wrap);
    lastW = plot.clientWidth;
    rowH = lastW < 560 ? 8 : 9;
    draw();
  }

  WALK.register('f11', { data: 'F11_DATA', mount });
})();
