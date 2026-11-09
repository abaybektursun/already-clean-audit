// f13: own-direction edits against random edits of the same size, per layer. Registers the mount "f13" with site.js;
// reads window.F13_DATA (f13/data.js). Two sections, the 8 control sides and the 42 sides of the 21 pairs; each has
// one chart of the change R to the shared vector and one of the loss change per token, against the layer. Each side
// is two marks, its own-direction edit and its random edit, joined by a line. Hover, tap or the arrow keys give the
// side and both edits' values.
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
    if (!m) throw new Error('f13: cannot read the side ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative' };
  };
  const median = (a) => {
    const s = a.slice().sort((x, y) => x - y);
    return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
  };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const NLAYERS = 32;

  function mount(root, D) {
    const F = WALK.fmt;
    const TAU = D.rounding.tie_margin;
    const CT = D.control;
    const PRS = D.pairs_rule_setting;

    // ---------- the sides: one own-direction and one random edit each ----------
    const bySide = new Map();
    D.edits.forEach((e) => {
      if ((e.role !== 'control' && e.role !== 'pair') || (e.kind !== 'own_direction' && e.kind !== 'random_direction')) return;
      if (!bySide.has(e.side)) bySide.set(e.side, { side: e.side, role: e.role, w: sideOf(e.side), edits: {} });
      const s = bySide.get(e.side);
      if (s.edits[e.kind] || s.role !== e.role) throw new Error('f13: side ' + e.side + ' has two ' + e.kind + ' edits');
      s.edits[e.kind] = e;
    });
    const sides = [...bySide.values()];
    sides.forEach((s) => {
      const o = s.edits.own_direction;
      const r = s.edits.random_direction;
      if (!o || !r) throw new Error('f13: side ' + s.side + ' lacks an edit');
      if (o.edit_size_c !== r.edit_size_c || o.layer !== r.layer || s.w.layer !== o.layer) throw new Error('f13: side ' + s.side + ': the two edits differ in size or layer');
      s.layer = o.layer;
      s.o = o;
      s.r = r;
      s.ratio = r.shared_vector_change / o.shared_vector_change;
    });
    const ctrl = sides.filter((s) => s.role === 'control').sort((a, b) => a.layer - b.layer);
    const prs = sides.filter((s) => s.role === 'pair').sort((a, b) => a.layer - b.layer || a.o.shared_vector_change - b.o.shared_vector_change);
    const NP = PRS.pairs.recomputed;
    if (ctrl.length !== CT.of_layers) throw new Error('f13: ' + ctrl.length + ' control sides, the data says ' + CT.of_layers);
    if (prs.length !== 2 * NP) throw new Error('f13: ' + prs.length + ' pair sides for ' + NP + ' pairs');
    ctrl.forEach((s, i) => {
      const c = CT.layers[i];
      if (c.layer !== s.layer || c.side !== s.side) throw new Error('f13: control layer ' + c.layer + ' does not match its edits');
      if (Math.abs(c.shared_vector_ratio_random_over_own_unrounded - s.ratio) > 1e-12) throw new Error('f13: control layer ' + c.layer + ': the ratio does not follow from R');
      s.ownMore = s.o.loss_change_per_token > s.r.loss_change_per_token + TAU;
      s.randMore = s.r.loss_change_per_token > s.o.loss_change_per_token + TAU;
      if (s.ownMore !== c.own_cost_more_than_random || s.randMore !== c.random_cost_more_than_own) throw new Error('f13: control layer ' + c.layer + ': which edit cost more does not follow from the loss changes');
    });
    const nOwn = ctrl.filter((s) => s.ownMore).length;
    const nRand = ctrl.filter((s) => s.randMore).length;
    if (nOwn !== CT.own_cost_more_count.recomputed || nRand !== CT.random_cost_more_count.recomputed) throw new Error('f13: the counts of which edit cost more do not match the data');
    const ratios = ctrl.map((s) => s.ratio);
    const rLo = Math.min(...ratios);
    const rHi = Math.max(...ratios);
    if (rLo !== CT.ratio_range[0] || rHi !== CT.ratio_range[1]) throw new Error('f13: the ratio range does not match the data');
    const medOwn = median(prs.map((s) => s.o.shared_vector_change));
    const medRand = median(prs.map((s) => s.r.shared_vector_change));
    if (Math.abs(medOwn - PRS.median_shared_vector_change_own.recomputed) > 1e-9 || Math.abs(medRand - PRS.median_shared_vector_change_random.recomputed) > 1e-9) throw new Error('f13: the pair medians do not match the data');
    prs.forEach((s) => {
      if (s.o.pair === null || s.o.member === null) throw new Error('f13: pair side ' + s.side + ' has no pair');
      s.mate = prs.find((t) => t.o.pair === s.o.pair && t !== s);
      if (!s.mate) throw new Error('f13: pair side ' + s.side + ' has no partner');
    });
    const pRatios = prs.map((s) => s.ratio);
    const PW = D.power_check;
    const dAbs = PRS.pair_difference_own.map((p) => Math.abs(p.difference_own));
    if (Math.abs(median(dAbs) - PW.median_abs_pair_difference.recomputed) > 1e-18) throw new Error('f13: the median |pair difference| does not match the data');

    // ---------- formatting ----------
    const sig = (x, n) => {
      const a = Math.abs(x);
      if (a === 0) return '0';
      const d = Math.max(0, n - 1 - Math.floor(Math.log10(a)));
      return F.fixed(x, d);
    };
    const sci = (x, n) => {
      if (x === 0) return '0';
      const p = Math.floor(Math.log10(Math.abs(x)));
      if (p >= -4 && p < 6) return sig(x, n);
      return F.fixed(x / Math.pow(10, p), n - 1) + '\u00a0×\u00a010' + String(p).split('').map((ch) => SUP[ch]).join('');
    };
    const big = (x) => (Math.abs(x) >= 100 ? F.fixed(x, 1) : sig(x, 3));
    const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
    const sideText = (w) => 'layer ' + w.layer + ', unit ' + F.int(w.unit) + ', ' + w.side + ' side';

    // ---------- the frame ----------
    root.replaceChildren();
    const wrap = el('div', 'f13-wrap');
    root.appendChild(wrap);
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f13-key ' + cls), text); return s; };
    const legend = el('p', 'f13-legend');
    legend.append(key('own', 'own-direction edit, δ = −(c/‖w‖) · w'), key('rand', 'random edit, δ = c · r, the same c'), key('link', 'the two edits of one side'));
    wrap.appendChild(legend);

    const charts = [];
    function section(title, intro) {
      const s = el('section', 'f13-sec');
      s.appendChild(el('h3', 'f13-h', title));
      intro.forEach((t) => s.appendChild(el('p', 'f13-p', t)));
      wrap.appendChild(s);
      return s;
    }
    function chartBox(sec, title, spec) {
      sec.appendChild(el('p', 'f13-ct', title));
      const box = el('div', 'f13-chart');
      box.tabIndex = 0;
      box.setAttribute('role', 'group');
      box.setAttribute('aria-label', title + '; the arrow keys move between sides');
      box.setAttribute('aria-describedby', 'walk-value-box');
      sec.appendChild(box);
      sec.appendChild(el('p', 'f13-xname', 'layer'));
      const c = Object.assign({ box, title, at: -1 }, spec);
      charts.push(c);
      wire(c);
      return c;
    }

    const A = section('The ' + ctrl.length + ' control sides: in each of layers ' + ctrl.map((s) => s.layer).join(', ') + ', the side with the largest E⁺ (the sum of its squared own-sign activations over the evaluation text), its write removed fully', [
      'The random edit moved the shared vector ' + F.fixed(rLo, 2) + ' (layer ' + ctrl[ratios.indexOf(rLo)].layer + ') to ' + F.fixed(rHi, 2) + ' (layer ' +
        ctrl[ratios.indexOf(rHi)].layer + ') times as far as the own-direction edit.',
      'The random edit cost more in ' + nRand + ' of ' + ctrl.length + ' layers; the own-direction edit cost more in ' + nOwn + ' (layers ' +
        ctrl.filter((s) => s.ownMore).map((s) => s.layer).join(' and ') + '), each by more than the tie margin of ' + sci(TAU, 3) + ' nats per token. The control needed the own-direction edit to cost more in at least ' + CT.needed + '.',
    ]);
    const val = {
      R: { o: (s) => s.o.shared_vector_change, r: (s) => s.r.shared_vector_change },
      L: { o: (s) => s.o.loss_change_per_token, r: (s) => s.r.loss_change_per_token },
    };
    chartBox(A, 'The change to the shared vector after the layer, R (log scale)', { set: ctrl, v: val.R, scale: 'log' });
    chartBox(A, 'The loss change per token, in nats (log scale)', { set: ctrl, v: val.L, scale: 'log' });

    const B = section('The ' + NP + ' pairs: ' + prs.length + ' sides in ' + new Set(prs.map((s) => s.layer)).size + ' layers, the four edits of a pair at one size c', [
      'The two edits moved the shared vector alike: the median over the ' + prs.length + ' own-direction edits is ' + F.fixed(medOwn, 1) + ', over the ' + prs.length +
        ' random edits ' + F.fixed(medRand, 1) + '; on one side, random ÷ own is ' + F.fixed(Math.min(...pRatios), 3) + ' to ' + F.fixed(Math.max(...pRatios), 3) + '.',
      'A pair’s loss result is the early side’s own-direction loss change minus the late side’s. Its resampling spread over the windows, median over the ' + NP +
        ' pairs ' + sci(PW.window_spread_median.recomputed, 3) + ' nats per token, had to be at most half the median |difference|, ' + sci(PW.half_median_abs_pair_difference, 3) +
        '; it was ' + F.fixed(PW.ratio.recomputed, 3) + ' times the median |difference|, against at most ' + F.fixed(PW.ratio_needed_at_most, 1) + '.',
    ]);
    if (PW.passed !== false || !(PW.ratio.recomputed > PW.ratio_needed_at_most)) throw new Error('f13: the power check sentence states that it was not met');
    chartBox(B, 'The change to the shared vector after the layer, R (log scale)', { set: prs, v: val.R, scale: 'log' });
    chartBox(B, 'The loss change per token, in 10⁻⁶ nats', { set: prs, v: val.L, scale: 'lin', unit: 1e-6 });

    // ---------- drawing ----------
    const GL = 54;
    const GR = 8;
    const HGT = 210;
    const PT = 10;
    const PB = 24;
    const OFF = 3.5;
    const logTicks = (lo, hi) => {
      const sets = [[1], [1, 3], [1, 2, 5], [1, 2, 3, 5]];
      let first = null;
      for (const set of sets) {
        const t = [];
        for (let p = Math.floor(Math.log10(lo)) - 1; p <= Math.ceil(Math.log10(hi)); p++) {
          set.forEach((m) => { const v = m * Math.pow(10, p); if (v >= lo && v <= hi) t.push(v); });
        }
        if (t.length >= 3 && t.length <= 6) return t;
        if (t.length >= 3 && !first) first = t;
      }
      if (first) return first;
      throw new Error('f13: no ticks between ' + lo + ' and ' + hi);
    };
    const tickText = (v) => (v >= 1 ? F.int(v) : String(Number(v.toPrecision(2))));

    function draw(c) {
      const W = Math.max(300, Math.floor(c.box.clientWidth));
      c.box.replaceChildren();
      c.box.style.height = HGT + 'px';
      const s = svgEl(c.box, 'svg', { width: W, height: HGT, viewBox: '0 0 ' + W + ' ' + HGT, 'aria-hidden': 'true' });
      const colW = (W - GL - GR) / NLAYERS;
      const off = Math.max(OFF, Math.min(6, colW * 0.22));
      const xL = (l) => GL + (l + 0.5) * colW;
      const vals = c.set.flatMap((d) => [c.v.o(d), c.v.r(d)]);
      let y;
      if (c.scale === 'log') {
        if (!vals.every((v) => v > 0)) throw new Error('f13: a value on a log scale is not positive');
        const lo = Math.min(...vals) / 1.25;
        const hi = Math.max(...vals) * 1.25;
        y = (v) => PT + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (HGT - PT - PB);
        logTicks(lo, hi).forEach((v) => {
          svgEl(s, 'line', { x1: GL, x2: W - GR, y1: y(v), y2: y(v), class: 'f13-grid' });
          const t = svgEl(s, 'text', { x: GL - 6, y: y(v) + 4, class: 'f13-tick', 'text-anchor': 'end' });
          t.textContent = tickText(v);
        });
      } else {
        const u = c.unit;
        const lo = Math.floor(Math.min(0, ...vals) / u - 0.3);
        const hi = Math.ceil(Math.max(0, ...vals) / u + 0.3);
        y = (v) => PT + (1 - (v / u - lo) / (hi - lo)) * (HGT - PT - PB);
        const step = hi - lo > 6 ? 2 : 1;
        for (let k = Math.ceil(lo / step) * step; k <= hi; k += step) {
          svgEl(s, 'line', { x1: GL, x2: W - GR, y1: y(k * u), y2: y(k * u), class: k === 0 ? 'f13-zero' : 'f13-grid' });
          const t = svgEl(s, 'text', { x: GL - 6, y: y(k * u) + 4, class: 'f13-tick', 'text-anchor': 'end' });
          t.textContent = F.int(k);
        }
      }
      svgEl(s, 'line', { x1: GL, x2: W - GR, y1: HGT - PB, y2: HGT - PB, class: 'f13-axis' });
      for (let l = 0; l < NLAYERS; l += 4) {
        const t = svgEl(s, 'text', { x: xL(l), y: HGT - 6, class: 'f13-tick', 'text-anchor': 'middle' });
        t.textContent = String(l);
      }
      c.pts = c.set.map((d) => {
        const xo = xL(d.layer) - off;
        const xr = xL(d.layer) + off;
        const yo = y(c.v.o(d));
        const yr = y(c.v.r(d));
        svgEl(s, 'line', { x1: xo, x2: xr, y1: yo, y2: yr, class: 'f13-link' });
        return { d, xo, yo, xr, yr };
      });
      c.marks = c.pts.map((p) => {
        const r = svgEl(s, 'rect', { x: p.xr - 3.5, y: p.yr - 3.5, width: 7, height: 7, class: 'f13-rand' });
        const o = svgEl(s, 'circle', { cx: p.xo, cy: p.yo, r: 4, class: 'f13-own' });
        return [o, r];
      });
      if (c.at >= 0) mark(c, c.at);
    }

    function boxFor(d) {
      const rows = [['own-direction edit: R', big(d.o.shared_vector_change), 'claim'], ['random edit: R', big(d.r.shared_vector_change), 'control'],
        ['random ÷ own, R', F.fixed(d.ratio, 3)],
        ['own-direction edit: loss change', sci(d.o.loss_change_per_token, 3) + ' nats', 'claim'], ['random edit: loss change', sci(d.r.loss_change_per_token, 3) + ' nats', 'control'],
        ['the edit size c', F.fixed(d.o.edit_size_c, 4)]];
      if (d.role === 'control') return { title: cap(sideText(d.w)) + ', the control side of its layer', rows };
      const m = d.mate;
      return {
        title: cap(sideText(d.w)) + ', the ' + d.o.member + ' side of a pair, born at snapshot ' + d.o.side_birth_snapshot,
        rows: rows.concat([['its pair', 'unit ' + F.int(m.w.unit) + ', ' + m.w.side + ' side, born at snapshot ' + m.o.side_birth_snapshot]]),
      };
    }
    function mark(c, i) {
      c.marks.forEach((m, k) => m.forEach((e) => e.classList.toggle('cur', k === i)));
    }
    function clear(c) {
      c.at = -1;
      if (c.marks) c.marks.forEach((m) => m.forEach((e) => e.classList.remove('cur')));
      WALK.valueBox.hide();
    }
    function nearest(c, e) {
      const b = c.box.getBoundingClientRect();
      const x = e.clientX - b.left;
      const y = e.clientY - b.top;
      let best = -1;
      let bd = 28;
      c.pts.forEach((p, i) => {
        const dd = Math.min(Math.hypot(p.xo - x, p.yo - y), Math.hypot(p.xr - x, p.yr - y));
        if (dd < bd) { bd = dd; best = i; }
      });
      return best;
    }
    function wire(c) {
      c.box.addEventListener('pointermove', (e) => {
        if (e.pointerType !== 'mouse') return;
        const i = nearest(c, e);
        if (i < 0) { clear(c); return; }
        c.at = i;
        mark(c, i);
        WALK.valueBox.show(boxFor(c.set[i]), e.clientX, e.clientY);
      });
      c.box.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') clear(c); });
      c.box.addEventListener('pointerdown', (e) => {
        if (e.pointerType === 'mouse') return;
        const i = nearest(c, e);
        if (i < 0) return;
        c.at = i;
        mark(c, i);
        WALK.valueBox.showFor(c.marks[i][0], boxFor(c.set[i]));
      });
      c.box.addEventListener('keydown', (e) => {
        const n = c.set.length;
        const move = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -n, End: n }[e.key];
        if (move === undefined) return;
        e.preventDefault();
        const from = c.at >= 0 ? c.at : move > 0 ? -1 : n;
        c.at = Math.max(0, Math.min(n - 1, from + move));
        mark(c, c.at);
        WALK.valueBox.showFor(c.marks[c.at][0], boxFor(c.set[c.at]));
      });
      c.box.addEventListener('blur', () => clear(c));
    }

    let lastW = -1;
    const all = () => { lastW = wrap.clientWidth; charts.forEach(draw); };
    new ResizeObserver(() => { if (wrap.clientWidth !== lastW) all(); }).observe(wrap);
    all();
  }

  WALK.register('f13', { data: 'F13_DATA', mount });
})();
