// m09: the gradient prediction of switching a side off, and what 16-bit broke. Registers the mount "m09" with site.js;
// reads window.M09_DATA (m09/data.js). The formula; for the 284 held-out pairs, the 16-bit prediction against the
// measured change with one toggle between the 16-bit and the 32-bit measurement; the 16-bit prediction against the
// 32-bit prediction; the sizes of the three quantities with their medians; and the memory that forced forward mode.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (parent, tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    parent.appendChild(e);
    return e;
  };
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])(?::c(\d+))?$/.exec(id);
    if (!m) throw new Error('m09: cannot read the id ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), sign: m[3] === '+' ? 'positive' : 'negative', group: m[4] === undefined ? null : Number(m[4]) };
  };
  const ORD = ['', '1st', '2nd', '3rd'];

  // rank agreement: the correlation of the two rankings, ties sharing their mean rank
  const ranks = (x) => {
    const o = x.map((v, i) => i).sort((a, b) => x[a] - x[b]);
    const r = new Array(x.length);
    for (let s = 0; s < o.length;) {
      let e = s;
      while (e + 1 < o.length && x[o[e + 1]] === x[o[s]]) e++;
      for (let k = s; k <= e; k++) r[o[k]] = (s + e) / 2;
      s = e + 1;
    }
    return r;
  };
  const rankAgreement = (x, y) => {
    const a = ranks(x), b = ranks(y);
    const n = a.length;
    const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
    return sab / Math.sqrt(saa * sbb);
  };
  const median = (x) => {
    const s = x.slice().sort((a, b) => a - b);
    return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
  };

  // scatter axes: log scale of the size on either side of 0, linear within ±C; the order of the values is kept
  const C = 1e-4;
  const VMAX = 5;
  const f = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v) / C);
  const FMAX = f(VMAX);
  const pos = (v) => ((f(v) + FMAX) / (2 * FMAX)) * 100;
  const TICKS = [-1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1];
  const MAJOR = new Set([-1, -0.01, 0, 0.01, 1]);

  function mount(root, D) {
    const F = WALK.fmt;
    const sig = (x, n) => (x === 0 ? '0' : F.fixed(x, Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(x))))));
    const P = D.pairs;
    const N = P.half.length;
    const T16 = D.toggle_16bit_measurement;
    const T32 = D.toggle_32bit_measurement;
    const PR = D.prediction;
    const WF = D.why_forward_mode;

    // ---------- self-checks: the rows must give the numbers the caption states ----------
    ['birth', 'side', 'place_in_list', 'prediction_16bit', 'measurement_16bit', 'measurement_32bit', 'prediction_32bit'].forEach((k) => {
      if (P[k].length !== N) throw new Error('m09: pairs.' + k + ' has ' + P[k].length + ' rows, not ' + N);
    });
    const held = [], rule = [];
    P.half.forEach((h, i) => {
      if (h === 'held_out') held.push(i);
      else if (h === 'rule_setting') rule.push(i);
      else throw new Error('m09: unknown half ' + h);
    });
    if (held.length !== 284 || rule.length !== 328) throw new Error('m09: the halves hold ' + rule.length + ' and ' + held.length + ' pairs, not 328 and 284');
    const col = (k, ids) => ids.map((i) => P[k][i]);
    const agreeCheck = (name, x, y, rec) => {
      const r = rankAgreement(x, y);
      if (Math.abs(r - rec.banked) > 1e-9 || rec.recomputed !== rec.banked) throw new Error('m09: ' + name + ' is ' + r + ' from the rows, ' + rec.banked + ' banked');
      return rec.banked;
    };
    const rho = {
      16: { rule: agreeCheck('16-bit rule-setting', col('prediction_16bit', rule), col('measurement_16bit', rule), T16.rank_agreement_rule_setting),
        held: agreeCheck('16-bit held-out', col('prediction_16bit', held), col('measurement_16bit', held), T16.rank_agreement_held_out) },
      32: { rule: agreeCheck('32-bit rule-setting', col('prediction_16bit', rule), col('measurement_32bit', rule), T32.rank_agreement_rule_setting),
        held: agreeCheck('32-bit held-out', col('prediction_16bit', held), col('measurement_32bit', held), T32.rank_agreement_held_out) },
    };
    const rhoPred = agreeCheck('16-bit against 32-bit prediction', col('prediction_16bit', held), col('prediction_32bit', held), PR.rank_agreement_16bit_vs_32bit_held_out);
    const all = held.concat(rule);
    const medCheck = (name, v, rec, tol) => {
      if (Math.abs(v - rec.banked) > tol) throw new Error('m09: median ' + name + ' is ' + v + ' from the rows, ' + rec.banked + ' banked');
      return rec.banked;
    };
    const med16m = medCheck('|16-bit measurement|', median(col('measurement_16bit', all).map(Math.abs)), T16.median_abs_measurement_both_halves, 1e-12);
    const med16p = medCheck('|16-bit prediction|', median(col('prediction_16bit', all).map(Math.abs)), PR.median_abs_16bit_both_halves, 1e-9);
    const med32m = medCheck('|32-bit measurement|', median(col('measurement_32bit', held).map(Math.abs)), T32.median_abs_measurement_held_out, 1e-12);
    const relDiff = medCheck('relative difference', median(held.map((i) => Math.abs(P.prediction_16bit[i] - P.prediction_32bit[i]) / Math.max(Math.abs(P.prediction_32bit[i]), 1e-30))),
      PR.median_relative_difference_held_out, 1e-12);
    [[rhoPred, 3, '0.995'], [rho[16].rule, 3, '0.100'], [rho[16].held, 3, '0.043'], [rho[32].held, 3, '0.960'], [T32.band_held_out[0], 3, '0.915'],
      [T32.band_held_out[1], 3, '0.990'], [med16m, 2, '0.65'], [med16p, 3, '0.022'], [relDiff * 100, 1, '2.6'], [WF.backward_32bit_peak_GB_layer0, 2, '67.17'],
      [WF.gpu_memory_GB, 2, '48.31']].forEach(([v, d, s]) => {
      if (F.fixed(v, d) !== s) throw new Error('m09: the data gives ' + F.fixed(v, d) + ' where the page says ' + s);
    });
    if (!(WF.backward_32bit_peak_GB_layer0 > WF.gpu_memory_GB)) throw new Error('m09: the 32-bit backward pass fits the GPU');
    const sm = WF.small_model_check;
    if (Math.abs(Math.abs(sm.forward_mode - sm.reverse_mode) / Math.abs(sm.reverse_mode) - sm.relative_difference) > 1e-15) throw new Error('m09: the small-model difference does not follow from its two values');
    if (Math.max(...[P.prediction_16bit, P.measurement_16bit, P.measurement_32bit, P.prediction_32bit].flat().map(Math.abs)) > VMAX) throw new Error('m09: a value lies outside the axes');

    root.replaceChildren();
    const wrap = el('div', 'm09-wrap');
    root.appendChild(wrap);

    // ---------- the formula ----------
    const top = el('section', 'm09-top');
    wrap.appendChild(top);
    top.appendChild(el('h3', 'm09-h', 'The first-order prediction of switching a side off'));
    top.appendChild(el('p', 'm09-formula', 'ΔL̂ = Σ over the group’s positions i of −h(i) · ∂L/∂h(i)'));
    top.appendChild(el('p', 'm09-note', 'h(i) is the unit’s input to the down matrix at position i, and L is the window’s summed next-token loss. Switching off changes h(i) by −h(i), ' +
      'so to first order the loss changes by −h(i) times the derivative; hence the minus sign. ΔL̂ is summed over the birth’s windows, in nats, and is compared with ΔL, the switched-off loss minus the clean loss summed the same way. ' +
      'A pair is a birth and one switched-off side: the carrier, or one of the three other sides of its layer that ranked highest one snapshot before the birth.'));

    // ---------- a scatter on the signed log scale ----------
    const legendKey = (cls, text) => { const s = el('span'); s.append(el('i', 'm09-key ' + cls), text); return s; };
    function scatter(parent, ids, xs, boxFor, aria) {
      const yName = el('p', 'm09-yname');
      parent.appendChild(yName);
      const plot = el('div', 'm09-plot');
      plot.tabIndex = 0;
      plot.setAttribute('role', 'group');
      plot.setAttribute('aria-label', aria);
      parent.appendChild(plot);
      const svg = svgEl(plot, 'svg', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      TICKS.forEach((v) => {
        const p = pos(v);
        svgEl(svg, 'line', { x1: p, x2: p, y1: 0, y2: 100, class: v === 0 ? 'm09-zero' : 'm09-gl' });
        svgEl(svg, 'line', { x1: 0, x2: 100, y1: 100 - p, y2: 100 - p, class: v === 0 ? 'm09-zero' : 'm09-gl' });
        const lab = v === 0 ? '0' : sig(v, 1);
        const xt = el('span', 'm09-xtick' + (MAJOR.has(v) ? '' : ' minor'), lab);
        xt.style.left = p + '%';
        const yt = el('span', 'm09-ytick' + (MAJOR.has(v) ? '' : ' minor'), lab);
        yt.style.top = 100 - p + '%';
        plot.append(xt, yt);
      });
      svgEl(svg, 'line', { x1: 0, y1: 100, x2: 100, y2: 0, class: 'm09-diag' });
      const dots = ids.map((i) => {
        const d = el('i', 'm09-dot');
        d.style.left = pos(xs[i]) + '%';
        plot.appendChild(d);
        return d;
      });
      let cur = -1;
      const mark = (k) => {
        if (cur >= 0) dots[cur].classList.remove('cur');
        cur = k;
        if (k >= 0) dots[k].classList.add('cur');
      };
      const nearest = (e) => {
        let best = -1, bd = 24 * 24;
        dots.forEach((d, k) => {
          const r = d.getBoundingClientRect();
          const dx = r.left + r.width / 2 - e.clientX, dy = r.top + r.height / 2 - e.clientY;
          if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = k; }
        });
        return best;
      };
      plot.addEventListener('pointermove', (e) => {
        if (e.pointerType !== 'mouse') return;
        const k = nearest(e);
        mark(k);
        if (k < 0) WALK.valueBox.hide();
        else WALK.valueBox.show(boxFor(ids[k]), e.clientX, e.clientY);
      });
      plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { mark(-1); WALK.valueBox.hide(); } });
      plot.addEventListener('pointerdown', (e) => {
        if (e.pointerType === 'mouse') return;
        const k = nearest(e);
        if (k < 0) return;
        mark(k);
        WALK.valueBox.showFor(dots[k], boxFor(ids[k]));
      });
      const order = ids.map((i, k) => k).sort((a, b) => xs[ids[a]] - xs[ids[b]]);
      plot.addEventListener('keydown', (e) => {
        const move = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, Home: -order.length, End: order.length }[e.key];
        if (move === undefined) return;
        e.preventDefault();
        const at = order.indexOf(cur);
        const from = at >= 0 ? at : move > 0 ? -1 : order.length;
        const k = order[Math.max(0, Math.min(order.length - 1, from + move))];
        mark(k);
        WALK.valueBox.showFor(dots[k], boxFor(ids[k]));
      });
      plot.addEventListener('blur', () => { mark(-1); WALK.valueBox.hide(); });
      return {
        yName,
        plot,
        setY(ys) { dots.forEach((d, k) => { d.style.top = 100 - pos(ys[ids[k]]) + '%'; }); },
        refresh() { if (cur >= 0) WALK.valueBox.showFor(dots[cur], boxFor(ids[cur])); },
      };
    }
    const scaleNote = 'Both axes are on a log scale of the size, on either side of 0 (linear within ±0.0001), so a value’s sign is the side of 0 it falls on. The scale keeps the order of the values, and rank agreement reads only that order.';
    const pairBox = (i) => {
      const b = sideOf(P.birth[i]);
      const s = sideOf(P.side[i]);
      const q = P.place_in_list[i];
      const rows = [['side switched off', q === 0 ? 'the carrier' : 'unit ' + F.int(s.unit) + ', ' + s.sign]];
      if (q > 0) rows.push(['its rank one snapshot before', ORD[q] + ' of the layer']);
      rows.push(['16-bit prediction', sig(P.prediction_16bit[i], 3)], ['32-bit prediction', sig(P.prediction_32bit[i], 3)],
        ['16-bit measurement', sig(P.measurement_16bit[i], 3)], ['32-bit measurement', sig(P.measurement_32bit[i], 3)]);
      return { title: 'Birth: layer ' + b.layer + ', unit ' + F.int(b.unit) + ', ' + b.sign + ' side, group ' + b.group, rows };
    };

    // ---------- the prediction against the measurement ----------
    const main = el('section', 'm09-main');
    wrap.appendChild(main);
    main.appendChild(el('h3', 'm09-h', 'The 16-bit prediction against the measured ΔL, ' + held.length + ' held-out pairs'));
    const controls = el('div', 'control-row');
    controls.appendChild(el('span', 'control-label', 'Measured in'));
    const tog = el('div', 'toggle');
    tog.setAttribute('role', 'group');
    tog.setAttribute('aria-label', 'Measurement precision');
    let state = '16';
    const stBtns = ['16', '32'].map((s) => {
      const b = el('button', '', s + '-bit');
      b.type = 'button';
      b.addEventListener('click', () => { state = s; update(); });
      tog.appendChild(b);
      return b;
    });
    controls.appendChild(tog);
    main.appendChild(controls);
    const grid = el('div', 'm09-cols');
    main.appendChild(grid);
    const left = el('div', 'm09-col');
    grid.appendChild(left);
    const leg = el('p', 'm09-legend');
    leg.append(legendKey('dot', 'one pair'), legendKey('diag', 'measurement = prediction'));
    left.appendChild(leg);
    const S1 = scatter(left, held, P.prediction_16bit, pairBox, held.length + ' held-out pairs: the 16-bit prediction across, the chosen measurement up; arrow keys move between pairs in order of the prediction');
    left.appendChild(el('p', 'm09-xname', '16-bit prediction ΔL̂, nats'));
    left.appendChild(el('p', 'm09-note', scaleNote));
    const read = el('div', 'm09-read');
    read.setAttribute('aria-live', 'polite');
    grid.appendChild(read);

    function update() {
      stBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(['16', '32'][k] === state)));
      const ys = state === '16' ? P.measurement_16bit : P.measurement_32bit;
      S1.setY(ys);
      S1.yName.textContent = state + '-bit measurement ΔL, nats';
      S1.refresh();
      read.replaceChildren();
      const fact = (big, text) => { const p = el('p', 'm09-fact'); p.append(el('b', '', big), el('span', '', text)); read.appendChild(p); };
      fact(F.fixed(rho[state].held, 3), 'rank agreement of the 16-bit prediction with the ' + state + '-bit measurement over the ' + held.length + ' held-out pairs drawn' +
        (state === '32' ? '; band ' + F.fixed(T32.band_held_out[0], 3) + ' to ' + F.fixed(T32.band_held_out[1], 3) + ', the range consistent with the data when the births are resampled' : ''));
      fact(F.fixed(rho[state].rule, 3), 'the same over the ' + rule.length + ' pairs of the half used to set the rules');
      read.appendChild(el('p', 'm09-note', state === '16'
        ? 'The 16-bit measurement is far larger than the prediction and unrelated to it in order. The study had required a rank agreement of at least ' + F.fixed(D.threshold_set_in_advance, 1) + ' before its gradient readings could count.'
        : 'The 32-bit measurement lies along the line where it equals the prediction. Shuffle reference: with the prediction shuffled across the pairs 1,000 times, 97.5 % of the shuffles agree less than ' +
          F.fixed(T32.shuffle_reference_held_out, 3) + '. The threshold set in advance was ' + F.fixed(D.threshold_set_in_advance, 1) + '.'));
    }

    // ---------- the two predictions, and the sizes ----------
    const two = el('div', 'm09-two');
    wrap.appendChild(two);
    const pa = el('section', 'm09-pa');
    two.appendChild(pa);
    pa.appendChild(el('h3', 'm09-h', 'The prediction in 16-bit against the prediction in 32-bit'));
    const S2 = scatter(pa, held, P.prediction_16bit, pairBox, held.length + ' held-out pairs: the 16-bit prediction across, the 32-bit prediction up');
    S2.setY(P.prediction_32bit);
    S2.yName.textContent = '32-bit prediction ΔL̂, nats';
    pa.appendChild(el('p', 'm09-xname', '16-bit prediction ΔL̂, nats'));
    const paFacts = el('div', 'm09-facts');
    const pfact = (big, text) => { const p = el('p', 'm09-fact small'); p.append(el('b', '', big), el('span', '', text)); paFacts.appendChild(p); };
    pfact(F.fixed(rhoPred, 3), 'rank agreement over the ' + held.length + ' held-out pairs');
    pfact(F.fixed(relDiff * 100, 1) + ' %', 'median relative difference, |16-bit − 32-bit| / |32-bit|');
    pa.appendChild(paFacts);
    pa.appendChild(el('p', 'm09-note', 'The 32-bit prediction was taken in forward mode along the direction −h. The two predictions agree, so the 16-bit prediction was not what failed.'));

    const pb = el('section', 'm09-pb');
    two.appendChild(pb);
    pb.appendChild(el('h3', 'm09-h', 'Sizes: what the 16-bit sum could not resolve'));
    pb.appendChild(el('p', 'm09-note', 'One tick per pair, at the size of the value; the dot is the median.'));
    const strip = el('div', 'm09-strip');
    pb.appendChild(strip);
    const LO = -5, HI = 1;
    const xs = (v) => ((Math.log10(v) - LO) / (HI - LO)) * 100;
    const sizeRow = (name, ids, k, med, what) => {
      const r = el('div', 'm09-srow');
      r.appendChild(el('span', 'm09-sname', name));
      const track = el('div', 'm09-strack');
      const svg = svgEl(track, 'svg', { viewBox: '0 0 100 10', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      let d = '';
      ids.forEach((i) => { d += 'M' + xs(Math.abs(P[k][i])).toFixed(3) + ' 0V10'; });
      svgEl(svg, 'path', { d, class: 'm09-ticks' });
      const dot = el('i', 'm09-mdot');
      dot.style.left = xs(med) + '%';
      const lab = el('span', 'm09-mlab' + (xs(med) > 60 ? ' left' : ''), 'median ' + (med >= 0.1 ? F.fixed(med, 2) : F.fixed(med, 3)));
      lab.style.left = xs(med) + '%';
      track.append(dot, lab);
      r.appendChild(track);
      WALK.hover(r, { title: name, rows: [['pairs', F.int(ids.length)], ['median size, nats', sig(med, 4), 'ink'], ['median of', what]] });
      strip.appendChild(r);
    };
    sizeRow('|16-bit prediction|, both halves', all, 'prediction_16bit', med16p, 'both halves');
    sizeRow('|16-bit measurement|, both halves', all, 'measurement_16bit', med16m, 'both halves');
    sizeRow('|32-bit measurement|, held-out half', held, 'measurement_32bit', med32m, 'the held-out half');
    const sax = el('div', 'm09-saxis');
    sax.setAttribute('aria-hidden', 'true');
    for (let p = -4; p <= 0; p++) {
      const t = el('span', 'm09-xtick', p === 0 ? '1' : F.fixed(Math.pow(10, p), -p));
      t.style.left = ((p - LO) / (HI - LO)) * 100 + '%';
      sax.appendChild(t);
    }
    strip.appendChild(sax);
    strip.appendChild(el('p', 'm09-xname', 'size in nats, log scale'));
    pb.appendChild(el('p', 'm09-note', 'The 16-bit measurement’s median size is ' + F.fixed(med16m, 2) + ' nats where the prediction says ' + F.fixed(med16p, 3) +
      ': summed in 16-bit over each window’s 2,048 losses, the measurement could not resolve a change of that size. The 32-bit measurement’s median, ' + F.fixed(med32m, 3) + ', matches the prediction’s.'));

    // ---------- why forward mode ----------
    const mem = el('section', 'm09-mem');
    wrap.appendChild(mem);
    mem.appendChild(el('h3', 'm09-h', 'Why the 32-bit derivative was taken in forward mode'));
    const bars = el('div', 'm09-bars');
    mem.appendChild(bars);
    const GBMAX = Math.ceil(WF.backward_32bit_peak_GB_max_over_births / 10) * 10;
    const bar = (name, gb, cls) => {
      const r = el('div', 'm09-brow');
      r.appendChild(el('span', 'm09-bname', name + ': ' + F.fixed(gb, 2) + ' GB'));
      const t = el('div', 'm09-btrack');
      const b = el('i', 'm09-bar ' + cls);
      b.style.width = (gb / GBMAX) * 100 + '%';
      t.appendChild(b);
      r.appendChild(t);
      WALK.hover(r, { title: name, rows: [['memory', F.fixed(gb, 2) + ' GB', cls]] });
      bars.appendChild(r);
    };
    bar('a 32-bit backward pass, for a birth in layer 0', WF.backward_32bit_peak_GB_layer0, 'ink');
    bar('the GPU’s memory', WF.gpu_memory_GB, 'control');
    mem.appendChild(el('p', 'm09-note', 'The largest need over the ' + WF.births + ' births is ' + F.fixed(WF.backward_32bit_peak_GB_max_over_births, 2) + ' GB; only ' + WF.births_that_would_fit + ' of the ' + WF.births +
      ' would fit with a margin. Forward mode carries the derivative along the one direction −h through a forward pass and stores no backward pass. On a small 32-bit model, for unit ' + sm.unit +
      ', forward mode gave ' + sm.forward_mode + ' and reverse mode ' + sm.reverse_mode + ': a relative difference of ' + sig(sm.relative_difference, 2) + '.'));

    update();
  }

  WALK.register('m09', { data: 'M09_DATA', mount });
})();
