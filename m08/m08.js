// m08: switching a side off at its group's positions. Registers the mount "m08" with site.js; reads window.M08_DATA
// (m08/data.js). One held-out birth read in 32-bit at the snapshot before its birth: for the chosen one of its five
// corpus windows, the clean and the switched-off loss at each of the 2,048 positions, their difference, and the sum;
// the five window sums added beside the banked value; the rounding check and the rank agreement's definition.
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
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m08: cannot read the side ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), sign: m[3] === '+' ? 'positive' : 'negative' };
  };
  const WINDOW = 2048; // losses per window: input j predicts input j + 1, j = 0 ... 2,047

  function mount(root, D) {
    const F = WALK.fmt;
    // a number with n significant digits, written in fixed notation
    const sig = (x, n) => (x === 0 ? '0' : F.fixed(x, Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(x))))));
    const B = D.birth;
    const W = D.windows;
    const who = sideOf(B.side);
    const sideName = (s) => 'layer ' + s.layer + ', unit ' + F.int(s.unit) + ', ' + s.sign + ' side';

    // ---------- self-checks: the data must say what the caption and the panels say ----------
    if (W.length !== B.unit_windows) throw new Error('m08: ' + W.length + ' windows, the birth has ' + B.unit_windows);
    if (B.measured_at_snapshot !== B.birth_snapshot - 1) throw new Error('m08: the switch-off is not read at the snapshot before the birth');
    const fromCorpus = new Map();
    B.group_positions_in_corpus.forEach((tok) => {
      const w = Math.floor(tok / WINDOW);
      const i = 1 + (tok % WINDOW);
      if (i <= WINDOW - 1) fromCorpus.set(w, (fromCorpus.get(w) || []).concat(i));
    });
    let total = 0;
    let rounding = 0;
    W.forEach((w) => {
      const pos = (fromCorpus.get(w.window) || []).sort((a, b) => a - b);
      if (pos.join() !== w.group_positions.join()) throw new Error('m08: window ' + w.window + ' positions do not follow from the corpus indices');
      if (w.loss_clean.length !== WINDOW) throw new Error('m08: window ' + w.window + ' has ' + w.loss_clean.length + ' losses, not 2,048');
      const i0 = w.first_changed_loss_index;
      if (i0 !== w.group_positions[0] || i0 < 1) throw new Error('m08: window ' + w.window + ' first changed index is not the first group position');
      const off = w.loss_switched_off_from_first_position;
      const dif = w.difference_from_first_position;
      if (off.length !== WINDOW - i0 || dif.length !== WINDOW - i0) throw new Error('m08: window ' + w.window + ' rows have the wrong length');
      let s = 0;
      let changed = 0;
      let first = -1;
      dif.forEach((d, k) => {
        // the banked difference is the float64 difference of two float32 losses: recompute it exactly
        if (Math.fround(off[k]) - Math.fround(w.loss_clean[i0 + k]) !== d) throw new Error('m08: window ' + w.window + ' difference at ' + (i0 + k) + ' does not equal switched off minus clean');
        s += d;
        if (d !== 0) { changed++; if (first < 0) first = i0 + k; }
      });
      if (changed !== w.tokens_changed || first !== w.first_loss_that_differs) throw new Error('m08: window ' + w.window + ' changed tokens do not match');
      if (Math.abs(s - w.window_sum) > 1e-9) throw new Error('m08: window ' + w.window + ' differences sum to ' + s + ', the data says ' + w.window_sum);
      total += w.window_sum;
      rounding += w.window_sum_rounding_check;
    });
    const T = D.loss_change_summed_over_windows;
    const carrier = D.sides_switched_off[0];
    if (carrier.side !== B.side || carrier.place_in_list !== 0) throw new Error('m08: the first switched-off side is not the carrier');
    if (total !== T.banked || T.recomputed !== T.banked || carrier.loss_change_32bit.banked !== T.banked) {
      throw new Error('m08: the five window sums add to ' + total + ', the banked value is ' + T.banked);
    }
    const RC = D.rounding_check_summed_over_windows;
    if (Math.abs(rounding - RC.banked) > 1e-12) throw new Error('m08: the rounding check sums to ' + rounding + ', the data says ' + RC.banked);
    const RA = D.rank_agreement_held_out;
    const MED = D.rounding_check_median_held_out;
    if (F.fixed(RA.value, 3) !== '0.960' || F.fixed(RA.band_low, 3) !== '0.915' || F.fixed(RA.band_high, 3) !== '0.990') throw new Error('m08: the rank agreement is not 0.960 [0.915, 0.990]');
    if (F.fixed(RA.shuffle_reference, 3) !== '0.118') throw new Error('m08: the shuffle reference is not 0.118');
    if (sig(MED.banked, 3) !== '0.000147' || MED.recomputed !== MED.banked) throw new Error('m08: the median rounding check is not 1.47e-4 nats');
    if (RA.n_pairs !== 4 * RA.n_births || D.sides_switched_off.length !== 4) throw new Error('m08: a birth does not give four pairs');

    root.replaceChildren();
    const wrap = el('div', 'm08-wrap');
    root.appendChild(wrap);

    // ---------- one window, token by token ----------
    const one = el('section', 'm08-one');
    wrap.appendChild(one);
    one.appendChild(el('h3', 'm08-h', sideName(who).replace(/^l/, 'L') + ', group ' + B.id.split(':c')[1]));
    one.appendChild(el('p', 'm08-note', 'The group is ' + B.group_size + ' of the side’s fifty contexts, named “' + B.group_name +
      '”. It is born at snapshot ' + B.birth_snapshot + ' (step ' + F.int(B.birth_step) + ') and is switched off here at snapshot ' +
      B.measured_at_snapshot + ' (step ' + F.int(B.measured_at_step) + '), the snapshot before the birth, in 32-bit. It is in the held-out half. ' +
      'Its positions fall in ' + W.length + ' corpus windows of the start-of-text token and 2,048 tokens; in each, the unit’s input to the down matrix is set to 0 at the group’s positions and nowhere else, and the window is run again.'));
    const formula = el('p', 'm08-formula');
    formula.textContent = 'ΔL = Σ over the windows, Σ over j = 0 … 2,047 of [loss switched off(j) − loss clean(j)], in nats';
    one.appendChild(formula);
    one.appendChild(el('p', 'm08-note', 'The loss at position j is the loss of predicting token j + 1. ΔL is a sum over every position, not a mean. The window’s text is not in the published files, so positions are indices.'));

    const controls = el('div', 'control-row');
    controls.appendChild(el('span', 'control-label', 'Corpus window'));
    const tog = el('div', 'toggle m08-win');
    tog.setAttribute('role', 'group');
    tog.setAttribute('aria-label', 'Corpus window');
    const winButtons = W.map((w, k) => {
      const b = el('button', '', F.int(w.window));
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => choose(k));
      tog.appendChild(b);
      return b;
    });
    controls.appendChild(tog);
    one.appendChild(controls);

    const legend = el('p', 'm08-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm08-key ' + cls), text); return s; };
    legend.append(key('clean', 'loss, clean'), key('off', 'loss, switched off (wide line, under the clean one)'), key('pos', 'the group’s position'), key('dif', 'switched off minus clean'));
    one.appendChild(legend);

    const chart = el('div', 'm08-chart');
    chart.tabIndex = 0;
    chart.setAttribute('role', 'group');
    chart.setAttribute('aria-label', 'Loss per position in the window; arrow keys move one position, Page Up and Page Down move 64');
    one.appendChild(chart);

    const lossMax = Math.ceil(Math.max(...W.map((w) => Math.max(...w.loss_clean, ...w.loss_switched_off_from_first_position)))) ;
    const LOSS_TICKS = [0, 5, 10, 15].filter((v) => v <= lossMax);
    const makePlot = (cls, name) => {
      const p = el('div', 'm08-plot ' + cls);
      p.appendChild(el('span', 'm08-yname', name));
      const svg = svgEl(p, 'svg', { viewBox: '0 0 ' + WINDOW + ' 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      chart.appendChild(p);
      return { p, svg, g: svgEl(svg, 'g', {}) };
    };
    const lossP = makePlot('loss', 'loss per position, nats');
    const difP = makePlot('dif', 'switched off minus clean, nats');
    const xAxis = el('div', 'm08-xaxis');
    xAxis.setAttribute('aria-hidden', 'true');
    [0, 500, 1000, 1500, 2000].forEach((v) => {
      const s = el('span', 'm08-xtick', F.int(v));
      s.style.left = ((v + 0.5) / WINDOW) * 100 + '%';
      xAxis.appendChild(s);
    });
    chart.appendChild(xAxis);
    chart.appendChild(el('span', 'm08-xname', 'position j in the window'));
    const yL = (v) => 100 - (v / lossMax) * 100;
    let difMax = 1;
    const yD = (v) => 50 - (v / difMax) * 50;
    const xp = (j) => ((j + 0.5) / WINDOW) * 100;

    // crosshair: one hairline across both plots, a dot on each series
    const cross = el('div', 'm08-cross');
    cross.hidden = true;
    chart.appendChild(cross);
    const dots = { clean: el('i', 'm08-dot clean'), off: el('i', 'm08-dot off'), dif: el('i', 'm08-dot dif') };
    lossP.p.append(dots.clean, dots.off);
    difP.p.appendChild(dots.dif);
    Object.values(dots).forEach((d) => { d.hidden = true; });

    const out = el('div', 'm08-out');
    out.setAttribute('aria-live', 'polite');
    one.appendChild(out);
    one.appendChild(el('p', 'm08-note m08-how', 'Hover or tap the chart for the exact losses at a position, or focus it and use the arrow keys.'));

    let cur = 0;
    let at = -1;
    const lossAt = (w, j) => {
      const i0 = w.first_changed_loss_index;
      const clean = w.loss_clean[j];
      return { clean, off: j < i0 ? clean : w.loss_switched_off_from_first_position[j - i0], dif: j < i0 ? 0 : w.difference_from_first_position[j - i0] };
    };
    const boxFor = (j) => {
      const w = W[cur];
      const v = lossAt(w, j);
      const before = j < w.first_changed_loss_index;
      return {
        title: 'Window ' + F.int(w.window) + ', position ' + F.int(j) + (w.group_positions.includes(j) ? ' (the group’s position)' : ''),
        rows: [['loss, clean', F.fixed(v.clean, 5), 'ink'], ['loss, switched off', F.fixed(v.off, 5), 'claim'],
          ['difference', before ? '0, bitwise equal' : sig(v.dif, 3)]],
      };
    };
    const show = (j, how, e) => {
      at = Math.max(0, Math.min(WINDOW - 1, j));
      const w = W[cur];
      const v = lossAt(w, at);
      cross.hidden = false;
      cross.style.left = xp(at) + '%';
      const place = (d, p, y) => { d.hidden = false; d.style.left = xp(at) + '%'; d.style.top = y + '%'; };
      place(dots.clean, lossP, yL(v.clean));
      place(dots.off, lossP, yL(v.off));
      place(dots.dif, difP, yD(v.dif));
      if (how === 'mouse') WALK.valueBox.show(boxFor(at), e.clientX, e.clientY);
      else WALK.valueBox.showFor(cross, boxFor(at));
    };
    const hide = () => {
      at = -1;
      cross.hidden = true;
      Object.values(dots).forEach((d) => { d.hidden = true; });
      WALK.valueBox.hide();
    };
    const jAt = (e) => {
      const r = lossP.svg.getBoundingClientRect();
      return Math.floor(((e.clientX - r.left) / r.width) * WINDOW);
    };
    chart.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') show(jAt(e), 'mouse', e); });
    chart.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hide(); });
    chart.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') show(jAt(e), 'tap'); });
    chart.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowLeft: -1, ArrowUp: 1, ArrowDown: -1, PageUp: 64, PageDown: -64, Home: -WINDOW, End: WINDOW }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      show((at < 0 ? W[cur].first_changed_loss_index : at) + move, 'key');
    });
    chart.addEventListener('focus', () => show(at < 0 ? W[cur].first_changed_loss_index : at, 'key'));
    chart.addEventListener('blur', hide);

    const yTicks = (p, ticks, y, d) => {
      p.p.querySelectorAll('.m08-ytick').forEach((t) => t.remove());
      ticks.forEach((v) => {
        svgEl(p.g, 'line', { x1: 0, x2: WINDOW, y1: y(v), y2: y(v), class: v === 0 ? 'm08-zero' : 'm08-grid' });
        const t = el('span', 'm08-ytick', F.fixed(v, d));
        t.style.top = y(v) + '%';
        p.p.appendChild(t);
      });
    };
    const posMarks = [];
    function draw() {
      const w = W[cur];
      const i0 = w.first_changed_loss_index;
      lossP.g.replaceChildren();
      difP.g.replaceChildren();
      posMarks.forEach((m) => m.remove());
      posMarks.length = 0;
      yTicks(lossP, LOSS_TICKS, yL, 0);
      // switched off first and wider, the clean loss on top: before the group's position the two are bitwise equal
      let d = '';
      w.loss_switched_off_from_first_position.forEach((v, k) => { d += (k ? 'L' : 'M') + (i0 + k + 0.5) + ' ' + yL(v).toFixed(3); });
      svgEl(lossP.g, 'path', { d, class: 'm08-offline' });
      d = '';
      w.loss_clean.forEach((v, j) => { d += (j ? 'L' : 'M') + (j + 0.5) + ' ' + yL(v).toFixed(3); });
      svgEl(lossP.g, 'path', { d, class: 'm08-cleanline' });
      // the difference: one needle per position from 0
      const amax = Math.max(...w.difference_from_first_position.map(Math.abs));
      const step = [1, 2, 5].map((m) => m * Math.pow(10, Math.floor(Math.log10(amax / 2)))).concat(10 * Math.pow(10, Math.floor(Math.log10(amax / 2)))).find((s) => 2 * s >= amax);
      difMax = 2 * step;
      const dec = Math.max(0, -Math.floor(Math.log10(step)));
      yTicks(difP, [-2 * step, -step, 0, step, 2 * step], yD, dec);
      d = '';
      w.difference_from_first_position.forEach((v, k) => { if (v !== 0) d += 'M' + (i0 + k + 0.5) + ' 50V' + yD(v).toFixed(3); });
      svgEl(difP.g, 'path', { d, class: 'm08-needles' });
      w.group_positions.forEach((i) => {
        [lossP, difP].forEach((p) => {
          const m = el('i', 'm08-posline');
          m.style.left = xp(i) + '%';
          p.p.appendChild(m);
          posMarks.push(m);
        });
        const lab = el('span', 'm08-poslab', 'group’s position ' + F.int(i));
        lab.style.left = xp(i) + '%';
        if (xp(i) > 60) lab.classList.add('left');
        lossP.p.appendChild(lab);
        posMarks.push(lab);
      });

      out.replaceChildren();
      const pos = w.group_positions.map(F.int).join(', ');
      out.appendChild(el('p', 'm08-outhead', 'Window ' + F.int(w.window) + ': sum of the 2,048 differences = ' + sig(w.window_sum, 4) + ' nats'));
      out.appendChild(el('p', 'm08-note', 'The group’s position in this window is ' + pos + '. Every loss before it is bitwise equal in the two runs. From it on, ' +
        F.int(w.tokens_changed) + ' of the ' + F.int(WINDOW - i0) + ' losses changed. The 32-bit prediction for this window is ' + sig(w.window_sum_32bit_prediction, 4) +
        ' nats; the rounding check (below) reads ' + sig(w.window_sum_rounding_check, 3) + ' nats.'));
      rows.forEach((r, k) => r.classList.toggle('cur', k === cur));
      if (at >= 0) show(at, 'key');
    }
    function choose(k) {
      cur = k;
      winButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
      at = -1;
      hide();
      draw();
    }

    // ---------- the birth's ΔL: the five window sums ----------
    const sum = el('section', 'm08-sum');
    wrap.appendChild(sum);
    sum.appendChild(el('h3', 'm08-h', 'The birth’s ΔL: the ' + W.length + ' window sums added'));
    const tab = el('table', 'm08-tab');
    const head = el('tr');
    ['window', 'position', 'changed', 'sum, nats'].forEach((h, i) => head.appendChild(el('th', i > 0 ? 'num' : '', h)));
    tab.appendChild(head);
    sum.appendChild(el('p', 'm08-note', 'Position: the group’s position in the window. Changed: the losses that differ between the two runs.'));
    const rows = W.map((w, k) => {
      const r = el('tr');
      r.append(el('td', '', F.int(w.window)), el('td', 'num', w.group_positions.map(F.int).join(', ')), el('td', 'num', F.int(w.tokens_changed)), el('td', 'num', sig(w.window_sum, 4)));
      r.addEventListener('click', () => choose(k));
      tab.appendChild(r);
      return r;
    });
    const totRow = (cls, text, v) => {
      const r = el('tr', cls);
      const th = el('th', '', text);
      th.colSpan = 3;
      r.append(th, el('td', 'num', sig(v, 6)));
      return r;
    };
    const tr1 = totRow('tot', 'the ' + W.length + ' sums added here', total);
    const tr2 = totRow('tot2', 'the banked ΔL of the carrier', T.banked);
    tab.append(tr1, tr2);
    const tabWrap = el('div', 'table-scroll');
    tabWrap.appendChild(tab);
    sum.appendChild(tabWrap);
    sum.appendChild(el('p', 'm08-note', 'The sum added on this page equals the banked value exactly (the page stops with an error if it does not). ' +
      'Select a row, or a window above, to draw it.'));

    // ---------- the measurement's floor and the rank agreement ----------
    const floor = el('section', 'm08-floor');
    wrap.appendChild(floor);
    floor.appendChild(el('h3', 'm08-h', 'The rounding check, and rank agreement'));
    floor.appendChild(el('p', 'm08-note', 'The rounding check runs the same window with the carrier’s change to the layer’s output added and then subtracted in the same forward pass. ' +
      'In exact arithmetic it changes nothing, so the loss change it reads is rounding: the smallest change the measurement can resolve.'));
    const strip = el('div', 'm08-strip');
    floor.appendChild(strip);
    const LO = -5, HI = 0; // log10 nats
    const xs = (v) => ((Math.log10(Math.abs(v)) - LO) / (HI - LO)) * 100;
    const stripRow = (name, v, cls) => {
      const r = el('div', 'm08-srow');
      r.appendChild(el('span', 'm08-sname', name));
      const track = el('div', 'm08-strack');
      const d = el('i', 'm08-sdot ' + cls);
      d.style.left = xs(v) + '%';
      const val = el('span', 'm08-sval', sig(Math.abs(v), 3));
      val.style.left = xs(v) + '%';
      if (xs(v) > 70) val.classList.add('left');
      track.append(d, val);
      r.appendChild(track);
      WALK.hover(r, { title: name, rows: [['reads, nats', sig(v, 4)], ['size, nats', sig(Math.abs(v), 4), cls === 'ink' ? 'ink' : 'control']] });
      strip.appendChild(r);
    };
    stripRow('Median size of the rounding check over the ' + RA.n_births + ' held-out births', MED.banked, 'control');
    stripRow('Size of this birth’s rounding check, ' + W.length + ' windows (it reads ' + sig(RC.banked, 3) + ')', RC.banked, 'control ring');
    stripRow('Size of this birth’s ΔL (it reads ' + sig(T.banked, 3) + ')', T.banked, 'ink');
    const sax = el('div', 'm08-saxis');
    sax.setAttribute('aria-hidden', 'true');
    for (let p = -4; p <= 0; p++) {
      const t = el('span', 'm08-xtick', p === 0 ? '1' : F.fixed(Math.pow(10, p), -p));
      t.style.left = ((p - LO) / (HI - LO)) * 100 + '%';
      sax.appendChild(t);
    }
    strip.appendChild(sax);
    strip.appendChild(el('span', 'm08-xname', 'size in nats, log scale'));

    floor.appendChild(el('p', 'm08-note', 'Rank agreement. Each held-out birth gives four pairs: the carrier, and the three other sides of its layer that ranked highest one snapshot before the birth, each switched off in turn. ' +
      RA.n_births + ' births give ' + RA.n_pairs + ' pairs. The pairs are ranked once by the prediction and once by the measured ΔL; the rank agreement is the correlation of the two rankings, 1 when the orders match and near 0 when they are unrelated.'));
    const facts = el('div', 'm08-facts');
    const fact = (big, text) => { const f = el('p', 'm08-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(F.fixed(RA.value, 3), 'rank agreement of the 16-bit gradient prediction with this 32-bit ΔL over the ' + RA.n_pairs + ' held-out pairs; band ' +
      F.fixed(RA.band_low, 3) + ' to ' + F.fixed(RA.band_high, 3) + ', the range consistent with the data when the births are resampled');
    fact(F.fixed(RA.shuffle_reference, 3), 'the shuffle reference: with the prediction shuffled across the pairs 1,000 times, 97.5 % of the shuffles agree less than this');
    floor.appendChild(facts);

    floor.appendChild(el('p', 'm08-note', 'This birth’s four pairs:'));
    const t4 = el('table', 'm08-tab');
    const h4 = el('tr');
    ['switched-off side', '16-bit prediction', '32-bit ΔL'].forEach((h, i) => h4.appendChild(el('th', i ? 'num' : '', h)));
    t4.appendChild(h4);
    D.sides_switched_off.forEach((s) => {
      const r = el('tr');
      const name = el('td');
      name.append(el('span', 'm08-sidename', sideName(sideOf(s.side))), el('span', 'm08-role', s.place_in_list === 0 ? 'the carrier' : s.role.replace(/^other side/, 'another side,')));
      r.append(name, el('td', 'num', sig(s.prediction_16bit, 3)), el('td', 'num', sig(s.loss_change_32bit.banked, 3)));
      t4.appendChild(r);
    });
    const t4w = el('div', 'table-scroll');
    t4w.appendChild(t4);
    floor.appendChild(t4w);
    floor.appendChild(el('p', 'm08-note', 'Values in nats, summed over the birth’s ' + W.length + ' windows.'));

    choose(0);
  }

  WALK.register('m08', { data: 'M08_DATA', mount });
})();
