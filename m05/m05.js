// m05: the onset rule on one cluster. Registers the mount "m05" with site.js; reads window.M05_DATA (m05/data.js).
// One cluster's median activation in its pole's direction against the pole's own 99.9th percentile at each of the
// sixteen checkpoints, with one drag over the checkpoints that applies the rule (at or above at t and at t + 1) and the
// onset marked; below, the ratio of the median to the percentile at the checkpoint before the onset for all 153 onsets.
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
  const flat = (s) => s.replace(/\r\n|\r|\n/g, '↵').replace(/\t/g, '⇥');
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])(?::c(\d+))?$/.exec(id);
    if (!m) throw new Error('m05: cannot read the id ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative', group: m[4] === undefined ? null : Number(m[4]),
      name: 'L' + Number(m[1]) + '.N' + Number(m[2]) + m[3] };
  };
  // The rule: the onset is the first t that is present at t and at t + 1; present at the last checkpoint alone counts there.
  const birthOf = (present) => present.findIndex((p, t) => p && (t === present.length - 1 || present[t + 1]));

  function mount(root, D) {
    const F = WALK.fmt;
    const steps = D.snapshot_steps;
    const NS16 = steps.length;
    const LAST = NS16 - 1;
    const E = D.example;
    const med = E.median_by_snapshot;
    const thr = E.threshold_by_snapshot;
    const present = E.present_by_snapshot;
    if (NS16 !== 16 || med.length !== NS16 || thr.length !== NS16 || present.length !== NS16) throw new Error('m05: expected sixteen checkpoints');
    present.forEach((p, t) => {
      if (p !== (med[t] > 0 && med[t] >= thr[t])) throw new Error('m05: present at checkpoint ' + t + ' does not follow from the median and the percentile');
    });
    const birth = birthOf(present);
    if (birth !== E.birth_snapshot.banked || birth !== E.birth_snapshot.recomputed) throw new Error('m05: the rule gives checkpoint ' + birth + ', the data ' + E.birth_snapshot.banked);
    const stepText = (t) => 'step ' + F.int(steps[t]);
    const who = sideOf(E.id);

    root.replaceChildren();
    const wrap = el('div', 'm05-wrap');
    root.appendChild(wrap);

    // ---------- the rule on one group ----------
    const one = el('section', 'm05-one');
    wrap.appendChild(one);
    one.appendChild(el('h3', 'm05-h', who.name + ' (neuron ' + F.int(who.unit) + ' of layer ' + who.layer + ', ' + E.side + ' pole), cluster ' + E.group));
    one.appendChild(el('p', 'm05-note', 'The cluster is ' + E.group_size + ' of the pole’s top-50 activating tokens (ranks ' + E.ranks_in_fifty.join(', ') +
      '), which the clustering LLM put together and named “' + E.group_name + '”. Its tokens are fixed corpus tokens, taken at the last ' +
      'checkpoint. At each checkpoint t we recomputed the activation at each of the ' + E.group_size + ' tokens in the pole’s direction ' +
      '(0 where it has the other sign) and compared the median of the ' + E.group_size + ' values with the pole’s own 99.9th percentile at t. ' +
      'The onset is the first checkpoint at which the median is at or above the percentile, and still is at the next checkpoint.'));

    const legend = el('p', 'm05-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm05-key ' + cls), text); return s; };
    legend.append(key('med', 'median over the cluster’s tokens'), key('thr', 'the pole’s own 99.9th percentile'), key('birth', 'onset'),
      key('on', 'median ≥ percentile'), key('off', 'median below it'));
    one.appendChild(legend);

    const chart = el('div', 'm05-chart');
    one.appendChild(chart);
    const plot = el('div', 'm05-plot');
    chart.appendChild(plot);
    const svg = svgEl(plot, 'svg', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    const positive = med.concat(thr).filter((v) => v > 0);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...positive))));
    const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...positive))));
    const LOGH = 80, ZERO = 93; // percent of the plot height: the log region, and the row that holds zeros
    const y = (v) => (v > 0 ? (1 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * LOGH : ZERO);
    const x = (t) => ((t + 0.5) / NS16) * 100;
    const put = (node, xp, yp) => { node.style.left = xp + '%'; node.style.top = yp + '%'; plot.appendChild(node); return node; };
    for (let v = lo; v <= hi * 1.0001; v *= 10) {
      svgEl(svg, 'line', { x1: 0, x2: 100, y1: y(v), y2: y(v), class: 'm05-grid' });
      put(el('span', 'm05-ytick', v >= 1 ? F.int(v) : String(v)), 0, y(v));
    }
    svgEl(svg, 'line', { x1: 0, x2: 100, y1: ZERO, y2: ZERO, class: 'm05-grid' });
    put(el('span', 'm05-ytick', '0'), 0, ZERO);
    const band = put(el('div', 'm05-cursor'), 0, 0);
    band.style.width = 100 / NS16 + '%';
    svgEl(svg, 'line', { x1: x(birth), x2: x(birth), y1: 0, y2: 100, class: 'm05-birthline' });
    put(el('span', 'm05-birthlab', 'onset'), x(birth), 0);
    const path = (vals, cls) => {
      let d = '';
      vals.forEach((v, t) => {
        if (!(v > 0)) return;
        d += (t > 0 && vals[t - 1] > 0 ? ' L' : ' M') + x(t) + ' ' + y(v);
      });
      svgEl(svg, 'path', { d: d.trim(), class: cls });
    };
    path(thr, 'm05-thrline');
    path(med, 'm05-medline');
    thr.forEach((v, t) => put(el('i', 'm05-dot thr'), x(t), y(v)));
    med.forEach((v, t) => put(el('i', 'm05-dot med' + (v > 0 ? '' : ' zero')), x(t), y(v)));
    put(el('span', 'm05-direct thr', '99.9th percentile'), x(1), y(thr[1])).classList.add('above');
    const low = med.reduce((b, v, t) => (v > 0 && t > 0 && (b < 0 || v < med[b]) ? t : b), -1);
    put(el('span', 'm05-direct med', 'median'), x(low), y(med[low])).classList.add('below');

    // the sixteen present cells and the checkpoints' training steps, under the plot
    const pres = el('div', 'm05-present');
    pres.setAttribute('aria-hidden', 'true');
    const cells = present.map((p, t) => {
      const c = el('i', 'm05-cell' + (p ? ' on' : ''));
      c.classList.toggle('birth', t === birth); // the onset's cell
      pres.appendChild(c);
      return c;
    });
    const presRow = el('div', 'm05-row-lab');
    presRow.appendChild(pres);
    chart.appendChild(presRow);
    const xl = el('div', 'm05-xlab');
    xl.setAttribute('aria-hidden', 'true');
    steps.forEach((s) => xl.appendChild(el('span', '', F.int(s))));
    const xRow = el('div', 'm05-row-lab');
    xRow.append(xl, el('span', 'm05-xname', 'training step of the checkpoint (the sixteen in order, not to scale)'));
    chart.appendChild(xRow);

    const scrubWrap = el('div', 'm05-scrubwrap');
    const scrub = el('input', 'scrub');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = String(LAST); scrub.step = '1'; scrub.value = String(birth);
    scrub.setAttribute('aria-label', 'Checkpoint');
    scrubWrap.appendChild(scrub);
    chart.appendChild(scrubWrap);

    // hover columns: one per checkpoint; a tap or click also moves the drag there
    const colBox = (t) => ({
      title: 'Checkpoint at step ' + F.int(steps[t]) + ' (stage1-step' + steps[t] + ')',
      rows: [['median', F.fixed(med[t], 4), 'ink'], ['99.9th percentile', F.fixed(thr[t], 4), 'control'], ['median ≥ percentile', present[t] ? 'yes' : 'no']],
    });
    steps.forEach((s, t) => {
      const c = put(el('div', 'm05-col'), x(t), 0);
      c.style.width = 100 / NS16 + '%';
      c.setAttribute('aria-label', 'Checkpoint at ' + stepText(t));
      c.tabIndex = -1; // the drag is the keyboard control; the columns serve the pointer
      WALK.hover(c, () => colBox(t));
      c.addEventListener('click', () => { scrub.value = String(t); update(); });
    });

    // the rule applied at the chosen checkpoint
    const out = el('div', 'm05-out');
    out.setAttribute('aria-live', 'polite');
    one.appendChild(out);
    const posList = el('div', 'm05-pos');
    one.appendChild(posList);
    const posHead = el('p', 'm05-note m05-poshead');
    posList.appendChild(posHead);
    const pgrid = el('div', 'm05-pgrid');
    posList.appendChild(pgrid);
    const prow = E.positions.map((p, j) => {
      const r = el('div', 'm05-prow');
      const before = el('span', 'm05-before');
      before.appendChild(el('span', '', flat(p.before_tail)));
      const v = el('span', 'm05-pval');
      r.append(el('span', 'm05-rank', String(p.rank)), before, el('span', 'm05-tok', flat(p.token)), v);
      pgrid.appendChild(r);
      return v;
    });
    const others = E.other_groups_of_side.filter((g) => g.c !== E.group).map((g) => g.birth);
    one.appendChild(el('p', 'm05-note', 'A second reading compares the median with a fixed share of the pole’s largest |a| at t; it puts this ' +
      'cluster’s onset at ' + stepText(E.birth_by_raw_ratio_reading) + '. We count an onset as late only when both readings put it after the onset of ' +
      'the pole’s earliest cluster (' + stepText(Math.min(...others)) + '); here both do.'));
    if (!E.late_by_both_readings) throw new Error('m05: the example is not late by both readings');

    function update() {
      const t = Number(scrub.value);
      band.style.left = (t / NS16) * 100 + '%';
      cells.forEach((c, i) => c.classList.toggle('pair', i === t || i === t + 1));
      scrub.setAttribute('aria-valuetext', 'checkpoint at ' + stepText(t));
      out.replaceChildren();
      out.appendChild(el('p', 'm05-outhead', 'Checkpoint at ' + stepText(t) + ' (stage1-step' + steps[t] + ')'));
      const tab = el('table', 'm05-tab');
      const tr = (cellsText, th) => { const r = el('tr'); cellsText.forEach((c, i) => r.appendChild(el(i === 0 || th ? 'th' : 'td', i ? 'num' : '', c))); tab.appendChild(r); };
      const has1 = t < LAST;
      tr(['', 'at ' + stepText(t), has1 ? 'next: ' + stepText(t + 1) : 'next'], true);
      tr(['median', F.fixed(med[t], 4), has1 ? F.fixed(med[t + 1], 4) : 'no next checkpoint']);
      tr(['99.9th percentile', F.fixed(thr[t], 4), has1 ? F.fixed(thr[t + 1], 4) : 'no next checkpoint']);
      tr(['median ≥ percentile', present[t] ? 'yes' : 'no', has1 ? (present[t + 1] ? 'yes' : 'no') : '']);
      out.appendChild(tab);
      let verdict;
      if (t === birth) verdict = 'The median is at or above the percentile here' + (has1 ? ' and at the next checkpoint' : ', the last checkpoint') +
        ', for the first time: the cluster’s onset is at ' + stepText(t) + '.';
      else if (t > birth) verdict = (present[t] ? 'The median is at or above the percentile here. ' : 'The median is below the percentile here. ') +
        'The onset came earlier, at ' + stepText(birth) + '.';
      else if (!present[t]) verdict = 'The median is below the percentile, so the onset is not here.';
      else verdict = 'The median is at or above the percentile here but not at the next checkpoint, so the onset is not here.';
      out.appendChild(el('p', 'm05-verdict', verdict));
      posHead.textContent = 'The ' + E.group_size + ' tokens at ' + stepText(t) + ' (rank in the top-50, text before, token, activation in the pole’s direction); the median of ' +
        E.group_size + ' values is the mean of the middle two.';
      E.activation_own_direction_by_snapshot[t].forEach((v, j) => { prow[j].textContent = F.fixed(v, 4); });
    }
    scrub.addEventListener('input', update);
    update();

    // ---------- all 153 births ----------
    const G = D.gradual;
    const P = G.per_birth;
    if (P.length !== G.births) throw new Error('m05: ' + P.length + ' onsets, the data says ' + G.births);
    const pre = P.map((b) => {
      if (b.ratio_by_snapshot.length !== NS16) throw new Error('m05: ' + b.birth + ' lacks a checkpoint');
      if (b.birth_snapshot < 3) throw new Error('m05: ' + b.birth + ' has its onset before the fourth checkpoint');
      const r = b.ratio_by_snapshot[b.birth_snapshot - 1];
      if (!(r >= 0 && r < 1)) throw new Error('m05: ' + b.birth + ' is at or above the threshold at the checkpoint before its onset');
      return r;
    });
    const less = P.filter((b) => !b.dated_step_holds_more_rise).length;
    if (less !== G.dated_step_holds_less_rise_than_two_before.banked) throw new Error('m05: ' + less + ' onsets with the smaller rise, the data says ' + G.dated_step_holds_less_rise_than_two_before.banked);
    const sorted = pre.slice().sort((a, b) => a - b);
    const mid = (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
    const MEDIAN = G.median_ratio_one_snapshot_before.recomputed;
    if (Math.abs(mid - MEDIAN) > 0.0005) throw new Error('m05: the median ratio is ' + mid + ', the data says ' + MEDIAN);

    const all = el('section', 'm05-all');
    wrap.appendChild(all);
    all.appendChild(el('h3', 'm05-h', 'All ' + G.births + ' onsets with p ≤ 0.05, at the checkpoint before the onset'));
    all.appendChild(el('p', 'm05-note', 'These are the ' + G.births + ' clusters whose onset has an empirical p-value of at most 0.05 against random ' +
      'clusters of the same size (the next picture shows the test). For each, the ratio is the cluster’s median divided by the pole’s 99.9th percentile.'));
    const facts = el('div', 'm05-facts');
    const fact = (big, text) => { const f = el('p', 'm05-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(F.fixed(MEDIAN, 3), 'the median ratio at the checkpoint before the onset');
    fact(less + ' of ' + G.births, 'onsets whose ratio rose less (log scale) in the interval into the onset than over the two intervals before it');
    all.appendChild(facts);
    all.appendChild(el('p', 'm05-note', 'Each dot is one onset, placed by that ratio; the ratio reaches 1 when the median equals the percentile. ' +
      'By the rule, a cluster is below 1 at the checkpoint before its onset. Median ratio by split: ' + F.fixed(G.median_ratio_one_snapshot_before_by_half.rule_setting, 4) +
      ' in the development split (neurons with an even index), ' + F.fixed(G.median_ratio_one_snapshot_before_by_half.held_out, 4) + ' in the held-out split (odd index).'));
    const leg2 = el('p', 'm05-legend');
    leg2.append(key('fill', 'smaller rise into the onset (' + less + ')'), key('ring', 'larger rise into the onset (' + (G.births - less) + ')'));
    all.appendChild(leg2);

    const BIN = 0.05;
    const NB = Math.round(1 / BIN);
    const bins = Array.from({ length: NB }, () => []);
    P.forEach((b, i) => bins[Math.min(NB - 1, Math.floor(pre[i] / BIN))].push(i));
    bins.forEach((list) => list.sort((a, b) => (P[a].dated_step_holds_more_rise - P[b].dated_step_holds_more_rise) || pre[a] - pre[b]));
    const dp = el('div', 'm05-dots');
    dp.tabIndex = 0;
    dp.setAttribute('role', 'group');
    dp.setAttribute('aria-label', G.births + ' onsets by their ratio at the checkpoint before the onset; arrow keys move between onsets');
    all.appendChild(dp);
    const dplot = el('div', 'm05-dplot');
    dp.appendChild(dplot);
    const order = [];
    const dotEls = [];
    for (let k = NB - 1; k >= 0; k--) {
      const row = el('div', 'm05-drow');
      dplot.appendChild(row);
      bins[k].forEach((i) => {
        const d = el('i', 'm05-bdot' + (P[i].dated_step_holds_more_rise ? ' ring' : ''));
        d.dataset.i = String(i);
        row.appendChild(d);
        order.push(i);
        dotEls[i] = d;
      });
    }
    const yr = (r) => (1 - r) * 100;
    for (let v = 0; v <= 10; v += 2) {
      const tk = el('span', 'm05-dtick', F.fixed(v / 10, 1));
      tk.style.top = yr(v / 10) + '%';
      dplot.appendChild(tk);
    }
    const thrLine = el('div', 'm05-dline thr');
    thrLine.style.top = '0%';
    thrLine.appendChild(el('span', 'm05-dlab', 'threshold: ratio 1'));
    const medLine = el('div', 'm05-dline med');
    medLine.style.top = yr(MEDIAN) + '%';
    medLine.appendChild(el('span', 'm05-dlab', 'median ' + F.fixed(MEDIAN, 3)));
    dplot.append(thrLine, medLine);
    all.appendChild(el('p', 'm05-note', 'Rows are bins of ' + F.fixed(BIN, 2) + ' in the ratio. Hover or tap a dot, or focus the chart and use the arrow keys.'));

    const birthBox = (i) => {
      const b = P[i];
      const w = sideOf(b.birth);
      const k = b.birth_snapshot;
      return {
        title: w.name + ', cluster ' + w.group,
        rows: [['onset at', stepText(k)],
          ['ratio at ' + stepText(k - 1), F.fixed(b.ratio_by_snapshot[k - 1], 5), 'ink'],
          ['ratio at ' + stepText(k), F.fixed(b.ratio_by_snapshot[k], 5)],
          ['rise into the onset', b.dated_step_holds_more_rise ? 'larger than over the two intervals before' : 'smaller than over the two intervals before'],
          ['split', b.half === 'rule_setting' ? 'development' : 'held-out']],
      };
    };
    let focusAt = -1;
    const mark = (i) => { dotEls.forEach((d) => d.classList.remove('cur')); if (i !== undefined) dotEls[i].classList.add('cur'); };
    dp.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const d = e.target.closest('.m05-bdot');
      if (!d) { WALK.valueBox.hide(); mark(); return; }
      mark(Number(d.dataset.i));
      WALK.valueBox.show(birthBox(Number(d.dataset.i)), e.clientX, e.clientY);
    });
    dp.addEventListener('pointerleave', () => { WALK.valueBox.hide(); mark(); });
    dp.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const d = e.target.closest('.m05-bdot');
      if (!d) return;
      focusAt = order.indexOf(Number(d.dataset.i));
      mark(Number(d.dataset.i));
      WALK.valueBox.showFor(d, birthBox(Number(d.dataset.i)));
    });
    dp.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -order.length, End: order.length }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      focusAt = Math.max(0, Math.min(order.length - 1, (focusAt < 0 ? 0 : focusAt + move)));
      const i = order[focusAt];
      mark(i);
      WALK.valueBox.showFor(dotEls[i], birthBox(i));
    });
    dp.addEventListener('blur', () => { WALK.valueBox.hide(); mark(); });
  }

  WALK.register('m05', { data: 'M05_DATA', mount });
})();
