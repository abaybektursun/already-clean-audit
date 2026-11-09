// m05: the birth rule on one unit. Registers the mount "m05" with site.js; reads window.M05_DATA (m05/data.js).
// One group's median own-direction activation against its side's own 99.9th percentile at each of the sixteen
// snapshots, with one drag over the snapshots that applies the rule (present at t and at t + 1) and the crossing
// marked; below, the ratio of the median to the percentile one snapshot before the birth for all 153 births.
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
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative', group: m[4] === undefined ? null : Number(m[4]) };
  };
  // The rule: born at the first t that is present at t and at t + 1; present at the final snapshot alone counts there.
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
    if (NS16 !== 16 || med.length !== NS16 || thr.length !== NS16 || present.length !== NS16) throw new Error('m05: expected sixteen snapshots');
    present.forEach((p, t) => {
      if (p !== (med[t] > 0 && med[t] >= thr[t])) throw new Error('m05: present at snapshot ' + t + ' does not follow from the median and the percentile');
    });
    const birth = birthOf(present);
    if (birth !== E.birth_snapshot.banked || birth !== E.birth_snapshot.recomputed) throw new Error('m05: the rule gives snapshot ' + birth + ', the data ' + E.birth_snapshot.banked);
    const who = sideOf(E.id);

    root.replaceChildren();
    const wrap = el('div', 'm05-wrap');
    root.appendChild(wrap);

    // ---------- the rule on one group ----------
    const one = el('section', 'm05-one');
    wrap.appendChild(one);
    one.appendChild(el('h3', 'm05-h', 'Layer ' + who.layer + ', unit ' + F.int(who.unit) + ', ' + E.side + ' side, group ' + E.group));
    one.appendChild(el('p', 'm05-note', 'The group is ' + E.group_size + ' of the side’s fifty contexts (ranks ' + E.ranks_in_fifty.join(', ') +
      '), named “' + E.group_name + '”. Its positions are fixed at the final snapshot. At each snapshot t the activation is read at each ' +
      'position in the side’s own direction (0 where it has the other sign), and the median of the ' + E.group_size +
      ' values is compared with the side’s own 99.9th percentile at t.'));

    const legend = el('p', 'm05-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm05-key ' + cls), text); return s; };
    legend.append(key('med', 'median over the group’s positions'), key('thr', 'the side’s own 99.9th percentile'), key('birth', 'birth'),
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
    put(el('span', 'm05-birthlab', 'born at ' + birth), x(birth), 0);
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

    // the sixteen present cells and the snapshot numbers, under the plot
    const pres = el('div', 'm05-present');
    pres.setAttribute('aria-hidden', 'true');
    const cells = present.map((p, t) => { const c = el('i', 'm05-cell' + (p ? ' on' : '') + (t === birth ? ' birth' : '')); pres.appendChild(c); return c; });
    const presRow = el('div', 'm05-row-lab');
    presRow.appendChild(pres);
    chart.appendChild(presRow);
    const xl = el('div', 'm05-xlab');
    xl.setAttribute('aria-hidden', 'true');
    steps.forEach((s, t) => xl.appendChild(el('span', '', String(t))));
    const xRow = el('div', 'm05-row-lab');
    xRow.append(xl, el('span', 'm05-xname', 'snapshot'));
    chart.appendChild(xRow);

    const scrubWrap = el('div', 'm05-scrubwrap');
    const scrub = el('input', 'scrub');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = String(LAST); scrub.step = '1'; scrub.value = String(birth);
    scrub.setAttribute('aria-label', 'Snapshot');
    scrubWrap.appendChild(scrub);
    chart.appendChild(scrubWrap);

    // hover columns: one per snapshot; a tap or click also moves the drag there
    const colBox = (t) => ({
      title: 'Snapshot ' + t + ', step ' + F.int(steps[t]),
      rows: [['median', F.fixed(med[t], 4), 'ink'], ['99.9th percentile', F.fixed(thr[t], 4), 'control'], ['median ≥ percentile', present[t] ? 'yes' : 'no']],
    });
    steps.forEach((s, t) => {
      const c = put(el('div', 'm05-col'), x(t), 0);
      c.style.width = 100 / NS16 + '%';
      c.setAttribute('aria-label', 'Snapshot ' + t);
      c.tabIndex = -1; // the drag is the keyboard control; the columns serve the pointer
      WALK.hover(c, () => colBox(t));
      c.addEventListener('click', () => { scrub.value = String(t); update(); });
    });

    // the rule applied at the chosen snapshot
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
    one.appendChild(el('p', 'm05-note', 'A second reading, the median against a fixed share of the side’s largest |a| at t, dates this group to snapshot ' +
      E.birth_by_raw_ratio_reading + '. A birth is counted as late only when both readings date it after the side’s first-born group (snapshot ' +
      Math.min(...others) + '); here both do.'));
    if (!E.late_by_both_readings) throw new Error('m05: the example is not late by both readings');

    function update() {
      const t = Number(scrub.value);
      band.style.left = (t / NS16) * 100 + '%';
      cells.forEach((c, i) => c.classList.toggle('pair', i === t || i === t + 1));
      scrub.setAttribute('aria-valuetext', 'snapshot ' + t + ', step ' + F.int(steps[t]));
      out.replaceChildren();
      out.appendChild(el('p', 'm05-outhead', 'Snapshot ' + t + ', step ' + F.int(steps[t])));
      const tab = el('table', 'm05-tab');
      const tr = (cellsText, th) => { const r = el('tr'); cellsText.forEach((c, i) => r.appendChild(el(i === 0 || th ? 'th' : 'td', i ? 'num' : '', c))); tab.appendChild(r); };
      const has1 = t < LAST;
      tr(['', 'at t = ' + t, has1 ? 'at t + 1 = ' + (t + 1) : 'at t + 1'], true);
      tr(['median', F.fixed(med[t], 4), has1 ? F.fixed(med[t + 1], 4) : 'no snapshot']);
      tr(['99.9th percentile', F.fixed(thr[t], 4), has1 ? F.fixed(thr[t + 1], 4) : 'no snapshot']);
      tr(['median ≥ percentile', present[t] ? 'yes' : 'no', has1 ? (present[t + 1] ? 'yes' : 'no') : '']);
      out.appendChild(tab);
      let verdict;
      if (t === birth) verdict = 'Present at t' + (has1 ? ' and at t + 1' : ', the final snapshot') + ', and the first snapshot where that holds: the group is born at snapshot ' + t + ', step ' + F.int(steps[t]) + '.';
      else if (t > birth) verdict = (present[t] ? 'Present at t. ' : 'Not present at t. ') + 'The group was already born, at snapshot ' + birth + '.';
      else if (!present[t]) verdict = 'The median is below the percentile: not present at t, so not born at t.';
      else verdict = 'Present at t but not at t + 1: not born at t.';
      out.appendChild(el('p', 'm05-verdict', verdict));
      posHead.textContent = 'The ' + E.group_size + ' positions at snapshot ' + t + ' (rank in the fifty, text before, token, activation in the side’s own direction); the median of ' +
        E.group_size + ' values is the mean of the middle two.';
      E.activation_own_direction_by_snapshot[t].forEach((v, j) => { prow[j].textContent = F.fixed(v, 4); });
    }
    scrub.addEventListener('input', update);
    update();

    // ---------- all 153 births ----------
    const G = D.gradual;
    const P = G.per_birth;
    if (P.length !== G.births) throw new Error('m05: ' + P.length + ' births, the data says ' + G.births);
    const pre = P.map((b) => {
      if (b.ratio_by_snapshot.length !== NS16) throw new Error('m05: ' + b.birth + ' lacks a snapshot');
      if (b.birth_snapshot < 3) throw new Error('m05: ' + b.birth + ' is born before snapshot 3');
      const r = b.ratio_by_snapshot[b.birth_snapshot - 1];
      if (!(r >= 0 && r < 1)) throw new Error('m05: ' + b.birth + ' is at or above the threshold one snapshot before its birth');
      return r;
    });
    const less = P.filter((b) => !b.dated_step_holds_more_rise).length;
    if (less !== G.dated_step_holds_less_rise_than_two_before.banked) throw new Error('m05: ' + less + ' births hold less of the rise, the data says ' + G.dated_step_holds_less_rise_than_two_before.banked);
    const sorted = pre.slice().sort((a, b) => a - b);
    const mid = (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
    const MEDIAN = G.median_ratio_one_snapshot_before.recomputed;
    if (Math.abs(mid - MEDIAN) > 0.0005) throw new Error('m05: the median ratio is ' + mid + ', the data says ' + MEDIAN);

    const all = el('section', 'm05-all');
    wrap.appendChild(all);
    all.appendChild(el('h3', 'm05-h', 'All ' + G.births + ' births that beat chance, one snapshot before the birth'));
    const facts = el('div', 'm05-facts');
    const fact = (big, text) => { const f = el('p', 'm05-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(F.fixed(MEDIAN, 3), 'the median ratio of the group’s median to the side’s 99.9th percentile one snapshot before the birth');
    fact(less + ' of ' + G.births, 'births in which the step into the birth snapshot holds less of the rise (on a log scale) than the two steps before it');
    all.appendChild(facts);
    all.appendChild(el('p', 'm05-note', 'Each dot is one birth, placed by that ratio; the ratio reaches 1 when the median equals the percentile. ' +
      'By the rule, a group is below 1 one snapshot before its birth. Medians by half: ' + F.fixed(G.median_ratio_one_snapshot_before_by_half.rule_setting, 4) +
      ' in the half used to set the rules, ' + F.fixed(G.median_ratio_one_snapshot_before_by_half.held_out, 4) + ' in the held-out half.'));
    const leg2 = el('p', 'm05-legend');
    leg2.append(key('fill', 'the birth step holds less of the rise (' + less + ')'), key('ring', 'it holds more (' + (G.births - less) + ')'));
    all.appendChild(leg2);

    const BIN = 0.05;
    const NB = Math.round(1 / BIN);
    const bins = Array.from({ length: NB }, () => []);
    P.forEach((b, i) => bins[Math.min(NB - 1, Math.floor(pre[i] / BIN))].push(i));
    bins.forEach((list) => list.sort((a, b) => (P[a].dated_step_holds_more_rise - P[b].dated_step_holds_more_rise) || pre[a] - pre[b]));
    const dp = el('div', 'm05-dots');
    dp.tabIndex = 0;
    dp.setAttribute('role', 'group');
    dp.setAttribute('aria-label', G.births + ' births by ratio one snapshot before the birth; arrow keys move between births');
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
        title: 'Layer ' + w.layer + ', unit ' + F.int(w.unit) + ', ' + w.side + ' side, group ' + w.group,
        rows: [['born at', 'snapshot ' + k + ', step ' + F.int(steps[k])],
          ['ratio at snapshot ' + (k - 1), F.fixed(b.ratio_by_snapshot[k - 1], 5), 'ink'],
          ['ratio at snapshot ' + k, F.fixed(b.ratio_by_snapshot[k], 5)],
          ['the birth step holds', b.dated_step_holds_more_rise ? 'more of the rise' : 'less of the rise'],
          ['half', b.half === 'rule_setting' ? 'used to set the rules' : 'held out']],
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
