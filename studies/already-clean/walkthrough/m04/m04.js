// m04: the parent test and its control (PLAN_PUBLIC_2026-11-08.md §3a row M4).
// Builds inside #m04 from window.M04_DATA (m04/data.js): the judge's prompt word for word, the three banked calls
// with the group's contexts and the two records the judge saw, and the unrelated-share formula with the call counts
// it comes from. Every number shown is read from the data object or computed from it here and checked against it.
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
  const fmt = () => WALK.fmt;

  function h(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function s(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  // One value box for a list of rows: the pointer, a tap, or the arrow keys once the list has focus.
  function listHover(list, rows, content) {
    let cur = -1;
    list.tabIndex = 0;
    list.setAttribute('aria-describedby', 'walk-value-box');
    const mark = (i) => {
      rows.forEach((r, j) => r.classList.toggle('is-cur', j === i));
      cur = i;
    };
    list.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const r = e.target.closest('[data-i]');
      if (!r) return;
      mark(+r.dataset.i);
      WALK.valueBox.show(content(cur), e.clientX, e.clientY);
    });
    list.addEventListener('pointerleave', (e) => {
      if (e.pointerType !== 'mouse') return;
      mark(-1);
      WALK.valueBox.hide();
    });
    list.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const r = e.target.closest('[data-i]');
      if (!r) return;
      mark(+r.dataset.i);
      WALK.valueBox.showFor(r, content(cur));
    });
    list.addEventListener('focus', () => {
      mark(cur < 0 ? 0 : cur);
      WALK.valueBox.showFor(rows[cur], content(cur));
    });
    list.addEventListener('blur', () => { mark(-1); WALK.valueBox.hide(); });
    list.addEventListener('keydown', (e) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      mark(Math.max(0, Math.min(rows.length - 1, cur + step)));
      rows[cur].scrollIntoView({ block: 'nearest' });
      WALK.valueBox.showFor(rows[cur], content(cur));
    });
  }

  // A context as the judge saw it: the text before, the marked token, the text after, on one line.
  function contexts(parent, list, heading) {
    const box = h('div', 'm04-list', parent);
    box.setAttribute('role', 'list');
    box.setAttribute('aria-label', heading + ': ' + list.length + ' contexts; arrow keys show each in full');
    const rows = list.map((c, i) => {
      const row = h('div', 'm04-ctx', box);
      row.dataset.i = i;
      row.setAttribute('role', 'listitem');
      h('span', null, h('span', 'm04-b', row), c.before);
      h('span', 'm04-t', row, c.token);
      h('span', 'm04-a', row, c.after);
      return row;
    });
    listHover(box, rows, (i) => ({
      title: heading + ', context ' + (i + 1) + ' of ' + list.length,
      rows: [[list[i].before + ' ⟦' + list[i].token + '⟧ ' + list[i].after, '']],
    }));
  }

  const OUTCOME = {
    'late-right': 'Late-born group, judge right',
    'late-wrong': 'Late-born group, judge wrong',
    'control-right': 'Control, judge right',
  };

  function mount(root, D) {
    const F = D.formula;
    const C = D.control_check;
    const n = F.calls_right;
    const A = n.late_born / n.late_born_calls;
    const Ac = n.control / n.control_calls;
    const chance = F.inputs_as_reported.chance;
    const pu = (Ac - A) / (Ac - chance);
    const near = (x, y, tol, what) => {
      if (Math.abs(x - y) > tol) throw new Error('m04: ' + what + ' computed here ' + x + ' differs from data.js ' + y);
    };
    near(A, F.judge_accuracy_late_born.recomputed, 5e-6, 'A');
    near(Ac, F.control_accuracy.recomputed, 5e-6, 'a_control');
    near(pu, F.unrelated_share.recomputed, 5e-5, 'p_u');
    near(n.late_born_calls, F.late_born_groups * D.judgments_per_group, 0, 'late-born calls');
    near(n.control_calls, F.control_groups * D.judgments_per_group, 0, 'control calls');
    const f = fmt();

    root.replaceChildren();

    // ---------- groups ----------
    const g = h('section', 'm04-part', root);
    h('h3', null, g, 'Groups');
    const gp = h('p', 'm04-text', g);
    gp.append('A separate model, ' + D.sorter_model + ', sorts a side’s fifty contexts into groups and names each one. ' +
      'A second, independent sorting must give the same number of groups within one (');
    h('code', null, gp, D.sorting_agreement_rule);
    gp.append('). Contexts where both of the unit’s input terms are negative form one flagged group that is left out of every count.');

    // ---------- the question ----------
    const q = h('section', 'm04-part', root);
    h('h3', null, q, 'The judge’s question, word for word');
    const qp = h('p', 'm04-text', q);
    qp.append('Judge model ');
    h('code', null, qp, D.judge_model);
    qp.append('. Prompt file checksum ');
    h('code', 'm04-hash', qp, D.prompt_sha256);
    qp.append('. The judge answers ' + D.judgments_per_group + ' times per group.');
    const pr = h('div', 'm04-prompt', q);
    h('p', 'm04-turn', pr, 'System turn');
    h('pre', null, pr, D.prompt_system);
    h('p', 'm04-turn', pr, 'User turn: the three places in braces are filled with the contexts below');
    h('pre', null, pr, D.prompt_user);

    // ---------- one call ----------
    const c = h('section', 'm04-part', root);
    h('h3', null, c, 'One call, as the judge saw it');
    const tg = h('div', 'toggle m04-toggle', c);
    tg.setAttribute('role', 'group');
    tg.setAttribute('aria-label', 'Banked call');
    const out = h('p', 'm04-text m04-summary', c);
    out.setAttribute('aria-live', 'polite');
    const body = h('div', null, c);
    const buttons = D.example_calls.map((call, i) => {
      if (!OUTCOME[call.outcome]) throw new Error('m04: unknown outcome ' + call.outcome);
      const b = h('button', null, tg, OUTCOME[call.outcome]);
      b.type = 'button';
      b.addEventListener('click', () => show(i));
      return b;
    });

    function show(i) {
      const call = D.example_calls[i];
      buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
      const control = call.outcome.startsWith('control');
      out.textContent = (control
        ? 'The control: the side’s main group, with records from snapshot ' + call.snapshot_before + '. '
        : 'A late-born group, with records from snapshot ' + call.snapshot_before + ', the snapshot before its birth. ') +
        'The group’s own side is record ' + call.own_side_record + '. The judge answered ' + call.judge_answer +
        ': ' + (call.judge_right ? 'right.' : 'wrong.');
      body.replaceChildren();
      const ex = h('div', 'm04-block', body);
      h('p', 'm04-head', ex, 'The group’s contexts at the end of training: ' + call.group_contexts.length +
        ' (the judge is shown at most ' + D.group_contexts_shown_at_most + ')');
      contexts(ex, call.group_contexts, 'Group');
      const recs = h('div', 'm04-records', body);
      for (const [key, list] of [['A', call.record_a], ['B', call.record_b]]) {
        const own = key === call.own_side_record;
        const r = h('div', 'm04-block', recs);
        const head = h('p', 'm04-head', r);
        h('b', null, head, 'Record ' + key);
        head.append(' ');
        const tag = h('span', 'm04-tag', head);
        h('i', 'm04-key ' + (own ? 'm04-own' : 'm04-other'), tag);
        tag.append(own ? 'the group’s own side' : 'a random side of the same layer');
        if (key === call.judge_answer) h('span', 'm04-answer', head, 'the judge’s answer');
        h('p', 'm04-sub', r, list.length + ' strongest contexts at snapshot ' + call.snapshot_before +
          ' (the judge saw them without this heading)');
        contexts(r, list, 'Record ' + key);
      }
    }
    show(0);

    // ---------- the formula ----------
    const fx = h('section', 'm04-part', root);
    h('h3', null, fx, 'The unrelated share');
    const defs = {
      A: () => ({ title: 'A: the judge’s accuracy on late-born groups', rows: [
        ['right answers', f.int(n.late_born) + ' of ' + f.int(n.late_born_calls)],
        ['calls', f.int(F.late_born_groups) + ' groups × ' + D.judgments_per_group],
        ['A', f.fixed(A, 5)]] }),
      Ac: () => ({ title: 'a_control: the judge’s accuracy on the control (each side’s main group, records from snapshot ' +
        D.example_calls.find((x) => x.outcome.startsWith('control')).snapshot_before + ')', rows: [
        ['right answers', f.int(n.control) + ' of ' + f.int(n.control_calls)],
        ['calls', f.int(F.control_groups) + ' main groups × ' + D.judgments_per_group],
        ['a_control', f.fixed(Ac, 5)]] }),
      chance: () => ({ title: 'Chance: a two-choice question answered at random is right half the time', rows: [['chance', f.fixed(chance, 1)]] }),
      pu: () => ({ title: 'p_u: the unrelated share. If a share p_u of late-born groups is judged at chance and the rest as ' +
        'accurately as the control, A = (1 − p_u)·a_control + p_u·0.5; solved for p_u', rows: [
        ['p_u', f.fixed(pu, 4) + ', reported ' + f.fixed(F.unrelated_share.banked, 3)],
        ['range consistent with the data', f.fixed(F.range[0], 3) + ' to ' + f.fixed(F.range[1], 3)],
        ['line set in advance', f.fixed(F.rule_line, 2)]] }),
      thr: () => ({ title: 'The threshold set in advance: the control had to reach it on the tuning sides before the test sides were judged',
        rows: [['threshold', f.fixed(C.threshold, 2)]] }),
      tune: () => ({ title: 'The control on the tuning sides', rows: [['sides', f.int(C.tuning_sides)], ['accuracy', f.fixed(C.tuning_accuracy, 3)]] }),
    };
    const term = (parent, text, key, cls) => {
      const t = h('span', 'm04-term' + (cls ? ' ' + cls : ''), parent, text);
      WALK.hover(t, defs[key]);
      return t;
    };
    const fl = h('div', 'm04-formula', fx);
    const l1 = h('p', null, fl);
    term(l1, 'p_u', 'pu');
    l1.append(' = (');
    term(l1, 'a_control', 'Ac');
    l1.append(' − ');
    term(l1, 'A', 'A');
    l1.append(') / (');
    term(l1, 'a_control', 'Ac');
    l1.append(' − ');
    term(l1, f.fixed(chance, 1), 'chance');
    l1.append(')');
    const l2 = h('p', null, fl);
    l2.append('= (');
    term(l2, f.fixed(F.inputs_as_reported.a_control, 3), 'Ac');
    l2.append(' − ');
    term(l2, f.fixed(F.inputs_as_reported.A, 3), 'A');
    l2.append(') / (');
    term(l2, f.fixed(F.inputs_as_reported.a_control, 3), 'Ac');
    l2.append(' − ');
    term(l2, f.fixed(chance, 1), 'chance');
    l2.append(') = ');
    term(l2, f.fixed(F.unrelated_share.banked, 3), 'pu');

    const counts = h('ul', 'm04-counts', fx);
    const li = (parent, key, label, rest) => { const x = h('li', null, parent); term(x, label, key); x.append(rest); return x; };
    li(counts, 'A', 'A', ' = ' + f.int(n.late_born) + ' right of ' + f.int(n.late_born_calls) + ' calls (' +
      f.int(F.late_born_groups) + ' late-born groups × ' + D.judgments_per_group + ') = ' + f.fixed(A, 5));
    li(counts, 'Ac', 'a_control', ' = ' + f.int(n.control) + ' right of ' + f.int(n.control_calls) + ' calls (' +
      f.int(F.control_groups) + ' main groups × ' + D.judgments_per_group + ') = ' + f.fixed(Ac, 5));
    li(counts, 'pu', 'p_u', ' = (' + f.fixed(Ac, 5) + ' − ' + f.fixed(A, 5) + ') / (' + f.fixed(Ac, 5) + ' − ' +
      f.fixed(chance, 1) + ') = ' + f.fixed(pu, 4) + ', reported as ' + f.fixed(F.unrelated_share.banked, 3) +
      '; the three-decimal values above give ' + f.fixed(F.unrelated_share_from_rounded_inputs, 4));

    // the accuracy axis: chance, the threshold, the control on tuning and test sides, A, and the two differences
    const ax = h('div', 'm04-axis', fx);
    ax.setAttribute('role', 'img');
    ax.setAttribute('aria-label', 'Judge accuracy from ' + f.fixed(chance, 1) + ' to 1: A ' + f.fixed(A, 3) +
      ', a_control ' + f.fixed(Ac, 3) + ', the control threshold ' + f.fixed(C.threshold, 2) + ', the control on ' +
      C.tuning_sides + ' tuning sides ' + f.fixed(C.tuning_accuracy, 3));
    const ctl = h('p', 'm04-text', fx);
    ctl.append('The control check: before the test sides were judged, the control had to reach at least ');
    term(ctl, f.fixed(C.threshold, 2), 'thr');
    ctl.append(' on the tuning sides. It reached ');
    term(ctl, f.fixed(C.tuning_accuracy, 3), 'tune');
    ctl.append(' on ' + C.tuning_sides + ' sides, and ' + f.fixed(Ac, 3) + ' on the test sides.');

    let lastW = 0;
    function drawAxis() {
      const W = ax.clientWidth;
      if (!W || W === lastW) return;
      lastW = W;
      const padL = 8, padR = 14;
      const x = (v) => padL + ((v - chance) / (1 - chance)) * (W - padL - padR);
      const rowH = 20, y0 = 2 * rowH + 18; // two label rows above the axis line
      const H = y0 + 2 * rowH + 24 + 2 * 26 + 8;
      ax.replaceChildren();
      const svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H }, ax);
      s('line', { x1: x(chance), x2: x(1), y1: y0, y2: y0, class: 'm04-rule' }, svg);
      for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
        s('line', { x1: x(v), x2: x(v), y1: y0, y2: y0 + 5, class: 'm04-rule' }, svg);
      }
      // label rows: above 1 and 2, below 1 and 2; each label ends at its mark when the mark is right of centre
      const label = (v, row, text, cls) => {
        const ty = row < 0 ? y0 + row * rowH - 6 : y0 + 12 + row * rowH;
        const right = x(v) > W * 0.5;
        s('line', { x1: x(v), x2: x(v), y1: y0, y2: row < 0 ? ty + 4 : ty - 12, class: 'm04-leader' }, svg);
        s('text', { x: x(v) + (right ? 4 : -4), y: ty, 'text-anchor': right ? 'end' : 'start', class: 'm04-lab' + (cls ? ' ' + cls : '') }, svg, text);
      };
      s('text', { x: x(1), y: y0 + 12 + rowH, 'text-anchor': 'end', class: 'm04-lab' }, svg, f.fixed(1, 1));
      label(Ac, -2, 'a_control ' + f.fixed(Ac, 3) + ', test sides');
      label(A, -1, 'A ' + f.fixed(A, 3) + ', late-born groups');
      label(chance, 1, 'chance ' + f.fixed(chance, 1));
      label(C.threshold, 1, 'threshold ' + f.fixed(C.threshold, 2));
      label(C.tuning_accuracy, 2, 'control ' + f.fixed(C.tuning_accuracy, 3) + ', ' + C.tuning_sides + ' tuning sides');
      s('line', { x1: x(C.threshold), x2: x(C.threshold), y1: y0 - 8, y2: y0 + 8, class: 'm04-thr' }, svg);
      s('circle', { cx: x(C.tuning_accuracy), cy: y0, r: 4, class: 'm04-dot-tune' }, svg);
      s('circle', { cx: x(Ac), cy: y0, r: 5, class: 'm04-dot-control' }, svg);
      s('circle', { cx: x(A), cy: y0, r: 5, class: 'm04-dot-ink' }, svg);
      s('circle', { cx: x(chance), cy: y0, r: 4, class: 'm04-dot-muted' }, svg);
      // the two differences of the formula, as lengths on the same axis
      const yb = y0 + 2 * rowH + 30;
      const bar = (from, to, y, text) => {
        s('rect', { x: x(from), y: y - 4, width: x(to) - x(from), height: 8, rx: 2, class: 'm04-bar' }, svg);
        s('text', { x: x(from) - 6, y: y + 5, 'text-anchor': 'end', class: 'm04-lab' }, svg, text);
      };
      bar(A, Ac, yb, 'a_control − A = ' + f.fixed(Ac - A, 5));
      const yd = yb + 26;
      s('rect', { x: x(chance), y: yd - 4, width: x(Ac) - x(chance), height: 8, rx: 2, class: 'm04-bar' }, svg);
      s('text', { x: x(chance) + 2, y: yd + 22, class: 'm04-lab' }, svg,
        'a_control − ' + f.fixed(chance, 1) + ' = ' + f.fixed(Ac - chance, 5));
      svg.setAttribute('height', yd + 30);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + (yd + 30));
    }
    drawAxis();
    new ResizeObserver(drawAxis).observe(ax);
  }

  WALK.register('m04', { data: 'M04_DATA', mount });
})();
