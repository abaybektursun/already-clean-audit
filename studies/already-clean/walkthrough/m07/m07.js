// m07: ranking a layer's poles one checkpoint before an onset (PLAN_PUBLIC_2026-11-08.md §3a row M7).
// Builds inside #m07 from window.M07_DATA (m07/data.js): pick one of the 71 held-out onsets; the layer's ten
// highest-scoring poles at the checkpoint before the onset, and the rank of the cluster's own pole on a 1-to-22,016 axis;
// a toggle switches to the comparison cluster's own pole under the same score. Below, the share of the 71 onsets in the
// top ten against the random baseline and the threshold set in advance. The full list of 22,016 scores is not saved;
// only the top ten and each scored pole's own rank and score are.
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';

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
  function onWidth(host, draw) {
    let last = 0;
    const run = () => { const w = host.clientWidth; if (w && w !== last) { last = w; draw(w); } };
    new ResizeObserver(run).observe(host);
    run();
    return () => { last = 0; run(); };
  }
  function sideName(id) {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m07: unexpected pole id ' + id);
    return 'neuron ' + WALK.fmt.int(+m[2]) + ' of layer ' + +m[1] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' pole';
  }
  const poleName = (id) => id.replace(/^L0*(\d+)N(\d+)([+-])$/, 'L$1.N$2$3');
  // Labels in rows above (row < 0) and below (row > 0) an axis line at y0, each ending at its mark when the mark is
  // right of centre, joined to the mark by a hairline.
  function rowLabel(svg, W, y0, x, row, text, cls) {
    const rowH = 20;
    const ty = row < 0 ? y0 + row * rowH - 6 : y0 + 12 + row * rowH;
    const right = x > W * 0.5;
    s('line', { x1: x, x2: x, y1: y0, y2: row < 0 ? ty + 4 : ty - 12, class: 'm07-leader' }, svg);
    s('text', { x: x + (right ? 4 : -4), y: ty, 'text-anchor': right ? 'end' : 'start', class: 'm07-lab' + (cls ? ' ' + cls : '') }, svg, text);
  }

  function mount(root, D) {
    const f = WALK.fmt;
    const N = D.sides_per_layer;
    const TOP = D.top;
    const B = D.births;
    const hits = B.filter((b) => b.carrier.rank <= TOP).length;
    const withCg = B.filter((b) => b.comparison_group);
    const cgHits = withCg.filter((b) => b.comparison_group.rank <= TOP).length;
    if (B.length !== D.held_out_births || hits !== D.carrier_in_top_ten.recomputed || withCg.length !== D.with_comparison_group.recomputed ||
      cgHits !== D.comparison_group_in_top_ten.recomputed) {
      throw new Error('m07: counts computed here differ from data.js');
    }
    const STEPS = D.checkpoint_steps;
    if (STEPS.length !== 16) throw new Error('m07: expected 16 checkpoint steps');
    const at = (k) => 'step ' + f.int(STEPS[k]);
    root.replaceChildren();

    // ---------- the rule ----------
    const rule = h('section', 'm07-part', root);
    h('h3', null, rule, 'The score and the rank');
    const ul = h('ul', 'm07-rule', rule);
    h('li', null, ul, 'A cluster’s own pole is the pole whose top-50 activating tokens the cluster came from.');
    h('li', null, ul, 'For a cluster whose onset is at checkpoint k, we score every pole of the layer at checkpoint k − 1, the checkpoint before the onset.');
    h('li', null, ul, 'For each pole: v = the median of that pole’s activation in its own direction over the cluster’s tokens. Its score = the share of its own-sign tokens at k − 1 whose |activation| is below v, read from the pole’s ' +
      D.histogram.bins + '-bin histogram of log₁₀|activation| (' + f.int(D.histogram.log10_low) + ' to ' + f.int(D.histogram.log10_high) +
      '), with log-linear interpolation inside v’s bin.');
    h('li', null, ul, 'Rank = the number of the layer’s ' + f.int(N) + ' poles whose score is at least the pole’s own; ties count against it. In the top ten means rank ≤ ' + TOP + '.');
    h('li', null, ul, 'The comparison cluster: a random cluster drawn from another pole’s top-50 tokens, of the same size and layer, whose onset falls at the same checkpoint by the same rule. We rank its own pole under the same score, on that cluster’s tokens.');

    // ---------- one onset ----------
    const one = h('section', 'm07-part', root);
    h('h3', null, one, 'One onset');
    const ctrl = h('div', 'control-row m07-controls', one);
    const lab = h('label', 'control-label', ctrl, 'Onset');
    lab.htmlFor = 'm07-birth';
    const sel = h('select', 'm07-select', ctrl);
    sel.id = 'm07-birth';
    B.forEach((b, i) => {
      const [side, c] = b.birth.split(':c');
      const name = b.group_name.length > 44 ? b.group_name.slice(0, 43) + '…' : b.group_name;
      const o = h('option', null, sel, poleName(side) + ' cluster ' + c + ', rank ' + f.int(b.carrier.rank) + ': ' + name);
      o.value = i;
    });
    const tg = h('div', 'toggle m07-toggle', one);
    tg.setAttribute('role', 'group');
    tg.setAttribute('aria-label', 'Pole ranked');
    const bCar = h('button', null, tg, 'The cluster’s own pole');
    const bCg = h('button', null, tg, 'The comparison pole');
    bCar.type = bCg.type = 'button';
    const who = h('p', 'm07-text', one);
    who.setAttribute('aria-live', 'polite');
    const axis = h('div', 'm07-plot', one);
    axis.setAttribute('role', 'img');
    const listHead = h('p', 'm07-head', one);
    const list = h('ol', 'm07-list', one);

    let birth = B[0];
    let mode = 'carrier';
    function show() {
      const cg = birth.comparison_group;
      if (!cg && mode === 'comparison_group') mode = 'carrier';
      bCar.setAttribute('aria-pressed', String(mode === 'carrier'));
      bCg.setAttribute('aria-pressed', String(mode === 'comparison_group'));
      bCg.disabled = !cg;
      const [side, c] = birth.birth.split(':c');
      who.replaceChildren();
      h('b', 'm07-mono', who, poleName(side));
      who.append(' (' + sideName(side) + '), cluster ' + c + ', “' + birth.group_name + '”: ' + birth.group_size +
        ' tokens, onset at ' + at(birth.birth_snapshot) + ', scored at ' + at(birth.scored_at_snapshot) + '. ' +
        'The cluster’s own pole ranks ' + f.int(birth.carrier.rank) + ' of ' + f.int(N) + '. ' +
        (cg ? 'The comparison cluster’s own pole, ' + poleName(cg.side_id) + ', ranks ' + f.int(cg.rank) + '.'
          : 'We found no comparison cluster for this onset (' + (B.length - withCg.length) + ' of ' + B.length + ' onsets).'));
      const S = birth[mode];
      const key = mode === 'carrier' ? 'claim' : 'control';
      const whose = mode === 'carrier' ? 'the cluster’s own pole' : 'the comparison pole';
      listHead.textContent = 'The layer’s ten highest scores at ' + at(birth.scored_at_snapshot) + ', on the tokens of ' +
        (mode === 'carrier' ? 'the cluster' : 'the comparison cluster') + '; ' + whose + ' is marked';
      list.replaceChildren();
      const row = (rank, side, score, mine) => {
        const li = h('li', 'm07-row' + (mine ? ' m07-mine' : ''), list);
        h('span', 'm07-rank', li, f.int(rank));
        const sd = h('span', 'm07-side', li);
        if (mine) h('i', 'm07-key m07-key-' + key, sd);
        h('span', 'm07-mono', sd, poleName(side));
        if (mine) h('span', 'm07-tag', sd, mode === 'carrier' ? 'own pole' : 'comparison pole');
        h('span', 'm07-score', li, f.fixed(score, 6));
        WALK.hover(li, { title: poleName(side) + (mine ? ' (' + whose + ')' : ''), rows: [
          ['pole', sideName(side)], ['rank', f.int(rank) + ' of ' + f.int(N)], ['score', f.fixed(score, 9)]] });
        return li;
      };
      S.top_ten.forEach((t, i) => row(i + 1, t.side_id, t.score, t.side_id === S.side_id));
      if (S.rank > TOP) {
        const gap = h('li', 'm07-gap', list);
        gap.setAttribute('aria-hidden', 'true');
        gap.textContent = '⋮ ' + f.int(S.rank - TOP - 1) + ' poles between';
        row(S.rank, S.side_id, S.score, true);
      }
      axis.setAttribute('aria-label', 'Rank on a log scale from 1 to ' + f.int(N) + ': the cluster’s own pole ' + birth.carrier.rank +
        (cg ? ', the comparison pole ' + cg.rank : ''));
      redraw();
    }
    sel.addEventListener('change', () => { birth = B[+sel.value]; show(); });
    bCar.addEventListener('click', () => { mode = 'carrier'; show(); });
    bCg.addEventListener('click', () => { mode = 'comparison_group'; show(); });

    function drawAxis(W) {
      const padL = 10, padR = 18;
      const lx = (r) => padL + (Math.log10(r) / Math.log10(N)) * (W - padL - padR);
      const y0 = 2 * 20 + 16;
      const H = y0 + 2 * 20 + 40;
      axis.replaceChildren();
      const svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H }, axis);
      s('rect', { x: lx(1), y: y0 - 12, width: lx(TOP) - lx(1), height: 24, rx: 3, class: 'm07-top' }, svg);
      s('line', { x1: lx(1), x2: lx(N), y1: y0, y2: y0, class: 'm07-axis' }, svg);
      const T = [1, 10, 100, 1000, 10000, N];
      for (const t of T) {
        s('line', { x1: lx(t), x2: lx(t), y1: y0, y2: y0 + 5, class: 'm07-axis' }, svg);
        if (t !== 10000 || lx(N) - lx(10000) > 44) {
          s('text', { x: lx(t), y: H - 22, 'text-anchor': t === 1 ? 'start' : t === N ? 'end' : 'middle', class: 'm07-tick' }, svg, f.int(t));
        }
      }
      s('text', { x: lx(1), y: H - 4, class: 'm07-axt' }, svg, 'rank among the layer’s ' + f.int(N) + ' poles, log scale');
      s('text', { x: lx(TOP) + 4, y: y0 - 16, class: 'm07-axt' }, svg, 'top ten');
      const cg = birth.comparison_group;
      const car = birth.carrier;
      rowLabel(svg, W, y0, lx(car.rank), -2, 'own pole: rank ' + f.int(car.rank), mode === 'carrier' ? 'm07-on' : '');
      if (cg) rowLabel(svg, W, y0, lx(cg.rank), 1, 'comparison pole: rank ' + f.int(cg.rank), mode === 'comparison_group' ? 'm07-on' : '');
      if (cg) s('circle', { cx: lx(cg.rank), cy: y0, r: mode === 'comparison_group' ? 6 : 4.5, class: 'm07-dot-control' }, svg);
      s('circle', { cx: lx(car.rank), cy: y0, r: mode === 'carrier' ? 6 : 4.5, class: 'm07-dot-claim' }, svg);
    }
    const redraw = onWidth(axis, drawAxis);
    show();

    // ---------- the 71 onsets ----------
    const all = h('section', 'm07-part', root);
    h('h3', null, all, 'Across the ' + B.length + ' held-out onsets');
    h('p', 'm07-text', all, 'The share of onsets whose own pole is in the top ten. A pole picked at random is in the top ten with probability ' +
      TOP + ' / ' + f.int(N) + ' = ' + f.fixed(D.chance_level.recomputed, 5) + ', the random baseline. The threshold set in advance, ' +
      f.fixed(D.threshold_set_in_advance, 4) + ', is the share reached by poles of the same layer matched on activation size.');
    const sum = h('div', 'm07-plot', all);
    const pc = hits / B.length, pg = cgHits / withCg.length;
    sum.setAttribute('role', 'img');
    sum.setAttribute('aria-label', 'Share in the top ten on a log scale: own pole ' + hits + ' of ' + B.length + ', comparison pole ' +
      cgHits + ' of ' + withCg.length + ', random baseline ' + f.fixed(D.chance_level.recomputed, 5) + ', threshold ' + f.fixed(D.threshold_set_in_advance, 4));
    // one row per quantity on a shared log axis; the two reference values are drawn through every row
    onWidth(sum, (W) => {
      const padL = 10, padR = 14;
      const lo = -4;
      const lx = (p) => padL + ((Math.log10(p) - lo) / -lo) * (W - padL - padR);
      const ch = D.chance_level.recomputed, th = D.threshold_set_in_advance;
      const rows = [
        ['own pole in the top ten: ' + hits + ' of ' + B.length + ' = ' + f.fixed(pc, 3), pc, 'm07-dot-claim'],
        ['comparison pole in the top ten: ' + cgHits + ' of ' + withCg.length + ' = ' + f.fixed(pg, 3), pg, 'm07-dot-control'],
        ['threshold set in advance: ' + f.fixed(th, 4), th, 'ref'],
        ['random baseline: ' + TOP + ' / ' + f.int(N) + ' = ' + f.fixed(ch, 5), ch, 'ref'],
      ];
      const rowH = 44, top = 4;
      const yAxis = top + rows.length * rowH + 4;
      const H = yAxis + 44;
      sum.replaceChildren();
      const svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H }, sum);
      for (const t of [1e-4, 1e-3, 1e-2, 1e-1, 1]) {
        s('line', { x1: lx(t), x2: lx(t), y1: yAxis, y2: yAxis + 5, class: 'm07-axis' }, svg);
        s('text', { x: lx(t), y: yAxis + 18, 'text-anchor': t === 1e-4 ? 'start' : t === 1 ? 'end' : 'middle', class: 'm07-tick' }, svg,
          f.fixed(t, Math.max(0, -Math.log10(t))));
      }
      s('line', { x1: lx(1e-4), x2: lx(1), y1: yAxis, y2: yAxis, class: 'm07-axis' }, svg);
      s('text', { x: lx(1e-4), y: yAxis + 38, class: 'm07-axt' }, svg, 'share of onsets in the top ten, log scale');
      rows.forEach(([text, v, cls], i) => {
        const y = top + i * rowH;
        // gridlines and the two reference values run through each row's mark band, never through its text
        for (const t of [1e-4, 1e-3, 1e-2, 1e-1, 1]) s('line', { x1: lx(t), x2: lx(t), y1: y + 22, y2: y + rowH, class: 'm07-grid' }, svg);
        for (const r of [ch, th]) s('line', { x1: lx(r), x2: lx(r), y1: y + 22, y2: y + rowH, class: 'm07-ref' }, svg);
        s('text', { x: padL, y: y + 16, class: 'm07-lab' }, svg, text);
        s('line', { x1: lx(1e-4), x2: lx(1), y1: y + 32, y2: y + 32, class: 'm07-row-rule' }, svg);
        if (cls !== 'ref') s('circle', { cx: lx(v), cy: y + 32, r: 5, class: cls }, svg);
        else s('rect', { x: lx(v) - 2, y: y + 24, width: 4, height: 16, rx: 1, class: 'm07-ref-mark' }, svg);
      });
    });
  }

  WALK.register('m07', { data: 'M07_DATA', mount });
})();
