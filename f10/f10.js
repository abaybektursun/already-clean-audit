// f10: naming the carrier one snapshot before the birth (PLAN_PUBLIC_2026-11-08.md §4 row F10, chapter 5).
// Builds inside #f10 from window.F10_DATA (f10/data.js): across the 71 held-out births, the rank of the carrier among
// its layer's 22,016 sides one snapshot before the birth, beside the two comparison groups' own sides, as dots in
// half-decade rank bins with the top-ten line; the paired differences and the share over chance; and the ten banked
// raw examples (the group's contexts and the layer's ten highest-scoring sides with their strongest contexts).
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
  }
  function sideName(id) {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('f10: unexpected side id ' + id);
    return 'layer ' + +m[1] + ', unit ' + WALK.fmt.int(+m[2]) + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' side';
  }
  // A banked context: up to 160 characters before, the marked token between [[ and ]], then the text after.
  function splitContext(text) {
    const i = text.indexOf('[[');
    const j = text.indexOf(']]', i);
    if (i < 0 || j < 0) throw new Error('f10: a context has no marked token: ' + text.slice(0, 60));
    const nl = (t) => t.replace(/\n/g, '↵');
    return { before: nl(text.slice(0, i)), token: nl(text.slice(i + 2, j)), after: nl(text.slice(j + 2)) };
  }
  // One line per context: the text before (clipped at the left), the token, the text after (clipped at the right).
  function contextRow(parent, text, extra) {
    const c = splitContext(text);
    const row = h('div', 'f10-ctx' + (extra != null ? ' f10-ctx-a' : ''), parent);
    h('span', null, h('span', 'f10-b', row), c.before);
    h('span', 'f10-t', row, c.token);
    h('span', 'f10-a', row, c.after);
    if (extra != null) h('span', 'f10-v', row, extra);
    return row;
  }
  // One value box for a list of rows: the pointer, a tap, or the arrow keys once the list has focus.
  function listHover(list, rows, content, label) {
    let cur = -1;
    list.tabIndex = 0;
    list.setAttribute('role', 'group');
    list.setAttribute('aria-label', label);
    list.setAttribute('aria-describedby', 'walk-value-box');
    const mark = (i) => { rows.forEach((r, j) => r.classList.toggle('is-cur', j === i)); cur = i; };
    const idx = (e) => rows.indexOf(e.target.closest('.f10-ctx'));
    list.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const i = idx(e);
      if (i < 0) return;
      mark(i);
      WALK.valueBox.show(content(i), e.clientX, e.clientY);
    });
    list.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { mark(-1); WALK.valueBox.hide(); } });
    list.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const i = idx(e);
      if (i < 0) return;
      mark(i);
      WALK.valueBox.showFor(rows[i], content(i));
    });
    list.addEventListener('focus', () => { mark(cur < 0 ? 0 : cur); WALK.valueBox.showFor(rows[cur], content(cur)); });
    list.addEventListener('blur', () => { mark(-1); WALK.valueBox.hide(); });
    list.addEventListener('keydown', (e) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      mark(Math.max(0, Math.min(rows.length - 1, cur + step)));
      WALK.valueBox.showFor(rows[cur], content(cur));
    });
  }
  const marked = (text) => text.replace('[[', '⟦').replace(']]', '⟧');
  const short = (text) => {
    const c = splitContext(text);
    return '…' + c.before.slice(-26) + '⟦' + c.token + '⟧' + c.after.slice(0, 18) + '…';
  };

  // half-decade bins of rank: bin 0 holds ranks 1 to 3, then (10^(i/2), 10^((i+1)/2)]; the top-ten line is an edge
  const binOf = (r) => (r <= 1 ? 0 : Math.min(8, Math.ceil(2 * Math.log10(r)) - 1));
  const NBINS = 9;

  function mount(root, D) {
    const f = WALK.fmt;
    const N = D.sides_per_layer;
    const TOP = D.top;
    const S = D.summary.held_out;
    const births = D.births.filter((b) => b.half === 'held_out');
    const series = [
      { key: 'carrier', cls: 'claim', name: 'The carrier', legend: 'the carrier', row: 'carrier' },
      { key: 'comparison_group', cls: 'control', name: 'A random group of another side’s contexts, same size and birth snapshot: its own side',
        legend: 'the comparison group’s own side', row: 'comparison side' },
      { key: 'comparison_group_distinct_tokens', cls: 'control-ring', name: 'The same, also matched on the number of distinct tokens: its own side',
        legend: 'the distinct-token comparison group’s own side', row: 'distinct-token comparison side' },
    ];
    for (const sr of series) {
      sr.items = births.filter((b) => b[sr.key]).map((b) => ({ birth: b, side: b[sr.key] }));
      sr.items.sort((a, b) => a.side.rank - b.side.rank);
      sr.hits = sr.items.filter((x) => x.side.rank <= TOP).length;
    }
    const want = [[S.births, S.carrier_in_top_ten], [S.with_comparison_group, S.comparison_group_in_top_ten],
      [S.with_distinct_token_comparison_group, S.distinct_token_comparison_group_in_top_ten]];
    series.forEach((sr, i) => {
      if (sr.items.length !== want[i][0] || sr.hits !== want[i][1]) {
        throw new Error('f10: ' + sr.key + ' has ' + sr.hits + ' of ' + sr.items.length + ' here; data.js says ' + want[i][1] + ' of ' + want[i][0]);
      }
    });
    const raw = new Map(D.raw_examples.map((e) => [e.birth, e]));
    root.replaceChildren();

    // ---------- the ranks ----------
    const top = h('section', 'f10-part', root);
    h('h3', null, top, 'Rank one snapshot before the birth, ' + births.length + ' held-out births');
    h('p', 'f10-text', top, 'Each dot is one birth: the rank of a side among the ' + f.int(N) + ' sides of its layer, by its own percentile on the group’s positions at the snapshot before the birth. Rank 1 is the highest score. Carrier dots with a ring have raw examples below; tap or click one to open it there.');
    const legend = h('ul', 'f10-legend', top);
    series.forEach((sr) => {
      const li = h('li', null, legend);
      h('i', 'f10-key f10-key-' + sr.cls, li);
      li.append(sr.name + ': ' + sr.hits + ' of ' + sr.items.length + ' in the top ten');
    });
    const plot = h('div', 'f10-plot', top);
    plot.tabIndex = 0;
    plot.setAttribute('role', 'group');
    plot.setAttribute('aria-describedby', 'walk-value-box');
    plot.setAttribute('aria-label', 'Ranks of three series in half-decade bins; left and right arrows move through the dots of a row, up and down change the row');

    let marks = [];
    let cur = -1;
    const pitch = 11;
    function draw(W) {
      const padL = 8, padR = 16;
      const lx = (r) => padL + (Math.log10(r) / Math.log10(N)) * (W - padL - padR);
      const edge = (i) => (i >= NBINS ? N : 10 ** (i / 2));
      const sub = [];
      for (let i = 0; i < NBINS; i++) sub.push(Math.max(1, Math.floor((lx(edge(i + 1)) - lx(edge(i)) - 2) / pitch)));
      // rows of dots per series: the same height for all three, set by the fullest bin
      const counts = series.map((sr) => { const c = new Array(NBINS).fill(0); sr.items.forEach((x) => c[binOf(x.side.rank)]++); return c; });
      const rowsMax = Math.max(...counts.flatMap((c) => c.map((n, i) => Math.ceil(n / sub[i]))));
      const bandH = rowsMax * pitch + 6;
      const headH = 26;
      const y0 = 22;
      const rowTop = (k) => y0 + k * (headH + bandH);
      const yAxis = rowTop(series.length);
      const H = yAxis + 46;
      plot.replaceChildren();
      const svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H }, plot);
      s('line', { x1: lx(TOP), x2: lx(TOP), y1: y0 - 14, y2: y0 - 2, class: 'f10-topline' }, svg);
      s('text', { x: lx(TOP) + 5, y: y0 - 4, class: 'f10-lab' }, svg, 'top ten (rank ≤ ' + TOP + ')');
      marks = [];
      series.forEach((sr, k) => {
        const yt = rowTop(k);
        s('text', { x: padL, y: yt + 16, class: 'f10-lab' }, svg, sr.row + ': ' + sr.hits + ' of ' + sr.items.length);
        const base = yt + headH + bandH - 4;
        // the top-ten band, the bin edges and the top-ten line run through the row's dots, never through its text
        const yb = yt + headH - 2;
        s('rect', { x: lx(1), y: yb, width: lx(TOP) - lx(1), height: base + 5 - yb, class: 'f10-top' }, svg);
        for (let i = 0; i <= NBINS; i++) s('line', { x1: lx(edge(i)), x2: lx(edge(i)), y1: yb, y2: base + 5, class: 'f10-grid' }, svg);
        s('line', { x1: lx(TOP), x2: lx(TOP), y1: yb, y2: k === series.length - 1 ? yAxis : base + 5, class: 'f10-topline' }, svg);
        s('line', { x1: lx(1), x2: lx(N), y1: base + 5, y2: base + 5, class: 'f10-rule' }, svg);
        const filled = new Array(NBINS).fill(0);
        sr.items.forEach((x) => {
          const b = binOf(x.side.rank);
          const n = filled[b]++;
          const x0 = lx(edge(b)) + 1 + ((lx(edge(b + 1)) - lx(edge(b)) - 2) - sub[b] * pitch) / 2;
          const cx = x0 + (n % sub[b]) * pitch + pitch / 2;
          const cy = base - Math.floor(n / sub[b]) * pitch - pitch / 2 + 1;
          const isRaw = sr.key === 'carrier' && raw.has(x.birth.birth);
          if (isRaw) s('circle', { cx, cy, r: 6.5, class: 'f10-raw' }, svg);
          s('circle', { cx, cy, r: 4, class: 'f10-dot-' + sr.cls }, svg);
          marks.push({ x: cx, y: cy, row: k, sr, item: x, isRaw });
        });
      });
      for (const t of [1, 10, 100, 1000, 10000, N]) {
        s('line', { x1: lx(t), x2: lx(t), y1: yAxis, y2: yAxis + 5, class: 'f10-axis' }, svg);
        if (t !== 10000 || lx(N) - lx(10000) > 44) {
          s('text', { x: lx(t), y: yAxis + 18, 'text-anchor': t === 1 ? 'start' : t === N ? 'end' : 'middle', class: 'f10-tick' }, svg, f.int(t));
        }
      }
      s('line', { x1: lx(1), x2: lx(N), y1: yAxis, y2: yAxis, class: 'f10-axis' }, svg);
      s('text', { x: lx(1), y: yAxis + 38, class: 'f10-axt' }, svg, 'rank among the layer’s ' + f.int(N) + ' sides, log scale');
      s('circle', { class: 'f10-cur', r: 7.5, cx: -20, cy: -20 }, svg);
      if (cur >= 0) ring(cur);
    }

    function content(m) {
      const b = m.item.birth;
      const [side, c] = b.birth.split(':c');
      const rows = [['birth', sideName(side) + ', group ' + c], ['group', b.group_name],
        ['contexts', b.group_size + ', ' + b.distinct_tokens + ' distinct tokens'],
        ['born at snapshot', String(b.birth_snapshot)], ['the carrier’s rank', f.int(b.carrier.rank) + ' of ' + f.int(N), 'claim']];
      if (m.sr.key !== 'carrier') {
        rows.push([m.sr.legend, m.item.side.side_id + ', rank ' + f.int(m.item.side.rank), 'control']);
        if (m.item.side.distinct_tokens != null) rows.push(['its distinct tokens', String(m.item.side.distinct_tokens)]);
      }
      const e = raw.get(b.birth);
      if (e) {
        e.group_contexts.slice(0, 4).forEach((g) => rows.push([short(g.text), '']));
        if (e.group_contexts.length > 4) rows.push(['and ' + (e.group_contexts.length - 4) + ' more contexts in the raw examples below', '']);
      }
      return { title: side + ', group ' + c + (m.sr.key === 'carrier' ? '' : ': ' + m.sr.legend), rows };
    }
    function ring(i) {
      const g = plot.querySelector('.f10-cur');
      if (g) { g.setAttribute('cx', i < 0 ? -20 : marks[i].x); g.setAttribute('cy', i < 0 ? -20 : marks[i].y); }
    }
    function at(i) {
      cur = i;
      ring(i);
      const r = plot.getBoundingClientRect();
      return [r.left + marks[i].x, r.top + marks[i].y - 6];
    }
    const nearest = (e) => {
      const r = plot.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = -1, d = 14 * 14;
      marks.forEach((m, i) => { const dd = (m.x - px) ** 2 + (m.y - py) ** 2; if (dd < d) { d = dd; best = i; } });
      return best;
    };
    const off = () => { cur = -1; ring(-1); WALK.valueBox.hide(); };
    plot.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const i = nearest(e);
      if (i < 0) { off(); return; }
      at(i);
      WALK.valueBox.show(content(marks[i]), e.clientX, e.clientY);
    });
    plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') off(); });
    plot.addEventListener('pointerdown', (e) => {
      const i = nearest(e);
      if (i < 0) return;
      if (marks[i].isRaw) pick(marks[i].item.birth.birth);
      if (e.pointerType === 'mouse') return;
      const [x, y] = at(i);
      WALK.valueBox.show(content(marks[i]), x, y);
    });
    plot.addEventListener('focus', () => { const [x, y] = at(cur < 0 ? 0 : cur); WALK.valueBox.show(content(marks[cur]), x, y); });
    plot.addEventListener('blur', off);
    plot.addEventListener('keydown', (e) => {
      if (cur < 0) return;
      let next = -1;
      const m = marks[cur];
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const same = marks.map((x, i) => i).filter((i) => marks[i].row === m.row);
        const j = same.indexOf(cur) + (e.key === 'ArrowRight' ? 1 : -1);
        if (j >= 0 && j < same.length) next = same[j];
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        const row = m.row + (e.key === 'ArrowDown' ? 1 : -1);
        let d = Infinity;
        marks.forEach((x, i) => { if (x.row === row && Math.abs(x.x - m.x) < d) { d = Math.abs(x.x - m.x); next = i; } });
      } else if (e.key === 'Enter' && m.isRaw) {
        pick(m.item.birth.birth);
        return;
      } else return;
      e.preventDefault();
      if (next < 0) return;
      const [x, y] = at(next);
      WALK.valueBox.show(content(marks[next]), x, y);
    });
    onWidth(plot, draw);

    // ---------- the differences ----------
    const B = D.banked;
    const diff = h('section', 'f10-part', root);
    h('h3', null, diff, 'The differences, paired by birth');
    const ratio = h('p', 'f10-text', diff);
    ratio.textContent = 'The carrier is in the top ten in ' + S.carrier_in_top_ten + ' of ' + S.births + ' births, ' +
      f.fixed(S.carrier_in_top_ten / S.births, 3) + ' (range consistent with the data ' + f.fixed(B.carrier_share_range[0], 3) + ' to ' +
      f.fixed(B.carrier_share_range[1], 3) + '). A side picked at random is in the top ten with probability ' + TOP + ' / ' + f.int(N) + ' = ' +
      f.fixed(B.chance_level, 5) + ', so the carrier is named at ' + f.fixed(B.carrier_share_over_chance, 1) +
      ' times chance. The threshold set in advance was ' + f.fixed(B.threshold_set_in_advance, 4) + '.';
    const near = (a, b, what) => { if (Math.abs(a - b) > 0.05) throw new Error('f10: ' + what + ' ' + a + ' against ' + b); };
    near((S.carrier_in_top_ten / S.births) / B.chance_level, B.carrier_share_over_chance, 'share over chance');
    const P = D.post_hoc_split;
    const rows = [
      { label: 'Carrier minus the comparison group’s own side, ' + B.carrier_minus_comparison_group.n + ' births',
        sub: 'in the top ten only for the carrier: ' + B.carrier_minus_comparison_group.x_only + '; only for the comparison side: ' +
          B.carrier_minus_comparison_group.y_only, d: B.carrier_minus_comparison_group },
      { label: 'Carrier minus the distinct-token comparison group’s own side, ' + B.carrier_minus_distinct_token_comparison_group.n + ' births',
        sub: 'only for the carrier: ' + B.carrier_minus_distinct_token_comparison_group.x_only + '; only for the comparison side: ' +
          B.carrier_minus_distinct_token_comparison_group.y_only, d: B.carrier_minus_distinct_token_comparison_group },
      { head: 'Found after the data were seen, both halves, births with a comparison group' },
      { label: 'Contexts that repeat one token (distinct tokens under two thirds of the group), ' + P.repeated_token.births.recomputed + ' births',
        sub: 'carrier ' + P.repeated_token.carrier_in_top_ten.recomputed + ', comparison side ' + P.repeated_token.comparison_group_in_top_ten.recomputed +
          ' in the top ten', d: { diff: P.repeated_token.difference, lo: P.repeated_token.difference_range[0], hi: P.repeated_token.difference_range[1] } },
      { label: 'Contexts with varied tokens, ' + P.varied_token.births.recomputed + ' births',
        sub: 'carrier ' + P.varied_token.carrier_in_top_ten.recomputed + ', comparison side ' + P.varied_token.comparison_group_in_top_ten.recomputed +
          ' in the top ten', d: { diff: P.varied_token.difference, lo: P.varied_token.difference_range[0], hi: P.varied_token.difference_range[1] } },
    ];
    // the paired differences must be the counts' differences
    near(B.carrier_minus_comparison_group.diff, (B.carrier_minus_comparison_group.x_only - B.carrier_minus_comparison_group.y_only) / B.carrier_minus_comparison_group.n, 'difference');
    const forest = h('div', 'f10-forest', diff);
    const signed = (x) => (x > 0 ? '+' : '') + f.fixed(x, 3);
    const lo = -0.6, hi = 0.2;
    const ticksF = [-0.6, -0.4, -0.2, 0, 0.2];
    for (const r of rows) {
      if (r.head) { h('p', 'f10-forest-head', forest, r.head); continue; }
      const row = h('div', 'f10-frow', forest);
      const t = h('p', 'f10-flabel', row);
      t.append(r.label + ': ');
      h('b', null, t, signed(r.d.diff));
      t.append(' (' + signed(r.d.lo) + ' to ' + signed(r.d.hi) + ')');
      h('p', 'f10-fsub', row, r.sub);
      const track = h('div', 'f10-track', row);
      const pos = (v) => (100 * (v - lo)) / (hi - lo) + '%';
      const zero = h('span', 'f10-zero', track);
      zero.style.left = pos(0);
      const band = h('span', 'f10-band', track);
      band.style.left = pos(r.d.lo);
      band.style.width = (100 * (r.d.hi - r.d.lo)) / (hi - lo) + '%';
      const dot = h('span', 'f10-fdot', track);
      dot.style.left = pos(r.d.diff);
      if (r.d.lo < lo || r.d.hi > hi) throw new Error('f10: a range leaves the axis: ' + r.d.lo + ' to ' + r.d.hi);
      WALK.hover(row, { title: r.label, rows: [['difference in the share in the top ten', signed(r.d.diff)],
        ['range consistent with the data', signed(r.d.lo) + ' to ' + signed(r.d.hi)], [r.sub, '']] });
    }
    const axisF = h('div', 'f10-faxis', forest);
    for (const v of ticksF) {
      const tk = h('span', 'f10-ftick', axisF, v === 0 ? '0' : (v > 0 ? '+' : '') + f.fixed(v, 1));
      tk.style.left = (100 * (v - lo)) / (hi - lo) + '%';
      if (v === lo) tk.style.transform = 'none';
      if (v === hi) tk.style.transform = 'translateX(-100%)';
    }
    h('p', 'f10-fcap', forest, 'difference in the share of births in the top ten, carrier minus comparison');

    // ---------- the raw examples ----------
    const rx = h('section', 'f10-part', root);
    h('h3', null, rx, 'The raw examples');
    h('p', 'f10-text', rx, 'Ten held-out births have their texts saved in the record: the group’s contexts at the end of training, and the layer’s ten highest-scoring sides at the snapshot before the birth, each with its own three strongest contexts and activations at that snapshot.');
    const ctl = h('div', 'control-row f10-controls', rx);
    const lab = h('label', 'control-label', ctl, 'Raw example');
    lab.htmlFor = 'f10-raw';
    const sel = h('select', 'f10-select', ctl);
    sel.id = 'f10-raw';
    D.raw_examples.forEach((e, i) => {
      const b = births.find((x) => x.birth === e.birth);
      if (!b || b.carrier.rank !== e.carrier_rank) throw new Error('f10: raw example ' + e.birth + ' does not match its birth');
      const [side, c] = e.birth.split(':c');
      const o = h('option', null, sel, side + ' group ' + c + ', carrier rank ' + f.int(e.carrier_rank) + ': ' +
        (b.group_name.length > 40 ? b.group_name.slice(0, 39) + '…' : b.group_name));
      o.value = i;
    });
    const body = h('div', null, rx);
    function showRaw(i) {
      const e = D.raw_examples[i];
      const b = births.find((x) => x.birth === e.birth);
      const [side, c] = e.birth.split(':c');
      body.replaceChildren();
      const who = h('p', 'f10-text', body);
      h('b', 'f10-mono', who, side);
      who.append(' (' + sideName(side) + '), group ' + c + ', “' + b.group_name + '”, born at snapshot ' + b.birth_snapshot +
        '. Scored at snapshot ' + e.scored_at_snapshot + ': the carrier’s rank ' + f.int(e.carrier_rank) + ', score ' + f.fixed(e.carrier_score, 6) + '.');
      h('p', 'f10-head', body, 'The group’s contexts (' + e.group_contexts.length + ')');
      const gl = h('div', 'f10-list', body);
      const gr = e.group_contexts.map((g) => contextRow(gl, g.text));
      listHover(gl, gr, (k) => ({ title: 'Rank ' + (e.group_contexts[k].rank_in_fifty + 1) + ' of the side’s fifty',
        rows: [[marked(e.group_contexts[k].text), '']] }), 'The group’s contexts; arrow keys show each in full');
      const det = h('details', 'f10-details', body);
      h('summary', null, det, 'The layer’s ten highest-scoring sides at snapshot ' + e.scored_at_snapshot + ', with their own strongest contexts');
      e.top_ten.forEach((t, k) => {
        const blk = h('div', 'f10-side', det);
        const hd = h('p', 'f10-side-head', blk);
        hd.append((k + 1) + '. ');
        if (t.side_id === e.carrier_side) h('i', 'f10-key f10-key-claim', hd);
        h('span', 'f10-mono', hd, t.side_id);
        hd.append((t.side_id === e.carrier_side ? ' (the carrier)' : '') + ', score ' + f.fixed(t.score, 6));
        const l = h('div', 'f10-list', blk);
        const sr = t.strongest_contexts.map((sc) => contextRow(l, sc.text, f.fixed(sc.a, 3)));
        listHover(l, sr, (j) => {
          const sc = t.strongest_contexts[j];
          return { title: t.side_id + ', its context ' + (sc.rank + 1) + ' at snapshot ' + e.scored_at_snapshot,
            rows: [['activation', f.fixed(sc.a, 4)], [marked(sc.text), '']] };
        }, t.side_id + ': its three strongest contexts; arrow keys show each in full');
      });
    }
    function pick(id) {
      const i = D.raw_examples.findIndex((e) => e.birth === id);
      sel.value = String(i);
      showRaw(i);
    }
    sel.addEventListener('change', () => showRaw(+sel.value));
    showRaw(0);
  }

  WALK.register('f10', { data: 'F10_DATA', mount });
})();
