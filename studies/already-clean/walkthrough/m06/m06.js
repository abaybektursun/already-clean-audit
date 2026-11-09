// m06: births that beat chance, and the split (PLAN_PUBLIC_2026-11-08.md §3a row M6).
// Builds inside #m06 from window.M06_DATA (m06/data.js): for each of the two banked example groups, the side's fifty
// with the group's positions and the birth snapshots of 2,000 random groups of the same size; below, every tested
// group's chance share against the 0.05 line, and the split of the 153 passing births by unit index.
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
  const SNAPSHOTS = 16; // snapshots 0 to 15

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

  // One value box for a row of columns: the pointer picks the column under it, a tap shows it above the column,
  // and the arrow keys move through the columns once the plot has focus. cols: [{x0, x1, ax, ay}] in host pixels.
  function columnHover(host, getCols, content, label, onCur) {
    let cur = -1;
    host.tabIndex = 0;
    host.setAttribute('role', 'group');
    host.setAttribute('aria-label', label);
    host.setAttribute('aria-describedby', 'walk-value-box');
    const ring = () => host.querySelector('.m06-cur');
    const at = (i) => {
      cur = i;
      const c = getCols()[i];
      const r = host.getBoundingClientRect();
      const g = ring();
      if (g) { g.setAttribute('x', c.x0); g.setAttribute('width', c.x1 - c.x0); g.style.display = ''; }
      if (onCur) onCur(i);
      return [r.left + c.ax, r.top + c.ay];
    };
    const pick = (e) => {
      const x = e.clientX - host.getBoundingClientRect().left;
      return getCols().findIndex((c) => x >= c.x0 && x < c.x1);
    };
    const off = () => {
      cur = -1;
      const g = ring();
      if (g) g.style.display = 'none';
      if (onCur) onCur(-1);
      WALK.valueBox.hide();
    };
    host.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const i = pick(e);
      if (i < 0) { off(); return; }
      at(i);
      WALK.valueBox.show(content(i), e.clientX, e.clientY);
    });
    host.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') off(); });
    host.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const i = pick(e);
      if (i < 0) return;
      const [x, y] = at(i);
      WALK.valueBox.show(content(i), x, y);
    });
    host.addEventListener('focus', () => { const [x, y] = at(cur < 0 ? 0 : cur); WALK.valueBox.show(content(cur), x, y); });
    host.addEventListener('blur', off);
    host.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      const [x, y] = at(Math.max(0, Math.min(getCols().length - 1, cur + step)));
      WALK.valueBox.show(content(cur), x, y);
    });
  }

  // Round tick values for a count axis that reaches max.
  function ticks(max) {
    const raw = max / 4;
    const p = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((v) => v >= raw);
    const out = [];
    for (let v = 0; v <= max; v += step) out.push(v);
    return out;
  }

  function sideName(id) {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m06: unexpected side id ' + id);
    return 'layer ' + +m[1] + ', unit ' + WALK.fmt.int(+m[2]) + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' side';
  }

  // Draws on resize of the host's width only, so drawing never feeds back into the observer.
  function onWidth(host, draw) {
    let last = 0;
    const run = () => { const w = host.clientWidth; if (w && w !== last) { last = w; draw(w); } };
    new ResizeObserver(run).observe(host);
    run();
  }

  function mount(root, D) {
    const f = WALK.fmt;
    const R = D.rule;
    const thr = R.threshold;
    const P = D.all_groups.share_random_born_same_or_later;
    const passing = P.filter((p) => p <= thr).length;
    if (passing !== D.counts.passing.recomputed || P.length !== D.counts.groups_tested.recomputed) {
      throw new Error('m06: ' + passing + ' of ' + P.length + ' pass here; data.js says ' + D.counts.passing.recomputed +
        ' of ' + D.counts.groups_tested.recomputed);
    }
    root.replaceChildren();

    // ---------- one group against its random groups ----------
    const one = h('section', 'm06-part', root);
    h('h3', null, one, 'One group against ' + f.int(R.random_groups_per_group) + ' random groups of the same size');
    const tg = h('div', 'toggle m06-toggle', one);
    tg.setAttribute('role', 'group');
    tg.setAttribute('aria-label', 'Example group');
    const NAMES = { beyond_chance_late: 'Born later than chance', at_chance: 'Born at chance' };
    const buttons = D.examples.map((ex, i) => {
      if (!NAMES[ex.example]) throw new Error('m06: unknown example ' + ex.example);
      const b = h('button', null, tg, NAMES[ex.example]);
      b.type = 'button';
      b.addEventListener('click', () => show(i));
      return b;
    });
    const who = h('p', 'm06-text', one);
    who.setAttribute('aria-live', 'polite');

    h('p', 'm06-head', one, 'The side’s fifty contexts, strongest first; the group’s positions are filled');
    const strip = h('div', 'm06-strip', one);
    const stripAxis = h('div', 'm06-strip-axis', one);
    h('span', null, stripAxis, 'rank 1');
    h('span', null, stripAxis, 'rank 50');
    const groups = h('ul', 'm06-groups', one);

    h('p', 'm06-head', one, 'Birth snapshot of each random group');
    const hist = h('div', 'm06-plot', one);
    const res = h('p', 'm06-text m06-result', one);
    res.setAttribute('aria-live', 'polite');

    let ex = null;
    function show(i) {
      ex = D.examples[i];
      buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
      who.replaceChildren();
      h('b', 'm06-mono', who, ex.id);
      who.append(' (' + sideName(ex.id) + '). Group ' + ex.group + ', “' + ex.group_name + '”: ' + ex.group_size +
        ' of the fifty, born at snapshot ' + ex.birth_snapshot + '.');
      // the fifty
      strip.replaceChildren();
      const inGroup = new Set(ex.ranks_in_fifty);
      const sizes = ex.groups_of_side.reduce((a, g) => a + g.size, 0);
      if (sizes !== 50) throw new Error('m06: the side’s groups hold ' + sizes + ' contexts, not 50');
      const cells = [];
      for (let r = 0; r < 50; r++) {
        const c = h('span', 'm06-cell' + (inGroup.has(r) ? ' m06-in' : ''), strip);
        cells.push(c);
      }
      strip.setAttribute('aria-label', 'Ranks ' + ex.ranks_in_fifty.map((r) => r + 1).join(', ') + ' of 50 are in group ' + ex.group +
        '; arrow keys move through the ranks');
      stripHover.cells = cells;
      groups.replaceChildren();
      for (const g of ex.groups_of_side) {
        const li = h('li', g.group === ex.group ? 'm06-this' : null, groups);
        h('i', 'm06-key ' + (g.group === ex.group ? 'm06-key-claim' : 'm06-key-bar'), li);
        li.append('group ' + g.group + ': ' + g.size + ' contexts, born at snapshot ' + g.birth_snapshot +
          (g.main ? ', the main group' : '') + (g.group === ex.group ? ' (this group)' : ''));
      }
      // the random groups
      const counts = new Array(SNAPSHOTS).fill(0);
      let total = 0;
      for (const [k, n] of Object.entries(ex.random_group_births_fresh_draw)) { counts[+k] = n; total += n; }
      if (total !== R.random_groups_per_group) throw new Error('m06: the histogram holds ' + total + ' random groups');
      const later = counts.slice(ex.birth_snapshot).reduce((a, b) => a + b, 0);
      if (Math.abs(later / total - ex.share_random_born_same_or_later_fresh_draw) > 1e-9) {
        throw new Error('m06: share ' + later / total + ' differs from data.js ' + ex.share_random_born_same_or_later_fresh_draw);
      }
      ex.counts = counts;
      ex.later = later;
      res.textContent = 'The chance share in this draw: ' + f.int(later) + ' of ' + f.int(total) + ' random groups born at snapshot ' +
        ex.birth_snapshot + ' or later, ' + f.fixed(later / total, 4) + '. The histogram is a second draw of ' + f.int(total) +
        '; the main run gave this group a chance share of ' + f.fixed(ex.share_random_born_same_or_later, 3) + '. Median birth of the random groups: snapshot ' +
        ex.random_group_median_birth + '. ' + (ex.passes ? 'The birth passes (chance share ≤ ' + f.fixed(thr, 2) + ').'
          : 'The birth does not pass (chance share above ' + f.fixed(thr, 2) + ').');
      drawHist(hist.clientWidth);
    }

    // hover on the fifty: one listener, the cell under the pointer
    const stripHover = { cells: [] };
    columnHover(strip, () => {
      const r0 = strip.getBoundingClientRect();
      return stripHover.cells.map((c) => {
        const r = c.getBoundingClientRect();
        return { x0: r.left - r0.left - 1, x1: r.right - r0.left + 1, ax: r.left - r0.left + r.width / 2, ay: 0 };
      });
    }, (i) => ({ title: 'Rank ' + (i + 1) + ' of 50', rows: [[ex.ranks_in_fifty.includes(i) ? 'in group ' + ex.group + ' (this group)'
      : 'in another group of this side', '', ex.ranks_in_fifty.includes(i) ? 'claim' : 'muted']] }),
    'The fifty contexts; arrow keys move through the ranks',
    (i) => stripHover.cells.forEach((c, j) => c.classList.toggle('is-cur', i === j)));

    let histCols = [];
    function drawHist(W) {
      if (!W || !ex) return;
      const padL = 40, padR = 6, top = 40, ph = 150, bot = 40;
      const cw = (W - padL - padR) / SNAPSHOTS;
      const max = Math.max(...ex.counts);
      const T = ticks(max);
      const ymax = Math.max(max, T[T.length - 1]);
      const y = (v) => top + ph - (v / ymax) * ph;
      hist.replaceChildren();
      const svg = s('svg', { width: W, height: top + ph + bot, viewBox: '0 0 ' + W + ' ' + (top + ph + bot) }, hist);
      s('rect', { x: padL + ex.birth_snapshot * cw, y: top - 4, width: (SNAPSHOTS - ex.birth_snapshot) * cw, height: ph + 4, class: 'm06-band' }, svg);
      for (const v of T) {
        s('line', { x1: padL, x2: W - padR, y1: y(v), y2: y(v), class: v ? 'm06-grid' : 'm06-axis' }, svg);
        s('text', { x: padL - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'm06-tick' }, svg, f.int(v));
      }
      s('rect', { class: 'm06-cur', x: 0, y: top - 4, width: 0, height: ph + 8, style: 'display:none' }, svg);
      const bw = Math.min(24, cw - 2);
      histCols = [];
      for (let k = 0; k < SNAPSHOTS; k++) {
        const cx = padL + (k + 0.5) * cw;
        const n = ex.counts[k];
        if (n) {
          const hgt = Math.max(1, y(0) - y(n));
          s('path', { d: barPath(cx - bw / 2, y(0) - hgt, bw, hgt), class: 'm06-bar-control' }, svg);
        }
        // counts of the counted columns are printed where a column is wide enough; the hover and the text below carry them
        if (k >= ex.birth_snapshot && cw >= 30) s('text', { x: cx, y: y(n) - 5, 'text-anchor': 'middle', class: 'm06-val' }, svg, f.int(n));
        // on a narrow plot every other snapshot is numbered, and always the birth
        const b = ex.birth_snapshot;
        if (cw >= 22 || k === b || (k % 2 === 0 && Math.abs(k - b) !== 1)) {
          s('text', { x: cx, y: top + ph + 16, 'text-anchor': 'middle', class: 'm06-tick' + (k === b ? ' m06-tick-cur' : '') }, svg, String(k));
        }
        histCols.push({ x0: padL + k * cw, x1: padL + (k + 1) * cw, ax: cx, ay: y(n) });
      }
      // the birth: a line at the left edge of the birth snapshot's column, where the counted region starts
      const bx = padL + ex.birth_snapshot * cw;
      s('line', { x1: bx, x2: bx, y1: top - 10, y2: top + ph, class: 'm06-birth' }, svg);
      const right = bx > W * 0.6;
      s('text', { x: bx + (right ? -6 : 6), y: top - 14, 'text-anchor': right ? 'end' : 'start', class: 'm06-lab' }, svg,
        'this group: born at snapshot ' + ex.birth_snapshot);
      s('text', { x: padL, y: top + ph + 34, class: 'm06-axt' }, svg, 'snapshot');
    }
    columnHover(hist, () => histCols, (k) => ({
      title: 'Snapshot ' + k + (k >= ex.birth_snapshot ? ': counted in the chance share' : ''),
      rows: [['random groups born here', f.int(ex.counts[k]) + ' of ' + f.int(R.random_groups_per_group), 'control']],
    }), 'Birth snapshots of the random groups; arrow keys move through the snapshots');
    onWidth(hist, drawHist);
    show(0);

    // ---------- every group ----------
    const all = h('section', 'm06-part', root);
    h('h3', null, all, 'Every tested group’s chance share');
    const allText = h('p', 'm06-text', all);
    allText.textContent = 'The test runs on every group of the ' + f.int(R.sides_with_two_or_more_groups) +
      ' sides with two or more groups: ' + f.int(P.length) + ' groups. ' + f.int(passing) + ' have a chance share ≤ ' +
      f.fixed(thr, 2) + ', against ' + f.fixed(D.counts.expected_at_chance.recomputed, 1) +
      ' expected at chance (the sum over groups of each group’s chance of passing).';
    const BINS = 20;
    const width = 1 / BINS; // bins of 0.05: [0, 0.05], then (0.05, 0.10], ...
    const bins = Array.from({ length: BINS }, () => []);
    // the pass rule decides the first bin; the other edges read the stored value to nine decimals
    P.forEach((p) => {
      const k = p <= thr ? 0 : Math.max(1, Math.min(BINS - 1, Math.ceil(+(p / width).toFixed(9)) - 1));
      bins[k].push(p);
    });
    if (bins[0].length !== passing) throw new Error('m06: the first bin holds ' + bins[0].length);
    const allPlot = h('div', 'm06-plot', all);
    let allCols = [];
    function drawAll(W) {
      const padL = 40, padR = 6, top = 22, ph = 170, bot = 46;
      const cw = (W - padL - padR) / BINS;
      const max = Math.max(...bins.map((b) => b.length));
      const T = ticks(max);
      const ymax = Math.max(max, T[T.length - 1]);
      const y = (v) => top + ph - (v / ymax) * ph;
      const x = (p) => padL + (p / width) * cw;
      allPlot.replaceChildren();
      const svg = s('svg', { width: W, height: top + ph + bot, viewBox: '0 0 ' + W + ' ' + (top + ph + bot) }, allPlot);
      for (const v of T) {
        s('line', { x1: padL, x2: W - padR, y1: y(v), y2: y(v), class: v ? 'm06-grid' : 'm06-axis' }, svg);
        s('text', { x: padL - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'm06-tick' }, svg, f.int(v));
      }
      s('rect', { class: 'm06-cur', x: 0, y: top - 4, width: 0, height: ph + 8, style: 'display:none' }, svg);
      const bw = Math.min(24, cw - 2);
      allCols = [];
      bins.forEach((b, k) => {
        const cx = padL + (k + 0.5) * cw;
        const hgt = y(0) - y(b.length);
        if (b.length) s('path', { d: barPath(cx - bw / 2, y(b.length), bw, hgt), class: k === 0 ? 'm06-bar-claim' : 'm06-bar' }, svg);
        allCols.push({ x0: padL + k * cw, x1: padL + (k + 1) * cw, ax: cx, ay: y(b.length) });
      });
      // chance level for the first bin, and the 0.05 line
      const e = D.counts.expected_at_chance.recomputed;
      s('line', { x1: padL + 0.5 * cw - bw / 2 - 3, x2: padL + 0.5 * cw + bw / 2 + 3, y1: y(e), y2: y(e), class: 'm06-chance' }, svg);
      s('line', { x1: x(thr), x2: x(thr), y1: top - 12, y2: top + ph, class: 'm06-thr' }, svg);
      s('text', { x: x(thr) + 4, y: top - 4, class: 'm06-tick' }, svg, f.fixed(thr, 2));
      for (const v of [0, 0.25, 0.5, 0.75, 1]) {
        s('text', { x: x(v), y: top + ph + 16, 'text-anchor': v === 0 ? 'start' : v === 1 ? 'end' : 'middle', class: 'm06-tick' }, svg, f.fixed(v, 2));
      }
      s('text', { x: padL, y: top + ph + 36, class: 'm06-axt' }, svg, 'chance share');
      // the legend, in the open space above the short middle bars
      const lx = padL + 1.5 * cw, ly = top + 20;
      s('rect', { x: lx, y: ly - 10, width: 10, height: 10, rx: 2, class: 'm06-bar-claim' }, svg);
      s('text', { x: lx + 16, y: ly, class: 'm06-lab' }, svg, f.int(passing) + ' pass (chance share ≤ ' + f.fixed(thr, 2) + ')');
      s('line', { x1: lx, x2: lx + 10, y1: ly + 17, y2: ly + 17, class: 'm06-chance' }, svg);
      s('text', { x: lx + 16, y: ly + 22, class: 'm06-lab' }, svg, f.fixed(e, 1) + ' expected at chance');
    }
    columnHover(allPlot, () => allCols, (k) => {
      const b = bins[k];
      const lo = k * width, hi = (k + 1) * width;
      const rows = [['groups', f.int(b.length) + ' of ' + f.int(P.length), k === 0 ? 'claim' : 'muted']];
      if (k === 0) {
        rows.push(['at exactly 0', f.int(b.filter((p) => p === 0).length)]);
        rows.push(['expected at chance', f.fixed(D.counts.expected_at_chance.recomputed, 1), 'control']);
      }
      const edge = b.filter((p) => p > thr && f.fixed(p, 3) === f.fixed(thr, 3));
      edge.forEach((p) => rows.push(['stored as ' + String(p) + ', above ' + f.fixed(thr, 2) + ': does not pass', '']));
      return { title: k === 0 ? 'chance share at most ' + f.fixed(thr, 2) + ': passes'
        : 'chance share above ' + f.fixed(lo, 2) + ', at most ' + f.fixed(hi, 2), rows };
    }, 'Every tested group’s chance share in bins of 0.05; arrow keys move through the bins');
    onWidth(allPlot, drawAll);

    // ---------- the split ----------
    const sp = h('section', 'm06-part', root);
    h('h3', null, sp, 'The ' + f.int(passing) + ' passing births, split by unit index');
    h('p', 'm06-text', sp, 'The rules of the later studies were written from the first half before the second half was read. ' +
      'Every carrier in the later chapters is one of these births.');
    const S = D.split;
    const halves = [
      ['Even unit index: the half that set the rules', S.rule_setting.recomputed, S.by_birth_snapshot.rule_setting],
      ['Odd unit index: the held-out half', S.held_out.recomputed, S.by_birth_snapshot.held_out],
    ];
    if (halves[0][1] + halves[1][1] !== passing) throw new Error('m06: the halves do not add up to ' + passing);
    const rows = h('div', 'm06-split', sp);
    for (const [name, n, byK] of halves) {
      const sum = Object.values(byK).reduce((a, b) => a + b, 0);
      if (sum !== n) throw new Error('m06: ' + name + ' by snapshot adds to ' + sum);
      const row = h('div', 'm06-split-row', rows);
      h('span', 'm06-split-name', row, name);
      const line = h('span', 'm06-split-line', row);
      const bar = h('span', 'm06-split-bar', line);
      bar.style.width = 'calc((100% - 4ch) * ' + n / passing + ')';
      h('span', 'm06-split-n', line, f.int(n));
      WALK.hover(row, { title: name + ': ' + n + ' births, by birth snapshot',
        rows: Object.entries(byK).map(([k, c]) => ['snapshot ' + k, f.int(c)]) });
    }
  }

  // A bar with a 4px rounded top and a square base.
  function barPath(x, y, w, hgt) {
    const r = Math.min(4, w / 2, hgt);
    return 'M' + x + ',' + (y + hgt) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + w - r) +
      'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hgt) + 'Z';
  }

  WALK.register('m06', { data: 'M06_DATA', mount });
})();
