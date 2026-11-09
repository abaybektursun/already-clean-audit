// f08: each condition's score on 500 competition problems. Registers the mount "f08" with site.js; reads
// window.F08_DATA (f08/data.js). Every banked condition (31) as a bar on a 0-to-100 axis with its range, the base model
// and the mathematics model as reference lines, the equal-count pair marked, and each condition's right or wrong
// answer on problems 1 to 3. A menu, a click or the arrow keys pick a condition; the panel shows what it copies and
// its three answers in full. Below, the registered paired differences, with the held-out word problems where banked.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const FAMILIES = ['reference', 'most-changed units', 'random units of equal energy', 'random sample of MLP entries',
    'half of the network', 'mathematics model minus the most-changed units', 'repeat reading'];
  const FROM = { base: 'the base model', math: 'the mathematics model' };
  const nb = (t) => t.replace(/ %/g, '\u00a0%'); // keeps "0.1 %" on one line

  function mount(root, D) {
    const F = WALK.fmt;
    const C = D.conditions.map((c) => ({ ...c, name: nb(c.name), mlp_part: nb(c.mlp_part) }));
    const byId = new Map(C.map((c) => [c.id, c]));
    const G = D.gain;
    const EQ = D.equal_count_comparison;
    const NP = D.problems.length;
    const THR = D.noise_of_one_reading.difference_must_exceed_points;
    const pct = (x) => F.fixed(x, 1);
    const signed = (x) => (x > 0 ? '+' : '') + F.fixed(x, 1);

    // ---------- the data must hold together before anything is drawn ----------
    C.forEach((c) => {
      if (!FAMILIES.includes(c.family)) throw new Error('f08: unknown family ' + c.family + ' for ' + c.id);
      if (c.n_problems !== 500) throw new Error('f08: ' + c.id + ' was scored on ' + c.n_problems + ' problems');
      if (c.examples.length !== NP) throw new Error('f08: ' + c.id + ' lacks an example answer');
      const chk = D.checks[c.id];
      if (!chk || Math.abs(chk.recomputed - c.accuracy) > 1e-9) throw new Error('f08: ' + c.id + ' accuracy does not match its recount');
      if (!(c.range_consistent_with_data[0] <= c.accuracy && c.accuracy <= c.range_consistent_with_data[1])) throw new Error('f08: ' + c.id + ' lies outside its range');
    });
    if (byId.get('base').accuracy !== G.base || byId.get('math').accuracy !== G.math) throw new Error('f08: the reference scores differ from the gain block');
    if (byId.get(EQ.most_changed).accuracy !== EQ.most_changed_accuracy || byId.get(EQ.scaled_sample).accuracy !== EQ.scaled_sample_accuracy) throw new Error('f08: the equal-count pair differs from its conditions');
    const allComps = D.comparisons.map((c) => ({ ...c, set: 500 })).concat(D.word_problems_1319.comparisons.map((c) => ({ ...c, set: 1319 })));
    allComps.forEach((c) => { if (!byId.has(c.first) || !byId.has(c.second)) throw new Error('f08: a comparison names an unknown condition'); });

    const ordered = FAMILIES.map((f) => C.filter((c) => c.family === f)
      .sort((a, b) => (a.percent_of_units || 0) - (b.percent_of_units || 0) || a.name.localeCompare(b.name)));
    const pair = new Set([EQ.most_changed, EQ.scaled_sample]);

    root.replaceChildren();
    const wrap = el('div', 'f08-wrap');
    root.appendChild(wrap);

    // ---------- the equal-count pair, stated ----------
    const mc = byId.get(EQ.most_changed);
    const ss = byId.get(EQ.scaled_sample);
    const facts = el('div', 'f08-facts');
    const fact = (big, text) => { const f = el('p', 'f08-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(pct(mc.accuracy), mc.name + ': ' + F.int(EQ.most_changed_entries) + ' MLP entries copied (base model: ' + pct(G.base) + ')');
    fact(pct(ss.accuracy), ss.name + ': ' + F.int(EQ.scaled_sample_entries_kept) + ' MLP entries kept at random and scaled up (mathematics model: ' + pct(G.math) + ')');
    fact(signed(EQ.comparison.difference), 'points between the two on the same ' + EQ.comparison.n_paired + ' problems; the range ' +
      signed(EQ.comparison.range[0]) + ' to ' + signed(EQ.comparison.range[1]) + '; a difference counts beyond ' + F.fixed(THR, 2));
    wrap.appendChild(facts);

    // ---------- controls ----------
    const controls = el('div', 'control-row f08-controls');
    const pick = el('label', 'f08-pick');
    pick.appendChild(el('span', 'control-label', 'Condition'));
    const sel = el('select', 'f08-select');
    ordered.forEach((group, k) => {
      const og = el('optgroup');
      og.label = FAMILIES[k] + ' (' + group.length + ')';
      group.forEach((c) => {
        const o = el('option', '', c.name + ': ' + pct(c.accuracy));
        o.value = c.id;
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    pick.appendChild(sel);
    controls.appendChild(pick);
    wrap.appendChild(controls);

    const legend = el('p', 'f08-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f08-key ' + cls), text); return s; };
    legend.append(key('bar', 'score: % of the 500 answered right'), key('pairbar', 'the equal-count pair'), key('whisker', 'the range (95 %)'),
      key('base', 'base model, ' + pct(G.base)), key('math', 'mathematics model, ' + pct(G.math)),
      key('right', 'right'), key('wrong', 'wrong, on problems 1 to ' + NP));
    wrap.appendChild(legend);

    // ---------- the list and the panel ----------
    const main = el('div', 'f08-main');
    wrap.appendChild(main);
    const listBox = el('section', 'f08-listbox');
    main.appendChild(listBox);
    const panel = el('section', 'f08-panel');
    panel.setAttribute('aria-live', 'polite');
    main.appendChild(panel);

    const x = (v) => v + '%'; // the axis runs 0 to 100 points
    const axis = el('div', 'f08-row f08-axisrow');
    axis.setAttribute('aria-hidden', 'true');
    axis.appendChild(el('span', 'f08-name'));
    const at = el('div', 'f08-track f08-axis');
    for (let v = 0; v <= 100; v += 20) {
      const t = el('span', 'f08-tick', String(v));
      t.style.left = x(v);
      at.appendChild(t);
    }
    axis.appendChild(at);
    const ph = el('span', 'f08-phead');
    for (let p = 0; p < NP; p++) ph.appendChild(el('span', '', String(p + 1)));
    axis.append(el('span', 'f08-val'), ph);
    listBox.appendChild(axis);

    const list = el('div', 'f08-list');
    list.tabIndex = 0;
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', 'All ' + C.length + ' conditions by score; arrow keys pick a condition');
    listBox.appendChild(list);
    const rows = new Map();
    const flat = [];
    ordered.forEach((group, k) => {
      list.appendChild(el('p', 'f08-group', FAMILIES[k]));
      group.forEach((c) => {
        const r = el('div', 'f08-row' + (pair.has(c.id) ? ' pair' : ''));
        r.setAttribute('role', 'option');
        r.id = 'f08-row-' + c.id;
        r.dataset.id = c.id;
        r.appendChild(el('span', 'f08-name', c.name));
        const tr = el('div', 'f08-track');
        [G.base, G.math].forEach((v, i) => { const l = el('i', 'f08-ref ' + (i ? 'math' : 'base')); l.style.left = x(v); tr.appendChild(l); });
        const bar = el('i', 'f08-bar');
        bar.style.width = x(c.accuracy);
        const wh = el('i', 'f08-whisker');
        wh.style.left = x(c.range_consistent_with_data[0]);
        wh.style.width = (c.range_consistent_with_data[1] - c.range_consistent_with_data[0]) + '%';
        tr.append(bar, wh);
        r.appendChild(tr);
        r.appendChild(el('span', 'f08-val', pct(c.accuracy)));
        const marks = el('span', 'f08-marks');
        c.examples.forEach((ex, p) => {
          const m = el('i', 'f08-pm ' + (ex.score ? 'right' : 'wrong'));
          m.dataset.p = String(p);
          marks.appendChild(m);
        });
        r.appendChild(marks);
        list.appendChild(r);
        rows.set(c.id, r);
        flat.push(c.id);
      });
    });
    listBox.appendChild(el('p', 'f08-note', 'Score: the share of the 500 problems answered right, one answer each. ' +
      'Hover or tap a row for its numbers and a square for that answer; the menu, a click or the arrow keys pick a condition.'));

    const rowBox = (c) => ({
      title: c.name,
      rows: [['score', pct(c.accuracy), pair.has(c.id) ? 'claim' : 'ink'],
        ['the range (95 %)', pct(c.range_consistent_with_data[0]) + ' to ' + pct(c.range_consistent_with_data[1])],
        ['answers at the ' + F.int(D.length_cap_tokens) + '-token limit', F.percent(c.share_hit_length_cap, 1)],
        ['problems 1 to ' + NP, c.examples.map((e) => (e.score ? 'right' : 'wrong')).join(', ')]],
    });
    const answerBox = (c, p) => {
      const ex = c.examples[p];
      return {
        title: 'Problem ' + (p + 1) + ', ' + c.name,
        rows: [['answer', ex.extracted === null ? 'none found' : ex.extracted], ['gold', D.problems[p].gold],
          ['graded', ex.score ? 'right' : 'wrong'], ['new tokens', F.int(ex.new_tokens)]],
      };
    };

    // ---------- the panel ----------
    let cur = null;
    function select(id, quiet) {
      const c = byId.get(id);
      if (!c) throw new Error('f08: no condition ' + id);
      cur = id;
      sel.value = id;
      rows.forEach((r, k) => { r.classList.toggle('cur', k === id); r.setAttribute('aria-selected', String(k === id)); });
      list.setAttribute('aria-activedescendant', rows.get(id).id);
      drawPanel(c);
      compRows.forEach(({ row, comp }) => row.classList.toggle('cur', comp.first === id || comp.second === id));
      if (!quiet) WALK.valueBox.showFor(rows.get(id).querySelector('.f08-bar'), rowBox(c));
    }

    function drawPanel(c) {
      panel.replaceChildren();
      panel.appendChild(el('p', 'f08-pfam', c.family));
      panel.appendChild(el('h3', 'f08-ph', c.name));
      const score = el('p', 'f08-pscore');
      score.append(el('b', '', pct(c.accuracy)), el('span', '', ' % of the ' + c.n_problems + ' problems answered right; the range ' +
        pct(c.range_consistent_with_data[0]) + ' to ' + pct(c.range_consistent_with_data[1])));
      panel.appendChild(score);
      const dl = el('dl', 'f08-dl');
      const item = (k, v) => { dl.append(el('dt', '', k), el('dd', '', v)); };
      item('MLP', c.mlp_part);
      item('Attention', FROM[c.tensors_from.attention]);
      item('Norm weights', FROM[c.tensors_from.norm_weights]);
      item('Embeddings and output matrix', FROM[c.tensors_from.embeddings_and_output_matrix]);
      item('Sampling seeds', c.seed_set === 'first' ? 'the first set, 1,000,000 + problem index' : 'the second set, 2,000,000 + problem index');
      item('Answers at the ' + F.int(D.length_cap_tokens) + '-token limit', F.percent(c.share_hit_length_cap, 1));
      if (c.built) {
        item('MLP entries in the mask', F.int(c.built.mlp_entries_in_mask) + ', of which ' + F.int(c.built.mlp_entries_changed) + ' changed');
        item('Squared change added, MLP', F.fixed(c.built.realized_mlp_change_energy, 4));
        item('Squared change added, other tensors', F.fixed(c.built.realized_non_mlp_change_energy, 4));
      }
      panel.appendChild(dl);

      const comps = allComps.filter((k) => k.first === c.id || k.second === c.id);
      if (comps.length) {
        panel.appendChild(el('p', 'f08-plabel', 'Registered paired differences, first minus second, in points'));
        const ul = el('ul', 'f08-pcomps');
        comps.forEach((k) => {
          ul.appendChild(el('li', '', byId.get(k.first).name + ' − ' + byId.get(k.second).name + ': ' + signed(k.difference) +
            ' (the range ' + signed(k.range[0]) + ' to ' + signed(k.range[1]) + ')' + (k.set === 1319 ? ', on the 1,319 held-out word problems' : '')));
        });
        panel.appendChild(ul);
      }

      panel.appendChild(el('p', 'f08-plabel', 'Problems 1 to ' + NP + ' of the 500, in file order'));
      c.examples.forEach((ex, p) => {
        const P = D.problems[p];
        const box = el('div', 'f08-prob');
        const head = el('p', 'f08-probhead');
        head.append(el('i', 'f08-pm ' + (ex.score ? 'right' : 'wrong')), ' ', el('b', '', 'Problem ' + (p + 1) + ': ' + (ex.score ? 'right' : 'wrong')));
        box.appendChild(head);
        box.appendChild(el('p', 'f08-stmt', P.statement));
        const ans = el('p', 'f08-ans');
        ans.append('Answer: ', el('code', '', ex.extracted === null ? 'none found' : ex.extracted), '  ·  gold: ', el('code', '', P.gold));
        box.appendChild(ans);
        box.appendChild(el('p', 'f08-meta', F.int(ex.new_tokens) + ' new tokens' + (ex.hit_length_cap ? ', stopped at the token limit' : '') +
          '; ' + (ex.extraction === 'answer_line' ? 'answer read from the last “Answer:” line' : ex.extraction === 'boxed' ? 'answer read from the last boxed expression' : 'no “Answer:” line and no boxed expression found')));
        const det = el('details', 'f08-det');
        const text = D.texts[ex.text];
        det.appendChild(el('summary', '', 'The full answer (' + F.int(text.length) + ' characters)'));
        det.appendChild(el('div', 'f08-text', text));
        box.appendChild(det);
        panel.appendChild(box);
      });
    }

    // ---------- list interaction ----------
    const rowOf = (e) => e.target.closest('.f08-row[data-id]');
    list.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const r = rowOf(e);
      if (!r) { WALK.valueBox.hide(); return; }
      const c = byId.get(r.dataset.id);
      const m = e.target.closest('.f08-pm');
      WALK.valueBox.show(m ? answerBox(c, Number(m.dataset.p)) : rowBox(c), e.clientX, e.clientY);
    });
    list.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') WALK.valueBox.hide(); });
    list.addEventListener('click', (e) => {
      const r = rowOf(e);
      if (!r) return;
      select(r.dataset.id, true);
      if (e.pointerType && e.pointerType !== 'mouse') {
        const m = e.target.closest('.f08-pm');
        const c = byId.get(r.dataset.id);
        WALK.valueBox.showFor(m || r.querySelector('.f08-bar'), m ? answerBox(c, Number(m.dataset.p)) : rowBox(c));
      }
    });
    list.addEventListener('keydown', (e) => {
      const move = { ArrowDown: 1, ArrowUp: -1, Home: -flat.length, End: flat.length }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      const i = Math.max(0, Math.min(flat.length - 1, flat.indexOf(cur) + move));
      select(flat[i]);
      rows.get(flat[i]).scrollIntoView({ block: 'nearest' });
    });
    list.addEventListener('focus', () => WALK.valueBox.showFor(rows.get(cur).querySelector('.f08-bar'), rowBox(byId.get(cur))));
    list.addEventListener('blur', () => WALK.valueBox.hide());
    sel.addEventListener('change', () => select(sel.value, true));

    // ---------- the paired differences ----------
    const diffs = el('section', 'f08-diffs');
    wrap.appendChild(diffs);
    diffs.appendChild(el('h3', 'f08-h', 'The registered paired differences'));
    diffs.appendChild(el('p', 'f08-note', 'Each row is the first condition’s score minus the second’s on the same problems, with its range. ' +
      'The shaded band is ±' + F.fixed(THR, 2) + ' points: on the 500 problems a difference counts only beyond it. ' +
      'The ' + D.word_problems_1319.comparisons.length + ' rows marked with a diamond are on the 1,319 held-out word problems, where the base model scores ' +
      pct(D.word_problems_1319.base) + ' and the mathematics model ' + pct(D.word_problems_1319.math) + '.'));
    const lo = Math.floor(Math.min(...allComps.map((k) => k.range[0])) / 5) * 5;
    const hi = Math.ceil(Math.max(...allComps.map((k) => k.range[1]), THR) / 5) * 5;
    const dx = (v) => ((v - lo) / (hi - lo)) * 100 + '%';
    const dhead = el('div', 'f08-drow f08-daxis');
    dhead.setAttribute('aria-hidden', 'true');
    dhead.appendChild(el('span', 'f08-dname'));
    const dat = el('div', 'f08-dtrack');
    for (let v = lo; v <= hi; v += 5) {
      const t = el('span', 'f08-tick' + (v % 10 ? ' odd' : ''), v > 0 ? '+' + v : v < 0 ? '−' + -v : '0');
      t.style.left = dx(v);
      dat.appendChild(t);
    }
    dhead.appendChild(dat);
    diffs.appendChild(dhead);
    const compRows = [];
    // each word-problem row goes right after the 500-problem row of the same pair
    const order = [];
    D.comparisons.forEach((k) => {
      order.push({ ...k, set: 500 });
      D.word_problems_1319.comparisons.filter((w) => w.first === k.first && w.second === k.second).forEach((w) => order.push({ ...w, set: 1319 }));
    });
    if (order.length !== allComps.length) throw new Error('f08: a word-problem comparison has no 500-problem pair');
    order.forEach((k) => {
      const isPair = pair.has(k.first) && pair.has(k.second);
      const row = el('div', 'f08-drow' + (k.set === 1319 ? ' held' : '') + (isPair ? ' pair' : ''));
      row.appendChild(el('span', 'f08-dname', (k.set === 1319 ? 'the same, 1,319 word problems' : byId.get(k.first).name + ' − ' + byId.get(k.second).name)));
      const tr = el('div', 'f08-dtrack');
      const band = el('i', 'f08-dband');
      band.style.left = dx(-THR);
      band.style.width = ((2 * THR) / (hi - lo)) * 100 + '%';
      const zero = el('i', 'f08-dzero');
      zero.style.left = dx(0);
      const wh = el('i', 'f08-dwhisker');
      wh.style.left = dx(k.range[0]);
      wh.style.width = ((k.range[1] - k.range[0]) / (hi - lo)) * 100 + '%';
      const dot = el('i', 'f08-ddot');
      dot.style.left = dx(k.difference);
      tr.append(band, zero, wh, dot);
      row.appendChild(tr);
      row.appendChild(el('span', 'f08-dval', signed(k.difference)));
      diffs.appendChild(row);
      WALK.hover(row, () => ({
        title: byId.get(k.first).name + ' − ' + byId.get(k.second).name,
        rows: [['on', k.set === 1319 ? '1,319 word problems' : '500 competition problems'], ['difference', signed(k.difference) + ' points', isPair ? 'claim' : 'ink'],
          ['the range (95 %)', signed(k.range[0]) + ' to ' + signed(k.range[1])]].concat(k.set === 500 ? [['problems paired', String(k.n_paired)]] : []),
      }));
      row.setAttribute('aria-label', (k.set === 1319 ? 'On the 1,319 word problems, ' : '') + byId.get(k.first).name + ' minus ' + byId.get(k.second).name + ': ' + signed(k.difference));
      compRows.push({ row, comp: k });
    });

    select(EQ.most_changed, true);
  }

  WALK.register('f08', { data: 'F08_DATA', mount });
})();
