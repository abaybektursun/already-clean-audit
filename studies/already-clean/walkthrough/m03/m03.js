// m03: the corpus and a pole's top-50 activating tokens. Registers the mount "m03" with site.js; reads window.M03_DATA
// (m03/data.js). The corpus rule drawn from the data's counts; one pole's top-50 tokens at the last checkpoint,
// added in rank order by the reader, each marked on a strip of the corpus windows; then three clusters' tokens, fixed
// at the last checkpoint, with their activations at every checkpoint, and one drag over the sixteen checkpoints.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const flat = (s) => s.replace(/\r\n|\r|\n/g, '↵').replace(/\t/g, '⇥');
  // The fifty's activations are stored rounded to four significant digits; print them that way.
  const sig4 = (x) => x.toPrecision(4).replace('-', '\u2212');
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m03: cannot read the pole id ' + id);
    const layer = Number(m[1]), unit = Number(m[2]);
    return { layer, unit, positive: m[3] === '+', name: 'L' + layer + '.N' + unit + m[3] };
  };

  function mount(root, D) {
    const F = WALK.fmt;
    const C = D.corpus;
    const steps = D.snapshot_steps;
    const LAST = steps.length - 1;
    if (steps.length !== 16) throw new Error('m03: expected 16 checkpoint steps');
    if (C.directories.length !== C.directories_used) throw new Error('m03: directory count differs from directories_used');
    if (C.tokens_in_windows !== C.windows * C.window_tokens) throw new Error('m03: windows × window tokens is not tokens_in_windows');
    if (C.scored_tokens !== C.tokens_in_windows - C.windows) throw new Error('m03: scored tokens are not one fewer per window');
    const fifty = D.fifty_at_end;
    if (fifty.contexts.length !== C.fifty) throw new Error('m03: the pole does not have 50 top tokens');
    if (fifty.step !== steps[LAST]) throw new Error('m03: the top-50 are not from the last checkpoint');
    const branch = (step) => 'stage1-step' + step;
    const own = sideOf(fifty.id);
    if (own.layer !== fifty.layer || own.unit !== fifty.unit || own.positive !== (fifty.side === 'positive')) throw new Error('m03: pole id ' + fifty.id + ' does not match its fields');

    root.replaceChildren();
    const wrap = el('div', 'm03-wrap');
    root.appendChild(wrap);

    // ---------- the corpus ----------
    const corpus = el('section', 'm03-corpus');
    wrap.appendChild(corpus);
    corpus.appendChild(el('h3', 'm03-h', 'The corpus'));
    const stepBox = (num, text) => {
      const s = el('div', 'm03-step');
      s.append(el('span', 'm03-num', num), el('div', 'm03-steptext'));
      s.lastChild.append(...[].concat(text));
      corpus.appendChild(s);
      return s.lastChild;
    };
    const cm = C.directories.filter((d) => d.code_or_math).length;
    const s1 = stepBox('1', el('p', '', C.directories_used + ' of the ' + C.directories_listed + ' directories of ' + C.dataset +
      ', the model’s own stage-1 training mix: the directory at index ⌊i × ' + C.directories_listed + ' / ' + C.directories_used +
      '⌋ of the sorted list, for i = 0 to ' + (C.directories_used - 1) + '.'));
    const grid = el('div', 'm03-dirs');
    grid.setAttribute('role', 'list');
    grid.setAttribute('aria-label', C.directories_used + ' directories');
    C.directories.forEach((d) => {
      const c = el('i', 'm03-dir' + (d.code_or_math ? ' cm' : ''));
      c.setAttribute('role', 'listitem');
      c.setAttribute('aria-label', d.name);
      WALK.hover(c, { title: d.name, rows: [['kind', d.kind], ['code or mathematics', d.code_or_math ? 'yes' : 'no']] });
      grid.appendChild(c);
    });
    s1.appendChild(grid);
    const leg = el('p', 'm03-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm03-key ' + cls), text); return s; };
    leg.append(key('cm', 'code or mathematics (' + cm + ')'), key('other', 'web text and documents (' + (C.directories_used - cm) + ')'));
    s1.appendChild(leg);
    stepBox('2', el('p', '', 'From each directory, the documents that fill its first ' + F.int(C.tokens_per_directory) +
      ' tokens, with duplicate texts removed, joined with the end-of-text token.'));
    stepBox('3', el('p', '', 'Cut into ' + F.int(C.windows) + ' windows of ' + F.int(C.window_tokens) + ' tokens: ' +
      F.int(C.tokens_in_windows) + ' tokens.'));
    const s4 = stepBox('4', el('p', '', 'Each window is run with the start-of-text token in front. The start token and window position 0 ' +
      'are not scored: at position 0 the model has seen no text before it.'));
    const win = el('div', 'm03-window');
    win.setAttribute('aria-hidden', 'true');
    const cell = (cls, text, sub) => { const c = el('div', 'm03-cell ' + cls); c.append(el('span', 'm03-cell-top', text), el('span', 'm03-cell-sub', sub)); return c; };
    win.append(cell('off', 'start', 'not scored'), cell('off', '0', 'not scored'),
      cell('on', 'positions 1 to ' + F.int(C.window_tokens - 1), 'scored'));
    s4.appendChild(win);
    const res = el('p', 'm03-result');
    res.append(F.int(C.tokens_in_windows) + ' − ' + F.int(C.windows) + ' = ', el('b', '', F.int(C.scored_tokens)), ' scored tokens.');
    s4.appendChild(res);

    // ---------- one side's fifty ----------
    const fs = el('section', 'm03-fifty');
    wrap.appendChild(fs);
    fs.appendChild(el('h3', 'm03-h', 'The top-50 activating tokens of one pole at the last checkpoint'));
    fs.appendChild(el('p', 'm03-note', 'Pole ' + own.name + ' is the ' + fifty.side + ' activations of neuron ' + F.int(fifty.unit) + ' of layer ' +
      fifty.layer + '. At the checkpoint ' + branch(fifty.step) + ' (step ' + F.int(fifty.step) + ') we computed its activation a at all ' +
      F.int(C.scored_tokens) + ' scored tokens and kept the ' + C.fifty + ' tokens with the largest a, each with ' + C.tokens_before_context +
      ' tokens of text before it and ' + C.tokens_after_context + ' after: the pole’s top-50 activating tokens. They are fixed corpus tokens, ' +
      'so the activation at them can be recomputed at any checkpoint.'));
    const ctl = el('div', 'control-row');
    const next = el('button', 'button', 'Add the next token');
    const allB = el('button', 'button', 'Add all 50');
    const clear = el('button', 'button', 'Start over');
    [next, allB, clear].forEach((b) => { b.type = 'button'; ctl.appendChild(b); });
    const count = el('span', 'control-label');
    count.setAttribute('aria-live', 'polite');
    ctl.appendChild(count);
    fs.appendChild(ctl);

    const strip = el('div', 'm03-strip');
    const stripPlot = el('div', 'm03-strip-plot');
    strip.appendChild(stripPlot);
    const sa = el('div', 'm03-strip-axis');
    sa.append(el('span', 'm03-tick l', '0'), el('span', 'm03-strip-name', 'window of the corpus'), el('span', 'm03-tick r', F.int(C.windows - 1)));
    strip.appendChild(sa);
    fs.appendChild(strip);

    const list = el('div', 'm03-kwic');
    list.setAttribute('role', 'list');
    const head = el('div', 'm03-khead');
    head.append(el('span', '', 'rank'), el('span', 'm03-k-ctx', 'the token in its text, cut to one line'), el('span', 'm03-k-a', 'a'));
    list.appendChild(head);
    fs.appendChild(list);
    fs.appendChild(el('p', 'm03-note', '↵ marks a line break in the text. The text around each token is shortened here; a is rounded to four significant digits.'));

    const ctxBox = (c) => ({
      title: 'Rank ' + c.rank + ' of ' + C.fifty + ': “' + flat(c.token) + '”',
      rows: [['a', sig4(c.a), 'ink'], ['window', F.int(c.window)], ['position in the window', F.int(c.position_in_window)], ['document', F.int(c.document)]],
    });
    let shown = 0;
    function drawFifty() {
      stripPlot.replaceChildren();
      list.querySelectorAll('.m03-row').forEach((r) => r.remove());
      fifty.contexts.slice(0, shown).forEach((c, i) => {
        const newest = i === shown - 1;
        const t = el('i', 'm03-wtick' + (newest ? ' new' : ''));
        t.style.left = (100 * c.window) / (C.windows - 1) + '%';
        t.setAttribute('aria-label', 'Rank ' + c.rank + ', window ' + c.window);
        WALK.hover(t, ctxBox(c));
        stripPlot.appendChild(t);
        const r = el('div', 'm03-row' + (newest ? ' new' : ''));
        r.setAttribute('role', 'listitem');
        const before = el('span', 'm03-before');
        before.appendChild(el('span', '', flat(c.before)));
        r.append(el('span', 'm03-rank', String(c.rank)), before, el('span', 'm03-tok', flat(c.token)), el('span', 'm03-after', flat(c.after)), el('span', 'm03-a', sig4(c.a)));
        r.setAttribute('aria-label', 'Rank ' + c.rank + ', token ' + JSON.stringify(c.token) + ', a = ' + c.a);
        WALK.hover(r, ctxBox(c));
        list.appendChild(r);
      });
      count.textContent = shown + ' of ' + C.fifty + ' shown';
      next.disabled = shown === C.fifty;
      allB.disabled = shown === C.fifty;
      clear.disabled = shown === 1;
    }
    next.addEventListener('click', () => { shown = Math.min(C.fifty, shown + 1); drawFifty(); });
    allB.addEventListener('click', () => { shown = C.fifty; drawFifty(); });
    clear.addEventListener('click', () => { shown = 1; drawFifty(); });
    shown = 1;
    drawFifty();

    // ---------- the same positions at an earlier snapshot ----------
    const rb = el('section', 'm03-back');
    wrap.appendChild(rb);
    rb.appendChild(el('h3', 'm03-h', 'Three clusters: the activation at their tokens at every checkpoint'));
    const ex = D.positions_read_back[0];
    rb.appendChild(el('p', 'm03-note', 'The data file does not hold this pole’s top-50 at earlier checkpoints. It holds ' +
      D.positions_read_back.length + ' clusters from other poles. A cluster is a subset of a pole’s top-50 tokens that an LLM put ' +
      'together and named; for example, ' + ex.group_size + ' tokens of ' + sideOf(ex.id).name + ' that it named “' + ex.group_name + '”. ' +
      'We took each cluster’s tokens at the last checkpoint and recomputed the activation at them at all ' + steps.length + ' checkpoints. ' +
      'The value drawn is the activation in the pole’s direction: a for a positive pole, −a for a negative pole, and 0 where a has the other sign.'));
    const sctl = el('div', 'm03-scrub');
    const lab = el('label', 'm03-scrub-label');
    lab.htmlFor = 'm03-snap';
    const scrub = el('input', 'scrub');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = String(LAST); scrub.step = '1'; scrub.value = String(LAST); scrub.id = 'm03-snap';
    const ticks = el('div', 'm03-sticks');
    ticks.setAttribute('aria-hidden', 'true');
    // Step labels: every fifth checkpoint on a phone; every third as well on a wider screen. The checkpoints are evenly spaced.
    steps.forEach((s, i) => {
      if (i % 5 && i % 3) return;
      const t = el('span', 'm03-stick' + (i % 5 ? ' minor' : '') + (i === LAST ? ' last' : ''), F.int(s));
      t.style.left = (100 * i) / LAST + '%';
      ticks.appendChild(t);
    });
    sctl.append(lab, scrub, ticks, el('p', 'm03-sname', 'training step of the checkpoint (the sixteen in order, not to scale)'));
    rb.appendChild(sctl);

    const panels = D.positions_read_back.map((g) => {
      const who = sideOf(g.id);
      const A = g.activation_own_direction_by_snapshot;
      if (A.length !== steps.length) throw new Error('m03: ' + g.id + ' is not given at every checkpoint');
      if ((g.side === 'positive') !== who.positive) throw new Error('m03: the pole of ' + g.id + ' does not match its id');
      A.forEach((row) => { if (row.length !== g.group_size) throw new Error('m03: ' + g.id + ' has a checkpoint with a different cluster size'); });
      const max = Math.max(...A.flat());
      const p = el('div', 'm03-panel');
      p.appendChild(el('h4', 'm03-h4', who.name + ' (neuron ' + F.int(who.unit) + ' of layer ' + who.layer + ', ' + g.side + ' pole): cluster “' +
        g.group_name + '”, ' + g.group_size + ' of its top-50 tokens (ranks ' + g.ranks_in_fifty.join(', ') + ')'));
      const bgrid = el('div', 'm03-bgrid');
      p.appendChild(bgrid);
      const rows = g.positions_at_end.map((pos, j) => {
        const r = el('div', 'm03-brow');
        const before = el('span', 'm03-before');
        before.appendChild(el('span', '', flat(pos.before_tail)));
        const track = el('span', 'm03-track');
        const bar = el('i', 'm03-bar');
        const end = el('i', 'm03-end');
        end.style.left = (100 * A[LAST][j]) / max + '%';
        track.append(bar, end);
        const val = el('span', 'm03-bval');
        r.append(before, el('span', 'm03-tok', flat(pos.token)), track, val);
        bgrid.appendChild(r);
        return { r, bar, val, pos, j };
      });
      p.appendChild(el('p', 'm03-note m03-scale', 'Bars on one scale per cluster, 0 to ' + F.fixed(max, 4) + ' (the largest value in the cluster at any checkpoint); the thin mark is the value at the last checkpoint.'));
      rb.appendChild(p);
      return { g, A, rows, max };
    });
    const leg2 = el('p', 'm03-legend');
    leg2.append(key('bar', 'value at the chosen checkpoint'), key('end', 'value at the last checkpoint, step ' + F.int(steps[LAST])));
    rb.appendChild(leg2);

    function drawBack() {
      const t = Number(scrub.value);
      lab.textContent = 'Checkpoint at step ' + F.int(steps[t]) + ' (' + branch(steps[t]) + ')';
      scrub.setAttribute('aria-valuetext', 'checkpoint at step ' + F.int(steps[t]));
      panels.forEach(({ A, rows, max }) => rows.forEach(({ r, bar, val, pos, j }) => {
        const v = A[t][j];
        bar.style.width = (100 * v) / max + '%';
        val.textContent = F.fixed(v, 4);
        r.setAttribute('aria-label', 'Rank ' + pos.rank + ', token ' + JSON.stringify(pos.token) + ', ' + F.fixed(v, 4) + ' at step ' + F.int(steps[t]));
      }));
    }
    panels.forEach(({ g, A, rows }) => {
      const who = sideOf(g.id);
      rows.forEach(({ r, pos, j }) => WALK.hover(r, () => {
        const t = Number(scrub.value);
        return {
          title: 'Rank ' + pos.rank + ' of the top-50 of ' + who.name + ': “' + flat(pos.token) + '”',
          rows: [['at step ' + F.int(steps[t]), F.fixed(A[t][j], 4), 'ink'], ['at step ' + F.int(steps[LAST]) + ', the last checkpoint', F.fixed(A[LAST][j], 4), 'muted']],
        };
      }));
    });
    scrub.addEventListener('input', drawBack);
    drawBack();
  }

  WALK.register('m03', { data: 'M03_DATA', mount });
})();
