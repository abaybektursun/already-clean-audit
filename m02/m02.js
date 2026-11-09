// m02: a unit and its two sides. Registers the mount "m02" with site.js; reads window.M02_DATA (m02/data.js).
// The layer's three MLP matrices drawn to true shape with the unit's row and column lit, the formula for the
// unit's activation a, and a toggle between the unit's positive side and negative side, each with its fifty
// contexts laid out around the token, with g, u and a from the data on hover, tap or keyboard focus.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  // Line breaks and tabs are drawn as visible marks so that each context fits on one line.
  const flat = (s) => s.replace(/\r\n|\r|\n/g, '↵').replace(/\t/g, '⇥');
  const FIRST = 10; // contexts shown before "Show all fifty"
  // g and u are 16-bit values; print every decimal they hold.
  const exact = (x) => WALK.fmt.fixed(x, Math.min(8, (String(x).split('.')[1] || '').length));

  function mount(root, D) {
    const F = WALK.fmt;
    const H = D.hidden_size;
    const sides = D.unit_shown;
    if (sides.length !== 2 || sides[0].side !== 'positive' || sides[1].side !== 'negative') throw new Error('m02: expected the positive then the negative side');
    if (sides[0].unit !== sides[1].unit || sides[0].layer !== sides[1].layer) throw new Error('m02: the two sides are not of one unit');
    const n = sides[0].unit;
    const layerNo = sides[0].layer;
    const [gR, gC] = D.matrix_shapes.gate;
    const [dR, dC] = D.matrix_shapes.down;
    if (gR !== D.units_per_layer || dC !== D.units_per_layer || gC !== H || dR !== H) throw new Error('m02: matrix shapes do not match the layer width');
    sides.forEach((s) => s.contexts.forEach((c, i) => {
      if (c.rank !== i) throw new Error('m02: contexts are not in rank order');
      if ((s.side === 'positive') !== (c.a > 0)) throw new Error('m02: a context has the other side’s sign');
    }));

    root.replaceChildren();
    const wrap = el('div', 'm02-wrap');
    root.appendChild(wrap);

    // ---------- the three matrices ----------
    const mats = el('section', 'm02-mats-sec');
    wrap.appendChild(mats);
    mats.appendChild(el('h3', 'm02-h', 'Unit ' + F.int(n) + ' of layer ' + layerNo + ' in ' + D.model));
    const row = el('div', 'm02-mats');
    mats.appendChild(row);
    const shapeText = (s) => s.map(F.int).join(' × ');
    const drawMatrix = (key, name, ownsRow) => {
      const [r, c] = D.matrix_shapes[key];
      const fig = el('div', 'm02-m');
      const frame = el('div', 'm02-frame');
      frame.style.width = 'calc(var(--u) * ' + c / H + ')';
      frame.style.height = 'calc(var(--u) * ' + r / H + ')';
      const box = el('div', 'm02-box');
      WALK.hover(box, { title: name, rows: [['shape', shapeText([r, c])], ['numbers', F.int(r * c)]] });
      box.setAttribute('aria-label', name + ', ' + shapeText([r, c]));
      const hl = el('div', 'm02-hl ' + (ownsRow ? 'row' : 'col'));
      hl.style[ownsRow ? 'top' : 'left'] = (100 * n) / (ownsRow ? r : c) + '%';
      const what = ownsRow ? 'row ' + F.int(n) + ' of the ' + name : 'column ' + F.int(n) + ' of the ' + name;
      WALK.hover(hl, { title: what, rows: [['numbers', F.int(ownsRow ? c : r)], ['position', F.int(n) + ' of ' + F.int(ownsRow ? r : c) + ', from 0']]});
      hl.setAttribute('aria-label', what);
      frame.append(box, hl);
      fig.append(frame, el('span', 'm02-mname', name), el('span', 'm02-mshape', shapeText([r, c])),
        el('span', 'm02-mown', (ownsRow ? 'row ' : 'column ') + F.int(n)));
      return fig;
    };
    row.append(drawMatrix('gate', 'gate matrix', true), drawMatrix('up', 'up matrix', true), drawMatrix('down', 'down matrix', false)); // vocab-ok
    const own = el('p', 'm02-note');
    own.textContent = 'Drawn to true shape (rows × columns, numbered from 0); the lit line is the unit’s own row or column. ' +
      'The unit owns ' + F.int(D.numbers_per_unit) + ' numbers: ' + F.int(H) + ' in each of its two rows and ' + F.int(H) +
      ' in its column. The weight values are not drawn: they are not in the data file.';
    mats.appendChild(own);

    // ---------- the formula ----------
    const formula = el('section', 'm02-formula');
    wrap.appendChild(formula);
    [
      ['g = gate row ' + F.int(n) + ' · h', 'h is the shared vector at one token, ' + F.int(H) + ' numbers'], // vocab-ok
      ['u = up row ' + F.int(n) + ' · h', ''],
      ['a = silu(g) · u', 'g and u are 16-bit (bfloat16); a is computed from them in 32-bit'],
      ['silu(x) = x·σ(x) = x/(1 + e^(−x))', ''],
    ].forEach(([f, note]) => {
      const p = el('p', 'm02-eq');
      p.appendChild(el('code', '', f));
      if (note) p.appendChild(el('span', 'm02-eqnote', note));
      formula.appendChild(p);
    });
    formula.appendChild(el('p', 'm02-note', 'a is signed. The tokens where a > 0 are the unit’s positive side; the tokens where a < 0 are its negative side. ' +
      'Each side has its own fifty contexts: the fifty tokens of the corpus where a is largest (positive side) or most negative (negative side).'));

    // ---------- the two sides ----------
    const ctx = el('section', 'm02-ctx');
    wrap.appendChild(ctx);
    const controls = el('div', 'control-row');
    const toggle = el('div', 'toggle');
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Side of the unit');
    const btns = sides.map((s) => {
      const b = el('button', '', s.side === 'positive' ? 'Positive side' : 'Negative side');
      b.type = 'button';
      toggle.appendChild(b);
      return b;
    });
    controls.appendChild(toggle);
    ctx.appendChild(controls);
    const summary = el('p', 'm02-summary');
    summary.setAttribute('aria-live', 'polite');
    ctx.appendChild(summary);
    const signs = el('p', 'm02-note m02-signs');
    ctx.appendChild(signs);
    const head = el('div', 'm02-khead');
    head.append(el('span', 'm02-k-rank', 'rank'), el('span', 'm02-k-ctx', 'context, cut to one line around the token'), el('span', 'm02-k-a', 'a'));
    const list = el('div', 'm02-kwic');
    list.setAttribute('role', 'list');
    list.appendChild(head);
    ctx.appendChild(list);
    const more = el('button', 'button m02-more');
    more.type = 'button';
    ctx.appendChild(more);
    ctx.appendChild(el('p', 'm02-note', '↵ marks a line break in the text, ⇥ a tab. The contexts are read at step ' + F.int(D.snapshot.step) +
      ', the snapshot whose files keep g and u for each of a side’s fifty; activations at the other tokens of a context are not in the data file.'));

    let cur = 0;
    let all = false;
    const SIGN_TEXT = { 'g>=0,u>=0': 'g ≥ 0, u ≥ 0', 'g>=0,u<0': 'g ≥ 0, u < 0', 'g<0,u>=0': 'g < 0, u ≥ 0', 'g<0,u<0': 'g < 0, u < 0' };
    function draw() {
      const s = sides[cur];
      btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === cur)));
      summary.textContent = (s.side === 'positive' ? 'Positive side (a > 0)' : 'Negative side (a < 0)') + ' of unit ' + F.int(n) + ', layer ' + layerNo +
        ': its fifty contexts, largest |a| first. The largest |a| is ' + F.fixed(s.max_abs_activation, 3) + '.';
      signs.textContent = 'Of these fifty, by the signs of g and u: ' + Object.keys(SIGN_TEXT).map((k) => {
        if (!(k in s.sign_pattern_counts)) throw new Error('m02: sign pattern ' + k + ' is missing');
        return SIGN_TEXT[k] + ': ' + s.sign_pattern_counts[k];
      }).join('; ') + '. Contexts with g < 0 and u < 0 are left out of groups.';
      list.querySelectorAll('.m02-row').forEach((r) => r.remove());
      s.contexts.slice(0, all ? s.contexts.length : FIRST).forEach((c) => {
        const r = el('div', 'm02-row');
        r.setAttribute('role', 'listitem');
        const before = el('span', 'm02-before');
        before.appendChild(el('span', '', flat(c.before).slice(-90)));
        const after = el('span', 'm02-after', flat(c.after).slice(0, 60));
        r.append(el('span', 'm02-rank', String(c.rank)), before, el('span', 'm02-tok', flat(c.token)), after, el('span', 'm02-a', F.fixed(c.a, 3)));
        r.setAttribute('aria-label', 'Rank ' + c.rank + ', token ' + JSON.stringify(c.token) + ', a = ' + F.fixed(c.a, 4));
        WALK.hover(r, {
          title: 'Rank ' + c.rank + ' of the ' + s.side + ' side’s fifty: “' + flat(c.token) + '”',
          rows: [['g', exact(c.g)], ['u', exact(c.u)], ['a = silu(g) · u', F.fixed(c.a, 4), 'ink'],
            ['side', s.side === 'positive' ? 'positive (a > 0)' : 'negative (a < 0)'],
            ['document, token', F.int(c.document) + ', ' + F.int(c.position_in_document)]],
        });
        list.appendChild(r);
      });
      more.textContent = all ? 'Show the first ten' : 'Show all fifty';
      more.setAttribute('aria-expanded', String(all));
    }
    btns.forEach((b, i) => b.addEventListener('click', () => { cur = i; draw(); }));
    more.addEventListener('click', () => { all = !all; draw(); });
    draw();

    // ---------- the layer's write ----------
    const write = el('section', 'm02-write');
    wrap.appendChild(write);
    write.appendChild(el('p', 'm02-eq')).appendChild(el('code', '', 'MLP(h) = Σ over n of a_n · (column n of the down matrix)'));
    write.appendChild(el('p', 'm02-note', 'The layer’s write to h is the sum over all ' + F.int(D.units_per_layer) +
      ' units of a_n times column n, a vector of ' + F.int(H) + ' numbers; the sum is RMS-normalized, scaled by the layer’s norm weight (' +
      F.int(H) + ' numbers), then added to h. One unit’s write is never added alone. A layer has ' + F.int(D.sides_per_layer) +
      ' sides; the model has ' + F.int(D.sides_total) + '.'));
  }

  WALK.register('m02', { data: 'M02_DATA', mount });
})();
