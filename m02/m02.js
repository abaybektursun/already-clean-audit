// m02: a neuron and its two poles. Registers the mount "m02" with site.js; reads window.M02_DATA (m02/data.js).
// The layer's gate_proj, up_proj and down_proj drawn to true shape with the neuron's row and column highlighted, the
// formula for the neuron's activation a, and a toggle between the neuron's positive pole and negative pole, each with
// its top-50 activating tokens laid out in their text, with g, u and a from the data on hover, tap or keyboard focus.
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
  const FIRST = 10; // tokens shown before "Show all 50"
  // g and u are bfloat16 values; print every decimal they hold.
  const exact = (x) => WALK.fmt.fixed(x, Math.min(8, (String(x).split('.')[1] || '').length));

  function mount(root, D) {
    const F = WALK.fmt;
    const H = D.hidden_size;
    const sides = D.unit_shown;
    if (sides.length !== 2 || sides[0].side !== 'positive' || sides[1].side !== 'negative') throw new Error('m02: expected the positive then the negative pole');
    if (sides[0].unit !== sides[1].unit || sides[0].layer !== sides[1].layer) throw new Error('m02: the two poles are not of one neuron');
    const n = sides[0].unit;
    const layerNo = sides[0].layer;
    const [gR, gC] = D.matrix_shapes.gate;
    const [dR, dC] = D.matrix_shapes.down;
    if (gR !== D.units_per_layer || dC !== D.units_per_layer || gC !== H || dR !== H) throw new Error('m02: matrix shapes do not match the layer width');
    sides.forEach((s) => s.contexts.forEach((c, i) => {
      if (c.rank !== i) throw new Error('m02: contexts are not in rank order');
      if ((s.side === 'positive') !== (c.a > 0)) throw new Error('m02: a token has the other pole’s sign');
    }));
    const neuronName = 'L' + layerNo + '.N' + n;
    const poleName = (s) => neuronName + (s.side === 'positive' ? '+' : '-');

    root.replaceChildren();
    const wrap = el('div', 'm02-wrap');
    root.appendChild(wrap);

    // ---------- the three matrices ----------
    const mats = el('section', 'm02-mats-sec');
    wrap.appendChild(mats);
    mats.appendChild(el('h3', 'm02-h', 'Neuron ' + F.int(n) + ' of layer ' + layerNo + ' (' + neuronName + ') in ' + D.model));
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
      const what = (ownsRow ? 'row ' : 'column ') + F.int(n) + ' of ' + name;
      WALK.hover(hl, { title: what, rows: [['numbers', F.int(ownsRow ? c : r)], ['position', F.int(n) + ' of ' + F.int(ownsRow ? r : c) + ', from 0']]});
      hl.setAttribute('aria-label', what);
      frame.append(box, hl);
      fig.append(frame, el('span', 'm02-mname', name), el('span', 'm02-mshape', shapeText([r, c])),
        el('span', 'm02-mown', (ownsRow ? 'row ' : 'column ') + F.int(n)));
      return fig;
    };
    row.append(drawMatrix('gate', 'gate_proj', true), drawMatrix('up', 'up_proj', true), drawMatrix('down', 'down_proj', false));
    const own = el('p', 'm02-note');
    own.textContent = 'The three weight matrices of the layer’s MLP, drawn to true shape (rows × columns, numbered from 0); the highlighted line is ' +
      'the neuron’s own row or column. The neuron owns ' + F.int(D.numbers_per_unit) + ' numbers: ' + F.int(H) + ' in its row of gate_proj, ' +
      F.int(H) + ' in its row of up_proj and ' + F.int(H) + ' in its column of down_proj. The data file holds no weight values, so the picture does not draw them.';
    mats.appendChild(own);

    // ---------- the formula ----------
    const formula = el('section', 'm02-formula');
    wrap.appendChild(formula);
    [
      ['g = (row ' + F.int(n) + ' of gate_proj) · h', 'h is the residual stream at one token, ' + F.int(H) + ' numbers'],
      ['u = (row ' + F.int(n) + ' of up_proj) · h', ''],
      ['a = silu(g) · u', 'the model computes g and u in bfloat16; we computed a from them in float32'],
      ['silu(x) = x·σ(x) = x/(1 + e^(−x))', ''],
    ].forEach(([f, note]) => {
      const p = el('p', 'm02-eq');
      p.appendChild(el('code', '', f));
      if (note) p.appendChild(el('span', 'm02-eqnote', note));
      formula.appendChild(p);
    });
    formula.appendChild(el('p', 'm02-note', 'a is the neuron’s activation, and it is signed. Following the Goodfire essay, we call the tokens where a > 0 ' +
      'the neuron’s positive pole, written ' + neuronName + '+, and the tokens where a < 0 its negative pole, ' + neuronName + '-. ' +
      'Each pole has its own top-50 activating tokens: the 50 corpus tokens where a is largest (positive pole) or most negative ' +
      '(negative pole), each shown below with its surrounding text.'));

    // ---------- the two sides ----------
    const ctx = el('section', 'm02-ctx');
    wrap.appendChild(ctx);
    const controls = el('div', 'control-row');
    const toggle = el('div', 'toggle');
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Pole of the neuron');
    const btns = sides.map((s) => {
      const b = el('button', '', s.side === 'positive' ? 'Positive pole' : 'Negative pole');
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
    head.append(el('span', 'm02-k-rank', 'rank'), el('span', 'm02-k-ctx', 'the token in its text, cut to one line'), el('span', 'm02-k-a', 'a'));
    const list = el('div', 'm02-kwic');
    list.setAttribute('role', 'list');
    list.appendChild(head);
    ctx.appendChild(list);
    const more = el('button', 'button m02-more');
    more.type = 'button';
    ctx.appendChild(more);
    ctx.appendChild(el('p', 'm02-note', '↵ marks a line break in the text, ⇥ a tab. The activations are from the checkpoint at step ' +
      F.int(D.snapshot.step) + ' (' + D.snapshot.branch + '), the only checkpoint at which our files hold g and u for all of a pole’s top-50 tokens. ' +
      'The data file holds no activations at the other tokens of the text.'));

    let cur = 0;
    let all = false;
    const SIGN_TEXT = { 'g>=0,u>=0': 'g ≥ 0, u ≥ 0', 'g>=0,u<0': 'g ≥ 0, u < 0', 'g<0,u>=0': 'g < 0, u ≥ 0', 'g<0,u<0': 'g < 0, u < 0' };
    function draw() {
      const s = sides[cur];
      btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === cur)));
      summary.textContent = (s.side === 'positive' ? 'Positive pole ' : 'Negative pole ') + poleName(s) + (s.side === 'positive' ? ' (a > 0)' : ' (a < 0)') +
        ': its top-50 activating tokens, largest |a| first. The largest |a| is ' + F.fixed(s.max_abs_activation, 3) + '.';
      signs.textContent = 'Of these 50 tokens, by the signs of g and u: ' + Object.keys(SIGN_TEXT).map((k) => {
        if (!(k in s.sign_pattern_counts)) throw new Error('m02: sign pattern ' + k + ' is missing');
        return SIGN_TEXT[k] + ': ' + s.sign_pattern_counts[k];
      }).join('; ') + '. We leave tokens with g < 0 and u < 0 out of every cluster.';
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
          title: 'Rank ' + c.rank + ' of the top-50 of ' + poleName(s) + ': “' + flat(c.token) + '”',
          rows: [['g', exact(c.g)], ['u', exact(c.u)], ['a = silu(g) · u', F.fixed(c.a, 4), 'ink'],
            ['pole', s.side === 'positive' ? 'positive (a > 0)' : 'negative (a < 0)'],
            ['document, position in it', F.int(c.document) + ', ' + F.int(c.position_in_document)]],
        });
        list.appendChild(r);
      });
      more.textContent = all ? 'Show the first 10' : 'Show all 50';
      more.setAttribute('aria-expanded', String(all));
    }
    btns.forEach((b, i) => b.addEventListener('click', () => { cur = i; draw(); }));
    more.addEventListener('click', () => { all = !all; draw(); });
    draw();

    // ---------- the layer's write ----------
    const write = el('section', 'm02-write');
    wrap.appendChild(write);
    write.appendChild(el('p', 'm02-eq')).appendChild(el('code', '', 'MLP(h) = Σ over n of a_n · (column n of down_proj)'));
    write.appendChild(el('p', 'm02-note', 'The MLP’s output is the sum over all ' + F.int(D.units_per_layer) +
      ' neurons of a_n times column n of down_proj, a vector of ' + F.int(H) + ' numbers. post_feedforward_layernorm RMS-normalizes the sum and ' +
      'scales it by its weight (' + F.int(H) + ' numbers), and the block adds the result to the residual stream h, so no neuron’s write ' +
      'reaches h alone. A layer has ' + F.int(D.sides_per_layer) + ' poles; the model has ' + F.int(D.sides_total) + '.'));
  }

  WALK.register('m02', { data: 'M02_DATA', mount });
})();
