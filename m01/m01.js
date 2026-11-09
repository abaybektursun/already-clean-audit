// m01: the model and the snapshots. Registers the mount "m01" with site.js; reads window.M01_DATA (m01/data.js).
// One layer's tensors drawn to true shape, the column of 32 layers between the embedding matrix and the output
// matrix, and the timeline of the sixteen released snapshots with the three model points outside them.
// Every number on screen is a field of M01_DATA or a product of its tensor shapes.
(() => {
  'use strict';

  // Plain names for the model file's tensor names. A tensor in the data that is not listed here throws.
  const TENSOR = {
    'self_attn.q_proj.weight': 'query matrix',
    'self_attn.k_proj.weight': 'key matrix',
    'self_attn.v_proj.weight': 'value matrix',
    'self_attn.q_norm.weight': 'query norm weight',
    'self_attn.k_norm.weight': 'key norm weight',
    'self_attn.o_proj.weight': 'attention output matrix',
    'post_attention_layernorm.weight': 'norm weight after attention',
    'mlp.gate_proj.weight': 'gate matrix', // vocab-ok: OLMo's own tensor name
    'mlp.up_proj.weight': 'up matrix',
    'mlp.down_proj.weight': 'down matrix',
    'post_feedforward_layernorm.weight': 'norm weight after the MLP',
    'model.embed_tokens.weight': 'embedding matrix',
    'lm_head.weight': 'output matrix',
    'model.norm.weight': 'final norm weight',
  };

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const prod = (shape) => shape.reduce((a, b) => a * b, 1);

  function mount(root, D) {
    const F = WALK.fmt;
    const H = D.hidden_size;
    const tensors = new Map();
    D.layer_tensors_in_order.concat(D.outside_layers).forEach((t) => {
      if (!(t.name in TENSOR)) throw new Error('m01: no plain name for tensor ' + t.name);
      tensors.set(t.name, t);
    });
    const perLayer = D.layer_tensors_in_order.reduce((s, t) => s + prod(t.shape), 0);
    const outside = D.outside_layers.reduce((s, t) => s + prod(t.shape), 0);
    if (perLayer * D.layers + outside !== D.parameters.banked) {
      throw new Error('m01: the tensor shapes sum to ' + (perLayer * D.layers + outside) + ', not ' + D.parameters.banked);
    }
    if (D.layer_types.length !== D.layers) throw new Error('m01: layer_types has ' + D.layer_types.length + ' entries');
    const shapeText = (s) => s.map(F.int).join(' × ');
    const tensorBox = (name) => {
      const t = tensors.get(name);
      return { title: TENSOR[name], rows: [['shape', shapeText(t.shape)], ['parameters', F.int(prod(t.shape))], ['name in the model file', name]] };
    };
    // A matrix or a weight vector drawn to true shape: width = columns, height = rows, in units of --u per H.
    const drawTensor = (name, caption) => {
      const t = tensors.get(name);
      const fig = el('div', 'm01-t');
      const box = el('div', t.shape.length === 2 ? 'm01-mat' : 'm01-vec');
      const rows = t.shape.length === 2 ? t.shape[0] : 1;
      const cols = t.shape.length === 2 ? t.shape[1] : t.shape[0];
      box.style.width = 'calc(var(--u) * ' + cols / H + ')';
      if (t.shape.length === 2) box.style.height = 'calc(var(--u) * ' + rows / H + ')';
      fig.append(box, el('span', 'm01-tcap', caption), el('span', 'm01-tshape', shapeText(t.shape)));
      WALK.hover(fig, tensorBox(name));
      fig.setAttribute('aria-label', TENSOR[name] + ', ' + shapeText(t.shape));
      return fig;
    };

    root.replaceChildren();
    const wrap = el('div', 'm01-wrap');
    root.appendChild(wrap);

    // ---------- the column of 32 layers ----------
    const model = el('section', 'm01-model');
    model.appendChild(el('h3', 'm01-h', D.model + ': ' + F.int(D.parameters.banked) + ' parameters'));
    const cols = el('div', 'm01-cols');
    model.appendChild(cols);
    wrap.appendChild(model);

    const stack = el('div', 'm01-stack');
    cols.appendChild(stack);
    stack.appendChild(el('p', 'm01-flow', 'tokens in'));
    const outer = (name, text) => {
      const t = tensors.get(name);
      const b = el('div', 'm01-outer');
      b.append(el('span', 'm01-outer-name', text), el('span', 'm01-tshape', shapeText(t.shape)));
      WALK.hover(b, tensorBox(name));
      b.setAttribute('aria-label', TENSOR[name] + ', ' + shapeText(t.shape));
      return b;
    };
    const embed = D.outside_layers.find((t) => t.name === 'model.embed_tokens.weight');
    const head = D.outside_layers.find((t) => t.name === 'lm_head.weight');
    const fnorm = D.outside_layers.find((t) => t.name === 'model.norm.weight');
    if (!embed || !head || !fnorm) throw new Error('m01: an embedding, output or final norm tensor is missing');
    stack.appendChild(outer(embed.name, 'embedding matrix'));
    const layersBox = el('div', 'm01-layers');
    layersBox.setAttribute('role', 'list');
    layersBox.setAttribute('aria-label', D.layers + ' layers');
    D.layer_types.forEach((type, i) => {
      const full = type === 'full_attention';
      if (!full && type !== 'sliding_attention') throw new Error('m01: unknown layer type ' + type);
      const row = el('div', 'm01-layer' + (full ? ' full' : ''));
      row.setAttribute('role', 'listitem');
      const num = el('span', 'm01-lnum', i === 0 || full ? String(i) : '');
      row.append(num, el('i', 'm01-lbar'));
      row.setAttribute('aria-label', 'Layer ' + i + (full ? ', full attention' : ', sliding-window attention'));
      WALK.hover(row, {
        title: 'Layer ' + i + ' of ' + D.layers + ' (numbered from 0)',
        rows: [
          ['attention', full ? 'full: the whole input' : 'sliding window of ' + F.int(D.sliding_window) + ' tokens'],
          ['tensors', String(D.layer_tensors_in_order.length)],
          ['parameters', F.int(perLayer)],
        ],
      });
      layersBox.appendChild(row);
    });
    stack.appendChild(layersBox);
    stack.appendChild(outer(fnorm.name, 'final norm'));
    stack.appendChild(outer(head.name, 'output matrix'));
    stack.appendChild(el('p', 'm01-flow', 'a score for each of ' + F.int(D.vocab_size) + ' tokens'));
    const legend = el('p', 'm01-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm01-key ' + cls), text); return s; };
    legend.append(key('full', 'full attention'), key('sliding', 'sliding window of ' + F.int(D.sliding_window) + ' tokens'));
    stack.appendChild(legend);

    // ---------- inside one layer ----------
    const layer = el('div', 'm01-inside');
    cols.appendChild(layer);
    layer.appendChild(el('h4', 'm01-h4', 'Inside each layer (the same tensors in all ' + D.layers + ')'));
    const hline = el('p', 'm01-h-line');
    hline.append(el('b', '', 'h'), ': the shared vector every layer reads from and adds to, ' + F.int(H) + ' numbers per token');
    layer.appendChild(hline);
    const hvec = el('div', 'm01-vec m01-hvec');
    hvec.style.width = 'calc(var(--u) * 1)';
    layer.appendChild(hvec);

    const step = (formula, note, names, extra) => {
      const s = el('div', 'm01-op');
      s.appendChild(el('p', 'm01-formula', formula));
      const row = el('div', 'm01-trow');
      names.forEach(([n, c]) => row.appendChild(drawTensor(n, c)));
      s.appendChild(row);
      if (extra) s.appendChild(extra);
      s.appendChild(el('p', 'm01-note', note));
      return s;
    };
    layer.appendChild(step('h ← h + RMSNorm(Attention(h))',
      D.attention_heads + ' attention heads of ' + D.head_dim + ' numbers each, with ' + D.key_value_heads +
      ' key and value attention heads. Layers ' + D.layer_types.map((t, i) => (t === 'full_attention' ? i : null)).filter((i) => i !== null).join(', ') +
      ' attend to the whole input; the others to a window of ' + F.int(D.sliding_window) + ' tokens, which holds the whole ' +
      F.int(D.input_tokens_per_window) + '-token input, so every layer sees every earlier token.',
      [['self_attn.q_proj.weight', 'query'], ['self_attn.k_proj.weight', 'key'], ['self_attn.v_proj.weight', 'value'],
       ['self_attn.o_proj.weight', 'output'], ['self_attn.q_norm.weight', 'query norm'], ['self_attn.k_norm.weight', 'key norm'],
       ['post_attention_layernorm.weight', 'RMSNorm weight']]));
    layer.appendChild(step('h ← h + RMSNorm(MLP(h))',
      'MLP(h) = down matrix · (silu(gate matrix · h) ⊙ (up matrix · h)): each of the ' + F.int(D.mlp_units_per_layer) + // vocab-ok
      ' units computes one number per token, a = silu(g) · u. There is no norm on a sublayer’s input; the norm is on its output, before the add.',
      [['mlp.gate_proj.weight', 'gate matrix'], ['mlp.up_proj.weight', 'up matrix'], ['mlp.down_proj.weight', 'down matrix'], // vocab-ok
       ['post_feedforward_layernorm.weight', 'RMSNorm weight']]));
    layer.appendChild(el('p', 'm01-note', 'RMSNorm(x) = w ⊙ x / √(mean(x²) + ' + D.rms_norm_eps + '), with w the norm weight of ' +
      F.int(H) + ' numbers. Shapes are rows × columns, drawn to scale; a weight vector is drawn as a line ' + F.int(H) + ' long. The weight values are not drawn.'));

    // ---------- the snapshots ----------
    const time = el('section', 'm01-time');
    wrap.appendChild(time);
    time.appendChild(el('h3', 'm01-h', 'The sixteen released snapshots of training stage 1'));
    time.appendChild(el('p', 'm01-note', 'Tokens seen = step × ' + F.int(D.tokens_per_step) + '. Top row: the snapshots in order; ' +
      'bottom row: the same snapshots at their true token count. Hover or tap a point.'));
    const S = D.snapshots;
    if (S.length !== 16) throw new Error('m01: expected 16 snapshots, got ' + S.length);
    const last = S[S.length - 1];
    const [mixStart, base, rl] = D.off_grid;
    const [b0, b1] = mixStart.between_snapshots;
    const SPAN = 80; // percent of the width given to stage 1; the rest holds the two models after stage 1
    const xOrd = (i) => ((i + 0.5) / S.length) * SPAN;
    const xTok = (tok) => xOrd(0) + (tok / last.tokens) * (xOrd(S.length - 1) - xOrd(0));
    const plot = el('div', 'm01-tl');
    time.appendChild(plot);
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-hidden', 'true');
    plot.appendChild(svg);
    const line = (x1, y1, x2, y2, cls) => {
      const l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('class', cls);
      svg.appendChild(l);
    };
    const YT = 24, YB = 74; // percent of the plot height: the ordered row and the token axis
    line(xOrd(0), YB, xTok(last.tokens), YB, 'm01-axis');
    const place = (node, x, y) => { node.style.left = x + '%'; node.style.top = y + '%'; plot.appendChild(node); return node; };
    const tokText = (n) => F.int(n) + ' tokens';
    // token axis ticks, in trillions
    for (let tr = 0; tr * 1e12 <= last.tokens; tr++) {
      line(xTok(tr * 1e12), YB, xTok(tr * 1e12), YB + 4, 'm01-axis');
      place(el('span', 'm01-tick', String(tr)), xTok(tr * 1e12), YB + 6);
    }
    place(el('span', 'm01-axis-name', 'tokens seen, trillions'), xOrd(0), 100).classList.add('left');
    S.forEach((s, i) => {
      line(xOrd(i), YT, xTok(s.tokens), YB, 'm01-conn');
      place(el('i', 'm01-dot low'), xTok(s.tokens), YB);
      place(el('span', 'm01-tick top' + (i % 5 === 0 || i === S.length - 1 ? '' : ' minor'), String(i)), xOrd(i), YT - 20);
      const hit = place(el('div', 'm01-hit'), xOrd(i), YT);
      hit.style.width = SPAN / S.length + '%';
      hit.appendChild(el('i', 'm01-dot'));
      hit.setAttribute('aria-label', 'Snapshot ' + i);
      WALK.hover(hit, { title: 'Snapshot ' + s.snapshot + ' of the sixteen', rows: [['step', F.int(s.step)], ['tokens seen', F.int(s.tokens)], ['branch', s.branch]] });
    });
    place(el('span', 'm01-tick top-name', 'snapshot'), xOrd(0), YT - 34).classList.add('left');
    // the start of the three data mixes: between snapshots b0 and b1, at its true token count below
    const xm = (xOrd(b0) + xOrd(b1)) / 2;
    line(xm, YT + 12, xTok(mixStart.tokens), YB, 'm01-conn off');
    place(el('i', 'm01-off low'), xTok(mixStart.tokens), YB);
    const mixBox = {
      title: 'a: ' + mixStart.name,
      rows: [['step', F.int(mixStart.step)], ['tokens seen', F.int(mixStart.tokens)], ['between snapshots', b0 + ' and ' + b1]]
        .concat(mixStart.mixes.map((m) => ['mix with ' + m.code_and_math_percent + ' % code and math', F.int(mixStart.mix_steps) + ' steps']))
        .concat([['tokens per mix step', F.int(mixStart.mix_tokens_per_step)], ['tokens per mix', F.int(mixStart.mix_tokens)]]),
    };
    const offHit = (x, y, letter, content) => {
      const h = place(el('div', 'm01-hit off'), x, y);
      h.append(el('i', 'm01-off'), el('span', 'm01-letter', letter));
      h.setAttribute('aria-label', content.title);
      WALK.hover(h, content);
      h.classList.add('below');
      return h;
    };
    offHit(xm, YT + 12, 'a', mixBox);
    // after stage 1: no token count in the data, so these two sit in their own zone, off the token axis
    const zone = place(el('span', 'm01-zone', 'after stage 1'), 100, YT - 34);
    zone.setAttribute('aria-hidden', 'true');
    line(SPAN + 3, 4, SPAN + 3, 96, 'm01-sep');
    offHit(SPAN + 7, YT, 'b', { title: 'b: ' + base.name, rows: [['model', base.repo], ['branch', base.branch + ' = ' + base.equals_branch], ['commit', base.commit.slice(0, 12)]] });
    offHit(SPAN + 15, YT, 'c', { title: 'c: ' + rl.name, rows: [['model', rl.repo], ['branch', rl.branch], ['commit', rl.commit.slice(0, 12)]] });
    const keys = el('ul', 'm01-offkeys');
    [['a', mixStart.name + ': step ' + F.int(mixStart.step) + ', ' + tokText(mixStart.tokens) + ', between snapshots ' + b0 + ' and ' + b1 +
       '; three mixes of ' + mixStart.mixes.map((m) => m.code_and_math_percent + ' %').join(', ') + ' code and math continue from it for ' +
       F.int(mixStart.mix_tokens) + ' tokens each'],
     ['b', base.name + ' (' + base.repo + ')'],
     ['c', rl.name + ' (' + rl.repo + ')']].forEach(([l, t]) => {
      const li = el('li');
      li.append(el('span', 'm01-keyoff'), el('b', '', l + ' '), t);
      keys.appendChild(li);
    });
    time.appendChild(keys);
    time.appendChild(el('p', 'm01-note', 'The last snapshot: step ' + F.int(last.step) + ', ' + tokText(last.tokens) +
      '. The models b and c come after stage 1; their token counts are not in the data file, so they stand apart from the token axis.'));
  }

  WALK.register('m01', { data: 'M01_DATA', mount });
})();
