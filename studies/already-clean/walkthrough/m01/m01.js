// m01: the model and its checkpoints. Registers the mount "m01" with site.js; reads window.M01_DATA (m01/data.js).
// Top: the architecture as a paper draws it, one direction with arrows: token ids -> model.embed_tokens -> the block,
// repeated 32 times (the residual stream as one line, two branches, two adds) -> model.norm -> lm_head -> logits.
// Every box carries its Hugging Face tensor names and shapes; the parameter counts are products of those shapes and
// must sum to the banked total. Bottom: the sixteen stage-1 checkpoints by step and by tokens seen, and the later
// models named by their exact Hugging Face repository and branch.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const code = (text) => el('code', '', text);
  const prod = (shape) => shape.reduce((a, b) => a * b, 1);

  function mount(root, D) {
    const F = WALK.fmt;
    const H = D.hidden_size;
    const shapeText = (s) => s.map(F.int).join(' × ');

    // ---------- the tensors, each used exactly once ----------
    const tensors = new Map();
    D.layer_tensors_in_order.concat(D.outside_layers).forEach((t) => tensors.set(t.name, t));
    const used = new Set();
    const take = (name) => {
      const t = tensors.get(name);
      if (!t) throw new Error('m01: the data file has no tensor ' + name);
      if (used.has(name)) throw new Error('m01: tensor drawn twice: ' + name);
      used.add(name);
      return t;
    };
    const sum = (names) => names.reduce((s, n) => s + prod(tensors.get(n).shape), 0);

    const ATTN = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
      'self_attn.q_norm.weight', 'self_attn.k_norm.weight', 'self_attn.o_proj.weight'];
    const MLP = ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'];
    const NA = 'post_attention_layernorm.weight';
    const NF = 'post_feedforward_layernorm.weight';
    const EMB = 'model.embed_tokens.weight';
    const FIN = 'model.norm.weight';
    const OUT = 'lm_head.weight';

    const perBlock = sum(ATTN) + sum([NA]) + sum(MLP) + sum([NF]);
    const total = sum([EMB]) + D.layers * perBlock + sum([FIN]) + sum([OUT]);
    if (total !== D.parameters.banked) throw new Error('m01: the drawn tensors sum to ' + total + ', not ' + D.parameters.banked);
    if (D.layer_types.length !== D.layers) throw new Error('m01: layer_types has ' + D.layer_types.length + ' entries');
    const fullLayers = [];
    D.layer_types.forEach((t, i) => {
      if (t === 'full_attention') fullLayers.push(i);
      else if (t !== 'sliding_attention') throw new Error('m01: unknown layer type ' + t);
    });
    const params = (n) => F.int(n) + ' parameters';

    // One tensor row: its short name and shape; hover or tap shows the full Hugging Face name.
    const row = (name, label, inBlock) => {
      const t = take(name);
      const full = inBlock ? 'model.layers.<i>.' + name : name;
      const r = el('div', 'm01-tr');
      r.append(code(label), el('span', 'm01-shape', shapeText(t.shape)));
      r.setAttribute('aria-label', full + ', shape ' + shapeText(t.shape));
      WALK.hover(r, { title: full, rows: [['shape', shapeText(t.shape)], ['parameters', F.int(prod(t.shape))]] });
      return r;
    };
    // One component box: its Hugging Face module name, what it is, its parameter count, its tensors, a note.
    const box = (cls, name, what, count, rows, note) => {
      const b = el('div', 'm01-box ' + cls);
      const head = el('div', 'm01-bh');
      const left = el('span', 'm01-bname');
      left.appendChild(code(name));
      head.append(left, el('span', 'm01-pc', params(count)));
      b.append(head, el('p', 'm01-what', what));
      const list = el('div', 'm01-trs');
      rows.forEach((r) => list.appendChild(r));
      b.appendChild(list);
      if (note) b.appendChild(el('p', 'm01-bnote', note));
      return b;
    };
    const arrow = (label) => {
      const a = el('div', 'm01-ar' + (label ? ' lab' : ''));
      a.setAttribute('aria-hidden', 'true');
      if (label) a.appendChild(el('span', '', label));
      return a;
    };

    root.replaceChildren();
    const wrap = el('div', 'm01-wrap');
    root.appendChild(wrap);

    // ---------- the architecture ----------
    const model = el('section', 'm01-model');
    wrap.appendChild(model);
    const title = el('h3', 'm01-h');
    title.append(code(D.model), ': ' + params(D.parameters.banked));
    model.appendChild(title);
    const arch = el('div', 'm01-arch');
    model.appendChild(arch);

    arch.appendChild(el('p', 'm01-io', 'input: token ids, one per position'));
    arch.appendChild(arrow());
    const emb = tensors.get(EMB);
    arch.appendChild(box('outer', 'model.embed_tokens', 'token embedding: one row of ' + F.int(H) +
      ' numbers for each of the ' + F.int(D.vocab_size) + ' vocabulary entries', prod(emb.shape), [row(EMB, 'weight', false)]));
    arch.appendChild(arrow('residual stream h: ' + F.int(H) + ' numbers per token'));

    const block = el('div', 'm01-block');
    block.setAttribute('role', 'group');
    block.setAttribute('aria-label', 'One block, repeated ' + D.layers + ' times');
    arch.appendChild(block);
    block.appendChild(el('span', 'm01-x', '× ' + D.layers));
    const bhd = el('p', 'm01-bhd');
    bhd.append(code('model.layers.0'), ' to ', code('model.layers.' + (D.layers - 1)), ': ' + D.layers +
      ' blocks of this form, each with its own weights; ' + params(perBlock) + ' per block');
    block.appendChild(bhd);

    const sublayer = (branchBoxes, addNote) => {
      const s = el('div', 'm01-sub');
      const br = el('div', 'm01-branch');
      branchBoxes.forEach((b, i) => {
        if (i) { const a = arrow(); a.classList.add('spine'); br.appendChild(a); }
        br.appendChild(b);
      });
      const add = el('div', 'm01-add');
      add.append(el('span', 'm01-plus', '+'), el('span', 'm01-ret'), el('span', 'm01-addnote', addNote));
      add.lastChild.setAttribute('aria-hidden', 'true');
      s.append(br, add);
      return s;
    };

    const slidingLayers = D.layers - fullLayers.length;
    const attn = box('branch', 'self_attn', 'causal self-attention: ' + D.attention_heads + ' attention heads of dimension ' +
      D.head_dim + ', with ' + D.key_value_heads + ' key-value attention heads', sum(ATTN),
      [row(ATTN[0], 'q_proj', true), row(ATTN[1], 'k_proj', true), row(ATTN[2], 'v_proj', true),
       row(ATTN[3], 'q_norm', true), row(ATTN[4], 'k_norm', true), row(ATTN[5], 'o_proj', true)],
      'q_proj, k_proj and v_proj make the queries, keys and values; q_norm and k_norm apply RMSNorm to the queries and keys; ' +
      'o_proj maps the attention heads’ outputs back to ' + F.int(H) + ' numbers. Layers ' + fullLayers.join(', ') +
      ' use full attention; the other ' + slidingLayers + ' use a sliding window of ' + F.int(D.sliding_window) +
      ' tokens. Our inputs are ' + F.int(D.input_tokens_per_window) + ' tokens long, so in every layer each token attends to all earlier tokens.');
    const na = box('norm', 'post_attention_layernorm', 'RMSNorm on the attention output, before the add', sum([NA]), [row(NA, 'weight', true)]);
    block.appendChild(sublayer([attn, na], 'h ← h + RMSNorm(Attention(h))'));

    const mlp = box('branch', 'mlp', 'SwiGLU MLP with ' + F.int(D.mlp_units_per_layer) + ' neurons', sum(MLP),
      [row(MLP[0], 'gate_proj', true), row(MLP[1], 'up_proj', true), row(MLP[2], 'down_proj', true)],
      'MLP(h) = down_proj(silu(gate_proj h) ⊙ up_proj h). A neuron is one of the ' + F.int(D.mlp_units_per_layer) +
      ' entries of silu(gate_proj h) ⊙ up_proj h; its activation is one number per token.');
    const nf = box('norm', 'post_feedforward_layernorm', 'RMSNorm on the MLP output, before the add', sum([NF]), [row(NF, 'weight', true)]);
    block.appendChild(sublayer([mlp, nf], 'h ← h + RMSNorm(MLP(h))'));

    arch.appendChild(arrow());
    arch.appendChild(box('norm outer', 'model.norm', 'final RMSNorm', sum([FIN]), [row(FIN, 'weight', false)]));
    arch.appendChild(arrow());
    arch.appendChild(box('outer', 'lm_head', 'output matrix, not tied to the embedding', sum([OUT]), [row(OUT, 'weight', false)]));
    arch.appendChild(arrow());
    arch.appendChild(el('p', 'm01-io', 'output: next-token logits, ' + F.int(D.vocab_size) + ' per position'));

    if (used.size !== tensors.size) throw new Error('m01: ' + (tensors.size - used.size) + ' tensors of the data file are not drawn');

    const total_ = el('p', 'm01-sum');
    total_.textContent = 'Parameters: ' + F.int(sum([EMB])) + ' (model.embed_tokens) + ' + D.layers + ' × ' + F.int(perBlock) +
      ' (blocks) + ' + F.int(sum([FIN])) + ' (model.norm) + ' + F.int(sum([OUT])) + ' (lm_head) = ' + F.int(total) + '.';
    model.appendChild(total_);
    model.appendChild(el('p', 'm01-note', 'RMSNorm(x) = w ⊙ x / √(mean(x²) + ' + D.rms_norm_eps + '), with w the norm’s own weight of ' +
      F.int(H) + ' numbers; shapes are rows × columns as stored in the model file; hover or tap a tensor for its full name.'));

    // ---------- the checkpoints ----------
    const S = D.snapshots;
    if (S.length !== 16) throw new Error('m01: expected 16 checkpoints, got ' + S.length);
    const last = S[S.length - 1];
    const [mixStart, base, rl] = D.off_grid;
    const [b0, b1] = mixStart.between_snapshots;
    const repo = D.model;

    const time = el('section', 'm01-time');
    wrap.appendChild(time);
    time.appendChild(el('h3', 'm01-h', 'The sixteen stage-1 checkpoints we used, and the later models'));
    const intro = el('p', 'm01-note');
    intro.append('A checkpoint is the model’s weights saved at one training step. Each of the sixteen is a branch of ',
      code(repo), ' named by its step. Top row: the sixteen in order. Bottom row: the same checkpoints placed by tokens seen, step × ' +
      F.int(D.tokens_per_step) + '. Hover or tap a point.');
    time.appendChild(intro);

    const LAB = 140, YT = 156, YM = 182, YB = 268, PLOT_H = 336; // px from the plot's top
    const xOrd = (i) => ((i + 0.5) / S.length) * 100;
    const xTok = (tok) => xOrd(0) + (tok / last.tokens) * (xOrd(S.length - 1) - xOrd(0));
    const plot = el('div', 'm01-tlp');
    plot.style.height = PLOT_H + 'px';
    time.appendChild(plot);
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 100 ' + PLOT_H);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-hidden', 'true');
    plot.appendChild(svg);
    const line = (x1, y1, x2, y2, cls) => {
      const l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('class', cls);
      svg.appendChild(l);
    };
    const place = (node, x, y) => { node.style.left = x + '%'; node.style.top = y + 'px'; plot.appendChild(node); return node; };

    line(xOrd(0), YB, xTok(last.tokens), YB, 'm01-axis');
    for (let tr = 0; tr * 1e12 <= last.tokens; tr++) {
      line(xTok(tr * 1e12), YB, xTok(tr * 1e12), YB + 5, 'm01-axis');
      place(el('span', 'm01-tick', String(tr)), xTok(tr * 1e12), YB + 8);
    }
    S.forEach((s, i) => {
      line(xOrd(i), YT, xTok(s.tokens), YB, 'm01-conn');
      place(el('i', 'm01-dot low'), xTok(s.tokens), YB);
      const lab = el('span', 'm01-blab', s.branch);
      lab.style.left = xOrd(i) + '%';
      lab.style.bottom = (PLOT_H - LAB) + 'px';
      lab.setAttribute('aria-hidden', 'true');
      plot.appendChild(lab);
      const hit = place(el('div', 'm01-hit'), xOrd(i), YT);
      hit.style.width = 100 / S.length + '%';
      hit.appendChild(el('i', 'm01-dot'));
      hit.setAttribute('aria-label', 'Checkpoint ' + s.branch + ': step ' + F.int(s.step) + ', ' + F.int(s.tokens) + ' tokens seen');
      WALK.hover(hit, { title: s.branch, rows: [['training step', F.int(s.step)], ['tokens seen', F.int(s.tokens)], ['repository', repo]] });
    });

    // stage1-step476838: between two of the sixteen in the top row, at its true token count on the axis
    const xm = (xOrd(b0) + xOrd(b1)) / 2;
    line(xm, YM, xTok(mixStart.tokens), YB, 'm01-conn off');
    place(el('i', 'm01-off low'), xTok(mixStart.tokens), YB);
    const mh = place(el('div', 'm01-hit off'), xm, YM);
    mh.appendChild(el('i', 'm01-off'));
    mh.setAttribute('aria-label', 'Checkpoint ' + mixStart.branch + ': step ' + F.int(mixStart.step) + ', where the three stage-2 data mixes start');
    WALK.hover(mh, {
      title: mixStart.branch,
      rows: [['training step', F.int(mixStart.step)], ['tokens seen', F.int(mixStart.tokens)],
        ['between', S[b0].branch + ' and ' + S[b1].branch]]
        .concat(mixStart.mixes.map((m) => ['mix with ' + m.code_and_math_percent + ' % code and mathematics', m.branch])),
    });
    const ml = place(el('span', 'm01-offlab'), xTok(mixStart.tokens), YB + 30);
    ml.append(el('i', 'm01-keyoff'), code(mixStart.branch));
    place(el('span', 'm01-axis-name', 'tokens seen, trillions'), xOrd(0), YB + 52).classList.add('left');

    // the later models, by their exact names
    const keys = el('ul', 'm01-offkeys');
    const li1 = el('li');
    li1.append(el('i', 'm01-keyoff'), code(mixStart.branch), ' (step ' + F.int(mixStart.step) + ', ' + F.int(mixStart.tokens) +
      ' tokens seen, between ', code(S[b0].branch), ' and ', code(S[b1].branch), '): three stage-2 data mixes start here. Each trains ' +
      F.int(mixStart.mix_steps) + ' more steps of ' + F.int(mixStart.mix_tokens_per_step) + ' tokens, ' + F.int(mixStart.mix_tokens) + ' tokens in all:');
    const mixes = el('ul', 'm01-mixes');
    mixStart.mixes.forEach((m) => {
      const li = el('li');
      li.append(code(m.branch), ': ' + m.code_and_math_percent + ' % code and mathematics');
      mixes.appendChild(li);
    });
    li1.appendChild(mixes);
    const li2 = el('li');
    li2.append(code(base.repo), ', branch ', code(base.branch), ' (the same weights as branch ', code(base.equals_branch),
      '): the finished base model, after training stages 2 and 3.');
    const li3 = el('li');
    li3.append(code(rl.repo), ', branch ', code(rl.branch), ': a model trained further on mathematics with reinforcement learning.');
    [li1, li2, li3].forEach((li) => keys.appendChild(li));
    time.appendChild(keys);
    time.appendChild(el('p', 'm01-note', 'The last stage-1 checkpoint is ' + last.branch + ': step ' + F.int(last.step) + ', ' +
      F.int(last.tokens) + ' tokens seen. The data file holds no token count for the two later models, so the plot does not place them.'));
  }

  WALK.register('m01', { data: 'M01_DATA', mount });
})();
