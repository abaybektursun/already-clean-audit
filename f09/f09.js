// f09: gradient × activation attribution in bfloat16 against the zero-ablation's loss change measured in bfloat16 or in
// float32, one point per ablation. Reads window.F09_DATA (f09/data.js). Two mounts build the same picture: "f09" opens
// on the bfloat16 measurement (chapter 6), "f09-32" on the float32 measurement (chapter 8); in both, the toggle switches
// the measurement and a second control switches between the held-out split (the default) and the development split.
// The Spearman correlation for the shown split and measurement is drawn against the threshold set in advance and the
// permutation baseline. Hover, tap or keyboard focus shows an ablation's cluster and values; for ten held-out clusters,
// the cluster's tokens in their text.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (parent, tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    parent.appendChild(e);
    return e;
  };
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])(?::c(\d+))?$/.exec(id);
    if (!m) throw new Error('f09: cannot read the id ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), sign: m[3] === '+' ? 'positive' : 'negative', group: m[4] === undefined ? null : Number(m[4]) };
  };
  const flat = (s) => s.replace(/\r\n|\r|\n/g, '↵').replace(/\t/g, '⇥');
  const ORD = ['', '1st', '2nd', '3rd'];
  const PREC = { 16: 'bfloat16', 32: 'float32' };
  const poleId = (p) => 'L' + p.layer + '.N' + p.unit + (p.sign === 'positive' ? '+' : '-');
  const clusterName = (b) => 'Cluster ' + b.group + ' of ' + poleId(b) + ' (layer ' + b.layer + ', neuron ' + b.unit.toLocaleString('en-US') + ', ' + b.sign + ' pole)';

  // Spearman correlation: the correlation of the two rankings, ties sharing their mean rank
  const ranks = (x) => {
    const o = x.map((v, i) => i).sort((a, b) => x[a] - x[b]);
    const r = new Array(x.length);
    for (let s = 0; s < o.length;) {
      let e = s;
      while (e + 1 < o.length && x[o[e + 1]] === x[o[s]]) e++;
      for (let k = s; k <= e; k++) r[o[k]] = (s + e) / 2;
      s = e + 1;
    }
    return r;
  };
  const rankAgreement = (x, y) => {
    const a = ranks(x), b = ranks(y);
    const n = a.length;
    const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
    return sab / Math.sqrt(saa * sbb);
  };

  // both axes: log scale of the size on either side of 0, linear within ±C; the order of the values is kept
  const C = 1e-4;
  const VMAX = 5;
  const f = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v) / C);
  const FMAX = f(VMAX);
  const pos = (v) => ((f(v) + FMAX) / (2 * FMAX)) * 100;
  const TICKS = [-1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1];
  const MAJOR = new Set([-1, -0.01, 0, 0.01, 1]);

  function mount(root, D, startState) {
    const F = WALK.fmt;
    const sig = (x, n) => (x === 0 ? '0' : F.fixed(x, Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(x))))));
    const P = D.pairs;
    const N = P.half.length;
    const RA = D.rank_agreement;
    const BAR = RA.threshold_set_in_advance;

    // ---------- self-checks: the rows must give the numbers the captions state ----------
    ['birth', 'side', 'place_in_list', 'layer', 'snapshot_read', 'prediction_16bit', 'measurement_16bit', 'measurement_32bit'].forEach((k) => {
      if (P[k].length !== N) throw new Error('f09: pairs.' + k + ' has ' + P[k].length + ' rows, not ' + N);
    });
    const HALVES = ['held_out', 'rule_setting'];
    const idx = { held_out: [], rule_setting: [] };
    P.half.forEach((h, i) => {
      if (!idx[h]) throw new Error('f09: unknown split ' + h);
      idx[h].push(i);
    });
    if (idx.rule_setting.length !== 328 || idx.held_out.length !== 284) throw new Error('f09: the splits hold ' + idx.rule_setting.length + ' and ' + idx.held_out.length + ' ablations, not 328 and 284');
    const perBirth = new Map();
    P.birth.forEach((b, i) => {
      if (!(b in D.births)) throw new Error('f09: the data holds no cluster ' + b);
      if (D.births[b].half !== P.half[i]) throw new Error('f09: ' + b + ' is in the wrong split');
      if (sideOf(P.side[i]).layer !== D.births[b].layer) throw new Error('f09: an ablated pole of ' + b + ' is in another layer');
      if (P.snapshot_read[i] !== D.births[b].birth_snapshot - 1) throw new Error('f09: ' + b + ' is not measured at the checkpoint before its onset');
      perBirth.set(b, (perBirth.get(b) || []).concat(P.place_in_list[i]));
    });
    perBirth.forEach((places, b) => {
      if (places.slice().sort().join() !== '0,1,2,3') throw new Error('f09: ' + b + ' does not have its four ablated poles');
    });
    P.place_in_list.forEach((q, i) => {
      if (q === 0 && P.side[i] !== P.birth[i].split(':')[0]) throw new Error('f09: place 0 of ' + P.birth[i] + ' is not the cluster’s own pole');
    });
    const MEAS = { 16: 'measurement_16bit', 32: 'measurement_32bit' };
    const rho = {};
    [16, 32].forEach((s) => {
      rho[s] = {};
      HALVES.forEach((h) => {
        const r = rankAgreement(idx[h].map((i) => P.prediction_16bit[i]), idx[h].map((i) => P[MEAS[s]][i]));
        const banked = RA['measurement_' + s + 'bit'][h].banked;
        if (Math.abs(r - banked) > 1e-9) throw new Error('f09: Spearman correlation, ' + PREC[s] + ', ' + h + ' is ' + r + ' from the rows, ' + banked + ' saved');
        rho[s][h] = banked;
      });
    });
    const want = [[rho[16].rule_setting, '0.100'], [rho[16].held_out, '0.043'], [rho[32].held_out, '0.960'],
      [RA.measurement_32bit.band_held_out[0], '0.915'], [RA.measurement_32bit.band_held_out[1], '0.990'], [RA.measurement_32bit.shuffle_reference_held_out, '0.118']];
    want.forEach(([v, s]) => { if (F.fixed(v, 3) !== s) throw new Error('f09: the data gives ' + F.fixed(v, 3) + ' where the caption says ' + s); });
    if (BAR !== 0.7) throw new Error('f09: the threshold set in advance is ' + BAR + ', not 0.7');
    if (Math.max(...[P.prediction_16bit, P.measurement_16bit, P.measurement_32bit].flat().map(Math.abs)) > VMAX) throw new Error('f09: a value lies outside the axes');
    const STEPS = D.snapshot_steps;
    if (!STEPS || STEPS.length !== 16) throw new Error('f09: the data does not hold the sixteen checkpoint steps');
    const ctx = new Map();
    D.hover_examples_held_out.forEach((h) => {
      if (h.half !== 'held_out' || !perBirth.has(h.birth)) throw new Error('f09: the saved text of ' + h.birth + ' is not of a held-out cluster in the rows');
      ctx.set(h.birth, h.group_contexts.map((c) => {
        const m = /^([\s\S]*?)\[\[([\s\S]*?)\]\]([\s\S]*)$/.exec(c);
        if (!m || /\[\[/.test(m[3])) throw new Error('f09: a saved text of ' + h.birth + ' does not mark exactly one token');
        return { before: m[1], token: m[2], after: m[3] };
      }));
    });
    // a bootstrap interval and a permutation baseline, where the files hold them
    const extra = (s, h) => {
      const r = RA['measurement_' + s + 'bit'];
      return { band: r['band_' + h] || null, shuffle: r['shuffle_reference_' + h] === undefined ? null : r['shuffle_reference_' + h] };
    };

    // ---------- the caption: the figure may hold one sentence per measurement ----------
    const figure = root.closest('.figure');
    const captions = figure ? [...figure.querySelectorAll('.figure-caption [data-f09-state]')] : [];

    root.classList.add('f09-pic');
    root.replaceChildren();
    const wrap = el('div', 'f09-wrap');
    root.appendChild(wrap);

    const controls = el('div', 'control-row f09-controls');
    const toggle = (name, options, onPick) => {
      const box = el('div', 'f09-control');
      box.appendChild(el('span', 'control-label', name));
      const g = el('div', 'toggle');
      g.setAttribute('role', 'group');
      g.setAttribute('aria-label', name);
      const bs = options.map(([value, text]) => {
        const b = el('button', '', text);
        b.type = 'button';
        b.dataset.value = value;
        b.addEventListener('click', () => onPick(value));
        g.appendChild(b);
        return b;
      });
      box.appendChild(g);
      controls.appendChild(box);
      return (value) => bs.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.value === value)));
    };
    let state = startState;
    let half = 'held_out';
    const pressState = toggle('Measured in', [['16', 'bfloat16'], ['32', 'float32']], (v) => { state = v; update(); });
    const pressHalf = toggle('Split', [['held_out', 'held-out, ' + idx.held_out.length], ['rule_setting', 'development, ' + idx.rule_setting.length]], (v) => { half = v; update(); });
    wrap.appendChild(controls);

    const main = el('div', 'f09-main');
    wrap.appendChild(main);

    // ---------- the scatter ----------
    const sc = el('section', 'f09-scatter');
    main.appendChild(sc);
    const legend = el('p', 'f09-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f09-key ' + cls), text); return s; };
    legend.append(key('dot', 'one zero-ablation: one cluster and one ablated pole'), key('ring', 'an ablation whose cluster’s text is saved (ten held-out clusters)'),
      key('diag', 'ΔL = attribution'));
    sc.appendChild(legend);
    const yName = el('p', 'f09-yname');
    sc.appendChild(yName);
    const plot = el('div', 'f09-plot');
    plot.tabIndex = 0;
    plot.setAttribute('role', 'group');
    sc.appendChild(plot);
    const svg = svgEl(plot, 'svg', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    TICKS.forEach((v) => {
      const p = pos(v);
      svgEl(svg, 'line', { x1: p, x2: p, y1: 0, y2: 100, class: v === 0 ? 'f09-zero' : 'f09-grid' });
      svgEl(svg, 'line', { x1: 0, x2: 100, y1: 100 - p, y2: 100 - p, class: v === 0 ? 'f09-zero' : 'f09-grid' });
      const lab = (v === 0 ? '0' : sig(v, 1));
      const xt = el('span', 'f09-xtick' + (MAJOR.has(v) ? '' : ' minor'), lab);
      xt.style.left = p + '%';
      const yt = el('span', 'f09-ytick' + (MAJOR.has(v) ? '' : ' minor'), lab);
      yt.style.top = 100 - p + '%';
      plot.append(xt, yt);
    });
    svgEl(svg, 'line', { x1: 0, y1: 100, x2: 100, y2: 0, class: 'f09-diag' });
    const dots = [];
    for (let i = 0; i < N; i++) {
      const d = el('i', 'f09-dot' + (ctx.has(P.birth[i]) ? ' ring' : ''));
      d.style.left = pos(P.prediction_16bit[i]) + '%';
      plot.appendChild(d);
      dots.push(d);
    }
    sc.appendChild(el('p', 'f09-xname', 'attribution ΔL̂ in bfloat16, nats'));
    sc.appendChild(el('p', 'f09-note', 'Both axes are on a log scale of the size, on either side of 0 (linear within ±0.0001), so negative values lie left of and below 0, and positive values right of and above it. The scale keeps the order of the values, and the Spearman correlation depends only on that order.'));

    // ---------- the rank agreement ----------
    const halfName = () => (half === 'held_out' ? 'held-out split' : 'development split');
    const ag = el('section', 'f09-agree');
    main.appendChild(ag);
    ag.setAttribute('aria-live', 'polite');
    const big = el('p', 'f09-big');
    const bigWhat = el('p', 'f09-note f09-bigwhat');
    ag.append(el('h3', 'f09-h', 'Spearman correlation with the attribution'), big, bigWhat);
    const strip = el('div', 'f09-strip');
    ag.appendChild(strip);
    const A0 = -0.2, A1 = 1;
    const ax = (v) => ((v - A0) / (A1 - A0)) * 100;
    const stripRows = {};
    [16, 32].forEach((s) => {
      const r = el('div', 'f09-srow');
      const name = el('span', 'f09-sname', 'ΔL measured in ' + PREC[s]);
      const val = el('span', 'f09-sval');
      const head = el('div', 'f09-shead');
      head.append(name, val);
      const track = el('div', 'f09-strack');
      const band = el('i', 'f09-band');
      const shuf = el('i', 'f09-shuf');
      const dot = el('i', 'f09-adot');
      const zero = el('i', 'f09-tzero');
      zero.style.left = ax(0) + '%';
      const bar = el('i', 'f09-tbar');
      bar.style.left = ax(BAR) + '%';
      track.append(zero, bar, band, shuf, dot);
      r.append(head, track);
      strip.appendChild(r);
      stripRows[s] = { r, val, band, shuf, dot };
      WALK.hover(r, () => {
        const x = extra(s, half);
        return {
          title: 'ΔL measured in ' + PREC[s] + ', ' + halfName(),
          rows: [['Spearman correlation', F.fixed(rho[s][half], 3), 'ink'],
            ['bootstrap 95 % CI', x.band ? F.fixed(x.band[0], 3) + ' to ' + F.fixed(x.band[1], 3) : 'not in the files'],
            ['permutation baseline', x.shuffle === null ? 'not in the files' : F.fixed(x.shuffle, 3), 'control'],
            ['threshold set in advance', F.fixed(BAR, 1)]],
        };
      });
    });
    const sAxis = el('div', 'f09-saxis');
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((v) => {
      const t = el('span', 'f09-xtick', v === 0 || v === 1 ? String(v) : F.fixed(v, 1));
      t.style.left = ax(v) + '%';
      sAxis.appendChild(t);
    });
    strip.appendChild(sAxis);
    const sleg = el('p', 'f09-legend');
    sleg.append(key('adot', 'Spearman correlation'), key('band', 'bootstrap 95 % confidence interval, clusters resampled'), key('shuf', 'permutation baseline'), key('bar', 'threshold set in advance, ' + F.fixed(BAR, 1)));
    ag.appendChild(sleg);
    const agNote = el('p', 'f09-note');
    ag.appendChild(agNote);

    // ---------- the tokens of the chosen cluster, in their text ----------
    const cx = el('section', 'f09-ctx');
    wrap.appendChild(cx);
    const cxHead = el('h3', 'f09-h');
    const cxNote = el('p', 'f09-note');
    const cxList = el('ol', 'f09-ctxlist');
    cx.append(cxHead, cxNote, cxList);
    let ctxBirth = null;
    const showCtx = () => {
      cxList.replaceChildren();
      if (!ctxBirth) {
        cxHead.textContent = 'The cluster’s tokens in their text';
        cxNote.textContent = half === 'held_out'
          ? 'The data file holds the text around the cluster’s tokens for ten held-out clusters, drawn as rings. Hover, tap or focus a ring to list that cluster’s tokens here.'
          : 'The data file holds this text for ten clusters of the held-out split only; switch to that split to read them.';
        return;
      }
      const b = sideOf(ctxBirth);
      cxHead.textContent = clusterName(b) + ': “' + D.births[ctxBirth].group_name + '”';
      cxNote.textContent = 'The cluster’s ' + ctx.get(ctxBirth).length + ' tokens, each marked in its surrounding text as saved; the text is cut to 90 characters before and 40 after.';
      ctx.get(ctxBirth).forEach((c) => {
        const li = el('li', 'f09-ctxrow');
        const before = flat(c.before);
        const after = flat(c.after);
        li.append(el('span', 'f09-cbefore', (before.length > 90 ? '…' : '') + before.slice(-90)), el('span', 'f09-ctok', flat(c.token)),
          el('span', 'f09-cafter', after.slice(0, 40) + (after.length > 40 ? '…' : '')));
        cxList.appendChild(li);
      });
    };

    // ---------- hover, tap and keys ----------
    const shown = () => idx[half];
    const boxFor = (i) => {
      const b = sideOf(P.birth[i]);
      const s = sideOf(P.side[i]);
      const q = P.place_in_list[i];
      const rows = [
        ['zero-ablated pole', q === 0 ? 'the cluster’s own pole' : poleId(s) + ', neuron ' + F.int(s.unit) + ', ' + s.sign],
      ];
      if (q > 0) rows.push(['rank at the checkpoint before the onset', ORD[q] + ' in the layer']);
      const kOn = D.births[P.birth[i]].birth_snapshot;
      rows.push(['attribution, bfloat16', sig(P.prediction_16bit[i], 3)],
        ['ΔL, bfloat16' + (state === '16' ? ' (drawn)' : ''), sig(P.measurement_16bit[i], 3)],
        ['ΔL, float32' + (state === '32' ? ' (drawn)' : ''), sig(P.measurement_32bit[i], 3)],
        ['onset at the checkpoint of step', F.int(STEPS[kOn])], ['measured at the checkpoint of step', F.int(STEPS[P.snapshot_read[i]])]);
      const cs = ctx.get(P.birth[i]);
      if (cs) {
        const c = cs[0];
        const before = flat(c.before);
        rows.push(['“' + (before.length > 50 ? '…' : '') + before.slice(-50) + '[' + flat(c.token) + ']' + flat(c.after).slice(0, 20) + '…”', ''],
          ['cluster tokens saved', cs.length + ', listed below']);
      }
      return { title: clusterName(b) + ': “' + D.births[P.birth[i]].group_name + '”', rows };
    };
    let cur = -1;
    const mark = (i) => {
      if (cur >= 0) dots[cur].classList.remove('cur');
      cur = i;
      if (i >= 0) {
        dots[i].classList.add('cur');
        if (ctx.has(P.birth[i]) && ctxBirth !== P.birth[i]) { ctxBirth = P.birth[i]; showCtx(); }
      }
    };
    const nearest = (e) => {
      let best = -1, bd = 24 * 24;
      shown().forEach((i) => {
        const r = dots[i].getBoundingClientRect();
        const dx = r.left + r.width / 2 - e.clientX, dy = r.top + r.height / 2 - e.clientY;
        const dd = dx * dx + dy * dy;
        if (dd < bd) { bd = dd; best = i; }
      });
      return best;
    };
    plot.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const i = nearest(e);
      if (i < 0) { mark(-1); WALK.valueBox.hide(); return; }
      mark(i);
      WALK.valueBox.show(boxFor(i), e.clientX, e.clientY);
    });
    plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { mark(-1); WALK.valueBox.hide(); } });
    plot.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const i = nearest(e);
      if (i < 0) return;
      mark(i);
      WALK.valueBox.showFor(dots[i], boxFor(i));
    });
    let order = [];
    plot.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, Home: -N, End: N }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      const k = order.indexOf(cur);
      const from = k >= 0 ? k : move > 0 ? -1 : order.length;
      const i = order[Math.max(0, Math.min(order.length - 1, from + move))];
      mark(i);
      WALK.valueBox.showFor(dots[i], boxFor(i));
    });
    plot.addEventListener('blur', () => { mark(-1); WALK.valueBox.hide(); });

    function update() {
      pressState(state);
      pressHalf(half);
      captions.forEach((c) => { c.hidden = c.dataset.f09State !== state; });
      const on = new Set(shown());
      order = shown().slice().sort((a, b) => P.prediction_16bit[a] - P.prediction_16bit[b]);
      dots.forEach((d, i) => {
        d.hidden = !on.has(i);
        d.style.top = 100 - pos(P[MEAS[state]][i]) + '%';
      });
      if (cur >= 0 && !on.has(cur)) { mark(-1); WALK.valueBox.hide(); }
      if (ctxBirth && half !== 'held_out') ctxBirth = null;
      showCtx();
      cx.classList.toggle('off', half !== 'held_out');
      yName.textContent = 'ΔL measured in ' + PREC[state] + ', nats';
      plot.setAttribute('aria-label', shown().length + ' ablations of the ' + halfName() + ': bfloat16 attribution across, ΔL measured in ' + PREC[state] + ' up; arrow keys move between ablations in order of the attribution');
      big.textContent = F.fixed(rho[state][half], 3);
      bigWhat.textContent = 'The bfloat16 attribution against ΔL measured in ' + PREC[state] + ', over the ' + shown().length + ' ablations of the ' + halfName() + '.';
      [16, 32].forEach((s) => {
        const R = stripRows[s];
        const v = rho[s][half];
        const x = extra(s, half);
        R.r.classList.toggle('on', String(s) === state);
        R.dot.style.left = ax(v) + '%';
        R.val.textContent = F.fixed(v, 3) + (x.band ? ' [' + F.fixed(x.band[0], 3) + ', ' + F.fixed(x.band[1], 3) + ']' : '');
        R.band.hidden = !x.band;
        if (x.band) { R.band.style.left = ax(x.band[0]) + '%'; R.band.style.width = ax(x.band[1]) - ax(x.band[0]) + '%'; }
        R.shuf.hidden = x.shuffle === null;
        if (x.shuffle !== null) R.shuf.style.left = ax(x.shuffle) + '%';
      });
      const x = extra(Number(state), half);
      agNote.textContent = (rho[state][half] >= BAR ? 'Above' : 'Below') + ' the threshold of ' + F.fixed(BAR, 1) + ' set before the run. ' +
        (x.shuffle === null
          ? 'The files hold no bootstrap interval and no permutation baseline for this split and precision.'
          : 'Permutation baseline ' + F.fixed(x.shuffle, 3) + ': we shuffled the attribution across the ablations 1,000 times, and 97.5 % of the shuffles gave a lower Spearman correlation than this.') +
        ' The attribution is gradient × activation attribution, the first-order estimate of a zero-ablation’s loss change. Each late-onset cluster gives four zero-ablations, one at a time: its own pole and the three other poles of its layer that ranked highest on the cluster’s tokens at the checkpoint before the onset. ' +
        'A pole is one sign of a neuron’s activations; a cluster is a group of a pole’s top-50 activating tokens that an LLM named; its onset is the first checkpoint at which the pole’s median activation on those tokens is in the pole’s own top 0.1 % and stays there.';
    }
    update();
  }

  // the same picture twice in the article: opening on the bfloat16 measurement (chapter 6) and on the float32 (chapter 8)
  WALK.register('f09', { data: 'F09_DATA', mount: (root, D) => mount(root, D, '16') });
  WALK.register('f09-32', { data: 'F09_DATA', mount: (root, D) => mount(root, D, '32') });
})();
