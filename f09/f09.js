// f09: the 16-bit gradient prediction against the 16-bit or the 32-bit switch-off measurement, one point per pair.
// Reads window.F09_DATA (f09/data.js). Two mounts build the same picture: "f09" opens on the 16-bit measurement
// (chapter 6), "f09-32" on the 32-bit measurement (chapter 8); in both, the toggle switches the measurement and a
// second control switches between the held-out half (the default) and the half used to set the rules. The rank
// agreement for the shown half and measurement is drawn against the threshold set in advance and the shuffle reference.
// Hover, tap or keyboard focus shows a pair's birth and values; for ten held-out births, the group's contexts.
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

  // rank agreement: the correlation of the two rankings, ties sharing their mean rank
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
      if (!idx[h]) throw new Error('f09: unknown half ' + h);
      idx[h].push(i);
    });
    if (idx.rule_setting.length !== 328 || idx.held_out.length !== 284) throw new Error('f09: the halves hold ' + idx.rule_setting.length + ' and ' + idx.held_out.length + ' pairs, not 328 and 284');
    const perBirth = new Map();
    P.birth.forEach((b, i) => {
      if (!(b in D.births)) throw new Error('f09: no record of the birth ' + b);
      if (D.births[b].half !== P.half[i]) throw new Error('f09: ' + b + ' is in the wrong half');
      if (sideOf(P.side[i]).layer !== D.births[b].layer) throw new Error('f09: a switched-off side of ' + b + ' is in another layer');
      if (P.snapshot_read[i] !== D.births[b].birth_snapshot - 1) throw new Error('f09: ' + b + ' is not read at the snapshot before its birth');
      perBirth.set(b, (perBirth.get(b) || []).concat(P.place_in_list[i]));
    });
    perBirth.forEach((places, b) => {
      if (places.slice().sort().join() !== '0,1,2,3') throw new Error('f09: ' + b + ' does not have its four switched-off sides');
    });
    P.place_in_list.forEach((q, i) => {
      if (q === 0 && P.side[i] !== P.birth[i].split(':')[0]) throw new Error('f09: place 0 of ' + P.birth[i] + ' is not the carrier');
    });
    const MEAS = { 16: 'measurement_16bit', 32: 'measurement_32bit' };
    const rho = {};
    [16, 32].forEach((s) => {
      rho[s] = {};
      HALVES.forEach((h) => {
        const r = rankAgreement(idx[h].map((i) => P.prediction_16bit[i]), idx[h].map((i) => P[MEAS[s]][i]));
        const banked = RA['measurement_' + s + 'bit'][h].banked;
        if (Math.abs(r - banked) > 1e-9) throw new Error('f09: rank agreement ' + s + '-bit, ' + h + ' is ' + r + ' from the rows, ' + banked + ' banked');
        rho[s][h] = banked;
      });
    });
    const want = [[rho[16].rule_setting, '0.100'], [rho[16].held_out, '0.043'], [rho[32].held_out, '0.960'],
      [RA.measurement_32bit.band_held_out[0], '0.915'], [RA.measurement_32bit.band_held_out[1], '0.990'], [RA.measurement_32bit.shuffle_reference_held_out, '0.118']];
    want.forEach(([v, s]) => { if (F.fixed(v, 3) !== s) throw new Error('f09: the data gives ' + F.fixed(v, 3) + ' where the caption says ' + s); });
    if (BAR !== 0.7) throw new Error('f09: the threshold set in advance is ' + BAR + ', not 0.7');
    if (Math.max(...[P.prediction_16bit, P.measurement_16bit, P.measurement_32bit].flat().map(Math.abs)) > VMAX) throw new Error('f09: a value lies outside the axes');
    const ctx = new Map();
    D.hover_examples_held_out.forEach((h) => {
      if (h.half !== 'held_out' || !perBirth.has(h.birth)) throw new Error('f09: the contexts of ' + h.birth + ' are not of a held-out birth in the rows');
      ctx.set(h.birth, h.group_contexts.map((c) => {
        const m = /^([\s\S]*?)\[\[([\s\S]*?)\]\]([\s\S]*)$/.exec(c);
        if (!m || /\[\[/.test(m[3])) throw new Error('f09: a context of ' + h.birth + ' does not mark exactly one position');
        return { before: m[1], token: m[2], after: m[3] };
      }));
    });
    // a band and a shuffle reference, where the files hold them
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
    const pressState = toggle('Measured in', [['16', '16-bit'], ['32', '32-bit']], (v) => { state = v; update(); });
    const pressHalf = toggle('Pairs', [['held_out', 'held-out half, ' + idx.held_out.length], ['rule_setting', 'rule-setting half, ' + idx.rule_setting.length]], (v) => { half = v; update(); });
    wrap.appendChild(controls);

    const main = el('div', 'f09-main');
    wrap.appendChild(main);

    // ---------- the scatter ----------
    const sc = el('section', 'f09-scatter');
    main.appendChild(sc);
    const legend = el('p', 'f09-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f09-key ' + cls), text); return s; };
    legend.append(key('dot', 'one pair: a birth and one switched-off side'), key('ring', 'a pair whose birth’s contexts are banked (ten held-out births)'),
      key('diag', 'measurement = prediction'));
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
    sc.appendChild(el('p', 'f09-xname', '16-bit prediction ΔL̂, nats'));
    sc.appendChild(el('p', 'f09-note', 'Both axes are on a log scale of the size, on either side of 0 (linear within ±0.0001), so a value’s sign is the side of 0 it falls on. The scale keeps the order of the values, and rank agreement reads only that order.'));

    // ---------- the rank agreement ----------
    const halfName = () => (half === 'held_out' ? 'held-out half' : 'half used to set the rules');
    const ag = el('section', 'f09-agree');
    main.appendChild(ag);
    ag.setAttribute('aria-live', 'polite');
    const big = el('p', 'f09-big');
    const bigWhat = el('p', 'f09-note f09-bigwhat');
    ag.append(el('h3', 'f09-h', 'Rank agreement with the prediction'), big, bigWhat);
    const strip = el('div', 'f09-strip');
    ag.appendChild(strip);
    const A0 = -0.2, A1 = 1;
    const ax = (v) => ((v - A0) / (A1 - A0)) * 100;
    const stripRows = {};
    [16, 32].forEach((s) => {
      const r = el('div', 'f09-srow');
      const name = el('span', 'f09-sname', s + '-bit measurement');
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
          title: s + '-bit measurement, ' + halfName(),
          rows: [['rank agreement', F.fixed(rho[s][half], 3), 'ink'],
            ['band', x.band ? F.fixed(x.band[0], 3) + ' to ' + F.fixed(x.band[1], 3) : 'not in the files'],
            ['shuffle reference', x.shuffle === null ? 'not in the files' : F.fixed(x.shuffle, 3), 'control'],
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
    sleg.append(key('adot', 'rank agreement'), key('band', 'band: the range consistent with the data, births resampled'), key('shuf', 'shuffle reference'), key('bar', 'threshold set in advance, ' + F.fixed(BAR, 1)));
    ag.appendChild(sleg);
    const agNote = el('p', 'f09-note');
    ag.appendChild(agNote);

    // ---------- the contexts of the chosen birth ----------
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
        cxHead.textContent = 'The group’s contexts';
        cxNote.textContent = half === 'held_out'
          ? 'The contexts are banked for ten held-out births, drawn as rings. Hover, tap or focus a ring to list its group’s contexts here.'
          : 'The contexts are banked for ten births of the held-out half only; switch to that half to read them.';
        return;
      }
      const b = sideOf(ctxBirth);
      cxHead.textContent = 'Layer ' + b.layer + ', unit ' + F.int(b.unit) + ', ' + b.sign + ' side, group ' + b.group + ': “' + D.births[ctxBirth].group_name + '”';
      cxNote.textContent = 'The group’s ' + ctx.get(ctxBirth).length + ' contexts as banked, the group’s position marked; texts cut to 90 characters before and 40 after.';
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
        ['side switched off', q === 0 ? 'the carrier' : 'unit ' + F.int(s.unit) + ', ' + s.sign],
      ];
      if (q > 0) rows.push(['its rank one snapshot before', ORD[q] + ' of the layer']);
      rows.push(['16-bit prediction', sig(P.prediction_16bit[i], 3)],
        ['16-bit measurement' + (state === '16' ? ' (drawn)' : ''), sig(P.measurement_16bit[i], 3)],
        ['32-bit measurement' + (state === '32' ? ' (drawn)' : ''), sig(P.measurement_32bit[i], 3)],
        ['born at snapshot', D.births[P.birth[i]].birth_snapshot + ', read at ' + P.snapshot_read[i]]);
      const cs = ctx.get(P.birth[i]);
      if (cs) {
        const c = cs[0];
        const before = flat(c.before);
        rows.push(['“' + (before.length > 50 ? '…' : '') + before.slice(-50) + '[' + flat(c.token) + ']' + flat(c.after).slice(0, 20) + '…”', ''],
          ['contexts banked', cs.length + ', listed below']);
      }
      return { title: 'Layer ' + b.layer + ', unit ' + F.int(b.unit) + ', ' + b.sign + ' side, group ' + b.group + ': “' + D.births[P.birth[i]].group_name + '”', rows };
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
      yName.textContent = state + '-bit measurement ΔL, nats';
      plot.setAttribute('aria-label', shown().length + ' pairs of the ' + halfName() + ': 16-bit prediction across, ' + state + '-bit measurement up; arrow keys move between pairs in order of the prediction');
      big.textContent = F.fixed(rho[state][half], 3);
      bigWhat.textContent = 'The 16-bit prediction against the ' + state + '-bit measurement, over the ' + shown().length + ' pairs of the ' + halfName() + '.';
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
          ? 'The files hold no band and no shuffle reference for this half and measurement.'
          : 'Shuffle reference ' + F.fixed(x.shuffle, 3) + ': with the prediction shuffled across the pairs 1,000 times, 97.5 % of the shuffles agree less than this.') +
        ' Each birth gives four pairs: the carrier and the three other sides of its layer that ranked highest one snapshot before the birth, each switched off in turn.';
    }
    update();
  }

  // the same picture twice in the article: opening on the 16-bit measurement (chapter 6) and on the 32-bit (chapter 8)
  WALK.register('f09', { data: 'F09_DATA', mount: (root, D) => mount(root, D, '16') });
  WALK.register('f09-32', { data: 'F09_DATA', mount: (root, D) => mount(root, D, '32') });
})();
