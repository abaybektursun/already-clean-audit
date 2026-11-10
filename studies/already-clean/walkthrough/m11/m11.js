// m11: copying part of the weight delta onto the base model, and how each condition is built. Registers the mount "m11"
// with site.js; reads window.M11_DATA (m11/data.js). The build formula; a menu of constructions and a toggle of sizes;
// the weight-delta norm of all 352,256 MLP neurons as a sorted curve (399 saved points) with the chosen condition's
// mask on the ranking; the squared delta norm each condition adds against the whole MLP delta; where its neurons sit by
// layer and along one layer; the delta by tensor family and which model each family comes from; and how one MATH-500
// score is taken, with its noise.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svgEl = (parent, tag, attrs) => {
    const e = document.createElementNS(SVGNS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    parent.appendChild(e);
    return e;
  };
  const NB = ' ';
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const pow10 = (k) => (k === 0 ? '1' : k === 1 ? '10' : k === 2 ? '100' : k === 3 ? '1,000' : k === 4 ? '10,000' : k === 5 ? '100,000' : '10' + String(k).replace(/./g, (c) => SUP[c]));

  const CONS = [
    { id: 'most', name: 'the top-k neurons by weight-delta norm', sized: true },
    { id: 'scalar', name: 'random neurons matched on squared delta norm, one scalar', sized: true },
    { id: 'own', name: 'random neurons matched on squared delta norm, at their own delta', sized: true },
    { id: 'sample', name: 'DARE on the MLP delta: keep a random share q, rescale by 1/q', sized: true },
    { id: 'all', name: 'the whole MLP delta', sized: false },
    { id: 'attn', name: 'the attention and norm-weight deltas', sized: false },
    { id: 'minus', name: 'RL-Zero-Math with its top-k neurons reset to the base model', sized: true },
  ];
  const FAM = [
    { key: 'mlp', name: 'MLP' },
    { key: 'attn', name: 'attention' },
    { key: 'norm', name: 'norm weights' },
    { key: 'embed', name: 'embeddings and output matrix' },
  ];

  function mount(root, D) {
    const F = WALK.fmt;
    const U = D.units_total;
    const W = D.units_per_layer;
    const NL = D.layers;
    const EPU = D.entries_per_unit;
    const E0 = D.whole_mlp_change_energy;
    const S = D.sizes;
    const PL = D.per_layer.units_per_set;
    const ONE = D.one_layer;
    const pts = D.sorted_curve.points;
    const N = D.noise_of_one_reading;
    const fam = D.change_by_tensor_family;
    const tag = (p) => String(p).replace('.', '_') + 'pct';
    const pctText = (p) => String(p) + NB + '%';
    const fmtE = (e) => (e >= 100 ? F.fixed(e, 1) : e >= 1 ? F.fixed(e, 3) : F.fixed(e, 4));
    const sig3 = (x) => x.toPrecision(3);

    // ---------- the data must hold together before anything is drawn ----------
    if (NL * W !== U) throw new Error('m11: ' + NL + ' layers x ' + W + ' is not ' + U);
    if (pts[0][0] !== 1 || pts[pts.length - 1][0] !== U) throw new Error('m11: the sorted curve does not run from rank 1 to ' + U);
    pts.forEach((p, i) => { if (i && !(p[0] > pts[i - 1][0] && p[1] <= pts[i - 1][1])) throw new Error('m11: the sorted curve is not sorted at point ' + i); });
    const setKeys = (p) => ({ most: 'most_changed_' + tag(p), scalar: 'random_equal_energy_scaled_' + tag(p), own: 'random_equal_energy_own_change_' + tag(p) });
    S.forEach((s) => {
      const k = setKeys(s.percent_of_units);
      const top = D.top_sets_from_curve[pctText(s.percent_of_units).replace(NB, ' ')];
      if (!top || top.units !== s.units) throw new Error('m11: the curve’s top set at ' + s.percent_of_units + ' % differs from the size block');
      if (Math.abs(top.energy - s.most_changed.energy) > 1e-12) throw new Error('m11: the top-k squared delta norm differs between blocks at ' + s.percent_of_units + ' %');
      if (s.units !== Math.round((s.percent_of_units / 100) * U)) throw new Error('m11: ' + s.units + ' neurons is not ' + s.percent_of_units + ' % of ' + U);
      [['most', s.units], ['scalar', s.random_equal_energy_scaled.units], ['own', s.random_equal_energy_own_change.units]].forEach(([c, n]) => {
        const per = PL[k[c]];
        if (!per || per.length !== NL) throw new Error('m11: no per-layer counts for ' + k[c]);
        const tot = per.reduce((a, b) => a + b, 0);
        if (tot !== n) throw new Error('m11: ' + k[c] + ' holds ' + tot + ' neurons by layer, the size block says ' + n);
        if (ONE.unit_indices[k[c]].length !== per[ONE.layer]) throw new Error('m11: ' + k[c] + ' layer ' + ONE.layer + ' indices do not match its count');
      });
      if (Math.abs(s.scaled_sample.scale * s.scaled_sample.keep_probability - 1) > 1e-9) throw new Error('m11: the sample scale is not 1/q at ' + s.percent_of_units + ' %');
    });
    const famSum = FAM.reduce((a, f) => a + fam[f.key].energy, 0);
    FAM.forEach((f) => { if (Math.abs(fam[f.key].energy / famSum - fam[f.key].share_of_energy) > 1e-9) throw new Error('m11: the ' + f.key + ' share of the squared delta norm does not add up'); });
    if (Math.abs(fam.mlp.energy - E0) > 1e-9) throw new Error('m11: the MLP family’s squared delta norm is not the whole MLP delta’s');

    // ---------- one condition at one size: everything the picture draws for it ----------
    function condition(id, si) {
      const s = S[si];
      const k = setKeys(s.percent_of_units);
      const allIdx = () => Array.from({ length: W }, (_, i) => i);
      const base = { id, size: s, sized: CONS.find((c) => c.id === id).sized };
      if (id === 'most') {
        return { ...base, units: s.units, entries: s.units * EPU, c: 1, energy: s.most_changed.realized_mlp_energy, built: true,
          ranks: [1, s.units], kind: 'exact', perLayer: PL[k.most], idx: ONE.unit_indices[k.most], from: 'base',
          mask: 'the ' + F.int(s.units) + ' neurons at ranks 1 to ' + F.int(s.units) + ', every entry of each' };
      }
      if (id === 'scalar') {
        const r = s.random_equal_energy_scaled;
        return { ...base, units: r.units, entries: r.units * EPU, c: r.scale_built, cPlanned: r.scale_planned, energy: r.realized_mlp_energy, built: true,
          ranks: [s.units + 1, U], kind: 'random', perLayer: PL[k.scalar], idx: ONE.unit_indices[k.scalar], from: 'base',
          mask: F.int(r.units) + ' neurons drawn at random from ranks ' + F.int(s.units + 1) + ' to ' + F.int(U) };
      }
      if (id === 'own') {
        const r = s.random_equal_energy_own_change;
        return { ...base, units: r.units, entries: r.units * EPU, c: 1, energy: r.realized_mlp_energy === null ? r.energy : r.realized_mlp_energy,
          built: r.realized_mlp_energy !== null, ranks: [s.units + 1, U], kind: 'random', perLayer: PL[k.own], idx: ONE.unit_indices[k.own], from: 'base',
          mask: F.int(r.units) + ' neurons drawn at random from ranks ' + F.int(s.units + 1) + ' to ' + F.int(U) + ', added until their squared delta norm reached the top-k set’s' };
      }
      if (id === 'sample') {
        const r = s.scaled_sample;
        return { ...base, units: null, entries: r.entries_kept, c: r.scale, q: r.keep_probability, energy: r.realized_mlp_energy, built: true,
          ranks: [1, U], kind: 'entries', perLayer: null, idx: null, from: 'base',
          mask: F.int(r.entries_kept) + ' of the ' + F.int(fam.mlp.entries) + ' MLP entries, each kept with probability q = ' + sig3(r.keep_probability) + ' (expected ' + F.int(r.entries_expected) + ')' };
      }
      if (id === 'all') {
        return { ...base, units: U, entries: fam.mlp.entries, c: 1, energy: E0, built: true, ranks: [1, U], kind: 'exact',
          perLayer: Array(NL).fill(W), idx: allIdx(), from: 'base', mask: 'all ' + F.int(U) + ' neurons, every entry' };
      }
      if (id === 'attn') {
        return { ...base, units: 0, entries: 0, c: null, energy: 0, built: true, ranks: null, kind: 'none',
          perLayer: Array(NL).fill(0), idx: [], from: 'attn', mask: 'no MLP entry' };
      }
      const most = new Set(ONE.unit_indices[k.most]);
      return { ...base, units: U - s.units, entries: (U - s.units) * EPU, c: 1, energy: s.most_changed.rest_energy, built: false,
        ranks: [s.units + 1, U], kind: 'exact', perLayer: PL[k.most].map((n) => W - n), idx: allIdx().filter((i) => !most.has(i)), from: 'math',
        mask: 'the ' + F.int(U - s.units) + ' neurons at ranks ' + F.int(s.units + 1) + ' to ' + F.int(U) + '; the top ' + F.int(s.units) + ' stay at the base model' };
    }
    const fromFor = (cond, fk) => {
      if (fk === 'mlp') return cond.id === 'attn' ? 'base' : 'built';
      if (cond.from === 'math') return 'math';
      if (cond.from === 'attn') return fk === 'attn' || fk === 'norm' ? 'math' : 'base';
      return 'base';
    };

    root.replaceChildren();
    const wrap = el('div', 'm11-wrap');
    root.appendChild(wrap);

    // ---------- the formula ----------
    const formula = el('div', 'm11-formula');
    formula.appendChild(el('p', 'm11-fline', 'Δ = Olmo-3.1-7B-RL-Zero-Math − Olmo-3-1025-7B, the weight delta, tensor by tensor, in float32'));
    formula.appendChild(el('p', 'm11-fline', 'sₙ = √(‖Δ gate_proj row‖² + ‖Δ up_proj row‖² + ‖Δ down_proj column‖²), over ' + F.int(EPU) + ' entries per neuron'));
    formula.appendChild(el('p', 'm11-fline', 'built model = bfloat16( base + c ⊙ M ⊙ Δ ), per tensor'));
    wrap.appendChild(formula);
    wrap.appendChild(el('p', 'm11-note', 'The base model is Olmo-3-1025-7B; Olmo-3.1-7B-RL-Zero-Math is the same model after reinforcement learning on mathematics. ' +
      'sₙ is a neuron’s weight-delta norm; we rank all ' + F.int(U) + ' MLP neurons of the ' + NL + ' layers together by it. ' +
      'M is a mask of entries (1 where we copy the delta) and c a scale. Each condition below is one choice of M and c, and of which model the other tensors come from. ' +
      'The size k is a share of the ' + F.int(U) + ' neurons.'));

    // ---------- controls ----------
    const controls = el('div', 'control-row m11-controls');
    const pick = el('label', 'm11-pick');
    pick.appendChild(el('span', 'control-label', 'Condition'));
    const sel = el('select', 'm11-select');
    CONS.forEach((c) => { const o = el('option', '', c.name); o.value = c.id; sel.appendChild(o); });
    pick.appendChild(sel);
    const sizeBox = el('div', 'm11-size');
    sizeBox.appendChild(el('span', 'control-label', 'Size'));
    const toggle = el('div', 'toggle');
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Size k, in percent of the ' + F.int(U) + ' neurons');
    const sizeBtns = S.map((s, i) => {
      const b = el('button', '', pctText(s.percent_of_units));
      b.type = 'button';
      b.addEventListener('click', () => { si = i; draw(); });
      toggle.appendChild(b);
      return b;
    });
    sizeBox.appendChild(toggle);
    controls.append(pick, sizeBox);
    wrap.appendChild(controls);

    // ---------- the chosen condition, in numbers ----------
    const sumBox = el('div', 'm11-sum');
    sumBox.setAttribute('aria-live', 'polite');
    wrap.appendChild(sumBox);

    // ---------- the ranking ----------
    const rank = el('section', 'm11-sec');
    wrap.appendChild(rank);
    rank.appendChild(el('h3', 'm11-h', 'The ranking: every neuron’s weight-delta norm, largest first'));
    const rLegend = el('p', 'm11-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'm11-key ' + cls), text); return s; };
    rLegend.append(key('pt', 'one of the ' + pts.length + ' saved points of the sorted curve'), key('ptc', 'inside the mask'), key('band', 'the mask: every neuron in the band'), key('bandr', 'neurons drawn at random from the band'));
    rank.appendChild(rLegend);
    const plot = el('div', 'm11-plot');
    plot.tabIndex = 0;
    plot.setAttribute('role', 'group');
    plot.setAttribute('aria-label', 'The sorted weight-delta norms of all ' + F.int(U) + ' neurons by rank; arrow keys move between the saved points');
    rank.appendChild(plot);
    const LMAX = Math.log10(U);
    const xr = (r) => (Math.log10(r) / LMAX) * 100;
    const YMAX = Math.ceil(pts[0][1] / 0.005) * 0.005;
    const ys = (v) => (1 - v / YMAX) * 100;
    for (let v = 0; v <= YMAX + 1e-12; v += 0.005) {
      const g = el('div', 'm11-grid');
      g.style.top = ys(v) + '%';
      g.appendChild(el('span', 'm11-ytick', F.fixed(v, 3)));
      plot.appendChild(g);
    }
    const cutLine = el('div', 'm11-vline');
    const cutLab = el('span', 'm11-vlab');
    plot.append(cutLine, cutLab);
    const svg = svgEl(plot, 'svg', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true', class: 'm11-svg' });
    const dotEls = pts.map(([r, v]) => svgEl(svg, 'line', { x1: xr(r), x2: xr(r), y1: ys(v), y2: ys(v), class: 'm11-pt' }));
    const cursor = el('i', 'm11-cursor');
    cursor.hidden = true;
    plot.appendChild(cursor);
    const xaxis = el('div', 'm11-xaxis');
    for (let k = 0; k <= Math.floor(LMAX); k++) {
      const t = el('span', 'm11-xtick' + (k % 2 ? ' odd' : ''), pow10(k));
      t.style.left = xr(10 ** k) + '%';
      xaxis.appendChild(t);
    }
    const tEnd = el('span', 'm11-xtick end', F.int(U));
    tEnd.style.left = '100%';
    xaxis.appendChild(tEnd);
    rank.appendChild(xaxis);
    rank.appendChild(el('p', 'm11-xname', 'rank among the ' + F.int(U) + ' neurons (log scale); up: sₙ'));
    const maskLane = el('div', 'm11-masklane');
    const maskBand = el('i', 'm11-maskband');
    maskLane.appendChild(maskBand);
    rank.appendChild(maskLane);
    const maskText = el('p', 'm11-masktext');
    rank.appendChild(maskText);

    // ---------- the squared change each condition adds ----------
    const en = el('section', 'm11-sec');
    wrap.appendChild(en);
    const enHead = el('h3', 'm11-h');
    en.appendChild(enHead);
    en.appendChild(el('p', 'm11-note', 'The squared delta norm is the sum of the squares of the entries added to the base model (Σ sₙ² when whole neurons are copied at c = 1); the whole MLP delta’s is ' + fmtE(E0) +
      '. A filled dot is the value the build measured on the built model; a ring is the planned value, where no build file was saved. Click a row, or press Enter on it, to choose that condition.'));
    const eLegend = el('p', 'm11-legend');
    eLegend.append(key('edot', 'the condition’s squared delta norm'), key('ering', 'planned, no build file'), key('whole', 'the whole MLP delta, ' + fmtE(E0)));
    en.appendChild(eLegend);
    const eplot = el('div', 'm11-eplot');
    en.appendChild(eplot);
    const eAxis = el('div', 'm11-eaxis');
    en.appendChild(eAxis);
    en.appendChild(el('p', 'm11-xname m11-ename', 'squared delta norm added to the base model (log scale)'));
    const ELO = -3;
    const EHI = 5;
    const ex = (e) => ((Math.log10(e) - ELO) / (EHI - ELO)) * 100;
    for (let k = ELO; k <= EHI; k++) {
      const t = el('span', 'm11-xtick' + ((k - ELO) % 2 ? ' odd' : '') + (k === EHI ? ' end' : k === ELO ? ' start' : ''), k < 0 ? (10 ** k).toFixed(-k) : pow10(k));
      t.style.left = ex(10 ** k) + '%';
      eAxis.appendChild(t);
    }

    // ---------- where the units sit ----------
    const lay = el('section', 'm11-sec');
    wrap.appendChild(lay);
    const layHead = el('h3', 'm11-h');
    lay.appendChild(layHead);
    const lplot = el('div', 'm11-lplot');
    lay.appendChild(lplot);
    const lcols = el('div', 'm11-lcols');
    const lgrid = el('div', 'm11-lgrid');
    lplot.append(lgrid, lcols);
    const lcolEls = Array.from({ length: NL }, () => { const c = el('div', 'm11-lcol'); c.appendChild(el('i', 'm11-lbar')); lcols.appendChild(c); return c; });
    const lx = el('div', 'm11-lxaxis');
    for (let l = 0; l < NL; l++) lx.appendChild(el('span', 'm11-lxtick' + (l % 4 ? ' minor' : ''), String(l)));
    lay.appendChild(lx);
    lay.appendChild(el('p', 'm11-xname', 'layer'));
    const stripHead = el('p', 'm11-striphead');
    lay.appendChild(stripHead);
    const strip = el('div', 'm11-strip');
    const ssvg = svgEl(strip, 'svg', { viewBox: '0 0 ' + W + ' 10', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    const spath = svgEl(ssvg, 'path', { class: 'm11-spath', d: '' });
    lay.appendChild(strip);
    const sx = el('div', 'm11-sxaxis');
    [0, 2500, 5000, 7500, 10000].forEach((u) => { const t = el('span', 'm11-xtick', F.int(u)); t.style.left = (u / (W - 1)) * 100 + '%'; sx.appendChild(t); });
    lay.appendChild(sx);
    lay.appendChild(el('p', 'm11-xname', 'neuron index in layer ' + ONE.layer + '. The strip shows layer ' + ONE.layer + ' for every condition and size, because it is ' + ONE.rule + '.'));

    // ---------- the change by tensor family ----------
    const famSec = el('section', 'm11-sec');
    wrap.appendChild(famSec);
    famSec.appendChild(el('h3', 'm11-h', 'The weight delta Δ by tensor family, and which model the chosen condition takes each family from'));
    const fLegend = el('p', 'm11-legend');
    fLegend.append(key('f0', 'entries unchanged'), key('f1', 'moved by one bfloat16 value'), key('f2', 'moved by two or more'));
    famSec.appendChild(fLegend);
    const famList = el('div', 'm11-fams');
    famSec.appendChild(famList);
    const famFrom = FAM.map((f) => {
      const r = fam[f.key];
      const row = el('div', 'm11-fam');
      const head = el('p', 'm11-famhead');
      head.append(el('b', '', f.name), ' ' + F.int(r.entries) + ' entries; squared delta norm ' + fmtE(r.energy) + ' (' + F.percent(r.share_of_energy, r.share_of_energy < 0.001 ? 4 : 1) + ' of the whole delta)');
      row.appendChild(head);
      const bar = el('div', 'm11-fbar');
      [['f0', r.unchanged], ['f1', r.one_16bit_step], ['f2', r.two_or_more_steps]].forEach(([c, n]) => {
        const seg = el('i', 'm11-fseg ' + c);
        seg.style.flexGrow = String(n);
        bar.appendChild(seg);
      });
      if (r.unchanged + r.one_16bit_step + r.two_or_more_steps !== r.entries) throw new Error('m11: the ' + f.key + ' entry counts do not add up');
      WALK.hover(bar, () => ({ title: f.name, rows: [['entries', F.int(r.entries)], ['unchanged', F.int(r.unchanged) + ' (' + F.percent(r.unchanged / r.entries, 1) + ')'],
        ['moved by one bfloat16 value', F.int(r.one_16bit_step) + ' (' + F.percent(r.one_16bit_step / r.entries, 1) + ')'],
        ['moved by two or more', F.int(r.two_or_more_steps) + ' (' + F.percent(r.two_or_more_steps / r.entries, 1) + ')'], ['squared delta norm', fmtE(r.energy)]] }));
      bar.setAttribute('aria-label', f.name + ': entries unchanged, moved by one bfloat16 value, moved by two or more');
      row.appendChild(bar);
      const from = el('p', 'm11-famfrom');
      row.appendChild(from);
      famList.appendChild(row);
      return from;
    });

    // ---------- the score and its noise ----------
    const sc = el('section', 'm11-sec');
    wrap.appendChild(sc);
    sc.appendChild(el('h3', 'm11-h', 'How we score a built model, and the noise of one score'));
    const P = D.score_protocol;
    sc.appendChild(el('p', 'm11-text', 'The score is the share of the ' + P.problems + ' MATH-500 problems answered right: one sampled answer each, temperature ' + P.temperature +
      ', top-p ' + P.top_p + ', up to ' + F.int(P.max_new_tokens) + ' new tokens, sampling seed ' + F.int(P.first_problem_seed) + ' + the problem’s index.'));
    const MARKS = [[N.sigma_points, 2, 'lo', 'points: the noise of one score (its standard deviation), from two sets of sampling seeds on the same weights'],
      [N.difference_must_exceed_points, 2, 'hi', 'points (the shaded stretch ends here): the study counted a difference between two conditions only above this'],
      [N.gain_required_above, 2, 'lo', 'points: the gain from Olmo-3-1025-7B to Olmo-3.1-7B-RL-Zero-Math had to exceed this'],
      [N.gain_points, 1, 'hi', 'points: that gain']];
    const nl = el('div', 'm11-nl');
    sc.appendChild(nl);
    const NMAX = Math.ceil((N.gain_points + 1) / 5) * 5;
    const nx = (v) => (v / NMAX) * 100 + '%';
    const nlTrack = el('div', 'm11-nltrack');
    nl.appendChild(nlTrack);
    const noiseBand = el('i', 'm11-nlband');
    noiseBand.style.width = nx(N.difference_must_exceed_points);
    nlTrack.appendChild(noiseBand);
    MARKS.forEach(([v, d, lane, name]) => {
      const m = el('span', 'm11-nlmark ' + lane);
      m.style.left = nx(v);
      m.appendChild(el('span', 'm11-nllab ' + lane, F.fixed(v, d)));
      nlTrack.appendChild(m);
      WALK.hover(m, { title: F.fixed(v, d) + ' ' + name, rows: [['points', F.fixed(v, 3)]] });
    });
    const nlAxis = el('div', 'm11-nlaxis');
    for (let v = 0; v <= NMAX; v += 5) { const t = el('span', 'm11-xtick', String(v)); t.style.left = nx(v); nlAxis.appendChild(t); }
    nl.appendChild(nlAxis);
    sc.appendChild(el('p', 'm11-xname', 'MATH-500 score, points'));
    const nlist = el('ul', 'm11-nlist');
    MARKS.forEach(([v, d, , name]) => { const li = el('li'); li.append(el('b', '', F.fixed(v, d)), ' ' + name); nlist.appendChild(li); });
    sc.appendChild(nlist);
    sc.appendChild(el('p', 'm11-note', 'This picture’s data holds how we score and the noise of one score, not the scores themselves; the results picture draws every condition’s score.'));

    // ---------- interaction on the ranking ----------
    const banked = new Map(D.cumulative_energy_at_banked_ranks.map((b) => [b.rank, b.energy]));
    let pi = -1;
    const ptBox = (i) => {
      const [r, v] = pts[i];
      const rows = [['sₙ', F.fixed(v, 5), 'ink'], ['share of neurons at or above this rank', F.percent(r / U, r / U < 0.01 ? 3 : 1)]];
      if (banked.has(r)) rows.push(['squared delta norm of ranks 1 to ' + F.int(r), fmtE(banked.get(r)) + ' (' + F.percent(banked.get(r) / E0, 2) + ')']);
      return { title: 'Rank ' + F.int(r) + ' of ' + F.int(U), rows };
    };
    const showPt = (i, e) => {
      pi = i;
      cursor.hidden = false;
      cursor.style.left = xr(pts[i][0]) + '%';
      cursor.style.top = ys(pts[i][1]) + '%';
      if (e) WALK.valueBox.show(ptBox(i), e.clientX, e.clientY);
      else WALK.valueBox.showFor(cursor, ptBox(i));
    };
    const nearest = (e) => {
      const r = plot.getBoundingClientRect();
      const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * 100;
      let best = 0;
      pts.forEach((p, i) => { if (Math.abs(xr(p[0]) - f) < Math.abs(xr(pts[best][0]) - f)) best = i; });
      return best;
    };
    const hidePt = () => { pi = -1; cursor.hidden = true; WALK.valueBox.hide(); };
    plot.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') showPt(nearest(e), e); });
    plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hidePt(); });
    plot.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') showPt(nearest(e), null); });
    plot.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowLeft: -1, PageUp: -20, PageDown: 20, Home: -pts.length, End: pts.length }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      showPt(Math.max(0, Math.min(pts.length - 1, pi < 0 ? 0 : pi + move)), null);
    });
    plot.addEventListener('blur', hidePt);

    // ---------- draw one condition ----------
    let cid = 'most';
    let si = 0;
    function draw() {
      const meta = CONS.find((c) => c.id === cid);
      const cond = condition(cid, si);
      const s = cond.size;
      sel.value = cid;
      sizeBtns.forEach((b, i) => { b.setAttribute('aria-pressed', String(i === si)); b.disabled = !meta.sized; });
      sizeBox.classList.toggle('off', !meta.sized);
      const label = meta.name + (meta.sized ? ', k = ' + pctText(s.percent_of_units) : '');

      // the numbers
      sumBox.replaceChildren();
      sumBox.appendChild(el('p', 'm11-sumhead', label));
      const dl = el('dl', 'm11-dl');
      const item = (k, v) => dl.append(el('dt', '', k), el('dd', '', v));
      item('M, the mask', cond.mask);
      item('MLP entries in M', F.int(cond.entries) + (cond.units ? ' (' + F.int(cond.units) + ' neurons × ' + F.int(EPU) + ')' : ''));
      item('c, the scale', cond.c === null ? 'none: the MLP stays at the base model' : cond.id === 'scalar' ? F.fixed(cond.c, 4) + ' as built, after a fix to the stochastic rounding (' + F.fixed(cond.cPlanned, 4) + ' planned)' :
        cond.id === 'sample' ? '1/q = ' + F.fixed(cond.c, 2) : '1');
      const ratio = cond.energy / E0;
      item('squared delta norm added', cond.energy === 0 ? '0' : fmtE(cond.energy) + ' = ' + (ratio === 1 ? 'the whole MLP delta' : ratio < 1 ? F.percent(ratio, ratio < 0.1 ? 2 : 1) + ' of the whole MLP delta' :
        F.fixed(ratio, 1) + ' times the whole MLP delta') + (cond.built ? '' : ' (planned)'));
      item('Other tensors', cond.from === 'math' ? 'all from Olmo-3.1-7B-RL-Zero-Math' : cond.from === 'attn' ? 'attention and every norm weight from Olmo-3.1-7B-RL-Zero-Math; embeddings and output matrix from the base model' : 'all from the base model');
      sumBox.appendChild(dl);

      // the ranking
      dotEls.forEach((d, i) => {
        const r = pts[i][0];
        d.setAttribute('class', 'm11-pt' + (cond.kind === 'exact' && cond.ranks && r >= cond.ranks[0] && r <= cond.ranks[1] ? ' in' : ''));
      });
      const showCut = meta.sized;
      cutLine.hidden = !showCut;
      cutLab.hidden = !showCut;
      if (showCut) {
        const top = D.top_sets_from_curve[String(s.percent_of_units) + ' %'];
        cutLine.style.left = xr(s.units) + '%';
        cutLab.style.left = xr(s.units) + '%';
        cutLab.classList.toggle('left', xr(s.units) > 45);
        cutLab.textContent = 'rank ' + F.int(s.units) + ': sₙ = ' + F.fixed(top.smallest_s, 5);
      }
      maskBand.className = 'm11-maskband ' + cond.kind;
      if (cond.ranks) {
        maskBand.hidden = false;
        maskBand.style.left = xr(cond.ranks[0]) + '%';
        maskBand.style.width = (xr(cond.ranks[1]) - xr(cond.ranks[0])) + '%';
      } else maskBand.hidden = true;
      maskText.textContent = cond.kind === 'exact' ? 'M takes every neuron in the band: ranks ' + F.int(cond.ranks[0]) + ' to ' + F.int(cond.ranks[1]) + '.'
        : cond.kind === 'random' ? 'M takes ' + F.int(cond.units) + ' neurons at random from the hatched band, ranks ' + F.int(cond.ranks[0]) + ' to ' + F.int(cond.ranks[1]) + '; this picture’s data does not hold which ranks they are.'
          : cond.kind === 'entries' ? 'M takes entries, not neurons: DARE keeps each of the ' + F.int(fam.mlp.entries) + ' MLP entries, from every rank, with probability q = ' + sig3(cond.q) + '.'
            : 'M is empty: no MLP entry is copied.';

      // the energies at this size
      enHead.textContent = 'The squared delta norm each condition adds' + (meta.sized ? ', at k = ' + pctText(s.percent_of_units) : '');
      eplot.replaceChildren();
      const rowsE = CONS.filter((c) => c.id !== 'attn').map((c) => condition(c.id, si));
      rowsE.forEach((r) => {
        const m = CONS.find((c) => c.id === r.id);
        const row = el('div', 'm11-erow' + (r.id === cid ? ' cur' : ''));
        row.appendChild(el('span', 'm11-ename', m.name + (m.sized ? ', k = ' + pctText(s.percent_of_units) : '')));
        row.setAttribute('role', 'button');
        const tr = el('div', 'm11-etrack');
        const ref = el('i', 'm11-ewhole');
        ref.style.left = ex(E0) + '%';
        tr.appendChild(ref);
        const dot = el('i', 'm11-edot' + (r.built ? '' : ' ring'));
        dot.style.left = ex(r.energy) + '%';
        tr.appendChild(dot);
        if (r.id === cid) {
          const v = el('span', 'm11-eval' + (ex(r.energy) > 70 ? ' left' : ''), fmtE(r.energy));
          v.style.left = ex(r.energy) + '%';
          tr.appendChild(v);
        }
        row.appendChild(tr);
        eplot.appendChild(row);
        WALK.hover(row, () => ({
          title: m.name + (m.sized ? ', k = ' + pctText(s.percent_of_units) : ''),
          rows: [['squared delta norm added', fmtE(r.energy) + (r.built ? '' : ' (planned)'), 'ink'], ['over the whole MLP delta', F.fixed(r.energy / E0, r.energy / E0 < 0.1 ? 5 : 3)],
            ['over the top-k set', F.fixed(r.energy / s.most_changed.energy, 3)]],
        }));
        row.dataset.id = r.id;
        row.addEventListener('click', () => { cid = r.id; draw(); });
        row.addEventListener('keydown', (e) => {
          if (e.key !== 'Enter' && e.key !== ' ') return;
          e.preventDefault();
          cid = r.id;
          draw();
          eplot.querySelector('.m11-erow[data-id="' + r.id + '"]').focus();
        });
      });
      if (cid === 'attn') {
        eplot.appendChild(el('p', 'm11-note', 'This condition adds no MLP delta; it takes attention (squared delta norm ' + fmtE(fam.attn.energy) + ') and the norm weights (' + fmtE(fam.norm.energy) + ') from Olmo-3.1-7B-RL-Zero-Math.'));
      }

      // the layers
      curCond = cond;
      curLabel = label;
      layHead.textContent = cond.perLayer ? 'Where its neurons sit: neurons per layer, of ' + F.int(W) : 'Where its entries sit';
      lgrid.replaceChildren();
      if (cond.perLayer) {
        lplot.hidden = false;
        lx.hidden = false;
        const mx = Math.max(1, ...cond.perLayer);
        const step = niceStep(mx / 3);
        const top = Math.ceil(mx / step) * step;
        for (let v = 0; v <= top + 1e-9; v += step) {
          const g = el('div', 'm11-grid');
          g.style.top = (1 - v / top) * 100 + '%';
          g.appendChild(el('span', 'm11-ytick', F.int(v)));
          lgrid.appendChild(g);
        }
        lcolEls.forEach((c, l) => {
          c.firstChild.style.height = (cond.perLayer[l] / top) * 100 + '%';
        });
      } else {
        lplot.hidden = true;
        lx.hidden = true;
      }
      stripHead.textContent = cond.idx ? 'Layer ' + ONE.layer + ': the mask M along its ' + F.int(W) + ' neurons in index order, ' + F.int(cond.idx.length) + ' marked'
        : 'Counts per neuron do not apply: M keeps entries at random in every neuron of every layer, about ' + sig3(cond.q * EPU) + ' of each neuron’s ' + F.int(EPU) + ' entries.';
      strip.hidden = !cond.idx;
      sx.hidden = !cond.idx;
      spath.setAttribute('d', cond.idx ? cond.idx.map((i) => 'M' + (i + 0.5) + ' 0v10').join('') : '');

      // the families
      FAM.forEach((f, k) => {
        const where = fromFor(cond, f.key);
        famFrom[k].textContent = 'In this condition: ' + (where === 'built' ? 'base + c ⊙ M ⊙ Δ, as above' : where === 'math' ? 'from Olmo-3.1-7B-RL-Zero-Math' : 'from the base model');
      });
    }
    // one listener set for the columns; the arrow keys move between layers
    let curCond = null;
    let curLabel = '';
    let li = -1;
    const colAt = (e) => { const r = lcols.getBoundingClientRect(); return Math.max(0, Math.min(NL - 1, Math.floor(((e.clientX - r.left) / r.width) * NL))); };
    const markCol = (l) => lcolEls.forEach((c, k) => c.classList.toggle('hov', k === l));
    lplot.tabIndex = 0;
    lplot.setAttribute('role', 'group');
    lplot.setAttribute('aria-label', 'Neurons of the chosen condition in each layer; arrow keys move between layers');
    lplot.addEventListener('pointermove', (e) => { if (e.pointerType !== 'mouse') return; const l = colAt(e); markCol(l); WALK.valueBox.show(layerBox(curCond, l, curLabel), e.clientX, e.clientY); });
    lplot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { markCol(-1); WALK.valueBox.hide(); } });
    lplot.addEventListener('pointerdown', (e) => { if (e.pointerType === 'mouse') return; const l = colAt(e); li = l; markCol(l); WALK.valueBox.showFor(lcolEls[l], layerBox(curCond, l, curLabel)); });
    lplot.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowLeft: -1, Home: -NL, End: NL }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      li = Math.max(0, Math.min(NL - 1, li < 0 ? NL - 1 : li + move));
      markCol(li);
      WALK.valueBox.showFor(lcolEls[li], layerBox(curCond, li, curLabel));
    });
    lplot.addEventListener('blur', () => { markCol(-1); WALK.valueBox.hide(); });
    const layerBox = (cond, l, label) => ({
      title: 'Layer ' + l + ', ' + label,
      rows: [['neurons in M', F.int(cond.perLayer[l]) + ' of ' + F.int(W), 'claim'], ['the layer’s squared MLP delta norm', fmtE(D.per_layer.mlp_change_energy[l])],
        ['largest sₙ in the layer', F.fixed(D.per_layer_from_curve[l].max_s, 5)], ['median sₙ in the layer', F.fixed(D.per_layer_from_curve[l].median_s, 5)]],
    });

    sel.addEventListener('change', () => { cid = sel.value; draw(); });
    draw();
  }

  function niceStep(x) {
    const p = 10 ** Math.floor(Math.log10(x));
    return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= x);
  }

  WALK.register('m11', { data: 'M11_DATA', mount });
})();
