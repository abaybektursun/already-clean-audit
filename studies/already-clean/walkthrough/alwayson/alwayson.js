// alwayson: the 3,412 always-on neurons against the rest. Registers the mount "alwayson" with site.js; reads
// window.ALWAYSON_DATA (alwayson/data.js). A histogram of |mean activation| over 200 tokens of corpus text for the
// 341,248 neurons of layers 0 to 30, the always-on neurons and the rest drawn apart, with the cut; a menu picks one
// layer or all. Below, the always-on neurons per layer; a tap on a layer's column picks that layer for the histogram.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const pow10 = (k) => (k === 0 ? '1' : k === 1 ? '10' : '10' + String(k).replace(/./g, (c) => SUP[c]));
  const sum = (a) => a.reduce((s, v) => s + v, 0);

  function mount(root, D) {
    const F = WALK.fmt;
    const P = D.per_layer;
    const E = P.histogram_log10_edges;
    const NB = E.length - 1;
    const NL = D.layers_considered[1] - D.layers_considered[0] + 1;
    const W = D.units_per_layer;
    const N = D.always_on_count;
    const CUT = D.cut;
    const REST = D.units_considered - N;
    const BW = E[1] - E[0];

    // ---------- the data must hold together before anything is drawn ----------
    if (NL * W !== D.units_considered) throw new Error('alwayson: ' + NL + ' layers x ' + W + ' is not ' + D.units_considered);
    if (P.histogram_always_on.length !== NL || P.histogram_rest.length !== NL || P.always_on.length !== NL) throw new Error('alwayson: expected ' + NL + ' layers');
    P.histogram_always_on.forEach((h, l) => {
      if (h.length !== NB || P.histogram_rest[l].length !== NB) throw new Error('alwayson: layer ' + l + ' histogram has the wrong number of bins');
      if (sum(h) !== P.always_on[l]) throw new Error('alwayson: layer ' + l + ' histogram holds ' + sum(h) + ' always-on neurons, the data says ' + P.always_on[l]);
      if (sum(h) + sum(P.histogram_rest[l]) !== W) throw new Error('alwayson: layer ' + l + ' histograms do not hold ' + W + ' neurons');
    });
    if (sum(P.always_on) !== N || D.members.length !== N) throw new Error('alwayson: the members do not number ' + N);
    const smallest = Math.min(...D.members.map((m) => Math.abs(m[2])));
    if (smallest !== CUT) throw new Error('alwayson: the smallest member |mean| ' + smallest + ' is not the cut ' + CUT);
    const negative = D.members.filter((m) => m[2] < 0).length;
    if (negative !== D.members_with_negative_mean) throw new Error('alwayson: ' + negative + ' members have a negative mean, the data says ' + D.members_with_negative_mean);
    const LOGCUT = Math.log10(CUT);
    if (LOGCUT < E[0] || LOGCUT > E[NB]) throw new Error('alwayson: the cut is outside the histogram');

    const lateFrom = 27;
    const late = sum(P.always_on.slice(lateFrom));
    const R = D.first_run_with_the_cut;

    root.replaceChildren();
    const wrap = el('div', 'ao-wrap');
    root.appendChild(wrap);

    // ---------- facts ----------
    const facts = el('div', 'ao-facts');
    const fact = (big, text) => { const f = el('p', 'ao-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(F.int(N), 'always-on neurons: the 1 % of the ' + F.int(D.units_considered) + ' neurons in layers ' + D.layers_considered[0] + ' to ' + D.layers_considered[1] + ' with the largest |mean activation|');
    fact(F.fixed(CUT, 3), 'the cut: the smallest |mean activation| among them');
    fact(F.int(late) + ' of ' + F.int(N), 'are in layers ' + lateFrom + ' to ' + D.layers_considered[1]);
    wrap.appendChild(facts);

    // ---------- controls ----------
    const controls = el('div', 'control-row ao-controls');
    const selWrap = el('label', 'ao-pick');
    selWrap.appendChild(el('span', 'control-label', 'Layer'));
    const sel = el('select', 'ao-select');
    const all = el('option', '', 'All ' + NL + ' layers (' + D.layers_considered[0] + ' to ' + D.layers_considered[1] + ')');
    all.value = 'all';
    sel.appendChild(all);
    P.always_on.forEach((c, l) => {
      const o = el('option', '', 'Layer ' + l + ': ' + F.int(c) + ' always-on of ' + F.int(W));
      o.value = String(l);
      sel.appendChild(o);
    });
    selWrap.appendChild(sel);
    const toggle = el('div', 'toggle');
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Count scale');
    const bLin = el('button', '', 'Linear count');
    const bLog = el('button', '', 'Log count');
    [bLin, bLog].forEach((b) => { b.type = 'button'; toggle.appendChild(b); });
    controls.append(selWrap, toggle);
    wrap.appendChild(controls);

    // ---------- the histogram ----------
    const hist = el('section', 'ao-hist');
    wrap.appendChild(hist);
    const hHead = el('h3', 'ao-h');
    hist.appendChild(hHead);
    const legend = el('p', 'ao-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'ao-key ' + cls), text); return s; };
    legend.append(key('on', 'always-on neurons'), key('rest', 'the rest'), key('cut', 'the cut, ' + F.fixed(CUT, 3)));
    hist.appendChild(legend);
    const plot = el('div', 'ao-plot');
    plot.tabIndex = 0;
    plot.setAttribute('role', 'group');
    hist.appendChild(plot);
    const grid = el('div', 'ao-gridlayer');
    const bars = el('div', 'ao-bars');
    const cursor = el('div', 'ao-cursor');
    cursor.hidden = true;
    plot.append(grid, cursor, bars);
    const xpct = (v) => ((v - E[0]) / (E[NB] - E[0])) * 100;
    const cutLine = el('div', 'ao-cutline');
    cutLine.style.left = xpct(LOGCUT) + '%';
    cutLine.appendChild(el('span', 'ao-cutlab', 'cut ' + F.fixed(CUT, 3)));
    plot.appendChild(cutLine);
    const xaxis = el('div', 'ao-xaxis');
    for (let k = Math.ceil(E[0]); k <= Math.floor(E[NB]); k++) {
      const t = el('span', 'ao-xtick' + ((k - Math.ceil(E[0])) % 2 ? ' odd' : ''), pow10(k));
      t.style.left = xpct(k) + '%';
      xaxis.appendChild(t);
    }
    hist.appendChild(xaxis);
    hist.appendChild(el('p', 'ao-xname', '|mean activation| over ' + D.positions_averaged + ' tokens of corpus text (log scale; bins of ' + F.fixed(BW, 1) + ' in log₁₀)'));

    // ---------- the per-layer columns ----------
    const per = el('section', 'ao-per');
    wrap.appendChild(per);
    per.appendChild(el('h3', 'ao-h', 'Always-on neurons per layer'));
    per.appendChild(el('p', 'ao-note', 'Each column is one layer’s count out of its ' + F.int(W) + ' neurons. Tap or click a column to show that layer above; arrow keys move between layers.'));
    const cplot = el('div', 'ao-cplot');
    cplot.tabIndex = 0;
    cplot.setAttribute('role', 'group');
    cplot.setAttribute('aria-label', 'Always-on neurons in each of layers ' + D.layers_considered[0] + ' to ' + D.layers_considered[1] + '; arrow keys pick a layer, Escape returns to all layers');
    per.appendChild(cplot);
    const cgrid = el('div', 'ao-gridlayer');
    const cols = el('div', 'ao-cols');
    cplot.append(cgrid, cols);
    const cmax = Math.max(...P.always_on);
    const cstep = niceStep(cmax / 4);
    const ctop = Math.ceil(cmax / cstep) * cstep;
    for (let v = 0; v <= ctop + 1e-9; v += cstep) {
      const g = el('div', 'ao-grid');
      g.style.bottom = (v / ctop) * 100 + '%';
      g.appendChild(el('span', 'ao-ytick', F.int(v)));
      cgrid.appendChild(g);
    }
    const colEls = P.always_on.map((c, l) => {
      const slot = el('div', 'ao-col');
      const b = el('i', 'ao-colbar');
      b.style.height = (c / ctop) * 100 + '%';
      slot.appendChild(b);
      if (l >= lateFrom) {
        const v = el('span', 'ao-colval', F.int(c));
        v.style.bottom = (c / ctop) * 100 + '%';
        slot.appendChild(v);
      }
      cols.appendChild(slot);
      return slot;
    });
    const cx = el('div', 'ao-cxaxis');
    P.always_on.forEach((c, l) => {
      const t = el('span', 'ao-cxtick' + (l % 5 ? ' minor' : ''), String(l));
      cx.appendChild(t);
    });
    per.appendChild(cx);
    per.appendChild(el('p', 'ao-xname', 'layer'));

    per.appendChild(el('p', 'ao-note', 'The activation is a neuron’s input to down_proj, SiLU(gate_proj · x) × (up_proj · x); we averaged it over ' + D.positions_averaged +
      ' tokens of corpus text, read at ' + D.model + '. ' + F.int(D.members_with_negative_mean) + ' of the ' + F.int(N) +
      ' always-on neurons have a negative mean; the ranking uses the absolute value. Layer 31 is outside the ranking: it has no later layer to feed. ' +
      'The run of ' + R.date.slice(8, 10).replace(/^0/, '') + ' November computed its own cut on another machine, ' + F.fixed(R.cut, 4) +
      ', which also takes ' + F.int(R.count) + ' neurons; ' + F.int(R.members_shared_with_this_bank) + ' of them are the same neurons.'));

    // ---------- state ----------
    let layer = 'all';
    let logScale = true;
    let bin = -1;
    let counts = null; // {on: [], rest: []}

    const countsFor = (l) => (l === 'all'
      ? { on: E.slice(0, NB).map((_, i) => sum(P.histogram_always_on.map((h) => h[i]))), rest: E.slice(0, NB).map((_, i) => sum(P.histogram_rest.map((h) => h[i]))) }
      : { on: P.histogram_always_on[l], rest: P.histogram_rest[l] });

    const lo = Math.log10(0.5); // the log scale's baseline sits half a unit below 1, so a bin holding one unit shows

    function draw() {
      counts = countsFor(layer);
      const top = Math.max(...counts.on, ...counts.rest);
      let y;
      grid.replaceChildren();
      if (logScale) {
        const kmax = Math.ceil(Math.log10(top));
        y = (c) => (c > 0 ? ((Math.log10(c) - lo) / (kmax - lo)) * 100 : 0);
        for (let k = 0; k <= kmax; k++) {
          const g = el('div', 'ao-grid');
          g.style.bottom = y(10 ** k) + '%';
          g.appendChild(el('span', 'ao-ytick', F.int(10 ** k)));
          grid.appendChild(g);
        }
      } else {
        const step = niceStep(top / 4);
        const ymax = Math.ceil(top / step) * step;
        y = (c) => (c / ymax) * 100;
        for (let v = 0; v <= ymax + 1e-9; v += step) {
          const g = el('div', 'ao-grid');
          g.style.bottom = y(v) + '%';
          g.appendChild(el('span', 'ao-ytick', F.int(v)));
          grid.appendChild(g);
        }
      }
      bars.replaceChildren();
      const w = 100 / NB;
      for (let i = 0; i < NB; i++) {
        const on = counts.on[i];
        const rest = counts.rest[i];
        const both = on > 0 && rest > 0;
        const put = (c, cls, half) => {
          if (!(c > 0)) return;
          const b = el('i', 'ao-bar ' + cls);
          b.style.left = (i + (half === 2 ? 0.5 : 0)) * w + '%';
          b.style.width = (half ? w / 2 : w) + '%';
          b.style.height = y(c) + '%';
          bars.appendChild(b);
        };
        put(rest, 'rest', both ? 1 : 0);
        put(on, 'on', both ? 2 : 0);
      }
      const nOn = sum(counts.on);
      const nRest = sum(counts.rest);
      hHead.textContent = layer === 'all'
        ? 'All ' + F.int(D.units_considered) + ' neurons of layers ' + D.layers_considered[0] + ' to ' + D.layers_considered[1] + ': ' + F.int(nOn) + ' always-on, ' + F.int(nRest) + ' the rest'
        : 'Layer ' + layer + ': ' + F.int(nOn) + ' always-on, ' + F.int(nRest) + ' the rest, of ' + F.int(W);
      plot.setAttribute('aria-label', hHead.textContent + '. Histogram of |mean activation|; arrow keys move between bins.');
      bLin.setAttribute('aria-pressed', String(!logScale));
      bLog.setAttribute('aria-pressed', String(logScale));
      sel.value = String(layer);
      colEls.forEach((c, l) => c.classList.toggle('cur', l === layer));
      if (bin >= 0) showBin(bin, null);
    }

    const fmtA = (x) => {
      if (x >= 0.001) return x.toPrecision(3);
      const e = Math.floor(Math.log10(x));
      return (x / 10 ** e).toFixed(2) + ' × ' + pow10(e);
    };
    function binBox(i) {
      const a = 10 ** E[i];
      const b = 10 ** E[i + 1];
      return {
        title: '|mean activation| ' + fmtA(a) + ' to ' + fmtA(b),
        rows: [['always-on neurons', F.int(counts.on[i]), 'claim'], ['the rest', F.int(counts.rest[i]), 'muted'],
          ['layer', layer === 'all' ? 'all ' + NL : String(layer)]],
      };
    }
    function showBin(i, ev) {
      bin = i;
      cursor.hidden = false;
      cursor.style.left = (i / NB) * 100 + '%';
      cursor.style.width = 100 / NB + '%';
      if (ev) WALK.valueBox.show(binBox(i), ev.clientX, ev.clientY);
      else WALK.valueBox.showFor(cursor, binBox(i));
    }
    const binAt = (e) => {
      const r = plot.getBoundingClientRect();
      return Math.max(0, Math.min(NB - 1, Math.floor(((e.clientX - r.left) / r.width) * NB)));
    };
    const clearBin = () => { bin = -1; cursor.hidden = true; WALK.valueBox.hide(); };
    plot.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') showBin(binAt(e), e); });
    plot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') clearBin(); });
    plot.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') showBin(binAt(e), null); });
    plot.addEventListener('keydown', (e) => {
      const move = { ArrowRight: 1, ArrowLeft: -1, Home: -NB, End: NB }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      const start = bin < 0 ? Math.floor((LOGCUT - E[0]) / BW) : bin + move;
      showBin(Math.max(0, Math.min(NB - 1, start)), null);
    });
    plot.addEventListener('blur', clearBin);

    // the per-layer columns
    const colBox = (l) => ({
      title: 'Layer ' + l,
      rows: [['always-on neurons', F.int(P.always_on[l]) + ' of ' + F.int(W), 'claim'],
        ['median |mean activation|', fmtA(P.median_abs_mean_activation[l])],
        ['largest |mean activation|', fmtA(P.max_abs_mean_activation[l])],
        ['always-on in the run of ' + R.date.slice(8, 10).replace(/^0/, '') + ' November', F.int(R.per_layer[l])]],
    });
    const colAt = (e) => {
      const r = cols.getBoundingClientRect();
      return Math.max(0, Math.min(NL - 1, Math.floor(((e.clientX - r.left) / r.width) * NL)));
    };
    const markHover = (l) => colEls.forEach((c, k) => c.classList.toggle('hov', k === l));
    cplot.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const l = colAt(e);
      markHover(l);
      WALK.valueBox.show(colBox(l), e.clientX, e.clientY);
    });
    cplot.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { markHover(-1); WALK.valueBox.hide(); } });
    cplot.addEventListener('click', (e) => {
      const l = colAt(e);
      layer = layer === l ? 'all' : l;
      draw();
      if (e.pointerType && e.pointerType !== 'mouse') WALK.valueBox.showFor(colEls[l], colBox(l));
    });
    cplot.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { layer = 'all'; draw(); return; }
      const move = { ArrowRight: 1, ArrowLeft: -1, Home: -NL, End: NL }[e.key];
      if (move === undefined) return;
      e.preventDefault();
      const l = Math.max(0, Math.min(NL - 1, layer === 'all' ? NL - 1 : layer + move));
      layer = l;
      draw();
      WALK.valueBox.showFor(colEls[l], colBox(l));
    });
    cplot.addEventListener('blur', () => { WALK.valueBox.hide(); markHover(-1); });

    sel.addEventListener('change', () => { layer = sel.value === 'all' ? 'all' : Number(sel.value); bin = -1; cursor.hidden = true; draw(); });
    bLin.addEventListener('click', () => { logScale = false; draw(); });
    bLog.addEventListener('click', () => { logScale = true; draw(); });
    draw();
  }

  // 1, 2 or 5 times a power of ten, at least x
  function niceStep(x) {
    const p = 10 ** Math.floor(Math.log10(x));
    return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= x);
  }

  WALK.register('alwayson', { data: 'ALWAYSON_DATA', mount });
})();
