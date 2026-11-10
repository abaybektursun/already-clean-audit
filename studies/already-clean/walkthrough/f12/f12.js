// f12: per band of eight layers, the share of a pole's top-50 activating tokens that come from code and mathematics,
// at the start (stage1-step476838) and after each of the three data mixes, against the drift band. Registers the mount
// "f12" with site.js; reads window.F12_DATA (f12/data.js). Four small panels, one per band, on one scale in tokens of
// the 50 (share x 50), each centred on its own start; in each panel one lane for the start and one for each mix. The
// drift band is the start plus or minus the largest band-mean change between neighbouring stage-1 checkpoints. A toggle
// picks the mix whose move is drawn and written.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  const MIXES = [
    { key: 'no_code_or_math', short: '0\u00a0%', shape: 'circle', branch: 'stage2-step47684-mix-gen-mc-only-from-2T-ckpt' },
    { key: 'code_and_math_40', short: '40\u00a0%', shape: 'square', branch: 'stage2-step47684-mix-round5-from-2T-ckpt' },
    { key: 'code_and_math_65', short: '65\u00a0%', shape: 'diamond', branch: 'stage2-step47684-mix-math-code-reasoning-web-from-2T-ckpt' },
  ];

  function mount(root, D) {
    const F = WALK.fmt;
    const K = D.slots_per_side;
    const NB = D.bands.length;
    const H = D.harvests;
    const S = D.shift_from_start;
    const DR = D.drift_bar;
    const snaps = DR.band_mean_share_at_snapshots;

    // ---------- the data must hold together before anything is drawn ----------
    MIXES.forEach((m) => {
      if (!H[m.key] || !S[m.key]) throw new Error('f12: no run or shift for ' + m.key);
      if (H[m.key].band_mean_share.length !== NB || S[m.key].band_mean_shift.length !== NB || S[m.key].range_banked.length !== NB) throw new Error('f12: ' + m.key + ' lacks a band');
    });
    if (H.start.band_mean_share.length !== NB || DR.band_bar.length !== NB) throw new Error('f12: the start or the drift band lacks a band');
    // The drift band per band is the largest |change| over the neighbouring snapshot pairs; recompute it here.
    const pairDiffs = DR.snapshot_pairs.map(([a, b]) => snaps[b].map((v, i) => v - snaps[a][i]));
    for (let b = 0; b < NB; b++) {
      const big = Math.max(...pairDiffs.map((d) => Math.abs(d[b])));
      if (Math.abs(big - DR.band_bar[b]) > 1e-12) throw new Error('f12: band ' + b + ' drift band ' + DR.band_bar[b] + ' is not the largest neighbouring change ' + big);
    }
    if (D.identity_pair.sides_whose_counts_differ !== 0) throw new Error('f12: the start read twice differs');
    H.start.band_mean_share.forEach((v, b) => {
      if (v !== H.start_second_harvest.band_mean_share[b]) throw new Error('f12: band ' + b + ' differs between the two runs of the start');
    });

    const pos = (share) => share * K;                       // share of fifty -> positions of fifty
    const STEPS = D.snapshot_steps;
    if (!STEPS || STEPS.length !== 16) throw new Error('f12: the data does not hold the sixteen checkpoint steps');
    const snapName = (t) => F.int(STEPS[Number(/^t(\d+)/.exec(t)[1])]);
    const bandName = (b) => 'Layers ' + D.bands[b].layers[0] + ' to ' + D.bands[b].layers[1];
    const signed = (x, d) => (x > 0 ? '+' : '') + F.fixed(x, d);

    // one scale for every panel: the widest reach of any move, range or drift band from the start, plus a margin
    let relLo = 0;
    let relHi = 0;
    for (let b = 0; b < NB; b++) {
      relLo = Math.min(relLo, -pos(DR.band_bar[b]));
      relHi = Math.max(relHi, pos(DR.band_bar[b]));
      MIXES.forEach((m) => S[m.key].range_banked[b].forEach((r) => { relLo = Math.min(relLo, pos(r)); relHi = Math.max(relHi, pos(r)); }));
    }
    const PAD = 0.08;
    relLo -= PAD;
    relHi += PAD;

    const movesNone = S.no_code_or_math.band_mean_shift_in_slots.map(Math.abs);
    const moves65 = S.code_and_math_65.band_mean_shift_in_slots.map(Math.abs);
    const drifts = DR.band_bar.map(pos);

    root.replaceChildren();
    const wrap = el('div', 'f12-wrap');
    root.appendChild(wrap);

    // ---------- facts ----------
    const facts = el('div', 'f12-facts');
    const fact = (big, text) => { const f = el('p', 'f12-fact'); f.append(el('b', '', big), el('span', '', text)); facts.appendChild(f); };
    fact(F.fixed(Math.min(...movesNone), 2) + ' to ' + F.fixed(Math.max(...movesNone), 2),
      'tokens of the 50: how far 100 billion training tokens with no code or mathematics moved the band mean, by band');
    fact(F.fixed(Math.max(...moves65), 2), 'tokens of the 50: the largest move under the ' + MIXES[2].short + ' mix');
    fact('±' + F.fixed(Math.min(...drifts), 2) + ' to ±' + F.fixed(Math.max(...drifts), 2),
      'the drift band: the largest change of the band mean between neighbouring stage-1 checkpoints, by band');
    wrap.appendChild(facts);

    // ---------- controls and legend ----------
    const controls = el('div', 'control-row f12-controls');
    controls.appendChild(el('span', 'control-label', 'Mix, by share of code and mathematics'));
    const toggle = el('div', 'toggle');
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Mix whose move is drawn');
    const buttons = MIXES.map((m, i) => {
      const b = el('button', '', m.short);
      b.type = 'button';
      b.addEventListener('click', () => select(i));
      toggle.appendChild(b);
      return b;
    });
    controls.appendChild(toggle);
    wrap.appendChild(controls);

    const legend = el('p', 'f12-legend');
    const key = (cls, text) => { const s = el('span'); s.append(el('i', 'f12-key ' + cls), text); return s; };
    legend.append(key('start', 'the start'), key('drift', 'the drift band'),
      key('mk circle', MIXES[0].short), key('mk square', MIXES[1].short), key('mk diamond', MIXES[2].short),
      key('whisker', 'bootstrap 95 % CI, 2,000 resamples'), key('move', 'the chosen mix’s move'));
    wrap.appendChild(legend);

    // ---------- the four panels ----------
    const panels = el('div', 'f12-panels');
    wrap.appendChild(panels);
    const x = (rel) => ((rel - relLo) / (relHi - relLo)) * 100;
    const parts = [];
    for (let b = 0; b < NB; b++) {
      const start = pos(H.start.band_mean_share[b]);
      const panel = el('section', 'f12-panel');
      panel.appendChild(el('h3', 'f12-h', bandName(b) + ' (' + F.int(D.bands[b].sides) + ' poles)'));
      const body = el('div', 'f12-body');
      panel.appendChild(body);
      const lanes = el('div', 'f12-lanes');
      body.appendChild(lanes);
      const plot = el('div', 'f12-plot');
      body.appendChild(plot);

      // the drift band and the start, across every lane
      const drift = el('div', 'f12-drift');
      drift.style.left = x(-pos(DR.band_bar[b])) + '%';
      drift.style.width = (x(pos(DR.band_bar[b])) - x(-pos(DR.band_bar[b]))) + '%';
      plot.appendChild(drift);
      WALK.hover(drift, () => ({
        title: bandName(b) + ': the drift band',
        rows: DR.snapshot_pairs.map(([a, c], j) => ['checkpoints at steps ' + snapName(a) + ' to ' + snapName(c), signed(pos(pairDiffs[j][b]), 3) + ' tokens of 50'])
          .concat([['the band: the start ±', F.fixed(pos(DR.band_bar[b]), 3), 'control']]),
      }));
      drift.setAttribute('aria-label', bandName(b) + ', the drift band');
      const startLine = el('div', 'f12-startline');
      startLine.style.left = x(0) + '%';
      plot.appendChild(startLine);

      const laneNames = ['the start'].concat(MIXES.map((m) => m.short));
      laneNames.forEach((n, j) => {
        const l = el('span', 'f12-lane', n);
        lanes.appendChild(l);
        const g = el('div', 'f12-laneline');
        g.style.top = ((j + 0.5) / laneNames.length) * 100 + '%';
        plot.insertBefore(g, drift);
      });
      const laneY = (j) => ((j + 0.5) / laneNames.length) * 100 + '%';

      // the start's mark
      const sm = el('i', 'f12-smark');
      sm.style.left = x(0) + '%';
      sm.style.top = laneY(0);
      plot.appendChild(sm);
      WALK.hover(sm, () => ({
        title: bandName(b) + ': ' + H.start.name,
        rows: [['share of the top-50', F.fixed(H.start.band_mean_share[b], 4), 'control'], ['tokens of the 50', F.fixed(start, 3)],
          ['second run of the same weights', F.fixed(H.start_second_harvest.band_mean_share[b], 4)]],
      }));
      sm.setAttribute('aria-label', bandName(b) + ', the start');

      // the three mixes
      const mixParts = MIXES.map((m, i) => {
        const shift = S[m.key].band_mean_shift_in_slots[b];
        const rng = S[m.key].range_banked[b].map(pos);
        const wh = el('div', 'f12-whisker');
        wh.style.left = x(rng[0]) + '%';
        wh.style.width = (x(rng[1]) - x(rng[0])) + '%';
        wh.style.top = laneY(i + 1);
        const arrow = el('div', 'f12-arrow');
        arrow.style.top = laneY(i + 1);
        const lo = Math.min(0, shift);
        const hi = Math.max(0, shift);
        arrow.style.left = x(lo) + '%';
        arrow.style.width = (x(hi) - x(lo)) + '%';
        arrow.classList.add(shift < 0 ? 'left' : 'right');
        const mk = el('i', 'f12-mark ' + m.shape);
        mk.style.left = x(shift) + '%';
        mk.style.top = laneY(i + 1);
        const lab = el('span', 'f12-movelab', signed(shift, 2));
        lab.style.left = x(shift / 2) + '%';
        lab.style.top = laneY(i + 1);
        plot.append(arrow, wh, mk, lab);
        WALK.hover(mk, () => ({
          title: bandName(b) + ': ' + H[m.key].name,
          rows: [['share of the top-50', F.fixed(H[m.key].band_mean_share[b], 4), 'ink'], ['tokens of the 50', F.fixed(pos(H[m.key].band_mean_share[b]), 3)],
            ['move from the start', signed(shift, 3) + ' tokens of 50'], ['bootstrap 95 % CI', signed(rng[0], 3) + ' to ' + signed(rng[1], 3)],
            ['the drift band', '±' + F.fixed(pos(DR.band_bar[b]), 3), 'control']],
        }));
        mk.setAttribute('aria-label', bandName(b) + ', ' + m.short + ' mix');
        return { mk, arrow, lab, wh };
      });
      parts.push(mixParts);

      // the axis: tokens of the 50, at quarter steps
      const axis = el('div', 'f12-axis');
      const a0 = Math.ceil((start + relLo) * 4) / 4;
      for (let v = a0; v <= start + relHi + 1e-9; v += 0.25) {
        const t = el('span', 'f12-tick' + (Math.abs(v * 2 - Math.round(v * 2)) > 1e-9 ? ' quarter' : ''), F.fixed(v, 2));
        t.style.left = x(v - start) + '%';
        axis.appendChild(t);
        const g = el('div', 'f12-vgrid');
        g.style.left = x(v - start) + '%';
        plot.insertBefore(g, plot.firstChild);
      }
      body.appendChild(axis);
      panels.appendChild(panel);
    }
    wrap.appendChild(el('p', 'f12-xname', 'Tokens from code and mathematics among a pole’s top-50 activating tokens (the share × ' + K + '), mean over the band’s poles. ' +
      'Every panel has the same scale, centred on its own start.'));

    const out = el('p', 'f12-out');
    out.setAttribute('aria-live', 'polite');
    wrap.appendChild(out);

    wrap.appendChild(el('p', 'f12-note', 'Each mix is 100 billion tokens of training from the start, the checkpoint stage1-step476838; the three results are the Hugging Face branches ' +
      MIXES.map((m) => m.branch + ' (' + m.short + ')').join(', ') + '. ' +
      'A pole is one sign of a neuron’s activations. Code and mathematics means the corpus directories ' + D.code_and_math_directories.map((d) => d.replace(/^data\//, '')).join(', ') + '. ' +
      'We read the start twice on the same weights: ' + F.int(D.identity_pair.sides_whose_counts_differ) + ' of ' + F.int(D.identity_pair.sides) + ' poles differ. ' +
      'The bootstrap 95 % confidence interval is over 2,000 resamples of neurons, both poles of a neuron together. ' +
      'The drift band is the start plus or minus the largest change of the band mean between neighbouring stage-1 checkpoints (' +
      DR.snapshot_pairs.map(([a, c]) => 'steps ' + snapName(a) + ' to ' + snapName(c)).join(', ') + '). Hover, tap or tab to a mark for its numbers.'));

    function select(i) {
      buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
      parts.forEach((mixParts) => mixParts.forEach((p, j) => {
        p.mk.classList.toggle('cur', j === i);
        p.wh.classList.toggle('cur', j === i);
        p.arrow.hidden = j !== i;
        p.lab.hidden = j !== i;
      }));
      const m = MIXES[i];
      const mv = S[m.key].band_mean_shift_in_slots;
      out.textContent = H[m.key].name.replace(/^after/, 'After').replace(/ %/g, '\u00a0%') + ': the band mean moved by ' +
        mv.map((v, b) => signed(v, 2) + ' (layers ' + D.bands[b].layers[0] + ' to ' + D.bands[b].layers[1] + ')').join(', ') +
        ' tokens of the 50; ' + mv.filter((v, b) => Math.abs(v) > pos(DR.band_bar[b])).length + ' of ' + NB + ' bands moved past the drift band.';
    }
    select(0);
  }

  WALK.register('f12', { data: 'F12_DATA', mount });
})();
