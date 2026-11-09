// m12: an edit along a direction, on one side. Registers the mount "m12" with site.js; reads window.M12_DATA
// (m12/data.js). For one of the 8 control sides: the three edits of its down-matrix column (own direction, random
// direction of the same size, own direction at half), each traced through its parts: the edit vector delta and its
// size c, the change to the down matrix's output over the side's tokens, the change R to the shared vector after the
// layer, and the loss change per token. Below: the measurement's rounding checks and the rule that forms the pairs.
(() => {
  'use strict';

  const MINUS = '−';
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const sideOf = (id) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    if (!m) throw new Error('m12: cannot read the side ' + id);
    return { layer: Number(m[1]), unit: Number(m[2]), side: m[3] === '+' ? 'positive' : 'negative' };
  };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

  function mount(root, D) {
    const F = WALK.fmt;
    const L = D.control_layers;
    if (L.length !== D.control_of_layers) throw new Error('m12: ' + L.length + ' control sides, the data says ' + D.control_of_layers);
    const KINDS = ['own_direction', 'random_direction', 'own_direction_half_fraction'];
    const NAME = { own_direction: 'own direction', random_direction: 'random direction', own_direction_half_fraction: 'own direction, half' };
    const CLS = { own_direction: 'own', random_direction: 'rand', own_direction_half_fraction: 'half' };
    const SHORT = { own_direction: 'own', random_direction: 'random', own_direction_half_fraction: 'own, half' };
    const TAU = D.tie_margin;
    L.forEach((x) => {
      if (x.edits.map((e) => e.kind).join() !== KINDS.join()) throw new Error('m12: layer ' + x.layer + ': the edits are not own, random, half');
      const [o, r, h] = x.edits;
      if (o.edit_size_c !== r.edit_size_c || o.edit_size_c !== o.column_norm || o.fraction_removed !== 1) throw new Error('m12: layer ' + x.layer + ': the own and random edits do not share c = ‖w‖');
      if (Math.abs(h.edit_size_c - 0.5 * h.column_norm) > 1e-6 || h.fraction_removed !== 0.5) throw new Error('m12: layer ' + x.layer + ': the half edit is not −0.5 · w');
      x.edits.forEach((e) => {
        if (Math.abs(e.edit_vector_norm - e.edit_size_c) > 1e-6 * e.edit_size_c) throw new Error('m12: layer ' + x.layer + ': ‖δ‖ as applied differs from c');
        if (!(e.loss_change_per_token > 0) || !(e.shared_vector_change > 0)) throw new Error('m12: layer ' + x.layer + ': a loss or shared-vector change is not positive; the bars start at zero');
      });
      if (Math.abs(r.shared_vector_change / o.shared_vector_change - x.shared_vector_ratio_random_over_own) > 1e-12) throw new Error('m12: layer ' + x.layer + ': the ratio does not follow from R');
      if ((o.loss_change_per_token > r.loss_change_per_token + TAU) !== x.own_cost_more_than_random) throw new Error('m12: layer ' + x.layer + ': which edit cost more does not follow from the loss changes');
      x.w = sideOf(x.side);
      if (x.w.layer !== x.layer) throw new Error('m12: layer ' + x.layer + ': the side is in layer ' + x.w.layer);
    });
    const ownMore = L.filter((x) => x.own_cost_more_than_random).length;
    if (ownMore !== D.control_own_cost_more_count.recomputed) throw new Error('m12: ' + ownMore + ' layers where the own direction cost more, the data ' + D.control_own_cost_more_count.recomputed);
    const EV = D.evaluation;
    if (EV.windows * EV.tokens_per_window !== EV.tokens) throw new Error('m12: windows × tokens per window is not the token count');
    const PR = D.pairs;

    // ---------- formatting ----------
    const sig = (x, n) => {
      const a = Math.abs(x);
      if (a === 0) return '0';
      const d = Math.max(0, n - 1 - Math.floor(Math.log10(a)));
      return F.fixed(x, d);
    };
    const sci = (x, n) => {
      if (x === 0) return '0';
      const p = Math.floor(Math.log10(Math.abs(x)));
      if (p >= -3 && p < 6) return sig(x, n);
      const m = x / Math.pow(10, p);
      return F.fixed(m, n - 1) + '\u00a0×\u00a010' + String(p).split('').map((ch) => SUP[ch]).join('');
    };
    const big = (x) => (Math.abs(x) >= 100 ? F.fixed(x, 1) : sig(x, 4));
    const c4 = (x) => F.fixed(x, 4);
    const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
    const sideText = (w) => 'layer ' + w.layer + ', unit ' + F.int(w.unit) + ', ' + w.side + ' side';

    // ---------- the frame ----------
    root.replaceChildren();
    const wrap = el('div', 'm12-wrap');
    root.appendChild(wrap);
    const pickRow = el('div', 'control-row');
    const pickLab = el('label', 'control-label', 'Control side');
    pickLab.htmlFor = 'm12-pick';
    const pick = el('select', 'm12-select');
    pick.id = 'm12-pick';
    L.forEach((x, i) => {
      const o = el('option', '', cap(sideText(x.w)));
      o.value = String(i);
      pick.appendChild(o);
    });
    pickRow.append(pickLab, pick);
    wrap.appendChild(pickRow);
    const who = el('p', 'm12-who');
    wrap.appendChild(who);
    const edits = el('div', 'm12-edits');
    wrap.appendChild(edits);
    const chain = el('div', 'm12-chain');
    wrap.appendChild(chain);
    const sumBox = el('p', 'm12-sum');
    wrap.appendChild(sumBox);

    // the parts that do not depend on the side
    const RC = D.rounding_checks_and_zero_edit;
    const checks = RC.filter((r) => r.kind === 'rounding_check');
    const zero = RC.filter((r) => r.kind === 'zero_edit');
    if (zero.length !== 1 || !checks.length) throw new Error('m12: expected rounding checks and one zero edit');
    const maxL = Math.max(...checks.map((r) => Math.abs(r.loss_change_per_token)));
    const maxR = Math.max(...checks.map((r) => r.shared_vector_change));
    if (Math.abs(10 * maxL - TAU) > 1e-20) throw new Error('m12: the tie margin is not ten times the largest rounding check');
    const floor = el('div', 'm12-block');
    floor.appendChild(el('h3', 'm12-h', 'How small a change the measurement resolves'));
    floor.appendChild(el('p', 'm12-p', 'A rounding check adds a random δ and subtracts it again in the same pass, which is zero in exact arithmetic. The ' + checks.length +
      ' rounding checks moved the loss by at most ' + sci(maxL, 3) + ' nats per token and the shared vector by at most ' + sig(maxR, 3) + '. The edit with δ = 0 moved the loss by ' +
      sci(zero[0].loss_change_per_token, 2) + ' and the shared vector by ' + big(zero[0].shared_vector_change) + '. Two loss changes closer than ' + sci(TAU, 3) +
      ' (ten times the largest rounding check) count as a tie.'));
    wrap.appendChild(floor);

    if (PR.edit_size_rule_holds !== PR.edit_size_rule_of_pairs || PR.rule_setting.recomputed !== PR.edit_size_rule_of_pairs) throw new Error('m12: the edit size rule does not hold in every pair');
    const pairs = el('div', 'm12-block');
    pairs.appendChild(el('h3', 'm12-h', 'The pairs'));
    pairs.appendChild(el('p', 'm12-p', 'A pair is two dated sides of one layer, of different units, whose births are at least three snapshots apart, matched on activation ' +
      'size and density (ln max |a| and ln density), then on ln ‖w‖ and ln E⁺. All four edits of a pair, own and random on each side, have one size: c = min(‖w_early‖, ‖w_late‖). ' +
      'The side with the smaller column has its write removed fully; the other loses the fraction c/‖w‖ of it. The data holds this in ' + PR.edit_size_rule_holds + ' of ' +
      PR.edit_size_rule_of_pairs + ' pairs of the half used to set the rules; the ' + PR.held_out + ' pairs of the held-out half were never read.'));
    pairs.appendChild(el('p', 'm12-p', 'R is ‖h_edit − h_base‖ over all evaluation positions and the 4,096 coordinates of the shared vector, read after the edited layer (after the norm and the add). ' +
      'The loss change is the mean over ' + F.int(EV.windows) + ' windows × ' + F.int(EV.tokens_per_window) + ' tokens (' + F.int(EV.tokens) +
      ' tokens, from outside the 20.6-million-token corpus) of the edited loss minus the base loss, in nats per token, in 32-bit.'));
    wrap.appendChild(pairs);

    // ---------- one side ----------
    function draw(i) {
      const x = L[i];
      const [o, r] = x.edits;
      const w = o.column_norm;
      const rootE = Math.sqrt(x.own_sign_energy);
      pick.value = String(i);

      who.replaceChildren();
      who.append(el('b', '', cap(sideText(x.w))), ': the side of layer ' + x.layer + ' with the largest E⁺, the sum over the evaluation tokens of its own-sign ' +
        'activation squared: E⁺ = ' + F.fixed(x.own_sign_energy, 1) + '. Its down-matrix column w has ‖w‖ = ' + c4(w) +
        '. At each token x where the side fires, an edit adds a(x) · δ to the down matrix’s output, a change of size |a(x)| · ‖δ‖; over all its tokens the energy added is ‖δ‖² · E⁺.');

      edits.replaceChildren();
      const line = (e, formula) => {
        const p = el('p', 'm12-edit');
        p.append(el('i', 'm12-key ' + CLS[e.kind]), el('b', '', cap(NAME[e.kind]) + ': '), formula);
        edits.appendChild(p);
      };
      line(x.edits[0], 'δ = −(c/‖w‖) · w with c = ' + c4(o.edit_size_c) + ', so c/‖w‖ = ' + F.fixed(o.fraction_removed, 3) + ': the side’s write is removed fully. ‖δ‖ as applied: ' + c4(o.edit_vector_norm) + '.');
      line(x.edits[1], 'δ = c · r, r a seeded random unit vector, with the same c = ' + c4(r.edit_size_c) + '. ‖δ‖ as applied: ' + c4(r.edit_vector_norm) + '.');
      line(x.edits[2], 'δ = −0.5 · w, half the side’s write removed: c = ' + c4(x.edits[2].edit_size_c) + '. ‖δ‖ as applied: ' + c4(x.edits[2].edit_vector_norm) + '.');

      chain.replaceChildren();
      const group = (n, title, unit, val, fmtv, exact) => {
        const g = el('section', 'm12-group');
        g.appendChild(el('h3', 'm12-h', n + '. ' + title));
        const max = Math.max(...x.edits.map(val));
        x.edits.forEach((e) => {
          const row = el('div', 'm12-row');
          const nm = el('span', 'm12-name');
          nm.append(el('i', 'm12-key ' + CLS[e.kind]), SHORT[e.kind]);
          const track = el('span', 'm12-track');
          const bar = el('span', 'm12-bar ' + CLS[e.kind]);
          bar.style.width = (100 * val(e)) / max + '%';
          track.appendChild(bar);
          const v = el('span', 'm12-val', fmtv(val(e)));
          row.append(nm, track, v);
          WALK.hover(row, () => ({ title: cap(NAME[e.kind]) + ', ' + sideText(x.w), rows: [[title + (unit ? ' (' + unit + ')' : ''), exact(val(e)), CLS[e.kind] === 'own' ? 'claim' : CLS[e.kind] === 'rand' ? 'control' : 'muted']] }));
          g.appendChild(row);
        });
        chain.appendChild(g);
        return g;
      };
      const full = (v) => String(v);
      group(1, 'The edit’s size, ‖δ‖ as applied', '', (e) => e.edit_vector_norm, c4, full);
      const g2 = group(2, 'The change to the down matrix’s output over the side’s tokens, ‖δ‖ · √E⁺', '', (e) => e.edit_size_c * rootE, big, (v) => F.fixed(v, 3));
      g2.appendChild(el('p', 'm12-note', 'c · √E⁺ with √E⁺ = ' + F.fixed(rootE, 2) + ': ' + c4(o.edit_size_c) + ' × ' + F.fixed(rootE, 2) + ' = ' + big(o.edit_size_c * rootE) +
        ' for the own and the random edit alike.'));
      const g3 = group(3, 'The change to the shared vector after the layer, R', '', (e) => e.shared_vector_change, big, (v) => F.fixed(v, 4));
      g3.appendChild(el('p', 'm12-note', 'Random ÷ own: ' + F.fixed(x.shared_vector_ratio_random_over_own, 2) + '.'));
      const g4 = group(4, 'The loss change per token', 'nats', (e) => e.loss_change_per_token, (v) => sig(v, 3), (v) => sci(v, 6));
      const dl = r.loss_change_per_token - o.loss_change_per_token;
      g4.appendChild(el('p', 'm12-note', (Math.abs(dl) <= TAU ? 'A tie: the two loss changes are within ' + sci(TAU, 3) + '.' :
        (dl > 0 ? 'The random edit cost more, by ' : 'The own-direction edit cost more, by ') + sig(Math.abs(dl), 3) + ' nats per token.') + ' The edit at half cost ' +
        sig(x.edits[2].loss_change_per_token, 3) + '.'));

      sumBox.textContent = 'The control was set up so that, on each of the ' + L.length + ' control sides, the own-direction edit should cost more than the random edit; ' +
        'it needed this in at least ' + D.control_needed + ' of ' + L.length + '. It cost more in ' + ownMore + ' (layers ' +
        L.filter((y) => y.own_cost_more_than_random).map((y) => y.layer).join(' and ') + '). The random edit moved the shared vector more on every one of them.';
      if (!L.every((y) => y.shared_vector_ratio_random_over_own > 1)) throw new Error('m12: the random edit did not move the shared vector more on every control side');
    }
    pick.addEventListener('change', () => draw(Number(pick.value)));
    draw(0);
  }

  WALK.register('m12', { data: 'M12_DATA', mount });
})();
