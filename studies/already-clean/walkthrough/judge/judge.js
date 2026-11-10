// "The judge's calls", shared by the walkthrough page and index.html.
// It builds itself inside the element with id "judgecalls" and reads window.JUDGE_DATA, which data.js sets.
// With no #judgecalls on the page it does nothing. Every call is one dot on a canvas, in one of two columns
// (late-onset groups, control groups), right calls at the bottom and wrong calls above them. The dots fall into
// place once, when the picture first comes into view, while the two shares count up (at once under
// prefers-reduced-motion, or after any press inside the picture). A button replaces every outcome with a fair
// coin flip and back; a menu opens one call with the text the judge saw.
(() => {
  'use strict';
  const root = document.getElementById('judgecalls');
  if (!root) return;
  const D = window.JUDGE_DATA;
  if (!D) {
    root.innerHTML = '<p class="jd-missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      'judge to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in judge');
  }

  // ---------- formatting ----------
  const int = (x) => Math.round(x).toLocaleString('en-US');
  const pct1 = (x) => (100 * x).toFixed(1);
  const step = (t) => int(D.steps[t]);
  const parseId = (id) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    return { layer: Number(m[1]), neuron: m[2], sign: m[3] === '+' ? 'positive' : 'negative' };
  };
  const tokText = (tok) => (tok.trim() === '' ? tok.replace(/ /g, '␣') : tok.trim());
  const COLS = [
    { name: 'Late-onset groups', n: D.n_late, right: D.right.late },
    { name: 'Control groups', n: D.n_control, right: D.right.control },
  ];

  root.innerHTML = `
  <div class="jd-top">
    <p class="jd-mode" id="jd-mode" aria-live="polite"></p>
    <button class="jd-btn" id="jd-coin" type="button" aria-pressed="false">Flip a coin for every call</button>
  </div>
  <div class="jd-plot" id="jd-plot">
    <div class="jd-heads" id="jd-heads">
      ${COLS.map((c, i) => `
      <div class="jd-head" id="jd-head${i}">
        <div class="jd-name">${c.name}</div>
        <div class="jd-sub">${int(c.n)} groups, 4 calls each</div>
        <div class="jd-pct"><span id="jd-pct${i}">0.0</span><span class="jd-unit">%</span></div>
        <div class="jd-count" id="jd-count${i}"></div>
      </div>`).join('')}
    </div>
    <canvas id="jd-canvas" role="img"></canvas>
  </div>
  <div class="jd-legend">
    <span><i class="jd-key jd-r"></i>right: the judge picked the neuron’s own record</span>
    <span><i class="jd-key jd-w"></i>wrong: it picked the other neuron’s record</span>
  </div>
  <p class="jd-share" id="jd-share"></p>
  <div class="jd-call" id="jd-call">
    <div class="jd-pick">
      <label for="jd-select">Open one call</label>
      <div class="jd-selwrap"><select id="jd-select"></select></div>
    </div>
    <p class="jd-who" id="jd-who"></p>
    <h4 id="jd-exh"></h4>
    <div class="jd-ctxs" id="jd-ex"></div>
    <h4 id="jd-rech"></h4>
    <div class="jd-recs">
      <div class="jd-rec" id="jd-rec0"></div>
      <div class="jd-rec" id="jd-rec1"></div>
    </div>
    <p class="jd-verdict" id="jd-verdict"></p>
  </div>
  <details>
    <summary>Show the counts as a table</summary>
    <div class="jd-tablewrap"><table id="jd-table"></table></div>
  </details>
  <p class="jd-caption" id="jd-caption"></p>
  <div class="jd-tip" id="jd-tip" aria-hidden="true"></div>`;

  const $ = (id) => root.querySelector('#jd-' + id);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const plot = $('plot'), canvas = $('canvas'), cx = canvas.getContext('2d');
  const BAND = 4;                      // px between the right calls and the wrong calls of a column
  const PAD = 2;                       // px under the bottom row

  // ---------- the calls ----------
  // D.groups[i] = [neuron, group name, birth point, control (0 or 1), contexts, "1011" outcomes, decoy neurons]
  const calls = [];
  D.groups.forEach((G, g) => {
    for (let k = 0; k < 4; k++) {
      calls.push({ i: calls.length, g, k, col: G[3], judge: G[5][k] === '1', out: G[5][k] === '1',
        x: 0, y: 0, fx: 0, fy: 0, t0: 0, dur: 0, ease: 0 });
    }
  });
  const byCol = [0, 1].map((c) => calls.filter((d) => d.col === c));
  byCol.forEach((list, c) => {
    if (list.length !== 4 * COLS[c].n || list.filter((d) => d.judge).length !== COLS[c].right) {
      throw new Error('judge: data.js counts disagree with its groups');
    }
  });
  const exampleAt = new Map(D.examples.map((E, i) => [E.g + ':' + E.k, i]));
  const callOf = (E) => calls[4 * E.g + E.k];

  // ---------- layout ----------
  let W = 0, H = 0, P = 6, R = 2.2, dpr = 1;
  const geo = [{}, {}];
  let order = [[], []];                // calls of each column in slot order
  let nRight = [0, 0];
  function sortCols() {
    order = byCol.map((list) => list.slice().sort((a, b) => (b.out - a.out) || (a.i - b.i)));
    nRight = order.map((list) => list.filter((d) => d.out).length);
  }
  function slotXY(c, s) {
    const G = geo[c];
    const row = Math.floor(s / G.cols), col = s % G.cols;
    return [G.x0 + col * P + P / 2, H - PAD - row * P - P / 2 - (s >= nRight[c] ? BAND : 0)];
  }
  function place() {
    order.forEach((list, c) => list.forEach((d, s) => { [d.x, d.y] = slotXY(c, s); }));
  }
  function layout() {
    W = plot.clientWidth;
    const narrow = W < 560;
    P = narrow ? 5 : 6;
    R = P * 0.36;
    const gap = narrow ? 18 : 40;
    const tot = Math.floor((W - gap) / P);
    const nL = byCol[0].length, nC = byCol[1].length;
    let rows = Math.ceil((nL + nC) / tot);
    while (Math.ceil(nL / rows) + Math.ceil(nC / rows) > tot) rows++;
    const cL = Math.ceil(nL / rows), cC = Math.ceil(nC / rows);
    geo[0] = { x0: 0, cols: cL, rows: Math.ceil(nL / cL) };
    geo[1] = { x0: cL * P + gap, cols: cC, rows: Math.ceil(nC / cC) };
    H = Math.max(geo[0].rows, geo[1].rows) * P + BAND + PAD + 2;
    dpr = window.devicePixelRatio || 1;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    [0, 1].forEach((c) => {
      const h = $('head' + c);
      h.style.left = geo[c].x0 + 'px';
      h.style.width = (c === 0 ? geo[1].x0 - 12 : W - geo[1].x0) + 'px';
    });
    $('heads').style.setProperty('--jd-headh', (Math.max($('head0').offsetHeight, $('head1').offsetHeight) + 14) + 'px');
    place();
  }

  // ---------- drawing ----------
  const ease = [(t) => t * t, (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)];
  let anim = null;                      // { kind: 'fall' | 'move', end, from: [v0, v1], to: [v0, v1], t0, dur }
  let selected = null, hovered = null;
  function pos(d, now) {
    if (!anim || now >= d.t0 + d.dur) return [d.x, d.y, true];
    if (now <= d.t0) return [d.fx, d.fy, false];
    const t = ease[d.ease]((now - d.t0) / d.dur);
    return [d.fx + (d.x - d.fx) * t, d.fy + (d.y - d.fy) * t, false];
  }
  function draw(now) {
    const css = getComputedStyle(root);
    const color = { right: css.getPropertyValue('--jd-right').trim(), wrong: css.getPropertyValue('--jd-wrong').trim(),
      rule: css.getPropertyValue('--rule').trim(), ink: css.getPropertyValue('--ink').trim(), bg: css.getPropertyValue('--bg').trim() };
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, W, H);
    cx.fillStyle = color.rule;
    geo.forEach((G) => cx.fillRect(G.x0, H - 1, G.cols * P, 1));
    if (waiting) return;
    [true, false].forEach((out) => {
      cx.beginPath();
      calls.forEach((d) => {
        if (d.out !== out) return;
        const [x, y] = pos(d, now);
        if (y < -R) return;
        cx.moveTo(x + R, y);
        cx.arc(x, y, R, 0, 2 * Math.PI);
      });
      cx.fillStyle = out ? color.right : color.wrong;
      cx.fill();
    });
    [[selected, 1.6], [hovered, 1.2]].forEach(([d, w]) => {
      if (!d) return;
      const [x, y, landed] = pos(d, now);
      if (!landed) return;
      cx.beginPath();
      cx.arc(x, y, R + 2.6, 0, 2 * Math.PI);
      cx.lineWidth = 2.4;
      cx.strokeStyle = color.bg;
      cx.stroke();
      cx.beginPath();
      cx.arc(x, y, R + 2.6, 0, 2 * Math.PI);
      cx.lineWidth = w;
      cx.strokeStyle = color.ink;
      cx.stroke();
    });
  }

  // ---------- the two shares ----------
  let shown = [0, 0];                   // share right now on screen, per column
  function target() { return [0, 1].map((c) => nRight[c] / order[c].length); }
  function setNumbers(v, landedRight) {
    shown = v;
    const coin = mode === 'coin';
    [0, 1].forEach((c) => {
      $('pct' + c).textContent = pct1(v[c]);
      const k = landedRight ? landedRight[c] : Math.round(v[c] * order[c].length);
      $('count' + c).textContent = (coin ? 'right by coin: ' : 'right: ') + int(k) + ' of ' + int(order[c].length) + ' calls';
    });
  }
  function numbers(now) {
    if (anim && anim.kind === 'fall' && now < anim.end) {
      const landed = [0, 1].map((c) => order[c].reduce((k, d) => k + (d.out && now >= d.t0 + d.dur ? 1 : 0), 0));
      setNumbers(landed.map((k, c) => k / order[c].length), landed);
    } else if (anim && anim.kind === 'move' && now < anim.end) {
      const t = ease[1](Math.min(1, Math.max(0, (now - anim.t0) / anim.dur)));
      setNumbers([0, 1].map((c) => anim.from[c] + (anim.to[c] - anim.from[c]) * t));
    } else {
      setNumbers(target());
    }
  }
  function describePlot() {
    const t = target();
    canvas.setAttribute('aria-label', (mode === 'coin' ? 'A fair coin in place of the judge. ' : 'The judge’s calls. ') +
      COLS.map((c, i) => c.name + ': ' + int(nRight[i]) + ' of ' + int(order[i].length) + ' calls right, ' +
        pct1(t[i]) + '%.').join(' '));
  }

  // ---------- animation ----------
  let raf = 0;
  function loop(now) {
    draw(now);
    numbers(now);
    if (anim && now < anim.end) { raf = requestAnimationFrame(loop); return; }
    anim = null;
    raf = 0;
    draw(now);
    numbers(now);
  }
  function run(a) {
    anim = a;
    if (!raf) raf = requestAnimationFrame(loop);
  }
  function settle() {
    anim = null;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    const now = performance.now();
    draw(now);
    numbers(now);
  }
  function fall() {
    waiting = false;
    const now = performance.now() + 40;
    let end = now;
    order.forEach((list, c) => list.forEach((d, s) => {
      const row = Math.floor(s / geo[c].cols);
      d.fx = d.x;
      d.fy = -8 - Math.random() * 60;
      d.t0 = now + (row / geo[c].rows) * 1150 + Math.random() * 180;
      d.dur = 380 + 260 * (d.y / H);
      d.ease = 0;
      end = Math.max(end, d.t0 + d.dur);
    }));
    run({ kind: 'fall', end });
  }
  function move() {
    const now = performance.now();
    const from = shown.slice();
    const old = calls.map((d) => pos(d, now));
    sortCols();
    place();
    describePlot();
    if (reduce.matches) { settle(); return; }
    const t0 = now + 20;
    let end = t0;
    calls.forEach((d, i) => {
      [d.fx, d.fy] = old[i];
      d.t0 = t0 + Math.random() * 220;
      d.dur = 680;
      d.ease = 1;
      end = Math.max(end, d.t0 + d.dur);
    });
    run({ kind: 'move', end, t0, dur: end - t0, from, to: target() });
  }

  // ---------- the coin ----------
  let mode = 'judge';
  const coinBtn = $('coin');
  function drawMode() {
    const coin = mode === 'coin';
    coinBtn.setAttribute('aria-pressed', String(coin));
    coinBtn.textContent = coin ? 'Show the judge’s calls' : 'Flip a coin for every call';
    $('mode').textContent = coin
      ? 'A fair coin in place of the judge, flipped just now for each of the ' + int(calls.length) + ' calls.'
      : 'The judge’s ' + int(calls.length) + ' calls, one dot each.';
  }
  coinBtn.addEventListener('click', () => {
    userActed();
    mode = mode === 'coin' ? 'judge' : 'coin';
    calls.forEach((d) => { d.out = mode === 'coin' ? Math.random() < 0.5 : d.judge; });
    drawMode();
    move();
  });

  // ---------- one call ----------
  const select = $('select');
  const KINDS = [
    ['named', 'Named in the write-up'],
    ['late-right', 'Late-onset groups, the judge right (drawn at random)'],
    ['late-wrong', 'Late-onset groups, the judge wrong (drawn at random)'],
    ['control-right', 'Control groups, the judge right (drawn at random)'],
    ['control-wrong', 'Control groups, the judge wrong (drawn at random)'],
  ];
  const DRAWN = {
    'late-right': 'calls on late-onset groups that the judge got right',
    'late-wrong': 'calls on late-onset groups that the judge got wrong',
    'control-right': 'calls on control groups that the judge got right',
    'control-wrong': 'calls on control groups that the judge got wrong',
  };
  KINDS.forEach(([kind, label]) => {
    const og = document.createElement('optgroup');
    og.label = label;
    D.examples.forEach((E, i) => {
      if (E.kind !== kind) return;
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = E.label || (D.groups[E.g][0] + ': ' + D.groups[E.g][1]);
      og.appendChild(o);
    });
    select.appendChild(og);
  });

  function appendText(el, text) {
    text.split('↵').forEach((part, i) => {
      if (i > 0) {
        const nl = document.createElement('span');
        nl.className = 'jd-nl';
        nl.textContent = '↵';
        el.appendChild(nl);
      }
      if (part) el.appendChild(document.createTextNode(part.replace(/\t/g, '  ')));
    });
  }
  function contextRows(box, list) {
    box.textContent = '';
    list.forEach(([before, tok, after]) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'jd-ctx';
      row.setAttribute('aria-expanded', 'false');
      const b = document.createElement('span');
      b.className = 'jd-b';
      const bi = document.createElement('span');
      appendText(bi, before);
      b.appendChild(bi);
      const t = document.createElement('span');
      t.className = 'jd-t';
      t.textContent = tokText(tok);
      const a = document.createElement('span');
      a.className = 'jd-a';
      appendText(a, after);
      row.append(b, t, a);
      row.addEventListener('click', () => {
        const open = row.classList.toggle('jd-open');
        row.setAttribute('aria-expanded', String(open));
      });
      box.appendChild(row);
    });
  }
  const plural = (n, one, many) => int(n) + ' ' + (n === 1 ? one : many);

  function openCall(i) {
    const E = D.examples[i];
    const [neuron, name, birth, control, size, outs, decoys] = D.groups[E.g];
    const id = parseId(neuron);
    const decoy = decoys[E.k];
    const ownFirst = E.order[0] === 't';
    const right = (E.answer === 'A') === ownFirst;
    const nRightGroup = outs.split('').filter((x) => x === '1').length;
    select.value = String(i);
    selected = callOf(E);
    if (!anim) draw(performance.now());

    const who = $('who');
    who.textContent = '';
    const b = document.createElement('b');
    b.textContent = neuron;
    const q = document.createElement('q');
    q.textContent = name;
    who.append(b, ' Layer ' + id.layer + ', neuron ' + id.neuron + ', ' + id.sign + ' activations. Its group ', q,
      ' (' + plural(size, 'context', 'contexts') + ', named by the clustering LLM) ' +
      (control
        ? 'is the neuron’s strongest at the end of training, so this call is a control: the records come from step ' +
          step(E.t) + ', the last saved checkpoint before the end. '
        : 'had its onset at step ' + step(birth) + ', so the records come from step ' + step(E.t) +
          ', the checkpoint before. ') +
      (E.kind === 'named' ? '' : 'Drawn at random from the ' + DRAWN[E.kind] + '.'));

    $('exh').textContent = 'The group’s contexts at the end of training';
    const note = document.createElement('span');
    note.className = 'jd-h-note';
    note.textContent = E.ex.length < size ? ' (the judge saw its strongest ' + E.ex.length + ' of ' + size + ')' : '';
    $('exh').appendChild(note);
    contextRows($('ex'), E.ex);

    $('rech').textContent = 'Two records at step ' + step(E.t) + ', in the order the judge saw them';
    const rnote = document.createElement('span');
    rnote.className = 'jd-h-note';
    rnote.textContent = ' (press a context to read all of it)';
    $('rech').appendChild(rnote);
    [E.a, E.b].forEach((list, r) => {
      const box = $('rec' + r);
      const own = (r === 0) === ownFirst;
      const picked = (E.answer === 'A') === (r === 0);
      box.classList.toggle('jd-picked', picked);
      box.textContent = '';
      const head = document.createElement('div');
      head.className = 'jd-rec-head';
      const nm = document.createElement('span');
      nm.className = 'jd-rec-name';
      nm.textContent = 'Record ' + 'AB'[r];
      head.appendChild(nm);
      if (picked) {
        const tag = document.createElement('span');
        tag.className = 'jd-tag';
        tag.textContent = 'The judge’s pick';
        head.appendChild(tag);
      }
      const whose = document.createElement('div');
      whose.className = 'jd-whose';
      if (own) {
        whose.append('This neuron’s own record, ' + plural(list.length, 'context', 'contexts') + '.');
      } else {
        const db = document.createElement('b');
        db.textContent = decoy;
        whose.append('The record of ', db, ', a neuron of layer ' + id.layer + ' drawn at random, ' +
          plural(list.length, 'context', 'contexts') + '.');
      }
      const ctxs = document.createElement('div');
      ctxs.className = 'jd-ctxs';
      contextRows(ctxs, list);
      box.append(head, whose, ctxs);
    });

    const v = $('verdict');
    v.textContent = '';
    const key = document.createElement('i');
    key.className = 'jd-key ' + (right ? 'jd-r' : 'jd-w');
    const span = document.createElement('span');
    const strong = document.createElement('b');
    strong.textContent = right ? 'Right.' : 'Wrong.';
    span.append(strong, ' The judge picked record ' + E.answer + ', ' +
      (right ? 'the neuron’s own.' : 'the record of ' + decoy + '.') +
      ' On this group it was right in ' + nRightGroup + ' of its 4 calls.');
    v.append(key, span);
  }
  select.addEventListener('change', () => { userActed(); openCall(Number(select.value)); });

  // ---------- pointer: the call under the pointer ----------
  const tip = $('tip');
  function hit(px, py) {
    for (let c = 0; c < 2; c++) {
      const G = geo[c];
      if (px < G.x0 - P / 2 || px > G.x0 + G.cols * P + P / 2) continue;
      const col = Math.min(G.cols - 1, Math.max(0, Math.floor((px - G.x0) / P)));
      for (const shift of [0, BAND]) {
        const row = Math.floor((H - PAD - py - shift) / P);
        if (row < 0) continue;
        const s = row * G.cols + col;
        if (s >= order[c].length) continue;
        if ((s >= nRight[c]) === (shift === BAND)) return order[c][s];
      }
    }
    return null;
  }
  function showTip(d) {
    const [neuron, name, birth, control, size, , decoys] = D.groups[d.g];
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = neuron;
    const t = control ? D.control_before : birth - 1;
    tip.append(b, ' “' + name + '”, ' + plural(size, 'context', 'contexts') + '. ' +
      (control ? 'Control group, judged against the records at step ' + step(t)
        : 'Onset at step ' + step(birth) + ', judged against the records at step ' + step(t)) +
      ', beside ' + decoys[d.k] + '. The judge was ' + (d.judge ? 'right' : 'wrong') + '.' +
      (mode === 'coin' ? ' The coin was ' + (d.out ? 'right' : 'wrong') + '.' : '') +
      (exampleAt.has(d.g + ':' + d.k) ? ' Press to open this call below.' : ''));
    const rr = root.getBoundingClientRect();
    const cr = canvas.getBoundingClientRect();
    tip.classList.add('jd-on');
    const w = tip.offsetWidth;
    const x = cr.left - rr.left + d.x;
    tip.style.left = Math.max(0, Math.min(rr.width - w, x - w / 2)) + 'px';
    tip.style.top = (cr.top - rr.top + d.y - tip.offsetHeight - 12) + 'px';
  }
  function hideTip() {
    tip.classList.remove('jd-on');
    if (hovered) { hovered = null; if (!anim) draw(performance.now()); }
  }
  canvas.addEventListener('pointermove', (e) => {
    if (anim) return;
    const r = canvas.getBoundingClientRect();
    const d = hit(e.clientX - r.left, e.clientY - r.top);
    if (!d) { hideTip(); canvas.style.cursor = ''; return; }
    canvas.style.cursor = exampleAt.has(d.g + ':' + d.k) ? 'pointer' : '';
    if (d !== hovered) { hovered = d; draw(performance.now()); }
    showTip(d);
  });
  // a touch lifts with a pointerleave; its tooltip stays until the next press elsewhere
  canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideTip(); });
  document.addEventListener('pointerdown', (e) => { if (e.target !== canvas) hideTip(); });
  canvas.addEventListener('pointerdown', (e) => {
    userActed();
    const r = canvas.getBoundingClientRect();
    const d = hit(e.clientX - r.left, e.clientY - r.top);
    if (!d) { hideTip(); return; }
    hovered = d;
    draw(performance.now());
    showTip(d);
    const i = exampleAt.get(d.g + ':' + d.k);
    if (i !== undefined) {
      openCall(i);
      $('call').scrollIntoView({ behavior: reduce.matches ? 'auto' : 'smooth', block: 'start' });
    }
  });

  // ---------- text: the derived share, the table, the caption ----------
  $('share').textContent =
    'If the judge is right on a late-onset group that grew from its neuron’s earlier record as often as on a ' +
    'control group, ' + pct1(D.a_ctrl) + '% of the time, and on an unrelated one half the time, then ' + pct1(D.A) +
    '% right means ' + pct1(D.share) + '% of late-onset groups are unrelated to that record, with a 95% range of ' +
    Math.round(100 * D.ci95[0]) + ' to ' + Math.round(100 * D.ci95[1]) + '%.';

  const table = $('table');
  const head = table.insertRow();
  ['', 'Groups', 'Calls', 'Right', 'Share right', '0 of 4 right', '1 of 4', '2 of 4', '3 of 4', '4 of 4']
    .forEach((h) => { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); });
  const coinShares = [1, 4, 6, 4, 1].map((k) => k / 16);
  const coinCounts = coinShares.map((s) => Math.round(s * D.n_late));
  [
    ['Late-onset groups', D.n_late, 4 * D.n_late, D.right.late, D.right.late / (4 * D.n_late), D.hist.late],
    ['Control groups', D.n_control, 4 * D.n_control, D.right.control, D.right.control / (4 * D.n_control), D.hist.control],
    ['A fair coin on the late-onset groups, expected', D.n_late, 4 * D.n_late, 2 * D.n_late, 0.5, coinCounts],
  ].forEach(([label, g, n, k, s, hist]) => {
    const tr = table.insertRow();
    [label, int(g), int(n), int(k), pct1(s) + '%', ...hist.map(int)].forEach((v) => { tr.insertCell().textContent = v; });
  });

  $('caption').textContent =
    'Control groups are each neuron’s strongest group, judged against its own record at step ' + step(D.control_before) +
    ', so the right answer is known; they are easier, sharing a marked token with that record in ' + pct1(D.overlap.control) +
    '% of calls against ' + pct1(D.overlap.late) + '% for late-onset groups. A right call means the neuron’s own record ' +
    'looked closer to the group than a random neuron’s, not that the group grew from it.';

  // ---------- once, on view ----------
  let pending = !reduce.matches;
  let waiting = pending;               // before the fall the canvas shows only the two baselines
  let pendingTimer = 0;
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    seen.disconnect();
    if (!pending) return;
    pendingTimer = setTimeout(() => {
      if (!pending) return;
      pending = false;
      fall();
    }, 450);
  }, { threshold: 0.35 });
  function userActed() {
    if (pending) {
      pending = false;
      waiting = false;
      clearTimeout(pendingTimer);
      seen.disconnect();
      settle();
    } else if (anim && anim.kind === 'fall') {
      settle();
    }
  }

  // ---------- size, fonts and theme ----------
  let lastW = 0;
  new ResizeObserver(() => {
    if (plot.clientWidth === lastW) return;
    lastW = plot.clientWidth;
    layout();
    if (waiting) { draw(0); return; }
    settle();
  }).observe(plot);
  if (document.fonts) document.fonts.ready.then(() => { layout(); if (waiting) draw(0); else settle(); });
  const redraw = () => { if (!anim) draw(performance.now()); };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------- first render ----------
  sortCols();
  drawMode();
  layout();
  describePlot();
  openCall(0);
  if (waiting) {
    draw(0);
    setNumbers([0, 0], [0, 0]);
    seen.observe(plot);
  } else {
    settle();
  }
})();
