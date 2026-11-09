// "One neuron's 50 contexts, sorted into groups", shared by the walkthrough page and index.html.
// It builds itself inside the element with id "ngroups" and reads window.GROUPS_DATA, which data.js sets.
// With no #ngroups on the page it does nothing. The 50 contexts start as one list in rank order and move
// into their groups once, when the picture first comes into view (at once under prefers-reduced-motion).
(() => {
  'use strict';
  const root = document.getElementById('ngroups');
  if (!root) return;
  const D = window.GROUPS_DATA;
  if (!D) {
    root.innerHTML = '<p class="missing">data.js did not load. Run <code>python3 extract.py</code> in the folder ' +
      'groups to write it, then reload the page.</p>';
    throw new Error('data.js did not load: run python3 extract.py in groups');
  }

  root.innerHTML = `
  <div class="controls">
    <div class="pick">
      <label class="sr" for="ngroups-neuron">Neuron</label>
      <div class="select-wrap"><select id="ngroups-neuron"></select></div>
      <button class="btn" id="ngroups-random" type="button">Random neuron</button>
    </div>
    <div class="seg" role="group" aria-label="Which sorting">
      <button type="button" data-s="0" aria-pressed="true">First (used)</button>
      <button type="button" data-s="1" aria-pressed="false">Second (repeat)</button>
    </div>
  </div>
  <p class="who" id="ngroups-who"></p>
  <div class="head">
    <div class="count" aria-live="polite">
      <div class="fig" id="ngroups-fig"><span class="num" id="ngroups-num"></span><span class="unit" id="ngroups-unit"></span></div>
      <p class="note" id="ngroups-note"></p>
    </div>
    <figure class="dist">
      <h3 id="ngroups-distTitle"></h3>
      <div class="bars" id="ngroups-bars"></div>
      <div class="axis" id="ngroups-axis" aria-hidden="true"></div>
    </figure>
  </div>
  <div class="board" id="ngroups-board" role="group" aria-label="The neuron's 50 contexts, placed in their groups">
    <span class="pilehead" id="ngroups-pilehead">The 50 strongest contexts at the end of training, strongest first</span>
  </div>
  <div class="pane" id="ngroups-pane" hidden>
    <h3><span class="v">activation</span><span id="ngroups-paneTitle"></span></h3>
    <div class="ctxs" id="ngroups-ctxs"></div>
  </div>
  <details>
    <summary>Show how many groups the ${D.n_study} neurons got, as a table</summary>
    <div class="tablewrap"><table id="ngroups-table"></table></div>
  </details>
  <p class="caption" id="ngroups-caption"></p>
  <div class="tip" id="ngroups-tip" aria-hidden="true"></div>`;

  const $ = (id) => root.querySelector('#ngroups-' + id);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const ORD = ['first', 'second'];
  const TH = 26, GAP = 4, PAD = 10;
  const MOVE_MS = 720;

  // ---------- formatting ----------
  const int = (x) => Math.round(x).toLocaleString('en-US');
  const fmt = (x) => (Math.abs(x) >= 100 ? int(x) : String(Number(x.toPrecision(3))));
  const quoted = (tok) => "'" + tok.replace(/\n/g, '\\n').replace(/\t/g, '\\t') + "'";
  const pillText = (tok) => {
    const t = tok.replace(/\n/g, '↵').replace(/\t/g, '⇥');
    return t.trim() === '' ? t.replace(/ /g, '␣') : t.trim();
  };
  const flat = (s) => s.replace(/\n/g, '↵').replace(/\t/g, ' ');
  const plural = (n, one, many) => n + ' ' + (n === 1 ? one : many);
  const describe = (id) => {
    const m = /^L(\d+)N(\d+)([+-])$/.exec(id);
    return 'Layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' activations.';
  };
  const topToken = (n) => {
    const c = new Map();
    n.ctx.forEach(([, tok]) => c.set(tok, (c.get(tok) || 0) + 1));
    let best = '', k = 0;
    c.forEach((v, tok) => { if (v > k) { k = v; best = tok; } });
    return best;
  };
  const ordered = (n, s) => n.sortings[s].map(([name, ranks]) => ({ name, ranks }))
    .sort((a, b) => b.ranks.length - a.ranks.length || a.ranks[0] - b.ranks[0]);

  // ---------- picker ----------
  const select = $('neuron');
  [['named', 'Examples'], ['random', 'Drawn at random']].forEach(([kind, label]) => {
    const members = D.neurons.map((n, i) => [n, i]).filter(([n]) => n.kind === kind);
    const og = document.createElement('optgroup');
    og.label = label + ' (' + members.length + ')';
    members.forEach(([n, i]) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = n.id + '  ' + (kind === 'named' ? n.label : quoted(topToken(n)));
      og.appendChild(o);
    });
    select.appendChild(og);
  });
  const randomIdx = D.neurons.map((n, i) => (n.kind === 'random' ? i : -1)).filter((i) => i >= 0);

  // ---------- the distribution over the study's neurons ----------
  const NB = D.hist[0].length;
  const HMAX = Math.max(...D.hist[0], ...D.hist[1]);
  const bars = $('bars'), axis = $('axis');
  bars.style.setProperty('--nb', NB);
  axis.style.setProperty('--nb', NB);
  const barEls = [];
  for (let k = 1; k <= NB; k++) {
    const cell = document.createElement('div');
    cell.className = 'barcell';
    cell.dataset.k = String(k);
    const i = document.createElement('i');
    const lab = document.createElement('span');
    lab.className = 'barlab';
    cell.append(i, lab);
    bars.appendChild(cell);
    const ax = document.createElement('span');
    ax.textContent = String(k);
    if (k !== 1 && k % 5) ax.className = 'minor';
    axis.appendChild(ax);
    barEls.push({ cell, i, lab, ax });
  }
  const table = $('table');
  const head = table.insertRow();
  ['Groups', ...Array.from({ length: NB }, (_, k) => String(k + 1))].forEach((h) => {
    const th = document.createElement('th');
    th.textContent = h;
    head.appendChild(th);
  });
  D.hist.forEach((h, s) => {
    const tr = table.insertRow();
    [ORD[s][0].toUpperCase() + ORD[s].slice(1) + ' sorting', ...h.map(String)].forEach((v) => { tr.insertCell().textContent = v; });
  });

  // ---------- state ----------
  const board = $('board');
  const pilehead = $('pilehead');
  let N = D.neurons[0];
  let si = 0;
  let groups = ordered(N, si);
  let sorted = false;
  let sel = 0;
  let hlRank = -1;
  let tiles = [];
  let rows = [];
  let resetTimer = 0, heroTimer = 0;

  // ---------- tiles ----------
  function buildTiles() {
    tiles.forEach((t) => t.el.remove());
    tiles = N.ctx.map(([, tok], r) => {
      const el = document.createElement('span');
      el.className = 'tile';
      el.setAttribute('aria-hidden', 'true');
      const s = document.createElement('span');
      s.textContent = pillText(tok);
      el.appendChild(s);
      el.dataset.r = String(r);
      board.appendChild(el);
      return { el, r, w: 0, x: 0, y: 0 };
    });
    measure();
  }
  function measure() { tiles.forEach((t) => { t.w = t.el.offsetWidth; }); }
  const place = (t) => { t.el.style.transform = 'translate3d(' + t.x + 'px,' + t.y + 'px,0)'; };

  // ---------- layout ----------
  function geometry() {
    const W = board.clientWidth;
    const wide = W >= 560;
    const labW = wide ? Math.round(Math.max(170, Math.min(250, W * 0.32))) : W;
    return { W, wide, labW, x0: wide ? labW + 20 : 0, tw: wide ? W - labW - 20 : W };
  }
  function flow(list, x0, width, y0) {
    let x = x0, y = y0;
    list.forEach((t) => {
      const w = Math.min(t.w, width);
      if (x > x0 && x + w > x0 + width) { x = x0; y += TH + GAP; }
      t.x = x;
      t.y = y;
      x += w + GAP;
    });
    return list.length ? y + TH : y0;
  }
  function layoutPile() {
    const h = flow(tiles, 0, geometry().W, 44);
    board.style.height = (h + 16) + 'px';
  }
  function layoutGroups() {
    const G = geometry();
    let y = 0;
    rows.forEach((row) => {
      row.lab.style.width = G.labW + 'px';
      const hl = row.lab.offsetHeight;
      const list = row.g.ranks.map((r) => tiles[r]);
      let h;
      if (G.wide) {
        row.lab.style.top = (PAD + 4) + 'px';
        h = Math.max(hl + 4, flow(list, G.x0, G.tw, y + PAD) - y - PAD) + 2 * PAD;
      } else {
        row.lab.style.top = PAD + 'px';
        h = flow(list, 0, G.W, y + PAD + hl + 8) - y + PAD;
      }
      row.el.style.transform = 'translateY(' + y + 'px)';
      row.el.style.height = h + 'px';
      y += h;
    });
    board.style.height = y + 'px';
  }
  function layout() { (sorted ? layoutGroups : layoutPile)(); }
  // Apply a layout with no motion (a resize, a new neuron, reduced motion).
  function still(fn) {
    board.classList.add('still');
    fn();
    board.getBoundingClientRect();
    board.classList.remove('still');
  }

  // ---------- group rows ----------
  function buildRows() {
    rows.forEach((r) => r.el.remove());
    rows = groups.map((g, gi) => {
      const el = document.createElement('div');
      el.className = 'row';
      el.tabIndex = 0;
      el.setAttribute('role', 'button');
      el.setAttribute('aria-label', 'Group ' + (gi + 1) + ' of ' + groups.length + ', ' + plural(g.ranks.length, 'context', 'contexts') + ': ' + g.name);
      const lab = document.createElement('div');
      lab.className = 'lab';
      const n = document.createElement('span');
      n.className = 'n';
      n.textContent = String(g.ranks.length);
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = g.name;
      lab.append(n, name);
      el.appendChild(lab);
      el.addEventListener('click', () => { userActed(); choose(gi, -1); });
      el.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        userActed();
        choose(gi, -1);
      });
      board.insertBefore(el, pilehead.nextSibling);
      return { el, lab, g };
    });
  }

  // Move every tile to its place, group by group, and fade the rows in; returns when the last lands (ms).
  function moveTiles(perGroup, perTile) {
    clearTimeout(resetTimer);
    const calm = reduce.matches;
    let last = 0;
    rows.forEach((row, gi) => {
      row.g.ranks.forEach((r, k) => {
        const d = calm ? 0 : Math.round(gi * perGroup + k * perTile);
        last = Math.max(last, d);
        tiles[r].el.style.transitionDelay = d + 'ms';
        place(tiles[r]);
      });
      row.el.style.transitionDelay = (calm ? 0 : Math.round(gi * perGroup)) + 'ms';
      row.el.classList.add('in');
    });
    const end = calm ? 0 : last + MOVE_MS;
    resetTimer = setTimeout(() => {
      tiles.forEach((t) => { t.el.style.transitionDelay = '0ms'; });
      rows.forEach((r) => { r.el.style.transitionDelay = '0ms'; });
    }, end + 50);
    return end;
  }
  const spread = (n) => Math.min(90, 900 / Math.max(1, n));

  // ---------- views ----------
  function choose(gi, rank) {
    sel = gi;
    hlRank = rank;
    const inSel = new Set(groups[gi].ranks);
    rows.forEach((r, i) => {
      r.el.classList.toggle('sel', i === gi);
      r.el.setAttribute('aria-pressed', String(i === gi));
    });
    tiles.forEach((t) => {
      t.el.classList.toggle('on', inSel.has(t.r));
      t.el.classList.toggle('hl', t.r === rank);
    });
    drawPane();
  }

  function appendText(el, text) {
    text.split('\n').forEach((p, i) => {
      if (i > 0) {
        const nl = document.createElement('span');
        nl.className = 'nl';
        nl.textContent = '↵';
        el.appendChild(nl);
      }
      if (p) el.appendChild(document.createTextNode(p.replace(/\t/g, '  ')));
    });
  }
  function drawPane() {
    const pane = $('pane');
    pane.hidden = !sorted;
    if (!sorted) return;
    const g = groups[sel];
    $('paneTitle').textContent = 'The ' + plural(g.ranks.length, 'context', 'contexts') + ' in the group “' + g.name + '”';
    const box = $('ctxs');
    box.textContent = '';
    g.ranks.forEach((r) => {
      const [before, token, after, a] = N.ctx[r];
      const row = document.createElement('div');
      row.className = 'ctx' + (r === hlRank ? ' hl' : '');
      const b = document.createElement('span');
      b.className = 'b';
      const bi = document.createElement('span');
      appendText(bi, before);
      b.appendChild(bi);
      const t = document.createElement('span');
      t.className = 't';
      const pill = document.createElement('span');
      pill.className = 'pill';
      pill.textContent = pillText(token);
      pill.title = quoted(token);
      t.appendChild(pill);
      const af = document.createElement('span');
      af.className = 'a';
      appendText(af, after);
      const v = document.createElement('span');
      v.className = 'v';
      v.textContent = fmt(a);
      row.append(b, t, af, v);
      box.appendChild(row);
    });
  }

  function drawWho() {
    const who = $('who');
    who.textContent = '';
    const b = document.createElement('b');
    b.textContent = N.id;
    const what = N.kind === 'named'
      ? N.label[0].toUpperCase() + N.label.slice(1) + '.'
      : 'Drawn at random from the ' + D.n_study + ' neurons studied. The token it fired on most often in its 50: ' + quoted(topToken(N)) + '.';
    who.append(b, ' ' + describe(N.id) + ' ' + what);
  }

  function drawHero(fade) {
    const set = () => {
      if (!sorted) {
        $('num').textContent = String(D.k);
        $('unit').textContent = 'contexts';
        $('note').textContent = 'The neuron’s strongest at the end of training, before sorting.';
        return;
      }
      const n = groups.length;
      const other = N.sortings[1 - si].length;
      $('num').textContent = String(n);
      $('unit').textContent = n === 1 ? 'group' : 'groups';
      $('note').textContent = 'under the ' + ORD[si] + ' sorting. The ' + ORD[1 - si] + ' sorting gives ' + other + '.';
    };
    if (!fade || reduce.matches) { set(); return; }
    const fig = $('fig');
    fig.classList.add('fade');
    $('note').style.opacity = '0';
    setTimeout(() => {
      set();
      fig.classList.remove('fade');
      $('note').style.opacity = '';
    }, 200);
  }

  function drawDist() {
    const h = D.hist[si];
    const cur = sorted ? groups.length : -1;
    $('distTitle').textContent = 'How many groups each of the ' + D.n_study + ' neurons got under the ' + ORD[si] + ' sorting';
    barEls.forEach(({ cell, i, lab, ax }, k) => {
      const pct = (100 * h[k]) / HMAX;
      i.style.height = (h[k] ? Math.max(pct, 3) : 0) + '%';
      const isCur = k + 1 === cur;
      cell.classList.toggle('cur', isCur);
      ax.classList.toggle('cur', isCur);
      lab.textContent = isCur ? String(h[k]) : '';
      lab.style.bottom = (h[k] ? Math.max(pct, 3) : 0) + '%';
    });
  }

  function caption() {
    $('caption').textContent =
      'The model that sorted the contexts, ' + D.sorter + ', was told to make one group per job, but a second model asked about 129 ' +
      'of its groups said 109 were not about one thing, so a group is not a job. It sorted each of the ' + D.n_study +
      ' neurons twice, making ' + int(D.total[0]) + ' groups the first time and ' + int(D.total[1]) + ' the second, ' +
      'with the same count for ' + D.same + ' neurons.';
  }

  // ---------- transitions ----------
  function sortIn(animate) {
    sorted = true;
    pilehead.classList.add('off');
    buildRows();
    layoutGroups();
    sel = 0;
    hlRank = -1;
    if (animate) {
      const end = moveTiles(spread(groups.length), 12);
      clearTimeout(heroTimer);
      heroTimer = setTimeout(() => { drawHero(true); drawDist(); }, Math.max(0, end - 300));
      setTimeout(() => choose(sel, -1), end);
    } else {
      still(() => moveTiles(0, 0));
      drawHero(false);
      drawDist();
      choose(sel, -1);
    }
  }

  function setSorting(s) {
    if (s === si) return;
    userActed();
    root.querySelectorAll('.seg button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.s) === s)));
    const keep = groups[sel].ranks[0];
    si = s;
    groups = ordered(N, si);
    sel = groups.findIndex((g) => g.ranks.includes(keep));
    buildRows();
    layoutGroups();
    const end = moveTiles(spread(groups.length) * 0.6, 8);
    choose(sel, -1);
    clearTimeout(heroTimer);
    heroTimer = setTimeout(() => drawHero(true), Math.max(0, end - 300));
    drawDist();
  }

  function setNeuron(i) {
    userActed();
    N = D.neurons[i];
    select.value = String(i);
    groups = ordered(N, si);
    buildTiles();
    tiles.forEach((t) => t.el.classList.add('gone'));
    buildRows();
    still(() => { layoutGroups(); tiles.forEach(place); });
    sel = 0;
    choose(0, -1);
    drawWho();
    drawHero(false);
    drawDist();
    requestAnimationFrame(() => {
      rows.forEach((row, gi) => {
        row.g.ranks.forEach((r, k) => {
          tiles[r].el.style.transitionDelay = (reduce.matches ? 0 : gi * 40 + k * 6) + 'ms';
          tiles[r].el.classList.remove('gone');
        });
        row.el.classList.add('in');
      });
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => tiles.forEach((t) => { t.el.style.transitionDelay = '0ms'; }), 1400);
    });
  }

  // ---------- once, on view ----------
  let autoPending = !reduce.matches;
  let autoTimer = 0;
  const seen = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    seen.disconnect();
    if (!autoPending) return;
    autoTimer = setTimeout(() => {
      if (!autoPending) return;
      autoPending = false;
      sortIn(true);
    }, 700);
  }, { threshold: 0.35 });
  function userActed() {
    if (!autoPending && sorted) return;
    autoPending = false;
    clearTimeout(autoTimer);
    seen.disconnect();
    if (!sorted) sortIn(false);
  }

  // ---------- pointer: the tile under the pointer, and the group of a pressed tile ----------
  const tip = $('tip');
  board.addEventListener('pointermove', (e) => {
    const el = e.target.closest('.tile');
    if (!el || e.pointerType !== 'mouse') { tip.classList.remove('on'); return; }
    const r = Number(el.dataset.r);
    const [before, token, after, a] = N.ctx[r];
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = 'Rank ' + (r + 1) + ' of ' + D.k + ', activation ' + fmt(a);
    tip.append(b, '\n…' + flat(before.slice(-56)) + '[' + flat(token) + ']' + flat(after.slice(0, 24)) + '…');
    const rr = root.getBoundingClientRect();
    const tr = el.getBoundingClientRect();
    tip.classList.add('on');
    const w = tip.offsetWidth;
    const x = tr.left - rr.left + tr.width / 2;
    tip.style.left = Math.max(0, Math.min(rr.width - w, x - w / 2)) + 'px';
    tip.style.top = (tr.top - rr.top - tip.offsetHeight - 8) + 'px';
  });
  board.addEventListener('pointerleave', () => tip.classList.remove('on'));
  board.addEventListener('pointerdown', (e) => {
    const el = e.target.closest('.tile');
    if (!el) return;
    userActed();
    const r = Number(el.dataset.r);
    choose(groups.findIndex((g) => g.ranks.includes(r)), r);
  });
  bars.addEventListener('pointermove', (e) => {
    const cell = e.target.closest('.barcell');
    if (!cell) { tip.classList.remove('on'); return; }
    const k = Number(cell.dataset.k);
    tip.textContent = '';
    const b = document.createElement('b');
    b.textContent = D.hist[si][k - 1] + ' of ' + D.n_study + ' neurons';
    tip.append(b, '\ngot ' + plural(k, 'group', 'groups') + ' under the ' + ORD[si] + ' sorting');
    const rr = root.getBoundingClientRect();
    const cr = cell.getBoundingClientRect();
    tip.classList.add('on');
    const w = tip.offsetWidth;
    tip.style.left = Math.max(0, Math.min(rr.width - w, cr.left - rr.left + cr.width / 2 - w / 2)) + 'px';
    tip.style.top = (cr.top - rr.top - tip.offsetHeight - 8) + 'px';
  });
  bars.addEventListener('pointerleave', () => tip.classList.remove('on'));

  // ---------- controls ----------
  select.addEventListener('change', () => setNeuron(Number(select.value)));
  $('random').addEventListener('click', () => {
    const pool = randomIdx.filter((i) => D.neurons[i] !== N);
    setNeuron(pool[Math.floor(Math.random() * pool.length)]);
  });
  root.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => setSorting(Number(b.dataset.s))));

  let lastW = 0;
  new ResizeObserver(() => {
    const w = board.clientWidth;
    if (w === lastW) return;
    lastW = w;
    still(() => { layout(); tiles.forEach(place); });
  }).observe(board);
  if (document.fonts) document.fonts.ready.then(() => { measure(); still(() => { layout(); tiles.forEach(place); }); });

  // ---------- first render ----------
  buildTiles();
  drawWho();
  caption();
  if (autoPending) {
    still(() => { layoutPile(); tiles.forEach(place); });
    drawHero(false);
    drawDist();
    drawPane();
    seen.observe(board);
  } else {
    sortIn(false);
  }
})();
