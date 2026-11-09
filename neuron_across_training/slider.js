// "One neuron across training": the slider, shared by the walkthrough page and index.html.
// It registers its mount with site.js (load ../site.js first) under the id "nat"; site.js builds it inside
// the element with that id, passing window.NEURON_DATA, which data.js sets, and throws if data.js did not load.
// With no #nat on the page it does nothing. Arrow keys act while focus is inside the slider; a host that
// is only the slider sets data-keys="page" on #nat so they act anywhere on the page.
(() => {
  'use strict';

  function mount(root, D) {
    root.innerHTML = `
    <div class="pick">
      <label class="sr" for="nat-neuron">Neuron</label>
      <div class="select-wrap"><select id="nat-neuron"></select></div>
      <button class="btn" id="nat-random" type="button">Random neuron</button>
    </div>
    <p class="who" id="nat-who"></p>
    <div class="readout" aria-live="polite">
      <div class="tokenblock">
        <div class="tokenline">
          <span class="hero-pill" id="nat-heroPill"></span>
          <span class="hero-count"><b id="nat-heroCount"></b> of <span id="nat-heroOf"></span></span>
        </div>
        <p class="hero-note" id="nat-heroNote"></p>
      </div>
      <div class="stat">
        <div class="label">Training step</div>
        <div class="value" id="nat-stepVal"></div>
        <div class="pctbar" aria-hidden="true"><i id="nat-pctFill"></i></div>
        <div class="label" id="nat-pctVal" style="margin-top:4px"></div>
      </div>
      <div class="stat">
        <div class="label">Strongest activation</div>
        <div class="value" id="nat-maxVal"></div>
        <div class="label" style="margin-top:15px">of the 50 strongest contexts</div>
      </div>
    </div>
    <div class="history" role="group" aria-label="All 16 training points">
      <div class="hist-head">
        <h3>Top 20 contexts by the token they fire on</h3>
        <div class="legend">
          <span><i class="key own"></i><span id="nat-legOwn"></span></span>
          <span><i class="key other"></i>most common other token at that point</span>
          <span><i class="key rest"></i>any other token</span>
        </div>
      </div>
      <div class="scrub" id="nat-scrub">
        <div class="plot">
          <div class="band" id="nat-band1"></div>
          <div class="stacks" id="nat-stacks"></div>
          <span class="yl" style="top:2px">20</span>
          <span class="yl" style="bottom:-4px;top:auto;transform:none">0</span>
        </div>
        <h3 class="sub">Strongest activation <span>(log scale)</span></h3>
        <div class="plot act" id="nat-act">
          <div class="band" id="nat-band2"></div>
          <svg viewBox="0 0 160 100" preserveAspectRatio="none" aria-hidden="true"><path id="nat-line"></path></svg>
          <span class="dot-label" id="nat-dotLabel"></span>
        </div>
        <div class="plot steptrack" id="nat-track" role="slider" tabindex="0" aria-label="Training point"
             aria-valuemin="0" aria-valuemax="15">
          <div class="rail"></div>
          <div class="thumb" id="nat-thumb"></div>
        </div>
        <div class="plot ticklabels" id="nat-ticklabels"></div>
        <div class="tip" id="nat-tip"></div>
      </div>
      <div class="controls">
        <button class="btn" id="nat-play" type="button"></button>
        <span class="hint">Drag across the chart, or use the left and right arrow keys.</span>
      </div>
      <details>
        <summary>Show the 16 points as a table</summary>
        <div class="tablewrap"><table id="nat-table"></table></div>
      </details>
    </div>
    <div class="list">
      <div class="list-head mono"><h3>The 20 strongest contexts at this point</h3><span class="v">activation</span></div>
      <div class="rows" id="nat-rows"></div>
    </div>
    <p class="caption" id="nat-caption"></p>`;

    const $ = (id) => root.querySelector('#nat-' + id);

    const NS = D.steps.length;
    const LAST = NS - 1;
    const TOP = D.top;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)');
    const narrow = matchMedia('(max-width: 640px)');
    const PLAY_MS = 1600;

    // ---------- formatting ----------
    const int = (x) => Math.round(x).toLocaleString('en-US');
    const fmt = (x) => {
      const a = Math.abs(x);
      if (a >= 100) return int(x);
      if (a === 0) return '0';
      return String(Number(x.toPrecision(3)));
    };
    const fmtPct = (p) => (p === 0 ? '0' : p >= 100 ? '100' : String(Number(p.toPrecision(3))));
    const compact = (s) => {
      if (s < 1000) return String(s);
      if (s < 1e6) return String(Number((s / 1000).toPrecision(3))) + 'k';
      return String(Number((s / 1e6).toPrecision(3))) + 'M';
    };
    const quoted = (tok) => "'" + tok.replace(/\n/g, '\\n').replace(/\t/g, '\\t') + "'";
    const pillText = (tok) => {
      const t = tok.replace(/\n/g, '↵').replace(/\t/g, '⇥');
      return t.trim() === '' ? t.replace(/ /g, '␣') : t.trim();
    };
    const describePole = (p) => {
      const m = /^L(\d+)N(\d+)([+-])$/.exec(p);
      return 'Layer ' + Number(m[1]) + ', neuron ' + m[2] + ', ' + (m[3] === '+' ? 'positive' : 'negative') + ' activations.';
    };
    const GROUP_NOTE = {
      lithium: 'The lithium neuron.',
      late: 'Its main pattern at the final point appeared late in training, and a judge model could not link it to the neuron’s earlier top contexts.',
      random: 'Drawn at random from the 600 neurons studied.',
    };

    // ---------- per-neuron summary ----------
    function summarize(n) {
      let own = '', best = -1;
      n.steps.forEach((st) => st.counts.forEach(([tok, c]) => { if (c > best) { best = c; own = tok; } }));
      const ownCount = [], other = [], size = [], maxes = [];
      n.steps.forEach((st) => {
        const m = new Map(st.counts);
        ownCount.push(m.get(own) || 0);
        const o = st.counts.find(([tok]) => tok !== own);
        other.push(o ? { tok: o[0], c: o[1] } : { tok: '', c: 0 });
        size.push(st.top.length);
        maxes.push(st.max);
      });
      return { own, best, ownCount, other, size, maxes };
    }
    D.neurons.forEach((n) => { n.sum = summarize(n); });

    // ---------- static DOM: picker ----------
    const select = $('neuron');
    const groups = [
      ['lithium', 'The lithium neuron'],
      ['late', 'Final main pattern appeared late'],
      ['random', 'Drawn at random'],
    ];
    groups.forEach(([g, label]) => {
      const members = D.neurons.map((n, i) => [n, i]).filter(([n]) => n.group === g);
      if (!members.length) return;
      const og = document.createElement('optgroup');
      og.label = label + ' (' + members.length + ')';
      members.forEach(([n, i]) => {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = n.pole + '  ' + quoted(n.sum.own);
        og.appendChild(o);
      });
      select.appendChild(og);
    });
    const randomIdx = D.neurons.map((n, i) => (n.group === 'random' ? i : -1)).filter((i) => i >= 0);

    // ---------- static DOM: history ----------
    const colPct = (i) => ((i + 0.5) / NS) * 100;
    const stacksBox = $('stacks');
    const cells = [];
    for (let s = 0; s < NS; s++) {
      const col = document.createElement('div');
      col.className = 'stack';
      const cs = [];
      for (let k = 0; k < TOP; k++) {
        const c = document.createElement('i');
        c.className = 'cell';
        col.appendChild(c);
        cs.push(c);
      }
      stacksBox.appendChild(col);
      cells.push(cs);
    }
    const act = $('act');
    const dots = [];
    for (let s = 0; s < NS; s++) {
      const d = document.createElement('i');
      d.className = 'dot';
      d.style.left = colPct(s) + '%';
      act.appendChild(d);
      dots.push(d);
    }
    const track = $('track');
    const tickLabels = [];
    for (let s = 0; s < NS; s++) {
      const t = document.createElement('i');
      t.className = 'tick';
      t.style.left = colPct(s) + '%';
      track.insertBefore(t, $('thumb'));
      const l = document.createElement('span');
      l.className = 'tl' + (s === 0 ? ' first' : '') + (s === LAST ? ' last' : '');
      l.style.left = colPct(s) + '%';
      l.textContent = compact(D.steps[s].step);
      $('ticklabels').appendChild(l);
      tickLabels.push(l);
    }
    let gridEls = [];

    // ---------- state ----------
    let N = D.neurons[0];
    let cur = Math.max(0, D.steps.findIndex((s) => s.step === 60000));
    let rowEls = [];
    let shown = [];
    let dragging = false;
    let timer = 0;
    let swapToken = 0;

    // ---------- rows ----------
    const rowsBox = $('rows');
    const rh = () => parseFloat(getComputedStyle(root).getPropertyValue('--rh'));

    function appendText(el, text) {
      const parts = text.split('\n');
      parts.forEach((p, i) => {
        if (i > 0) {
          const nl = document.createElement('span');
          nl.className = 'nl';
          nl.textContent = '↵';
          el.appendChild(nl);
        }
        if (p) el.appendChild(document.createTextNode(p.replace(/\t/g, '  ')));
      });
    }

    function buildRows() {
      rowsBox.textContent = '';
      rowEls = N.ctx.map(([before, token, after]) => {
        const row = document.createElement('div');
        row.className = 'ctx';
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
        const a = document.createElement('span');
        a.className = 'a';
        appendText(a, after);
        const v = document.createElement('span');
        v.className = 'v';
        row.append(b, t, a, v);
        row._pill = pill;
        row._v = v;
        row._tok = token;
        row._y = 0;
        rowsBox.appendChild(row);
        return row;
      });
      shown = rowEls.map(() => false);
    }

    // mode: 'step' (direction-aware slide), 'reveal' (rise in), 'jump' (no motion)
    function placeRows(prev, mode) {
      const st = N.steps[cur];
      const H = rh();
      const dir = cur >= prev ? 1 : -1;
      const calm = reduce.matches || mode === 'jump';
      const shift = calm ? 0 : 36;
      const stagger = calm || dragging ? 0 : mode === 'reveal' ? 26 : 12;
      const rank = new Map(st.top.map(([ci], r) => [ci, r]));
      const own = N.sum.own;
      const other = N.sum.other[cur].tok;
      const entering = [];
      rowEls.forEach((el, ci) => {
        const r = rank.get(ci);
        if (r === undefined) {
          if (shown[ci]) {
            shown[ci] = false;
            el.style.transitionDelay = '0ms';
            el.style.transform = 'translate3d(' + (-dir * shift) + 'px,' + el._y + 'px,0)';
            el.style.opacity = '0';
            el.style.pointerEvents = 'none';
          }
          return;
        }
        el._v.textContent = fmt(st.top[r][1]);
        el._pill.className = 'pill' + (el._tok === own ? ' own' : el._tok === other ? ' other' : '');
        const y = r * H;
        if (!shown[ci]) {
          entering.push([el, y, r]);
          el.style.transition = 'none';
          el.style.transform = mode === 'reveal'
            ? 'translate3d(0,' + (y + (calm ? 0 : 14)) + 'px,0)'
            : 'translate3d(' + (dir * shift) + 'px,' + y + 'px,0)';
          el.style.opacity = '0';
          shown[ci] = true;
        } else {
          el.style.transitionDelay = (r * stagger) + 'ms';
          el.style.transform = 'translate3d(0,' + y + 'px,0)';
        }
        el._y = y;
        el.style.pointerEvents = '';
      });
      if (entering.length) {
        rowsBox.getBoundingClientRect();
        entering.forEach(([el, y, r]) => {
          el.style.transition = '';
          el.style.transitionDelay = (r * stagger) + 'ms';
          el.style.transform = 'translate3d(0,' + y + 'px,0)';
          el.style.opacity = '1';
        });
      }
      rowsBox.style.height = (Math.max(st.top.length, 1) * H) + 'px';
    }

    // ---------- per-neuron views ----------
    function drawNeuron() {
      const S = N.sum;
      const who = $('who');
      who.textContent = '';
      const idb = document.createElement('b');
      idb.textContent = N.pole;
      who.append(idb, ' ' + describePole(N.pole) + ' ' + GROUP_NOTE[N.group] + ' The token it fired on most: ' +
        quoted(S.own) + ', ' + S.best + ' of ' + TOP + ' at its peak.');
      $('legOwn').textContent = quoted(S.own);

      // stacks
      for (let s = 0; s < NS; s++) {
        const o = S.ownCount[s];
        const oc = S.other[s].c;
        for (let k = 0; k < TOP; k++) {
          const cls = k < o ? 'own' : k < o + oc ? 'other' : k < S.size[s] ? '' : 'empty';
          cells[s][k].className = 'cell' + (cls ? ' ' + cls : '');
          cells[s][k].style.transitionDelay = reduce.matches ? '0ms' : (s * 18 + k * 6) + 'ms';
        }
      }

      // strongest activation, log scale, own axis
      const logs = S.maxes.map((v) => Math.log10(Math.max(v, 1e-6)));
      const lo = Math.floor(Math.min(...logs));
      let hi = Math.ceil(Math.max(...logs));
      if (hi === lo) hi = lo + 1;
      const ypct = (l) => 8 + ((hi - l) / (hi - lo)) * 84;
      gridEls.forEach((e) => e.remove());
      gridEls = [];
      const every = hi - lo > 4 ? 2 : 1;
      for (let k = lo; k <= hi; k++) {
        const g = document.createElement('i');
        g.className = 'grid';
        g.style.top = ypct(k) + '%';
        act.insertBefore(g, act.firstChild);
        gridEls.push(g);
        if ((k - lo) % every) continue;
        const lab = document.createElement('span');
        lab.className = 'yl';
        lab.style.top = ypct(k) + '%';
        lab.textContent = k < 0 ? (10 ** k).toFixed(-k) : int(10 ** k);
        act.appendChild(lab);
        gridEls.push(lab);
      }
      const pts = logs.map((l, s) => [(s + 0.5) * 10, ypct(l)]);
      const d = 'M' + pts.map(([x, y]) => x.toFixed(2) + ' ' + y.toFixed(2)).join(' L');
      const line = $('line');
      line.setAttribute('d', d);
      line.style.d = 'path("' + d + '")';
      dots.forEach((el, s) => { el.style.top = pts[s][1] + '%'; });
      N._ypct = pts.map((p) => p[1]);

      // table
      const table = $('table');
      table.textContent = '';
      const head = table.insertRow();
      ['Point', 'Step', '% of training', quoted(S.own) + ' in top 20', 'Most common other token', 'Strongest activation']
        .forEach((h) => { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); });
      D.steps.forEach((st, s) => {
        const tr = table.insertRow();
        [st.t, int(st.step), fmtPct(st.pct), S.ownCount[s] + ' of ' + S.size[s],
          S.other[s].c ? quoted(S.other[s].tok) + ' ' + S.other[s].c : '-', fmt(S.maxes[s])]
          .forEach((v) => { tr.insertCell().textContent = v; });
      });
    }

    // ---------- per-step views ----------
    function drawStep() {
      const S = N.sum;
      const st = D.steps[cur];
      const c = S.ownCount[cur];
      const pill = $('heroPill');
      pill.textContent = pillText(S.own);
      pill.classList.toggle('gone', c === 0);
      $('heroCount').textContent = String(c);
      $('heroOf').textContent = String(S.size[cur]);
      const note = $('heroNote');
      note.textContent = '';
      if (c === 0) {
        let last = -1;
        for (let s = cur - 1; s >= 0; s--) if (S.ownCount[s] > 0) { last = s; break; }
        note.textContent = last >= 0
          ? quoted(S.own) + ' lost the top 20. It last held ' + S.ownCount[last] + ' of ' + S.size[last] +
            ', at step ' + int(D.steps[last].step) + '.'
          : quoted(S.own) + ' is not in the top 20 at this point.';
      } else {
        note.textContent = 'of the top 20 contexts fire on ' + quoted(S.own) + '.';
      }
      const o = S.other[cur];
      if (o.c > c) {
        const k = document.createElement('i');
        k.className = 'key other';
        note.append(' Most common here: ', k, quoted(o.tok) + ', ' + o.c + ' of ' + S.size[cur] + '.');
      }
      $('stepVal').textContent = int(st.step);
      $('pctVal').textContent = fmtPct(st.pct) + '% of training';
      $('pctFill').style.width = st.pct + '%';
      $('maxVal').textContent = fmt(S.maxes[cur]);

      const left = (cur / NS) * 100 + '%';
      $('band1').style.left = left;
      $('band2').style.left = left;
      if (!dragging) $('thumb').style.left = colPct(cur) + '%';
      dots.forEach((d, s) => d.classList.toggle('cur', s === cur));
      tickLabels.forEach((l, s) => l.classList.toggle('cur', s === cur));
      const dl = $('dotLabel');
      dl.textContent = fmt(S.maxes[cur]);
      dl.style.left = colPct(cur) + '%';
      dl.style.top = N._ypct[cur] + '%';
      dl.classList.toggle('below', N._ypct[cur] < 30);

      track.setAttribute('aria-valuenow', String(cur));
      track.setAttribute('aria-valuetext', 'step ' + int(st.step) + ', ' + fmtPct(st.pct) + '% of training, ' +
        quoted(S.own) + ' ' + c + ' of ' + S.size[cur] + ', strongest activation ' + fmt(S.maxes[cur]));
    }

    function setStep(s, mode) {
      s = Math.max(0, Math.min(LAST, s));
      if (s === cur && mode !== 'force') return;
      const prev = cur;
      cur = s;
      drawStep();
      placeRows(prev, mode === 'force' ? 'jump' : 'step');
    }

    function setNeuron(i) {
      const token = ++swapToken;
      N = D.neurons[i];
      select.value = String(i);
      drawNeuron();
      drawStep();
      rowsBox.classList.add('out');
      setTimeout(() => {
        if (token !== swapToken) return;
        buildRows();
        rowsBox.classList.remove('out');
        placeRows(cur, 'reveal');
      }, reduce.matches ? 0 : 170);
      caption();
    }

    // ---------- play ----------
    const playBtn = $('play');
    const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.6 6z"/></svg>';
    const PAUSE_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';
    function setPlayLabel() {
      playBtn.innerHTML = timer ? PAUSE_ICON : PLAY_ICON;
      playBtn.append(timer ? 'Pause' : 'Play through training');
    }
    function stop() {
      if (timer) clearInterval(timer);
      timer = 0;
      root.dataset.playing = '';
      setPlayLabel();
    }
    function play(by) {
      if (cur === LAST) setStep(0);
      timer = setInterval(() => {
        if (cur >= LAST) { stop(); return; }
        setStep(cur + 1);
        if (cur >= LAST) stop();
      }, PLAY_MS);
      root.dataset.playing = by;
      setPlayLabel();
    }
    playBtn.addEventListener('click', () => (timer ? stop() : play('button')));
    setPlayLabel();

    // ---------- autoplay: once, from the first step, when the chart is first in view ----------
    // Skipped under prefers-reduced-motion. Any pointer press inside the slider, or any key press while it
    // runs, stops it (the play button's own press goes to its click, which pauses); a press or focus inside
    // the slider before it starts cancels it.
    let autoPending = !reduce.matches;
    const seen = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      seen.disconnect();
      if (!autoPending) return;
      autoPending = false;
      setStep(0);
      play('auto');
    }, { threshold: 0.6 });
    if (autoPending) seen.observe($('scrub'));
    const userActed = () => {
      autoPending = false;
      seen.disconnect();
    };
    root.addEventListener('pointerdown', (e) => {
      userActed();
      if (root.dataset.playing === 'auto' && !playBtn.contains(e.target)) stop();
    });
    root.addEventListener('focusin', userActed);
    document.addEventListener('keydown', (e) => {
      if (root.dataset.playing === 'auto' && e.target !== playBtn) stop();
    });

    // ---------- scrub ----------
    const scrub = $('scrub');
    const tip = $('tip');
    const frac = (e) => {
      const r = track.getBoundingClientRect();
      return Math.max(0, Math.min(0.99999, (e.clientX - r.left) / r.width));
    };
    scrub.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      stop();
      scrub.classList.add('dragging');
      scrub.setPointerCapture(e.pointerId);
      tip.classList.remove('on');
      track.focus({ preventScroll: true });
      dragTo(e);
    });
    function dragTo(e) {
      const f = frac(e);
      $('thumb').style.left = Math.max(0.5 / NS, Math.min(1 - 0.5 / NS, f)) * 100 + '%';
      setStep(Math.floor(f * NS));
    }
    scrub.addEventListener('pointermove', (e) => {
      if (dragging) { dragTo(e); return; }
      if (e.pointerType !== 'mouse') return;
      showTip(Math.floor(frac(e) * NS));
    });
    const endDrag = () => {
      if (!dragging) return;
      dragging = false;
      scrub.classList.remove('dragging');
      $('thumb').style.left = colPct(cur) + '%';
    };
    scrub.addEventListener('pointerup', endDrag);
    scrub.addEventListener('pointercancel', endDrag);
    scrub.addEventListener('pointerleave', () => tip.classList.remove('on'));

    function showTip(s) {
      const S = N.sum;
      const st = D.steps[s];
      tip.textContent = '';
      const b = document.createElement('b');
      b.textContent = 'Step ' + int(st.step) + '  (' + fmtPct(st.pct) + '%)';
      const lines = [quoted(S.own) + '  ' + S.ownCount[s] + ' of ' + S.size[s]];
      if (S.other[s].c) lines.push(quoted(S.other[s].tok) + '  ' + S.other[s].c + ' of ' + S.size[s]);
      lines.push('strongest activation  ' + fmt(S.maxes[s]));
      tip.append(b, '\n' + lines.join('\n'));
      const sr = scrub.getBoundingClientRect();
      const tr = track.getBoundingClientRect();
      const x = tr.left - sr.left + tr.width * (s + 0.5) / NS;
      const w = tip.offsetWidth;
      tip.style.left = Math.max(w / 2, Math.min(sr.width - w / 2, x)) + 'px';
      tip.style.top = '0px';
      tip.classList.add('on');
    }

    // ---------- keys ----------
    (root.dataset.keys === 'page' ? document : root).addEventListener('keydown', (e) => {
      if (e.target === select || e.altKey || e.ctrlKey || e.metaKey) return;
      let s = -1;
      const onTrack = e.target === track;
      if (e.key === 'ArrowRight' || (onTrack && e.key === 'ArrowUp')) s = cur + 1;
      else if (e.key === 'ArrowLeft' || (onTrack && e.key === 'ArrowDown')) s = cur - 1;
      else if (e.key === 'Home') s = 0;
      else if (e.key === 'End') s = LAST;
      else if (e.key === ' ' && onTrack) { e.preventDefault(); timer ? stop() : play('key'); return; }
      else return;
      e.preventDefault();
      stop();
      setStep(s);
    });

    // ---------- neuron controls ----------
    select.addEventListener('change', () => { stop(); setNeuron(Number(select.value)); });
    $('random').addEventListener('click', () => {
      stop();
      const pool = randomIdx.filter((i) => D.neurons[i] !== N);
      setNeuron(pool[Math.floor(Math.random() * pool.length)]);
    });
    narrow.addEventListener('change', () => setStep(cur, 'force'));

    // ---------- caption ----------
    function caption() {
      const li = D.neurons.find((n) => n.group === 'lithium');
      const S = li.sum;
      const at = (step) => D.steps.findIndex((s) => s.step === step);
      const i60 = at(60000), i1m = at(1000000);
      const peak = S.maxes.indexOf(Math.max(...S.maxes));
      const end = S.ownCount[LAST] === 0
        ? 'had lost the top 20 by the final step'
        : 'held ' + S.ownCount[LAST] + ' of ' + S.size[LAST] + ' at the final step';
      $('caption').textContent =
        'Each view is one MLP neuron of ' + D.model + ' at ' + NS + ' saved training points; at each point ' +
        Number((D.n_tokens_scored / 1e6).toPrecision(3)) + ' million tokens were scored, the neuron’s ' + D.k +
        ' strongest contexts were kept, and the page shows the top ' + TOP + '. ' +
        'The lithium neuron (' + li.pole + ') fired on ' + quoted(S.own) + ' in ' + S.ownCount[i60] + ' of its top ' +
        S.size[i60] + ' contexts at step 60,000, still held ' + S.ownCount[i1m] + ' of ' + S.size[i1m] +
        ' at step 1,000,000, and ' + end + ', while its strongest activation fell from ' + fmt(S.maxes[peak]) +
        ' at step ' + int(D.steps[peak].step) + ' to ' + fmt(S.maxes[LAST]) + '.';
    }

    setNeuron(0);
  }

  WALK.register('nat', { data: 'NEURON_DATA', mount });
})();
