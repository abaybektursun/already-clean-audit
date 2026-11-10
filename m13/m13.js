// m13: the checks each study ran before its result could count. Registers the mount "m13" with site.js; reads
// window.M13_DATA (m13/data.js). The studies are rows, the checks are columns, and each cell is the status the
// study's files record: passed, failed, not applicable or not recorded. Hover, tap or keyboard focus on a cell names
// the file that records it; choosing a cell (click, tap, Enter) lists its files as links at the data's commit.
(() => {
  'use strict';

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const STATUS = ['passed', 'failed', 'not applicable', 'not recorded'];
  const CLS = { passed: 'pass', failed: 'fail', 'not applicable': 'na', 'not recorded': 'nr' };
  const wrapPath = (p) => p.replace(/\//g, '/​'); // lets a long path wrap inside the value box

  function mount(root, D) {
    const C = D.checks;
    const SD = D.studies;
    const NC = C.length;
    const NS = SD.length;

    // ---------- the data must hold together before anything is drawn ----------
    const counts = {};
    C.forEach((c) => { counts[c.id] = Object.fromEntries(STATUS.map((s) => [s, 0])); });
    const notRecorded = [];
    SD.forEach((s) => C.forEach((c) => {
      const cell = s[c.id];
      if (!cell) throw new Error('m13: ' + s.study + ' has no cell for ' + c.id);
      if (!STATUS.includes(cell.status)) throw new Error('m13: ' + s.study + ' ' + c.id + ' has the status ' + cell.status);
      if (!cell.source) throw new Error('m13: ' + s.study + ' ' + c.id + ' names no file');
      counts[c.id][cell.status] += 1;
      if (cell.status === 'not recorded') notRecorded.push(s.study + ': ' + c.id);
    }));
    C.forEach((c) => STATUS.forEach((st) => {
      if (counts[c.id][st] !== D.status_counts[c.id][st]) throw new Error('m13: ' + c.id + ' ' + st + ' counts ' + counts[c.id][st] + ', the data says ' + D.status_counts[c.id][st]);
    }));
    if (notRecorded.join('|') !== D.not_recorded.join('|')) throw new Error('m13: the cells not recorded differ from the data’s list');
    const total = Object.fromEntries(STATUS.map((st) => [st, C.reduce((n, c) => n + counts[c.id][st], 0)]));
    const files = (cell) => cell.source.split(' ; ').map((p) => p.trim()).filter(Boolean);
    const link = (p) => WALK.REPO + '/blob/' + D.meta.commit + '/' + p;

    root.replaceChildren();
    const wrap = el('div', 'm13-wrap');
    root.appendChild(wrap);

    // ---------- legend ----------
    wrap.appendChild(el('p', 'm13-note', NS + ' studies as rows, ' + NC + ' checks as columns: ' + NS * NC + ' cells. Each check names one thing a study does; each cell shows the status that the study’s files record for it.'));
    const legend = el('p', 'm13-legend');
    STATUS.forEach((st) => {
      const s = el('span');
      s.append(el('i', 'm13-glyph ' + CLS[st]), st + ' (' + total[st] + ')');
      legend.appendChild(s);
    });
    wrap.appendChild(legend);

    // ---------- the checks, numbered ----------
    const list = el('ol', 'm13-checks');
    C.forEach((c) => {
      const li = el('li');
      const n = counts[c.id];
      li.append(el('span', 'm13-cname', c.name), el('span', 'm13-ccount', STATUS.filter((st) => n[st]).map((st) => st + ' ' + n[st]).join(', ')));
      list.appendChild(li);
    });
    wrap.appendChild(list);

    // ---------- the grid ----------
    const scroll = el('div', 'm13-scroll');
    wrap.appendChild(scroll);
    const grid = el('div', 'm13-grid');
    grid.setAttribute('role', 'group');
    grid.setAttribute('aria-label', 'Checks by study; arrow keys move between cells, Enter lists the files of a cell');
    scroll.appendChild(grid);
    grid.appendChild(el('span', 'm13-corner'));
    C.forEach((c, j) => {
      const h = el('span', 'm13-colh', String(j + 1));
      h.setAttribute('aria-hidden', 'true');
      h.title = c.name;
      grid.appendChild(h);
    });
    const cells = [];
    SD.forEach((s, i) => {
      grid.appendChild(el('span', 'm13-rowh', s.name));
      cells.push(C.map((c, j) => {
        const cell = s[c.id];
        const b = el('span', 'm13-cell');
        b.appendChild(el('i', 'm13-glyph ' + CLS[cell.status]));
        b.setAttribute('role', 'button');
        b.setAttribute('aria-label', s.name + '; ' + (j + 1) + ', ' + c.name + ': ' + cell.status);
        b.setAttribute('aria-describedby', 'walk-value-box');
        b.tabIndex = -1;
        b.dataset.i = String(i);
        b.dataset.j = String(j);
        grid.appendChild(b);
        return b;
      }));
    });
    wrap.appendChild(el('p', 'm13-note', 'Hover, tap or tab into the grid for a cell’s file; arrow keys move between cells; click, tap or press Enter to list its files below.'));

    // ---------- the chosen cell ----------
    const panel = el('div', 'm13-panel');
    panel.setAttribute('aria-live', 'polite');
    wrap.appendChild(panel);

    const recount = (s, c) => (c.id === 'failing_inputs' && D.failing_input_recount[s.study]) || null;
    const boxFor = (i, j) => {
      const s = SD[i];
      const c = C[j];
      const cell = s[c.id];
      const rows = [[s.name, ''], ['status: ' + cell.status, '']];
      const rc = recount(s, c);
      if (rc) rows.push(['inputs that must fail', String(rc.records)], ['failed as required', String(rc.failed_as_required)]);
      const where = cell.status === 'not recorded' ? 'looked for in: ' : 'recorded in: ';
      files(cell).forEach((p) => rows.push([where + wrapPath(p), '']));
      return { title: (j + 1) + '. ' + c.name, rows };
    };

    let at = [0, 0];
    const setRoving = (i, j) => {
      cells[at[0]][at[1]].tabIndex = -1;
      at = [i, j];
      cells[i][j].tabIndex = 0;
    };
    function choose(i, j) {
      cells.forEach((r) => r.forEach((b) => b.classList.remove('cur')));
      cells[i][j].classList.add('cur');
      const s = SD[i];
      const c = C[j];
      const cell = s[c.id];
      panel.replaceChildren();
      const head = el('p', 'm13-phead');
      head.append(el('i', 'm13-glyph ' + CLS[cell.status]), ' ', el('b', '', cell.status), ': ' + (j + 1) + '. ' + c.name);
      panel.append(head, el('p', 'm13-pstudy', s.name));
      const rc = recount(s, c);
      if (rc) panel.appendChild(el('p', 'm13-pline', 'Inputs that must fail: ' + rc.records + '; failed as required: ' + rc.failed_as_required + '.'));
      const fl = files(cell);
      const where = cell.status === 'not recorded' ? 'Looked for in' : 'Recorded in';
      panel.appendChild(el('p', 'm13-plabel', fl.length === 1 ? where : where + ' these ' + fl.length + ' files'));
      const ul = el('ul', 'm13-files');
      fl.forEach((p) => {
        const li = el('li');
        const a = el('a');
        a.href = link(p);
        a.appendChild(el('code', '', p));
        li.appendChild(a);
        ul.appendChild(li);
      });
      panel.appendChild(ul);
      if (cell.key && cell.key !== '—') {
        const k = el('p', 'm13-pline');
        k.append(document.createTextNode('Where in the file: '), el('code', '', cell.key));
        panel.appendChild(k);
      }
    }

    const cellOf = (e) => e.target.closest('.m13-cell');
    grid.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const b = cellOf(e);
      if (!b) { WALK.valueBox.hide(); return; }
      WALK.valueBox.show(boxFor(Number(b.dataset.i), Number(b.dataset.j)), e.clientX, e.clientY);
    });
    grid.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') WALK.valueBox.hide(); });
    grid.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      const b = cellOf(e);
      if (b) WALK.valueBox.showFor(b, boxFor(Number(b.dataset.i), Number(b.dataset.j)));
    });
    grid.addEventListener('click', (e) => {
      const b = cellOf(e);
      if (!b) return;
      const i = Number(b.dataset.i);
      const j = Number(b.dataset.j);
      setRoving(i, j);
      choose(i, j);
    });
    grid.addEventListener('focusin', (e) => {
      const b = cellOf(e);
      if (b) WALK.valueBox.showFor(b, boxFor(Number(b.dataset.i), Number(b.dataset.j)));
    });
    grid.addEventListener('focusout', () => WALK.valueBox.hide());
    grid.addEventListener('keydown', (e) => {
      const b = cellOf(e);
      if (!b) return;
      let [i, j] = [Number(b.dataset.i), Number(b.dataset.j)];
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(i, j); return; }
      const mv = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[e.key];
      if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = NC - 1;
      else if (mv) { i = Math.max(0, Math.min(NS - 1, i + mv[0])); j = Math.max(0, Math.min(NC - 1, j + mv[1])); }
      else return;
      e.preventDefault();
      setRoving(i, j);
      cells[i][j].focus();
    });

    setRoving(0, 0);
    choose(0, 0);
  }

  WALK.register('m13', { data: 'M13_DATA', mount });
})();
