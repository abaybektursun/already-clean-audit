// site.js: the walkthrough's shared helpers (BUILD.md §1). No dependencies, no network.
//
//   WALK.register(id, {data, mount})  a picture registers its mount: data names the global its data.js sets
//                                      (for example 'NEURON_DATA'); mount(root, D) builds inside #id.
//   WALK.mountAll()                    mounts every registered picture whose element is on the page. It runs by
//                                      itself once the document has loaded; a second call mounts nothing twice.
//   WALK.fmt                           number formatting: int, fixed, percent, interval.
//   WALK.valueBox, WALK.hover          the one value box: shown near the pointer, on tap, or on keyboard focus.
//   WALK.dataLink(a, meta)             points a "data" link at meta.sources in the public repository at meta.commit.
//   data-load="a.css b/data.js b/b.js" on a mount: the article page loads these when the mount nears the viewport.
//
// A page loads site.js first, then each picture's data.js and script. Each mount runs as its own task, so a picture
// that throws is reported in the console and does not stop the others from mounting.
(() => {
  'use strict';

  const REPO = 'https://github.com/abaybektursun/already-clean-audit';

  // ---------- mount registry ----------
  const registry = new Map();

  function register(id, spec) {
    if (registry.has(id)) throw new Error('WALK.register: "' + id + '" is registered twice');
    if (typeof spec.mount !== 'function') throw new Error('WALK.register: "' + id + '" has no mount function');
    registry.set(id, { spec, mounted: false });
  }

  function mountOne(id, entry) {
    const root = document.getElementById(id);
    const name = entry.spec.data;
    const D = name ? window[name] : undefined;
    if (name && !D) {
      const p = document.createElement('p');
      p.className = 'walk-missing';
      p.textContent = 'The data file for this picture did not load (' + (name ? name.toLowerCase().replace(/_data$/, '') + '/data.js' : 'data.js') + ').';
      root.replaceChildren(p);
      throw new Error('#' + id + ': window.' + name + ' is not set; data.js did not load');
    }
    entry.spec.mount(root, D);
    const figure = root.closest('.figure');
    if (figure) figure.querySelectorAll('.data-link').forEach((a) => dataLink(a, D && D.meta));
  }

  function mountAll() {
    registry.forEach((entry, id) => {
      if (entry.mounted || !document.getElementById(id)) return;
      entry.mounted = true;
      queueMicrotask(() => mountOne(id, entry));
    });
  }

  // ---------- numbers ----------
  // Thousands separators, fixed decimals, a true minus sign; an interval is drawn as "a [lo, hi]".
  // A value that is not a finite number throws instead of printing NaN.
  const MINUS = '\u2212';
  function fixed(x, d) {
    if (!Number.isFinite(x)) throw new TypeError('WALK.fmt: not a finite number: ' + x);
    const s = Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    return x < 0 && /[1-9]/.test(s) ? MINUS + s : s;
  }
  const fmt = {
    int: (x) => fixed(x, 0),
    fixed,
    percent: (fraction, d) => fixed(100 * fraction, d) + '%',
    interval: (v, lo, hi, d) => fixed(v, d) + ' [' + fixed(lo, d) + ', ' + fixed(hi, d) + ']',
  };

  // ---------- the value box ----------
  // Content is a string, or {title, rows: [[label, value, key]]}, where key names a colour token
  // ('claim', 'control', 'ink', 'muted') drawn as a small square before the label.
  let box = null;
  let anchor = null; // the element the box is shown for; null while it follows the mouse
  const BOX_ID = 'walk-value-box';

  function fill(content) {
    if (!box) {
      box = document.createElement('div');
      box.className = 'value-box';
      box.id = BOX_ID;
      box.setAttribute('role', 'tooltip');
      box.hidden = true;
      document.body.appendChild(box);
    }
    box.replaceChildren();
    if (typeof content === 'string') {
      box.textContent = content;
      return box;
    }
    if (content.title) {
      const t = document.createElement('span');
      t.className = 'vb-title';
      t.textContent = content.title;
      box.appendChild(t);
    }
    (content.rows || []).forEach(([label, value, key]) => {
      const row = document.createElement('span');
      row.className = 'vb-row';
      const l = document.createElement('span');
      if (key) {
        const k = document.createElement('i');
        k.className = 'vb-key';
        k.style.background = 'var(--' + key + ')';
        l.appendChild(k);
      }
      l.append(label);
      const v = document.createElement('span');
      v.className = 'vb-value';
      v.textContent = value;
      row.append(l, v);
      box.appendChild(row);
    });
    return box;
  }

  // x, y: the anchor in viewport coordinates. The box sits centred above it, inside the viewport,
  // and flips below when there is no room above.
  function place(b, x, y) {
    b.hidden = false;
    const w = b.offsetWidth;
    const h = b.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const left = Math.max(8, Math.min(vw - w - 8, x - w / 2));
    const top = y - h - 12 < 8 ? y + 20 : y - h - 12;
    b.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
  }

  const placeFor = (b, el) => {
    const r = el.getBoundingClientRect();
    place(b, r.left + r.width / 2, r.top);
  };
  const valueBox = {
    show(content, x, y) { anchor = null; place(fill(content), x, y); },
    showFor(el, content) { anchor = el; placeFor(fill(content), el); },
    hide() { anchor = null; if (box) box.hidden = true; },
  };

  // Wire one mark: mouse hover follows the pointer, a tap or keyboard focus shows the box above the mark.
  // content is the content or a function that returns it. For a dense chart (hundreds of marks), attach one
  // listener to the plot, find the mark under the pointer, and call valueBox.show; give the plot arrow keys.
  function hover(el, content) {
    const get = typeof content === 'function' ? content : () => content;
    if (!el.matches('a[href], button, input, select, [tabindex]')) el.tabIndex = 0;
    el.setAttribute('aria-describedby', BOX_ID);
    el.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') valueBox.show(get(), e.clientX, e.clientY); });
    el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') valueBox.hide(); });
    el.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') valueBox.showFor(el, get()); });
    el.addEventListener('focus', () => valueBox.showFor(el, get()));
    el.addEventListener('blur', () => valueBox.hide());
  }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') valueBox.hide(); });
  document.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'mouse' && !e.target.closest('[aria-describedby="' + BOX_ID + '"]')) valueBox.hide();
  });
  // A scroll moves a box shown for an element along with it (keyboard focus scrolls the page) and hides one
  // that follows the mouse.
  addEventListener('scroll', () => {
    if (!box || box.hidden) return;
    if (anchor) placeFor(box, anchor);
    else valueBox.hide();
  }, { passive: true });

  // ---------- the data link ----------
  // meta = {commit, sources: [{path, sha256}], link?}. A source's path is repository-relative, an s3:// URI, or an
  // absolute local path for a file that exists only on the laptop. The link opens meta.link when it is set, else the
  // first repository-relative source. The commit is the public repository's.
  function dataLink(a, meta) {
    if (!meta || !/^[0-9a-f]{7,40}$/.test(meta.commit || '') || !Array.isArray(meta.sources) || !meta.sources.length) {
      throw new Error('data link: the data object needs meta.commit (a git sha) and a non-empty meta.sources');
    }
    const inRepo = meta.sources.find((s) => !/^(s3:\/\/|\/)/.test(s.path));
    const path = meta.link || (inRepo && inRepo.path);
    if (!path) throw new Error('data link: no source is in the repository and meta.link is not set');
    a.href = REPO + '/blob/' + meta.commit + '/' + path;
    a.title = path + ' at ' + meta.commit.slice(0, 9) + ' (' + meta.sources.length +
      (meta.sources.length === 1 ? ' input file' : ' input files') + ', each listed with its checksum in data.js)';
    if (!a.textContent.trim()) a.textContent = 'data';
  }

  // ---------- theme ----------
  // A button with data-theme-toggle switches between light and dark by setting data-theme on <html>.
  // Its label names the theme it switches to. The choice lasts until the page is left.
  const darkQuery = matchMedia('(prefers-color-scheme: dark)');
  const currentTheme = () => document.documentElement.dataset.theme || (darkQuery.matches ? 'dark' : 'light');

  function bindThemeToggles() {
    document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
      const label = () => {
        const next = currentTheme() === 'dark' ? 'light' : 'dark';
        btn.textContent = next === 'dark' ? 'Dark theme' : 'Light theme';
        btn.setAttribute('aria-label', 'Switch to the ' + next + ' theme');
      };
      btn.addEventListener('click', () => {
        document.documentElement.dataset.theme = currentTheme() === 'dark' ? 'light' : 'dark';
        label();
      });
      darkQuery.addEventListener('change', label);
      label();
    });
  }

  // ---------- the chapter list ----------
  // Marks the link of the chapter whose top part is in the upper third of the window with aria-current.
  function bindChapterList() {
    const links = [...document.querySelectorAll('.toc a[href^="#"]')];
    if (!links.length) return;
    const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const seen = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.removeAttribute('aria-current'));
        byId.get(e.target.id).setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '0px 0px -67% 0px' });
    byId.forEach((a, id) => {
      const el = document.getElementById(id);
      if (!el) throw new Error('chapter list: no element with id "' + id + '"');
      seen.observe(el);
    });
  }

  // ---------- loading a picture when it nears the viewport ----------
  // On the article page a mount names its files: <div id="m01" data-load="m01/m01.css m01/data.js m01/m01.js">.
  // They load in that order when the mount comes within one screen of the viewport (BUILD.md §2), then every
  // registered picture on the page is mounted. A script that builds itself as it runs (the first edition's) mounts
  // as it loads. A file named by two mounts loads once. A file that fails to load writes its path into the mount and
  // throws. A standalone page loads its own files with script tags and never uses this.
  const loading = new Map();
  function loadFile(path) {
    if (!loading.has(path)) {
      loading.set(path, new Promise((resolve, reject) => {
        const css = /\.css$/.test(path);
        const el = document.createElement(css ? 'link' : 'script');
        if (css) { el.rel = 'stylesheet'; el.href = path; } else { el.src = path; el.async = false; }
        el.onload = () => resolve(path);
        el.onerror = () => reject(new Error('load: ' + path + ' did not load'));
        document.head.appendChild(el);
      }));
    }
    return loading.get(path);
  }
  async function loadMount(el) {
    const files = el.dataset.load.split(/\s+/).filter(Boolean);
    if (!files.length) throw new Error('#' + el.id + ': data-load names no file');
    try {
      for (const f of files) await loadFile(f);
    } catch (err) {
      const p = document.createElement('p');
      p.className = 'walk-missing';
      p.textContent = 'The data file for this picture did not load (' + err.message.replace(/^load: | did not load$/g, '') + ').';
      el.replaceChildren(p);
      throw err;
    }
    mountAll();
  }
  function bindLazyMounts() {
    const mounts = [...document.querySelectorAll('[data-load]')];
    if (!mounts.length) return;
    if (!('IntersectionObserver' in window)) { mounts.forEach(loadMount); return; }
    const seen = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        seen.unobserve(e.target);
        loadMount(e.target);
      });
    }, { rootMargin: '100% 0px' });
    mounts.forEach((el) => seen.observe(el));
  }

  window.WALK = { REPO, register, mountAll, fmt, valueBox, hover, dataLink, loadFile };

  function init() {
    bindThemeToggles();
    bindChapterList();
    mountAll();
    bindLazyMounts();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
