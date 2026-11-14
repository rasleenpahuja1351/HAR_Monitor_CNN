/* HAR Monitor page logic (editorial redesign). */
(function () {
  const { LABEL, SHORT, DYNAMIC, esc } = window.Charts;

  // ------------------------------------------------------------------ config
  const CONFIG = {
    streamEveryMs: 2560,      // one 2.56 s window at a time, like the real sensor (at 1×)
    timelineLength: 60,       // windows kept in the timeline strip
    logLength: 8,             // rows kept in the event log
    uploadChunkRows: 250,     // rows per /api/predict/batch request (stays < 4.5 MB)
    lowConfidence: 0.7,
    // CNN results come from the project report (not served by the API).
    cnn: { acc: 0.891, f1: 0.890, size: '53,222 params' },
    // Example responses per activity. Replace with your line's real rules.
    responses: {
      LAYING: 'Possible fall: alert supervisor, stop nearby machines.',
      SITTING: 'Operator seated at station: normal operation.',
      STANDING: 'Operator standing at station: normal operation.',
      WALKING: 'Operator moving: keep robot cell at reduced speed.',
      WALKING_UPSTAIRS: 'Operator on stairs: pause forklift lane below.',
      WALKING_DOWNSTAIRS: 'Operator on stairs: pause forklift lane below.'
    },
    shortResponses: {
      LAYING: 'Alert supervisor', SITTING: 'No action', STANDING: 'No action', WALKING: 'Slow robot cell',
      WALKING_UPSTAIRS: 'Pause forklift lane', WALKING_DOWNSTAIRS: 'Pause forklift lane'
    }
  };

  const $ = (s, el = document) => el.querySelector(s);
  const setK = (k, v) => document.querySelectorAll(`[data-k="${k}"]`).forEach((e) => (e.textContent = v));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  // fetch with timeout + retries (network errors and 5xx are retried, 4xx are not)
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function api(path, opts = {}, { timeout = 10000, retries = 2 } = {}) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeout);
      try {
        const r = await fetch(path, { ...opts, signal: ctrl.signal });
        clearTimeout(t);
        if (r.ok) return await r.json();
        let msg = r.status + ' ' + r.statusText;
        try { const j = await r.json(); if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
        const err = new Error(msg); err.status = r.status;
        if (r.status < 500 && r.status !== 429) throw err;        // client error: do not retry
        lastErr = err;
      } catch (e) {
        clearTimeout(t);
        if (e.status && e.status < 500 && e.status !== 429) throw e;
        lastErr = e.name === 'AbortError' ? new Error('Request timed out') : e;
      }
      if (attempt < retries) await sleep(600 * 2 ** attempt);
    }
    throw lastErr;
  }

  let CLASSES = Object.keys(LABEL);
  let TEST_ACC = 0.935;
  Charts.wave($('#wave'));

  // ---------------------------------------------------------------- health
  api('/api/health').then((h) => {
    $('#api-dot').className = 'dot ok';
    $('#api-text').textContent = 'FastAPI /predict · online';
    CLASSES = h.classes;
  }).catch((e) => {
    $('#api-dot').className = 'dot bad';
    $('#api-text').textContent = e.status === 503 ? 'API degraded' : 'API offline';
  });

  // ---------------------------------------------------------------- metrics
  api('/api/metrics').then((m) => {
    CLASSES = m.classes;
    const d = m.dataset, k = m.keras, rf = m.random_forest;
    TEST_ACC = k.test_accuracy;
    setK('acc', pct(k.test_accuracy));
    setK('acc_s', rf ? `Keras Dense v1. Random Forest scores ${pct(rf.test_accuracy)} on the same split.` : 'Keras Dense v1 on held-out subjects.');
    setK('test', d.test_windows.toLocaleString('en-US'));
    setK('test_s', `From ${d.test_subjects} subjects the model never saw during training.`);
    setK('val', pct(k.val_accuracy));
    setK('val_s', k.best_epoch ? `Epoch ${k.best_epoch}, subject-wise validation. Early stopping restored this epoch.` : 'Subject-wise validation.');
    setK('feat', d.features);

    // model table
    const rows = [
      { name: 'Keras Dense v1', acc: k.test_accuracy, f1: k.macro_f1, size: k.params.toLocaleString('en-US') + ' params', tag: 'Deployed', cls: 'best' },
      rf && { name: 'Random Forest', acc: rf.test_accuracy, f1: rf.macro_f1, size: '300 trees', tag: 'Baseline', cls: 'mid' },
      { name: '1D CNN', acc: (m.cnn || {}).test_accuracy || CONFIG.cnn.acc, f1: (m.cnn || {}).macro_f1 || CONFIG.cnn.f1,
        size: m.cnn && m.cnn.params ? m.cnn.params.toLocaleString('en-US') + ' params' : CONFIG.cnn.size, tag: 'Compared', cls: '' }
    ].filter(Boolean);
    $('#models').innerHTML = rows.map((r) => `<tr class="${r.cls}"><td>${r.name}</td>
      <td><div class="acc"><span class="track"><i style="width:${Math.max(3, (r.acc * 100 - 85) / 10 * 100)}%"></i></span><b>${pct(r.acc)}</b></div></td>
      <td class="num">${r.f1.toFixed(3)}</td><td class="note">${r.size}</td>
      <td><span class="chip ${r.cls === 'best' ? 'solid' : ''}">${r.tag}</span></td></tr>`).join('');

    if (k.history) {
      Charts.curves($('#curve'), k.history, k.best_epoch);
      $('#curve-sub').textContent = `Subject-wise validation · early stopping restored epoch ${k.best_epoch}`;
    }

    const worst = Charts.confusion($('#cm'), m.classes, k.confusion);
    $('#cm-lab').textContent = `Section 03 · Confusion matrix · Keras Dense v1 · ${d.test_windows.toLocaleString('en-US')} test windows`;
    if (worst.v) {
      $('#worst-n').textContent = worst.v;
      $('#worst-t').textContent = `${SHORT[m.classes[worst.i]].toLowerCase()} windows were read as ${SHORT[m.classes[worst.j]].toLowerCase()}. A waist-worn phone sees almost the same posture in both; neither is a safety risk.`;
    }
    Charts.bars($('#f1'), m.classes.map((c, i) => ({ label: SHORT[c], value: k.f1[i], text: k.f1[i].toFixed(3), muted: k.f1[i] < 0.9 })));

    const order = m.classes.map((c, i) => i).sort((a, b) => d.train_counts[b] - d.train_counts[a]);
    const mx = Math.max(...d.train_counts);
    Charts.bars($('#dist'), order.map((i) => ({ label: SHORT[m.classes[i]], value: d.train_counts[i] / mx, text: d.train_counts[i].toLocaleString('en-US'), muted: !DYNAMIC.has(m.classes[i]) })));
  }).catch((e) => {
    console.error('metrics', e);
    ['#models', '#f1', '#dist'].forEach((id) => {
      const el = $(id);
      if (el) el.innerHTML = (id === '#models' ? '<tr><td colspan="5" class="note">' : '<p class="note">') +
        'Could not load results from the API (' + esc(e.message) + '). Refresh the page to try again.' + (id === '#models' ? '</td></tr>' : '</p>');
    });
  });

  // ------------------------------------------------------------------ live
  const tl = [], log = [], accSeries = [];
  let seen = 0, correct = 0, latSum = 0, okN = 0, revN = 0, timer = null, paused = false, speed = 1;
  const strip = $('#tl');
  strip.innerHTML = Array.from({ length: CONFIG.timelineLength }, () => '<div class="empty"></div>').join('');

  function showWindow(w, latency) {
    seen += 1; correct += w.correct ? 1 : 0; latSum += latency;
    w.low_confidence ? revN++ : okN++;
    accSeries.push(correct / seen);
    if (accSeries.length > 120) accSeries.shift();

    $('#cur-lab').textContent = `Current window · Subject ${w.subject} · #${seen.toLocaleString('en-US')}`;
    const words = LABEL[w.activity].split(' ');
    $('#pred').innerHTML = words.length > 1 ? `${esc(words[0])}<br><em>${esc(words.slice(1).join(' '))}</em>` : `<em>${esc(words[0])}</em>`;
    $('#conf').textContent = pct(w.confidence, 0);
    const probs = Object.entries(w.probabilities).sort((a, b) => b[1] - a[1]);
    Charts.bars($('#probs'), probs.map(([c, p], i) => ({ label: LABEL[c], value: p, text: p.toFixed(2), top: i === 0 })));
    $('#truth').innerHTML = `<span class="lab">True activity</span><b>${esc(LABEL[w.true_activity])}</b>` +
      (w.correct ? '<span class="chip solid">✓ correct</span>' : '<span class="chip ink">✗ misclassified</span>');
    const resp = $('#resp');
    if (w.low_confidence) {
      resp.className = 'resp warn';
      $('#resp-t').textContent = `Low confidence (${pct(w.confidence, 0)}): sent to a supervisor, no machine action.`;
    } else {
      resp.className = 'resp';
      $('#resp-t').textContent = CONFIG.responses[w.activity] || '';
    }
    $('#subj-chip').textContent = `Subject ${w.subject}`;
    $('#win-chip').textContent = `Window ${seen.toLocaleString('en-US')}`;

    $('#sacc').textContent = pct(correct / seen);
    $('#sacc-n').textContent = `of ${seen.toLocaleString('en-US')} window${seen === 1 ? '' : 's'}`;
    Charts.spark($('#spark'), accSeries, TEST_ACC);
    $('#spark-note').textContent = `Dashed line: ${pct(TEST_ACC)} full test-set accuracy (Keras Dense v1).`;
    $('#g-ok').textContent = okN; $('#g-rev').textContent = revN;

    tl.push({ n: seen, w });
    if (tl.length > CONFIG.timelineLength) tl.shift();
    const pad = CONFIG.timelineLength - tl.length;
    strip.innerHTML = '<div class="empty"></div>'.repeat(pad) + tl.map(({ w: t }) =>
      `<div class="${DYNAMIC.has(t.activity) ? 'dy' : ''} ${t.correct ? '' : 'x'} ${t.low_confidence ? 'lc' : ''}" title="${LABEL[t.activity]} · ${pct(t.confidence, 0)}${t.correct ? '' : ' · true: ' + LABEL[t.true_activity]}"></div>`).join('');
    $('#tl-n').textContent = CONFIG.timelineLength;
    $('#tl-a').textContent = '#' + tl[0].n.toLocaleString('en-US');
    $('#tl-b').textContent = 'Now · #' + seen.toLocaleString('en-US');

    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    log.unshift({ time, n: seen, w });
    if (log.length > CONFIG.logLength) log.pop();
    $('#log').innerHTML = log.map(({ time, n, w: x }) => `<tr><td>${time}</td><td>#${n.toLocaleString('en-US')}</td>
      <td>${esc(LABEL[x.activity])}</td><td class="${x.correct ? 'dim' : 'hi'}">${esc(LABEL[x.true_activity])}</td>
      <td class="${x.low_confidence ? '' : 'g'}">${x.confidence.toFixed(2)}</td>
      <td>${x.low_confidence ? '<span class="chip">Sent to review</span>' : esc(CONFIG.shortResponses[x.activity] || '')}</td></tr>`).join('');
    $('#lat').textContent = (latSum / seen).toFixed(2) + ' ms';
  }

  let fails = 0, busy = false;
  async function tick() {
    if (busy) return;                       // never overlap requests on slow networks
    busy = true;
    try {
      const r = await api('/api/stream?n=1', {}, { timeout: 8000, retries: 1 });
      fails = 0;
      if (r && r.windows && r.windows.length) showWindow(r.windows[0], r.latency_ms || 0);
    } catch (e) {
      fails += 1;
      $('#resp').className = 'resp warn';
      if (fails >= 3) {
        stop(); paused = true;
        toggle.innerHTML = '▶&nbsp; Reconnect';
        toggle.setAttribute('aria-pressed', 'true');
        $('#resp-t').textContent = 'The stream is paused because the API did not respond (' + e.message + '). Press Reconnect to try again.';
      } else {
        $('#resp-t').textContent = 'Connection problem (' + e.message + '); retrying…';
      }
    } finally {
      busy = false;
    }
  }
  const interval = () => CONFIG.streamEveryMs / speed;
  function start() { stop(); tick(); timer = setInterval(tick, interval()); }
  function stop() { clearInterval(timer); timer = null; }

  const toggle = $('#live-toggle');
  toggle.addEventListener('click', () => {
    paused = !paused;
    if (!paused) fails = 0;
    toggle.innerHTML = paused ? '▶&nbsp; Resume stream' : '❚❚&nbsp; Pause stream';
    toggle.setAttribute('aria-pressed', String(paused));
    paused ? stop() : start();
  });
  $('#live-next').addEventListener('click', () => tick());
  document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
    speed = Number(b.dataset.speed);
    if (!paused) start();
  }));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop(); else if (!paused && !timer) start();
  });
  start();

  // ------------------------------------------------------------- nav spy
  const links = [...document.querySelectorAll('.links a[data-sec]')];
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) links.forEach((l) => l.classList.toggle('on', l.dataset.sec === en.target.id));
    });
  }, { rootMargin: '-35% 0px -60% 0px' });
  links.forEach((l) => { const el = document.getElementById(l.dataset.sec); if (el) io.observe(el); });
  links[0].classList.add('on');

  // ------------------------------------------------------------- upload
  function splitCSVLine(line) {
    const out = []; let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) {
        if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
        else if (ch === '"') q = false;
        else cur += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }

  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
    if (!lines.length) throw new Error('The file is empty.');
    const first = splitCSVLine(lines[0]);
    const hasHeader = first.some((v) => v.trim() !== '' && isNaN(Number(v)));
    let featIdx, labelIdx = -1;
    if (hasHeader) {
      const names = first.map((s) => s.trim());
      labelIdx = names.findIndex((n) => n.toLowerCase() === 'activity');
      featIdx = names.map((n, i) => i).filter((i) => i !== labelIdx && names[i].toLowerCase() !== 'subject');
    } else {
      featIdx = first.map((_, i) => i).slice(0, 561);
    }
    if (featIdx.length !== 561) throw new Error(`Expected 561 feature columns, found ${featIdx.length}.`);
    const rows = [], labels = [];
    for (let r = hasHeader ? 1 : 0; r < lines.length; r++) {
      const cells = lines[r].split(',');
      const row = featIdx.map((i) => Math.round(Number(cells[i]) * 1e6) / 1e6);
      if (row.some((v) => !Number.isFinite(v))) throw new Error(`Row ${r + 1} has a missing or non-numeric value.`);
      rows.push(row);
      if (labelIdx >= 0) labels.push((cells[labelIdx] || '').replace(/"/g, '').trim());
    }
    return { rows, labels: labelIdx >= 0 ? labels : null };
  }

  function toCSV(preds, conf, labels) {
    const head = labels ? 'row,predicted,confidence,true_activity,correct' : 'row,predicted,confidence';
    const body = preds.map((p, i) => labels ? `${i + 1},${p},${conf[i]},${labels[i]},${p === labels[i]}` : `${i + 1},${p},${conf[i]}`);
    return [head, ...body].join('\n');
  }

  async function handleFile(file) {
    const res = $('#res');
    if (!file) return;
    res.innerHTML = `<p class="lab">${esc(file.name)} · ${(file.size / 1e6).toFixed(2)} MB</p><div class="prog"><i id="pbar"></i></div><p class="note" id="pstat">Reading file…</p>`;
    try {
      const { rows, labels } = parseCSV(await file.text());
      const preds = [], conf = [];
      let ms = 0;
      for (let i = 0; i < rows.length; i += CONFIG.uploadChunkRows) {
        const chunk = rows.slice(i, i + CONFIG.uploadChunkRows);
        const r = await api('/api/predict/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: chunk }) });
        preds.push(...r.predictions); conf.push(...r.confidence); ms += r.latency_ms;
        $('#pbar').style.width = pct(Math.min(1, (i + chunk.length) / rows.length));
        $('#pstat').textContent = `Classified ${preds.length.toLocaleString('en-US')} of ${rows.length.toLocaleString('en-US')} rows…`;
      }
      const counts = {}; CLASSES.forEach((c) => (counts[c] = 0));
      preds.forEach((p) => (counts[p] = (counts[p] || 0) + 1));
      const acc = labels ? preds.filter((p, i) => p === labels[i]).length / preds.length : null;
      const low = conf.filter((c) => c < CONFIG.lowConfidence).length;
      const mx = Math.max(...Object.values(counts));
      const url = URL.createObjectURL(new Blob([toCSV(preds, conf, labels)], { type: 'text/csv' }));
      res.innerHTML = `
        <p class="lab">${esc(file.name)} · done</p>
        <div class="rstats">
          <div><span class="lab">Rows</span><b>${preds.length.toLocaleString('en-US')}</b></div>
          <div><span class="lab">${labels ? 'Accuracy' : 'Avg confidence'}</span><b class="g">${labels ? pct(acc) : pct(conf.reduce((a, b) => a + b, 0) / conf.length)}</b></div>
          <div><span class="lab">Low confidence</span><b>${low.toLocaleString('en-US')}</b></div>
        </div>
        <div class="bars" id="rbars"></div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
          <a class="btn primary" href="${url}" download="${esc(file.name.replace(/\.csv$/i, ''))}_predictions.csv">Download predictions CSV</a>
          <span class="note">Server inference: ${ms.toFixed(1)} ms total</span>
        </div>`;
      Charts.bars($('#rbars'), CLASSES.map((c) => ({ label: SHORT[c], value: counts[c] / mx, text: String(counts[c]), muted: !DYNAMIC.has(c) })));
    } catch (e) {
      res.innerHTML = `<p class="err">Could not process the file: ${esc(e.message)}</p>`;
    }
  }

  const drop = $('#drop'), input = $('#csv');
  input.addEventListener('change', () => handleFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
})();
