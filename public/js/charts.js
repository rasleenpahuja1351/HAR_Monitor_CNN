/* Editorial chart helpers (plain SVG/HTML, no libraries). */
(function () {
  const G = '#1D3F04', G3 = '#92998E', GL = '#C9D2BE', RULE = '#DDD8CE', SOFT = '#ECE8DF', MUTED = '#6B6862', INK = '#111111';
  const LABEL = {
    LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking',
    WALKING_DOWNSTAIRS: 'Walking downstairs', WALKING_UPSTAIRS: 'Walking upstairs'
  };
  const SHORT = { LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking', WALKING_DOWNSTAIRS: 'Downstairs', WALKING_UPSTAIRS: 'Upstairs' };
  const TINY = { LAYING: 'Lay', SITTING: 'Sit', STANDING: 'Stand', WALKING: 'Walk', WALKING_DOWNSTAIRS: 'Down', WALKING_UPSTAIRS: 'Up' };
  const DYNAMIC = new Set(['WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const MONO = 'font-family="JetBrains Mono, monospace"';

  // Illustrative waveform (clearly labelled as not dataset output in the page)
  function wave(el) {
    let g = '';
    for (let i = 0; i <= 4; i++) g += `<line x1="8" y1="${30 + i * 40}" x2="584" y2="${30 + i * 40}" stroke="${SOFT}"/>`;
    g += `<rect x="8" y="10" width="288" height="200" fill="${G}" fill-opacity=".06"/>`;
    [[G, 2], [G3, 1.5], [GL, 1.5]].forEach(([c, w], k) => {
      let d = '';
      for (let i = 0; i <= 128; i++) {
        const x = 8 + i * 4.5, y = 110 + Math.sin(i * 0.19 + k * 1.3) * (48 - k * 12) + Math.sin(i * 0.61 + k) * (10 + k * 3) - k * 6;
        d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
      }
      g = g + `<path d="${d}" fill="none" stroke="${c}" stroke-width="${w}"/>`;
    });
    el.innerHTML = g;
  }

  function curves(el, hist, best) {
    const tr = hist.accuracy, va = hist.val_accuracy, n = tr.length;
    const L = 52, R = 1170, T = 16, B = 240;
    const lo = Math.floor(Math.min(...tr, ...va) * 20) / 20, hi = 1;
    const X = (i) => L + i * (R - L) / (n - 1), Y = (v) => B - (v - lo) / (hi - lo) * (B - T);
    let g = '';
    for (let v = Math.ceil(lo * 20) / 20; v <= hi + 1e-9; v += 0.05)
      g += `<line x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${SOFT}"/><text x="${L - 10}" y="${Y(v) + 4}" fill="${MUTED}" font-size="11" ${MONO} text-anchor="end">${Math.round(v * 100)}%</text>`;
    const step = Math.max(1, Math.round(n / 10));
    for (let i = 0; i < n; i += step) g += `<text x="${X(i)}" y="${B + 22}" fill="${MUTED}" font-size="11" ${MONO} text-anchor="middle">${i + 1}</text>`;
    g += `<text x="${R}" y="${B + 38}" fill="${MUTED}" font-size="10" ${MONO} text-anchor="end">EPOCH</text>`;
    const path = (a) => a.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
    if (best) g += `<line x1="${X(best - 1)}" x2="${X(best - 1)}" y1="${T}" y2="${B}" stroke="${INK}" stroke-dasharray="4 4" stroke-opacity=".5"/>`;
    g += `<path d="${path(tr)}" fill="none" stroke="${G3}" stroke-width="2"/><path d="${path(va)}" fill="none" stroke="${G}" stroke-width="2.4"/>`;
    if (best) {
      const vb = va[best - 1], x = X(best - 1), y = Y(vb), bx = Math.min(Math.max(x, L + 90), R - 90);
      g += `<circle cx="${x}" cy="${y}" r="5" fill="#fff" stroke="${G}" stroke-width="2.5"/>
        <rect x="${bx - 86}" y="${y + 14}" width="172" height="28" rx="4" fill="${G}"/>
        <text x="${bx}" y="${y + 32}" fill="#fff" font-size="11.5" ${MONO} text-anchor="middle">BEST · EPOCH ${best} · ${(vb * 100).toFixed(1)}%</text>`;
    }
    el.innerHTML = g;
  }

  function confusion(el, classes, cm) {
    let h = '<div></div>' + classes.map((c) => `<div class="h">${TINY[c]}</div>`).join('');
    let worst = { v: 0 };
    cm.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v > worst.v) worst = { v, i, j }; }));
    cm.forEach((row, i) => {
      h += `<div class="rl">${SHORT[classes[i]]}</div>`;
      row.forEach((v, j) => {
        let cls = 'c', style = '';
        if (i === j) cls += ' d';
        else if (i === worst.i && j === worst.j) cls += ' w';
        else if (!v) cls += ' z';
        else { const p = Math.min(1, 0.25 + v / 110); style = `background:rgb(${[246, 244, 239].map((a, k) => Math.round(a + ([122, 122, 114][k] - a) * p)).join(',')});${p > 0.7 ? 'color:#fff' : ''}`; }
        h += `<div class="${cls}" style="${style}" role="cell" title="True ${SHORT[classes[i]]}, predicted ${SHORT[classes[j]]}: ${v}">${v}</div>`;
      });
    });
    el.innerHTML = h;
    return worst;
  }

  // rows: [{label, value, text, muted}] with value in 0..1
  function bars(el, rows, cls = '') {
    el.innerHTML = rows.map((r) => `<div class="bar ${r.top ? 'top' : ''} ${cls}"><span>${esc(r.label)}</span><span class="t"><i class="${r.muted ? 'm' : ''}" style="width:${Math.max(0.6, r.value * 100)}%"></i></span><b>${esc(r.text)}</b></div>`).join('');
  }

  function spark(el, series, ref) {
    const W = 400, H = 64, lo = 0.6, Y = (v) => H - 4 - (Math.max(lo, v) - lo) / (1 - lo) * (H - 8);
    let g = `<line x1="0" x2="${W}" y1="${Y(ref)}" y2="${Y(ref)}" stroke="${RULE}" stroke-dasharray="4 4"/>`;
    if (series.length > 1) {
      const d = series.map((v, i) => (i ? 'L' : 'M') + (i * W / (series.length - 1)).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
      g += `<path d="${d}" fill="none" stroke="${G}" stroke-width="1.8" vector-effect="non-scaling-stroke"/>`;
    }
    el.innerHTML = g;
  }

  window.Charts = { wave, curves, confusion, bars, spark, LABEL, SHORT, DYNAMIC, esc };
})();
