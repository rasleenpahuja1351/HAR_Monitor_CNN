"""
Pipeline & Live Prediction page for HAR Monitor (Group EIC-16), written in Python.

Served by the FastAPI app at  /live  (see app.py: app.include_router(create_router(...))).

What Python does here
  * the pipeline diagram is generated from the PIPELINE / TRAINING lists below
  * the numbers on the page (accuracy, windows, split) are read from model/metrics.json
  * the first live prediction is made on the server: a real held-out test window is
    classified by the deployed model before the page is sent
After the page loads, the browser keeps asking /api/stream for a new window every
2.56 s (the same Python model answers each request), so the page stays live.
"""
from __future__ import annotations

import html
import json
import time
from typing import Callable

import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

# --------------------------------------------------------------------------- #
# Pipeline definition (edit here to change the diagram)
# --------------------------------------------------------------------------- #
PIPELINE = [
    {"tag": "On the worker", "title": "Wearable sensors",
     "text": "Waist-worn phone: accelerometer + gyroscope, sampled at 50 Hz."},
    {"tag": "Signal processing", "title": "Windowing",
     "text": "128 readings = one 2.56 s window, 50% overlap with the next."},
    {"tag": "Feature engineering", "title": "561 features",
     "text": "Time- and frequency-domain statistics (mean, energy, entropy, FFT…)."},
    {"tag": "Our model", "title": "Keras Dense network", "highlight": True,
     "text": "561 → 512 → 256 → 128 → 6 with BatchNorm, ReLU and Dropout; scaler folded in, runs in NumPy."},
    {"tag": "FastAPI on Vercel", "title": "/api/predict",
     "text": "Softmax gives the activity and its confidence in under a millisecond."},
]
DECISION = {
    "question": "Is confidence ≥ {threshold:.0%}?",
    "yes": ("Line response", "Walking → slow the robot cell · Stairs → pause the forklift lane · Laying → possible fall, alert supervisor."),
    "no": ("Supervisor review", "No machine action is taken; the window is flagged for a person to confirm."),
}


def training_steps(m: dict) -> list[dict]:
    """Offline path, filled in from model/metrics.json so the numbers always match the model."""
    d = m.get("dataset", {})
    total = d.get("train_windows", 7352) + d.get("test_windows", 2947)
    accs = []
    for key, name in (("keras", "Dense"), ("random_forest", "RF"), ("cnn", "CNN")):
        if key in m and "test_accuracy" in m[key]:
            accs.append(f"{name} {m[key]['test_accuracy'] * 100:.1f}%")
    return [
        {"title": "UCI HAR data", "text": f"{total:,} windows, 30 people, {len(m.get('classes', [])) or 6} activities."},
        {"title": "Person-wise split", "text": "16 train · 5 validation · 9 test people."},
        {"title": "Train 3 models", "text": "Dense NN, 1D CNN, Random Forest; early stopping."},
        {"title": "Evaluate", "text": f"{d.get('test_subjects', 9)} unseen people: " + ", ".join(accs) + "."},
        {"title": "Export & deploy", "text": "Best model folded into NumPy weights → step 4 above.", "highlight": True},
    ]


# --------------------------------------------------------------------------- #
# HTML building
# --------------------------------------------------------------------------- #
e = html.escape


def _step(s: dict, number: int | None) -> str:
    num = f'<span class="num">{number}</span>' if number else ""
    tag = f'<p class="tag">{e(s["tag"])}</p>' if s.get("tag") else ""
    cls = "step hi" if s.get("highlight") else "step"
    return f'<li class="{cls}">{num}{tag}<h3>{e(s["title"])}</h3><p class="note">{e(s["text"])}</p></li>'


def pipeline_html(metrics: dict, threshold: float) -> str:
    online = "\n        ".join(_step(s, i + 1) for i, s in enumerate(PIPELINE))
    offline = "\n        ".join(_step(s, None) for s in training_steps(metrics))
    yes_t, yes_d = DECISION["yes"]
    no_t, no_d = DECISION["no"]
    return f"""
  <section class="psec" id="pipeline">
    <div class="sec-h" style="padding-top:8px">
      <div><p class="lab">Part 1 · Pipeline</p><h2>From wearable sensor<br>to <em>factory response</em></h2></div>
      <p class="body">The model is trained once, offline, on the UCI HAR data. In use, every 2.56-second window from a worker's sensor goes through the online path below and ends in a safety decision.</p>
    </div>
    <div class="lane">
      <p class="lab lane-t"><span class="lane-dot"></span>Online · every 2.56 s</p>
      <ol class="flow">
        {online}
      </ol>
      <div class="decide">
        <div class="gate-q"><span class="num">{len(PIPELINE) + 1}</span><p class="tag">Decision</p><h3>{e(DECISION["question"].format(threshold=threshold))}</h3></div>
        <div class="branches">
          <div class="br yes"><p class="lab">Yes · automatic</p><h3>{e(yes_t)}</h3><p class="note">{e(yes_d)}</p></div>
          <div class="br no"><p class="lab">No · human check</p><h3>{e(no_t)}</h3><p class="note">{e(no_d)}</p></div>
        </div>
      </div>
    </div>
    <div class="lane off">
      <p class="lab lane-t"><span class="lane-dot"></span>Offline · training, done once</p>
      <ol class="flow small">
        {offline}
      </ol>
    </div>
  </section>"""


def render_page(metrics: dict, first: dict | None, threshold: float, group: str) -> str:
    k = metrics.get("keras", {})
    boot = {"window": first["window"] if first else None,
            "latency_ms": first["latency_ms"] if first else 0,
            "metrics": {"test_accuracy": k.get("test_accuracy")}}
    boot_json = json.dumps(boot).replace("</", "<\\/")
    rendered = ""
    if first:
        w = first["window"]
        name = w["activity"].replace("_", " ").capitalize()
        rendered = (f'Server-side prediction: window #1 was classified by Python before the page was sent '
                    f'({e(name)}, {w["confidence"] * 100:.0f}% confidence, subject {w["subject"]}).')
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline &amp; Live Prediction · HAR Monitor</title>
<meta name="description" content="Pipeline diagram and live activity prediction, generated by the HAR Monitor Python backend.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='6' fill='%231D3F04'/%3E%3Cpath d='M3 12h4l3-7 4 14 3-7h4' fill='none' stroke='white' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>""" + CSS + """</style>
</head>
<body>
<header class="nav">
  <div class="wrap nav-in">
    <a class="brand" href="/">HAR Monitor</a>
    <div class="nav-r">
      <span class="chip g"><i class="dot" id="api-dot"></i><span id="api-text">Connecting…</span></span>
      <span class="chip g hide-sm">Model: Keras Dense v1</span>
      <a class="back" href="/">← Dashboard</a>
    </div>
  </div>
</header>
<main class="wrap">
  <section class="head">
    <div>
      <p class="lab">HAR Monitor · how it works and what it does</p>
      <h1>Pipeline &amp;<br><em>live prediction</em></h1>
    </div>
    <div class="jump">
      <a class="btn" href="#pipeline">1 · The pipeline ↓</a>
      <a class="btn primary" href="#live">2 · Live prediction ↓</a>
    </div>
  </section>
""" + pipeline_html(metrics, threshold) + """

  <section class="sec-h" id="live">
    <div><p class="lab">Part 2 · Live prediction · held-out test windows</p><h2>Watch the pipeline<br><em>run live</em></h2></div>
    <p class="body">The first window below was classified by Python on the server before this page was sent. After that, every 2.56 s a real window from one of the 9 test subjects (never seen in training) is sent to <code>/api/stream</code>, checked against its true label, and the confidence decides the factory response (step 6).</p>
  </section>
""" + LIVE_BODY.replace("{{RENDERED}}", rendered) + """
</main>
<footer class="foot">
  <div class="wrap foot-in">
    <span style="font:400 28px var(--serif)">HAR Monitor</span>
    <span class="lab">Group """ + e(group) + """ · UCS321 EST AI Project · Thapar Institute · Data: UCI HAR (DOI 10.24432/C54S4K)</span>
  </div>
</footer>
<script>window.__FIRST__ = """ + boot_json + """;</script>
<script>""" + JS + """</script>
</body>
</html>
""")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def create_router(get_model: Callable, get_samples: Callable, get_metrics: Callable, describe: Callable,
                  threshold: float = 0.7, group: str = "EIC-16") -> APIRouter:
    router = APIRouter()

    @router.get("/live", response_class=HTMLResponse, include_in_schema=False)
    @router.get("/live/", response_class=HTMLResponse, include_in_schema=False)
    def live_page():
        try:
            metrics = get_metrics()
        except Exception:
            metrics = {}
        first = None
        try:  # one real prediction made in Python before the page is sent
            model = get_model()
            X, y, subj = get_samples()
            i = int(np.random.default_rng().integers(0, len(X)))
            t0 = time.perf_counter()
            probs = model.predict_proba(X[i:i + 1])[0]
            latency = (time.perf_counter() - t0) * 1000
            w = describe(probs, model.classes)
            true = model.classes[int(y[i])]
            w.update({"true_activity": true, "subject": int(subj[i]), "correct": w["activity"] == true})
            first = {"window": w, "latency_ms": round(latency, 3)}
        except Exception:
            first = None  # the page still works; the browser fetches windows from /api/stream
        return HTMLResponse(render_page(metrics, first, threshold, group), headers={"Cache-Control": "no-store"})

    return router


# --------------------------------------------------------------------------- #
# Static parts of the page (styles, live-panel markup and browser script)
# --------------------------------------------------------------------------- #
CSS = r"""
:root {
  --paper: #F6F4EF; --card: #FFFFFF; --ink: #111111; --body: #3A3833; --muted: #6B6862;
  --rule: #DDD8CE; --soft: #ECE8DF; --g: #1D3F04; --g2: #65705E; --g3: #92998E; --gt: #E7ECDF; --gl: #C9D2BE;
  --serif: "Instrument Serif", "Times New Roman", serif;
  --sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 72px; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 400 16px/1.55 var(--sans); -webkit-font-smoothing: antialiased; }
a { color: inherit; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 40px; }
.lab { font: 400 11px/1.4 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0; }
.note { font-size: 13px; color: var(--muted); margin: 0; line-height: 1.5; }
.g { color: var(--g); }
h1, h2, h3 { font-weight: 400; margin: 0; font-family: var(--serif); }
em { font-style: italic; color: var(--g); }

/* nav */
.nav { position: sticky; top: 0; z-index: 10; background: rgba(246,244,239,.92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--g); }
.nav-in { display: flex; align-items: center; justify-content: space-between; gap: 20px; height: 64px; }
.brand { font: 400 26px var(--serif); text-decoration: none; }
.nav-r { display: flex; align-items: center; gap: 10px; }
.chip { display: inline-flex; align-items: center; gap: 7px; font: 12px var(--mono); padding: 6px 12px; border: 1px solid var(--rule); border-radius: 999px; background: var(--card); white-space: nowrap; }
.chip.g { border-color: var(--g); }
.chip.solid { background: var(--g); border-color: var(--g); color: #fff; }
.chip.ink { background: var(--ink); border-color: var(--ink); color: #fff; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--g3); display: inline-block; }
.dot.ok { background: #3E8E14; animation: pulse 1.6s infinite; } .dot.bad { background: #B42318; }
@keyframes pulse { 50% { opacity: .35; } }
.back { font-size: 14px; color: var(--g); text-decoration: none; }
.back:hover { text-decoration: underline; }

/* header + controls */
.head { padding: 56px 0 28px; display: flex; justify-content: space-between; align-items: flex-end; gap: 40px; flex-wrap: wrap; }
.head h1 { font-size: clamp(52px, 7vw, 96px); line-height: .92; letter-spacing: -.015em; margin-top: 16px; }
.head p.body { color: var(--body); max-width: 480px; margin: 0; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding-bottom: 28px; }
.btn { display: inline-flex; align-items: center; gap: 8px; font: 500 14px var(--sans); padding: 12px 20px; border-radius: 6px; border: 1px solid var(--g); cursor: pointer; text-decoration: none; background: transparent; color: var(--g); }
.btn.primary { background: var(--g); color: #fff; }
.btn.primary:hover { background: #142D02; } .btn:not(.primary):hover { background: var(--gt); }
.btn:focus-visible, .seg button:focus-visible, select:focus-visible { outline: 2px solid var(--g); outline-offset: 2px; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; background: var(--card); }
.seg button { font: 13px var(--mono); border: 0; background: none; padding: 12px 14px; cursor: pointer; color: var(--ink); }
.seg button.on { background: var(--gt); color: var(--g); }
.spacer { flex: 1; }

/* layout */
.grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 24px; }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 28px; }
.stack { display: flex; flex-direction: column; gap: 24px; }

/* current window */
.cur { display: grid; grid-template-columns: 220px 1fr; gap: 32px; align-items: center; padding: 36px; }
.ring { position: relative; width: 220px; height: 220px; }
.ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.ring .c { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring .c b { font: 400 56px/1 var(--serif); }
.pred { font-size: clamp(42px, 4.6vw, 64px); line-height: .95; margin: 10px 0 16px; }
.truth { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; font-size: 14px; margin-bottom: 16px; }
.resp { background: var(--gt); border-left: 3px solid var(--g); padding: 16px 18px; }
.resp .lab { color: var(--g); }
.resp p.r { font: 400 24px/1.2 var(--serif); margin: 6px 0 0; }
.resp.warn { background: var(--soft); border-left-color: var(--ink); }
.resp.warn .lab { color: var(--ink); }
.probs { margin-top: 26px; display: flex; flex-direction: column; gap: 10px; grid-column: 1 / -1; }
.bar { display: grid; grid-template-columns: 170px 1fr 52px; align-items: center; gap: 16px; font-size: 14px; color: var(--muted); }
.bar.top { color: var(--ink); font-weight: 500; }
.bar .t { height: 8px; border-radius: 4px; background: var(--soft); overflow: hidden; }
.bar .t i { display: block; height: 100%; border-radius: 4px; background: var(--g3); transition: width .45s ease; }
.bar.top .t i { background: var(--g); }
.bar b { font: 400 13px var(--mono); text-align: right; }

/* stats */
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }
.stats > div { background: var(--card); padding: 20px; }
.stats b { display: block; font: 400 44px/1.05 var(--serif); margin-top: 6px; }
#spark { width: 100%; height: 70px; display: block; margin-top: 10px; }
.counts { display: flex; flex-direction: column; gap: 9px; margin-top: 14px; }
.counts .bar { grid-template-columns: 120px 1fr 40px; }

/* timeline + log */
.timeline { margin-top: 24px; }
.ch { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
.ch h3 { font-size: 32px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font: 11px var(--mono); text-transform: uppercase; letter-spacing: .06em; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sq { width: 12px; height: 12px; display: inline-block; }
.strip { display: grid; grid-template-columns: repeat(60, 1fr); gap: 4px; align-items: end; height: 76px; margin-top: 20px; }
.strip > div { position: relative; border-radius: 2px; background: var(--gl); height: 32px; transition: height .3s; }
.strip > div.dy { background: var(--g); height: 56px; }
.strip > div.empty { background: var(--soft); }
.strip > div.x::after { content: ""; position: absolute; left: 50%; top: -10px; width: 5px; height: 5px; margin-left: -2.5px; border-radius: 50%; background: var(--ink); }
.strip > div.lc { opacity: .5; }
.axis { display: flex; justify-content: space-between; margin-top: 10px; }
.tbl-wrap { overflow-x: auto; margin-top: 16px; }
table { width: 100%; border-collapse: collapse; }
th { font: 400 11px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); text-align: left; padding: 10px 12px 10px 0; border-bottom: 1px solid var(--ink); }
td { padding: 12px 12px 12px 0; border-bottom: 1px solid var(--rule); font-size: 14px; }
td.m { font: 13px var(--mono); } td.dim { color: var(--muted); } td.hi { font-weight: 500; }
.foot { background: var(--g); color: #fff; padding: 30px 0; margin-top: 64px; }
.foot-in { display: flex; justify-content: space-between; align-items: center; gap: 20px; flex-wrap: wrap; }
.foot .lab { color: var(--gl); }

/* pipeline */
.psec { padding: 8px 0 56px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }
.lane { border: 1px solid var(--rule); border-radius: 8px; padding: 28px; background: var(--card); }
.lane + .lane { margin-top: 20px; }
.lane.off { background: transparent; border-style: dashed; }
.lane-t { display: flex; align-items: center; gap: 8px; margin-bottom: 18px; color: var(--g); }
.lane.off .lane-t { color: var(--muted); }
.lane-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--g); }
.lane.off .lane-dot { background: var(--g3); }
.flow { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(5, 1fr); gap: 34px; }
.step { position: relative; background: var(--paper); border: 1px solid var(--rule); border-radius: 8px; padding: 20px 18px; display: flex; flex-direction: column; gap: 6px; }
.lane.off .step { background: var(--card); }
.step h3 { font-size: 22px; }
.step + .step::before { content: ""; position: absolute; left: -27px; top: 50%; width: 20px; height: 2px; background: var(--g); }
.step + .step::after { content: ""; position: absolute; left: -11px; top: calc(50% - 4px); border: 5px solid transparent; border-left: 6px solid var(--g); }
.lane.off .step + .step::before { background: var(--g3); }
.lane.off .step + .step::after { border-left-color: var(--g3); }
.step.hi, .lane.off .step.hi { background: var(--g); border-color: var(--g); color: #fff; }
.step.hi .note, .step.hi .tag { color: var(--gl); }
.step.hi .num { background: #fff; color: var(--g); }
.num { width: 28px; height: 28px; border-radius: 50%; background: var(--g); color: #fff; font: 500 13px var(--mono); display: inline-flex; align-items: center; justify-content: center; flex: none; }
.tag { font: 400 10.5px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 6px 0 0; }
.flow.small .step h3 { font-size: 19px; }
.decide { display: grid; grid-template-columns: repeat(5, 1fr); gap: 34px; margin-top: 40px; }
.gate-q { grid-column: 5; grid-row: 1; display: flex; flex-direction: column; justify-content: center; gap: 8px; background: var(--card); border: 1.5px solid var(--g); border-radius: 8px; padding: 18px; position: relative; }
.gate-q h3 { font-size: 22px; }
.gate-q::before { content: ""; position: absolute; left: 50%; top: -36px; width: 2px; height: 30px; background: var(--g); }
.gate-q::after { content: ""; position: absolute; left: -12px; top: calc(50% - 5px); border: 5px solid transparent; border-right: 6px solid var(--g); }
.branches { grid-column: 1 / span 4; grid-row: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; position: relative; }
.branches::after { content: ""; position: absolute; right: -30px; top: 50%; width: 22px; height: 2px; background: var(--g); }
.br { border-radius: 8px; padding: 18px 20px; display: flex; flex-direction: column; gap: 6px; }
.br.yes { background: var(--gt); border-left: 3px solid var(--g); }
.br.yes .lab { color: var(--g); }
.br.no { background: var(--soft); border-left: 3px solid var(--ink); }
.br.no .lab { color: var(--ink); }
.br h3, .gate-q h3 { margin: 0; }
.sec-h { padding: 56px 0 28px; display: flex; justify-content: space-between; align-items: flex-end; gap: 40px; flex-wrap: wrap; }
.sec-h h2 { font-size: clamp(40px, 4.5vw, 56px); line-height: 1; margin-top: 14px; }
.sec-h p.body { color: var(--body); max-width: 480px; margin: 0; }
.jump { display: flex; gap: 10px; flex-wrap: wrap; }

@media (max-width: 1000px) {
  .grid { grid-template-columns: 1fr; }
  .cur { grid-template-columns: 180px 1fr; }
  .ring { width: 180px; height: 180px; }
  .flow { grid-template-columns: repeat(3, 1fr); row-gap: 20px; }
  .flow .step:nth-child(3n+1)::before, .flow .step:nth-child(3n+1)::after { display: none; }
  .decide { grid-template-columns: 1fr; }
  .gate-q, .branches { grid-column: 1; }
  .gate-q { grid-row: 1; } .branches { grid-row: 2; }
  .gate-q::before, .gate-q::after, .branches::after { display: none; }
}
@media (max-width: 680px) {
  .wrap { padding: 0 16px; }
  .nav .hide-sm { display: none; }
  .head { padding-top: 36px; }
  .cur { grid-template-columns: 1fr; justify-items: center; text-align: center; padding: 24px; }
  .truth { justify-content: center; }
  .bar { grid-template-columns: 110px 1fr 44px; gap: 10px; font-size: 13px; }
  .strip { grid-template-columns: repeat(30, 1fr); }
  .strip > div:nth-child(-n+30) { display: none; }
  .card { padding: 20px; }
  .lane { padding: 18px; }
  .flow, .branches { grid-template-columns: 1fr; gap: 26px; }
  .flow .step + .step::before { display: block; left: 50%; top: -22px; width: 2px; height: 16px; }
  .flow .step + .step::after { display: block; left: calc(50% - 4px); top: -9px; border: 5px solid transparent; border-top: 6px solid var(--g); border-left-color: transparent; }
  .lane.off .step + .step::after { border-top-color: var(--g3); border-left-color: transparent; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

LIVE_BODY = r"""  <div class="controls">
    <button type="button" class="btn primary" id="toggle" aria-pressed="false">❚❚&nbsp; Pause</button>
    <button type="button" class="btn" id="next">Next window →</button>
    <div class="seg" role="group" aria-label="Speed">
      <button type="button" data-speed="0.5">0.5×</button><button type="button" data-speed="1" class="on">1×</button><button type="button" data-speed="2">2×</button>
    </div>
    <button type="button" class="btn" id="reset">Reset session</button>
    <span class="spacer"></span>
    <span class="chip" id="subj">Subject –</span>
    <span class="chip" id="win">Window 0</span>
  </div>

  <div class="grid">
    <div class="card cur" aria-live="polite">
      <div class="ring">
        <svg viewBox="0 0 120 120" aria-hidden="true">
          <circle cx="60" cy="60" r="52" fill="none" stroke="#ECE8DF" stroke-width="10"/>
          <circle id="arc" cx="60" cy="60" r="52" fill="none" stroke="#1D3F04" stroke-width="10" stroke-linecap="round" stroke-dasharray="0 327" style="transition: stroke-dasharray .5s ease"/>
        </svg>
        <div class="c"><b id="conf">–</b><span class="lab">Confidence</span></div>
      </div>
      <div>
        <p class="lab" id="cur-lab">Current window</p>
        <p class="note" id="ssr">{{RENDERED}}</p>
        <h2 class="pred" id="pred">Waiting…</h2>
        <div class="truth" id="truth"><span class="lab">True activity</span><span>–</span></div>
        <div class="resp" id="resp"><p class="lab">Line response</p><p class="r" id="resp-t">Waiting for the first window…</p></div>
      </div>
      <div class="probs" id="probs"></div>
    </div>

    <div class="stack">
      <div class="stats">
        <div><span class="lab">Session accuracy</span><b class="g" id="acc">–</b></div>
        <div><span class="lab">Windows</span><b id="n">0</b></div>
        <div><span class="lab">Auto-actioned</span><b class="g" id="ok">0</b></div>
        <div><span class="lab">Sent to review</span><b id="rev">0</b></div>
      </div>
      <div class="card">
        <p class="lab">Running accuracy</p>
        <svg id="spark" viewBox="0 0 400 70" preserveAspectRatio="none" aria-label="Running accuracy over the session"></svg>
        <p class="note" id="spark-note">Dashed line: full test-set accuracy.</p>
      </div>
      <div class="card">
        <p class="lab">Predicted activities this session</p>
        <div class="counts" id="counts"></div>
      </div>
    </div>
  </div>

  <section class="card timeline">
    <div class="ch"><h3>Last 60 windows</h3>
      <div class="legend"><span><i class="sq" style="background:#1D3F04"></i>Dynamic</span><span><i class="sq" style="background:#C9D2BE"></i>Static</span><span><i class="dot" style="background:#111"></i>Mismatch</span><span>Faded = low confidence</span></div></div>
    <div class="strip" id="tl"></div>
    <div class="axis"><span class="lab" id="tl-a">–</span><span class="lab" id="tl-b">Now</span></div>
  </section>

  <section class="card timeline">
    <div class="ch"><h3>Event log</h3><span class="note">Server inference per window: <span id="lat">–</span></span></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Time</th><th>Window</th><th>Subject</th><th>Predicted</th><th>True</th><th>Confidence</th><th>Response</th></tr></thead>
        <tbody id="log"><tr><td colspan="7" class="dim">Waiting for the first window…</td></tr></tbody>
      </table>
    </div>
  </section>
"""

JS = r"""
(function () {
  const CONFIG = { everyMs: 2560, timeline: 60, logRows: 10, low: 0.7 };
  const LABEL = { LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking', WALKING_DOWNSTAIRS: 'Walking downstairs', WALKING_UPSTAIRS: 'Walking upstairs' };
  const SHORT = { LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking', WALKING_DOWNSTAIRS: 'Downstairs', WALKING_UPSTAIRS: 'Upstairs' };
  const DYNAMIC = new Set(['WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']);
  const RESPONSE = {
    LAYING: 'Possible fall: alert supervisor, stop nearby machines.',
    SITTING: 'Operator seated at station: normal operation.',
    STANDING: 'Operator standing at station: normal operation.',
    WALKING: 'Operator moving: keep robot cell at reduced speed.',
    WALKING_UPSTAIRS: 'Operator on stairs: pause forklift lane below.',
    WALKING_DOWNSTAIRS: 'Operator on stairs: pause forklift lane below.'
  };
  const SHORT_RESP = { LAYING: 'Alert supervisor', SITTING: 'No action', STANDING: 'No action', WALKING: 'Slow robot cell', WALKING_UPSTAIRS: 'Pause forklift lane', WALKING_DOWNSTAIRS: 'Pause forklift lane' };

  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // fetch with timeout + retries (network errors / 5xx retried, 4xx not)
  async function api(path, { timeout = 8000, retries = 1 } = {}) {
    let last;
    for (let a = 0; a <= retries; a++) {
      const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), timeout);
      try {
        const r = await fetch(path, { signal: ctrl.signal }); clearTimeout(t);
        if (r.ok) return await r.json();
        let msg = r.status + ' ' + r.statusText;
        try { const j = await r.json(); if (j.detail) msg = j.detail; } catch (_) {}
        const e = new Error(msg); e.status = r.status;
        if (r.status < 500) throw e;
        last = e;
      } catch (e) {
        clearTimeout(t);
        if (e.status && e.status < 500) throw e;
        last = e.name === 'AbortError' ? new Error('request timed out') : e;
      }
      if (a < retries) await sleep(600 * 2 ** a);
    }
    throw last;
  }

  let TEST_ACC = 0.935, CLASSES = Object.keys(LABEL);
  api('/api/health').then((h) => { CLASSES = h.classes; $('#api-dot').className = 'dot ok'; $('#api-text').textContent = 'Live · /api/stream'; })
    .catch((e) => { $('#api-dot').className = 'dot bad'; $('#api-text').textContent = e.status === 503 ? 'API degraded' : 'API offline'; });
  

  // ------------------------------------------------------------ state
  let S;
  function resetState() {
    S = { seen: 0, correct: 0, ok: 0, rev: 0, lat: 0, acc: [], tl: [], log: [], counts: {} };
    CLASSES.forEach((c) => (S.counts[c] = 0));
    $('#tl').innerHTML = '<div class="empty"></div>'.repeat(CONFIG.timeline);
    ['#acc', '#conf'].forEach((s) => ($(s).textContent = '–'));
    ['#n', '#ok', '#rev'].forEach((s) => ($(s).textContent = '0'));
    $('#log').innerHTML = '<tr><td colspan="7" class="dim">Waiting for the first window…</td></tr>';
    renderCounts(); drawSpark();
  }

  function bars(el, rows) {
    el.innerHTML = rows.map((r) => `<div class="bar ${r.top ? 'top' : ''}"><span>${esc(r.label)}</span><span class="t"><i style="width:${Math.max(0.6, r.value * 100)}%"></i></span><b>${esc(r.text)}</b></div>`).join('');
  }
  function renderCounts() {
    const mx = Math.max(1, ...Object.values(S.counts));
    bars($('#counts'), CLASSES.map((c) => ({ label: SHORT[c], value: S.counts[c] / mx, text: String(S.counts[c]), top: false })));
    $('#counts').querySelectorAll('.bar').forEach((b, i) => { if (DYNAMIC.has(CLASSES[i])) b.querySelector('i').style.background = '#1D3F04'; });
  }
  function drawSpark() {
    const W = 400, H = 70, lo = 0.6, Y = (v) => H - 5 - (Math.max(lo, v) - lo) / (1 - lo) * (H - 10);
    let g = `<line x1="0" x2="${W}" y1="${Y(TEST_ACC)}" y2="${Y(TEST_ACC)}" stroke="#DDD8CE" stroke-dasharray="4 4"/>`;
    if (S && S.acc.length > 1) {
      const d = S.acc.map((v, i) => (i ? 'L' : 'M') + (i * W / (S.acc.length - 1)).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
      g += `<path d="${d}" fill="none" stroke="#1D3F04" stroke-width="1.8" vector-effect="non-scaling-stroke"/>`;
    }
    $('#spark').innerHTML = g;
    $('#spark-note').textContent = `Dashed line: ${pct(TEST_ACC)} full test-set accuracy (Keras Dense v1).`;
  }

  function show(w, latency) {
    S.seen++; if (w.correct) S.correct++; w.low_confidence ? S.rev++ : S.ok++; S.lat += latency;
    S.counts[w.activity] = (S.counts[w.activity] || 0) + 1;
    S.acc.push(S.correct / S.seen); if (S.acc.length > 150) S.acc.shift();

    $('#cur-lab').textContent = `Current window · Subject ${w.subject} · #${S.seen.toLocaleString('en-US')}`;
    const words = LABEL[w.activity].split(' ');
    $('#pred').innerHTML = words.length > 1 ? `${esc(words[0])}<br><em>${esc(words.slice(1).join(' '))}</em>` : `<em>${esc(words[0])}</em>`;
    $('#conf').textContent = pct(w.confidence, 0);
    $('#arc').setAttribute('stroke-dasharray', `${(w.confidence * 326.7).toFixed(1)} 327`);
    $('#arc').setAttribute('stroke', w.low_confidence ? '#92998E' : '#1D3F04');
    $('#truth').innerHTML = `<span class="lab">True activity</span><b>${esc(LABEL[w.true_activity])}</b>` +
      (w.correct ? '<span class="chip solid">✓ correct</span>' : '<span class="chip ink">✗ misclassified</span>');
    if (w.low_confidence) {
      $('#resp').className = 'resp warn';
      $('#resp-t').textContent = `Low confidence (${pct(w.confidence, 0)}): sent to a supervisor, no machine action.`;
    } else { $('#resp').className = 'resp'; $('#resp-t').textContent = RESPONSE[w.activity]; }
    const probs = Object.entries(w.probabilities).sort((a, b) => b[1] - a[1]);
    bars($('#probs'), probs.map(([c, p], i) => ({ label: LABEL[c], value: p, text: p.toFixed(2), top: i === 0 })));

    $('#subj').textContent = 'Subject ' + w.subject;
    $('#win').textContent = 'Window ' + S.seen.toLocaleString('en-US');
    $('#acc').textContent = pct(S.correct / S.seen);
    $('#n').textContent = S.seen.toLocaleString('en-US');
    $('#ok').textContent = S.ok; $('#rev').textContent = S.rev;
    $('#lat').textContent = (S.lat / S.seen).toFixed(2) + ' ms';
    drawSpark(); renderCounts();

    S.tl.push({ n: S.seen, w }); if (S.tl.length > CONFIG.timeline) S.tl.shift();
    $('#tl').innerHTML = '<div class="empty"></div>'.repeat(CONFIG.timeline - S.tl.length) + S.tl.map(({ w: t }) =>
      `<div class="${DYNAMIC.has(t.activity) ? 'dy' : ''} ${t.correct ? '' : 'x'} ${t.low_confidence ? 'lc' : ''}" title="${LABEL[t.activity]} · ${pct(t.confidence, 0)}${t.correct ? '' : ' · true: ' + LABEL[t.true_activity]}"></div>`).join('');
    $('#tl-a').textContent = '#' + S.tl[0].n.toLocaleString('en-US');
    $('#tl-b').textContent = 'Now · #' + S.seen.toLocaleString('en-US');

    S.log.unshift({ time: new Date().toLocaleTimeString('en-GB', { hour12: false }), n: S.seen, w });
    if (S.log.length > CONFIG.logRows) S.log.pop();
    $('#log').innerHTML = S.log.map(({ time, n, w: x }) => `<tr><td class="m dim">${time}</td><td class="m">#${n}</td><td class="m">${x.subject}</td>
      <td class="hi">${esc(LABEL[x.activity])}</td><td class="${x.correct ? 'dim' : 'hi'}">${esc(LABEL[x.true_activity])}</td>
      <td class="m ${x.low_confidence ? '' : 'g'}">${x.confidence.toFixed(2)}</td>
      <td>${x.low_confidence ? '<span class="chip">Sent to review</span>' : esc(SHORT_RESP[x.activity])}</td></tr>`).join('');
  }

  // ------------------------------------------------------------ loop
  let timer = null, paused = false, speed = 1, busy = false, fails = 0;
  async function tick() {
    if (busy) return; busy = true;
    try {
      const r = await api('/api/stream?n=1');
      fails = 0;
      if (r && r.windows && r.windows.length) show(r.windows[0], r.latency_ms || 0);
    } catch (e) {
      fails++;
      $('#resp').className = 'resp warn';
      if (fails >= 3) { stop(); paused = true; setToggle(); $('#resp-t').textContent = 'Stream paused: the API did not respond (' + e.message + '). Press Resume to try again.'; }
      else $('#resp-t').textContent = 'Connection problem (' + e.message + '); retrying…';
    } finally { busy = false; }
  }
  function start() { stop(); tick(); timer = setInterval(tick, CONFIG.everyMs / speed); }
  function stop() { clearInterval(timer); timer = null; }
  function setToggle() { $('#toggle').innerHTML = paused ? '▶&nbsp; Resume' : '❚❚&nbsp; Pause'; $('#toggle').setAttribute('aria-pressed', String(paused)); }
  $('#toggle').addEventListener('click', () => { paused = !paused; if (!paused) fails = 0; setToggle(); paused ? stop() : start(); });
  $('#next').addEventListener('click', tick);
  $('#reset').addEventListener('click', resetState);
  document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
    speed = Number(b.dataset.speed); if (!paused) start();
  }));
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else if (!paused && !timer) start(); });

  resetState();
  const F = window.__FIRST__;
  if (F && F.metrics && F.metrics.test_accuracy) TEST_ACC = F.metrics.test_accuracy;
  if (F && F.window) show(F.window, F.latency_ms || 0);   // first prediction was made by Python on the server
  start();
})();
"""
