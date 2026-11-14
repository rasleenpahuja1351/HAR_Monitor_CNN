"""
HAR Monitor API (Group EIC-16) — FastAPI backend for Vercel.

Serves the prediction API under /api and, locally, the website in public/.
The Keras model is exported to NumPy weights (src/export_model.py), so this
function needs only FastAPI + NumPy.

Hardening:
  * every input is validated (shape, finite numbers, value range, row limits)
  * oversized request bodies are refused before parsing (413)
  * model files are loaded lazily and cached; if they are missing the API
    answers 503 with a clear message instead of crashing
  * every unexpected error becomes a JSON 500 (no stack traces leak)
  * consistent JSON error format: {"detail": "...", "error": "<code>"}
  * security + cache headers on every response

Run locally:   uvicorn app:app --reload      -> http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.inference import HARModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
PUBLIC_DIR = ROOT / "public"

VERSION = "2.0.0"
GROUP_ID = "EIC-16"
N_FEATURES = 561
MAX_BATCH_ROWS = 1000            # each request stays far below Vercel's 4.5 MB body limit
MAX_BODY_BYTES = 4_000_000       # refuse larger bodies before parsing them
MAX_ABS_VALUE = 1e4              # UCI HAR features lie in [-1, 1]; anything huge is a mistake
LOW_CONFIDENCE = 0.70            # below this a prediction is sent for human review
MAX_STREAM = 60

log = logging.getLogger("har")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


class ModelUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Model & data (loaded once per function instance, then cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_model() -> HARModel:
    return HARModel(MODEL_DIR / "har_mlp.npz")


def get_model() -> HARModel:
    try:
        return _load_model()
    except Exception as e:  # missing or corrupt file
        log.error("model load failed: %s", e)
        raise ModelUnavailable("The model is not available on this deployment (model/har_mlp.npz).") from e


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    return json.loads((MODEL_DIR / "metrics.json").read_text())


def get_metrics() -> dict:
    try:
        return _load_metrics()
    except Exception as e:
        log.error("metrics load failed: %s", e)
        raise ModelUnavailable("Metrics are not available on this deployment (model/metrics.json).") from e


@lru_cache(maxsize=1)
def _load_samples():
    with np.load(MODEL_DIR / "samples.npz", allow_pickle=False) as d:
        return d["X"].astype(np.float32), d["y"].astype(int), d["subject"].astype(int)


def get_samples():
    try:
        return _load_samples()
    except Exception as e:
        log.error("samples load failed: %s", e)
        raise ModelUnavailable("Live samples are not available on this deployment (model/samples.npz).") from e


def describe(probs: np.ndarray, classes: List[str]) -> dict:
    i = int(np.argmax(probs))
    conf = float(probs[i])
    return {
        "activity": classes[i],
        "confidence": round(conf, 4),
        "low_confidence": bool(conf < LOW_CONFIDENCE),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #
def _check_row(row: List[float], where: str = "features") -> List[float]:
    if len(row) != N_FEATURES:
        raise ValueError(f"{where}: expected {N_FEATURES} values, got {len(row)}")
    for v in row:
        if not math.isfinite(v):
            raise ValueError(f"{where}: values must be finite numbers (no NaN or Infinity)")
        if abs(v) > MAX_ABS_VALUE:
            raise ValueError(f"{where}: value {v} is out of range (|x| must be <= {MAX_ABS_VALUE:g})")
    return row


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: List[float] = Field(..., description="561 pre-computed UCI HAR features, in dataset column order")

    @field_validator("features")
    @classmethod
    def _valid(cls, v):
        return _check_row(v)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: List[List[float]] = Field(..., description=f"1 to {MAX_BATCH_ROWS} rows of 561 features")

    @field_validator("rows")
    @classmethod
    def _valid(cls, v):
        if not v:
            raise ValueError("rows must not be empty")
        if len(v) > MAX_BATCH_ROWS:
            raise ValueError(f"at most {MAX_BATCH_ROWS} rows per request; split larger files into chunks")
        for i, r in enumerate(v):
            _check_row(r, f"row {i + 1}")
        return v


# --------------------------------------------------------------------------- #
# App, middleware and error handling
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="HAR Monitor API",
    description="Human activity recognition (UCI HAR, 6 classes) for factory automation · Group EIC-16.",
    version=VERSION,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


def _error(status: int, detail: str, code: str, request_id: Optional[str] = None) -> JSONResponse:
    body = {"detail": detail, "error": code}
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status, content=body)


@app.middleware("http")
async def guard(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    request.state.rid = rid
    if request.url.path.startswith("/api") and request.method in ("POST", "PUT", "PATCH"):
        size = request.headers.get("content-length")
        if size is not None:
            try:
                if int(size) > MAX_BODY_BYTES:
                    return _error(413, f"Request body too large (max {MAX_BODY_BYTES // 1_000_000} MB). "
                                       f"Send at most {MAX_BATCH_ROWS} rows per request.", "payload_too_large", rid)
            except ValueError:
                return _error(400, "Invalid Content-Length header.", "bad_request", rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # last line of defence: never return a raw crash
        log.exception("unhandled error [%s] %s %s", rid, request.method, request.url.path)
        response = _error(500, "Internal server error. Please try again.", "internal_error", rid)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    if request.url.path.startswith("/api") and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        msg = str(e.get("msg", "invalid value")).replace("Value error, ", "")
        msgs.append(msg if (not loc or msg.startswith(loc.split(".")[-1])) else f"{loc}: {msg}")
    return _error(422, "; ".join(msgs) or "Invalid request.", "validation_error", getattr(request.state, "rid", None))


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException):
    code = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, str(exc.detail), code, getattr(request.state, "rid", None))


@app.exception_handler(ModelUnavailable)
async def on_model_unavailable(request: Request, exc: ModelUnavailable):
    return _error(503, str(exc), "model_unavailable", getattr(request.state, "rid", None))


@app.exception_handler(Exception)
async def on_unexpected(request: Request, exc: Exception):
    rid = getattr(request.state, "rid", None)
    log.exception("unhandled error [%s]", rid)
    return _error(500, "Internal server error. Please try again.", "internal_error", rid)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    """Liveness + readiness. Never raises: reports 'degraded' if the model cannot be loaded."""
    try:
        m = get_model()
        return {"status": "ok", "version": VERSION, "group": GROUP_ID, "model": "keras-dense-v1",
                "classes": m.classes, "features": len(m.features)}
    except ModelUnavailable as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "version": VERSION,
                                                      "detail": str(e), "error": "model_unavailable"})


@app.get("/api/metrics")
def metrics():
    return JSONResponse(content=get_metrics(), headers={"Cache-Control": "public, max-age=300, s-maxage=3600"})


@app.get("/api/features")
def features():
    return JSONResponse(content={"count": N_FEATURES, "features": get_model().features},
                        headers={"Cache-Control": "public, max-age=3600, s-maxage=86400"})


@app.post("/api/predict")
def predict(req: PredictRequest):
    model = get_model()
    t0 = time.perf_counter()
    probs = model.predict_proba(np.asarray([req.features], dtype=np.float32))[0]
    out = describe(probs, model.classes)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return out


@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest):
    model = get_model()
    t0 = time.perf_counter()
    probs = model.predict_proba(np.asarray(req.rows, dtype=np.float32))
    idx = probs.argmax(1)
    conf = probs.max(1)
    return {
        "count": len(req.rows),
        "predictions": [model.classes[i] for i in idx],
        "confidence": [round(float(p), 4) for p in conf],
        "low_confidence": int((conf < LOW_CONFIDENCE).sum()),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


@app.get("/api/stream")
def stream(n: int = Query(1, ge=1, le=MAX_STREAM), seed: Optional[int] = Query(None, ge=0, le=2**32 - 1)):
    """Classify real held-out test windows, as a stand-in for a live sensor feed."""
    model = get_model()
    X, y, subj = get_samples()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=n)
    t0 = time.perf_counter()
    probs = model.predict_proba(X[idx])
    latency = (time.perf_counter() - t0) * 1000 / n
    items = []
    for k, i in enumerate(idx):
        d = describe(probs[k], model.classes)
        true = model.classes[int(y[i])]
        d.update({"true_activity": true, "subject": int(subj[i]), "correct": d["activity"] == true})
        items.append(d)
    return {"windows": items, "latency_ms": round(latency, 3)}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
def api_not_found(path: str):
    raise HTTPException(status_code=404, detail=f"Unknown API route: /api/{path}")


# Pipeline & live prediction page, generated in Python (live_page.py) and served at /live
from live_page import create_router  # noqa: E402

app.include_router(create_router(get_model, get_samples, get_metrics, describe,
                                 threshold=LOW_CONFIDENCE, group=GROUP_ID))


# Website. On Vercel the public/ files are served by the CDN and are not inside
# the function, so mount them only when the folder exists (local development).
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
