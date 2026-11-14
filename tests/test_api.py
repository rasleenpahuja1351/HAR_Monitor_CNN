"""API tests: normal use plus the failure cases the backend must survive."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app, get_samples

client = TestClient(app, raise_server_exceptions=False)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and len(body["classes"]) == 6 and body["features"] == 561
    assert body["group"] == "EIC-16"
    assert "X-Request-ID" in r.headers and r.headers["X-Content-Type-Options"] == "nosniff"


def test_metrics_cached_and_valid():
    r = client.get("/api/metrics")
    assert r.status_code == 200 and "max-age" in r.headers["Cache-Control"]
    m = r.json()
    assert 0.9 < m["keras"]["test_accuracy"] < 1 and len(m["keras"]["confusion"]) == 6


def test_predict_real_windows_mostly_correct():
    X, y, _ = get_samples()
    classes = client.get("/api/health").json()["classes"]
    correct = 0
    for i in range(60):
        r = client.post("/api/predict", json={"features": X[i].tolist()})
        assert r.status_code == 200
        body = r.json()
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
        correct += body["activity"] == classes[int(y[i])]
    assert correct / 60 > 0.8


@pytest.mark.parametrize("payload", [
    {"features": [0.0] * 10},                      # wrong length
    {"features": [0.0] * 560 + ["abc"]},           # not a number
    {"features": [0.0] * 560 + [1e9]},             # absurd value
    {"features": None},                            # null
    {},                                            # missing field
    {"features": [0.0] * 561, "extra": 1},         # unexpected field
    [1, 2, 3],                                     # wrong JSON type
])
def test_predict_rejects_bad_input_with_422(payload):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error" and isinstance(r.json()["detail"], str)


def test_predict_rejects_nan_and_malformed_json():
    body = '{"features": [' + ",".join(["0"] * 560) + ', NaN]}'
    r = client.post("/api/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    r = client.post("/api/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_ok_limits_and_empty():
    X, _, _ = get_samples()
    r = client.post("/api/predict/batch", json={"rows": X[:50].tolist()})
    assert r.status_code == 200 and r.json()["count"] == 50 and len(r.json()["predictions"]) == 50
    assert client.post("/api/predict/batch", json={"rows": []}).status_code == 422
    assert client.post("/api/predict/batch", json={"rows": [[0.0] * 561, [0.0] * 3]}).status_code == 422


def test_oversized_body_is_413():
    big = np.zeros((1500, 561)).tolist()
    r = client.post("/api/predict/batch", json={"rows": big})
    assert r.status_code in (413, 422)
    assert r.json()["error"] in ("payload_too_large", "validation_error")


def test_stream_and_bounds():
    r = client.get("/api/stream?n=5&seed=1")
    assert r.status_code == 200 and len(r.json()["windows"]) == 5
    assert client.get("/api/stream?n=0").status_code == 422
    assert client.get("/api/stream?n=999").status_code == 422
    assert client.get("/api/stream?n=abc").status_code == 422


def test_unknown_routes_and_methods():
    assert client.get("/api/nope").json()["error"] == "not_found"
    assert client.post("/api/nope").status_code == 404
    assert client.get("/api/predict").status_code in (404, 405)   # wrong method never crashes


def test_missing_model_returns_503_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_DIR", tmp_path)
    app_module._load_model.cache_clear()
    try:
        r = client.get("/api/health")
        assert r.status_code == 503 and r.json()["status"] == "degraded"
        r = client.post("/api/predict", json={"features": [0.0] * 561})
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
    finally:
        monkeypatch.undo()
        app_module._load_model.cache_clear()
    assert client.get("/api/health").status_code == 200


def test_unexpected_error_is_json_500(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(app_module, "describe", boom)
    r = client.post("/api/predict", json={"features": [0.0] * 561})
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "simulated" not in r.text


def test_website_served_locally():
    r = client.get("/")
    assert r.status_code == 200 and "HAR Monitor" in r.text and "EIC-16" in r.text
