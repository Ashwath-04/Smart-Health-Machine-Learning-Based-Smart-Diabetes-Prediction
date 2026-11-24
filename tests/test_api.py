"""FastAPI endpoint tests (uses the tiny session model, not the full artifact)."""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture
def client(predictor, monkeypatch):
    monkeypatch.setattr(api_main, "load_predictor", lambda: None)
    with TestClient(api_main.app) as c:
        api_main.STATE.update(predictor=predictor, load_error=None)
        yield c


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(api_main, "load_predictor", lambda: None)
    with TestClient(api_main.app) as c:
        api_main.STATE.update(predictor=None, load_error="no artifact")
        yield c


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
    assert client.get("/favicon.ico").status_code == 204


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == "test-0"


def test_health_degraded_and_503(client_without_model, valid_payload):
    assert client_without_model.get("/health").json()["status"] == "degraded"
    assert client_without_model.post("/predict", json=valid_payload).status_code == 503
    assert client_without_model.get("/model-info").status_code == 503


def test_predict_ok(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    for key in ["prediction", "probability", "threshold", "model_version", "explanation", "disclaimer"]:
        assert key in body
    assert body["threshold_mode"] == "balanced"
    assert len(body["explanation"]["contributions"]) == 8


def test_predict_mode_and_no_explain(client, valid_payload):
    body = client.post("/predict", params={"mode": "high_sensitivity", "explain": False}, json=valid_payload).json()
    assert body["threshold"] == 0.3 and body["explanation"] is None


@pytest.mark.parametrize("mutation", [
    {"glucose": "abc"}, {"bmi": -3}, {"glucose": 0}, {"age": 500}, {"unknown_field": 1}, {"pregnancies": 2.5},
])
def test_predict_rejects_malformed(client, valid_payload, mutation):
    r = client.post("/predict", json={**valid_payload, **mutation})
    assert r.status_code == 422
    assert r.json()["detail"] == "Invalid input"


def test_predict_missing_required_field(client, valid_payload):
    payload = dict(valid_payload)
    payload.pop("age")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and any(e["field"] == "age" for e in r.json()["errors"])


def test_predict_invalid_body_and_mode(client, valid_payload):
    assert client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/predict", params={"mode": "yolo"}, json=valid_payload).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "test_stacking" and "disclaimer" in body


def test_parse_and_predict_text(client):
    r = client.post("/parse", json={"text": "45-year-old with glucose 160 and BMI 31"})
    assert r.status_code == 200
    parsed = r.json()
    assert parsed["extracted"]["glucose"] == 160 and "pregnancies" in parsed["filled_with_training_median"]
    r = client.post("/predict/text", json={"text": "45-year-old with glucose 160 and BMI 31"})
    assert r.status_code == 200 and "result" in r.json()


def test_predict_text_missing_without_fill(client):
    r = client.post("/predict/text", json={"text": "glucose 160", "fill_missing_with_median": False})
    assert r.status_code == 422 and "age" in r.json()["detail"]["missing"]


def test_report(client, valid_payload):
    r = client.post("/report", json=valid_payload)
    assert r.status_code == 200 and "not a substitute for professional medical evaluation" in r.text
