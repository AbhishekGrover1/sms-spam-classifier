import pytest
from fastapi.testclient import TestClient

from app.main import app

SPAM = (
    "WINNER!! You have been selected to receive a £900 prize. "
    "Call 09061701234 now or text CLAIM to 81010 to collect. T&Cs apply."
)
HAM = "Running late, the bus is stuck near the station. Start without me, I'll be there in 20 mins."


@pytest.fixture(scope="module")
def client():
    # the with-block runs the startup code that loads the model
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Abhishek Grover" in response.text


def test_spam_is_flagged(client):
    body = client.post("/predict", json={"message": SPAM}).json()
    assert body["label"] == "spam"
    assert body["spam_probability"] > 0.9
    assert body["signals"], "a clear spam message should come with explanation words"


def test_normal_message_passes(client):
    body = client.post("/predict", json={"message": HAM}).json()
    assert body["label"] == "ham"
    assert body["spam_probability"] < 0.2


def test_blank_message_is_rejected(client):
    assert client.post("/predict", json={"message": "   "}).status_code == 422


def test_overlong_message_is_rejected(client):
    assert client.post("/predict", json={"message": "a" * 1001}).status_code == 422


def test_batch(client):
    response = client.post("/predict/batch", json={"messages": [SPAM, HAM]})
    labels = [item["label"] for item in response.json()["results"]]
    assert labels == ["spam", "ham"]


def test_model_info_has_test_metrics(client):
    body = client.get("/model").json()
    assert 0 < body["test"]["precision"] <= 1
    assert body["threshold"] > 0
