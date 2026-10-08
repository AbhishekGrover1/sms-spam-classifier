"""FastAPI app: serves the classifier page and a small JSON API.

Author: Abhishek Grover
"""
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import BatchIn, BatchOut, MessageIn, Prediction
from src.config import ROOT
from src.predictor import SpamClassifier

STATIC_DIR = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once at startup instead of on every request
    app.state.classifier = SpamClassifier()
    yield


app = FastAPI(
    title="SMS Spam Classifier",
    description="Tells spam text messages from normal ones, and shows which words decided it.",
    version="1.0.0",
    contact={"name": "Abhishek Grover"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_classifier(request: Request) -> SpamClassifier:
    return request.app.state.classifier


Classifier = Annotated[SpamClassifier, Depends(get_classifier)]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(classifier: Classifier):
    return {"status": "ok", "model": classifier.model_name}


@app.get("/model")
def model_info(classifier: Classifier):
    """Which model is running and how it scored on the held-out test set."""
    return classifier.metrics


@app.post("/predict", response_model=Prediction)
def predict(body: MessageIn, classifier: Classifier):
    return classifier.predict(body.message)


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(body: BatchIn, classifier: Classifier):
    return {"results": classifier.predict_many(body.messages)}
