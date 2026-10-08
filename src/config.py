"""Paths and constants shared by the training script and the API.

Author: Abhishek Grover
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

RANDOM_STATE = 42
