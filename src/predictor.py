"""Loads the trained model and turns a message into a prediction.

Author: Abhishek Grover
"""
import json

import joblib
import numpy as np
import sklearn

from src.config import METRICS_PATH, MODEL_PATH

# the explanation re-scores the message once per word, so cap how many it tries
MAX_EXPLAIN_WORDS = 60
PUNCTUATION = ".,!?;:\"'()[]{}<>*~_-…"

# common words matter to the model but say little to a reader, so they are not listed
FILLER = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "and", "or", "is",
    "are", "am", "was", "be", "it", "this", "that", "has", "have", "had", "me", "my",
    "i", "i'm", "i'll", "you", "your", "we", "with", "as", "by", "can", "will", "so",
    "if", "do",
}


def load_artifact():
    """Load the saved model, or train a new one if it is missing or unreadable.

    A pickle only loads safely with the scikit-learn version that wrote it, so a
    version mismatch also triggers a quick retrain (a few seconds of work).
    """
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if artifact["sklearn_version"] == sklearn.__version__:
                return artifact
        except Exception:
            pass  # fall through and rebuild the model below

    print("No usable model found, training one from data/spam.csv ...")
    from src.train import train
    return train()


def log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SpamClassifier:
    def __init__(self):
        artifact = load_artifact()
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact["threshold"]
        self.model_name = artifact["model_name"]
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    def spam_probabilities(self, messages):
        return self.pipeline.predict_proba(list(messages))[:, 1]

    def label_for(self, probability):
        return "spam" if probability >= self.threshold else "ham"

    def predict(self, message):
        probability = float(self.spam_probabilities([message])[0])
        label = self.label_for(probability)
        return {
            "label": label,
            "spam_probability": round(probability, 4),
            "threshold": round(self.threshold, 4),
            "signals": self.explain(message, probability, label),
        }

    def predict_many(self, messages):
        results = []
        for probability in self.spam_probabilities(messages):
            results.append({
                "label": self.label_for(probability),
                "spam_probability": round(float(probability), 4),
            })
        return results

    def explain(self, message, probability, label, top=6):
        """Find the words that pushed the score toward the predicted label.

        Every word is dropped in turn and the message is scored again. How far the
        log-odds move is that word's weight. This only needs predict_proba, so it
        works the same for any model in the pipeline.
        """
        words = message.split()
        tries = min(len(words), MAX_EXPLAIN_WORDS)
        if tries == 0:
            return []

        without_word = [" ".join(words[:i] + words[i + 1:]) for i in range(tries)]
        new_scores = self.spam_probabilities(without_word)

        # for a spam verdict we want words that raised the score, for ham the ones that lowered it
        direction = 1 if label == "spam" else -1
        weights = {}
        for word, new_score in zip(words, new_scores):
            key = word.lower().strip(PUNCTUATION)
            if key and key not in FILLER:
                push = direction * (log_odds(probability) - log_odds(new_score))
                weights[key] = weights.get(key, 0.0) + float(push)

        strongest = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [
            {"word": word, "weight": round(weight, 2)}
            for word, weight in strongest[:top]
            if weight >= 0.1
        ]
