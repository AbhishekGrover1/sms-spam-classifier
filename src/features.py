"""Text cleaning and feature building for the SMS spam classifier.

Author: Abhishek Grover
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(com|net|org|info|biz|co\.uk)\b\S*)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY = re.compile(r"[£$€]\s?\d[\d,.]*")
NUMBER = re.compile(r"\d+")
LONG_NUMBER = re.compile(r"\d{9,}")


def _number_token(match):
    # a phone number, a short code and a plain number mean very different things
    size = len(match.group())
    if size >= 9:
        return " phonetoken "
    if size >= 5:
        return " codetoken "
    return " numtoken "


def normalize_text(text):
    """Lowercase the text and swap urls, money and numbers for placeholder words.

    Spammers almost never reuse the exact same phone number or price, but the
    shape of it (a long number, a short code, a pound amount) shows up again
    and again, so the shape is what the model should learn.
    """
    text = text.lower()
    text = URL.sub(" urltoken ", text)
    text = EMAIL.sub(" emailtoken ", text)
    text = MONEY.sub(" moneytoken ", text)
    return NUMBER.sub(_number_token, text)


class MessageStats(BaseEstimator, TransformerMixin):
    """A handful of numbers about how a message looks (length, capitals, digits...)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            size = max(len(text), 1)
            rows.append([
                np.log1p(len(text)),
                np.log1p(len(text.split())),
                sum(ch.isdigit() for ch in text) / size,
                sum(ch.isupper() for ch in text) / size,
                text.count("!"),
                len(re.findall(r"[£$€]", text)),
                1 if URL.search(text.lower()) else 0,
                1 if LONG_NUMBER.search(text) else 0,
            ])
        return np.array(rows, dtype=float)


def build_features():
    """Word tf-idf + character tf-idf + the message stats, stacked side by side."""
    return FeatureUnion(
        [
            ("words", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2),
                                      min_df=2, sublinear_tf=True)),
            # character n-grams catch misspellings and txt-speak ("txt", "u", "2day")
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                      min_df=3, sublinear_tf=True)),
            ("stats", Pipeline([("raw", MessageStats()), ("scale", MaxAbsScaler())])),
        ],
        # the stats are only a hint, so they get less say than the text features
        transformer_weights={"stats": 0.3},
    )
