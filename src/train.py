"""Train the SMS spam classifier.

Run it from the project root:

    python -m src.train

Steps: clean the data, compare a few models with cross-validation, pick a
decision threshold, score the winner on a held-out test set, then save the
model together with its metrics.

Author: Abhishek Grover
"""
import json
import tempfile
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from src.features import build_features

# of the messages we call spam, at least this share has to really be spam
TARGET_PRECISION = 0.98


def load_data():
    # spam.csv is latin-1 encoded: label in the first column, text in the second.
    # The three unnamed columns after those are almost empty, so they are skipped.
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1],
                     names=["label", "message"], header=0)
    df["message"] = df["message"].str.strip()
    rows_in_file = len(df)

    # exact duplicates would end up in both train and test and flatter the scores
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df, rows_in_file


def candidates():
    """The models to compare, each with the settings to search over."""
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {"clf__alpha": [0.03, 0.1, 0.3]},
        ),
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [10, 30, 100], "clf__class_weight": [None, "balanced"]},
        ),
        "Linear SVM": (
            CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3),
            {"clf__estimator__C": [0.3, 1, 3]},
        ),
    }


def pick_threshold(pipe, X_train, y_train, cv):
    """Lowest cutoff whose out-of-fold precision still meets TARGET_PRECISION.

    A lower cutoff catches more spam, a higher one wrongly flags fewer real
    messages. Only training data is used here, the test set stays untouched.
    """
    scores = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, _, thresholds = precision_recall_curve(y_train, scores)
    good = np.where(precision[:-1] >= TARGET_PRECISION)[0]
    if len(good) == 0:
        return 0.5
    return float(thresholds[good[0]])


def train():
    started = time.time()
    df, rows_in_file = load_data()
    print(f"{rows_in_file} rows in file, {len(df)} unique messages, {df['is_spam'].sum()} spam")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["is_spam"], test_size=0.2,
        stratify=df["is_spam"], random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison, searches = {}, {}
    # the feature step is identical for every setting, so cache it between fits
    with tempfile.TemporaryDirectory() as cache_dir:
        for name, (estimator, grid) in candidates().items():
            pipe = Pipeline([("features", build_features()), ("clf", estimator)], memory=cache_dir)
            search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv)
            search.fit(X_train, y_train)

            test_scores = search.predict_proba(X_test)[:, 1]
            comparison[name] = {
                "cv_pr_auc": round(search.best_score_, 4),
                "test_pr_auc": round(average_precision_score(y_test, test_scores), 4),
                "test_roc_auc": round(roc_auc_score(y_test, test_scores), 4),
                "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            }
            searches[name] = search
            print(f"{name:20s} cv PR-AUC {search.best_score_:.4f}  {search.best_params_}")

        # pick by cross-validation score, never by the test set
        best_name = max(comparison, key=lambda n: comparison[n]["cv_pr_auc"])
        model = searches[best_name].best_estimator_
        threshold = pick_threshold(model, X_train, y_train, cv)

    model.set_params(memory=None)

    scores = model.predict_proba(X_test)[:, 1]
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    metrics = {
        "model": best_name,
        "threshold": round(threshold, 4),
        "target_precision": TARGET_PRECISION,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sklearn_version": sklearn.__version__,
        "data": {
            "rows_in_file": rows_in_file,
            "unique_messages": len(df),
            "spam": int(df["is_spam"].sum()),
            "ham": int((df["is_spam"] == 0).sum()),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "test": {
            "accuracy": round(accuracy_score(y_test, predicted), 4),
            "precision": round(precision_score(y_test, predicted), 4),
            "recall": round(recall_score(y_test, predicted), 4),
            "f1": round(f1_score(y_test, predicted), 4),
            "roc_auc": round(roc_auc_score(y_test, scores), 4),
            "pr_auc": round(average_precision_score(y_test, scores), 4),
            "always_ham_accuracy": round(1 - y_test.mean(), 4),
            "confusion_matrix": {
                "true_ham": int(tn), "false_spam": int(fp),
                "missed_spam": int(fn), "caught_spam": int(tp),
            },
        },
        "comparison": comparison,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    artifact = {
        "pipeline": model,
        "threshold": threshold,
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    t = metrics["test"]
    print(f"\nbest model: {best_name}, threshold {threshold:.3f}")
    print(f"test precision {t['precision']}  recall {t['recall']}  f1 {t['f1']}  "
          f"roc-auc {t['roc_auc']}  pr-auc {t['pr_auc']}")
    print(f"confusion matrix {t['confusion_matrix']}")
    print(f"done in {time.time() - started:.0f}s, saved to {MODEL_PATH.name} and {METRICS_PATH.name}")
    return artifact


if __name__ == "__main__":
    train()
