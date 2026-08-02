"""Classical classifiers with reproducible defaults."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import ClassifierConfig


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ClassifierConfig | dict[str, Any],
):
    if isinstance(config, dict):
        config = ClassifierConfig(**config)
    started = time.perf_counter()
    if config.kind == "linear_svc":
        params = {"C": 1.0, "dual": "auto", "max_iter": 10_000}
        params.update(config.params)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(random_state=config.seed, **params)),
        ])
    elif config.kind == "stochastic_svm":
        # Hinge-loss SGD optimises a linear SVM objective without liblinear's
        # prohibitively slow 500 one-vs-rest fits on this dataset.
        params = {
            "loss": "hinge",
            "alpha": 1e-4,
            "max_iter": 2_000,
            "tol": 1e-3,
            "average": True,
            "n_jobs": -1,
        }
        params.update(config.params)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SGDClassifier(random_state=config.seed, **params)),
        ])
    elif config.kind == "random_forest":
        params = {
            "n_estimators": 300,
            "max_features": "sqrt",
            "max_depth": None,
            "n_jobs": -1,
        }
        params.update(config.params)
        model = RandomForestClassifier(random_state=config.seed, **params)
    else:
        raise ValueError(f"Unknown classifier kind: {config.kind}")
    model.fit(X_train, y_train)
    model.fit_seconds_ = time.perf_counter() - started
    model.experiment_config_ = config.to_dict()
    return model


def class_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        raise TypeError("Classifier must expose decision_function or predict_proba")
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("Multiclass score matrix is required for top-5 accuracy")
    return scores


def model_classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        return np.asarray(model.named_steps["classifier"].classes_)
    raise TypeError("Unable to find classifier classes")
