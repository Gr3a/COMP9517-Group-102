"""Required metrics, predictions, confusion analysis, and report figures."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .models import class_scores, model_classes


def _top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean(np.any(top_labels == y_true[:, None], axis=1)))


def evaluate(model, X: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    started = time.perf_counter()
    predictions = np.asarray(model.predict(X))
    scores = class_scores(model, X)
    seconds = time.perf_counter() - started
    classes = model_classes(model)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=np.arange(len(class_names)), zero_division=0
    )
    metrics = {
        "top1_accuracy": float(accuracy_score(y, predictions)),
        "top5_accuracy": _top_k_accuracy(y, scores, classes, 5),
        "overall_accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "inference_seconds": seconds,
        "images_per_second": len(y) / max(seconds, 1e-12),
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "scores": scores,
        "classes": classes,
        "confusion_matrix": confusion_matrix(y, predictions, labels=np.arange(len(class_names))),
        "per_class": pd.DataFrame({
            "label": np.arange(len(class_names)),
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }),
    }


def most_confused_pairs(matrix: np.ndarray, class_names: list[str], n: int = 10) -> pd.DataFrame:
    work = matrix.copy()
    np.fill_diagonal(work, 0)
    rows = []
    for flat in np.argsort(work.ravel())[::-1]:
        true, predicted = np.unravel_index(flat, work.shape)
        if work[true, predicted] == 0 or len(rows) >= n:
            break
        rows.append({
            "true_label": true,
            "true_class": class_names[true],
            "predicted_label": predicted,
            "predicted_class": class_names[predicted],
            "count": int(work[true, predicted]),
        })
    return pd.DataFrame(rows)


def save_evaluation(result: dict, class_names: list[str], output_dir: str | Path, prefix: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{prefix}_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / f"{prefix}_predictions_scores.npz",
        predictions=result["predictions"],
        scores=result["scores"],
        classes=result["classes"],
    )
    result["per_class"].to_csv(output / f"{prefix}_per_class.csv", index=False)
    pairs = most_confused_pairs(result["confusion_matrix"], class_names)
    pairs.to_csv(output / f"{prefix}_confused_pairs.csv", index=False)

    matrix = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="mako", xticklabels=False, yticklabels=False, ax=ax)
    ax.set(title=f"{prefix}: all 500 classes", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_confusion_all.png", dpi=200)
    plt.close(fig)

    selected = sorted(set(pairs["true_label"]) | set(pairs["predicted_label"])) if len(pairs) else []
    if selected:
        subset = matrix[np.ix_(selected, selected)]
        labels = [class_names[i].split("_")[-2] + " " + class_names[i].split("_")[-1] for i in selected]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(subset, annot=True, fmt="d", cmap="mako", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(title="Most-confused class subset", xlabel="Predicted", ylabel="True")
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_confusion_subset.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(result["per_class"]["f1"], bins=20, ax=ax)
    ax.set(title="Per-class F1 distribution", xlabel="F1")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_per_class_f1.png", dpi=200)
    plt.close(fig)


def save_example_gallery(result: dict, manifest, class_names: list[str], output_dir: str | Path, n_each: int = 6) -> None:
    """Save reproducible high-confidence successes and failures for discussion."""
    output = Path(output_dir)
    predictions = result["predictions"]
    scores = result["scores"]
    y = manifest["label"].to_numpy()
    confidence = scores.max(axis=1)
    correct = np.flatnonzero(predictions == y)
    wrong = np.flatnonzero(predictions != y)
    chosen_correct = correct[np.argsort(confidence[correct])[::-1][:n_each]]
    chosen_wrong = wrong[np.argsort(confidence[wrong])[::-1][:n_each]]
    chosen = np.concatenate([chosen_correct, chosen_wrong])
    records = []
    fig, axes = plt.subplots(2, n_each, figsize=(3 * n_each, 7), squeeze=False)
    for ax, index in zip(axes.flat, chosen):
        path = Path(manifest.iloc[index]["path"])
        ax.imshow(plt.imread(path))
        true_name = class_names[y[index]].split("_")[-2:]
        pred_name = class_names[predictions[index]].split("_")[-2:]
        ax.set_title(f"True: {' '.join(true_name)}\nPred: {' '.join(pred_name)}", fontsize=8)
        ax.axis("off")
        records.append({
            "path": str(path), "true": class_names[y[index]],
            "predicted": class_names[predictions[index]],
            "score": float(confidence[index]), "correct": bool(predictions[index] == y[index]),
        })
    fig.suptitle("High-confidence successes (top) and failures (bottom)")
    fig.tight_layout()
    fig.savefig(output / "representative_examples.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / "representative_examples.csv", index=False)
