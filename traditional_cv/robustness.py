"""Deterministic test-only degradation study."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .evaluation import evaluate
from .features import read_rgb


DEGRADATION_LEVELS = {
    "gaussian_noise": [0.02, 0.05, 0.10, 0.20],
    "gaussian_blur": [1, 2, 3, 5],
    "motion_blur": [3, 7, 11, 15],
    "jpeg": [75, 50, 25, 10],
}


def degrade(image: np.ndarray, kind: str, severity: float, index: int, seed: int = 42) -> np.ndarray:
    if kind == "gaussian_noise":
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        value = image.astype(np.float32) / 255.0
        return np.clip((value + rng.normal(0, severity, value.shape)) * 255, 0, 255).astype(np.uint8)
    if kind == "gaussian_blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(severity), sigmaY=float(severity))
    if kind == "motion_blur":
        size = int(severity)
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        return cv2.filter2D(image, -1, kernel)
    if kind == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"))
    if kind == "clean":
        return image
    raise ValueError(f"Unknown degradation: {kind}")


def run_robustness(
    model,
    extractor,
    test_manifest,
    class_names: list[str],
    degradation_config: dict[str, list[float]] | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate a frozen model. The caller must pass only held-out test rows."""
    if set(test_manifest["split"]) != {"test"}:
        raise ValueError("Robustness evaluation accepts test rows only")
    config = degradation_config or DEGRADATION_LEVELS
    paths = test_manifest["path"].tolist()
    y = test_manifest["label"].to_numpy()
    records = []
    clean_X = extractor.transform(paths)
    clean = evaluate(model, clean_X, y, class_names)["metrics"]
    records.append({"degradation": "clean", "level": 0, **clean})
    for kind, levels in config.items():
        for level in levels:
            transform = lambda image, index, k=kind, s=level: degrade(image, k, s, index, seed)
            X = extractor.transform(paths, image_transform=transform)
            metrics = evaluate(model, X, y, class_names)["metrics"]
            records.append({"degradation": kind, "level": level, **metrics})
    result = pd.DataFrame(records)
    result["top1_relative_drop"] = 1 - result["top1_accuracy"] / clean["top1_accuracy"]
    result["macro_f1_relative_drop"] = 1 - result["macro_f1"] / clean["macro_f1"]
    return result


def plot_robustness(results: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "robustness_results.csv", index=False)
    for metric in ("top1_accuracy", "macro_f1"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (kind, group) in zip(axes.flat, results[results.degradation != "clean"].groupby("degradation")):
            clean_value = results.loc[results.degradation == "clean", metric].iloc[0]
            x = [0, *group["level"].tolist()]
            y = [clean_value, *group[metric].tolist()]
            ax.plot(range(len(x)), y, marker="o")
            ax.set_xticks(range(len(x)), [str(v) for v in x])
            ax.set(title=kind.replace("_", " ").title(), xlabel="Severity", ylabel=metric)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / f"robustness_{metric}.png", dpi=200)
        plt.close(fig)


def plot_degradation_examples(
    image_path: str | Path,
    output_dir: str | Path,
    degradation_config: dict[str, list[float]] | None = None,
    *,
    seed: int = 42,
) -> Path:
    """Create a report-ready grid of the exact test-time corruptions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = degradation_config or DEGRADATION_LEVELS
    image = read_rgb(image_path)
    rows, columns = len(config), 5
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), squeeze=False)
    for row, (kind, levels) in enumerate(config.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{kind.replace('_', ' ').title()}\nClean")
        axes[row, 0].axis("off")
        for column, level in enumerate(levels, 1):
            axes[row, column].imshow(degrade(image, kind, level, index=0, seed=seed))
            axes[row, column].set_title(f"Severity: {level}")
            axes[row, column].axis("off")
    fig.suptitle("Held-out test image degradation examples", fontsize=16)
    fig.tight_layout()
    path = output / "degradation_examples.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
