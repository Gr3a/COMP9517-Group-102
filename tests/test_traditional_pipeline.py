from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from traditional_cv.config import ClassifierConfig, FeatureConfig
from traditional_cv.data import build_manifest
from traditional_cv.evaluation import evaluate
from traditional_cv.features import HandcraftedFeatureExtractor, resize_with_padding
from traditional_cv.models import fit_classifier
from traditional_cv.robustness import degrade, run_robustness


def _dataset(root: Path, counts=(4, 2, 2), classes=6):
    rng = np.random.default_rng(42)
    for split, count in zip(("train", "val", "test"), counts):
        for label in range(classes):
            directory = root / split / f"class_{label:02d}"
            directory.mkdir(parents=True)
            colour = np.zeros((48 + label, 64, 3), dtype=np.uint8)
            colour[..., label % 3] = 80 + label * 20
            for index in range(count):
                image = np.clip(colour + rng.integers(0, 10, colour.shape, dtype=np.uint8), 0, 255)
                cv2.imwrite(str(directory / f"{split}_{label}_{index}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_manifest_and_features_end_to_end(tmp_path):
    _dataset(tmp_path)
    manifest = build_manifest(tmp_path, expected_classes=6, validate_counts=False)
    assert manifest.frame.groupby("split").size().to_dict() == {"test": 12, "train": 24, "val": 12}
    train, val = manifest.split("train"), manifest.split("val")
    config = FeatureConfig(names=("hsv", "lbp", "hog"), image_size=64, hog_pixels_per_cell=(8, 8))
    extractor = HandcraftedFeatureExtractor(config).fit(train.path)
    X_train, X_val = extractor.transform(train.path), extractor.transform(val.path)
    assert X_train.shape[0] == 24 and np.isfinite(X_train).all()
    model = fit_classifier(X_train, train.label.to_numpy(), ClassifierConfig(params={"C": 0.1}))
    result = evaluate(model, X_val, val.label.to_numpy(), manifest.class_names)
    assert set(("top1_accuracy", "top5_accuracy", "macro_f1")) <= result["metrics"].keys()
    assert result["scores"].shape == (12, 6)

    fast_model = fit_classifier(
        X_train, train.label.to_numpy(), ClassifierConfig(kind="stochastic_svm")
    )
    fast_result = evaluate(fast_model, X_val, val.label.to_numpy(), manifest.class_names)
    assert fast_result["scores"].shape == (12, 6)
    assert fast_model.named_steps["classifier"].loss == "hinge"


def test_padding_and_degradations_are_deterministic():
    image = np.full((40, 80, 3), 120, dtype=np.uint8)
    padded = resize_with_padding(image, 64)
    assert padded.shape == (64, 64, 3)
    first = degrade(image, "gaussian_noise", 0.1, index=3, seed=42)
    second = degrade(image, "gaussian_noise", 0.1, index=3, seed=42)
    assert np.array_equal(first, second)
    for kind, severity in (("gaussian_blur", 2), ("motion_blur", 7), ("jpeg", 50)):
        assert degrade(image, kind, severity, 0).shape == image.shape


def test_sift_bovw_fit_and_transform(tmp_path):
    rng = np.random.default_rng(7)
    paths = []
    for index in range(8):
        # Textured synthetic images guarantee enough SIFT keypoints.
        image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        path = tmp_path / f"texture_{index}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths.append(path)
    config = FeatureConfig(
        names=("bovw",), image_size=96, bovw_words=8,
        sift_max_per_image=30, sift_sample_limit=100,
    )
    extractor = HandcraftedFeatureExtractor(config).fit(paths)
    features = extractor.transform(paths)
    assert features.shape == (8, 8)
    assert np.isfinite(features).all()


def test_robustness_rejects_non_test_rows(tmp_path):
    _dataset(tmp_path, classes=2)
    manifest = build_manifest(tmp_path, expected_classes=2, validate_counts=False)
    with pytest.raises(ValueError, match="test rows only"):
        run_robustness(None, None, manifest.split("val"), manifest.class_names)


def test_streamlined_runner_creates_six_candidates(tmp_path, monkeypatch):
    from traditional_cv import experiment as experiment_module

    data = tmp_path / "data"
    rng = np.random.default_rng(11)
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for label in range(3):
            directory = data / split / f"species_{label}"
            directory.mkdir(parents=True)
            for index in range(count):
                image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
                cv2.imwrite(str(directory / f"{split}_{label}_{index}.jpg"), image)
    small = {
        "hsv": FeatureConfig(names=("hsv",), image_size=64),
        "lbp": FeatureConfig(names=("lbp",), image_size=64),
        "hog": FeatureConfig(
            names=("hog",), image_size=64, hog_pixels_per_cell=(8, 8)
        ),
        "bovw256": FeatureConfig(
            names=("bovw",), image_size=64, bovw_words=8,
            sift_max_per_image=20, sift_sample_limit=100,
        ),
    }
    monkeypatch.setattr(experiment_module, "FEATURE_CONFIGS", small)
    manifest = build_manifest(data, expected_classes=3, validate_counts=False)
    runner = experiment_module.ExperimentRunner(manifest, tmp_path / "output")
    summary = runner.run_streamlined_experiments()
    assert len(summary) == 6
    assert set(summary["classifier"]) == {"stochastic_linear_svm", "random_forest"}
    assert (tmp_path / "output/models/frozen_best.joblib").exists()
    assert (tmp_path / "output/STREAMLINED_COMPLETE.json").exists()
