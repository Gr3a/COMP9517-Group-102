"""Decision-complete experiment runner. Validation selects; test stays locked."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import ClassifierConfig, FeatureConfig, SEED
from .data import DatasetManifest
from .evaluation import evaluate, save_evaluation, save_example_gallery
from .features import FeatureFusion, HandcraftedFeatureExtractor, extract_features
from .models import fit_classifier


FEATURE_CONFIGS = {
    "hsv": FeatureConfig(names=("hsv",)),
    "lbp": FeatureConfig(names=("lbp",)),
    "hog": FeatureConfig(names=("hog",)),
    "bovw128": FeatureConfig(names=("bovw",), bovw_words=128),
    "bovw256": FeatureConfig(names=("bovw",), bovw_words=256),
}
STREAMLINED_FEATURES = ("hsv", "lbp", "hog", "bovw256")
SVC_C_VALUES = (0.01, 0.1, 1.0, 10.0)


class ExperimentRunner:
    def __init__(self, manifest: DatasetManifest, output_dir: str | Path = "artifacts/traditional"):
        self.manifest = manifest
        self.output = Path(output_dir)
        self.cache = self.output / "cache"
        self.models = self.output / "models"
        for directory in (self.output, self.cache, self.models):
            directory.mkdir(parents=True, exist_ok=True)
        self.train = manifest.split("train")
        self.val = manifest.split("val")
        results_path = self.output / "validation_results.csv"
        self.results: list[dict] = (
            pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
        )
        self.blocks: dict[str, dict[str, np.ndarray]] = {}
        self.extractors: dict[str, HandcraftedFeatureExtractor] = {}
        self._save_environment()

    def _save_environment(self):
        metadata = {
            "seed": SEED,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "class_count": len(self.manifest.class_names),
            "split_counts": self.manifest.frame.groupby("split").size().to_dict(),
        }
        (self.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.manifest.save(self.output)

    def benchmark(self, sample_size: int = 500) -> pd.DataFrame:
        rows = []
        sample = self.train.sample(min(sample_size, len(self.train)), random_state=SEED)
        for name in ("hsv", "lbp", "hog"):
            extractor = HandcraftedFeatureExtractor(FEATURE_CONFIGS[name]).fit(sample.path)
            start = time.perf_counter()
            extractor.transform(sample.path)
            elapsed = time.perf_counter() - start
            rows.append({
                "feature": name,
                "sample_images": len(sample),
                "seconds": elapsed,
                "estimated_30000_seconds": elapsed / len(sample) * 30_000,
            })
        result = pd.DataFrame(rows)
        result.to_csv(self.output / "benchmark.csv", index=False)
        return result

    def _record(self, stage: str, features: str, classifier: str, params: dict, result: dict, fit_seconds: float):
        row = {
            "stage": stage,
            "features": features,
            "classifier": classifier,
            "params": json.dumps(params, sort_keys=True),
            "fit_seconds": fit_seconds,
            **result["metrics"],
        }
        self.results.append(row)
        pd.DataFrame(self.results).to_csv(self.output / "validation_results.csv", index=False)

    def _select_svc(self, feature_name: str, X_train: np.ndarray, X_val: np.ndarray):
        y_train, y_val = self.train.label.to_numpy(), self.val.label.to_numpy()
        candidates = []
        for c in SVC_C_VALUES:
            config = ClassifierConfig(kind="linear_svc", params={"C": c})
            model = fit_classifier(X_train, y_train, config)
            result = evaluate(model, X_val, y_val, self.manifest.class_names)
            self._record("svc_tuning", feature_name, "linear_svc", {"C": c}, result, model.fit_seconds_)
            candidates.append((result["metrics"]["macro_f1"], c, model, result))
        return max(candidates, key=lambda item: item[0])

    def run_individual_features(self):
        summary = []
        for name, config in FEATURE_CONFIGS.items():
            print(f"\n=== {name} ===")
            X_train, extractor = extract_features(
                self.train.path, config, fit=True, cache_dir=self.cache, split=f"train_{name}"
            )
            X_val, _ = extract_features(
                self.val.path, config, extractor=extractor, cache_dir=self.cache, split=f"val_{name}"
            )
            self.blocks[name] = {"train": X_train, "val": X_val}
            self.extractors[name] = extractor
            extractor.save(self.models / f"extractor_{name}.joblib")
            score, c, model, result = self._select_svc(name, X_train, X_val)
            joblib.dump(model, self.models / f"svc_{name}.joblib")
            summary.append({"name": name, "macro_f1": score, "C": c})
        table = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
        table.to_csv(self.output / "individual_feature_summary.csv", index=False)
        return table

    def _load_or_extract_block(self, name: str) -> tuple[np.ndarray, np.ndarray, HandcraftedFeatureExtractor]:
        """Reuse a completed descriptor safely; fit only missing training extractors."""
        config = FEATURE_CONFIGS[name]
        extractor_path = self.models / f"extractor_{name}.joblib"
        if extractor_path.exists():
            extractor = joblib.load(extractor_path)
            X_train, _ = extract_features(
                self.train.path, config, extractor=extractor,
                cache_dir=self.cache, split=f"train_{name}",
            )
        else:
            X_train, extractor = extract_features(
                self.train.path, config, fit=True,
                cache_dir=self.cache, split=f"train_{name}",
            )
            extractor.save(extractor_path)
        X_val, _ = extract_features(
            self.val.path, config, extractor=extractor,
            cache_dir=self.cache, split=f"val_{name}",
        )
        self.blocks[name] = {"train": X_train, "val": X_val}
        self.extractors[name] = extractor
        return X_train, X_val, extractor

    def run_streamlined_experiments(self) -> pd.DataFrame:
        """Run the six controlled experiments selected for the time-bounded study.

        The classifier is fixed during descriptor ablation, then the complete
        representation is fixed during classifier ablation. Test data is never read.
        """
        y_train = self.train.label.to_numpy()
        y_val = self.val.label.to_numpy()
        candidates: list[dict] = []
        svm_config = ClassifierConfig(kind="stochastic_svm")

        print("Stage 1/3: descriptor ablation")
        for name in STREAMLINED_FEATURES:
            print(f"\n=== {name} + stochastic linear SVM ===")
            X_train, X_val, extractor = self._load_or_extract_block(name)
            model = fit_classifier(X_train, y_train, svm_config)
            result = evaluate(model, X_val, y_val, self.manifest.class_names)
            self._record(
                "descriptor_ablation", name, "stochastic_linear_svm",
                svm_config.params, result, model.fit_seconds_,
            )
            joblib.dump(model, self.models / f"streamlined_svm_{name}.joblib")
            candidates.append({
                "features": name, "classifier": "stochastic_linear_svm",
                "macro_f1": result["metrics"]["macro_f1"],
                "extractor": extractor, "model": model,
            })

        print("\nStage 2/3: feature-fusion ablation")
        fusion = FeatureFusion({name: self.extractors[name] for name in STREAMLINED_FEATURES})
        fusion.fit_blocks({name: self.blocks[name]["train"] for name in STREAMLINED_FEATURES})
        X_train_fused = fusion.transform_blocks(
            {name: self.blocks[name]["train"] for name in STREAMLINED_FEATURES}
        )
        X_val_fused = fusion.transform_blocks(
            {name: self.blocks[name]["val"] for name in STREAMLINED_FEATURES}
        )
        fused_name = "+".join(STREAMLINED_FEATURES)
        fusion.save(self.models / "streamlined_fusion.joblib")
        fused_svm = fit_classifier(X_train_fused, y_train, svm_config)
        fused_svm_result = evaluate(fused_svm, X_val_fused, y_val, self.manifest.class_names)
        self._record(
            "fusion_ablation", fused_name, "stochastic_linear_svm",
            svm_config.params, fused_svm_result, fused_svm.fit_seconds_,
        )
        joblib.dump(fused_svm, self.models / "streamlined_fused_svm.joblib")
        candidates.append({
            "features": fused_name, "classifier": "stochastic_linear_svm",
            "macro_f1": fused_svm_result["metrics"]["macro_f1"],
            "extractor": fusion, "model": fused_svm,
        })

        print("\nStage 3/3: classifier ablation")
        rf_params = {
            "n_estimators": 200, "max_features": "sqrt",
            "max_depth": None, "n_jobs": -1,
        }
        rf_config = ClassifierConfig(kind="random_forest", params=rf_params)
        fused_rf = fit_classifier(X_train_fused, y_train, rf_config)
        fused_rf_result = evaluate(fused_rf, X_val_fused, y_val, self.manifest.class_names)
        self._record(
            "classifier_ablation", fused_name, "random_forest",
            rf_params, fused_rf_result, fused_rf.fit_seconds_,
        )
        joblib.dump(fused_rf, self.models / "streamlined_fused_rf.joblib")
        candidates.append({
            "features": fused_name, "classifier": "random_forest",
            "macro_f1": fused_rf_result["metrics"]["macro_f1"],
            "extractor": fusion, "model": fused_rf,
        })

        best = max(candidates, key=lambda item: item["macro_f1"])
        joblib.dump({
            "extractor": best["extractor"],
            "model": best["model"],
            "feature_name": best["features"],
            "classifier": best["classifier"],
            "validation_macro_f1": best["macro_f1"],
            "selection_metric": "validation_macro_f1",
            "candidate_count": len(candidates),
        }, self.models / "frozen_best.joblib")

        new_rows = pd.DataFrame([
            {key: value for key, value in row.items() if key not in {"extractor", "model"}}
            for row in candidates
        ]).sort_values("macro_f1", ascending=False)
        new_rows.to_csv(self.output / "streamlined_summary.csv", index=False)
        (self.output / "STREAMLINED_COMPLETE.json").write_text(json.dumps({
            "experiments": len(candidates),
            "selected_features": best["features"],
            "selected_classifier": best["classifier"],
            "validation_macro_f1": best["macro_f1"],
        }, indent=2), encoding="utf-8")
        return new_rows

    def _ensure_loaded(self):
        if self.blocks:
            return
        for name, config in FEATURE_CONFIGS.items():
            extractor_path = self.models / f"extractor_{name}.joblib"
            if not extractor_path.exists():
                raise RuntimeError("Run individual feature experiments first")
            extractor = joblib.load(extractor_path)
            X_train, _ = extract_features(self.train.path, config, extractor=extractor, cache_dir=self.cache, split=f"train_{name}")
            X_val, _ = extract_features(self.val.path, config, extractor=extractor, cache_dir=self.cache, split=f"val_{name}")
            self.extractors[name] = extractor
            self.blocks[name] = {"train": X_train, "val": X_val}

    def run_fusion_ablation(self):
        self._ensure_loaded()
        individual = pd.read_csv(self.output / "individual_feature_summary.csv")
        # Retain only the better vocabulary to avoid including duplicate SIFT blocks.
        best_bovw = individual[individual.name.str.startswith("bovw")].iloc[0]["name"]
        candidates = [n for n in individual.name if not n.startswith("bovw")] + [best_bovw]
        candidates.sort(key=lambda n: float(individual.loc[individual.name == n, "macro_f1"].iloc[0]), reverse=True)
        chosen: list[str] = []
        best = None
        for name in candidates:
            chosen.append(name)
            fusion = FeatureFusion({n: self.extractors[n] for n in chosen})
            fusion.fit_blocks({n: self.blocks[n]["train"] for n in chosen})
            X_train = fusion.transform_blocks({n: self.blocks[n]["train"] for n in chosen})
            X_val = fusion.transform_blocks({n: self.blocks[n]["val"] for n in chosen})
            feature_name = "+".join(chosen)
            score, c, model, result = self._select_svc(feature_name, X_train, X_val)
            fusion.save(self.models / f"fusion_{len(chosen)}.joblib")
            joblib.dump(model, self.models / f"fusion_svc_{len(chosen)}.joblib")
            candidate = (score, feature_name, c, fusion, model, X_train, X_val)
            if best is None or score > best[0]:
                best = candidate
        assert best is not None
        score, feature_name, c, fusion, svc, X_train, X_val = best

        # Same selected representation, controlled classifier comparison.
        rf_candidates = []
        for max_features in ("sqrt", 0.2):
            for max_depth in (None, 30):
                params = {"max_features": max_features, "max_depth": max_depth}
                model = fit_classifier(X_train, self.train.label.to_numpy(), ClassifierConfig("random_forest", params))
                result = evaluate(model, X_val, self.val.label.to_numpy(), self.manifest.class_names)
                self._record("classifier_comparison", feature_name, "random_forest", params, result, model.fit_seconds_)
                rf_candidates.append((result["metrics"]["macro_f1"], model, params))
        best_rf = max(rf_candidates, key=lambda item: item[0])
        # Final choice remains entirely validation-macro-F1 based.
        final_model, final_kind = (svc, "linear_svc") if score >= best_rf[0] else (best_rf[1], "random_forest")
        joblib.dump({
            "extractor": fusion,
            "model": final_model,
            "feature_name": feature_name,
            "classifier": final_kind,
            "validation_macro_f1": max(score, best_rf[0]),
        }, self.models / "frozen_best.joblib")
        return {"features": feature_name, "classifier": final_kind, "validation_macro_f1": max(score, best_rf[0])}

    def evaluate_test_once(self):
        """Separate explicit step: do not call until all validation choices are frozen."""
        bundle_path = self.models / "frozen_best.joblib"
        marker = self.output / "TEST_EVALUATED.json"
        if marker.exists():
            raise RuntimeError("Test was already evaluated; refusing an accidental second selection loop")
        bundle = joblib.load(bundle_path)
        test = self.manifest.split("test")
        X_test = bundle["extractor"].transform(test.path)
        result = evaluate(bundle["model"], X_test, test.label.to_numpy(), self.manifest.class_names)
        save_evaluation(result, self.manifest.class_names, self.output / "test", "traditional_clean")
        save_example_gallery(result, test, self.manifest.class_names, self.output / "test")
        marker.write_text(json.dumps({
            "features": bundle["feature_name"],
            "classifier": bundle["classifier"],
            "metrics": result["metrics"],
        }, indent=2), encoding="utf-8")
        return result["metrics"]
