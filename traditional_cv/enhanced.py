"""Validation-only enhanced handcrafted experiment; baseline artifacts stay untouched."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ClassifierConfig, FeatureConfig
from .evaluation import evaluate, save_evaluation, save_example_gallery
from .features import FeatureFusion, extract_features
from .models import fit_classifier


ENHANCED_CONFIGS = {
    "spatial_hsv224": FeatureConfig(names=("spatial_hsv",), image_size=224),
    "lbp224": FeatureConfig(names=("lbp",), image_size=224),
    "hog224": FeatureConfig(names=("hog",), image_size=224),
    "spatial_bovw512": FeatureConfig(
        names=("spatial_bovw",), image_size=224, bovw_words=512,
        sift_max_per_image=500, sift_sample_limit=200_000,
    ),
}
SVM_ALPHAS = (1e-5, 1e-4, 1e-3)
RF_SCREEN = (
    {"max_depth": 15, "min_samples_leaf": 2},
    {"max_depth": 25, "min_samples_leaf": 2},
    {"max_depth": 25, "min_samples_leaf": 4},
    {"max_depth": None, "min_samples_leaf": 4},
)


class EnhancedExperimentRunner:
    def __init__(self, manifest, output_dir="artifacts/traditional_enhanced"):
        self.manifest = manifest
        self.output = Path(output_dir)
        self.cache = self.output / "cache"
        self.models = self.output / "models"
        for path in (self.output, self.cache, self.models):
            path.mkdir(parents=True, exist_ok=True)
        self.train, self.val = manifest.split("train"), manifest.split("val")

    def _features(self):
        train_blocks, val_blocks, extractors = {}, {}, {}
        for name, config in ENHANCED_CONFIGS.items():
            print(f"\n=== Extracting {name} ===", flush=True)
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
            train_blocks[name], val_blocks[name], extractors[name] = X_train, X_val, extractor
        fusion = FeatureFusion(extractors).fit_blocks(train_blocks)
        fusion.save(self.models / "enhanced_fusion.joblib")
        return fusion, fusion.transform_blocks(train_blocks), fusion.transform_blocks(val_blocks)

    def _row(self, family, params, model, result, train_accuracy):
        return {
            "family": family, "params": json.dumps(params, sort_keys=True),
            "train_accuracy": train_accuracy, "fit_seconds": model.fit_seconds_,
            **result["metrics"],
        }

    def run_validation_search(self):
        fusion, X_train, X_val = self._features()
        y_train, y_val = self.train.label.to_numpy(), self.val.label.to_numpy()
        rows, candidates = [], []

        for alpha in SVM_ALPHAS:
            params = {"alpha": alpha}
            print(f"\n=== Enhanced SVM alpha={alpha:g} ===", flush=True)
            model = fit_classifier(X_train, y_train, ClassifierConfig("stochastic_svm", params))
            result = evaluate(model, X_val, y_val, self.manifest.class_names)
            row = self._row("stochastic_linear_svm", params, model, result, float((model.predict(X_train) == y_train).mean()))
            rows.append(row); candidates.append((row["macro_f1"], model, row))
            joblib.dump(model, self.models / f"svm_alpha_{alpha:g}.joblib")

        for index, tuning in enumerate(RF_SCREEN, 1):
            params = {"n_estimators": 100, "max_features": "sqrt", "n_jobs": -1, **tuning}
            print(f"\n=== Regularised RF screen {index}/{len(RF_SCREEN)}: {tuning} ===", flush=True)
            model = fit_classifier(X_train, y_train, ClassifierConfig("random_forest", params))
            result = evaluate(model, X_val, y_val, self.manifest.class_names)
            row = self._row("random_forest_screen", params, model, result, float((model.predict(X_train) == y_train).mean()))
            rows.append(row); candidates.append((row["macro_f1"], model, row))

        best = max(candidates, key=lambda item: item[0])
        best_model, best_row = best[1], best[2]
        # If RF wins screening, refit the identical regularisation with 200 trees.
        if best_row["family"] == "random_forest_screen":
            params = json.loads(best_row["params"]); params["n_estimators"] = 200
            print(f"\n=== Final 200-tree regularised RF: {params} ===", flush=True)
            best_model = fit_classifier(X_train, y_train, ClassifierConfig("random_forest", params))
            result = evaluate(best_model, X_val, y_val, self.manifest.class_names)
            best_row = self._row("random_forest_final", params, best_model, result, float((best_model.predict(X_train) == y_train).mean()))
            rows.append(best_row)

        table = pd.DataFrame(rows)
        table.to_csv(self.output / "enhanced_validation_results.csv", index=False)
        final = max(
            [(r["macro_f1"], r) for r in rows if r["family"] != "random_forest_screen"],
            key=lambda item: item[0],
        )[1]
        # Resolve the final object (the refitted RF or matching SVM).
        if final is best_row:
            final_model = best_model
        else:
            alpha = json.loads(final["params"])["alpha"]
            final_model = joblib.load(self.models / f"svm_alpha_{alpha:g}.joblib")
        joblib.dump({
            "extractor": fusion, "model": final_model,
            "feature_name": "+".join(ENHANCED_CONFIGS),
            "classifier": final["family"], "validation_macro_f1": final["macro_f1"],
        }, self.models / "enhanced_frozen_best.joblib")
        (self.output / "ENHANCED_COMPLETE.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        self._plots(table)
        return table.sort_values("macro_f1", ascending=False)

    def _plots(self, table):
        sns.set_theme(style="whitegrid")
        labels = [f"{f}\n{json.loads(p)}" for f, p in zip(table.family, table.params)]
        for metrics, filename in [
            (("top1_accuracy", "top5_accuracy", "macro_f1"), "metrics_comparison.png"),
            (("train_accuracy", "top1_accuracy"), "train_validation_gap.png"),
        ]:
            frame = table.assign(configuration=labels).melt("configuration", value_vars=list(metrics), var_name="metric", value_name="score")
            fig, ax = plt.subplots(figsize=(14, 7)); sns.barplot(frame, x="configuration", y="score", hue="metric", ax=ax)
            ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(self.output / filename, dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6)); sns.scatterplot(table, x="fit_seconds", y="macro_f1", hue="family", s=100, ax=ax)
        ax.set_title("Validation macro-F1 versus training time"); fig.tight_layout(); fig.savefig(self.output / "performance_vs_time.png", dpi=200); plt.close(fig)

        baseline_path = self.output.parent / "traditional" / "validation_results.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            baseline = baseline[baseline["stage"].isin(["descriptor_ablation", "fusion_ablation", "classifier_ablation"])]
            base_best = baseline.loc[baseline["macro_f1"].idxmax()]
            enhanced_best = table.loc[table["macro_f1"].idxmax()]
            comparison = pd.DataFrame([
                {"method": "Baseline winner", "top1": base_best.top1_accuracy, "top5": base_best.top5_accuracy, "macro_f1": base_best.macro_f1},
                {"method": "Enhanced validation winner", "top1": enhanced_best.top1_accuracy, "top5": enhanced_best.top5_accuracy, "macro_f1": enhanced_best.macro_f1},
            ]).melt("method", var_name="metric", value_name="score")
            fig, ax = plt.subplots(figsize=(9, 6)); sns.barplot(comparison, x="method", y="score", hue="metric", ax=ax)
            ax.set_title("Baseline versus enhanced validation performance"); fig.tight_layout(); fig.savefig(self.output / "baseline_vs_enhanced.png", dpi=200); plt.close(fig)

    def evaluate_test_once(self):
        marker = self.output / "ENHANCED_TEST_EVALUATED.json"
        if marker.exists():
            raise RuntimeError("Enhanced test was already evaluated")
        bundle = joblib.load(self.models / "enhanced_frozen_best.joblib")
        test = self.manifest.split("test")
        X = bundle["extractor"].transform(test.path)
        result = evaluate(bundle["model"], X, test.label.to_numpy(), self.manifest.class_names)
        save_evaluation(result, self.manifest.class_names, self.output / "test", "enhanced_traditional")
        save_example_gallery(result, test, self.manifest.class_names, self.output / "test")
        marker.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        return result["metrics"]
