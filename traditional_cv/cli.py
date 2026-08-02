"""Command-line entry point used locally and by the Colab notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import build_manifest
from .experiment import ExperimentRunner
from .robustness import plot_robustness, run_robustness


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=["audit", "benchmark", "streamlined", "features", "fusion", "test", "robustness"],
    )
    result.add_argument("--data", default="Team_Dataset")
    result.add_argument("--output", default="artifacts/traditional")
    result.add_argument("--expected-classes", type=int, default=500)
    result.add_argument("--hash-files", action="store_true")
    result.add_argument("--sample-size", type=int, default=500)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = build_manifest(
        args.data,
        expected_classes=args.expected_classes,
        hash_files=args.hash_files,
    )
    runner = ExperimentRunner(manifest, args.output)
    if args.command == "audit":
        print(manifest.frame.groupby("split").size())
        print(f"Classes: {len(manifest.class_names)}")
    elif args.command == "benchmark":
        print(runner.benchmark(args.sample_size).to_string(index=False))
    elif args.command == "features":
        print(runner.run_individual_features().to_string(index=False))
    elif args.command == "streamlined":
        print(runner.run_streamlined_experiments().to_string(index=False))
    elif args.command == "fusion":
        print(json.dumps(runner.run_fusion_ablation(), indent=2))
    elif args.command == "test":
        print(json.dumps(runner.evaluate_test_once(), indent=2))
    elif args.command == "robustness":
        bundle = joblib.load(Path(args.output) / "models" / "frozen_best.joblib")
        results = run_robustness(
            bundle["model"], bundle["extractor"], manifest.split("test"), manifest.class_names
        )
        plot_robustness(results, Path(args.output) / "robustness")
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
