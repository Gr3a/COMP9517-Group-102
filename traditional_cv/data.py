"""Dataset discovery and leakage/integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_COUNTS = {"train": 40, "val": 10, "test": 10}


@dataclass
class DatasetManifest:
    frame: pd.DataFrame
    class_names: list[str]
    root: Path

    def split(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame["split"] == name].reset_index(drop=True)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        serial = self.frame.copy()
        serial["path"] = serial["path"].map(str)
        serial.to_csv(output / "dataset_manifest.csv", index=False)
        (output / "class_names.json").write_text(
            json.dumps(self.class_names, indent=2), encoding="utf-8"
        )


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: str | Path,
    *,
    expected_classes: int | None = 500,
    validate_counts: bool = True,
    hash_files: bool = False,
) -> DatasetManifest:
    """Build a deterministic manifest and reject split/class mismatches.

    Hashing is optional because reading all 2.6 GB adds startup time. Enable it once
    for the final integrity audit to detect byte-identical leakage across splits.
    """
    root = Path(root).expanduser().resolve()
    class_sets: dict[str, set[str]] = {}
    rows: list[dict] = []
    seen_names: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        class_sets[split] = set(classes)
        for class_name in classes:
            files = sorted(
                p for p in (split_dir / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            if validate_counts and len(files) != EXPECTED_COUNTS[split]:
                raise ValueError(
                    f"{split}/{class_name}: expected {EXPECTED_COUNTS[split]} "
                    f"images, found {len(files)}"
                )
            for path in files:
                # Official train and validation UUIDs should not repeat.
                if path.name in seen_names and seen_names[path.name] != split:
                    raise ValueError(f"Possible leakage: {path.name} occurs in two splits")
                seen_names[path.name] = split
                digest = _file_hash(path) if hash_files else None
                if digest and digest in seen_hashes and seen_hashes[digest] != split:
                    raise ValueError(f"Byte-identical image occurs in two splits: {path}")
                if digest:
                    seen_hashes[digest] = split
                rows.append({
                    "path": path,
                    "split": split,
                    "class_name": class_name,
                    "sha256": digest,
                })

    if not (class_sets["train"] == class_sets["val"] == class_sets["test"]):
        raise ValueError("Train, validation, and test class sets differ")
    class_names = sorted(class_sets["train"])
    if expected_classes is not None and len(class_names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(class_names)}")
    label_map = {name: index for index, name in enumerate(class_names)}
    frame = pd.DataFrame(rows)
    frame["label"] = frame["class_name"].map(label_map).astype(int)
    return DatasetManifest(frame, class_names, root)

