"""HSV, spatial LBP, HOG, SIFT-BoVW, fusion, and disk caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, normalize
from skimage.feature import hog, local_binary_pattern
from tqdm.auto import tqdm

from .config import FeatureConfig


ImageTransform = Callable[[np.ndarray, int], np.ndarray]
VALID_FEATURES = {"hsv", "spatial_hsv", "lbp", "hog", "bovw", "spatial_bovw"}


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Resize without distortion, padding with the image's median colour."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    fill = np.median(image.reshape(-1, image.shape[2]), axis=0).astype(image.dtype)
    canvas = np.empty((size, size, image.shape[2]), dtype=image.dtype)
    canvas[...] = fill
    x0, y0 = (size - new_width) // 2, (size - new_height) // 2
    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
    return canvas


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Train-only-fitted feature pipeline supporting serialisation."""

    def __init__(self, config: FeatureConfig):
        unknown = set(config.names) - VALID_FEATURES
        if unknown:
            raise ValueError(f"Unknown descriptors: {sorted(unknown)}")
        self.config = config
        self.kmeans_: MiniBatchKMeans | None = None
        self.scalers_: dict[str, StandardScaler] = {}
        self.block_order_: list[str] = list(config.names)
        self.timings_: dict[str, float] = {}

    def _sift(self):
        return cv2.SIFT_create(nfeatures=self.config.sift_max_per_image)

    def _sift_descriptors(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(resize_with_padding(image, self.config.image_size), cv2.COLOR_RGB2GRAY)
        _, descriptors = self._sift().detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, 128), dtype=np.float32)
        return descriptors[: self.config.sift_max_per_image].astype(np.float32)

    def fit(self, paths: Iterable[str | Path], y=None):
        paths = list(paths)
        started = time.perf_counter()
        if {"bovw", "spatial_bovw"} & set(self.config.names):
            rng = np.random.default_rng(self.config.seed)
            pool: list[np.ndarray] = []
            count = 0
            # Reservoir sampling avoids retaining millions of descriptors.
            reservoir = np.empty((self.config.sift_sample_limit, 128), dtype=np.float32)
            for path in tqdm(paths, desc="Fitting SIFT vocabulary"):
                descriptors = self._sift_descriptors(read_rgb(path))
                for descriptor in descriptors:
                    count += 1
                    if count <= len(reservoir):
                        reservoir[count - 1] = descriptor
                    else:
                        index = int(rng.integers(0, count))
                        if index < len(reservoir):
                            reservoir[index] = descriptor
            sample = reservoir[: min(count, len(reservoir))]
            if len(sample) < self.config.bovw_words:
                raise ValueError("Too few SIFT descriptors to construct vocabulary")
            self.kmeans_ = MiniBatchKMeans(
                n_clusters=self.config.bovw_words,
                batch_size=4096,
                n_init=3,
                random_state=self.config.seed,
                reassignment_ratio=0.01,
            ).fit(sample)

        # Fit block scalers only for fusion. Individual descriptors retain their
        # natural normalisation and classifiers add any required scaling.
        if len(self.config.names) > 1:
            raw = self._transform_blocks(paths)
            self.scalers_ = {
                name: StandardScaler().fit(block) for name, block in raw.items()
            }
        self.timings_["fit_seconds"] = time.perf_counter() - started
        return self

    def _hsv(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(resize_with_padding(image, self.config.image_size), cv2.COLOR_RGB2HSV)
        return self._hsv_region(hsv)

    def _hsv_region(self, hsv: np.ndarray) -> np.ndarray:
        histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None, list(self.config.hsv_bins), [0, 180, 0, 256, 0, 256]
        ).ravel().astype(np.float32)
        histogram /= histogram.sum() + 1e-12
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        moments = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])
        # Put moments on comparable fixed ranges before later standardisation.
        moments /= np.array([180, 256, 256, 180, 256, 256], dtype=np.float32)
        return np.concatenate([histogram, moments]).astype(np.float32)

    def _spatial_hsv(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(resize_with_padding(image, self.config.image_size), cv2.COLOR_RGB2HSV)
        regions = [hsv]
        regions.extend(cell for row in np.array_split(hsv, 2, axis=0) for cell in np.array_split(row, 2, axis=1))
        return np.concatenate([self._hsv_region(region) for region in regions])

    def _lbp(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(resize_with_padding(image, self.config.image_size), cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(
            gray, self.config.lbp_points, self.config.lbp_radius, method="uniform"
        )
        gy, gx = self.config.lbp_grid
        bins = self.config.lbp_points + 2
        blocks = []
        for row in np.array_split(lbp, gy, axis=0):
            for cell in np.array_split(row, gx, axis=1):
                hist = np.bincount(cell.astype(np.int32).ravel(), minlength=bins).astype(np.float32)
                blocks.append(hist / (hist.sum() + 1e-12))
        return np.concatenate(blocks)

    def _hog(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(resize_with_padding(image, self.config.image_size), cv2.COLOR_RGB2GRAY)
        return hog(
            gray,
            orientations=self.config.hog_orientations,
            pixels_per_cell=self.config.hog_pixels_per_cell,
            cells_per_block=self.config.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

    def _bovw(self, image: np.ndarray) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("BoVW extractor must be fitted on training data first")
        descriptors = self._sift_descriptors(image)
        vector = np.zeros(self.config.bovw_words, dtype=np.float32)
        if len(descriptors):
            words = self.kmeans_.predict(descriptors)
            vector = np.bincount(words, minlength=self.config.bovw_words).astype(np.float32)
            vector /= vector.sum() + 1e-12
            vector = np.sqrt(vector)
        return vector

    def _spatial_bovw(self, image: np.ndarray) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("Spatial BoVW extractor must be fitted on training data first")
        resized = resize_with_padding(image, self.config.image_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._sift().detectAndCompute(gray, None)
        output = np.zeros((5, self.config.bovw_words), dtype=np.float32)
        if descriptors is None:
            return output.ravel()
        descriptors = descriptors[: self.config.sift_max_per_image].astype(np.float32)
        keypoints = keypoints[: self.config.sift_max_per_image]
        words = self.kmeans_.predict(descriptors)
        size = self.config.image_size
        for word, keypoint in zip(words, keypoints):
            output[0, word] += 1
            x, y = keypoint.pt
            quadrant = 1 + min(1, int(y >= size / 2)) * 2 + min(1, int(x >= size / 2))
            output[quadrant, word] += 1
        output /= output.sum(axis=1, keepdims=True) + 1e-12
        return np.sqrt(output).ravel()

    def _one(self, image: np.ndarray, name: str) -> np.ndarray:
        return getattr(self, f"_{name}")(image)

    def _transform_blocks(
        self,
        paths: Iterable[str | Path],
        image_transform: ImageTransform | None = None,
    ) -> dict[str, np.ndarray]:
        rows = {name: [] for name in self.block_order_}
        for index, path in enumerate(tqdm(list(paths), desc="Extracting features")):
            image = read_rgb(path)
            if image_transform is not None:
                image = image_transform(image, index)
            for name in self.block_order_:
                rows[name].append(self._one(image, name))
        return {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}

    def transform(
        self,
        paths: Iterable[str | Path],
        image_transform: ImageTransform | None = None,
    ) -> np.ndarray:
        started = time.perf_counter()
        blocks = self._transform_blocks(paths, image_transform=image_transform)
        if len(blocks) == 1:
            result = next(iter(blocks.values()))
        else:
            result = np.concatenate(
                [self.scalers_[name].transform(blocks[name]) for name in self.block_order_], axis=1
            )
            result = normalize(result, norm="l2").astype(np.float32)
        self.timings_["last_transform_seconds"] = time.perf_counter() - started
        return result

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class FeatureFusion(BaseEstimator, TransformerMixin):
    """Fuse already-fitted single-descriptor extractors without refitting BoVW."""

    def __init__(self, extractors: dict[str, HandcraftedFeatureExtractor]):
        self.extractors = extractors
        self.block_order_ = list(extractors)
        self.scalers_: dict[str, StandardScaler] = {}

    def fit_blocks(self, blocks: dict[str, np.ndarray]):
        self.scalers_ = {name: StandardScaler().fit(blocks[name]) for name in self.block_order_}
        return self

    def transform_blocks(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        fused = np.concatenate(
            [self.scalers_[name].transform(blocks[name]) for name in self.block_order_], axis=1
        )
        return normalize(fused, norm="l2").astype(np.float32)

    def transform(self, paths, image_transform: ImageTransform | None = None) -> np.ndarray:
        blocks = {
            name: extractor.transform(paths, image_transform=image_transform)
            for name, extractor in self.extractors.items()
        }
        return self.transform_blocks(blocks)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def _cache_key(paths: list[str | Path], config: FeatureConfig, split: str) -> str:
    payload = json.dumps({"paths": [str(p) for p in paths], "config": config.to_dict(), "split": split}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def extract_features(
    image_paths: Iterable[str | Path],
    config: FeatureConfig,
    *,
    extractor: HandcraftedFeatureExtractor | None = None,
    fit: bool = False,
    cache_dir: str | Path | None = None,
    split: str = "data",
) -> tuple[np.ndarray, HandcraftedFeatureExtractor]:
    """Public extraction API with optional deterministic cache."""
    paths = list(image_paths)
    extractor = extractor or HandcraftedFeatureExtractor(config)
    if fit:
        extractor.fit(paths)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{split}_{_cache_key(paths, config, split)}.npy"
        if cache_path.exists():
            return np.load(cache_path), extractor
    features = extractor.transform(paths)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
    return features, extractor
