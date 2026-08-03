"""Central experiment defaults. Keep every graded run reproducible."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SEED = 42


@dataclass(frozen=True)
class FeatureConfig:
    names: tuple[str, ...] = ("hsv",)
    image_size: int = 128
    hsv_bins: tuple[int, int, int] = (8, 8, 8)
    lbp_points: int = 24
    lbp_radius: int = 3
    lbp_grid: tuple[int, int] = (4, 4)
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (16, 16)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    sift_max_per_image: int = 200
    sift_sample_limit: int = 200_000
    bovw_words: int = 256
    seed: int = SEED

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["names"] = list(self.names)
        return value


@dataclass(frozen=True)
class ClassifierConfig:
    kind: str = "linear_svc"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = SEED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

