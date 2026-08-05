"""Quick tests for the Grad-CAM code - no weights or dataset needed.

They build a random model and just check the hooks fire and the heatmap comes out the
right shape/range. Run with `pytest -q`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grad_cam.analysis import background_score, cam_to_mask, overlay_cam
from grad_cam.gradcam import GradCAM, GradCAMpp
from grad_cam.models import MODEL_REGISTRY, load_model


def _dummy_input() -> "torch.Tensor":
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


@pytest.mark.parametrize("cam_cls", [GradCAM, GradCAMpp])
def test_cam_shape_and_range(cam_cls):
    model = load_model("resnet18", num_classes=10, weights_path=None)
    target = MODEL_REGISTRY["resnet18"].target_layer(model)
    with cam_cls(model, target) as cam:
        heat = cam(_dummy_input(), class_idx=3)
    assert heat.shape == (224, 224)
    heat = heat.numpy()
    assert np.isfinite(heat).all()
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6


def test_default_class_is_argmax():
    model = load_model("resnet18", num_classes=10, weights_path=None)
    target = MODEL_REGISTRY["resnet18"].target_layer(model)
    with GradCAM(model, target) as cam:
        heat = cam(_dummy_input(), class_idx=None)  # should not raise
    assert heat.shape == (224, 224)


def test_swin_reshape_path_runs():
    """Swin's token-shaped activations must be reshaped before pooling."""
    model = load_model("swin_t", num_classes=10, weights_path=None)
    spec = MODEL_REGISTRY["swin_t"]
    with GradCAM(model, spec.target_layer(model), reshape=spec.reshape) as cam:
        heat = cam(_dummy_input(), class_idx=1)
    assert heat.shape == (224, 224)


def test_mask_and_background_score():
    # A compact central blob should read as focused (small area, low centroid offset).
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[96:128, 96:128] = 1.0
    mask = cam_to_mask(cam)
    assert mask.sum() > 0
    scores = background_score(cam)
    assert 0.0 < scores["mask_area_fraction"] < 0.1
    assert scores["centroid_offset"] < 0.05
    assert overlay_cam(np.zeros((224, 224, 3)), cam).shape == (224, 224, 3)
