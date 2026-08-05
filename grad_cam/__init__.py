"""Grad-CAM explainability for our deep-learning classifiers (advanced topic 1).

Loads the trained CNNs from Alex_Shim/, runs Grad-CAM on them, and answers the four
questions from the spec: correct vs incorrect maps, same-genus confusions, organism vs
background, and concrete failure cases. See notebooks/Grad_CAM_Explainability.ipynb to run
it and grad_cam/INTERPRETATION.md for the written-up findings.
"""

from __future__ import annotations

from .gradcam import GradCAM, GradCAMpp
from .models import EVAL_TRANSFORM, MODEL_REGISTRY, denormalize, genus_of, load_model

__all__ = [
    "GradCAM",
    "GradCAMpp",
    "MODEL_REGISTRY",
    "EVAL_TRANSFORM",
    "load_model",
    "denormalize",
    "genus_of",
]
