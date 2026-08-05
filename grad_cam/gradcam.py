"""Grad-CAM and Grad-CAM++ written from scratch with hooks.

I didn't use a library here (e.g. pytorch-grad-cam) on purpose - the assignment gives
more credit for implementing it yourself, and it's not that much code.

Idea (Selvaraju et al. 2017): grab the target layer's activations A on the forward pass
and the gradients of the class score w.r.t. A on the backward pass. Weight each channel
by how important its gradient is, sum them up, ReLU, and you get the heatmap. Grad-CAM++
(Chattopadhay et al. 2018) only changes how those channel weights are computed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer, reshape=None):
        # NOTE: don't call this under torch.no_grad() - we need the backward pass.
        self.model = model
        self.reshape = reshape          # only used for transformers (Swin)
        self._activations = None
        self._gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def _weights(self, activations, gradients):
        # plain Grad-CAM: just average the gradients over H and W
        return gradients.mean(dim=(2, 3), keepdim=True)

    def __call__(self, input_tensor, class_idx=None):
        """Return a (H, W) heatmap in [0, 1] for one image (input shape (1, C, H, W)).
        If class_idx is None we explain whatever the model predicts."""
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("expects a single image, shape (1, C, H, W)")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        logits[0, class_idx].backward()

        activations, gradients = self._activations, self._gradients
        if activations is None or gradients is None:
            raise RuntimeError("hooks didn't fire - wrong target layer?")
        if self.reshape is not None:
            activations = self.reshape(activations)
            gradients = self.reshape(gradients)

        weights = self._weights(activations, gradients)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)   # only care about evidence *for* the class, not against

        # upsample from the tiny conv-map back to the input size, then 0-1 normalise
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


class GradCAMpp(GradCAM):
    """Grad-CAM++ - weights each pixel of the gradient instead of a flat average, which
    gives sharper maps and handles multiple instances better."""

    def _weights(self, activations, gradients):
        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)
        global_sum = activations.sum(dim=(2, 3), keepdim=True)
        denom = 2 * grad2 + global_sum * grad3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))   # avoid /0
        alpha = grad2 / denom
        return (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
