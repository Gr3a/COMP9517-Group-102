"""Picking example images, drawing the overlays, and the two quantitative checks.

Everything works off the *_predictions.csv the training notebooks already save (columns:
filepath, true, pred, confidence, top1..top5, all integer class ids). Reusing that file
means the images I show match the numbers reported elsewhere, and I don't have to re-run
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.filters import threshold_otsu

from .gradcam import GradCAM
from .models import EVAL_TRANSFORM, denormalize, genus_of


def load_predictions(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def remap_filepaths(df: pd.DataFrame, data_dir, anchor: str = "Team_Dataset") -> pd.DataFrame:
    """The CSVs were made on Windows so paths look like C:\\...\\Team_Dataset\\test\\...
    Keep everything from Team_Dataset onwards and re-root it locally so the same CSV
    works on my Mac / Colab."""
    root = Path(data_dir)

    def fix(path):
        p = str(path).replace("\\", "/")
        rel = p.split(anchor, 1)[1].lstrip("/") if anchor in p else Path(p).name
        return str(root / rel)

    out = df.copy()
    out["filepath"] = out["filepath"].map(fix)
    return out


def add_genus(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    genera = [genus_of(name) for name in class_names]
    out = df.copy()
    out["true_genus"] = out["true"].map(lambda i: genera[int(i)])
    out["pred_genus"] = out["pred"].map(lambda i: genera[int(i)])
    return out


def select_correct_incorrect(df, n=4, seed=42):
    correct = df[df["pred"] == df["true"]]
    incorrect = df[df["pred"] != df["true"]]
    rng = np.random.default_rng(seed)

    def sample(frame):
        if len(frame) <= n:
            return frame
        return frame.loc[rng.choice(frame.index.to_numpy(), size=n, replace=False)]

    return sample(correct), sample(incorrect)


def most_confident_mistakes(df, n=6):
    # the wrong answers the model was most sure about - the interesting failures
    return df[df["pred"] != df["true"]].nlargest(n, "confidence")


@dataclass(frozen=True)
class ConfusedPair:
    true_idx: int
    pred_idx: int
    count: int
    same_genus: bool


def confusable_pairs(df, class_names, top=10, same_genus_only=False):
    """Most frequent (true -> predicted) mix-ups, counted straight from the CSV."""
    wrong = df[df["pred"] != df["true"]]
    counts = (
        wrong.groupby(["true", "pred"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    genera = [genus_of(name) for name in class_names]
    pairs = []
    for _, row in counts.iterrows():
        t, p = int(row["true"]), int(row["pred"])
        same = genera[t] == genera[p]
        if same_genus_only and not same:
            continue
        pairs.append(ConfusedPair(t, p, int(row["count"]), same))
        if len(pairs) >= top:
            break
    return pairs


def cam_for_path(gradcam, image_path, class_idx=None, device="cpu"):
    """Run one image through Grad-CAM. Returns (rgb, cam), both 224x224 - the image we
    can show and the heatmap in [0,1]."""
    image = Image.open(image_path).convert("RGB")
    x = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    cam = gradcam(x, class_idx).numpy()
    rgb = denormalize(x[0]).permute(1, 2, 0).cpu().numpy()
    return rgb, cam


def overlay_cam(rgb, cam, alpha=0.5):
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


# --- organism vs background --------------------------------------------------
# There are no ground-truth masks in iNat, so this is a proxy: threshold the CAM into a
# blob and measure how big / how central / how peaky it is. Small + central + peaky ==
# the model locked onto an object; big + spread out == more background-ish.

def cam_to_mask(cam, method="otsu", quantile=0.5):
    if method == "otsu":
        if np.ptp(cam) < 1e-6:            # flat map, otsu would blow up
            return np.zeros_like(cam, dtype=bool)
        return cam >= threshold_otsu(cam)
    if method == "quantile":
        return cam >= np.quantile(cam, quantile)
    raise ValueError(f"unknown mask method '{method}'")


def background_score(cam, method="otsu"):
    h, w = cam.shape
    mask = cam_to_mask(cam, method=method)
    area = float(mask.mean())

    total = cam.sum()
    if total > 0:
        ys, xs = np.mgrid[0:h, 0:w]
        cy, cx = (cam * ys).sum() / total, (cam * xs).sum() / total
        max_dist = np.hypot(h / 2, w / 2)
        offset = float(np.hypot(cy - h / 2, cx - w / 2) / max_dist)
    else:
        offset = float("nan")

    peak = float(cam[mask].mean() - cam.mean()) if mask.any() else 0.0
    return {"mask_area_fraction": area, "centroid_offset": offset, "peak_concentration": peak}


# --- faithfulness + locality studies ----------------------------------------

def faithfulness_deletion(gradcam, df, device="cpu", n=60,
                          fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), seed=3):
    """Deletion test: blank the most-activated pixels and see how far the confidence
    drops. If the map is honest, removing those pixels should tank the prediction.
    Returns {fraction_removed: mean_confidence} over n correct images."""
    correct = df[df["pred"] == df["true"]]
    rng = np.random.default_rng(seed)
    idx = rng.choice(correct.index.to_numpy(), size=min(n, len(correct)), replace=False)
    model = gradcam.model
    curve = {f: [] for f in fractions}

    for i in idx:
        row = df.loc[i]
        pred = int(row["pred"])
        image = Image.open(row["filepath"]).convert("RGB")
        x = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
        cam = gradcam(x, pred).to(x.device)
        with torch.no_grad():
            for f in fractions:
                if f <= 0:
                    masked = x
                else:
                    thresh = torch.quantile(cam, 1 - f)
                    keep = cam >= thresh
                    masked = x.clone()
                    masked[:, :, keep] = 0.0     # 0 is the ImageNet mean after norm
                curve[f].append(model(masked).softmax(1)[0, pred].item())

    return {f: float(np.mean(v)) for f, v in curve.items()}


def plot_deletion_curve(curve, out_dir=None, name="faithfulness_deletion.png"):
    fracs = sorted(curve)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot([f * 100 for f in fracs], [curve[f] for f in fracs], marker="o")
    ax.set_xlabel("% most-activated pixels removed")
    ax.set_ylabel("mean confidence in predicted class")
    ax.set_title("Grad-CAM faithfulness (deletion test)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def locality_distribution(gradcam, df, device="cpu", n_per=200, seed=5):
    """background_score over a bunch of correct and incorrect predictions, so we can show
    a distribution instead of a single number."""
    rng = np.random.default_rng(seed)
    out = {}
    groups = {"correct": df[df["pred"] == df["true"]], "incorrect": df[df["pred"] != df["true"]]}
    for kind, frame in groups.items():
        idx = rng.choice(frame.index.to_numpy(), size=min(n_per, len(frame)), replace=False)
        rows = []
        for i in idx:
            row = df.loc[i]
            _, cam = cam_for_path(gradcam, row["filepath"], int(row["pred"]), device)
            rows.append(background_score(cam))
        out[kind] = pd.DataFrame(rows)
    return out


def plot_locality_distribution(dists, out_dir=None, name="locality_distribution.png"):
    cols = ["mask_area_fraction", "centroid_offset"]
    fig, axes = plt.subplots(1, len(cols), figsize=(8, 3.2))
    for col, ax in zip(cols, axes):
        for kind, frame in dists.items():
            ax.hist(frame[col], bins=20, alpha=0.5, label=kind, density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Attention locality: correct vs incorrect", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# --- figures -----------------------------------------------------------------

def _save(fig, out_dir, name):
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    return fig


def _panel(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def fig_correct_vs_incorrect(gradcam, df, class_names, device="cpu", n=3, out_dir=None):
    """Original + CAM overlay for a few correct and a few wrong predictions."""
    correct, incorrect = select_correct_incorrect(df, n=n)
    rows = [("correct", r) for _, r in correct.iterrows()]
    rows += [("wrong", r) for _, r in incorrect.iterrows()]

    fig, axes = plt.subplots(len(rows), 2, figsize=(5, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (kind, row) in enumerate(rows):
        rgb, cam = cam_for_path(gradcam, row["filepath"], int(row["pred"]), device)
        true_name = class_names[int(row["true"])]
        pred_name = class_names[int(row["pred"])]
        tag = "OK" if kind == "correct" else "MISS"
        _panel(axes[i, 0], rgb, f"[{tag}] true: {true_name}")
        _panel(axes[i, 1], overlay_cam(rgb, cam), f"CAM -> pred: {pred_name}\nconf {row['confidence']:.2f}")
    fig.suptitle("Grad-CAM: correct vs incorrect predictions", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "correct_vs_incorrect.png")


def fig_confusable_pair(gradcam, df, class_names, pair, device="cpu", out_dir=None):
    """Take one image the model got wrong and show the CAM for BOTH the true and the
    predicted species. If the two heatmaps sit on the same spot, that's why it can't tell
    them apart."""
    misses = df[(df["true"] == pair.true_idx) & (df["pred"] == pair.pred_idx)]
    if misses.empty:
        raise ValueError("no misclassified examples for this pair")
    row = misses.iloc[0]

    rgb, cam_true = cam_for_path(gradcam, row["filepath"], pair.true_idx, device)
    _, cam_pred = cam_for_path(gradcam, row["filepath"], pair.pred_idx, device)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    _panel(axes[0], rgb, f"image of\n{class_names[pair.true_idx]}")
    _panel(axes[1], overlay_cam(rgb, cam_true), f"CAM for TRUE\n{class_names[pair.true_idx]}")
    _panel(axes[2], overlay_cam(rgb, cam_pred), f"CAM for PRED\n{class_names[pair.pred_idx]}")
    tag = "same genus" if pair.same_genus else "different genus"
    fig.suptitle(f"Confusable pair ({tag}) - confused {pair.count}x", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, f"confusable_{pair.true_idx}_{pair.pred_idx}.png")


def fig_organism_vs_background(gradcam, df, class_names, device="cpu", n=4, out_dir=None):
    """Image, CAM overlay, and the Otsu mask, with the background-proxy numbers.
    Returns the figure plus a small metrics table."""
    correct, _ = select_correct_incorrect(df, n=n)
    rows = list(correct.iterrows())
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    records = []
    for i, (_, row) in enumerate(rows):
        rgb, cam = cam_for_path(gradcam, row["filepath"], int(row["pred"]), device)
        mask = cam_to_mask(cam)
        scores = background_score(cam)
        records.append({"filepath": row["filepath"], "class": class_names[int(row["true"])], **scores})
        _panel(axes[i, 0], rgb, class_names[int(row["true"])])
        _panel(axes[i, 1], overlay_cam(rgb, cam), "Grad-CAM")
        _panel(axes[i, 2], np.dstack([mask] * 3).astype(float),
               f"mask {scores['mask_area_fraction']:.2f}\noffset {scores['centroid_offset']:.2f}")
    fig.suptitle("Organism vs background: CAM mask + locality proxy", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "organism_vs_background.png")
    return fig, pd.DataFrame.from_records(records)


def fig_failure_cases(gradcam, df, class_names, device="cpu", n=4, out_dir=None):
    """Most confident mistakes: image, CAM for the wrong predicted class, CAM for the true
    class. This is what I use to write the failure-case claims."""
    rows = list(most_confident_mistakes(df, n=n).iterrows())
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (_, row) in enumerate(rows):
        rgb, cam_pred = cam_for_path(gradcam, row["filepath"], int(row["pred"]), device)
        _, cam_true = cam_for_path(gradcam, row["filepath"], int(row["true"]), device)
        _panel(axes[i, 0], rgb, f"true: {class_names[int(row['true'])]}")
        _panel(axes[i, 1], overlay_cam(rgb, cam_pred), f"CAM pred: {class_names[int(row['pred'])]}\nconf {row['confidence']:.2f}")
        _panel(axes[i, 2], overlay_cam(rgb, cam_true), "CAM for true class")
    fig.suptitle("Most confident failure cases", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "failure_cases.png")
