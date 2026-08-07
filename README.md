# COMP9517 Group 102
## Setup
Hello Group, I have made a basic setup for our workflow based on what the specification recommended.
1. **GitHub is for the Code.** (No Heavy files - restraints are in place to prevent such actions)
2. **Google Drive is for Data.** (Already made a sample data pack for our assignment by splitting it into 500 classes, 40 training, 10 validation and 10 test - *"COMP9517_Team_500_40_10"*)
3. **Google Colab - Use Your own noteboook.** To avoid conflict use your own notebooks

## Traditional handcrafted-feature method

The reusable implementation is in `traditional_cv/`. The executed primary workflow and
results are in `Traditional & Advanced Methods/Traditional_Method_Submission.ipynb`.
Generated models, cached features, metrics, predictions, and figures are written under
`artifacts/traditional/`. The `artifacts/` directory is Git-ignored because it contains
large reproducible outputs.

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-traditional.txt
python -m traditional_cv.cli audit --data Team_Dataset
python -m traditional_cv.cli benchmark --data Team_Dataset --sample-size 500
```

Run the time-bounded comprehensive experiment. It reuses saved feature arrays and models:

```bash
python -m traditional_cv.cli streamlined --data Team_Dataset
python -m traditional_cv.cli test --data Team_Dataset
python -m traditional_cv.cli robustness --data Team_Dataset
```

The `streamlined` command performs six controlled experiments: four individual
descriptors with a fixed stochastic linear SVM, fused features with the same SVM,
and fused features with Random Forest. The older `features` and `fusion` commands
remain available only for reproducing the original exhaustive search and are not
part of the time-bounded notebook workflow.

The `test` command creates `TEST_EVALUATED.json` and refuses to run twice. Complete all
validation-based decisions before using it. Use `--hash-files` with `audit` for the final
byte-level leakage check. Run automated tests with `pytest -q`.

### Enhanced fully traditional follow-up

Open `Traditional & Advanced Methods/Enhanced_Traditional_Method.ipynb` for the
validation-only enhanced experiment. It uses 224px spatial HSV/LBP/HOG,
spatial-pyramid SIFT-BoVW-512, three
hinge-loss SVM regularisation settings, and four regularised Random-Forest screens.
Outputs are isolated under `artifacts/traditional_enhanced/`; the baseline is preserved.

## Robustness to image degradation (Advanced Topic 2)

The traditional robustness experiment is provided in
`Traditional & Advanced Methods/Traditional_Robustness.ipynb` and can also be run with:

```bash
python -m traditional_cv.cli robustness --data Team_Dataset
```

The experiment loads the already trained and frozen best model. Only the 5,000 held-out test images are degraded using four practical image
corruptions at four severity levels:

- Gaussian noise: standard deviation `0.02, 0.05, 0.10, 0.20`;
- Gaussian blur: sigma `1, 2, 3, 5`;
- motion blur: kernel size `3, 7, 11, 15`;
- JPEG compression: quality `75, 50, 25, 10`.

The clean test set is the severity-zero reference. Top-1 accuracy and macro-F1 are measured
at every level and plotted against severity. This shows how quickly performance falls as
image quality decreases while keeping the original train, validation, and test split intact.


## Grad-CAM explainability (Advanced Topic 1, deep-learning only)

`grad_cam/` provides a hand-implemented Grad-CAM / Grad-CAM++ (via forward/backward
hooks, no external CAM library) that plugs into the trained CNNs documented under
`Deep_Learning_Pipeline/` (ResNet18/50, ConvNeXt-Tiny, and Swin-T). The guided Colab
workflow is `Traditional & Advanced Methods/Grad_CAM_Explainability.ipynb`. The current
consolidated deep-learning workflow is
`Deep_Learning_Pipeline/New Notebook/Deep_Learning_Notebook.ipynb`; the original
per-architecture notebooks are retained under `Deep_Learning_Pipeline/Old Notebooks/`.

It answers the four required questions: correct-vs-incorrect maps, confusable
same-genus pairs, organism-vs-background attention (Otsu-mask proxy, since iNat2021 has
no GT masks), and concrete failure-case claims. It **reuses** each run's
`*_predictions.csv` to pick examples, so nothing is re-inferred.

```bash
python -m pip install -r requirements-gradcam.txt
pytest -q tests/test_gradcam.py        # smoke test, no weights/data needed
```

To produce figures, open the notebook and set `ARCH`, `WEIGHTS_PATH` (the saved
`state_dict` .pth), `DATA_DIR` (`Team_Dataset/`), and `PRED_CSV`. Figures are written to
`artifacts/grad_cam/` (git-ignored). Grad-CAM target layers: `layer4[-1]` (ResNet),
`features[-1]` (ConvNeXt), `features[-1]` + reshape (Swin).
