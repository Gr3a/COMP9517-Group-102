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

## Deep Learning Method

`Deep_Learning_Notebook.ipynb` trains and evaluates seven variations of CNN/transformer models
forclassification of 500 iNaturalist-2021 species (Sub-set created from `Data_Setup.ipynb`), and produces Performance metrics
and figure used in the report. 

### Two ways to run it

**1. Train a run from scratch (Local).** Set `TRAIN = True` and set `RUN_ID` to one of
the seven keys of the model that you wish to train. It will then train, evaluate on the test set, and append its row to
the results table. From-scratch runs take about 40–70 minutes while pretrained runs around 15–40.

**2. Reproduce results from the saved predictions (from history).** The seven runs have
already been trained and their `*_predictions.csv` and `*_history.csv` files are included (or are created if it is trained with the first method beforehand).
With `TRAIN = False` the notebook rebuilds the full results table, all figures, and the
confusion analysis from those CSVs. Makes it far easier to recover stats, even after closing and re-opening.


### Requirements

Python 3.10+ with `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
and `Pillow`. A CUDA GPU was used for local training (and times were given as such); reproducing from the CSVs runs on
CPU. For this pipeline, Sub-set data was kept locally, but it can be performed online using Colab.

### Before running: personalise the CONFIG cell

Three paths, near the top of the notebook:

```python
ROOT     = Path(r'C:\path\to\COMP9517')                 # searched recursively for the run CSVs
DATA_DIR = Path(r'C:\path\to\...\Team_Dataset')         # folder containing train/ val/ test/ of data sub-set
OUT_DIR  = ROOT / 'report_outputs'                      # figures + summary.csv are written here
```

`ROOT` is searched recursively, so the existing nested result folders (for example
`ResNet50/From Scratch/No Aug/`) are found in place — nothing needs to be moved or renamed.
Leave the two mode switches as they are unless you are training:

```python
TRAIN    = False                          # True only to train a run
RUN_ID   = 'convnext_tiny_Pre_Full_Aug'   # the model you wish to train (if TRAIN is active)
BEST_RUN = 'convnext_tiny_Pre_Full_Aug'   # run used for the detailed error analysis (Best Performing)
```

### Data layout expected

```
Team_Dataset/
├── train/<species>/*.jpg     40 images per class 
├── val/<species>/*.jpg       10 images per class 
└── test/<species>/*.jpg      10 images per class 
```

500 identically named species folders per split. More information on how in the `Data_Setup.ipynb` script.

### Running

Open the notebook and Run All. The sections run top to bottom:

- **Setup / Configuration / Helpers** — imports and the `RUNS` registry (the single place any
  experiment is defined; every results row derives its labels from it).
- **0. Prediction Finder** — lists which runs have predictions/histories on disk and flags any
  file whose name does not match `RUNS`.
- **1. Data Check** — confirms the three splits share identical class lists and that no image
  appears in more than one split (leakage check), then writes `class_list.txt`.
- **2. Training** — `train_run` and `predict`. Only trains if `TRAIN = True`. Otherwise this
  section just defines the functions and reports that it is using the existing predictions.
- **3. Performance Metrics** — rebuilds `summary.csv` (top-1, top-5, balanced accuracy, macro
  precision/recall/F1, species-never-correct) from the prediction files.
- **3.1 Training Time** — fills the training-time column. The per-architecture seconds/epoch
  have already been measured and the column is populated. Times  are extrapolated from 
  single-epoch due to time restraints.
- **3.2–3.5** — loss/accuracy curves, the confusion matrix (full 500-class view and an
  annotated zoom over the most-confused species), per-class F1, qualitative success/failure
  grids, and the cross-run comparison table and accuracy-versus-cost plot.

### Outputs

Everything lands in `OUT_DIR`: `summary.csv` (the results table), `class_list.txt`, and the
figures (`combined_curves.png`, `*_confusion_full.png`, `*_confusion_zoom.png`,
`*_perclass_f1.png`, `*_qualitative_success.png`, `*_qualitative_failure.png`,
`accuracy_vs_time.png`). 

### The seven runs

| RUN_ID | arch | init | aug | epochs | lr |
|---|---|---|---|---|---|
| r50_Scratch_No_Aug | resnet50 | scratch | none | 60 | 1e-3 |
| r50_Scratch_Full_Aug | resnet50 | scratch | full | 60 | 1e-3 |
| r50_Pre_Full_Aug | resnet50 | pretrained | full | 20 | 1e-4 |
| r18_Scratch_Full_Aug | resnet18 | scratch | full | 60 | 1e-3 |
| r18_Pre_Full_Aug | resnet18 | pretrained | full | 20 | 1e-4 |
| convnext_tiny_Pre_Full_Aug | convnext_tiny | pretrained | full | 20 | 1e-4 |
| swin_t_Pre_Full_Aug | swin_t | pretrained | full | 20 | 1e-4 |

Held constant across every run: batch size 32, AdamW, weight decay 1e-4, cosine annealing,
label smoothing 0.1, seed 42, 224×224 input, checkpoint selected on validation accuracy.

### Third-party code and attribution

PyTorch / torchvision provide the architectures and the ImageNet-1k pretrained weights
(`weights='DEFAULT'`, ImageNet-1k only). scikit-learn provides the precision/recall/F1, 
balanced accuracy and confusion-matrix routines. 
Dataset: Van Horn et al., *Benchmarking Representation Learning for Natural World Image Collections*, CVPR 2021.

### Not included

Trained checkpoints (`.pth`), dataset images, and generated figures are excluded to stay
under the 40 MB code-submission limit. Re-running the notebook regenerates the figures and CSVs (For at least one TRAIN complete).

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
