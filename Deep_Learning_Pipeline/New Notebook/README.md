# Deep Learning Pipeline — How to Run

`Deep_Learning_Notebook.ipynb` trains and evaluates seven variations of CNN/transformer models
forclassification of 500 iNaturalist-2021 species (Sub-set created from `Data_Setup.ipynb`), and produces Performance metrics
and figure used in the report. 

## Two ways to run it

** 1. Train a run from scratch (Local).** Set `TRAIN = True` and set `RUN_ID` to one of
the seven keys of the model that you wish to train. It will then train, evaluate on the test set, and append its row to
the results table. From-scratch runs take about 40–70 minutes while pretrained runs around 15–40.

**2. Reproduce results from the saved predictions (from history).** The seven runs have
already been trained and their `*_predictions.csv` and `*_history.csv` files are included (or are created if it is trained with the first method beforehand).
With `TRAIN = False` the notebook rebuilds the full results table, all figures, and the
confusion analysis from those CSVs. Makes it far easier to recover stats, even after closing and re-opening.


## Requirements

Python 3.10+ with `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
and `Pillow`. A CUDA GPU was used for local training (and times were given as such); reproducing from the CSVs runs on
CPU. For this pipeline, Sub-set data was kept locally, but it can be performed online using Colab.

## Before running: personalise the CONFIG cell

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

## Data layout expected

```
Team_Dataset/
├── train/<species>/*.jpg     40 images per class 
├── val/<species>/*.jpg       10 images per class 
└── test/<species>/*.jpg      10 images per class 
```

500 identically named species folders per split. More information on how in the `Data_Setup.ipynb` script.

## Running

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

## Outputs

Everything lands in `OUT_DIR`: `summary.csv` (the results table), `class_list.txt`, and the
figures (`combined_curves.png`, `*_confusion_full.png`, `*_confusion_zoom.png`,
`*_perclass_f1.png`, `*_qualitative_success.png`, `*_qualitative_failure.png`,
`accuracy_vs_time.png`). 

## The seven runs

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

## Third-party code and attribution

PyTorch / torchvision provide the architectures and the ImageNet-1k pretrained weights
(`weights='DEFAULT'`, ImageNet-1k only). scikit-learn provides the precision/recall/F1, 
balanced accuracy and confusion-matrix routines. 
Dataset: Van Horn et al., *Benchmarking Representation Learning for Natural World Image Collections*, CVPR 2021.

## Not included

Trained checkpoints (`.pth`), dataset images, and generated figures are excluded to stay
under the 40 MB code-submission limit. Re-running the notebook regenerates the figures and CSVs (For at least one TRAIN complete).

