# COMP9517 Group 102
## Setup
Hello Group, I have made a basic setup for our workflow based on what the specification recommended.
1. **GitHub is for the Code.** (No Heavy files - restraints are in place to prevent such actions)
2. **Google Drive is for Data.** (Already made a sample data pack for our assignment by splitting it into 500 classes, 40 training, 10 validation and 10 test - *"COMP9517_Team_500_40_10"*)
3. **Google Colab - Use Your own noteboook.** To avoid conflict use your own notebooks

## Traditional handcrafted-feature method

The implementation is in `traditional_cv/` and the guided Colab workflow is
`notebooks/Traditional_Method.ipynb`. Generated models, cached features, metrics,
predictions, and figures go under `artifacts/` and must remain outside Git.

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

Open `notebooks/Enhanced_Traditional_Method.ipynb` for the validation-only enhanced
experiment. It uses 224px spatial HSV/LBP/HOG, spatial-pyramid SIFT-BoVW-512, three
hinge-loss SVM regularisation settings, and four regularised Random-Forest screens.
Outputs are isolated under `artifacts/traditional_enhanced/`; the baseline is preserved.
