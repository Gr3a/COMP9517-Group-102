# Traditional and Advanced Methods

## Traditional method

`Traditional_Method_Submission.ipynb` is the main handcrafted computer-vision pipeline. Images are represented using four descriptors:

- HSV colour histograms for colour information;
- LBP for local texture;
- HOG for edges and shape;
- SIFT Bag of Visual Words for local visual patterns.

Each descriptor is first evaluated separately with a linear SVM. The descriptors are then combined into one fused feature vector, and the fused representation is compared using a linear SVM and a Random Forest. Validation macro-F1 is used to select the best configuration. The selected model is frozen before it is evaluated on the held-out test set. This keeps training, validation, and test data separate.

The traditional method performs much worse than the pretrained deep-learning models because handcrafted descriptors cannot capture all the subtle visual differences between 500 similar species. However, it provides a useful classical baseline and satisfies the required comparison of multiple descriptors and classifiers.

## Enhanced traditional experiment

`Enhanced_Traditional_Method.ipynb` investigates whether higher-resolution and spatially richer handcrafted features improve the baseline. It uses enhanced versions of HSV, LBP, HOG, and SIFT Bag of Visual Words and compares classifier settings using the validation set. The main traditional notebook remains the official baseline and final test evaluation.

## Robustness to test-time degradation

`Traditional_Robustness.ipynb` implements the advanced robustness study. It loads the already trained and frozen traditional model, degrades only the held-out test images, and evaluates the model without retraining or changing its parameters.

The test images are examined under four practical degradation types at several severity levels:

- Gaussian noise;
- Gaussian blur;
- motion blur;
- JPEG compression.

For every degradation level, the notebook reports top-1 accuracy and macro-F1 and plots how performance changes as degradation becomes stronger. The clean test result is used as the severity-zero reference. This experiment measures how sensitive the handcrafted representation is to image-quality changes while avoiding data leakage.


