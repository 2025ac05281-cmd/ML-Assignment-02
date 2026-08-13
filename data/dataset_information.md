# Dataset Information

## Selected Dataset

Breast Cancer Wisconsin (Diagnostic), originally published through the UCI Machine Learning Repository and available locally through `sklearn.datasets.load_breast_cancer`.

- UCI page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn page: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Suitability for the Assignment

- Public classification dataset: Yes
- Instances: 569, which is greater than the required minimum of 500
- Features: 30, which is greater than the required minimum of 12
- Target type: Binary classification
- Missing values: None in the scikit-learn copy

## Feature Groups

Ten measurements are calculated for each cell nucleus:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each measurement, the dataset contains the mean, standard error, and worst value, producing 30 numerical features.

## Target Convention Used in This Project

The source copy in scikit-learn uses `0 = malignant` and `1 = benign`. This project reverses that encoding so that:

- `0 = Benign`
- `1 = Malignant`

Malignant is therefore the positive class for Precision, Recall, F1, and AUC.

## Training and Test Partition

The supplied `test_data.csv` contains 114 labelled rows. The training script matches those rows against all 30 source features and removes them before fitting. The complement contains 455 rows and is used as the training partition.

This method provides three safeguards:

- The supplied test data remains unseen during training.
- All five models are evaluated on exactly the same records.
- The reported metrics are reproducible from the files in the repository.

## CSV Schema

The labelled test file contains 31 columns:

- 30 numeric feature columns
- 1 integer target column named `diagnosis`

A prediction-only upload may omit `diagnosis`, but all 30 feature columns are required.
