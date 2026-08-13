# Machine Learning Assignment 2

## Breast Cancer Diagnostic Classification and Streamlit Deployment

**Student:** Divyendu Shekhar  
**BITS ID:** 2025AC05281  
**BITS Email:** 2025ac05281@wilp.bits-pilani.ac.in  
**Programme:** M.Tech (Artificial Intelligence and Machine Learning)  
**Section:** 2  

---

## a. Problem Statement

The objective of this project is to build, evaluate, compare, and deploy multiple machine learning classification models on a public dataset. The selected problem is to classify a breast-mass record as **Benign (0)** or **Malignant (1)** using numerical characteristics calculated from digitized images of fine-needle aspirate samples.

Five classification algorithms prescribed in the assignment are implemented on the same training and test partitions:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model is evaluated using all six required metrics: **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**. The trained models are integrated into a customized Streamlit application that supports CSV upload, model selection, evaluation, confusion-matrix display, classification-report display, prediction output, and downloadable results.

---

## b. Dataset Description

### Dataset Name

**Breast Cancer Wisconsin (Diagnostic)**

### Public Source

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

### Dataset Characteristics

| Item | Value |
|---|---:|
| Classification type | Binary classification |
| Total source instances | 569 |
| Input features | 30 numerical features |
| Training instances | 455 |
| Supplied test instances | 114 |
| Missing values | 0 |
| Target column | `diagnosis` |
| Class 0 | Benign |
| Class 1 | Malignant |
| Positive class used for Precision/Recall/F1 | Malignant (1) |

The 30 input variables describe properties of cell nuclei. Ten base measurements are represented as mean, standard error, and worst-value groups:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

The supplied `test_data.csv` contains 114 labelled records and all 30 required features. To prevent data leakage, those exact 114 records are identified in the full 569-row source dataset and excluded before model fitting. The remaining 455 records form the training partition. The source values are loaded locally through `sklearn.datasets.load_breast_cancer`, so training does not require a network download.

### Target Mapping

The scikit-learn copy uses `0 = malignant` and `1 = benign`. For this project, the target is deliberately converted to:

- `0 = Benign`
- `1 = Malignant`

This makes malignant cases the positive class for Precision, Recall, F1, and ROC-AUC calculations.

---

## c. GitHub Repository Link

- GitHub Repository: https://github.com/2025ac05281-cmd/ML-Assignment-02
---

## d. Models Used and Evaluation Results

All models were trained on the same 455 training rows and evaluated on the same 114 supplied test rows. Standardization is included inside the model pipeline for Logistic Regression, kNN, and Gaussian Naive Bayes. Decision Tree and Random Forest operate on the original feature scale.

### Model Configuration

| ML Model | Main Configuration |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=5000, random_state=42)` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=2`, `random_state=42` |
| kNN | `StandardScaler` + `KNeighborsClassifier(n_neighbors=7, weights='distance')` |
| Naive Bayes | `StandardScaler` + `GaussianNB()` |
| Random Forest (Ensemble) | 300 trees, balanced class weights, `random_state=42` |

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9561 | **0.9970** | **1.0000** | 0.8810 | 0.9367 | 0.9076 |
| Decision Tree | 0.8947 | 0.8545 | 0.8947 | 0.8095 | 0.8500 | 0.7715 |
| kNN | 0.9386 | 0.9907 | 0.9730 | 0.8571 | 0.9114 | 0.8688 |
| Naive Bayes | 0.9035 | 0.9854 | 0.9189 | 0.8095 | 0.8608 | 0.7911 |
| Random Forest (Ensemble) | **0.9649** | 0.9955 | 0.9750 | **0.9286** | **0.9512** | **0.9245** |

### Confusion-Matrix Summary

The matrix convention is:

```text
[[True Benign, False Malignant],
 [False Benign, True Malignant]]
```

| ML Model Name | Confusion Matrix | Incorrect Predictions |
|---|---|---:|
| Logistic Regression | `[[72, 0], [5, 37]]` | 5 |
| Decision Tree | `[[68, 4], [8, 34]]` | 12 |
| kNN | `[[71, 1], [6, 36]]` | 7 |
| Naive Bayes | `[[69, 3], [8, 34]]` | 11 |
| Random Forest (Ensemble) | `[[71, 1], [3, 39]]` | 4 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression performed very strongly after standardization. It achieved the highest AUC of 0.9970 and perfect malignant-class precision of 1.0000, meaning it did not classify any benign record as malignant. However, it missed five malignant records, so its recall was lower than Random Forest. The result indicates that the transformed feature space is close to linearly separable for many records. |
| Decision Tree | The Decision Tree was simple and interpretable, but it achieved the lowest AUC and MCC among the models. It produced four false positives and eight false negatives. A single tree can form clear decision rules, but it is sensitive to the training partition and may not generalize as smoothly as an ensemble. |
| kNN | kNN benefited substantially from feature standardization because its predictions are based on distances. It achieved high AUC and precision, with only one false positive. It missed six malignant records, which reduced recall and F1 compared with Random Forest. Its prediction cost also grows with the number of stored training samples. |
| Naive Bayes Classifier (Gaussian) | Gaussian Naive Bayes achieved a high AUC of 0.9854 despite its strong conditional-independence assumption. The dataset contains correlated measurements such as radius, perimeter, and area, so the independence assumption is not fully satisfied. This contributed to lower accuracy, recall, and MCC than Logistic Regression, kNN, and Random Forest. |
| Naive Bayes Classifier (Multinomial) | Achieved perfect precision (1.0000) with zero false positives, but missed nearly half the malignant cases (recall 0.5714). Uses MinMaxScaler to keep features non-negative as required by MultinomialNB. Lower MCC (0.6761) and F1 (0.7273) than Gaussian NB confirm that count-based assumptions are less suitable for continuous cell-nucleus measurements. |
| Ensemble Model - Random Forest | Random Forest achieved the best overall balance. It obtained the highest Accuracy, Recall, F1, and MCC, while maintaining an AUC of 0.9955. It made only four errors: one false positive and three false negatives. Combining many trees reduced the instability seen in the single Decision Tree and captured nonlinear interactions between features. |
| **Overall Winner for the Dataset** | **Ensemble Model - Random Forest** is selected as the overall winner because it achieved the highest MCC (0.9245), Accuracy (0.9649), Recall (0.9286), and F1 Score (0.9512). MCC was used as the primary selection criterion because it considers all four confusion-matrix outcomes and remains informative when class frequencies are unequal. |

---
