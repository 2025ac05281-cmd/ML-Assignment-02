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

> **Educational-use notice:** This application is an academic machine learning demonstration. Its predictions must not be used for medical diagnosis, clinical decisions, or treatment.

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

**GitHub Repository:** `[PASTE CLICKABLE GITHUB REPOSITORY LINK HERE]`

Recommended repository name:

```text
ml-assignment-2-breast-cancer-classification
```

---

## Live Streamlit Application Link

**Streamlit Community Cloud App:** `[PASTE CLICKABLE STREAMLIT APP LINK HERE]`

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
| Decision Tree | The Decision Tree was simple and interpretable, but it achieved the lowest AUC and MCC among the five models. It produced four false positives and eight false negatives. A single tree can form clear decision rules, but it is sensitive to the training partition and may not generalize as smoothly as an ensemble. |
| kNN | kNN benefited substantially from feature standardization because its predictions are based on distances. It achieved high AUC and precision, with only one false positive. It missed six malignant records, which reduced recall and F1 compared with Random Forest. Its prediction cost also grows with the number of stored training samples. |
| Naive Bayes | Gaussian Naive Bayes achieved a high AUC of 0.9854 despite its strong conditional-independence assumption. The dataset contains correlated measurements such as radius, perimeter, and area, so the independence assumption is not fully satisfied. This contributed to lower accuracy, recall, and MCC than Logistic Regression, kNN, and Random Forest. |
| Random Forest (Ensemble) | Random Forest achieved the best overall balance. It obtained the highest Accuracy, Recall, F1, and MCC, while maintaining an AUC of 0.9955. It made only four errors: one false positive and three false negatives. Combining many trees reduced the instability seen in the single Decision Tree and captured nonlinear interactions between features. |
| **Overall Winner for the Dataset** | **Random Forest (Ensemble)** is selected as the overall winner because it achieved the highest MCC (0.9245), Accuracy (0.9649), Recall (0.9286), and F1 Score (0.9512). MCC was used as the primary selection criterion because it considers all four confusion-matrix outcomes and remains informative when class frequencies are unequal. |

---

## Evaluation Metric Definitions

- **Accuracy:** Proportion of all test records classified correctly.
- **AUC:** Area under the ROC curve; measures ranking ability across classification thresholds.
- **Precision:** Among records predicted as malignant, the proportion that are actually malignant.
- **Recall:** Among actual malignant records, the proportion identified correctly.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **MCC:** Correlation-style score based on all four confusion-matrix cells; ranges from -1 to +1.

---

## Streamlit Application Features

The Streamlit app implements all features requested in the assignment:

- CSV dataset upload option for test data
- Option to use the included `test_data.csv`
- Model-selection dropdown
- Precomputed comparison table for all five models
- Accuracy, AUC, Precision, Recall, F1, and MCC display
- Confusion matrix
- ROC curve
- Classification report
- Predictions and class probabilities
- Comparison of all models on any uploaded labelled dataset
- Download button for prediction results
- Input-schema validation and clear error messages
- Customized page layout, visual theme, assignment profile, and dataset guide

The app accepts either:

1. A labelled CSV containing all 30 feature columns plus `diagnosis`, or
2. A feature-only CSV containing the 30 feature columns.

When `diagnosis` is present, the app calculates all evaluation metrics. When it is absent, the app displays predictions and probabilities only.

---

## Repository Structure

```text
ML_Assignment_2_Divyendu_Shekhar/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── sample_upload_without_target.csv
├── SUBMISSION_LINKS.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── dataset_information.md
├── model/
│   ├── train_models.py
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest_ensemble.joblib
│   ├── feature_names.json
│   ├── model_metrics.json
│   ├── classification_reports.json
│   ├── confusion_matrices.json
│   └── training_metadata.json
├── notebook/
│   └── ML_Assignment_2_Divyendu_Shekhar.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── classification_reports.txt
│   ├── all_model_predictions.csv
│   ├── model_metric_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices/
│   └── predictions/
├── docs/
│   ├── BITS_VIRTUAL_LAB_EXECUTION_GUIDE.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── GITHUB_COMMIT_PLAN.md
│   └── FINAL_SUBMISSION_CHECKLIST.md
├── tests/
│   └── test_project_artifacts.py
└── submission/
    ├── Final_Submission_Report_DRAFT.docx
    └── Final_Submission_Report_DRAFT.pdf
```

---

## Local Execution

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Reproduce model training and saved results

```bash
python model/train_models.py
```

The script validates the supplied test file, reconstructs the leakage-free training partition, trains every model, prints the comparison table, and recreates all saved model and result files.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, normally `http://localhost:8501`.

### 5. Install development tools and run project checks

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Streamlit Community Cloud Deployment

1. Push the complete project to a public GitHub repository.
2. Sign in to Streamlit Community Cloud using the same GitHub account.
3. Select **Create app** or **New app**.
4. Choose the repository and the `main` branch.
5. Set the application file to `app.py`.
6. Deploy the application.
7. Open the public app link and test both the included test file and an uploaded CSV.
8. Add the live app link to this README and to the final submission PDF.

No secret key or external API is required.

---

## BITS Virtual Lab Execution Evidence

The assignment must be executed on BITS Virtual Lab. A suitable single screenshot should visibly include:

- The BITS Virtual Lab desktop or browser environment
- The terminal command, such as `python model/train_models.py` or `streamlit run app.py`
- Successful model output or the running Streamlit interface
- Enough context to show that the assignment was genuinely executed on the Virtual Lab

Insert this screenshot in the marked location in `submission/Final_Submission_Report_DRAFT.docx`, update the two mandatory links, export the document to PDF, and verify that the links are clickable.

---

## Reproducibility Notes

- Random seed: `42`
- The supplied 114 test records are never used for fitting.
- Scaling is fitted only on the training data because it is contained inside scikit-learn pipelines.
- All model objects are saved with Joblib.
- The positive class is consistently defined as Malignant (`1`).
- Package versions are pinned in `requirements.txt` to match the saved artifacts.
- Training metadata records feature names, row counts, model filenames, class mapping, winner, and software versions.

---

## Final Submission Checklist

- [ ] GitHub repository link added and tested
- [ ] Live Streamlit app link added and tested
- [ ] App opens without an error
- [ ] `test_data.csv` upload works
- [ ] Model-selection dropdown works
- [ ] All six metrics are visible
- [ ] Confusion matrix and classification report are visible
- [ ] All five models and saved artifacts are present
- [ ] One BITS Virtual Lab execution screenshot is inserted
- [ ] README content is included in the submitted PDF
- [ ] PDF links are clickable
- [ ] Final PDF is submitted before the deadline

---

## References

1. UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. scikit-learn, `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. scikit-learn model evaluation documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
4. Streamlit documentation: https://docs.streamlit.io/
