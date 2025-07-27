## Customer Loan Analytics & Prediction (Analytics_olympird.ipynb)

This repository contains a Jupyter notebook focused on advanced analytics and classification for a large loan/customer dataset. The pipeline includes extensive data exploration, cleaning, feature engineering, imputation, encoding, classification modeling, and evaluation.

## Overview

- **Goal:** Analyze customer and loan data, explore patterns, and predict target flags related to primary and final loan closure.
- **Techniques:** Data imputation, label and one-hot encoding, feature analysis, supervised learning (Decision Tree, Random Forest, KNN, SVM, Logistic Regression).
- **Data:** Private customer loans data, split into `train.csv` and `test.csv`.

## Workflow Summary

### 1. Library Imports

The notebook imports all essential Python libraries for data handling, visualization, preprocessing, modeling, and evaluation.

### 2. Data Loading

- Loads `train.csv` and `test.csv` into pandas DataFrames.
- Initial head and descriptive statistics output to understand dataset dimensions, feature types, and distributions.

### 3. Data Exploration

- Prints a sample of rows and computes summary statistics (mean, std, quantiles, etc.).
- Initial inspection reveals both numeric and categorical features, some missing values, and several encoded columns.

### 4. Data Preprocessing

- **Missing Value Handling:** Uses `KNNImputer` to fill missing values in numeric features.
- **Categorical Encoding:** Utilizes `LabelEncoder` and `OneHotEncoder` as appropriate for categorical columns like names and certain encoded columns.
- **Data Cleaning:** Considers columns that may be irrelevant or redundant and performs cleanup (potentially dropping columns or fixing anomalies, as inferred).
- **Feature Engineering:** Ensures all relevant features are in proper numeric form for modeling.

### 5. Train-Test Splitting

- Splits the training data for testing and validation (using `train_test_split` from scikit-learn).

### 6. Classification Models

Implements and trains multiple supervised learning algorithms:

- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **KNeighborsClassifier**
- **LogisticRegression**
- **SVC (Support Vector Machine)**

Each model is fit on training data and evaluated.

### 7. Model Evaluation

- Computes and prints accuracy, precision, recall, F1-score, and confusion matrix for model performance.
- Uses sklearn functions like `classification_report` and `confusion_matrix` for detailed diagnostics.

### 8. Prediction & Export

- Predictions for the test set are generated as per modeling pipeline.
- Results can be saved for downstream evaluation or submission.

### 9. Visualization

- Uses matplotlib, seaborn, and plotly for exploratory plots (distribution, correlation, etc.)

## Usage

1. Clone this repository.
2. Place the provided `train.csv` and `test.csv` in the repository's root directory.
3. Install the requirements as described below.
4. Launch JupyterLab or Jupyter Notebook.
5. Open and run `Analytics_olympird.ipynb` sequentially.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

Install via pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly regex
```

## Folder Structure

- `Analytics_olympird.ipynb` — The main notebook containing all code, analyses, and visualizations.
- `train.csv`, `test.csv` — Data files required for analysis.

## Noteworthy Code Steps

- **KNN Imputation:** Effectively fills missing numeric values, preserving dataset integrity.
- **Encoding:** Label/OneHot Encoding transforms categorical features for ML compatibility.
- **Multiple Classifiers:** Systematic comparison of several classification models for robust prediction.
- **Detailed Evaluation:** Precision, recall, F1-score, and confusion matrices for each model, enabling informed model selection.

## Notes

- Runs on standard CPUs; no GPU compute required.
- Models and preprocessing steps can be easily extended or modified for additional features.
- Recommended for tabular classification problems with mixed feature types.

## Results & Next Steps

- Models achieve strong predictive accuracy (see notebook outputs).
- Further improvements could include hyperparameter tuning, additional feature engineering, and handling class imbalance.

*For issues or suggestions, open an issue or submit a pull request.*
