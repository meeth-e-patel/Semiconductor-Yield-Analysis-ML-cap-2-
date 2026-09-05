# Semiconductor-Yield-Analysis-ML

## Executive Summary
This project focuses on predicting pass/fail yield types in semiconductor manufacturing using sensor signal data. Modern fabrication processes monitor hundreds of variables, many of which contain noise or redundant information. By applying automated machine learning pipelines, feature selection, missing value imputation, and class balancing, we identify critical signal indicators to optimize process throughput and minimize unit production costs.

---

## Workflow & Pipeline
1. **Data Exploration**: Analyzed raw dataset containing 1,567 entities across 591 sensor signals.
2. **Data Cleansing**: Dropped non-predictive timestamps, removed 116 zero-variance (constant) features, removed high-missingness attributes (>50%), and imputed remaining missing values using median strategies.
3. **Data Analysis & Visualization**: Executed univariate, bivariate, and correlation heatmap analysis to evaluate skewed sensor distributions and feature interactions.
4. **Data Pre-processing**: Implemented stratified train-test splitting (80/20), standardized features using `StandardScaler`, and resolved severe target imbalance (~93.4% Pass vs ~6.6% Fail) using **SMOTE**.
5. **Model Evaluation & Tuning**: Evaluated multiple classifiers using 5-fold `GridSearchCV`:
   * **Random Forest Classifier** (Selected Model)
   * **Support Vector Machine (SVM)**
   * **Gaussian Naïve Bayes**
6. **Model Deployment**: Exported optimal trained model (`.pkl`) along with `StandardScaler` and `SimpleImputer` preprocessing artifacts.

---

## Key Tech Stack
* **Language**: Python 3.x
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (`SMOTE`), `matplotlib`, `seaborn`, `joblib`
* **Environment**: JupyterLab

---

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Semiconductor-Yield-Prediction-ML.git](https://github.com/YourUsername/Semiconductor-Yield-Prediction-ML.git)
