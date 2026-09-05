# Semiconductor Manufacturing Yield Prediction

## Executive Summary & Background

### Domain
**Semiconductor Manufacturing Process**

### Context & Problem Statement
A modern semiconductor manufacturing process operates under continuous surveillance, capturing thousands of signal variables from sensors and process measurement points across every production entity (wafer).

However, not all collected signals carry equal diagnostic value—many represent noise, redundant metrics, or uninformative constants. Having a massive pool of redundant signals complicates monitoring, increases computing overhead, and obscures critical root causes when yield excursions occur downstream[cite: 2].

### Project Objective
The primary goal of this project is to build an automated machine learning classification model that accurately predicts the **Pass/Fail yield** of manufacturing entities based on sensor readings[cite: 2].

Additionally, this project evaluates whether all original features are necessary for predictive accuracy[cite: 2]. By identifying key sensor signals impacting yield type, process engineers can:
* Pinpoint critical factors driving yield excursions[cite: 2].
* Increase manufacturing throughput and reduce defect learning cycles[cite: 2].
* Minimize per-unit production costs by flagging potential defects early in the line[cite: 2].

---

## Dataset Overview

The dataset (`signal-data.csv`) comprises real-time sensor measurements taken during wafer manufacturing[cite: 2]:

* **Total Instances:** 1,567 wafers[cite: 2]
* **Original Features:** 592 columns (including timestamp `Time` and target label `Pass/Fail`)[cite: 2]
* **Target Variable:** `Pass/Fail`
  * `-1`: Pass[cite: 2]
  * `1`: Fail[cite: 2]
* **Initial Data Types:** `float64` (590 features), `int64` (1 target), `str` (1 timestamp)[cite: 2]

---

## Data Pipeline & Preprocessing Workflow
---

## Exploratory Data Analysis (EDA)

### Class Imbalance Analysis
* **Class Distribution:**
  * **Pass (`-1`):** ~1,463 samples (~93.36%)[cite: 2]
  * **Fail (`1`):** ~104 samples (~6.64%)[cite: 2]
* **Key Finding:** Severe class imbalance present in the target variable[cite: 2]. Predicting the majority class baseline yields high raw accuracy (~93.4%) but fails to detect critical production failures[cite: 2].

### Feature Distribution & Multicollinearity
* **Skews & Outliers:** Sensor readings exhibit significant right/left skewness and extreme outliers typical of real-world industrial IoT sensors[cite: 2].
* **Feature Correlation:** Strong pairwise correlations exist across multiple sensor channels, confirming high redundancy among physical measurements[cite: 2].

---

## Model Development & Class Imbalance Handling

To counter severe class imbalance, models were evaluated across three data configurations:
1. **Unbalanced Baseline Data**
2. **SMOTE (Synthetic Minority Over-sampling Technique)**
3. **Class-Weight Adjustment (Cost-Sensitive Learning)**

### Primary Evaluation Metrics
Standard **Accuracy** is misleading for this domain. The pipeline prioritizes:
* **Recall (Sensitivity):** Maximizes detection of true defects (`1`), minimizing undetected faulty chips.
* **PR-AUC (Precision-Recall Area Under Curve):** Provides a robust performance measure for highly imbalanced binary classification.
* **F1-Score (Minority Class):** Balances Precision and Recall for the `Fail` class.

---

## Model Experiments & Performance Summary

Multiple machine learning algorithms were trained and cross-validated:

| Model Architecture | Resampling / Balancing | Fail Class Recall | Fail Class Precision | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline | Low | Moderate | Low | Moderate |
| **Random Forest** | SMOTE | Moderate | High | Moderate | High |
| **XGBoost Classifier** | Scale Pos Weight | High | High | High | High |
| **Support Vector Classifier (SVC)** | Class Weight (`balanced`) | Moderate | Moderate | Moderate | Moderate |

---

## Dimensionality Reduction (PCA & Feature Importance)

To test the core hypothesis—*whether all features are necessary for prediction*—dimensionality reduction techniques were applied:

### Principal Component Analysis (PCA)
* **Variance Retention:** 90% of total variance captured with a significantly reduced set of components compared to the clean 446-feature set[cite: 2].
* **Impact:** Reduced training latency significantly while maintaining competitive downstream classification metrics.

### Feature Selection / Top Signal Drivers
* Using tree-based feature importance (Random Forest / XGBoost) and L1-regularization (Lasso), the top predictive sensor channels were isolated[cite: 2].
* **Outcome:** Less than 15% of the total sensor suite contributes to over 80% of predictive power, confirming that significant hardware/compute bandwidth can be saved by focusing on key signals[cite: 2].

---

## Key Business Insights & Process Recommendations

1. **Focus Sensor Monitoring:** Process engineers can streamline automated equipment checks by focusing diagnostic alarms on top-ranking sensor features[cite: 2].
2. **Early Yield Excursion Interception:** Implementing high-recall models allows early line intervention, saving cost before wafers undergo expensive downstream packaging and testing[cite: 2].
3. **Sensor Infrastructure Optimization:** Low-variance and highly redundant sensors can be decommissioned or monitored at lower sampling frequencies without degrading yield prediction capability[cite: 2].

---

## How to Run & Reproduce

```python
# Import main dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv('signal-data.csv')

# 2. Run preprocessing
X = df.drop(columns=['Time', 'Pass/Fail'])
y = df['Pass/Fail']

# 3. Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model fitting with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)
