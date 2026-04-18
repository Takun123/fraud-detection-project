# Fraud Detection Project

## Overview
Credit card fraud costs financial institutions billions annually, 
making automated detection a critical business need. This project 
builds an end-to-end ML pipeline using XGBoost to detect fraudulent 
transactions, with threshold tuning from 0.5 to 0.77 to optimize 
the precision-recall balance. The key challenge was severe class 
imbalance — fraud represents only 0.17% of transactions — which 
causes naive models to ignore fraud entirely and required specialized 
handling throughout.

## Dataset
Source: Kaggle Credit Card Fraud Dataset
- 284,315 transactions (after deduplication)
- 492 fraud cases (0.17% — highly imbalanced)
- 30 features: Time, Amount, V1-V28 (PCA-anonymized)
- No missing values

Note: Dataset not included (150MB). Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place file at: data/raw/creditcard.csv


## Methodology

### 1. Exploratory Data Analysis
- Original dataset: 284,807 rows, 31 columns
- Removed 1,081 duplicate rows → 283,726 rows remaining
- Class imbalance confirmed: 99.83% legitimate, 0.17% fraud
- Amount is heavily right-skewed (mean=$88, max=$25,691)
- 1,808 zero-amount transactions identified (25 fraud — kept as valid probing transactions)

### 2. Preprocessing
- Removed duplicates before splitting to prevent data leakage
- Stratified 80/20 train/test split to preserve fraud ratio in both sets
- StandardScaler applied to Amount and Time only — fit on train, applied to both
- Fraud rate preserved: 0.0017 in both train and test sets

### 3. Modeling
Three models trained with class imbalance handling:

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.87 | 0.11 | 0.92 |
| XGBoost | 0.93 | 0.78 | 0.85 | 0.89 |
| Random Forest | 0.97 | 0.71 | 0.82 | 0.85 |

**Selected: XGBoost** — Logistic Regression had highest AUC-ROC but precision of 0.06 
means 94% of fraud alerts would be false alarms. XGBoost delivers the best 
precision-recall balance for real deployment.

### 4. Threshold Tuning
- Default threshold (0.5): F1 = 0.85, Precision = 0.93, Recall = 0.78
- Optimal threshold (0.77): F1 = 0.86, Precision = 0.96, Recall = 0.78
- Same recall, higher precision — fewer false alarms at no cost to fraud detection

### 5. Explainability (SHAP)
- V14 is the most important feature — SHAP values reach approximately -6
- V4, V12, V10 are next most important
- Amount has almost no influence on predictions

## Results
- **74/95 fraud cases correctly detected** on test set
- **Only 3 legitimate transactions wrongly flagged**
- Final model: XGBoost at threshold 0.77

## Limitations
- V1-V28 are PCA-transformed — real feature meaning is unknown, 
  limiting business explainability for regulators
- Optimal threshold chosen to maximize F1, not business cost function
- Dataset is from 2013 — fraud patterns may have evolved

## How to Run
```bash
git clone https://github.com/Takun123/fraud-detection-project
cd fraud-detection-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Download creditcard.csv from Kaggle and place in data/raw/
jupyter notebook
```

## Tools
Python, pandas, scikit-learn, XGBoost, SHAP, joblib, matplotlib, seaborn
