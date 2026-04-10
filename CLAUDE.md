# Fraud Detection Project — ML Engineer Context

## Project Goal
Build an end-to-end fraud detection pipeline on a messy, real-world-style dataset.
Binary classification: fraudulent transaction (1) vs legitimate (0).

## Hard Rules (Never Break These)
- Split data BEFORE any transformation (no leakage)
- Use StratifiedKFold (not plain train_test_split) due to class imbalance
- Baseline model first (LogisticRegression), then XGBoost, then LightGBM
- Evaluation metrics: Precision, Recall, F1, AUC-ROC — NEVER just accuracy
- Save models with joblib, not pickle
- Every function needs a docstring

## Stack
Python 3.11, pandas, numpy, scikit-learn, xgboost, lightgbm, shap, imbalanced-learn

## Current Phase
EDA and data cleaning (Phase 1)

## What I Am NOT Asking You To Do
- Do not write the full pipeline at once
- Do not explain basic Python syntax
- Do not add features I didn't ask for