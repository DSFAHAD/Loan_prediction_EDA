# Loan Prediction: Advanced EDA & Feature Engineering

Project 1 of the DecodeLabs Data Science Industrial Training Kit (Batch 2026).

## Goal
Transform a raw, messy loan applications dataset into a mathematically clean dataset ready for machine learning, instead of just running algorithms on dirty data.

## Dataset
Loan Prediction dataset: 614 loan applications with a mix of numeric and categorical features, real missing values, and a binary target (Loan_Status).

## What Was Done

### 1. Missing Value Handling
Applied a missingness based decision rule:
- Columns under 5% missing were dropped
- Columns between 5% and 20% missing were imputed (median for skewed numeric columns, mode or group wise imputation for categorical columns)

### 2. Outlier Handling
Detected outliers using the IQR (Interquartile Range) method and capped them using numpy.clip() instead of deleting rows, to preserve data volume.

### 3. Feature Engineering
Created 4 new features:
- TotalIncome
- LoanAmount_to_Income_Ratio
- EMI
- Income_per_Dependent

### 4. Encoding
Binary mapping used for 2 category fields (Gender, Married, Education, Self_Employed). One hot encoding used for multi category nominal fields (Property_Area, Dependents), to avoid introducing false ordinal relationships.

### 5. Multicollinearity Check
Built a correlation matrix, flagged feature pairs with correlation above 0.80, and dropped whichever feature in the pair correlated weaker with the target variable.

## Result
573 clean rows, 0 missing values, no problematic multicollinearity left. Dataset is ready for the next stage: model training.

## Files
- Loan prediction.py: full Python pipeline script
- loan_train.csv: original raw dataset
- loan_cleaned.csv: final cleaned output

## Tech Used
Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn

## Author
Fahad Hameed
GitHub: DSFAHAD
LinkedIn: linkedin.com/in/fahad-hameed-7501b5333
