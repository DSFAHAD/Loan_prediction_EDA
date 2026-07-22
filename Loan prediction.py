import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
df = pd.read_csv(r"C:\Python For DS\codes\Superstore-EDA\intership\loan_train.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include='all').T)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct')
print(missing_report)

plt.figure(figsize=(8, 4))
sns.barplot(x=missing_report.index, y=missing_report['missing_pct'], color='#6FA0D8')
plt.axhline(5, color='green', linestyle='--', label='5% threshold')
plt.axhline(20, color='red', linestyle='--', label='20% threshold')
plt.xticks(rotation=45)
plt.ylabel('% Missing')
plt.title('Missingness Proportion per Feature')
plt.legend()
plt.tight_layout()
plt.show()

print("LoanAmount skew:", df['LoanAmount'].skew().round(2))
plt.figure(figsize=(6, 3))
sns.histplot(df['LoanAmount'].dropna(), kde=True, color='#E8637A')
plt.title('LoanAmount Distribution (before cleaning)')
plt.show()

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df = df.dropna(subset=['Gender', 'Married', 'Dependents', 'Loan_Amount_Term'])

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['Credit_History'] = df.groupby('Property_Area')['Credit_History'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median())
)

print("Remaining nulls:", df.isnull().sum().sum())
print(df.shape)

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color='#6FA0D8')
    ax.set_title(f'{col} (before capping)')
plt.tight_layout()
plt.show()

def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

outlier_summary = {}
for col in numeric_cols:
    lower, upper = iqr_bounds(df[col])
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = {'lower_bound': round(lower, 2), 'upper_bound': round(upper, 2), 'n_outliers': n_outliers}
    df[col] = np.clip(df[col], lower, upper)

print(pd.DataFrame(outlier_summary).T)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color='#E8637A')
    ax.set_title(f'{col} (after IQR capping)')
plt.tight_layout()
plt.show()

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_to_Income_Ratio'] = df['LoanAmount'] / df['TotalIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']

df['Dependents_numeric'] = df['Dependents'].replace('3+', '3').astype(int)
df['Income_per_Dependent'] = df['TotalIncome'] / (df['Dependents_numeric'] + 1)

print(df[['TotalIncome', 'LoanAmount_to_Income_Ratio', 'EMI', 'Income_per_Dependent']].describe().T)

binary_map_cols = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Loan_Status': {'Y': 1, 'N': 0}
}
for col, mapping in binary_map_cols.items():
    df[col] = df[col].map(mapping)

df = pd.get_dummies(df, columns=['Property_Area', 'Dependents'], drop_first=False)

corr_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix — All Numeric Features')
plt.show()

threshold = 0.80
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index
                    if pd.notnull(upper.loc[row, col]) and abs(upper.loc[row, col]) > threshold]

print("High-correlation pairs (>0.80):")
for a, b, val in high_corr_pairs:
    print(f"  {a} <-> {b} : {val:.2f}")

target_corr = corr_matrix['Loan_Status'].abs()
to_drop = set()
for a, b, val in high_corr_pairs:
    weaker = a if target_corr[a] < target_corr[b] else b
    to_drop.add(weaker)

print("Dropping (weaker link with target):", to_drop)
df = df.drop(columns=list(to_drop))

print("Final shape:", df.shape)
print("Remaining nulls:", df.isnull().sum().sum())
df.to_csv('loan_cleaned.csv', index=False)
print("Saved: loan_cleaned.csv")