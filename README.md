# Heart Disease Prediction Web App

Streamlit app for predicting heart disease using classical ML models (Logistic Regression, Random Forest, XGBoost).

## Dataset
- File used: `heart_cleaned.csv`
- Source: Kaggle — "Heart Disease UCI" (dataset: `ronitf/heart-disease-uci`). This dataset contains the UCI Cleveland Heart Disease data. Kaggle URL: https://www.kaggle.com/ronitf/heart-disease-uci

## Setup
1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run the app:
```powershell
streamlit run app.py
```

## Data cleaning — common Python commands used
Below are concise, commonly used Pandas / scikit-learn commands that were used during cleaning and preprocessing.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load
df = pd.read_csv('heart_cleaned.csv')

# quick checks
df.shape
df.info()
df.describe()
df.isnull().sum()

# remove exact duplicate rows
df.drop_duplicates(inplace=True)

# handle missing values (example strategies)
# option A: drop rows with missing values
df = df.dropna()
# option B: fill numeric NaNs with median
df.fillna(df.median(), inplace=True)

# convert dtypes when needed
df['some_int_col'] = df['some_int_col'].astype(int)

# categorical encoding
df = pd.get_dummies(df, columns=['cp','thal'], drop_first=True)

# feature scaling
scaler = StandardScaler()
num_cols = ['age','trestbps','chol','thalach','oldpeak']
df[num_cols] = scaler.fit_transform(df[num_cols])

# train-test split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save cleaned dataset
df.to_csv('heart_cleaned_processed.csv', index=False)
```

Adjust column names and strategies to match your dataset and goals.

## Project lifecycle (9 stages)
1. Problem definition — clarify prediction objective and success metrics.
2. Data collection — gather raw data and record provenance.
3. Data understanding / EDA — inspect distributions, correlations, and class balance.
4. Data cleaning / preprocessing — handle missing values, duplicates, outliers, and types.
5. Feature engineering — create, select, and transform informative features.
6. Model selection & training — compare models and tune hyperparameters.
7. Evaluation — compute metrics, plot ROC/PR curves, and validate with cross-validation.
8. Deployment — package model and serve via an app or API (e.g., Streamlit, Flask, FastAPI).
9. Monitoring & maintenance — track performance drift, retrain when necessary.

---
If you want I can insert the exact dataset URL, list the exact columns used in preprocessing, or restore a previous README version from source control. Tell me which details to fill in.
