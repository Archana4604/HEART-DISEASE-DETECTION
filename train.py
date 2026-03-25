"""
Heart Disease Detection - Training Script
Run this in Google Colab or locally. Place heart.csv in the training folder.
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# --- Load Dataset ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')

# Column mapping for alternative dataset formats
COL_MAP = {
    'chest_pain_type': 'cp', 'resting_blood_pressure': 'trestbps', 'cholesterol': 'chol',
    'fasting_blood_sugar': 'fbs', 'resting_ecg': 'restecg', 'max_heart_rate': 'thalach',
    'exercise_induced_angina': 'exang', 'st_depression': 'oldpeak', 'st_slope': 'slope',
    'num_major_vessels': 'ca', 'thalassemia': 'thal', 'heart_disease': 'target'
}

for path in ['heart.csv', 'heart_disease_dataset.csv']:
    p = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(p):
        df = pd.read_csv(p)
        df = df.rename(columns=COL_MAP)
        break
else:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(url, header=None, names=cols, na_values='?')
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])

# Ensure target column
if 'target' not in df.columns and 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
elif 'target' not in df.columns:
    df['target'] = df.iloc[:, -1]

# --- Preprocessing ---
df = df.dropna()
target_col = 'target'
y = df[target_col]
X = df.drop(columns=[target_col])

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scale_cols = [c for c in ['chol', 'trestbps', 'thalach'] if c in X.columns]
scaler = StandardScaler()
X_scaled = X.copy()
if scale_cols:
    X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Models with GridSearchCV ---
models = {
    'Decision Tree': (DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'criterion': ['gini', 'entropy']}),
    'Random Forest': (RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}),
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42),
        {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}),
    'SVM': (SVC(probability=True, random_state=42),
        {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']})
}

best_models = {}
for name, (model, params) in models.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

# --- Evaluate & Select Best ---
results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
results_df['Composite'] = results_df['ROC-AUC'] + results_df['F1-Score']
best_model_name = results_df.loc[results_df['Composite'].idxmax(), 'Model']
best_model = best_models[best_model_name]

print("\nPerformance Summary:")
print(results_df.to_string(index=False))
print(f"\nBest Model: {best_model_name}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_model.predict(X_test))
print(cm)

# --- Export ---
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
metadata = {
    'feature_columns': list(X.columns),
    'scale_columns': scale_cols,
    'best_model_name': best_model_name
}
with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

print("\nExported: heart_disease_model.pkl, scaler.pkl, metadata.pkl to model/")
