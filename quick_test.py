"""Quick test to verify models work and get initial results"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df_2019 = pd.read_excel('Dataset_Year_2019.xlsx')
df_2020 = pd.read_excel('Dataset_Year_2020.xlsx')

# Get target columns
target_2019 = [col for col in df_2019.columns if 'burned' in col.lower()][0]
target_2020 = [col for col in df_2020.columns if 'burned' in col.lower()][0]

print(f"2019: {len(df_2019)} rows, {df_2019[target_2019].sum()} failures ({df_2019[target_2019].mean()*100:.2f}%)")
print(f"2020: {len(df_2020)} rows, {df_2020[target_2020].sum()} failures ({df_2020[target_2020].mean()*100:.2f}%)")

# Combine and preprocess
df_2019_copy = df_2019.rename(columns={target_2019: 'Burned'})
df_2020_copy = df_2020.rename(columns={target_2020: 'Burned'})
df = pd.concat([df_2019_copy, df_2020_copy], ignore_index=True)

X = df.drop(columns=['Burned'])
y = df['Burned']

# Encode categoricals
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAfter SMOTE: {np.bincount(y_train_balanced)}")

# Models
models = {
    'SVM (Baseline)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

print("\n" + "="*70)
print(f"{'Model':<20} | {'Accuracy':>12} | {'F1-Score':>12} | {'ROC-AUC':>12}")
print("="*70)

for name, model in models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\r{name:<20} | {acc*100:>11.2f}% | {f1:>12.4f} | {auc:>12.4f}")

print("="*70)
print("\nBaseline to beat: 97.25%")
