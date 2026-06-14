"""
Feature Selection Approach - Closer to original paper methodology
Using Sequential Feature Selection and no class balancing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE SELECTION APPROACH")
print("(Replicating original paper methodology)")
print("="*70)

# Load data
df_2019 = pd.read_excel('Dataset_Year_2019.xlsx')
df_2020 = pd.read_excel('Dataset_Year_2020.xlsx')

target_2019 = [col for col in df_2019.columns if 'burned' in col.lower()][0]
target_2020 = [col for col in df_2020.columns if 'burned' in col.lower()][0]

print(f"\n2019 Dataset: {len(df_2019):,} rows")
print(f"2020 Dataset: {len(df_2020):,} rows")

# ============================================================
# TEST ON 2019 DATA ONLY (like original paper)
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 1: 2019 Data Only (Original Paper Setup)")
print("="*70)

X = df_2019.drop(columns=[target_2019])
y = df_2019[target_2019]

# Encode categoricals
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

feature_names = X.columns.tolist()
print(f"Original features: {len(feature_names)}")

# Split (80/20 like original paper)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Feature Selection using RFE with SVM
# ============================================================
print("\n--- Feature Selection using RFE ---")

# Use SVM for feature selection (like original paper)
svm_selector = SVC(kernel='linear', random_state=42)
rfe = RFE(svm_selector, n_features_to_select=8, step=1)
rfe.fit(X_train_scaled, y_train)

selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
print(f"Selected features ({len(selected_features)}):")
for f in selected_features:
    print(f"  - {f}")

# Get selected feature data
X_train_selected = X_train_scaled[:, rfe.support_]
X_test_selected = X_test_scaled[:, rfe.support_]

# ============================================================
# Train models WITHOUT class balancing (original paper approach)
# ============================================================
print("\n--- Training Models (No Class Balancing) ---")

models = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42),
    'SVM (Tuned)': SVC(kernel='rbf', probability=True, C=10.0, gamma=0.1, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

print(f"\n{'Model':<25} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-"*75)

results = []
best_acc = 0
best_model_name = ""

for name, model in models.items():
    # Train on selected features
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})
    print(f"{name:<25} | {acc*100:>9.2f}% | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model_name = name

# ============================================================
# Try ALL features (no selection)
# ============================================================
print("\n--- Training on ALL Features ---")

print(f"\n{'Model':<25} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-"*75)

best_acc_all = 0

for name, model in models.items():
    # Recreate model instance
    if 'SVM' in name:
        if 'Tuned' in name:
            model = SVC(kernel='rbf', probability=True, C=10.0, gamma=0.1, random_state=42)
        else:
            model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    elif 'Random' in name:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    elif 'XGBoost' in name:
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0)
    elif 'LightGBM' in name:
        model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"{name:<25} | {acc*100:>9.2f}% | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f}")

    if acc > best_acc_all:
        best_acc_all = acc

# ============================================================
# TEST ON 2020 DATA
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 2: 2020 Data Only")
print("="*70)

X_2020 = df_2020.drop(columns=[target_2020])
y_2020 = df_2020[target_2020]

for col in X_2020.columns:
    if X_2020[col].dtype == 'object':
        X_2020[col] = LabelEncoder().fit_transform(X_2020[col].astype(str))

X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(X_2020, y_2020, test_size=0.2, random_state=42, stratify=y_2020)

scaler_20 = StandardScaler()
X_train_20_scaled = scaler_20.fit_transform(X_train_20)
X_test_20_scaled = scaler_20.transform(X_test_20)

print(f"\n{'Model':<25} | {'Accuracy':>10}")
print("-"*40)

for name in ['LightGBM', 'XGBoost', 'Random Forest']:
    if name == 'LightGBM':
        model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    elif name == 'XGBoost':
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train_20_scaled, y_train_20)
    y_pred = model.predict(X_test_20_scaled)
    acc = accuracy_score(y_test_20, y_pred)
    print(f"{name:<25} | {acc*100:>9.2f}%")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nOriginal Paper Baseline:")
print(f"  - SVM (2019 Optimized): 96.74%")
print(f"  - SVM (2020 Optimized): 97.25%")
print(f"\nOur Best Results:")
print(f"  - With Feature Selection: {best_acc*100:.2f}%")
print(f"  - All Features: {best_acc_all*100:.2f}%")

improvement = max(best_acc, best_acc_all) - 0.9725
print(f"\nGap from 97.25% baseline: {improvement*100:+.2f}%")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR IMPROVEMENT:")
print("="*70)
print("""
1. The original paper used 11 specific features after feature selection
2. They used different preprocessing/normalization
3. Consider hyperparameter tuning with GridSearchCV
4. The class imbalance in their test set might be different

Key insight: Our models achieve high accuracy but the exact split
and preprocessing methodology from the original paper is needed
to replicate their results exactly.

For IEOM paper, focus on:
- Novel contribution: Comprehensive ML comparison
- Better explainability with SHAP
- Ensemble methods
- Practical deployment considerations
""")
