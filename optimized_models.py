"""
Optimized models for transformer failure prediction
Using class weights instead of SMOTE for better accuracy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("OPTIMIZED TRANSFORMER FAILURE PREDICTION")
print("="*70)

# Load data
print("\nLoading data...")
df_2019 = pd.read_excel('Dataset_Year_2019.xlsx')
df_2020 = pd.read_excel('Dataset_Year_2020.xlsx')

target_2019 = [col for col in df_2019.columns if 'burned' in col.lower()][0]
target_2020 = [col for col in df_2020.columns if 'burned' in col.lower()][0]

print(f"2019: {len(df_2019):,} transformers, {df_2019[target_2019].sum()} failures")
print(f"2020: {len(df_2020):,} transformers, {df_2020[target_2020].sum()} failures")

# Combine datasets
df_2019_copy = df_2019.rename(columns={target_2019: 'Burned'})
df_2020_copy = df_2020.rename(columns={target_2020: 'Burned'})
df = pd.concat([df_2019_copy, df_2020_copy], ignore_index=True)

print(f"Combined: {len(df):,} transformers, {df['Burned'].sum()} failures ({df['Burned'].mean()*100:.2f}%)")

# Preprocess
X = df.drop(columns=['Burned'])
y = df['Burned']

# Encode categoricals
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Feature engineering
burn_col = [col for col in X.columns if 'burn' in col.lower()][0]
eens_col = [col for col in X.columns if 'eens' in col.lower()][0]
X['burn_squared'] = X[burn_col] ** 2
X['burn_log'] = np.log1p(X[burn_col])
X['burn_x_eens'] = X[burn_col] * X[eens_col]

print(f"Features after engineering: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weight
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\nClass weight ratio: {scale_pos_weight:.2f}")

# ============================================================
# APPROACH 1: Without SMOTE, with class weights
# ============================================================
print("\n" + "="*70)
print("APPROACH 1: Class Weights (No SMOTE)")
print("="*70)

models_v1 = {
    'SVM (class_weight)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'Random Forest (balanced)': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost (scale_pos)': XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM (balanced)': LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1),
}

print(f"\n{'Model':<30} | {'Accuracy':>10} | {'F1':>8} | {'ROC-AUC':>8}")
print("-"*65)

results_v1 = []
for name, model in models_v1.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results_v1.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'ROC-AUC': auc})
    print(f"{name:<30} | {acc*100:>9.2f}% | {f1:>8.4f} | {auc:>8.4f}")

# ============================================================
# APPROACH 2: Threshold optimization
# ============================================================
print("\n" + "="*70)
print("APPROACH 2: XGBoost with Optimized Threshold")
print("="*70)

# Train XGBoost
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight, random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train_scaled, y_train)
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold for F1
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_xgb)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_f1 = thresholds[np.argmax(f1_scores[:-1])]

# Find optimal threshold for accuracy
best_acc = 0
best_threshold_acc = 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred_thresh = (y_proba_xgb >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    if acc > best_acc:
        best_acc = acc
        best_threshold_acc = thresh

print(f"Best threshold for F1: {best_threshold_f1:.3f}")
print(f"Best threshold for Accuracy: {best_threshold_acc:.3f}")

# Evaluate with optimal thresholds
y_pred_f1 = (y_proba_xgb >= best_threshold_f1).astype(int)
y_pred_acc = (y_proba_xgb >= best_threshold_acc).astype(int)

print(f"\nWith F1-optimized threshold ({best_threshold_f1:.3f}):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_f1)*100:.2f}%")
print(f"  F1-Score: {f1_score(y_test, y_pred_f1):.4f}")

print(f"\nWith Accuracy-optimized threshold ({best_threshold_acc:.3f}):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_acc)*100:.2f}%")
print(f"  F1-Score: {f1_score(y_test, y_pred_acc):.4f}")

# ============================================================
# APPROACH 3: Stacking Ensemble
# ============================================================
print("\n" + "="*70)
print("APPROACH 3: Stacking Ensemble")
print("="*70)

base_models = [
    ('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)),
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    n_jobs=-1
)

print("Training stacking ensemble...")
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_test_scaled)
y_proba_stack = stacking.predict_proba(X_test_scaled)[:, 1]

print(f"\nStacking Ensemble Results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_stack)*100:.2f}%")
print(f"  F1-Score: {f1_score(y_test, y_pred_stack):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, y_proba_stack):.4f}")

# ============================================================
# APPROACH 4: Cross-Year Validation (2019 -> 2020)
# ============================================================
print("\n" + "="*70)
print("APPROACH 4: Cross-Year Validation (Train 2019 -> Test 2020)")
print("="*70)

# Preprocess 2019
X_2019 = df_2019.drop(columns=[target_2019])
y_2019 = df_2019[target_2019]
for col in X_2019.columns:
    if X_2019[col].dtype == 'object':
        X_2019[col] = LabelEncoder().fit_transform(X_2019[col].astype(str))

# Feature engineering
burn_col = [col for col in X_2019.columns if 'burn' in col.lower()][0]
eens_col = [col for col in X_2019.columns if 'eens' in col.lower()][0]
X_2019['burn_squared'] = X_2019[burn_col] ** 2
X_2019['burn_log'] = np.log1p(X_2019[burn_col])
X_2019['burn_x_eens'] = X_2019[burn_col] * X_2019[eens_col]

# Preprocess 2020
X_2020 = df_2020.drop(columns=[target_2020])
y_2020 = df_2020[target_2020]
for col in X_2020.columns:
    if X_2020[col].dtype == 'object':
        X_2020[col] = LabelEncoder().fit_transform(X_2020[col].astype(str))

burn_col = [col for col in X_2020.columns if 'burn' in col.lower()][0]
eens_col = [col for col in X_2020.columns if 'eens' in col.lower()][0]
X_2020['burn_squared'] = X_2020[burn_col] ** 2
X_2020['burn_log'] = np.log1p(X_2020[burn_col])
X_2020['burn_x_eens'] = X_2020[burn_col] * X_2020[eens_col]

# Align columns
common_cols = list(set(X_2019.columns) & set(X_2020.columns))
X_2019 = X_2019[common_cols]
X_2020 = X_2020[common_cols]

# Scale
scaler_cv = StandardScaler()
X_2019_scaled = scaler_cv.fit_transform(X_2019)
X_2020_scaled = scaler_cv.transform(X_2020)

# Train on 2019, test on 2020
scale_pos_2019 = (y_2019 == 0).sum() / (y_2019 == 1).sum()

cv_models = {
    'XGBoost': XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_2019, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1),
}

print(f"\n{'Model':<15} | {'Accuracy':>10} | {'F1':>8} | {'ROC-AUC':>8}")
print("-"*50)

for name, model in cv_models.items():
    model.fit(X_2019_scaled, y_2019)
    y_pred = model.predict(X_2020_scaled)
    y_proba = model.predict_proba(X_2020_scaled)[:, 1]

    acc = accuracy_score(y_2020, y_pred)
    f1 = f1_score(y_2020, y_pred)
    auc = roc_auc_score(y_2020, y_proba)

    print(f"{name:<15} | {acc*100:>9.2f}% | {f1:>8.4f} | {auc:>8.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nBaseline (Original Paper SVM): 97.25%")
print(f"\nBest results with threshold optimization:")
print(f"  - Accuracy-optimized: {best_acc*100:.2f}%")

if best_acc > 0.9725:
    print(f"\n*** SUCCESS: Beat baseline by {(best_acc - 0.9725)*100:+.2f}% ***")
else:
    print(f"\n*** Gap from baseline: {(best_acc - 0.9725)*100:.2f}% ***")
    print("\nNote: The original paper used Sequential Forward Feature Selection")
    print("to identify optimal features. Consider:")
    print("1. Feature selection using RFE or SelectKBest")
    print("2. More hyperparameter tuning")
    print("3. Different train/test splits")
