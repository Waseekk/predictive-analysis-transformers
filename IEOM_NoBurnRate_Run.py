"""
IEOM Predictive Maintenance Pipeline — WITHOUT Burn Rate & Criticality
======================================================================
This is the IEOM notebook converted to a runnable script, with Burn Rate
and Criticality DROPPED to match the Transformer Copy approach.
Outputs go to: outputs_no_burn/
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

# ── Global configuration ─────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(r"E:\Projects_\predictive_analysis\ifti_vai\Transformer")
OUTPUT_DIR = Path(r"E:\Projects_\predictive_analysis\outputs_no_burn")
OUTPUT_DIR.mkdir(exist_ok=True)

PATH_2019 = DATA_DIR / "Dataset_Year_2019.xlsx"
PATH_2020 = DATA_DIR / "Dataset_Year_2020.xlsx"

TARGET_2019 = "Burned transformers 2019"
TARGET_2020 = "Burned transformers 2020"
EENS_COL    = "Electric power not supplied EENS [kWh] "
USERS_COL   = "Number of users"
KM_COL      = "km of network LT:"
AVG_DDT     = "Average earth discharge density DDT [Rays/km^2-año]"
MAX_DDT     = "Maximum ground discharge density DDT [Rays/km^2-año]"
POWER_COL   = "POWER"
BURN_RATE   = "Burning rate  [Failures/year]"
CRITICALITY = "Criticality according to previous study for ceramics level"

KNOWN_CATS = {
    "LOCATION", "SELF-PROTECTION", "Removable connectors",
    "Type of clients", "Type of installation", "Air network", "Circuit Queue",
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.figsize': (8, 5),
    'axes.grid': True, 'grid.alpha': 0.3,
})
sns.set_style("whitegrid")

print("=" * 70)
print("IEOM Pipeline — WITHOUT Burn Rate & Criticality")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

# ============================================================================
# Section 1: Data Loading
# ============================================================================
print("\n[1/12] Loading data...")
df_2019 = pd.read_excel(PATH_2019)
df_2020 = pd.read_excel(PATH_2020)

print(f"  2019: {df_2019.shape[0]:,} rows x {df_2019.shape[1]} cols")
print(f"  2020: {df_2020.shape[0]:,} rows x {df_2020.shape[1]} cols")

n_pos_19 = int(df_2019[TARGET_2019].sum())
n_pos_20 = int(df_2020[TARGET_2020].sum())
print(f"  2019 failures: {n_pos_19:,} ({n_pos_19/len(df_2019)*100:.2f}%)")
print(f"  2020 failures: {n_pos_20:,} ({n_pos_20/len(df_2020)*100:.2f}%)")

table1_data = {
    'Attribute': ['Total transformers', 'Burned (positive)', 'Not burned (negative)',
                  'Failure rate (%)', 'Number of features'],
    '2019 (Train)': [f"{len(df_2019):,}", f"{n_pos_19:,}",
                     f"{len(df_2019)-n_pos_19:,}", f"{n_pos_19/len(df_2019)*100:.2f}",
                     str(df_2019.shape[1]-1)],
    '2020 (Test)':  [f"{len(df_2020):,}", f"{n_pos_20:,}",
                     f"{len(df_2020)-n_pos_20:,}", f"{n_pos_20/len(df_2020)*100:.2f}",
                     str(df_2020.shape[1]-1)],
}
table1 = pd.DataFrame(table1_data)

# ============================================================================
# Section 2: Feature Engineering — NO BURN RATE, NO CRITICALITY
# ============================================================================
print("\n[2/12] Feature engineering (WITHOUT Burn Rate & Criticality)...")

def safe_div(a, b, eps=1e-6):
    return a / (b.replace(0, np.nan) + eps)

def add_engineered_features(df):
    df = df.copy()
    df["users_per_km"]   = safe_div(df[USERS_COL], df[KM_COL])
    df["eens_per_user"]  = safe_div(df[EENS_COL],  df[USERS_COL])
    df["eens_per_km"]    = safe_div(df[EENS_COL],  df[KM_COL])
    df["ddt_ratio"]      = safe_div(df[MAX_DDT],   df[AVG_DDT])
    df["log_eens"]       = np.log1p(df[EENS_COL].clip(lower=0))
    df["power_per_user"] = safe_div(df[POWER_COL], df[USERS_COL])
    # NOTE: burn_x_power and burn_x_eens are EXCLUDED because they depend on Burn Rate
    return df

def build_Xy(df, target_col):
    """Build X, y — DROP Burn Rate and Criticality."""
    df = add_engineered_features(df)
    y = df[target_col].astype(int).copy()
    X = df.drop(columns=[target_col]).copy()
    # >>> KEY DIFFERENCE: Drop expert features <<<
    for col in [BURN_RATE, CRITICALITY]:
        if col in X.columns:
            X = X.drop(columns=[col])
    return X, y

X_2019, y_2019 = build_Xy(df_2019, TARGET_2019)
X_2020, y_2020 = build_Xy(df_2020, TARGET_2020)

print(f"  X_2019: {X_2019.shape}  |  positive rate: {y_2019.mean():.4f}")
print(f"  X_2020: {X_2020.shape}  |  positive rate: {y_2020.mean():.4f}")
print(f"  Features ({X_2019.shape[1]}):")
for i, c in enumerate(X_2019.columns, 1):
    print(f"    {i:2d}. {c}")
print(f"\n  >>> DROPPED: {BURN_RATE}")
print(f"  >>> DROPPED: {CRITICALITY}")
print(f"  >>> DROPPED: burn_x_power, burn_x_eens (depend on Burn Rate)")

# ============================================================================
# Section 2b: Identify column types
# ============================================================================
cat_cols = [c for c in X_2019.columns if (X_2019[c].dtype == "object") or (c in KNOWN_CATS)]
num_cols = [c for c in X_2019.columns if c not in cat_cols]
print(f"\n  Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

# ============================================================================
# Section 3: EDA Figures
# ============================================================================
print("\n[3/12] Generating EDA figures...")

# Fig 1: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (year, y, n_pos) in zip(axes, [
    ("2019 (Train)", y_2019, n_pos_19), ("2020 (Test)", y_2020, n_pos_20)
]):
    counts = [len(y) - n_pos, n_pos]
    labels = [f"Not Burned\n({counts[0]:,})", f"Burned\n({counts[1]:,})"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar([0, 1], counts, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels); ax.set_ylabel("Count"); ax.set_title(year)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                f"{count/len(y)*100:.1f}%", ha='center', va='bottom', fontweight='bold')
fig.suptitle("Figure 1: Class Distribution (NO Burn Rate version)", fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig01_class_distribution.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 2: Correlation heatmap
numeric_df = X_2019[num_cols].copy()
short_names = {POWER_COL: "POWER", AVG_DDT: "Avg DDT", MAX_DDT: "Max DDT",
               USERS_COL: "Users", EENS_COL: "EENS", KM_COL: "km LT"}
disp_names = [short_names.get(c, c) for c in numeric_df.columns]
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=disp_names, yticklabels=disp_names, square=True, linewidths=0.5, ax=ax)
ax.set_title("Figure 2: Feature Correlation (NO Burn Rate)", fontweight='bold')
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig02_correlation_heatmap.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 3: EENS distribution by failure status (replacing Burn Rate fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (label, X, y) in zip(axes, [("2019 (Train)", X_2019, y_2019), ("2020 (Test)", X_2020, y_2020)]):
    for cls, color, lbl in [(0, '#2ecc71', 'Not Burned'), (1, '#e74c3c', 'Burned')]:
        subset = X.loc[y == cls, EENS_COL]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=lbl, density=True, edgecolor='black', linewidth=0.3)
    ax.set_xlabel("EENS [kWh]"); ax.set_ylabel("Density"); ax.set_title(label); ax.legend()
fig.suptitle("Figure 3: EENS Distribution by Failure Status", fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig03_eens_distribution.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 4: Top-5 feature boxplots (without burn rate)
top5 = [EENS_COL, POWER_COL, "eens_per_km", "users_per_km", "log_eens"]
top5_short = ["EENS", "POWER", "EENS/km", "Users/km", "log(EENS)"]
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
plot_df = X_2019.copy(); plot_df["Burned"] = y_2019.values
for ax, feat, short in zip(axes, top5, top5_short):
    if feat not in plot_df.columns:
        ax.set_visible(False); continue
    d0 = plot_df.loc[plot_df["Burned"]==0, feat]; d1 = plot_df.loc[plot_df["Burned"]==1, feat]
    bp = ax.boxplot([d0, d1], labels=["No","Yes"], patch_artist=True, widths=0.5, showfliers=False)
    bp['boxes'][0].set_facecolor('#2ecc71'); bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_title(short, fontsize=10); ax.set_xlabel("Burned")
fig.suptitle("Figure 4: Top-5 Feature Distributions (NO Burn Rate)", fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig04_boxplots_top5.png", dpi=300, bbox_inches='tight'); plt.close()
print("  Figures 1-4 saved.")

# ============================================================================
# Section 4: Preprocessing
# ============================================================================
print("\n[4/12] Preprocessing pipeline...")

def make_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

X_train_full, X_val_2019, y_train_full, y_val_2019 = train_test_split(
    X_2019, y_2019, test_size=0.20, random_state=SEED, stratify=y_2019)
X_train_2019, X_calib_2019, y_train_2019, y_calib_2019 = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=SEED, stratify=y_train_full)

print(f"  Train: {len(X_train_2019):,} | Calib: {len(X_calib_2019):,} | Val: {len(X_val_2019):,} | Test: {len(X_2020):,}")

preprocessor = make_preprocessor(num_cols, cat_cols)
preprocessor.fit(X_train_2019)
X_train_pp = preprocessor.transform(X_train_2019)
X_calib_pp = preprocessor.transform(X_calib_2019)
X_val_pp   = preprocessor.transform(X_val_2019)
X_test_pp  = preprocessor.transform(X_2020)
ohe_feature_names = preprocessor.get_feature_names_out()
print(f"  Transformed dimensions: {X_train_pp.shape[1]} features")

# ============================================================================
# Section 5: SMOTE
# ============================================================================
print("\n[5/12] SMOTE comparison...")
smote = SMOTE(random_state=SEED)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pp, y_train_2019)
print(f"  Before SMOTE: {len(X_train_pp):,} | After: {len(X_train_smote):,}")

strategies = {
    'No resampling': (X_train_pp, y_train_2019, {}),
    'SMOTE':         (X_train_smote, y_train_smote, {}),
    'Class weights': (X_train_pp, y_train_2019, {'class_weight': 'balanced'}),
}
print(f"\n  {'Strategy':<20s} {'ROC-AUC':>10s} {'PR-AUC':>10s} {'F2':>10s}")
print("  " + "-" * 55)
smote_results = []
for name, (Xtr, ytr, kwargs) in strategies.items():
    lr = LogisticRegression(max_iter=5000, solver='liblinear', random_state=SEED, **kwargs)
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(X_val_pp)[:, 1]
    roc = roc_auc_score(y_val_2019, proba)
    pr  = average_precision_score(y_val_2019, proba)
    best_f2 = max(fbeta_score(y_val_2019, (proba >= t).astype(int), beta=2, zero_division=0)
                  for t in np.linspace(0.01, 0.99, 99))
    smote_results.append({'Strategy': name, 'ROC-AUC': roc, 'PR-AUC': pr, 'F2': best_f2})
    print(f"  {name:<20s} {roc:>10.4f} {pr:>10.4f} {best_f2:>10.4f}")
table2 = pd.DataFrame(smote_results)

# ============================================================================
# Section 6: Model Training
# ============================================================================
print("\n[6/12] Training 6 models with RandomizedSearchCV...")
pos_weight = (y_train_2019 == 0).sum() / max((y_train_2019 == 1).sum(), 1)
print(f"  Class imbalance ratio: {pos_weight:.1f}:1")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

model_configs = {
    'SVM (RBF)': {
        'estimator': SVC(kernel='rbf', probability=True, class_weight='balanced',
                         random_state=SEED, cache_size=1000),
        'params': {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.01, 0.001]},
        'use_smote': False,
    },
    'Logistic Regression': {
        'estimator': LogisticRegression(max_iter=5000, solver='liblinear',
                                        class_weight='balanced', random_state=SEED),
        'params': {'C': np.logspace(-3, 2, 15), 'penalty': ['l1', 'l2']},
        'use_smote': False,
    },
    'Random Forest': {
        'estimator': RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                            random_state=SEED, n_jobs=-1),
        'params': {'max_depth': [None, 8, 16, 24], 'min_samples_leaf': [1, 3, 5, 10],
                   'min_samples_split': [2, 5, 10]},
        'use_smote': False,
    },
    'Gradient Boosting': {
        'estimator': GradientBoostingClassifier(random_state=SEED),
        'params': {'n_estimators': [200, 400, 600], 'learning_rate': [0.03, 0.05, 0.1],
                   'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0]},
        'use_smote': True,
    },
    'XGBoost': {
        'estimator': XGBClassifier(random_state=SEED, n_estimators=500, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
                                    tree_method='hist', scale_pos_weight=pos_weight),
        'params': {'max_depth': [3, 5, 7], 'min_child_weight': [1, 5, 10],
                   'reg_alpha': [0, 0.1, 1], 'reg_lambda': [1, 5, 10]},
        'use_smote': False,
    },
    'LightGBM': {
        'estimator': LGBMClassifier(random_state=SEED, n_estimators=500, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
                                     min_child_samples=50, num_leaves=31, max_depth=6, verbose=-1),
        'params': {'num_leaves': [15, 31, 63], 'min_child_samples': [20, 50, 100],
                   'max_depth': [4, 6, 8], 'reg_alpha': [0, 0.1, 1]},
        'use_smote': False,
    },
}

trained_models = {}
for name, config in model_configs.items():
    print(f"\n  [{name}]", end=" ")
    if config['use_smote']:
        Xtr, ytr = X_train_smote, y_train_smote
    else:
        Xtr, ytr = X_train_pp, y_train_2019
    rs = RandomizedSearchCV(config['estimator'], config['params'], n_iter=20,
                            scoring='average_precision', cv=cv, n_jobs=-1, random_state=SEED, verbose=0)
    rs.fit(Xtr, ytr)
    trained_models[name] = rs.best_estimator_
    val_proba = rs.best_estimator_.predict_proba(X_val_pp)[:, 1]
    val_pr = average_precision_score(y_val_2019, val_proba)
    val_roc = roc_auc_score(y_val_2019, val_proba)
    print(f"CV={rs.best_score_:.4f} | Val PR-AUC={val_pr:.4f} | Val ROC={val_roc:.4f}")
print("\n  All models trained.")

# ============================================================================
# Section 7: Calibration
# ============================================================================
print("\n[7/12] Probability calibration...")
calibrated_models = {}
for name, model in trained_models.items():
    cal = CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')
    cal.fit(X_calib_pp, y_calib_2019)
    calibrated_models[name] = cal
print(f"  {len(calibrated_models)} models calibrated.")

# Fig 5: Calibration curves
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
colors_cal = plt.cm.Set2(np.linspace(0, 1, len(calibrated_models)))
for (name, cal_model), color in zip(calibrated_models.items(), colors_cal):
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_2020, proba, n_bins=10, strategy='uniform')
    ax.plot(mean_pred, frac_pos, 's-', color=color, label=name, markersize=5)
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title('Figure 5: Calibration Curves (NO Burn Rate)', fontweight='bold')
ax.legend(loc='lower right'); ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig05_calibration_curves.png", dpi=300, bbox_inches='tight'); plt.close()
print("  Figure 5 saved.")

# ============================================================================
# Section 8: Evaluation
# ============================================================================
print("\n[8/12] Evaluating on 2020 test set...")

def optimize_threshold_f2(y_true, y_prob):
    best_t, best_f2 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2: best_f2, best_t = f2, t
    return best_t, best_f2

def precision_recall_at_k(y_true, y_prob, k_pct):
    n = len(y_true); k = max(1, int(n * k_pct / 100))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    top_k_true = np.array(y_true)[top_k_idx]
    prec = top_k_true.sum() / k
    rec = top_k_true.sum() / max(1, y_true.sum())
    lift = prec / max(1e-10, y_true.mean())
    return prec, rec, lift

# Dummy baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_pp, y_train_2019)
dummy_pred = dummy.predict(X_test_pp)
print(f"\n  DUMMY CLASSIFIER: Accuracy={accuracy_score(y_2020, dummy_pred)*100:.2f}%, "
      f"Recall={recall_score(y_2020, dummy_pred, zero_division=0)*100:.2f}%, "
      f"F1={f1_score(y_2020, dummy_pred, zero_division=0):.4f}")
print("  --> Accuracy is MEANINGLESS at ~4% failure rate.\n")

eval_results = []
for name, cal_model in calibrated_models.items():
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    val_proba = cal_model.predict_proba(X_val_pp)[:, 1]
    best_t, _ = optimize_threshold_f2(y_val_2019, val_proba)
    y_pred = (proba >= best_t).astype(int)

    roc = roc_auc_score(y_2020, proba)
    pr  = average_precision_score(y_2020, proba)
    f2  = fbeta_score(y_2020, y_pred, beta=2, zero_division=0)
    f1  = f1_score(y_2020, y_pred, zero_division=0)
    acc = accuracy_score(y_2020, y_pred)
    prec = precision_score(y_2020, y_pred, zero_division=0)
    rec  = recall_score(y_2020, y_pred, zero_division=0)
    p1, r1, l1 = precision_recall_at_k(y_2020, proba, 1)
    p5, r5, l5 = precision_recall_at_k(y_2020, proba, 5)
    p10, r10, l10 = precision_recall_at_k(y_2020, proba, 10)

    eval_results.append({
        'Model': name, 'ROC-AUC': roc, 'PR-AUC': pr, 'F2': f2, 'F1': f1,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'Threshold': best_t,
        'P@1%': p1, 'R@1%': r1, 'Lift@1%': l1,
        'P@5%': p5, 'R@5%': r5, 'Lift@5%': l5,
        'P@10%': p10, 'R@10%': r10, 'Lift@10%': l10,
    })

results_df = pd.DataFrame(eval_results).sort_values('PR-AUC', ascending=False)
print("  TABLE 3: Model Comparison (2020 Test Set, NO Burn Rate)")
print("  " + "=" * 95)
display_cols = ['Model', 'ROC-AUC', 'PR-AUC', 'F2', 'F1', 'Precision', 'Recall', 'Lift@5%', 'Threshold']
print(results_df[display_cols].to_string(index=False, float_format='{:.4f}'.format))
best_model_name = results_df.iloc[0]['Model']
print(f"\n  Best model by PR-AUC: {best_model_name}")

# ============================================================================
# Section 8b: Figures 6-8
# ============================================================================
print("\n[8b/12] Generating evaluation figures...")

colors = plt.cm.Set1(np.linspace(0, 1, len(calibrated_models)))

# Fig 6: PR curves
fig, ax = plt.subplots(figsize=(8, 6))
baseline_pr = y_2020.mean()
ax.axhline(y=baseline_pr, color='gray', linestyle=':', label=f'Baseline (P={baseline_pr:.3f})')
for (name, cal_model), color in zip(calibrated_models.items(), colors):
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    precision, recall, _ = precision_recall_curve(y_2020, proba)
    ap = average_precision_score(y_2020, proba)
    ax.plot(recall, precision, color=color, label=f'{name} (AP={ap:.3f})', linewidth=1.5)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Figure 6: PR Curves (NO Burn Rate)', fontweight='bold')
ax.legend(loc='upper right', fontsize=8); ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig06_pr_curves.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 7: ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
for (name, cal_model), color in zip(calibrated_models.items(), colors):
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    fpr, tpr, _ = roc_curve(y_2020, proba)
    auc = roc_auc_score(y_2020, proba)
    ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})', linewidth=1.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 7: ROC Curves (NO Burn Rate)', fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig07_roc_curves.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 8: Confusion matrices
n_models = len(calibrated_models)
fig, axes = plt.subplots(2, 3, figsize=(14, 9)); axes = axes.flatten()
for idx, (name, cal_model) in enumerate(calibrated_models.items()):
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    val_proba = cal_model.predict_proba(X_val_pp)[:, 1]
    best_t, _ = optimize_threshold_f2(y_val_2019, val_proba)
    y_pred = (proba >= best_t).astype(int)
    cm = confusion_matrix(y_2020, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Not Burned','Burned']).plot(ax=axes[idx], cmap='Blues', values_format='d')
    axes[idx].set_title(f"{name}\n(t={best_t:.2f})", fontsize=10)
for idx in range(n_models, len(axes)): axes[idx].set_visible(False)
fig.suptitle('Figure 8: Confusion Matrices (NO Burn Rate)', fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig08_confusion_matrices.png", dpi=300, bbox_inches='tight'); plt.close()
print("  Figures 5-8 saved.")

# ============================================================================
# Section 9: SHAP
# ============================================================================
print("\n[9/12] SHAP explainability...")
tree_models = ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']
best_tree_name = results_df[results_df['Model'].isin(tree_models)].iloc[0]['Model']
print(f"  Using {best_tree_name} for SHAP")

shap_model = trained_models[best_tree_name]
feature_display_names = []
replacements = {AVG_DDT: 'Avg DDT', MAX_DDT: 'Max DDT', EENS_COL: 'EENS',
                USERS_COL: 'Users', KM_COL: 'km Network', POWER_COL: 'POWER'}
for fname in ohe_feature_names:
    fname_str = str(fname)
    for prefix in ['num__', 'cat__']:
        if fname_str.startswith(prefix): fname_str = fname_str[len(prefix):]
    for long, short in replacements.items():
        fname_str = fname_str.replace(long, short)
    feature_display_names.append(fname_str)

explainer = shap.TreeExplainer(shap_model)
shap_sample_size = min(2000, len(X_test_pp))
rng = np.random.RandomState(SEED)
shap_idx = rng.choice(len(X_test_pp), shap_sample_size, replace=False)
X_shap = X_test_pp[shap_idx]
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list): shap_values = shap_values[1]
print(f"  SHAP values: {shap_values.shape}")

# Fig 9: SHAP beeswarm
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=feature_display_names, max_display=15, show=False)
plt.title(f'Figure 9: SHAP (NO Burn Rate) — {best_tree_name}', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig09_shap_beeswarm.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 10: SHAP dependence for top feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_feat_idx = int(np.argmax(mean_abs_shap))
fig, ax = plt.subplots(figsize=(8, 5))
shap.dependence_plot(top_feat_idx, shap_values, X_shap, feature_names=feature_display_names, show=False)
plt.title(f'Figure 10: SHAP Dependence — {feature_display_names[top_feat_idx]}', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig10_shap_dependence_top.png", dpi=300, bbox_inches='tight'); plt.close()

# Grouped SHAP importance
grouped_importance = {}
for i, fname in enumerate(feature_display_names):
    base_name = fname
    for cat in KNOWN_CATS:
        if fname.startswith(cat + '_') or fname.startswith(cat):
            base_name = cat; break
    grouped_importance[base_name] = grouped_importance.get(base_name, 0) + mean_abs_shap[i]
sorted_importance = sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Grouped SHAP Feature Importance:")
print(f"  {'Feature':<35s} {'Mean |SHAP|':>12s}")
print("  " + "-" * 50)
for feat, imp in sorted_importance[:15]:
    print(f"  {feat:<35s} {imp:>12.4f}")
print("  Figures 9-10 saved.")

# ============================================================================
# Section 10: Economic Risk
# ============================================================================
print("\n[10/12] Economic risk framework...")
COST_BASE_COP = 560_000; COST_INSPECTION_COP = 50_000
power_2020 = df_2020[POWER_COL].values
median_power = np.median(power_2020[power_2020 > 0])
replacement_cost = COST_BASE_COP * (power_2020 / median_power).clip(0.5, 5.0)

best_cal_model = calibrated_models[best_model_name]
p_fail = best_cal_model.predict_proba(X_test_pp)[:, 1]
expected_loss = p_fail * replacement_cost

risk_df = pd.DataFrame({
    'Transformer_ID': range(1, len(df_2020)+1), 'P_failure': p_fail,
    'Power_kVA': power_2020, 'Replacement_Cost_COP': replacement_cost,
    'Expected_Loss_COP': expected_loss, 'Actual_Burned': y_2020.values,
}).sort_values('Expected_Loss_COP', ascending=False).reset_index(drop=True)

print("\n  TOP 10 HIGHEST-RISK TRANSFORMERS:")
top10 = risk_df.head(10).copy(); top10['Rank'] = range(1, 11)
top10['Actual'] = top10['Actual_Burned'].map({0: 'No', 1: 'YES'})
print(top10[['Rank','Transformer_ID','P_failure','Power_kVA','Expected_Loss_COP','Actual']].to_string(index=False, float_format='{:.4f}'.format))

# Fig 11: Cost-benefit
n_total = len(risk_df); sorted_risk = risk_df
n_insp = np.arange(0, n_total+1)
cum_fail = np.zeros(n_total+1); cum_loss = np.zeros(n_total+1); cum_cost = np.zeros(n_total+1)
for i in range(n_total):
    row = sorted_risk.iloc[i]
    cum_fail[i+1] = cum_fail[i] + row['Actual_Burned']
    cum_loss[i+1] = cum_loss[i] + (row['Replacement_Cost_COP'] if row['Actual_Burned']==1 else 0)
    cum_cost[i+1] = (i+1) * COST_INSPECTION_COP
total_failures = int(y_2020.sum())
pct_found = cum_fail / total_failures * 100
net_benefit = cum_loss - cum_cost

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1 = axes[0]
ax1.plot(n_insp/n_total*100, pct_found, 'b-', linewidth=2, label='Model (no burn rate)')
ax1.plot([0,100],[0,100], 'k--', alpha=0.5, label='Random')
ax1.set_xlabel('Inspected (%)'); ax1.set_ylabel('Failures Found (%)'); ax1.set_title('Cumulative Gain', fontweight='bold')
ax1.legend(fontsize=8); ax1.set_xlim([0, 30])
ax2 = axes[1]
ax2.plot(n_insp, net_benefit/1e6, 'g-', linewidth=2)
opt_idx = np.argmax(net_benefit)
ax2.axvline(x=opt_idx, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_idx}')
ax2.scatter([opt_idx], [net_benefit[opt_idx]/1e6], color='red', s=100, zorder=5)
ax2.set_xlabel('Inspections'); ax2.set_ylabel('Net Benefit (M COP)'); ax2.set_title('Net Economic Benefit', fontweight='bold')
ax2.legend(fontsize=9)
fig.suptitle('Figure 11: Cost-Benefit (NO Burn Rate)', fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(OUTPUT_DIR / "fig11_cost_benefit.png", dpi=300, bbox_inches='tight'); plt.close()
print(f"\n  Optimal inspections: {opt_idx:,}")
print(f"  Max net benefit: {net_benefit[opt_idx]:,.0f} COP ({net_benefit[opt_idx]/1e6:.1f}M)")
print(f"  Failures caught: {int(cum_fail[opt_idx])}/{total_failures} ({pct_found[opt_idx]:.1f}%)")
print("  Figure 11 saved.")

# ============================================================================
# Section 11: Bootstrap CIs
# ============================================================================
print("\n[11/12] Bootstrap confidence intervals (1000 iterations)...")
N_BOOTSTRAP = 1000; rng = np.random.RandomState(SEED)
bootstrap_results = []
y_test_arr = np.array(y_2020)
for name, cal_model in calibrated_models.items():
    proba = cal_model.predict_proba(X_test_pp)[:, 1]
    val_proba = cal_model.predict_proba(X_val_pp)[:, 1]
    best_t, _ = optimize_threshold_f2(y_val_2019, val_proba)
    boot_roc, boot_pr, boot_f2 = [], [], []
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_test_arr), len(y_test_arr), replace=True)
        y_b, p_b = y_test_arr[idx], proba[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b): continue
        boot_roc.append(roc_auc_score(y_b, p_b))
        boot_pr.append(average_precision_score(y_b, p_b))
        boot_f2.append(fbeta_score(y_b, (p_b >= best_t).astype(int), beta=2, zero_division=0))
    bootstrap_results.append({
        'Model': name,
        'ROC-AUC': f"{np.mean(boot_roc):.4f} [{np.percentile(boot_roc, 2.5):.4f}, {np.percentile(boot_roc, 97.5):.4f}]",
        'PR-AUC':  f"{np.mean(boot_pr):.4f} [{np.percentile(boot_pr, 2.5):.4f}, {np.percentile(boot_pr, 97.5):.4f}]",
        'F2':      f"{np.mean(boot_f2):.4f} [{np.percentile(boot_f2, 2.5):.4f}, {np.percentile(boot_f2, 97.5):.4f}]",
    })
    print(f"  {name}: done")
boot_df = pd.DataFrame(bootstrap_results)
print("\n  Bootstrap CIs:")
print(boot_df.to_string(index=False))

# ============================================================================
# Section 12: Export
# ============================================================================
print("\n[12/12] Exporting results...")
table1.to_csv(OUTPUT_DIR / "table1_dataset_summary.csv", index=False)
table2.to_csv(OUTPUT_DIR / "table2_smote_comparison.csv", index=False)
results_df.to_csv(OUTPUT_DIR / "table3_model_comparison.csv", index=False)
risk_df.head(10).to_csv(OUTPUT_DIR / "table4_top10_risk.csv", index=False)
boot_df.to_csv(OUTPUT_DIR / "table3_bootstrap_CIs.csv", index=False)

# LaTeX tables
with open(OUTPUT_DIR / "table1_dataset_summary.tex", 'w') as f:
    f.write(table1.to_latex(index=False, caption="Dataset Summary", label="tab:dataset"))
with open(OUTPUT_DIR / "table2_smote_comparison.tex", 'w') as f:
    f.write(table2.to_latex(index=False, caption="Resampling Strategy Comparison", label="tab:smote"))
latex_cols = ['Model', 'ROC-AUC', 'PR-AUC', 'F2', 'Precision', 'Recall', 'Lift@5%']
with open(OUTPUT_DIR / "table3_model_comparison.tex", 'w') as f:
    f.write(results_df[latex_cols].to_latex(index=False, float_format='%.4f',
            caption="Model Comparison (NO Burn Rate)", label="tab:models_noburn"))

print("\n  Saved outputs:")
for f in sorted(OUTPUT_DIR.glob('*')):
    print(f"    {f.name:<45s} {f.stat().st_size/1024:>8.1f} KB")

# ============================================================================
# Verification
# ============================================================================
print("\n" + "=" * 60)
print("VERIFICATION CHECKLIST (NO BURN RATE VERSION)")
print("=" * 60)
print(f"[{'PASS' if len(df_2019)==15873 and n_pos_19==807 else 'FAIL'}] 2019: 15,873 rows, 807 positives")
print(f"[{'PASS' if len(df_2020)==15873 and n_pos_20==629 else 'FAIL'}] 2020: 15,873 rows, 629 positives")
print(f"[{'PASS' if BURN_RATE not in X_2019.columns else 'FAIL'}] Burn Rate EXCLUDED from features")
print(f"[{'PASS' if CRITICALITY not in X_2019.columns else 'FAIL'}] Criticality EXCLUDED from features")
print(f"[{'PASS' if results_df.iloc[0]['PR-AUC'] > y_2020.mean() else 'FAIL'}] Best PR-AUC ({results_df.iloc[0]['PR-AUC']:.4f}) > baseline ({y_2020.mean():.4f})")
print(f"[{'PASS' if len(X_train_smote) > len(X_train_pp) else 'FAIL'}] SMOTE applied")

print("\n" + "=" * 60)
print("DONE! All outputs saved to:", OUTPUT_DIR)
print("=" * 60)
