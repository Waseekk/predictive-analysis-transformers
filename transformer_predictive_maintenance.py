"""
Enhanced Predictive Maintenance of Distribution Transformers
Using Gradient Boosting Ensemble with Explainable AI

For IEOM Bangkok 2026 Conference Submission
Target: Beat baseline SVM accuracy of 97.25%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================

def load_data():
    """Load datasets for 2019 and 2020"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df_2019 = pd.read_excel('Dataset_Year_2019.xlsx')
    df_2020 = pd.read_excel('Dataset_Year_2020.xlsx')

    print(f"2019 Dataset: {df_2019.shape[0]} rows, {df_2019.shape[1]} columns")
    print(f"2020 Dataset: {df_2020.shape[0]} rows, {df_2020.shape[1]} columns")

    return df_2019, df_2020

def explore_data(df, year):
    """Exploratory Data Analysis"""
    print(f"\n{'=' * 60}")
    print(f"EXPLORATORY DATA ANALYSIS - {year}")
    print("=" * 60)

    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")

    # Target variable distribution
    target_col = df.columns[-1]  # Last column is 'Burned transformers'
    print(f"\nTarget Variable Distribution ({target_col}):")
    print(df[target_col].value_counts())
    print(f"Class Imbalance Ratio: {df[target_col].value_counts()[0] / df[target_col].value_counts()[1]:.2f}:1")

    return target_col

def visualize_data(df_2019, df_2020, target_col):
    """Create visualizations for the paper"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get target columns for each year (they have different names)
    target_2019 = [col for col in df_2019.columns if 'burned' in col.lower() or 'quemado' in col.lower()][0]
    target_2020 = [col for col in df_2020.columns if 'burned' in col.lower() or 'quemado' in col.lower()][0]

    # 1. Target distribution comparison
    ax1 = axes[0, 0]
    years = ['2019', '2020']
    failed = [df_2019[target_2019].sum(), df_2020[target_2020].sum()]
    total = [len(df_2019), len(df_2020)]
    x = np.arange(len(years))
    width = 0.35
    ax1.bar(x - width/2, total, width, label='Total', color='steelblue')
    ax1.bar(x + width/2, failed, width, label='Failed', color='coral')
    ax1.set_ylabel('Count')
    ax1.set_title('Transformer Failure Distribution by Year')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()

    # 2. Burn rate distribution (most important feature)
    ax2 = axes[0, 1]
    burn_rate_col = [col for col in df_2019.columns if 'burn' in col.lower() or 'tasa' in col.lower()][0]
    df_2019[burn_rate_col].hist(bins=50, ax=ax2, alpha=0.7, label='2019', color='steelblue')
    df_2020[burn_rate_col].hist(bins=50, ax=ax2, alpha=0.7, label='2020', color='coral')
    ax2.set_xlabel('Burn Rate')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Burn Rate Distribution')
    ax2.legend()

    # 3. Power distribution
    ax3 = axes[0, 2]
    power_col = [col for col in df_2019.columns if 'power' in col.lower() or 'potencia' in col.lower()][0]
    df_2019[power_col].hist(bins=30, ax=ax3, alpha=0.7, label='2019', color='steelblue')
    ax3.set_xlabel('Power (kVA)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Transformer Power Distribution')

    # 4. Correlation heatmap for numerical features (2019)
    ax4 = axes[1, 0]
    numeric_cols = df_2019.select_dtypes(include=[np.number]).columns
    corr_matrix = df_2019[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax4, center=0)
    ax4.set_title('Feature Correlation Heatmap (2019)')

    # 5. Failure rate by location
    ax5 = axes[1, 1]
    location_col = [col for col in df_2019.columns if 'ubica' in col.lower() or 'location' in col.lower()][0]
    fail_by_loc = df_2019.groupby(location_col)[target_2019].mean() * 100
    fail_by_loc.plot(kind='bar', ax=ax5, color=['coral', 'steelblue'])
    ax5.set_xlabel('Location (0=Rural, 1=Urban)')
    ax5.set_ylabel('Failure Rate (%)')
    ax5.set_title('Failure Rate by Location')
    ax5.tick_params(axis='x', rotation=0)

    # 6. Users vs Failures
    ax6 = axes[1, 2]
    users_col = [col for col in df_2019.columns if 'user' in col.lower() or 'usuario' in col.lower()][0]
    df_failed = df_2019[df_2019[target_2019] == 1]
    df_ok = df_2019[df_2019[target_2019] == 0]
    ax6.scatter(df_ok[users_col], df_ok[burn_rate_col], alpha=0.3, label='OK', s=10)
    ax6.scatter(df_failed[users_col], df_failed[burn_rate_col], alpha=0.5, label='Failed', s=10, color='red')
    ax6.set_xlabel('Number of Users')
    ax6.set_ylabel('Burn Rate')
    ax6.set_title('Users vs Burn Rate by Failure Status')
    ax6.legend()

    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: eda_visualizations.png")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()

    # Get column names
    cols = df_processed.columns.tolist()

    # Identify target column (last column)
    target_col = cols[-1]

    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Store column names for later use
    feature_names = X.columns.tolist()

    return X, y, feature_names, label_encoders

def create_engineered_features(X, feature_names):
    """Create additional engineered features"""
    X_eng = X.copy()

    # Find relevant columns by partial matching
    cols_lower = {col.lower(): col for col in feature_names}

    # Try to find burn rate column
    burn_col = None
    for key, col in cols_lower.items():
        if 'burn' in key or 'tasa' in key or 'quema' in key:
            burn_col = col
            break

    # Try to find EENS column
    eens_col = None
    for key, col in cols_lower.items():
        if 'eens' in key or 'ens' in key:
            eens_col = col
            break

    # Try to find users column
    users_col = None
    for key, col in cols_lower.items():
        if 'user' in key or 'usuario' in key:
            users_col = col
            break

    # Try to find power column
    power_col = None
    for key, col in cols_lower.items():
        if 'power' in key or 'potencia' in key:
            power_col = col
            break

    # Create interaction features if columns found
    if burn_col and eens_col:
        X_eng['burn_x_eens'] = X_eng[burn_col] * X_eng[eens_col]

    if burn_col and users_col:
        X_eng['burn_x_users'] = X_eng[burn_col] * X_eng[users_col]

    if power_col and users_col:
        X_eng['power_per_user'] = X_eng[power_col] / (X_eng[users_col] + 1)

    if burn_col:
        X_eng['burn_squared'] = X_eng[burn_col] ** 2
        X_eng['burn_log'] = np.log1p(X_eng[burn_col])

    new_feature_names = X_eng.columns.tolist()

    print(f"Original features: {len(feature_names)}")
    print(f"After engineering: {len(new_feature_names)}")
    print(f"New features added: {set(new_feature_names) - set(feature_names)}")

    return X_eng, new_feature_names

# ============================================================
# 3. MODEL TRAINING AND EVALUATION
# ============================================================

def get_models():
    """Define all models to compare"""
    models = {
        # Baseline (original paper)
        'SVM (Baseline)': SVC(kernel='rbf', probability=True, random_state=42),

        # Classical ML
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),

        # Gradient Boosting Methods
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),

        # Neural Network
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }

    return metrics, model, y_pred, y_pred_proba

def run_experiments(X, y, feature_names, use_smote=True, use_feature_eng=True):
    """Run all experiments and compare models"""
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENTS")
    print("=" * 60)

    # Feature engineering
    if use_feature_eng:
        X, feature_names = create_engineered_features(X, feature_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    if use_smote:
        print("\nApplying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"Before SMOTE: {np.bincount(y_train)}")
        print(f"After SMOTE: {np.bincount(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Get models
    models = get_models()

    # Results storage
    results = []
    trained_models = {}

    print("\nTraining and evaluating models...")
    print("-" * 80)

    for name, model in models.items():
        try:
            metrics, trained_model, y_pred, y_pred_proba = evaluate_model(
                model, X_train_balanced, X_test_scaled, y_train_balanced, y_test, name
            )
            results.append({'Model': name, **metrics})
            trained_models[name] = trained_model
            print(f"{name:25} | Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f if metrics['ROC-AUC'] else 'N/A'}")
        except Exception as e:
            print(f"{name:25} | Error: {str(e)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Sorted by Accuracy)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nSaved: model_comparison_results.csv")

    return results_df, trained_models, X_test_scaled, y_test, scaler, feature_names

# ============================================================
# 4. ENSEMBLE MODEL
# ============================================================

def create_ensemble(X_train, X_test, y_train, y_test):
    """Create stacking ensemble for best performance"""
    print("\n" + "=" * 60)
    print("CREATING ENSEMBLE MODEL")
    print("=" * 60)

    # Base models
    base_models = [
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    # Meta-learner
    meta_learner = LogisticRegression(max_iter=1000)

    # Stacking ensemble
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    # Train
    print("Training stacking ensemble...")
    stacking_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\nStacking Ensemble Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return stacking_clf, metrics, y_pred, y_pred_proba

# ============================================================
# 5. EXPLAINABILITY WITH SHAP
# ============================================================

def explain_model(model, X_test, feature_names, model_name='XGBoost'):
    """Generate SHAP explanations"""
    print("\n" + "=" * 60)
    print(f"SHAP EXPLAINABILITY ANALYSIS - {model_name}")
    print("=" * 60)

    # Create SHAP explainer
    if 'XGB' in model_name or 'LGBM' in model_name or 'Random' in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_test[:100])

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test[:500])  # Use subset for speed

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, get positive class

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: shap_summary_{model_name.lower().replace(' ', '_')}.png")

    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: shap_importance_{model_name.lower().replace(' ', '_')}.png")

    return shap_values

# ============================================================
# 6. CROSS-YEAR VALIDATION (Novel Contribution)
# ============================================================

def cross_year_validation(df_2019, df_2020):
    """Train on 2019, test on 2020 (temporal validation)"""
    print("\n" + "=" * 60)
    print("CROSS-YEAR VALIDATION (Train: 2019, Test: 2020)")
    print("=" * 60)

    # Standardize target column names
    df_2019_copy = df_2019.copy()
    df_2020_copy = df_2020.copy()
    target_2019 = [col for col in df_2019.columns if 'burned' in col.lower()][0]
    target_2020 = [col for col in df_2020.columns if 'burned' in col.lower()][0]
    df_2019_copy = df_2019_copy.rename(columns={target_2019: 'Burned'})
    df_2020_copy = df_2020_copy.rename(columns={target_2020: 'Burned'})

    # Preprocess both years
    X_2019, y_2019, feature_names, _ = preprocess_data(df_2019_copy)
    X_2020, y_2020, _, _ = preprocess_data(df_2020_copy)

    # Feature engineering
    X_2019, feature_names = create_engineered_features(X_2019, feature_names)
    X_2020, _ = create_engineered_features(X_2020, X_2020.columns.tolist())

    # Align columns
    common_cols = list(set(X_2019.columns) & set(X_2020.columns))
    X_2019 = X_2019[common_cols]
    X_2020 = X_2020[common_cols]

    # Scale
    scaler = StandardScaler()
    X_2019_scaled = scaler.fit_transform(X_2019)
    X_2020_scaled = scaler.transform(X_2020)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_2019_scaled, y_2019)

    # Train best models
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_2020_scaled)
        y_pred_proba = model.predict_proba(X_2020_scaled)[:, 1]

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_2020, y_pred),
            'Precision': precision_score(y_2020, y_pred),
            'Recall': recall_score(y_2020, y_pred),
            'F1-Score': f1_score(y_2020, y_pred),
            'ROC-AUC': roc_auc_score(y_2020, y_pred_proba)
        }
        results.append(metrics)
        print(f"{name:20} | Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1-Score']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('cross_year_validation_results.csv', index=False)
    print("\nSaved: cross_year_validation_results.csv")

    return results_df

# ============================================================
# 7. VISUALIZATION FOR PAPER
# ============================================================

def create_paper_figures(results_df, y_test, y_pred, y_pred_proba):
    """Create publication-quality figures"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Model comparison bar chart
    ax1 = axes[0]
    models = results_df['Model'].head(6)
    accuracies = results_df['Accuracy'].head(6) * 100
    colors = ['coral' if 'SVM' in m else 'steelblue' for m in models]
    bars = ax1.barh(models, accuracies, color=colors)
    ax1.axvline(x=97.25, color='red', linestyle='--', label='Baseline (97.25%)')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{acc:.2f}%', va='center', fontsize=9)

    # 2. Confusion Matrix (best model)
    ax2 = axes[1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Not Failed', 'Failed'],
                yticklabels=['Not Failed', 'Failed'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Best Model)')

    # 3. ROC Curve
    ax3 = axes[2]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax3.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('paper_figures.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: paper_figures.png")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE OF DISTRIBUTION TRANSFORMERS")
    print("Enhanced ML Approach for IEOM Bangkok 2026")
    print("=" * 60)

    # 1. Load data
    df_2019, df_2020 = load_data()

    # 2. Explore data
    target_col = explore_data(df_2019, 2019)
    explore_data(df_2020, 2020)

    # 3. Visualize data
    visualize_data(df_2019, df_2020, target_col)

    # 4. Preprocess data (combined dataset for more training data)
    # Standardize target column names before combining
    df_2019_copy = df_2019.copy()
    df_2020_copy = df_2020.copy()
    target_2019 = [col for col in df_2019.columns if 'burned' in col.lower()][0]
    target_2020 = [col for col in df_2020.columns if 'burned' in col.lower()][0]
    df_2019_copy = df_2019_copy.rename(columns={target_2019: 'Burned'})
    df_2020_copy = df_2020_copy.rename(columns={target_2020: 'Burned'})
    df_combined = pd.concat([df_2019_copy, df_2020_copy], ignore_index=True)
    X, y, feature_names, _ = preprocess_data(df_combined)

    # 5. Run experiments
    results_df, trained_models, X_test, y_test, scaler, feature_names = run_experiments(
        X, y, feature_names, use_smote=True, use_feature_eng=True
    )

    # 6. Create and evaluate ensemble
    # Re-split for ensemble training
    X_eng, feature_names = create_engineered_features(X, X.columns.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    ensemble_model, ensemble_metrics, y_pred_ensemble, y_pred_proba_ensemble = create_ensemble(
        X_train_balanced, X_test_scaled, y_train_balanced, y_test
    )

    # Add ensemble to results
    ensemble_result = {'Model': 'Stacking Ensemble', **ensemble_metrics}
    results_df = pd.concat([results_df, pd.DataFrame([ensemble_result])], ignore_index=True)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Including Ensemble)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 7. SHAP Explainability for best tree-based model
    best_model = trained_models.get('XGBoost') or trained_models.get('LightGBM')
    if best_model:
        explain_model(best_model, X_test_scaled, feature_names, 'XGBoost')

    # 8. Cross-year validation
    cross_year_results = cross_year_validation(df_2019, df_2020)

    # 9. Create paper figures
    create_paper_figures(results_df, y_test, y_pred_ensemble, y_pred_proba_ensemble)

    # 10. Final summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    best_accuracy = results_df['Accuracy'].max() * 100
    baseline_accuracy = 97.25
    improvement = best_accuracy - baseline_accuracy

    print(f"Baseline SVM Accuracy: {baseline_accuracy:.2f}%")
    print(f"Best Model Accuracy:   {best_accuracy:.2f}%")
    print(f"Improvement:           {improvement:+.2f}%")
    print(f"Best Model:            {results_df.iloc[0]['Model']}")

    # Save final results
    results_df.to_csv('final_results.csv', index=False)
    print("\nAll results saved to CSV files.")

    return results_df, trained_models, ensemble_model

if __name__ == "__main__":
    results, models, ensemble = main()
