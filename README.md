# Predictive Maintenance of Distribution Transformers

**Conference:** IEOM Asia Pacific 2026, Bangkok, Thailand (March 25-27, 2026)

**Paper:** Integrated Machine Learning and Economic Risk Modeling for Reliability-Centered Asset Management in Electrical Distribution Networks

**Authors:** Md. Efatuzzaman Efat, Irtefa Waseek, Sumiya Afrose, Sayed Md. Fazle Rabbi, Md. Saiful Islam

---

## What This Project Does

This project builds a machine learning pipeline to predict which distribution transformers in the Cauca department of Colombia are likely to fail in the coming year. The dataset comes from 15,873 real transformers monitored over 2019 and 2020, with a failure rate of roughly 4-5% per year.

The core idea is to train on 2019 data and test on 2020 data, so the model genuinely has to predict the future rather than interpolate within the same time period. Beyond classification, the pipeline converts predicted failure probabilities into economic risk scores to help utilities decide which transformers to inspect first given a limited budget.

**Dataset source:** Bravo Montenegro et al. (2021), *Data in Brief*, vol. 38, pp. 107454. Available on Mendeley Data.

---

## The Two Pipelines

There are two parallel pipelines in this repo. Only Pipeline 2 was used in the submitted paper.

| | Pipeline 1 | Pipeline 2 (Paper) |
|---|---|---|
| Notebook | `IEOM_Transformer_PredMaint.ipynb` | `IEOM_NoBurnRate_PredMaint.ipynb` |
| Script | `run_ieom.py` | `IEOM_NoBurnRate_Run.py` |
| Output folder | `outputs/` | `outputs_no_burn/` |
| Burn Rate feature | Included | Excluded |
| Criticality feature | Included | Excluded |

Burn Rate is the historical failure frequency of each transformer. It is a very strong predictor, but it requires years of records to compute and does not exist for new transformers. Pipeline 2 answers the more practically useful question: how well can we predict failures using only physical and operational data?

---

## Methods

- **Models tested:** Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Imbalance handling:** SMOTE and class weighting comparison (Table 2 in the paper)
- **Hyperparameter tuning:** RandomizedSearchCV with 5-fold stratified cross-validation, optimizing PR-AUC
- **Probability calibration:** Platt scaling (sigmoid) on a held-out calibration set
- **Evaluation:** PR-AUC (primary), ROC-AUC, F2, Precision, Recall, Lift@5%, P@5%, R@5%
- **Threshold selection:** F2-score maximization on the 2019 validation set
- **Explainability:** SHAP TreeExplainer on XGBoost
- **Economic layer:** Expected failure cost converted to inspection priority ranking with NPV optimization

---

## Key Results (Paper, Pipeline 2)

| Model | ROC-AUC | PR-AUC | F2 | Lift@5% |
|---|---|---|---|---|
| Logistic Regression | 0.727 | 0.153 | 0.282 | 5.70 |
| SVM (RBF) | 0.725 | 0.140 | 0.294 | 5.12 |
| XGBoost | 0.732 | 0.134 | 0.291 | 4.74 |
| Random Forest | 0.733 | 0.129 | 0.303 | 5.38 |

Logistic Regression achieves the best PR-AUC (0.153), which is the primary metric for this imbalanced dataset. Inspecting the top 5% of highest-risk transformers flagged by the model captures failures at 5.7x the rate of random inspection.

---

## How to Run

**Requirements**

```
pip install -r requirements.txt
```

The dataset files (`Dataset_Year_2019.xlsx`, `Dataset_Year_2020.xlsx`) are not included in this repo. Download them from Mendeley Data (Bravo Montenegro et al., 2021) and place them at:

```
ifti_vai/Transformer/Dataset_Year_2019.xlsx
ifti_vai/Transformer/Dataset_Year_2020.xlsx
```

Or update the `DATA_DIR` path at the top of the script.

**Run Pipeline 2 (the paper pipeline):**

```bash
python IEOM_NoBurnRate_Run.py
```

Outputs are saved to `outputs_no_burn/`.

**Run as a Jupyter notebook:**

```bash
jupyter notebook IEOM_NoBurnRate_PredMaint.ipynb
```

---

## Output Files

Both `outputs/` (Pipeline 1) and `outputs_no_burn/` (Pipeline 2) contain the same structure:

| File | Description |
|---|---|
| `fig01_class_distribution.png` | Class imbalance visualization |
| `fig05_calibration_curves.png` | Post-calibration reliability curves |
| `fig06_pr_curves.png` | Precision-Recall curves for all models |
| `fig07_roc_curves.png` | ROC curves for all models |
| `fig08_confusion_matrices.png` | Confusion matrices at F2-optimal threshold |
| `fig09_shap_beeswarm.png` | SHAP feature importance (XGBoost) |
| `fig11_cost_benefit.png` | Cumulative gain and net economic benefit curve |
| `table2_smote_comparison.csv` | Resampling strategy comparison |
| `table3_model_comparison.csv` | Full model evaluation metrics |
| `table3_bootstrap_CIs.csv` | 95% bootstrap confidence intervals |

---

## Project Structure

```
predictive_analysis/
├── IEOM_NoBurnRate_PredMaint.ipynb   # Final notebook (paper)
├── IEOM_NoBurnRate_Run.py            # Python script version (paper)
├── IEOM_Transformer_PredMaint.ipynb  # Pipeline 1 (with Burn Rate)
├── run_ieom.py                       # Python script for Pipeline 1
├── outputs_no_burn/                  # Paper figures and tables
├── outputs/                          # Pipeline 1 figures and tables
├── DOCUMENTATION.md                  # Detailed technical guide
├── requirements.txt
└── README.md
```

---

## References

Bravo M, D.A., Alvarez Q, L.I., and Lozano M, C.A. Dataset of distribution transformers for predictive maintenance. *Data in Brief*, vol. 38, pp. 107454, 2021.

Alvarez Quinones, L.I., Lozano-Moncada, C.A., and Bravo Montenegro, D.A. Machine learning for predictive maintenance scheduling of distribution transformers. *Journal of Quality in Maintenance Engineering*, vol. 29, no. 1, pp. 188-202, 2022.
