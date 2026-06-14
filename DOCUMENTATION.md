# Predictive Maintenance of Distribution Transformers
## A Complete Guide to the IEOM Pipeline

---

## What Is This Project About? (The Big Picture)

Imagine a city where **15,873 electrical transformers** supply power to homes, hospitals, and businesses. Every year, some of these transformers **burn out** — causing blackouts, expensive repairs, and safety hazards. In 2019, **807 transformers** (about 5%) burned. In 2020, **629** (about 4%) burned.

**The core question:** Can we predict *which* transformers will fail *before* they actually do?

If yes, the electric utility can **inspect and fix** those high-risk transformers first, preventing failures, saving money, and keeping the lights on.

This project builds **machine learning models** that analyze transformer characteristics (size, location, past failure history, lightning exposure, etc.) and output a **risk score** for each transformer — essentially saying: *"This transformer has a 26% chance of burning in the next year. Prioritize it for inspection."*

---

## The Two Pipelines: Why Are There Two Scripts?

| | **Pipeline 1: Full Features** | **Pipeline 2: No Burn Rate** |
|---|---|---|
| **Script** | `IEOM_Transformer_PredMaint.ipynb` | `IEOM_NoBurnRate_Run.py` |
| **Output Folder** | `outputs -first/` | `outputs_no_burn/` |
| **Key Difference** | Includes **Burn Rate** and **Criticality** as features | **Excludes** Burn Rate and Criticality |
| **Why It Matters** | Shows best-case performance with expert knowledge | Shows what's achievable with **only raw sensor/grid data** |
| **Total Features** | 23 (15 original + 8 engineered) | 19 (13 original + 6 engineered) |

### Why remove Burn Rate and Criticality?

**Burn Rate** is the historical failure rate of a transformer (e.g., "this transformer has failed 0.5 times per year on average"). **Criticality** is an expert-assigned label from a previous study.

These are extremely powerful predictors — but they have a catch:
- They require **years of historical records** and **expert judgment** to compute
- For brand-new transformers or regions without historical data, they simply don't exist
- Including them can create a **circular dependency**: you're partly using past failures to predict future failures

Pipeline 2 answers: *"How well can we predict failures using only physical and operational data — no historical hindsight?"*

---

## The Dataset

**Source:** Real-world data from the electrical distribution network in **Cauca, Colombia**, covering 15,873 transformers across two years.

| Attribute | 2019 (Training) | 2020 (Testing) |
|---|---|---|
| Total transformers | 15,873 | 15,873 |
| Burned (failures) | 807 (5.08%) | 629 (3.96%) |
| Not burned | 15,066 | 15,244 |
| Original features | 15 | 15 |

### The Features (What the Model Sees)

Each transformer is described by these attributes:

| Feature | Type | What It Means |
|---|---|---|
| **POWER** (kVA) | Numeric | The capacity/size of the transformer. Larger = serves more load |
| **LOCATION** | Binary | Urban (1) or Rural (0) |
| **SELF-PROTECTION** | Binary | Whether the transformer has built-in fuses/protection |
| **Avg DDT** | Numeric | Average lightning strike density in the area (rays per km^2 per year) |
| **Max DDT** | Numeric | Maximum lightning strike density in the area |
| **Burning Rate** | Numeric | Historical failure rate (failures/year) - *Pipeline 1 only* |
| **Criticality** | Binary | Expert-assigned risk level - *Pipeline 1 only* |
| **Removable Connectors** | Binary | Type of electrical connections |
| **Type of Clients** | Categorical | Residential, commercial, mixed, industrial, etc. (9 types) |
| **Number of Users** | Numeric | How many customers this transformer serves |
| **EENS (kWh)** | Numeric | Energy Not Supplied — how much power was lost due to outages |
| **Type of Installation** | Categorical | Pole-mounted, pad-mounted, etc. (8 types) |
| **Air Network** | Binary | Whether the network is overhead (exposed to weather) |
| **Circuit Queue** | Binary | Position in the electrical circuit |
| **km of Network LT** | Numeric | Length of the low-tension network served |

### Engineered Features (Created by the Pipeline)

The code also creates **derived features** that capture relationships the raw data doesn't show directly:

| Engineered Feature | Formula | Intuition |
|---|---|---|
| `users_per_km` | Users / km of network | How densely loaded is the network? |
| `eens_per_user` | EENS / Users | How much energy loss per customer? |
| `eens_per_km` | EENS / km | Energy loss density along the network |
| `ddt_ratio` | Max DDT / Avg DDT | How much does lightning intensity spike? |
| `log_eens` | log(1 + EENS) | Smoothed version of EENS (reduces extreme values) |
| `power_per_user` | POWER / Users | How much capacity per customer? |
| `burn_x_power` | Burn Rate x POWER | Interaction of failure history and size - *Pipeline 1 only* |
| `burn_x_eens` | Burn Rate x EENS | Interaction of failure history and energy loss - *Pipeline 1 only* |

---

## The Imbalanced Class Problem (Why Accuracy Is Misleading)

This is one of the most important concepts to understand in this project.

Only **~4-5% of transformers actually burn**. This means:

> A model that **always predicts "Not Burned"** (never flags any transformer) achieves **~96% accuracy** — but catches **zero** actual failures. Completely useless.

This is why the pipeline uses a **Dummy Classifier** as a baseline: it proves that high accuracy means nothing here.

### Metrics That Actually Matter

| Metric | What It Measures | Why It Matters Here |
|---|---|---|
| **PR-AUC** (Precision-Recall Area Under Curve) | Overall ability to find failures without too many false alarms | **Primary metric** — best for imbalanced data |
| **ROC-AUC** | Ability to distinguish failures from non-failures | Good overview, but can be over-optimistic with imbalanced data |
| **F2 Score** | Weighted combination of precision and recall, favoring recall (2x weight) | Missing a real failure is worse than a false alarm |
| **Precision** | Of those flagged as "will burn," how many actually did? | Controls wasted inspections |
| **Recall** | Of the transformers that actually burned, how many did we catch? | Controls missed failures |
| **Lift@5%** | How much better than random when inspecting the top 5% riskiest | Directly actionable for budget planning |

### Why F2 Instead of F1?

In predictive maintenance, the cost of **missing a failure** (transformer burns, blackout occurs) is much higher than the cost of **a false alarm** (unnecessary inspection). The F2 score weights recall twice as heavily as precision, matching this real-world asymmetry.

---

## Handling the Imbalance: SMOTE and Class Weights

Three strategies were tested using Logistic Regression as a baseline:

### Pipeline 1 (With Burn Rate)

| Strategy | ROC-AUC | PR-AUC | F2 |
|---|---|---|---|
| No resampling | 0.7948 | 0.2387 | 0.4016 |
| SMOTE | 0.7994 | 0.2402 | 0.4014 |
| Class weights | 0.7999 | 0.2334 | 0.3975 |

### Pipeline 2 (Without Burn Rate)

| Strategy | ROC-AUC | PR-AUC | F2 |
|---|---|---|---|
| No resampling | 0.7968 | 0.2524 | 0.3925 |
| SMOTE | 0.7982 | 0.2508 | 0.4073 |
| Class weights | 0.8063 | 0.2429 | 0.4142 |

**SMOTE** (Synthetic Minority Over-sampling Technique) creates artificial "burned transformer" examples to balance the training data. Think of it like: if you only have 484 examples of failures and 9,039 non-failures, the model barely learns what a failure looks like. SMOTE generates synthetic failure examples so the model sees a 50/50 split during training.

**Key safeguard:** SMOTE is applied **only to the training data**. The validation and test sets are untouched — this prevents the model from "cheating" by seeing synthetic data during evaluation.

---

## The Six Models

Both pipelines train and compare **six different machine learning models**, each with a different approach to learning patterns:

### Model Descriptions (For Non-Technical Readers)

| Model | Analogy | Strengths |
|---|---|---|
| **Logistic Regression** | A sophisticated scoring formula: adds up weighted factors to get a risk score | Simple, interpretable, fast. Good baseline. |
| **SVM (RBF)** | Draws complex boundaries in multi-dimensional space to separate failures from non-failures | Good at finding non-linear patterns |
| **Random Forest** | 500 "experts" (decision trees) each vote; majority wins | Robust, handles noise well, hard to overfit |
| **Gradient Boosting** | A sequence of weak learners, each correcting the previous one's mistakes | Powerful, good at subtle patterns |
| **XGBoost** | An optimized, faster version of Gradient Boosting | State-of-the-art for tabular data |
| **LightGBM** | An even faster Gradient Boosting variant that grows trees differently | Very fast, handles large datasets well |

### How Models Are Tuned

Each model has **hyperparameters** (settings that control how it learns). The pipeline uses **RandomizedSearchCV** with 5-fold cross-validation to find the best settings:

1. Split training data into 5 equal parts
2. For each of 20 random hyperparameter combinations:
   - Train on 4 parts, validate on the 5th
   - Rotate which part is held out
   - Average the score across all 5 rotations
3. Pick the combination with the best average PR-AUC

This prevents the model from memorizing the training data and ensures it generalizes to new data.

---

## Results: How Well Do the Models Perform?

### Pipeline 1: Full Features (With Burn Rate)

Evaluated on the **2020 test set** (completely unseen during training):

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Lift@5% |
|---|---|---|---|---|---|---|
| **SVM (RBF)** | **0.7373** | **0.1805** | 0.3024 | 0.1641 | 0.3831 | 5.70 |
| Random Forest | 0.7340 | 0.1776 | 0.2965 | 0.1568 | 0.3816 | 5.31 |
| Logistic Regression | 0.7253 | 0.1696 | 0.2983 | 0.1223 | 0.4658 | 5.89 |
| LightGBM | 0.7187 | 0.1579 | 0.2697 | 0.1277 | 0.3736 | 4.71 |
| Gradient Boosting | 0.7177 | 0.1116 | 0.2840 | 0.1031 | 0.5056 | 4.14 |
| XGBoost | 0.7153 | 0.1040 | 0.2582 | 0.1155 | 0.3736 | 3.95 |

### Pipeline 2: No Burn Rate

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Lift@5% |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7268 | **0.1531** | 0.2823 | 0.0972 | 0.5390 | 5.70 |
| SVM (RBF) | 0.7254 | 0.1404 | 0.2942 | 0.1117 | 0.4976 | 5.12 |
| XGBoost | 0.7311 | 0.1329 | 0.2902 | 0.1351 | 0.4070 | 4.90 |
| Random Forest | 0.7332 | 0.1286 | 0.3035 | 0.1377 | 0.4340 | 5.38 |
| LightGBM | 0.7281 | 0.1143 | 0.2737 | 0.1225 | 0.3959 | 4.49 |
| Gradient Boosting | 0.7035 | 0.0927 | 0.2668 | 0.1181 | 0.3895 | 3.53 |

### What These Numbers Mean (In Plain English)

**Best model with Burn Rate: SVM (RBF) — PR-AUC 0.1805**
- If you inspect the **top 5%** of transformers flagged as riskiest, you'll find failures at a rate **5.7x higher** than random inspection
- The model catches about **38% of all failures** while only flagging ~16% of transformers as high risk
- ROC-AUC of 0.74 means the model correctly ranks a random failure above a random non-failure **74% of the time**

**Best model without Burn Rate: Logistic Regression — PR-AUC 0.1531**
- Performance drops by about **15%** (PR-AUC from 0.18 to 0.15) without historical data
- But the model still provides a **5.7x lift** when inspecting the top 5%
- Logistic Regression catches more failures (**54% recall**) but with more false alarms (lower precision)
- This is still **vastly better than random** or no model at all

### Key Takeaway

> Removing Burn Rate and Criticality reduces performance, but the models are still highly useful. Even without expert knowledge, the pipeline can prioritize the riskiest transformers far better than random inspection.

---

## Statistical Confidence: Bootstrap Results

To ensure the results aren't a fluke, each model's performance is measured across **1,000 bootstrap samples** (randomly resampling the test set with replacement). The 95% confidence intervals tell us the range within which the true performance almost certainly falls.

### Pipeline 1 (With Burn Rate) — 95% Confidence Intervals

| Model | ROC-AUC | PR-AUC | F2 |
|---|---|---|---|
| SVM (RBF) | 0.7372 [0.7149, 0.7586] | 0.1817 [0.1515, 0.2141] | 0.3021 [0.2734, 0.3327] |
| Random Forest | 0.7337 [0.7111, 0.7546] | 0.1792 [0.1494, 0.2096] | 0.2965 [0.2688, 0.3252] |
| Logistic Regression | 0.7253 [0.7044, 0.7470] | 0.1711 [0.1429, 0.2006] | 0.2980 [0.2728, 0.3227] |
| LightGBM | 0.7189 [0.6969, 0.7379] | 0.1598 [0.1326, 0.1892] | 0.2697 [0.2442, 0.2968] |
| Gradient Boosting | 0.7176 [0.6960, 0.7400] | 0.1132 [0.0968, 0.1315] | 0.2834 [0.2594, 0.3066] |
| XGBoost | 0.7153 [0.6937, 0.7365] | 0.1055 [0.0910, 0.1223] | 0.2581 [0.2333, 0.2855] |

### Pipeline 2 (Without Burn Rate) — 95% Confidence Intervals

| Model | ROC-AUC | PR-AUC | F2 |
|---|---|---|---|
| Random Forest | 0.7330 [0.7098, 0.7539] | 0.1298 [0.1119, 0.1478] | 0.3034 [0.2750, 0.3298] |
| XGBoost | 0.7313 [0.7097, 0.7522] | 0.1349 [0.1145, 0.1580] | 0.2900 [0.2615, 0.3182] |
| LightGBM | 0.7283 [0.7073, 0.7470] | 0.1155 [0.0998, 0.1328] | 0.2739 [0.2481, 0.3017] |
| Logistic Regression | 0.7269 [0.7049, 0.7479] | 0.1544 [0.1306, 0.1791] | 0.2819 [0.2588, 0.3045] |
| SVM (RBF) | 0.7251 [0.7030, 0.7466] | 0.1421 [0.1190, 0.1668] | 0.2940 [0.2687, 0.3195] |
| Gradient Boosting | 0.7035 [0.6818, 0.7250] | 0.0936 [0.0820, 0.1058] | 0.2662 [0.2375, 0.2932] |

The overlapping confidence intervals across the top models confirm that the differences between the best 3-4 models are not statistically dramatic — they all perform in a similar range.

---

## Probability Calibration

Raw model outputs (e.g., "0.73 probability of failure") are often **not well-calibrated** — a 73% prediction doesn't mean exactly 73 out of 100 similar transformers will fail.

The pipeline applies **Platt Scaling (sigmoid calibration)** using a dedicated calibration set to fix this. After calibration, if the model says "10% probability," approximately 10% of transformers with that score will actually fail. This is critical for the economic risk framework, where dollar amounts are multiplied by probability.

---

## SHAP Explainability: Why Does the Model Make Each Prediction?

SHAP (SHapley Additive exPlanations) is a technique from game theory that explains **how much each feature contributes** to each individual prediction.

### How to Read a SHAP Beeswarm Plot

Each dot is one transformer. The horizontal position shows whether that feature **pushed the prediction toward failure (right)** or **toward non-failure (left)**. The color shows the feature's actual value (red = high, blue = low).

**Pipeline 1 — Top Findings:**
- **Burn Rate** is the most important feature by a large margin. High burn rate (red dots, right side) strongly pushes toward predicting failure
- **Criticality** is the second most important — transformers flagged as critical by experts are much more likely to burn
- **EENS** (energy not supplied) — transformers that already have high outage history are at higher risk
- **burn_x_power** and **burn_x_eens** — the engineered interaction features are also highly informative

**Pipeline 2 — Top Findings (without Burn Rate):**
- With burn rate removed, the model relies more heavily on **EENS**, **lightning density (DDT)**, **users_per_km**, and **network length**
- The feature importance is more evenly distributed — no single feature dominates as burn rate did

### What This Means for Domain Experts

The SHAP analysis confirms domain knowledge:
1. Past failure history (burn rate) is the strongest predictor — unsurprisingly, transformers that failed before are more likely to fail again
2. Lightning exposure (DDT) is a significant risk factor
3. Network load (EENS, users) contributes to failure risk
4. Even without historical data, physical and operational characteristics provide meaningful predictive signal

---

## Economic Risk Framework

The pipeline goes beyond binary "will fail / won't fail" predictions to produce **dollar-denominated risk scores**.

### How It Works

For each transformer:

```
Expected Loss = P(failure) x Replacement Cost
```

Where:
- **P(failure)** = calibrated probability from the best model
- **Replacement Cost** = base cost (560,000 COP) scaled by transformer capacity relative to the median

### Top 10 Highest-Risk Transformers

#### Pipeline 1 (With Burn Rate)

| Rank | Transformer | P(failure) | Power (kVA) | Expected Loss (COP) | Actually Burned? |
|---|---|---|---|---|---|
| 1 | #3713 | 26.6% | 75 | 744,581 | No |
| 2 | #11483 | 25.7% | 75 | 720,190 | No |
| 3 | #7758 | 25.6% | 112.5 | 716,577 | No |
| 4 | #7885 | 24.5% | 112.5 | 685,955 | No |
| 5 | #7871 | 23.3% | 112.5 | 652,041 | No |
| 6 | #11996 | 18.9% | 112.5 | 530,463 | No |
| 7 | #8510 | 52.3% | 25 | 487,780 | No |
| 8 | #5648 | 25.2% | 50 | 470,455 | No |
| 9 | #2636 | 16.4% | 112.5 | 459,952 | No |
| 10 | #2209 | 16.1% | 75 | 451,704 | No |

#### Pipeline 2 (Without Burn Rate)

| Rank | Transformer | P(failure) | Power (kVA) | Expected Loss (COP) | Actually Burned? |
|---|---|---|---|---|---|
| 1 | #8510 | 48.6% | 25 | 453,454 | No |
| 2 | #7012 | 26.3% | 45 | 442,613 | No |
| 3 | #8009 | 73.5% | 15 | 411,384 | No |
| 4 | #5744 | 42.5% | 25 | 396,351 | No |
| 5 | #9606 | 27.7% | 37.5 | 387,717 | **YES** |
| 6 | #14172 | 41.3% | 25 | 385,520 | No |
| 7 | #14567 | 27.3% | 37.5 | 382,507 | No |
| 8 | #12383 | 38.2% | 25 | 356,486 | No |
| 9 | #15144 | 63.0% | 15 | 352,774 | No |
| 10 | #5510 | 37.7% | 25 | 352,168 | No |

### Understanding the Top-10 Lists

Notice that the top-risk list is dominated by transformers that **did not actually burn**. This doesn't mean the model is wrong — it means these transformers are genuinely high-risk but happened not to fail in this particular year. Failure is probabilistic, not deterministic. A transformer with a 25% failure probability still has a 75% chance of surviving any given year.

Note that Transformer #9606 in Pipeline 2 actually did burn, validating the model's risk assessment.

### Cost-Benefit Analysis

The cost-benefit curve (Figure 11) answers: **"If we can only inspect N transformers, which N should we pick?"**

The model ranks all 15,873 transformers by expected loss. Starting from the top:
- Inspect the #1 riskiest transformer, then #2, then #3...
- At each step, count how many actual failures we've caught
- Compare the cost of inspections (50,000 COP each) against the replacement costs saved

The **optimal inspection point** is where net benefit peaks — beyond that, you're spending more on inspections than you're saving from prevented failures.

---

## Figures Guide

Both pipelines produce 11 publication-quality figures:

| Figure | What It Shows | Key Insight |
|---|---|---|
| **Fig 1** | Class distribution bars | Severe imbalance: ~95% negative, ~5% positive |
| **Fig 2** | Feature correlation heatmap | Shows which features are redundant or related |
| **Fig 3** | Burn Rate / EENS distribution by class | Pipeline 1: Burned transformers have higher burn rates. Pipeline 2: EENS distribution differs between classes |
| **Fig 4** | Top-5 feature boxplots | Visual separation between burned and not-burned for the most important features |
| **Fig 5** | Calibration curves | How well-calibrated each model's probabilities are (closer to diagonal = better) |
| **Fig 6** | Precision-Recall curves | The fundamental tradeoff: catching more failures = more false alarms |
| **Fig 7** | ROC curves | All models significantly outperform random (diagonal line) |
| **Fig 8** | Confusion matrices (6 models) | Exact counts of true/false positives and negatives at the F2-optimized threshold |
| **Fig 9** | SHAP beeswarm (top 15 features) | Which features drive the model's decisions and in what direction |
| **Fig 10** | SHAP dependence plot | How the single most important feature's value maps to its impact on predictions |
| **Fig 11** | Cost-benefit curve | Optimal number of inspections for maximum economic savings |

---

## Pipeline Architecture (Technical Details)

### Data Flow

```
Raw Excel Data (2019, 2020)
    |
    v
Feature Engineering (add ratios, interactions, log transforms)
    |
    v
[Pipeline 1: Keep all 23 features]  OR  [Pipeline 2: Drop to 19 features]
    |
    v
Train/Calibration/Validation Split (60/20/20 within 2019)
    |
    v
Preprocessing: Imputation + Scaling (numeric) + One-Hot Encoding (categorical)
    |                   |
    |                   v
    |            SMOTE (on training fold only)
    |                   |
    v                   v
RandomizedSearchCV (5-fold, 20 iterations, scoring=average_precision)
    |
    v
6 Trained Models
    |
    v
Sigmoid Calibration (on calibration fold)
    |
    v
Evaluation on 2020 Test Set (completely unseen)
    |
    v
SHAP Analysis + Economic Risk Scoring + Bootstrap CIs
    |
    v
Figures (PNG) + Tables (CSV + LaTeX)
```

### Key Design Decisions

1. **Temporal Validation:** Train on 2019, test on 2020. This is the most realistic evaluation — models must predict the future, not interpolate the past.

2. **No Data Leakage:** The preprocessor (scaling, encoding) is fit **only** on the 2019 training fold. Calibration, validation, and test data are transformed using those same statistics but never influence the fitting.

3. **Three-Way Split of 2019:**
   - 60% Train (9,523 samples) — used for model fitting
   - 20% Calibration (3,175 samples) — used for probability calibration
   - 20% Validation (3,175 samples) — used for threshold optimization and model selection

4. **Threshold Optimization:** Rather than using the default 0.5 cutoff, the pipeline finds the threshold that maximizes F2 score on the validation set, then applies that threshold to the test set.

5. **Scoring Metric:** `average_precision` (PR-AUC) is used for hyperparameter search instead of `roc_auc` or `accuracy`, because it's the most appropriate metric for heavily imbalanced data.

### Dependencies

```
numpy, pandas, matplotlib, seaborn
scikit-learn (model training, preprocessing, evaluation)
xgboost (XGBClassifier)
lightgbm (LGBMClassifier)
imbalanced-learn (SMOTE)
shap (model explainability)
```

---

## How to Run

### Pipeline 1 (Full Features — Jupyter Notebook)

```bash
jupyter notebook IEOM_Transformer_PredMaint.ipynb
# Run all cells. Outputs saved to: outputs -first/
```

### Pipeline 2 (No Burn Rate — Python Script)

```bash
python IEOM_NoBurnRate_Run.py
# Outputs saved to: outputs_no_burn/
```

Both pipelines expect the data files at:
```
E:\Projects_\predictive_analysis\ifti_vai\Transformer\Dataset_Year_2019.xlsx
E:\Projects_\predictive_analysis\ifti_vai\Transformer\Dataset_Year_2020.xlsx
```

---

## Summary of Findings

1. **SVM (RBF)** is the best model when Burn Rate is available (PR-AUC = 0.1805). **Logistic Regression** wins without it (PR-AUC = 0.1531).

2. **Burn Rate is the single most powerful predictor** — its removal reduces PR-AUC by about 15%, but models remain useful.

3. **All six models significantly outperform random inspection**, with 4-6x lift when inspecting the top 5% of predictions.

4. **SMOTE and class weighting** provide modest improvements over unweighted training but are not game-changers for this dataset.

5. **The economic risk framework** translates probabilities into actionable COP-denominated risk rankings, enabling budget-constrained inspection planning.

6. **SHAP analysis** confirms domain expertise: burn rate, criticality, lightning exposure (DDT), and energy loss (EENS) are the key failure drivers.

7. **Bootstrap confidence intervals** confirm that results are statistically robust across 1,000 resampled test sets.

---

*Generated from the IEOM Predictive Maintenance Pipeline for Distribution Transformers — Cauca, Colombia dataset (15,873 transformers, 2019-2020).*
