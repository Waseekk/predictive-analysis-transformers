# Predicting Transformer Failures in Cauca, Colombia: A Research Comparison

**IEOM Bangkok 2026 Conference Paper — Supporting Document**
**Last Updated:** February 12, 2026

---

## 1. The Big Picture

### What problem are we solving?

The Compania Energetica de Occidente operates 15,873 distribution transformers across the Cauca department in southwestern Colombia. These transformers step down high-voltage electricity (13.2 kV and 34.5 kV) to levels safe for homes and businesses. When a transformer fails — often called "burning" — it causes power outages, costs money to replace, and leaves communities without electricity for an average of 5 days (126 hours per incident).

In 2020 alone:
- **629 transformers failed** (a 3.96% failure rate)
- **Corrective maintenance cost 3.27 billion COP** (~750,000 USD)
  - Labor: 1.01 billion COP
  - Materials: 2.26 billion COP
- **82,458 hours of electricity were lost** across affected communities

### Why does prediction matter?

Currently, most transformers are maintained *after* they fail (corrective maintenance). This is the most expensive approach because:
1. Emergency repairs cost more than planned ones
2. Communities lose power for days while waiting for replacement
3. Failed transformers can damage surrounding equipment

If we can predict *which* transformers are most likely to fail next year, the utility can inspect and maintain them *before* they break. This is called **predictive maintenance**, and research suggests it can reduce maintenance costs by 18-25% and cut unplanned outages significantly (Vita et al., 2023).

### Who benefits?

- **The utility** saves money on emergency repairs and reduces liability
- **Communities in Cauca** experience fewer and shorter blackouts
- **The research community** gains a validated methodology applicable to other utilities worldwide

---

## 2. Data: Where It Comes From

### Source

All data comes from the operational records of Compania Energetica de Occidente, covering the Cauca department of Colombia. The dataset was published as an open research resource by Bravo Montenegro et al. (2021) in *Data in Brief*.

### Scope

- **15,873 distribution transformers** monitored continuously
- **Two years of data:** 2019 (used for training) and 2020 (used for testing)
- **Voltage levels:** 13.2 kV and 34.5 kV
- **Geography:** Mix of urban and rural areas across Cauca
- **Failure events:** 807 failures in 2019 (5.08%), 629 failures in 2020 (3.96%)

### How each variable is collected

The dataset contains 16 variables per transformer, collected from three sources:

**From utility operational records:**
- Transformer power rating (kVA), location (rural/urban), installation type, self-protection equipment, removable connectors, overhead network type, circuit position, low-voltage network length (km), number of customers served, customer type (residential/commercial/industrial), and energy not supplied during outages (EENS in kWh)

**From environmental monitoring (weather stations and ceraunic studies):**
- Average ground discharge density (lightning strikes per km² per year), maximum ground discharge density, and criticality score (a binary flag based on regional lightning risk studies)

**Derived from historical failure records:**
- Burn rate (number of past failures divided by observation period) — this captures how often a transformer has failed historically and is the single strongest predictor of future failure (Alvarez et al., 2022)

---

## 3. Variables Explained

### Original Dataset Variables

| Variable | Plain English | Type | Example |
|----------|---------------|------|---------|
| **Burn Rate** | How often this transformer has failed in the past (failures per year) | Continuous | 0.4 = failed roughly once every 2.5 years |
| **Location** | Where the transformer is installed | Binary | Rural or Urban |
| **Power (kVA)** | How much electricity it can handle | Continuous | 10, 15, 25, 37.5, 45, 50, 75, 112.5 kVA |
| **Self-protection** | Whether it has an internal low-voltage switch for safety | Binary | Yes or No |
| **Avg DDT** | Average lightning strike density in the area | Continuous | Strikes per km² per year |
| **Max DDT** | Peak lightning strike density in the area | Continuous | Strikes per km² per year |
| **Criticality** | Domain-expert risk score based on ceraunic studies | Binary | High-risk (1) or Not (0) |
| **Removable Connectors** | Whether it has medium-voltage removable connectors | Binary | Yes or No |
| **Customer Type** | What kind of customers it serves | Categorical | Residential (strata 1-6), Commercial, Industrial, Official |
| **Number of Users** | How many customers depend on this transformer | Integer | e.g., 15 households |
| **EENS** | Energy not supplied — revenue loss when it fails (kWh) | Continuous | Higher = more economic impact per failure |
| **Installation Type** | Physical mounting configuration | Categorical | Pole, cabin, H-structure, pad-mounted, tower, etc. |
| **Overhead Network** | Whether the connected network is aerial | Binary | Yes or No |
| **Circuit Tail** | Whether the transformer is at a terminal point or through point | Binary | Terminal or Through |
| **Km of LT Network** | Length of low-voltage network connected to it | Continuous | e.g., 2.3 km |

### Engineered Features (Created by the Pipeline)

| Feature | What it captures | Formula |
|---------|-----------------|---------|
| Users per km | Network density — how concentrated the load is | Number of Users / Km of LT Network |
| EENS per user | Impact per customer when failure occurs | EENS / Number of Users |
| Log(EENS) | Scaled version of energy losses (handles outliers) | Natural log of EENS |
| Log(Users) | Scaled version of user count | Natural log of Number of Users |
| Log(Km) | Scaled version of network length | Natural log of Km of LT Network |

---

## 4. Evolution of Our Approach

The project evolved through three major stages, each learning from the previous one's mistakes:

### Stage 1: The Colab Notebook (`Colab_Predictive_Maintenance.ipynb`)

**What it did:** Combined 2019 and 2020 data together, shuffled everything randomly, then split 80/20 for training and testing. Trained 6 models plus a stacking ensemble.

**What went wrong:**
- **Data leakage:** By mixing years before splitting, the model "sees" 2020 data during training — like a student who accidentally gets the exam answers while studying. Every reported result (e.g., 95.46% accuracy) is inflated and unreliable.
- **Wrong encoding:** Used LabelEncoder on categorical variables like "Rural" and "Urban," which creates a false mathematical relationship (Rural=1 > Urban=0) that misleads certain models.
- **Misleading metric:** Reported accuracy as the primary result. But since only ~4% of transformers fail, a model that simply says "nothing will fail" achieves 95.5% accuracy while catching zero actual failures.

**Lesson learned:** Temporal data must be split by time (train on the past, test on the future), not randomly.

### Stage 2: The Local Pipeline (`Transformer copy.ipynb`)

**What it fixed:**
- Proper temporal split (train on 2019, test on 2020)
- Correct categorical encoding (OneHotEncoder)
- PR-AUC as the primary metric (appropriate for rare events)
- Hyperparameter tuning with RandomizedSearchCV
- Probability calibration and F2-optimized thresholds
- 13 models tested comprehensively

**What went wrong:**
- **Dropped the two best predictors:** Set `use_expert_features=False`, removing Burn Rate and Criticality. This is like a doctor diagnosing heart disease while deliberately ignoring blood pressure and cholesterol.
- Best PR-AUC achieved: only 0.1520 (Logistic Regression)

**Lesson learned:** Sound methodology is necessary but not sufficient — you also need the right features.

### Stage 3: The IEOM Pipeline (`run_ieom.py`)

**What it fixed:**
- **Kept Burn Rate and Criticality** — restoring the strongest predictors
- Added SHAP explainability (showing *why* each prediction is made)
- Added bootstrap confidence intervals (quantifying uncertainty)
- Added economic risk analysis (Expected Loss = probability x consequence)
- Added SMOTE strategy comparison (systematic imbalance handling)
- Added 11 publication-quality figures
- Added dummy classifier baseline (proving accuracy is meaningless at 4% failure rate)

**Result:** Best PR-AUC = 0.1805 (SVM), a +18.8% improvement over Stage 2's best (0.1520). This is the version submitted for IEOM Bangkok 2026.

---

## 5. How the Pipeline Works (Step by Step)

For readers unfamiliar with machine learning, here is what the IEOM pipeline does:

1. **Load the data:** Read 2019 and 2020 transformer records from Excel files
2. **Engineer new features:** Calculate derived variables like users-per-km and log-transformed values
3. **Split the data:** Use 2019 as training data and 2020 as testing data (never mixing them)
4. **Handle missing values:** Fill gaps with the median (middle value) of each column
5. **Encode categories:** Convert text labels (like "Rural") into numeric format that models can process
6. **Handle class imbalance:** Since only ~4% of transformers fail, use SMOTE to create synthetic examples of failures so models don't just learn to predict "no failure" for everything
7. **Train 6 models:** Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, and Gradient Boosting — each learns different patterns in the data
8. **Tune hyperparameters:** Automatically find the best settings for each model using cross-validation
9. **Calibrate probabilities:** Adjust model outputs so that when a model says "20% chance of failure," it really does fail about 20% of the time
10. **Optimize the decision threshold:** Instead of the default 50% cutoff, find the threshold that maximizes the F2-score (which penalizes missed failures more heavily than false alarms)
11. **Evaluate on 2020 data:** Test all models on data they have never seen before
12. **Rank transformers by risk:** Calculate Expected Loss = P(failure) x Replacement Cost for every transformer
13. **Explain predictions:** Use SHAP to show which features drove each prediction

---

## 6. Actual Results

### Baseline: The Dummy Classifier

A model that predicts "no failure" for every transformer achieves:
- **Accuracy: 96.04%** (because 96% of transformers *don't* fail)
- **Failures caught: 0 out of 629**
- **PR-AUC: ~0.04** (the random baseline for a 4% event rate)

This is why accuracy is meaningless for this problem. Any useful model must beat PR-AUC = 0.04.

### IEOM Pipeline Results (WITH Burn Rate)

| Model | PR-AUC | ROC-AUC | F2 Score | Lift@5% | What it means |
|-------|--------|---------|----------|---------|---------------|
| **SVM (RBF)** | **0.1805** | 0.7373 | 0.3024 | 5.70x | If you inspect the top 5%, you find 5.7x more failures than random |
| Random Forest | 0.1776 | 0.7340 | 0.2965 | 5.31x | Close second |
| Logistic Regression | 0.1696 | 0.7253 | 0.2983 | **5.89x** | Best at targeted top-5% inspection |
| LightGBM | 0.1579 | 0.7187 | 0.2697 | 4.71x | Decent |
| Gradient Boosting | 0.1116 | 0.7177 | 0.2840 | 4.14x | Underperforms |
| XGBoost | 0.1040 | 0.7153 | 0.2582 | 3.95x | Underperforms |

**Best model: SVM (RBF)** with PR-AUC = 0.1805 — roughly **4.6x better than random guessing** (baseline 0.04).

### Ablation Study Results (WITHOUT Burn Rate)

| Model | PR-AUC | ROC-AUC | F2 Score | Lift@5% |
|-------|--------|---------|----------|---------|
| **Logistic Regression** | **0.1531** | 0.7268 | 0.2823 | 5.70x |
| SVM (RBF) | 0.1404 | 0.7254 | 0.2942 | 5.12x |
| XGBoost | 0.1329 | 0.7311 | 0.2902 | 4.90x |
| Random Forest | 0.1286 | 0.7332 | 0.3035 | 5.38x |
| LightGBM | 0.1143 | 0.7281 | 0.2737 | 4.49x |
| Gradient Boosting | 0.0927 | 0.7035 | 0.2668 | 3.53x |

Without Burn Rate, the best PR-AUC drops to 0.1531 — a **15.2% decrease** from the full model (0.1805).

### Bootstrap Confidence Intervals

To ensure results are statistically robust, we report 95% confidence intervals from bootstrap resampling:

**WITH Burn Rate:**
| Model | PR-AUC [95% CI] |
|-------|-----------------|
| SVM (RBF) | 0.1817 [0.1515, 0.2141] |
| Random Forest | 0.1792 [0.1494, 0.2096] |
| Logistic Regression | 0.1711 [0.1429, 0.2006] |

**WITHOUT Burn Rate:**
| Model | PR-AUC [95% CI] |
|-------|-----------------|
| Logistic Regression | 0.1544 [0.1306, 0.1791] |
| SVM (RBF) | 0.1421 [0.1190, 0.1668] |
| XGBoost | 0.1349 [0.1145, 0.1580] |

### Practical Impact: What Do These Numbers Mean?

At a **Lift@5% of 5.70x**, if the utility can only afford to inspect 5% of its transformers (~794 units):
- **Random inspection** would catch ~31 of the 629 actual failures
- **Model-guided inspection** would catch ~179 failures — nearly **6 times more**
- This means **148 additional failures prevented** per year compared to random selection

At **Lift@1% of 10.06x** (top ~159 transformers):
- **Random inspection** would catch ~6 failures
- **Model-guided inspection** would catch ~63 failures — **10 times more**

---

## 7. The Burn Rate Question

### What is Burn Rate?

Burn Rate = (Number of historical failures) / (Observation period). A transformer with a Burn Rate of 0.6 has failed roughly 3 times in 5 years. It is the single most predictive feature in the dataset (Alvarez et al., 2022).

### Is it circular reasoning?

A common concern: "Using past failures to predict future failures — isn't that circular?" The answer is **no**, for several reasons:

1. **Burn Rate captures physical degradation.** Transformers that have failed before often have underlying conditions (aging insulation, environmental stress, overloading) that make them prone to fail again. The rate encodes these latent physical states.

2. **Temporal separation prevents leakage.** The model trains on 2019 Burn Rate and predicts 2020 failures. It never sees 2020 failure data during training. This is standard temporal validation.

3. **It is analogous to established practices.** Credit scoring uses past default history to predict future defaults. Medical risk models use prior hospitalizations to predict future ones. In all cases, history is a legitimate predictor because it reflects underlying conditions.

4. **The ablation study validates this.** Without Burn Rate, the model still works (PR-AUC 0.1531 vs baseline 0.04) but loses 15-29% of its predictive power. This confirms Burn Rate adds genuine information that other features cannot fully replicate.

5. **Domain experts include it deliberately.** Alvarez et al. (2022) and Vita et al. (2023) both include Burn Rate as a core feature. It was designed by electrical engineers at Compania Energetica de Occidente who understand transformer degradation physics.

### Defense for the paper

If reviewers challenge Burn Rate's inclusion, the paper presents:
- The ablation study (Codebase 4) as empirical evidence of its contribution
- Temporal validation design as protection against leakage
- References to Alvarez (2022) and Vita (2023) who use the same feature
- A conceptual argument that Burn Rate is a proxy for latent physical condition

---

## 8. Economic Analysis

### Cost Methodology

Each transformer's replacement cost is estimated based on its power rating, using a base cost of **560,000 COP per failure** for a 15 kVA unit (derived from Alvarez et al., 2022), scaled proportionally by kVA:

| Power (kVA) | Estimated Replacement Cost (COP) |
|-------------|-----------------------------------|
| 15.0 | 560,000 |
| 25.0 | 933,333 |
| 37.5 | 1,400,000 |
| 45.0 | 1,680,000 |
| 50.0 | 1,866,667 |
| 75.0 | 2,800,000 |
| 112.5 | 2,800,000 (capped) |

### Context: The Scale of the Problem

From Alvarez et al. (2022):
- **3.27 billion COP** spent on corrective maintenance in 2020
  - 1.01 billion COP in labor
  - 2.26 billion COP in materials
- **82,458 hours** of electricity lost (average 126.47 hours = ~5 days per failure)
- **652 failures** recorded in 2020

### Expected Loss Framework

The IEOM pipeline computes **Expected Loss** for every transformer:

> **Expected Loss = P(failure) x Replacement Cost**

This prioritizes transformers that are both *likely to fail* AND *expensive to replace*. A small transformer with 70% failure probability (Expected Loss = 392,000 COP) ranks lower than a large transformer with 25% failure probability (Expected Loss = 700,000 COP).

### Top 10 Highest-Risk Transformers (IEOM Model)

| Rank | ID | Failure Prob. | Power (kVA) | Expected Loss (COP) |
|------|------|--------------|-------------|---------------------|
| 1 | 3713 | 26.6% | 75.0 | 744,581 |
| 2 | 11483 | 25.7% | 75.0 | 720,190 |
| 3 | 7758 | 25.6% | 112.5 | 716,577 |
| 4 | 7885 | 24.5% | 112.5 | 685,955 |
| 5 | 7871 | 23.3% | 112.5 | 652,041 |
| 6 | 11996 | 18.9% | 112.5 | 530,463 |
| 7 | 8510 | 52.3% | 25.0 | 487,780 |
| 8 | 5648 | 25.2% | 50.0 | 470,455 |
| 9 | 2636 | 16.4% | 112.5 | 459,952 |
| 10 | 2209 | 16.1% | 75.0 | 451,704 |

The top 10 represent a combined Expected Loss of **5.92 million COP**. Across all 15,873 transformers, the model enables systematic prioritization of inspection resources toward the highest economic risk.

### Comparison with Prior Work

| Study | Transformers Flagged | Method | Coverage |
|-------|---------------------|--------|----------|
| Alvarez et al. (2022) | 910 for 2021 inspection | SVM only | 870 rural, mostly 10 kVA |
| Vita et al. (2023) | 852 for 2021 inspection | k-means + SVM | 820 rural, 10 kVA most vulnerable |
| **This work (IEOM)** | Risk-ranked list of all 15,873 | 6 models, SVM best | All power classes, Expected Loss ordering |

Our approach differs from prior work in a key way: instead of a binary "inspect / don't inspect" list, we provide a **continuous risk ranking** weighted by economic consequence. This gives the utility flexibility to choose their inspection budget and see exactly how many failures they would catch at each level of effort.

---

## 9. Research Contributions for the IEOM Paper

The following contributions are **novel** relative to prior work on this dataset:

### 1. First SHAP Analysis on This Dataset
Neither Alvarez et al. (2022) nor Vita et al. (2023) provided feature-level explanations for individual predictions. Our SHAP analysis shows:
- Which features drive each transformer's risk score
- Global feature importance rankings (Burn Rate >> Location >> DDT)
- Non-linear interaction effects (e.g., high burn rate + high power = disproportionately higher risk)

### 2. Calibrated Risk Scoring with Expected Loss
Prior work predicts binary failure (yes/no). Our pipeline produces **calibrated probabilities** (e.g., "23% chance of failure"), which are then multiplied by replacement cost to yield Expected Loss. This bridges machine learning prediction with economic decision-making.

### 3. Comprehensive 6-Model Benchmark with Temporal Validation
Alvarez et al. (2022) used only SVM. Vita et al. (2023) compared SVM with k-means clustering. We benchmark 6 supervised models (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, Gradient Boosting) under identical conditions with temporal validation.

### 4. PR-AUC and F2 as Primary Metrics
Prior work reported accuracy (misleading at ~4% failure rate) or recall alone. We use PR-AUC (the standard for imbalanced classification) and F2-score (which weights recall 4x more than precision, appropriate when missed failures are costlier than false alarms).

### 5. Bootstrap Confidence Intervals
No prior work on this dataset provides uncertainty estimates. Our bootstrap CIs (e.g., SVM PR-AUC = 0.1817 [0.1515, 0.2141]) allow reviewers and practitioners to assess the reliability of reported performance.

### 6. Systematic SMOTE Strategy Comparison
First systematic comparison of three imbalance-handling strategies (SMOTE, class weights, no resampling) on this dataset. Finding: differences are marginal (<1% PR-AUC), suggesting that feature engineering matters more than resampling method.

### 7. Ablation Study on Burn Rate
First controlled experiment quantifying Burn Rate's contribution: +28.6% PR-AUC improvement for SVM (0.1404 → 0.1805). This justifies the feature's inclusion and provides evidence for its use in future studies.

---

## 10. References

1. **Bravo Montenegro, D., Rengifo Rengifo, H., Alvarez Torres, C.** (2021). Dataset for predicting the burning of distribution transformers. *Data in Brief*, 38, 107454. DOI: [10.1016/j.dib.2021.107454](https://doi.org/10.1016/j.dib.2021.107454)

2. **Alvarez Torres, C., Rengifo Rengifo, H., Bravo Montenegro, D.** (2022). Methodology for predictive maintenance of distribution transformers based on machine learning. *Ingenieria*, 27(2), e17742. DOI: [10.14483/23448393.17742](https://doi.org/10.14483/23448393.17742)

3. **Vita, V., Fotis, G., Pavlatos, C., Mladenov, V.** (2023). Predictive maintenance for distribution system operators in electricity smart grids. *Electronics*, 12(6), 1356. DOI: [10.3390/electronics12061356](https://doi.org/10.3390/electronics12061356)

4. **Transformer Predictive Maintenance Research Roadmap** (Internal document). 10-phase strategic framework for ML-driven predictive maintenance with economic evaluation.

---

## 11. Future Plans

Based on the Research Roadmap and directions identified by Vita et al. (2023):

### Near-Term (for extended paper or journal version)
- **Life-cycle costing with NPV/IRR analysis:** Quantify the net present value of deploying predictive maintenance over 5-10 years, including infrastructure investment costs
- **Budget-constrained optimization:** Solve the practical problem of "given M maintenance crews and budget B, which transformers should we inspect first?" using expected loss maximization subject to constraints
- **Multi-year sliding window validation:** Test model stability across multiple annual cycles (2019→2020, 2020→2021, etc.)

### Medium-Term (research extensions)
- **IoT integration for real-time monitoring:** Incorporate sensor data (oil temperature, dissolved gas analysis, load measurements) for continuous rather than annual prediction
- **Digital twin-assisted planning:** Build virtual replicas of transformer assets to simulate "maintain now vs. delay" decisions, potentially reducing unplanned outages by ~35%
- **Edge computing deployment:** Move inference to field devices for faster response times and lower outage duration

### Long-Term (vision)
- **Self-healing grid integration:** Combine failure prediction with automated switching to reduce restoration time by up to 60%
- **Transfer learning to other Colombian departments:** Adapt the trained model to Narino, Valle del Cauca, and other regions served by similar utilities
- **Climate change adaptation:** Incorporate projected changes in lightning patterns and temperature extremes as forward-looking features
