# 📖 EXPLAIN — What This Project Does (Step by Step)

> This document explains every part of the **Breast Cancer Prediction with Quantum SVM** project in **simple words**. It walks through the notebook code in the exact order it runs, tells you what each piece does, and why.

---

## 🔎 What Is This Project About?

We built a system that **predicts whether a breast tumor is benign (harmless) or malignant (cancerous)** using patient measurement data. We compared two approaches:

1. **Classical Machine Learning** — traditional algorithms like SVM, Random Forest, Logistic Regression, XGBoost, etc.
2. **Quantum Machine Learning** — a Quantum Support Vector Machine (QSVM) that uses quantum computing concepts to measure how similar two patients are.

The goal is to see if the quantum approach can match or beat the classical approach on this task.

> **Key takeaway from results:** The classical models won. The best classical model (Linear SVM) scored 97.6% F1, while the best quantum model (Z QSVM) scored 84.2% F1. This is expected — quantum advantage is not guaranteed on small, clean datasets.

---

## 🗂️ Project Files Overview

| File | What It Does |
|------|-------------|
| `wdbc_qsvm_notebook.ipynb` | The **main notebook** — all the code runs here, top to bottom |
| `helpers.py` | Two small helper functions used by the notebook |
| `requirements.txt` | List of Python packages needed to run this project |
| `README.md` | Overview and setup instructions |
| `report.md` | Short formal report with results |
| `results/` | Folder containing all saved charts, tables, and cached data |

### What's in `helpers.py`?

Just two simple functions:

1. **`classification_metrics(y_true, y_pred, y_score)`** — Takes the real labels and predicted labels, and calculates five scores: accuracy, precision, recall, F1, and ROC-AUC.
2. **`save_figure(figure, file_path)`** — Saves a chart as an image file, displays it in the notebook, and closes it to free memory.

---

## 🔢 Step-by-Step Code Walkthrough

Below is exactly what the notebook does, cell by cell, in the order it runs.

---

### 🛠️ SETUP — Importing Libraries & Setting Up

**What happens:**
- We import all the Python libraries we need: NumPy (math), Pandas (tables), Matplotlib & Seaborn (charts), Scikit-learn (classical ML), XGBoost, SHAP (explainability), and our custom `helpers.py`.
- We set a **random seed** (`RANDOM_SEED = 42`) so that every time we run the notebook, we get the same results (reproducibility).
- We create the `results/` and `results/kernels/` folders if they don't already exist.
- We print the versions of key packages used (so anyone can reproduce the exact same run).
- We set a clean plotting style using Seaborn.

**Output:** Prints package versions (e.g., scikit-learn 1.8.0, numpy 2.3.3, etc.)  
**Saved file:** `results/package_versions.csv`

---

### 📊 STEP 1 — Load and Inspect the Data

**What happens:**
- We try to download the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset from the UCI online repository.
- If the download fails (no internet), we automatically use the **same dataset bundled with scikit-learn**.
- We clean up the data: remove any ID column (it doesn't describe the tumor), and set the labels so that **Malignant = 1** and **Benign = 0**.
- We print basic info: the shape of the data (569 patients × 30 features), how many missing values (0), and the class counts (357 benign, 212 malignant).

**In simple words:** We loaded data about 569 breast cancer patients. Each patient has 30 numeric measurements of their tumor (like size, shape, texture). The label tells us if the tumor was benign or malignant.

**Output:** Prints data shape, missing values, and class counts  
**Saved files:** `results/data_source.txt`, `results/data_summary.csv`

---

### 📈 STEP 1 (continued) — Visualizing the Data

**What happens:**
- **Class Distribution Bar Chart** — Shows how many benign vs. malignant patients we have. This tells us if the dataset is balanced.
- **Correlation Heatmap** — A color grid showing how much each of the 30 features is related to every other feature. Bright/dark colors mean strong relationships. This helps us see that many features measure similar things (redundancy).
- **Box Plots** — For 4 example features, shows the range of values for benign vs. malignant tumors. This helps us see which features differ between the two classes.

**In simple words:** We drew charts to understand the data before building any models. We saw that many features are correlated (related), and some features clearly differ between benign and malignant tumors.

**Saved files:** `results/01_class_distribution.png`, `results/02_correlation_heatmap.png`, `results/03_feature_boxplots.png`

---

### ✂️ STEP 2 — Split the Data into Training and Testing Sets

**What happens:**
- We split the 569 patients into two groups:
  - **Training set (80%)** — 455 patients used to teach the models.
  - **Test set (20%)** — 114 patients held back to check how well the models learned.
- The split is **stratified**: both groups keep roughly the same ratio of benign to malignant patients.
- We print and verify the sizes and class balance of both groups.

**In simple words:** We kept 114 patients aside as a "test exam." The models never see these patients during training. This way we can fairly judge how well they'll work on new, unseen patients.

---

### ⚖️ STEP 3 — Scale the Features (Standardization)

**What happens:**
- We use **StandardScaler** to normalize the features: each feature is shifted and scaled so its values have a mean of 0 and a standard deviation of 1.
- **Important:** The scaler is fitted (learns the mean/std) only from the training set. The same learned transformation is then applied to the test set. This prevents **data leakage** — the test data doesn't influence the scaling.

**In simple words:** The 30 features have very different ranges (some go 0–1, others go 0–4000). Scaling puts them all on the same playing field, which makes the models work better.

---

### 🔬 STEP 4 — Reduce Features Using PCA

**What happens:**
- **PCA (Principal Component Analysis)** compresses the 30 features into just **4 components**. These 4 components capture the most important patterns from all 30 features.
- We plot how much of the total information each component captures (explained variance). The first 4 components capture a good chunk.
- PCA is needed because quantum simulators are **slow** — simulating more qubits takes exponentially longer. 4 features = 4 qubits, which is manageable.
- We also prepare 6-component and 8-component versions for a side experiment later.

**In simple words:** We condensed the 30 measurements into 4 "super-measurements" that capture the most important information. This is necessary because the quantum computer simulator can only handle a small number of features efficiently.

**Saved files:** `results/04_pca_explained_variance.png`, `results/pca_loadings_top_features.csv`

---

### 📐 STEP 4 (continued) — Scale for Quantum Circuits

**What happens:**
- For the quantum model, we map the PCA-reduced features to **angles between 0 and π** (about 0 to 3.14).
- This is done using `MinMaxScaler` with range `[0, π]`.
- These angles are used as rotation amounts in the quantum circuits.

**In simple words:** Quantum circuits use rotations (angles) to encode data. So we convert the patient measurements into angles that the quantum circuit can use.

---

### 🤖 STEP 5 — Build and Train Classical Models

**What happens:**
- We build 7 classical machine learning models:
  1. **Logistic Regression** — Draws a straight boundary to separate classes.
  2. **Decision Tree** — Makes yes/no decisions based on feature thresholds.
  3. **Random Forest** — Combines many decision trees and takes a vote.
  4. **Naive Bayes** — Uses probability theory assuming features are independent.
  5. **XGBoost** — Builds decision trees one after another, each correcting the previous one's mistakes.
  6. **SVM (Linear)** — Finds the best flat boundary with maximum margin.
  7. **SVM (RBF)** — Like linear SVM but can learn curved boundaries.

- Each model is tuned using **GridSearchCV with 5-fold cross-validation**: we try different settings and pick the best one.
- Each model is tested in two versions:
  - Using **all 30 original features**
  - Using **4 PCA-reduced features** (same as what the QSVM will use)

**In simple words:** We trained 7 different types of AI models, each in two configurations, tuned their settings to find the best version, and measured how well each one predicts cancer. This gives us a fair baseline to compare against the quantum model.

**Saved file:** `results/classical_results.csv`

---

### ⚛️ STEP 6 — Build Quantum Feature Maps (Circuits)

**What happens:**
- We import the **Qiskit** quantum computing library.
- We build two types of quantum circuits (**feature maps**):
  1. **ZFeatureMap** — Each qubit gets a rotation based on one feature. No connection between qubits.
  2. **ZZFeatureMap** — Same as Z, but also adds **entangling gates** that create connections between pairs of qubits, capturing relationships between features.
- Both use **4 qubits** (one per PCA component) and **2 repetitions** (the circuit pattern is applied twice for more expressiveness).
- We draw pictures of both circuits so you can see the gates.

**In simple words:** We designed two quantum circuits that convert patient data into quantum states. The Z circuit treats each measurement independently. The ZZ circuit also looks at relationships between pairs of measurements. Think of it like: Z looks at each feature alone, ZZ looks at features together.

**Saved files:** `results/05_z_feature_map_circuit.png`, `results/06_zz_feature_map_circuit.png`

---

### 🧮 STEP 7 — Compute Quantum Kernels

**What happens:**
- This is the **most computationally expensive step**.
- For each feature map (Z and ZZ), we compute a **kernel matrix** — a big table of similarity scores.
- Each entry says: "How similar is Patient A to Patient B according to the quantum circuit?"
  - A value of **1** means their quantum states are identical.
  - A value near **0** means their quantum states are very different.
- We compute two kernel matrices per feature map:
  - **Train-vs-Train** — Similarity between all pairs of training patients.
  - **Test-vs-Train** — Similarity of each test patient to every training patient.
- Because simulating quantum circuits is slow, we only use **50 training patients** (out of 455) for the main QSVM. Classical models use all 455.
- The kernel matrices are **cached** (saved to disk) so we don't have to recompute them if we run the notebook again.
- We **validate** each kernel matrix: it should be symmetric, have 1s on the diagonal, and all values should be between 0 and 1.

**In simple words:** We used the quantum circuits to measure how similar each pair of patients is. This creates a "similarity table" that the SVM can use instead of looking at the raw features. Computing this table is slow (~55–74 seconds), which is why we only use 50 training patients for the quantum model.

**Saved files:** `results/kernels/*.npz` (cached kernel matrices)

---

### 📉 STEP 7 (continued) — Visualize Kernels

**What happens:**
- We draw **heatmaps** of the training kernel matrices.
- A bright diagonal is expected (each patient is perfectly similar to itself = 1).
- We can visually see how "spread out" the similarity values are.

**Saved files:** `results/qsvm_zfeaturemap_kernel_heatmap.png`, `results/qsvm_zzfeaturemap_kernel_heatmap.png`

---

### 🎯 STEP 8 — Train the QSVM

**What happens:**
- We take the pre-computed quantum kernel (the similarity table) and feed it to a **regular SVM** with `kernel="precomputed"`.
- The SVM learns a boundary in the "quantum similarity space" to separate benign from malignant patients.
- We tune the **C parameter** (which controls how strictly the boundary is enforced) using cross-validation over values `[0.1, 1, 10, 100]`.
- The best model is then evaluated on the 114 test patients.
- We calculate the same 5 metrics: accuracy, precision, recall, F1, and ROC-AUC.

**In simple words:** We plugged the quantum similarity table into a standard SVM classifier. The SVM uses this table to decide which patients are benign and which are malignant. We picked the best settings and measured how well it works.

**Saved files:** `results/main_qsvm_results.csv`

---

### 📋 STEP 9a — Confusion Matrices

**What happens:**
- For each QSVM (Z and ZZ), we draw a **confusion matrix** — a 2×2 grid showing:
  - ✅ Correct benign predictions (top-left)
  - ✅ Correct malignant predictions (bottom-right)
  - ❌ Benign predicted as malignant — false positive (top-right)
  - ❌ **Malignant predicted as benign — false negative (bottom-left)** ← This is the most dangerous error in medical diagnosis

**In simple words:** The confusion matrix tells us exactly how many patients the model got right and wrong, and what types of mistakes it made. Missing a malignant case (false negative) is the most critical error.

**Saved files:** `results/qsvm_zfeaturemap_confusion_matrix.png`, `results/qsvm_zzfeaturemap_confusion_matrix.png`

---

### 📋 STEP 9b — Compare All Models

**What happens:**
- We combine the results from all classical models and both QSVMs into one big comparison table.
- We sort the table by malignant recall (how many malignant cases the model catches), then by F1 score.
- We create bar charts comparing the F1 scores of all models.

**In simple words:** We put all models side by side to see which one is the best at catching cancer. The Linear SVM on all 30 features was the winner.

**Saved files:** `results/final_comparison.csv`, `results/final_model_comparison.png`, `results/experiment_all_models.png`

---

### 📋 STEP 9c — ROC Curves

**What happens:**
- We draw **ROC curves** for the main models (the classical finalists + both QSVMs).
- An ROC curve shows the trade-off between catching malignant cases (recall) and falsely flagging benign cases (false positive rate) as you change the decision threshold.
- A curve closer to the **top-left corner** is better. The area under the curve (AUC) summarizes this — closer to 1.0 is better.

**In simple words:** ROC curves show how well a model can rank malignant patients above benign patients across all possible thresholds. A perfect model would have AUC = 1.0.

**Saved file:** `results/roc_curves_all_main_models.png`

---

### 📋 STEP 9d — SHAP Explainability (XGBoost)

**What happens:**
- We use **SHAP (SHapley Additive exPlanations)** on the XGBoost model to understand **which features matter most** for predictions.
- SHAP assigns each feature a contribution score for each patient's prediction.
- We create a summary plot showing the top features and save the importance ranking.

**In simple words:** SHAP tells us "why" XGBoost made each prediction. For example, it found that `worst concave points`, `mean concave points`, and `worst perimeter` are the most important features. These are model explanations, not medical advice.

**Saved files:** `results/shap_xgboost_summary.png`, `results/shap_feature_importance.csv`

---

### 📋 STEP 9e — Feature Map Comparison Experiment

**What happens:**
- We compare the two quantum feature maps (Z vs. ZZ) head-to-head.
- We also compare both QSVMs against the RBF SVM on the same 4 PCA features.
- Bar charts show the differences.

**In simple words:** We checked whether the Z or ZZ quantum circuit works better, and how they compare to a classical model using the same reduced features.

**Saved files:** `results/experiment_feature_maps.csv`, `results/experiment_feature_maps.png`, `results/experiment_qsvm_vs_rbf.csv`, `results/experiment_qsvm_vs_rbf.png`

---

### 📋 STEP 9f — Repetition Experiment

**What happens:**
- We test the ZZ feature map with **1, 2, and 3 repetitions** (how many times the circuit pattern is repeated).
- More repetitions make the circuit deeper and more expressive, but also more complex.
- We compare the results.

**Result:** More repetitions did NOT automatically improve the QSVM in this experiment. F1 went from 0.762 (1 rep) → 0.583 (2 reps) → 0.556 (3 reps).

**Saved files:** `results/experiment_repetitions.csv`, `results/experiment_repetitions.png`

---

### 📋 STEP 9g — Entanglement Experiment

**What happens:**
- We test the ZZ feature map with different **entanglement patterns**: `linear` (each qubit connects to the next), `full` (every qubit connects to every other).
- This tests whether more entanglement (more connections between qubits) helps.

**Saved files:** `results/experiment_entanglement.csv`, `results/experiment_entanglement.png`

---

### 📋 STEP 9h — Qubit Count Experiment

**What happens:**
- We test the ZZ feature map with **4, 6, and 8 qubits** (which means 4, 6, or 8 PCA components).
- Each uses only 25 training patients to keep the comparison fair.
- More qubits = more features used = more complex quantum simulation.

**Result:** 6 and 8 qubits predicted no malignant patients at all (F1 = 0). More qubits hurt performance in this small experiment because of the tiny training set.

**Saved files:** `results/experiment_qubits.csv`, `results/experiment_qubits.png`

---

### 📋 STEP 9i — Misclassified Patients Analysis

**What happens:**
- We look at the specific test patients that the Z QSVM got **wrong**.
- For each misclassified patient, we save:
  - Their original features
  - Their true label
  - The predicted label
  - The SVM decision score (how confident the model was — scores close to 0 mean the model was very uncertain)

**In simple words:** We examined the mistakes to understand what went wrong. Patients with decision scores close to 0 were right on the boundary — the model was unsure about them.

**Saved file:** `results/qsvm_misclassified_test_patients.csv`

---

### 📋 STEP 9j — Stability Test (5 Random Seeds)

**What happens:**
- We repeat the entire comparison using **5 different random seeds** (7, 21, 84, 123, and the original 42).
- Each seed creates a different train/test split.
- We compute the **mean and standard deviation** of each metric across all 5 runs.
- This tells us how consistent and reliable the results are.

**Results across 5 splits:**

| Model | F1 (mean ± std) |
|-------|-----------------|
| Linear SVM (all 30 features) | 0.959 ± 0.021 |
| RBF SVM (PCA 4) | 0.925 ± 0.016 |
| QSVM Z | 0.870 ± 0.032 |
| QSVM ZZ | 0.523 ± 0.047 |

**In simple words:** Running the experiment multiple times confirmed that the classical models are consistently better. The Z QSVM is decent but trails behind, and the ZZ QSVM is unreliable on this dataset.

**Saved files:** `results/stability_by_seed.csv`, `results/stability_summary.csv`, `results/stability_f1_mean_std.png`

---

## 🧠 Key Concepts Explained Simply

### What is an SVM?
A **Support Vector Machine (SVM)** is like drawing a line (or a plane) between two groups of points. It finds the line that has the **biggest gap (margin)** between the groups. Points closest to the line are called "support vectors."

### What is a Kernel?
A **kernel** is a shortcut for measuring how similar two data points are. Instead of transforming data into a complex space, the kernel directly computes a similarity score. Example: the **RBF kernel** measures similarity based on distance between points.

### What is a Quantum Kernel?
A **quantum kernel** uses a quantum circuit to compute similarity. Each patient's features are encoded as rotations on qubits. The similarity between two patients is the squared overlap of their quantum states: a value near 1 means similar, near 0 means different.

### What is PCA?
**Principal Component Analysis (PCA)** is a technique that compresses many features into a smaller set of "principal components." Each component is a combination of the original features that captures as much variation in the data as possible.

### What is SHAP?
**SHAP** explains AI predictions by assigning each feature a contribution score. It tells you how much each feature pushed the prediction toward benign or malignant.

---

## 📊 Final Results Summary

| Rank | Model | F1 Score | Malignant Recall |
|------|-------|----------|-----------------|
| 1 | SVM (Linear) — all 30 features | 0.976 | 95.2% |
| 2 | Random Forest — all 30 features | 0.963 | 92.9% |
| 3 | SVM (RBF) — all 30 features | 0.963 | 92.9% |
| ... | ... | ... | ... |
| 15 | **QSVM Z** — 4 PCA, 50 train patients | **0.842** | **76.2%** |
| 16 | **QSVM ZZ** — 4 PCA, 50 train patients | **0.583** | **50.0%** |

> **Bottom line:** Classical models are faster, more accurate, and more reliable on this dataset. The quantum models are a useful learning demonstration, but they do NOT show quantum advantage here.

---

## ⚠️ Limitations

- All quantum results are from **simulation**, not real quantum hardware.
- The QSVM used only **50 training patients** (vs. 455 for classical models) because quantum simulation is slow.
- WDBC is a **small, clean dataset** that classical models handle easily.
- This is a **learning project**, not a medical diagnostic tool.
- No noise simulation was performed.

---

## 📚 References

1. Islam et al. (2024). *Predictive Modeling for Breast Cancer Classification* — [arXiv:2404.04686](https://arxiv.org/abs/2404.04686)
2. [UCI WDBC Dataset (ID 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
3. [Qiskit Machine Learning Docs](https://qiskit-community.github.io/qiskit-machine-learning/)
4. Havlíček et al. (2019). *Supervised learning with quantum-enhanced feature spaces* — [Nature](https://doi.org/10.1038/s41586-019-0980-2)

