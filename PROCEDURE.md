# 📋 PROCEDURE.MD — Complete Code Sequence & Execution Guide

> **Project:** Breast Cancer Prediction with a Quantum-Kernel Support Vector Machine (QSVM)  
> **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
> **Main File:** `wdbc_qsvm_notebook.ipynb`  
> **Purpose of this document:** A step-by-step procedure guide explaining the **exact sequence of code execution**, cell by cell, what each code block does, inputs/outputs, and how to explain it in simple words for presentations or viva.

---

## 🗺️ High-Level Execution Pipeline Flowchart

```
[Cell 2: Setup & Environment]
       │
       ▼
[Cell 4: Data Ingestion (UCI / Scikit-learn fallback)]
       │
       ▼
[Cell 6: Exploratory Data Analysis & Plots]
       │
       ▼
[Cell 8: Stratified Train/Test Split (80/20)]
       │
       ▼
[Cell 10: Standard Scaling & PCA Dimensionality Reduction]
       │
       ▼
[Cell 12: 7 Classical ML Baselines (GridSearchCV 5-Fold CV)]
       │
       ▼
[Cell 14: Quantum Feature Maps (Z & ZZ) & Kernel Computation]
       │
       ▼
[Cell 16: QSVM Training (Precomputed Kernel SVC)]
       │
       ▼
[Cell 18: Quantum Sensitivity Experiments (Reps, Entanglement, Qubits)]
       │
       ▼
[Cell 20: 5-Fold Cross-Seed Stability Verification]
       │
       ▼
[Cell 22: Model Explainability (SHAP on XGBoost & QSVM Misclassifications)]
       │
       ▼
[Cell 24: Final Synthesis & Performance Benchmarking]
```

---

## 🔍 Detailed Cell-by-Cell Code Sequence

---

### 🔹 Cell 2: Setup, Dependencies & Reproducibility Setup
- **Notebook Section:** `## Setup`
- **Key Code Functions:** `np.random.seed(42)`, `Path.cwd()`, `mkdir()`

#### What the code does:
1. Imports core data science libraries (`numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`, `xgboost`, `shap`).
2. Imports custom helper functions from `helpers.py`:
   - `classification_metrics`: Standardized evaluation (Accuracy, Precision, Recall, F1, ROC-AUC).
   - `save_figure`: Saves charts to disk at 180 DPI and displays them inline.
3. Sets a fixed global random seed `RANDOM_SEED = 42` to guarantee complete reproducibility across all splits and stochastic models.
4. Creates output directories: `results/` and `results/kernels/`.
5. Prints installed library versions to document the exact software environment.

- **Outputs:**
  - Standard output of package versions.
  - Directories `results/` and `results/kernels/` created.

---

### 🔹 Cell 4: Data Ingestion, Cleaning & Label Formatting
- **Notebook Section:** `## Step 1: Load and inspect the data`
- **Key Code Functions:** `ucimlrepo.fetch_ucirepo(id=17)` with `try-except` fallback to `sklearn.datasets.load_breast_cancer()`

#### What the code does:
1. Attempts to download the WDBC dataset directly from the UCI Machine Learning Repository (Dataset ID 17).
2. If internet connection is unavailable or UCI is down, gracefully catches the exception and loads Scikit-learn’s built-in 569-patient WDBC dataset.
3. Drops non-predictive identifiers (such as `id` or patient numbers).
4. Standardizes the binary diagnosis label:
   - **`1` = Malignant (Cancerous)** — set as the positive class.
   - **`0` = Benign (Non-cancerous)** — set as the negative class.
5. Verifies data integrity: checks for `NaN` / missing values (confirmed `0` missing).
6. Assigns standard anatomical feature names (30 features total: mean, standard error, and "worst" of 10 cellular characteristics).

- **Outputs:**
  - Console verification: 569 rows, 30 columns, 0 missing values (357 Benign, 212 Malignant).
  - Saved file: `results/data_source.txt`.

---

### 🔹 Cell 6: Exploratory Data Analysis (EDA) & Visualization
- **Notebook Section:** Visualizing Class Distribution and Feature Correlation
- **Key Code Functions:** `plt.subplots()`, `sns.heatmap()`, `sns.boxplot()`

#### What the code does:
1. **Class Distribution Chart:** Plots a bar chart showing the balance of Benign (357) vs. Malignant (212) cases.
2. **Correlation Heatmap:** Computes the Pearson correlation matrix of all 30 features and renders a 15x12 heatmap. It reveals severe multicollinearity among size/area/perimeter features.
3. **Boxplots:** Visualizes the distribution of four critical features (`mean radius`, `mean texture`, `mean smoothness`, `mean concave points`) partitioned by diagnosis.

- **Outputs:**
  - `results/01_class_distribution.png`
  - `results/02_correlation_heatmap.png`
  - `results/03_feature_boxplots.png`

---

### 🔹 Cell 8: Stratified Splitting & Leakage-Free Standardization
- **Notebook Section:** `## Step 2: Split and scale without leakage`
- **Key Code Functions:** `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`

#### What the code does:
1. Divides the 569 patients into:
   - **Training Set (80%):** 455 patients (285 Benign, 170 Malignant).
   - **Test Set (20%):** 114 patients (72 Benign, 42 Malignant).
2. Uses **stratification** so that both splits preserve the identical ~62.7% / 37.3% ratio of benign to malignant cases.
3. Saves the original patient row indices (`train_indices`, `test_indices`) so any individual test case can be traced back.
4. Fits `StandardScaler()` **strictly on `X_train`**, and then transforms `X_test` using those exact training statistics (mean and variance). This prevents data leakage from the test set into model training.

- **Outputs:**
  - Data structures: `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`.
  - Saved file: `results/data_summary.csv`.

---

### 🔹 Cell 10: Dimensionality Reduction via PCA & Angle Mapping
- **Notebook Section:** `## Step 3: Reduce features for the simulator`
- **Key Code Functions:** `PCA(n_components=4)`, `MinMaxScaler(feature_range=(0, np.pi))`

#### What the code does:
1. Runs Principal Component Analysis (PCA) on the standardized training data to compress 30 correlated features down to **4 orthogonal principal components**.
2. Plots the cumulative explained variance curve across all 30 components.
3. Generates a table of top feature loadings (weights) showing which original cell measurements contribute most to PC1, PC2, PC3, and PC4.
4. **Quantum Encoding Preparation:** Quantum phase rotation gates accept angles. A `MinMaxScaler` maps the 4 PCA components into the interval $[0, \pi]$:
   - `X_train_angle = minmax.fit_transform(X_train_pca4)`
   - `X_test_angle = minmax.transform(X_test_pca4)`
5. Prepares 6-component and 8-component datasets for subsequent width-scaling experiments.

- **Outputs:**
  - `results/04_pca_explained_variance.png`
  - `results/pca_loadings_top_features.csv`
  - Variables: `X_train_angle`, `X_test_angle`.

---

### 🔹 Cell 12: Training & Hyperparameter Tuning of 7 Classical Baselines
- **Notebook Section:** `## Step 4: Classical baselines`
- **Key Code Functions:** `GridSearchCV(..., cv=StratifiedKFold(5), scoring="f1")`, `Pipeline`

#### What the code does:
1. Defines 7 classical machine learning classifiers with hyperparameter search grids:
   - **Logistic Regression:** $C \in [0.1, 1, 10]$
   - **Decision Tree:** $\text{max\_depth} \in [\text{None}, 3, 6]$
   - **Random Forest:** 100 trees, $\text{max\_depth} \in [\text{None}, 8]$
   - **Gaussian Naive Bayes:** $\text{var\_smoothing} \in [10^{-9}, 10^{-8}, 10^{-7}]$
   - **XGBoost:** 80 estimators, $\text{max\_depth} \in [2, 3]$, $\text{learning\_rate} \in [0.05, 0.1]$
   - **Support Vector Classifier (Linear):** $C \in [0.1, 1, 10]$
   - **Support Vector Classifier (RBF):** $C \in [0.1, 1, 10]$
2. Evaluates each classifier under **two distinct regimes**:
   - **Regime A:** Using all 30 scaled features.
   - **Regime B:** Using the 4-component PCA pipeline (matching the quantum simulator's input dimension).
3. Evaluates all 14 configurations on the held-out 114 test patients and records Accuracy, Precision, Recall, F1, ROC-AUC, and execution time.

- **Outputs:**
  - Saved file: `results/classical_results.csv` (14 rows comparing model performance).

---

### 🔹 Cell 14: Quantum Feature Map Construction & Kernel Matrix Simulation
- **Notebook Section:** `## Step 5: Quantum feature maps and kernels`
- **Key Code Functions:** `ZFeatureMap`, `ZZFeatureMap`, `StatevectorSampler`, `ComputeUncompute`, `FidelityQuantumKernel`

#### What the code does:
1. Defines two quantum circuits using Qiskit Machine Learning:
   - **`ZFeatureMap`:** 4 qubits, 2 repetitions. Features are encoded as independent single-qubit phase rotations ($R_z$ gates) with Hadamard gates. No qubit entanglement.
   - **`ZZFeatureMap`:** 4 qubits, 2 repetitions, linear entanglement. Encodes both single-qubit rotations and pairwise two-qubit interactions ($R_{zz}$ gates implemented via CNOTs).
2. Draws and saves circuit diagrams of both feature maps.
3. Subsamples **50 training patients** (stratified) for quantum kernel evaluation to balance simulation time with statistical representation.
4. Evaluates the **Quantum Kernel Matrix** using statevector simulation:
   - Train kernel: $K_{\text{train}}[i, j] = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$ ($50 \times 50$ matrix).
   - Test kernel: $K_{\text{test}}[i, j] = |\langle \phi(x_{\text{test}, i}) | \phi(x_{\text{train}, j}) \rangle|^2$ ($114 \times 50$ matrix).
5. Enforces mathematical validation: checks that $K_{\text{train}}$ is symmetric, has unit diagonal ($K_{ii} = 1.0$), and values are bounded in $[0, 1]$.
6. Saves matrices to `.npz` files for instant reloading.
7. Plots heatmaps of the computed quantum similarity kernels.

- **Outputs:**
  - Circuit images: `results/05_z_feature_map_circuit.png`, `results/06_zz_feature_map_circuit.png`
  - Kernel caches: `results/kernels/Z_4q_reps2_none_seed42.npz`, `results/kernels/ZZ_4q_reps2_linear_seed42.npz`
  - Heatmaps: `results/qsvm_zfeaturemap_kernel_heatmap.png`, `results/qsvm_zzfeaturemap_kernel_heatmap.png`

---

### 🔹 Cell 16: Quantum Support Vector Classifier (QSVM) Fitting
- **Notebook Section:** `## Step 6: Train and test the QSVM`
- **Key Code Functions:** `SVC(kernel="precomputed")`, `GridSearchCV`

#### What the code does:
1. Passes the precomputed quantum kernel matrices into an SVM classifier.
2. Performs 5-fold cross-validation on the $50 \times 50$ training kernel to optimize regularization parameter $C \in [0.1, 1, 10, 100]$.
3. Fits the optimal SVM model on the training kernel and predicts on the $114 \times 50$ test kernel.
4. Generates confusion matrices for both Z-QSVM and ZZ-QSVM.

- **Outputs:**
  - `results/main_qsvm_results.csv`
  - `results/qsvm_zfeaturemap_confusion_matrix.png`
  - `results/qsvm_zzfeaturemap_confusion_matrix.png`

---

### 🔹 Cell 18: Quantum Sensitivity & Ablation Experiments
- **Notebook Section:** `## Step 7: Small quantum and model comparisons`
- **Key Code Functions:** Iterative test runner for feature map variations

#### What the code does:
Tests the impact of different quantum circuit architectures:
1. **Repetitions Experiment:** Evaluates ZZFeatureMap with 1, 2, and 3 repetitions.
   - *Result:* Repetition 1: F1 = 0.762; Repetition 2: F1 = 0.583; Repetition 3: F1 = 0.556. Deeper circuits degraded accuracy due to over-rotation.
2. **Entanglement Experiment:** Compares `linear` entanglement vs. `full` all-to-all entanglement.
3. **Qubit Scaling Experiment:** Evaluates 4, 6, and 8 qubits (PCA components) with 25 training patients.
   - *Result:* 6 and 8 qubits collapsed to predicting 0 malignant cases because the Hilbert space grew too large for the small training sample.

- **Outputs:**
  - `results/experiment_repetitions.csv` & `.png`
  - `results/experiment_entanglement.csv` & `.png`
  - `results/experiment_qubits.csv` & `.png`
  - `results/experiment_feature_maps.csv` & `.png`
  - `results/experiment_qsvm_vs_rbf.csv` & `.png`

---

### 🔹 Cell 20: 5-Fold Random Split Cross-Validation Stability Test
- **Notebook Section:** `### Stability across five random splits`
- **Key Code Functions:** Multi-seed evaluation loop over `seeds = [42, 7, 21, 84, 123]`

#### What the code does:
1. Re-splits and re-runs the four key models across 5 different random splits:
   - Linear SVM (30 features)
   - RBF SVM (4 PCA components)
   - QSVM (ZFeatureMap)
   - QSVM (ZZFeatureMap)
2. Computes the **Mean and Standard Deviation** for Accuracy, Recall, F1, and ROC-AUC.
3. Confirms that classical superiority is statistically robust across data splits, not an artifact of a single lucky split.

- **Outputs:**
  - `results/stability_by_seed.csv`
  - `results/stability_summary.csv`
  - `results/stability_f1_mean_std.png`

---

### 🔹 Cell 22: Explainable AI (XAI) & Error Analysis
- **Notebook Section:** `## Step 8: Explainability and difficult test cases`
- **Key Code Functions:** `shap.TreeExplainer(xgboost_model)`, `decision_function()`

#### What the code does:
1. **SHAP Analysis (XGBoost):** Calculates Shapley additive values on the test set to determine which of the 30 cellular features most heavily push predictions towards malignant or benign.
2. **Error Audit:** Extracts the 12 test patients misclassified by Z-QSVM. Calculates their signed decision boundary margin scores ($f(x) = \text{sign}(\sum \alpha_i y_i K(x_i, x) + b)$) to identify whether errors were near the margin or severe anomalies.

- **Outputs:**
  - `results/shap_xgboost_summary.png`
  - `results/shap_feature_importance.csv`
  - `results/qsvm_misclassified_test_patients.csv`

---

### 🔹 Cell 24: Final Benchmarking Synthesis & ROC Curve Generation
- **Notebook Section:** `## Step 9: Final comparison`
- **Key Code Functions:** `RocCurveDisplay.from_predictions()`, bar plot consolidation

#### What the code does:
1. Combines all 16 model runs (14 classical + 2 QSVM) into a master table sorted by Malignant Recall and F1 score.
2. Plots comparative bar charts of F1 scores across all models.
3. Overlays ROC curves for all primary competitors on a single plot.

- **Outputs:**
  - `results/final_comparison.csv`
  - `results/final_model_comparison.png`
  - `results/roc_curves_all_main_models.png`
  - `results/experiment_all_models.png`

---

## 🗣️ How to Present This Procedure (Viva / Presentation Talking Points)

When presenting what was done in this project, explain the work in this 5-point sequence:

1. **Problem Definition & Ingestion:**
   > *"We took 569 patient biopsy samples from the WDBC dataset with 30 cell nucleus measurements. Our goal is binary classification: predicting malignant vs. benign, prioritizing malignant recall because false negatives can be fatal."*

2. **Preprocessing Without Data Leakage:**
   > *"We held out 20% of the data (114 patients) in a stratified split. We fitted standard scaling and 4-component PCA strictly on the training partition. For the quantum circuits, we mapped the 4 PCA components to rotation angles between $0$ and $\pi$."*

3. **Classical Benchmark Setup:**
   > *"We trained 7 diverse classical algorithms using 5-fold cross-validation grid search on both all 30 features and the 4 PCA components. The Linear SVM on all 30 features achieved the best performance with an F1 score of 97.6% and a recall of 95.2%."*

4. **Quantum Kernel Engineering:**
   > *"We implemented Quantum Support Vector Machines using Qiskit. We tested two feature maps: ZFeatureMap (single-qubit rotations, no entanglement) and ZZFeatureMap (with two-qubit entangling gates). We simulated statevector fidelity overlaps to generate similarity kernels, which were then used to train an SVM."*

5. **Findings & Sensitivity Analysis:**
   > *"Z-QSVM performed reasonably well with 84.2% F1, but ZZ-QSVM degraded to 58.3% F1. Classical models outperformed both QSVMs. Our ablation tests showed that increasing circuit repetitions or qubit counts did not improve accuracy because the small training sample could not support higher-dimensional Hilbert spaces. We verified these conclusions across 5 independent random splits and explained the decision drivers using SHAP."*

