# Breast Cancer Prediction with a Quantum SVM

This project teaches the basic steps behind a **quantum-kernel support vector machine (QSVM)** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. It compares two simulated quantum feature maps with classical machine-learning models.

The main workflow is in [wdbc_qsvm_notebook.ipynb](wdbc_qsvm_notebook.ipynb). Open it, read the short explanation before each section, and run the cells from top to bottom. The notebook prints intermediate shapes and checks, displays the graphs below, and saves the tables and images in `results/`.

> **In this run:** the UCI download was unavailable, so the notebook used scikit-learn's bundled copy of WDBC. It is the same dataset. The source is recorded in [results/data_source.txt](results/data_source.txt).

## Quick setup

Use Python 3.10 or newer. In PowerShell, open this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name wdbc-qsvm --display-name "Python (wdbc-qsvm)"
jupyter lab wdbc_qsvm_notebook.ipynb
```

In JupyterLab, choose the `Python (wdbc-qsvm)` kernel, then select **Run → Run All Cells**. The notebook tries UCI dataset 17 first; if the download is unavailable, it automatically loads scikit-learn's WDBC copy and prints which source it used.

The completed run used Qiskit 2.6.1, Qiskit Aer 0.18.2, and Qiskit Machine Learning 0.9.1. The versions are saved in [results/package_versions.csv](results/package_versions.csv).

## What a QSVM is doing

A classical SVM separates the two diagnoses by learning a boundary between patients. A kernel gives the SVM a way to compare two patients by their similarity.

For the QSVM, a **feature map** turns each patient's reduced measurements into a quantum state. The quantum kernel estimates how similar two of those states are. The resulting similarity table is then passed to a normal SVM. Here the quantum states are simulated on a computer; no quantum hardware is used.

- **ZFeatureMap:** applies separate phase rotations to the features, without entangling qubits.
- **ZZFeatureMap:** also encodes pairwise feature relationships using entangling gates.
- **Kernel value:** ranges from 0 to 1. A value near 1 means the two encoded states are similar.

## How to follow the project, step by step

The notebook follows these steps. Each step prints or displays its result so you can see what the model is doing.

1. **Load and inspect WDBC.** The dataset has 569 patient records, 30 numeric measurements, 357 benign labels, and 212 malignant labels. The notebook drops an ID column if one is present, sets `malignant = 1` and `benign = 0`, prints the shape and missing values, and plots the class counts and feature correlations.
2. **Split the data.** A stratified 80/20 split creates 455 training records and 114 held-out test records. This keeps the class balance similar in both groups.
3. **Scale without leakage.** Classical pipelines learn `StandardScaler` from their training fold. PCA and the quantum angle scaler are fit on training data only. The quantum features are mapped to angles from 0 to π.
4. **Reduce features for the simulator.** PCA compresses the 30 measurements into four components for the main QSVM. One component becomes one qubit. Six and eight components are also tried in a smaller side experiment.
5. **Build classical baselines.** Logistic Regression, Decision Tree, Random Forest, Naive Bayes, XGBoost, linear SVM, and RBF SVM are tuned with stratified 5-fold cross-validation. Each is tested using all 30 features and the same four-component PCA pipeline.
6. **Build quantum feature maps.** The notebook draws Z and ZZ circuits with four qubits and two repetitions. It computes a train-versus-train kernel and a test-versus-train kernel for each map.
7. **Check and cache the kernels.** The notebook checks that the training kernel is symmetric, its diagonal is 1, and its values are in the range 0 to 1. The matrices are saved in `results/kernels/`, so later runs can reuse them.
8. **Fit the QSVM.** A normal SVM with `kernel="precomputed"` learns from the quantum similarity matrix. Cross-validation selects `C` from 0.1, 1, 10, and 100. The final metrics are calculated on the held-out test set.
9. **Explain and compare.** The notebook creates confusion matrices, ROC curves, a SHAP summary for XGBoost, PCA loading tables, misclassified-patient details, and small comparisons for feature map, repetitions, entanglement, and qubit count.

### How to tell whether a model is working

Start with the **confusion matrix**. It counts correct and incorrect benign/malignant predictions. Since malignant is label 1, the bottom-left cell is a false negative: a malignant case predicted as benign.

Then read the metrics:

- **Accuracy:** fraction of all test patients classified correctly.
- **Precision:** among patients predicted malignant, the fraction that are malignant.
- **Malignant recall:** among actual malignant patients, the fraction the model catches. Low recall means more missed malignant cases.
- **F1:** one score that balances precision and recall.
- **ROC-AUC:** how well the model ranks malignant cases above benign cases across thresholds.

In the notebook, inspect the intermediate output after each step. To see individual QSVM errors and their signed decision scores, open [results/qsvm_misclassified_test_patients.csv](results/qsvm_misclassified_test_patients.csv). A score close to zero means the prediction was close to the learned SVM boundary; it does not explain a medical cause.

## Results from the saved test run

Malignant is the positive class. The complete table is also available in [results/final_comparison.csv](results/final_comparison.csv). Classical training time includes grid search and refitting; QSVM training time includes kernel calculation and SVM tuning.

| Model | Features | Accuracy | Precision | Malignant recall | F1 | ROC-AUC | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| SVM (linear) | All 30 features | 0.982 | 1.000 | 0.952 | 0.976 | 0.996 | 0.146 |
| Random Forest | All 30 features | 0.974 | 1.000 | 0.929 | 0.963 | 0.993 | 1.836 |
| SVM (RBF) | All 30 features | 0.974 | 1.000 | 0.929 | 0.963 | 0.993 | 0.140 |
| Logistic Regression | PCA 4 components | 0.974 | 1.000 | 0.929 | 0.963 | 0.998 | 0.172 |
| SVM (linear) | PCA 4 components | 0.974 | 1.000 | 0.929 | 0.963 | 0.998 | 0.120 |
| Logistic Regression | All 30 features | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 | 0.190 |
| XGBoost | All 30 features | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 | 1.193 |
| Random Forest | PCA 4 components | 0.956 | 0.974 | 0.905 | 0.938 | 0.994 | 1.553 |
| SVM (RBF) | PCA 4 components | 0.947 | 0.974 | 0.881 | 0.925 | 0.997 | 0.158 |
| Decision Tree | PCA 4 components | 0.939 | 0.949 | 0.881 | 0.914 | 0.927 | 0.145 |
| XGBoost | PCA 4 components | 0.939 | 0.973 | 0.857 | 0.911 | 0.992 | 0.507 |
| Decision Tree | All 30 features | 0.930 | 0.905 | 0.905 | 0.905 | 0.925 | 0.188 |
| Naive Bayes | All 30 features | 0.921 | 0.923 | 0.857 | 0.889 | 0.989 | 0.102 |
| Naive Bayes | PCA 4 components | 0.904 | 0.919 | 0.810 | 0.861 | 0.981 | 0.139 |
| QSVM ZFeatureMap | 4 PCA components; 50 quantum training patients | 0.895 | 0.941 | 0.762 | 0.842 | 0.977 | 55.185 |
| QSVM ZZFeatureMap | 4 PCA components; 50 quantum training patients | 0.737 | 0.700 | 0.500 | 0.583 | 0.784 | 74.266 |

The best overall model in this run was the **linear SVM on all 30 features**. The best QSVM was the **ZFeatureMap**. Its recall was lower than the best classical model, and the ZZFeatureMap scored lower still. These results do not show a quantum advantage.

In the small repetition check, ZZ F1 was 0.762 at one repetition, 0.583 at two, and 0.556 at three. In the qubit-count check, F1 was 0.433 for four qubits and 0 for both six and eight. The six- and eight-qubit models predicted no malignant patients in that run. Because those comparisons used only 25 training patients, they should not be read as evidence that more repetitions or qubits are inherently worse.

The main QSVMs use 50 of the 455 training patients because simulating every pair in a large quantum kernel is slow. They still test on all 114 held-out patients. Classical models train on all 455 patients, so the comparison is not equal in training-set size. The qubit-count experiment uses 25 training patients for each of 4, 6, and 8 qubits. Those smaller runs are only a quick check, not a reliable ranking of qubit counts.

### Stability across five splits

The notebook repeats the finalist comparison using five random seeds. The means and standard deviations are in [results/stability_summary.csv](results/stability_summary.csv).

| Model | Malignant recall, mean ± std | F1, mean ± std |
|---|---:|---:|
| Linear SVM, all 30 features | 0.943 ± 0.036 | 0.959 ± 0.021 |
| RBF SVM, PCA 4 components | 0.905 ± 0.038 | 0.925 ± 0.016 |
| QSVM ZFeatureMap | 0.848 ± 0.069 | 0.870 ± 0.032 |
| QSVM ZZFeatureMap | 0.476 ± 0.061 | 0.523 ± 0.047 |

The five-split results also favor the classical models. The Z QSVM is stronger and more consistent than the ZZ QSVM in these runs, but it still trails the classical finalists.

## Important graphs

### Overall model comparison

This graph compares malignant-class F1 across all classical and main quantum models. Taller bars mean a better balance between precision and recall.

![Final model comparison](results/final_model_comparison.png)

The ROC graph shows the tradeoff between false positives and malignant recall as the decision threshold changes.

![ROC curves for all main models](results/roc_curves_all_main_models.png)

This chart isolates the Z and ZZ feature-map comparison and compares the QSVMs with the reduced-feature RBF SVM.

![Feature map comparison](results/experiment_feature_maps.png)

![QSVM compared with RBF SVM](results/experiment_qsvm_vs_rbf.png)

The next chart includes all classical and main quantum models.

![All main models](results/experiment_all_models.png)

### Quantum model checks

The heatmaps show the train-versus-train fidelity similarities. A bright diagonal is expected because each patient is perfectly similar to themself. The confusion matrices show which classes each QSVM gets right or wrong.

![ZFeatureMap training kernel](results/qsvm_zfeaturemap_kernel_heatmap.png)

![ZZFeatureMap training kernel](results/qsvm_zzfeaturemap_kernel_heatmap.png)

![ZFeatureMap confusion matrix](results/qsvm_zfeaturemap_confusion_matrix.png)

![ZZFeatureMap confusion matrix](results/qsvm_zzfeaturemap_confusion_matrix.png)

### Data and feature maps

The class plot and correlation heatmap help explain the dataset before modeling. PCA reduces correlated measurements to a smaller set for the simulator.

![Class distribution](results/01_class_distribution.png)

![Feature correlation heatmap](results/02_correlation_heatmap.png)

![Example feature box plots](results/03_feature_boxplots.png)

![PCA explained variance](results/04_pca_explained_variance.png)

The circuit pictures show the gates used by each feature map. Z applies feature rotations; ZZ includes pairwise entangling operations.

![ZFeatureMap circuit](results/05_z_feature_map_circuit.png)

![ZZFeatureMap circuit](results/06_zz_feature_map_circuit.png)

### Explainability and small experiments

SHAP summarizes which features influenced XGBoost. The experiment graphs compare repetitions and entanglement. More repetitions or qubits did not automatically improve the QSVM in this small run.

The largest SHAP contributions in this saved run were `worst concave points`, `mean concave points`, and `worst perimeter`. See [results/shap_feature_importance.csv](results/shap_feature_importance.csv) for the full ranking.

![XGBoost SHAP summary](results/shap_xgboost_summary.png)

![Five-split stability](results/stability_f1_mean_std.png)

![Repetition comparison](results/experiment_repetitions.png)

![Entanglement comparison](results/experiment_entanglement.png)

![Qubit-count comparison](results/experiment_qubits.png)

## Reference paper: what is different?

The supplied paper, [Islam et al. (2024), arXiv:2404.04686](https://arxiv.org/abs/2404.04686), is a classical machine-learning and SHAP study on a separate private Bangladeshi patient dataset. It has no quantum-computing part. This project borrows the idea of comparing classical baselines and using SHAP, applies those ideas to WDBC, and adds QSVM as the quantum experiment. The datasets are different, so do not compare their scores directly.

## Project files

- [wdbc_qsvm_notebook.ipynb](wdbc_qsvm_notebook.ipynb) — main, step-by-step notebook.
- [helpers.py](helpers.py) — two small helpers for metrics and saved figures.
- [requirements.txt](requirements.txt) — Python dependencies.
- [report.md](report.md) — short project report with results.
- `results/` — CSV tables, graphs, data-source record, and cached kernel matrices.
- [results/final_comparison.csv](results/final_comparison.csv) — full model table.
- [results/stability_summary.csv](results/stability_summary.csv) — five-split mean and standard deviation.
- [results/pca_loadings_top_features.csv](results/pca_loadings_top_features.csv) — which original features contribute most to PCA components.
- [results/shap_feature_importance.csv](results/shap_feature_importance.csv) — XGBoost feature importance from SHAP.
- [results/qsvm_misclassified_test_patients.csv](results/qsvm_misclassified_test_patients.csv) — QSVM test errors and decision scores.

## Limitations

This is a learning project, not a medical diagnostic tool. All quantum results are from statevector simulation, use a small training sample, and do not demonstrate quantum advantage. The optional noise demo was omitted. No result here comes from physical quantum hardware.

## References

1. T. Islam et al. (2024). [Predictive Modeling for Breast Cancer Classification in the Context of Bangladeshi Patients](https://arxiv.org/abs/2404.04686).
2. [UCI Breast Cancer Wisconsin (Diagnostic), dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
3. [Qiskit Machine Learning documentation](https://qiskit-community.github.io/qiskit-machine-learning/).
4. V. Havlíček et al. (2019). [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2).
