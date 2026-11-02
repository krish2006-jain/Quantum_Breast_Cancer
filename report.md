# WDBC Breast Cancer Classification with a Simulated QSVM

## Introduction

This project compares classical machine-learning models with a quantum-kernel support vector machine on the Wisconsin Diagnostic Breast Cancer dataset. The question is whether the simulated quantum kernel is competitive on this small, clean, two-class dataset. The notebook reports accuracy, precision, malignant recall, F1, ROC-AUC, and training time. Malignant recall is important because a malignant patient predicted as benign is a missed positive case.

## Dataset

The notebook tried dataset 17 from the UCI Machine Learning Repository. UCI was unavailable in this saved run, so it used scikit-learn's bundled copy of WDBC. It contains 569 patients and 30 numeric measurements per patient. There are 357 benign cases and 212 malignant cases, with 0 missing feature values. The labels are encoded as 0 for benign and 1 for malignant.

We used a stratified 80/20 train/test split. The training set contains 455 patients and the held out test set contains 114. Standard scaling, PCA, and angle scaling are fitted on training data only. The classical model pipelines also fit their scaling and PCA inside each cross-validation fold.

## Related paper and how we adapted it

The supplied paper is Islam et al.'s *Predictive Modeling for Breast Cancer Classification in the Context of Bangladeshi Patients: A Supervised Machine Learning Approach with Explainable AI* [1]. It studies classical models and SHAP on a separate private Bangladeshi patient dataset. It does not use quantum computing. We reuse its classical-baseline comparison and XGBoost/SHAP explanation ideas, but evaluate them on public WDBC data. The QSVM is the new part of this project. The two studies use different datasets, so their scores should not be compared as if they were one experiment.

## Simple theory

A support vector machine (SVM) separates two classes by finding a boundary with a wide margin. A kernel lets the SVM compare examples through similarity values, without explicitly building a large feature vector by hand.

A quantum feature map places each patient into a quantum state. The quantum kernel estimates the squared overlap between two states, `|<phi(x)|phi(x')>|^2`. A value near one means the encoded states are similar; a value near zero means they are different. The SVM receives this similarity matrix and learns a boundary from it.

The Z feature map applies individual phase rotations to features. The ZZ feature map adds pairwise interactions between features, which entangle qubits. In simple terms, the ZZ map can represent relationships between pairs of measurements. These circuits create states in a Hilbert space, the mathematical space used to describe quantum states. Here, all circuits and kernel measurements are simulated; no quantum processor is used.

## Method

The notebook compares logistic regression, decision tree, random forest, Gaussian naive Bayes, XGBoost, RBF SVM, and linear SVM. Each has a small parameter grid and stratified five-fold cross-validation. They are evaluated on all original features and a four-component PCA pipeline.

The quantum models use four principal components, two circuit repetitions, and 1,024 sampler shots. PCA is fit on the training split, and a MinMaxScaler maps the components to angles from 0 to π. To keep circuit simulation within a practical runtime, the main QSVM kernels use a stratified sample of 50 training patients; each QSVM still predicts the full held out test set. The classical models use the full training split. The 4/6/8-qubit side experiment uses 25 training patients at each width so its qubit counts use equal sample sizes.

For each map, the notebook builds a train/train kernel and a test/train kernel, saves both matrices, tunes SVM `C` over `[0.1, 1, 10, 100]`, and checks symmetry, a unit diagonal, and values between zero and one. A seeded statevector sampler makes the finite-shot kernel repeatable. Saved matrices are checked before reuse. The optional noise demo was omitted to keep the run focused on the main comparison.

## Results

The following held out test scores are copied from `results/final_comparison.csv`. Training seconds include cross-validation and refitting for classical models; for QSVMs they include kernel creation and SVM tuning.

| Model | Features | Accuracy | Precision | Malignant recall | F1 | ROC-AUC | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| SVM (linear) | All 30 features | 0.982 | 1.000 | 0.952 | 0.976 | 0.996 | 0.146 |
| SVM (RBF) | PCA 4 components | 0.947 | 0.974 | 0.881 | 0.925 | 0.997 | 0.158 |
| QSVM ZFeatureMap | 4 PCA components; 50 quantum training patients | 0.895 | 0.941 | 0.762 | 0.842 | 0.977 | 55.185 |
| QSVM ZZFeatureMap | 4 PCA components; 50 quantum training patients | 0.737 | 0.700 | 0.500 | 0.583 | 0.784 | 74.266 |

The best overall model by test F1 was **SVM (linear) using All 30 features**. Its malignant recall was 0.952. The best QSVM by malignant recall, then F1, was **QSVM ZFeatureMap**: accuracy 0.895, recall 0.762, F1 0.842, and ROC-AUC 0.977. The ZZ map scored lower on this split. These results do not show a quantum advantage.

The five-split stability results for the finalists are in `results/stability_summary.csv`:

| Model | Accuracy mean ± std | Malignant recall mean ± std | F1 mean ± std | ROC-AUC mean ± std |
|---|---:|---:|---:|---:|
| SVM (linear) (All 30 features) | 0.970 ± 0.015 | 0.943 ± 0.036 | 0.959 ± 0.021 | 0.994 ± 0.005 |
| SVM (RBF) (PCA 4 components) | 0.946 ± 0.011 | 0.905 ± 0.038 | 0.925 ± 0.016 | 0.989 ± 0.007 |
| QSVM ZFeatureMap | 0.907 ± 0.021 | 0.848 ± 0.069 | 0.870 ± 0.032 | 0.959 ± 0.021 |
| QSVM ZZFeatureMap | 0.681 ± 0.033 | 0.476 ± 0.061 | 0.523 ± 0.047 | 0.700 ± 0.053 |

Across these splits, the full-feature linear SVM had higher mean F1 than either QSVM. The Z QSVM was more consistent and had higher malignant recall than the ZZ QSVM on these runs, but it remained below the classical finalists. These are small experiments with a sampled quantum training set, so they are evidence about this setup, not a general claim about all QSVMs.

The repetition experiment and qubit-count experiment are saved as separate CSV tables. They show that more repetitions and more qubits did not automatically improve this result:

| ZZ repetitions | Accuracy | Malignant recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.825 | 0.762 | 0.762 | 0.881 |
| 2 | 0.737 | 0.500 | 0.583 | 0.784 |
| 3 | 0.649 | 0.595 | 0.556 | 0.667 |

| Qubits | Quantum training patients | Accuracy | Malignant recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 4 | 25 | 0.702 | 0.310 | 0.433 | 0.758 |
| 6 | 25 | 0.632 | 0.000 | 0.000 | 0.602 |
| 8 | 25 | 0.632 | 0.000 | 0.000 | 0.602 |

## Explainability

The XGBoost SHAP summary ranks worst concave points, mean concave points, worst perimeter among its most influential features on the held out data. The PCA loading table shows that the first component is most associated with mean concave points, mean concavity, worst concave points among its listed high-loading features. These are model explanations, not clinical conclusions.

The Z QSVM misclassified 12 held out patients. Examples closest to its decision boundary were row 205 (Malignant predicted as Benign, decision score -0.004); row 75 (Malignant predicted as Benign, decision score -0.044); row 563 (Malignant predicted as Benign, decision score -0.133). The signed SVM score is a measure of distance from the learned boundary; it does not explain a biological cause.

## Discussion

The classical models are strong on WDBC, with the best test accuracy at 0.982. The Z QSVM is a useful demonstration that a fidelity kernel can train a standard SVM, but its test F1 and malignant recall are lower than the best classical model. The ZZ map performs worse here. The differences may reflect feature-map choice, finite shots, PCA compression, the small quantum training sample, or the random held out split. The experiment does not isolate one cause.

## Limitations

The quantum experiments use simulation, a small number of qubits, finite shots, and fewer training patients than the classical models. The 6 and 8 qubit side runs also use a smaller training sample. The dataset is small and relatively easy for classical models. The test set is one split; the five-split stability check gives more context but is still limited. The result is not a clinical diagnostic tool. The noise demonstration was left out, and no physical quantum hardware was tested.

## Future work

Future work could compare exact statevector kernels with measured-shot kernels, test noise models, use more patients in the quantum training set where hardware permits, test alternative feature maps, and repeat the study on additional datasets. A larger nested cross-validation study would give a stronger estimate of model performance.

## Conclusion

The project completed a classical and quantum-kernel comparison on WDBC with saved tables, figures, and kernel caches. The full-feature linear SVM was the strongest model in this run. The four-qubit Z QSVM was the better of the two main quantum maps, but neither map beat the classical baselines. This is an honest, useful result for a small simulator-based study.

## References

1. T. Islam et al. (2024). *Predictive Modeling for Breast Cancer Classification in the Context of Bangladeshi Patients: A Supervised Machine Learning Approach with Explainable AI*. [arXiv:2404.04686](https://arxiv.org/abs/2404.04686).
2. UCI Machine Learning Repository. *Breast Cancer Wisconsin (Diagnostic)*, dataset 17. [Dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
3. Qiskit Machine Learning. [Project documentation](https://qiskit-community.github.io/qiskit-machine-learning/).
4. F. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. V. Havlíček et al. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature, 567, 209–212. [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2).
