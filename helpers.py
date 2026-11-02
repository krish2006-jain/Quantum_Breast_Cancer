"""Small helpers used by the WDBC QSVM notebook."""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import Image, display


def classification_metrics(y_true, y_pred, y_score):
    # Return the requested scores, treating malignant (label 1) as the positive class.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def save_figure(figure, file_path):
    # Save a chart, show it in the notebook, and close it to free memory.
    figure.tight_layout()
    figure.savefig(file_path, dpi=180, bbox_inches="tight")
    display(Image(filename=str(file_path)))
    plt.close(figure)
