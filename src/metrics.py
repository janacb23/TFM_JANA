"""
metrics.py

Cálculo de métricas para clasificación binaria y multiclase
- Binario: AUC-ROC, sensibilidad (recall clase positiva), especificidad, F1
- Multiclase: AUC-ROC macro (one-vs-rest), F1 macro, accuracy
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, confusion_matrix, accuracy_score
)


def compute_binary_metrics(y_true, y_pred, y_proba) -> dict:
    """
    y_true: array de labels reales (0/1)
    y_pred: array de labels predichos (0/1)
    y_proba: array de probabilidad de la clase positiva (1 = con RD)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": roc_auc_score(y_true, y_proba),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_multiclass_metrics(y_true, y_pred, y_proba) -> dict:
    """
    y_proba: matriz (n_samples, n_classes) de probabilidades softmax
    """
    return {
        "auc_macro": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def aggregate_runs(list_of_metric_dicts: list) -> dict:
    """
    Recibe una lista de diccionarios de métricas (uno por run) y devuelve media +- desviación estándar de cada métrica
    """
    keys = list_of_metric_dicts[0].keys()
    summary = {}
    for k in keys:
        values = np.array([m[k] for m in list_of_metric_dicts])
        summary[k] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_runs": len(values),
            "values": values.tolist(),
        }
    return summary


def format_metric_summary(summary: dict, metric_name: str) -> str:
    """Devuelve string tipo: 'F1: 90.15% ± 3.00% (n=3)' """
    m = summary[metric_name]
    return f"{metric_name}: {m['mean']*100:.2f}% \u00b1 {m['std']*100:.2f}% (n={m['n_runs']})"
