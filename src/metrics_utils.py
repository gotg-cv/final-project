import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
