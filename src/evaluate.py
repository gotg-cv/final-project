import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import Trainer, TrainingArguments, VideoMAEForVideoClassification

from src.daisee_io import parse_daisee_csv
from src.data_loader import DaiseeDataset
from src.metrics_utils import compute_metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on DAiSEE Validation")
    p.add_argument("--data_root", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    paths, labels = parse_daisee_csv(args.data_root, "Validation")
    if args.max_samples:
        paths = paths[: args.max_samples]
        labels = labels[: args.max_samples]
    if not paths:
        raise RuntimeError("No validation clips.")

    ds = DaiseeDataset(paths, labels)
    model = VideoMAEForVideoClassification.from_pretrained(args.model_path)

    out_dir = args.out_dir or args.model_path
    os.makedirs(out_dir, exist_ok=True)

    ta = TrainingArguments(
        output_dir=os.path.join(out_dir, "_eval_scratch"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=ta, compute_metrics=compute_metrics, eval_dataset=ds)

    metrics = trainer.evaluate()
    po = trainer.predict(ds)

    logits = np.asarray(po.predictions)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    y_pred = logits.argmax(axis=-1)
    y_true = np.asarray(po.label_ids)

    names = ["Boredom", "Confusion", "Engagement", "Frustration"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=names, zero_division=0)

    def scalar(x):
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    payload = {
        "trainer_metrics": {k: scalar(v) for k, v in metrics.items()},
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    mf = os.path.join(out_dir, "metrics.json")
    with open(mf, "w") as f:
        json.dump(payload, f, indent=2)
    print(report)
    print("Wrote", mf)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(4))
        ax.set_yticklabels(names)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        pth = os.path.join(out_dir, "confusion_matrix.png")
        plt.savefig(pth, dpi=150)
        plt.close()
        print("Wrote", pth)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
