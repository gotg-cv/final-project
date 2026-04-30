#!/usr/bin/env python3
"""
Build report figures from a Hugging Face Trainer run directory:
  - training_curves.png  (loss + eval loss / accuracy / macro F1 if present)
  - confusion_matrix.png from metrics.json (optional)

trainer_state.json lives under each checkpoint-* folder; the script picks the
longest log_history (full training to date).

You can also pass --ipynb to rebuild training loss / eval loss curves from
saved notebook stdout (Kaggle downloads often have Trainer dict lines but no
embedded PNG plots). Confusion matrices are not present in that text; use
--metrics_json from evaluate.py for those.

Usage (after you copy the run folder from Kaggle):
  python plot_run_artifacts.py --run_dir outputs/run01
  python plot_run_artifacts.py --run_dir outputs/run01 \\
    --metrics_json outputs/run01/eval_report/metrics.json

  python plot_run_artifacts.py --ipynb ~/Downloads/my_run.ipynb
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from glob import glob


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_ESCAPE.sub("", s)


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def parse_trainer_dicts_from_notebook_text(text: str) -> list[dict]:
    """Parse HF Trainer / tqdm dict lines from captured notebook stdout."""
    rows: list[dict] = []
    for raw in text.splitlines():
        line = _strip_ansi(raw)
        i = line.find("{")
        j = line.rfind("}")
        if i < 0 or j <= i:
            continue
        chunk = line[i : j + 1]
        try:
            d = ast.literal_eval(chunk)
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and "epoch" in d:
            rows.append(d)
    return rows


def stdout_dicts_to_log_history(rows: list[dict]) -> list[dict]:
    """Map stdout dicts (string or float values) to plot_training_curves rows."""
    hist: list[dict] = []
    for d in rows:
        ep = _to_float(d.get("epoch"))
        if ep is None:
            continue
        ev_loss = _to_float(d.get("eval_loss"))
        tr_loss = _to_float(d.get("loss"))
        if ev_loss is not None:
            row: dict = {"epoch": ep, "eval_loss": ev_loss}
            for k_src, k_dst in (
                ("eval_accuracy", "eval_accuracy"),
                ("eval_f1_macro", "eval_f1_macro"),
            ):
                v = _to_float(d.get(k_src))
                if v is not None:
                    row[k_dst] = v
            hist.append(row)
        elif tr_loss is not None:
            hist.append({"epoch": ep, "loss": tr_loss})
    return hist


def load_log_histories_from_ipynb(ipynb_path: str) -> list[tuple[int, list[dict]]]:
    with open(ipynb_path) as f:
        nb = json.load(f)
    results: list[tuple[int, list[dict]]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "src.train" not in src:
            continue
        text = ""
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                text += "".join(o.get("text", []))
        if not text.strip():
            continue
        hist = stdout_dicts_to_log_history(parse_trainer_dicts_from_notebook_text(text))
        if hist:
            results.append((i, hist))
    return results


def load_log_history(run_dir: str) -> list[dict]:
    paths = glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not paths:
        return []
    best: list[dict] = []
    best_step = -1
    for p in paths:
        with open(p) as f:
            state = json.load(f)
        step = int(state.get("global_step", 0))
        hist = state.get("log_history") or []
        if step >= best_step and len(hist) >= len(best):
            best_step = step
            best = hist
    return best


def plot_training_curves(
    log_history: list[dict],
    out_path: str,
    *,
    suptitle: str = "Training curves (from Trainer log_history)",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    train_epochs, train_loss = [], []
    eval_epochs, eval_loss, eval_acc, eval_f1 = [], [], [], []

    for row in log_history:
        ep = row.get("epoch")
        if "loss" in row and ep is not None and "eval_loss" not in row:
            train_epochs.append(float(ep))
            train_loss.append(float(row["loss"]))
        if "eval_loss" in row and ep is not None:
            eval_epochs.append(float(ep))
            eval_loss.append(float(row["eval_loss"]))
            if "eval_accuracy" in row:
                eval_acc.append(float(row["eval_accuracy"]))
            else:
                eval_acc.append(np.nan)
            if "eval_f1_macro" in row:
                eval_f1.append(float(row["eval_f1_macro"]))
            else:
                eval_f1.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if train_epochs:
        axes[0].plot(train_epochs, train_loss, "b.", alpha=0.35, label="train (per log)")
        # light moving average for readability
        if len(train_loss) > 10:
            w = min(25, len(train_loss) // 5)
            kern = np.ones(w) / w
            smoothed = np.convolve(train_loss, kern, mode="valid")
            e_smooth = np.convolve(train_epochs, kern, mode="valid")
            axes[0].plot(e_smooth, smoothed, "b-", lw=2, label=f"train smoothed (w={w})")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Training loss")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No training loss in log_history", ha="center")

    if eval_epochs:
        axes[1].plot(eval_epochs, eval_loss, "o-", color="C0", label="eval loss")
        if not all(np.isnan(eval_acc)):
            ax2 = axes[1].twinx()
            ax2.plot(eval_epochs, eval_acc, "s-", color="C1", label="eval accuracy")
            if not all(np.isnan(eval_f1)):
                ax2.plot(eval_epochs, eval_f1, "^-", color="C2", label="eval macro F1")
            ax2.set_ylabel("accuracy / macro F1")
            ax2.legend(loc="lower right", fontsize=8)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("eval loss")
        axes[1].set_title("Validation")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval metrics in log_history", ha="center")

    fig.suptitle(suptitle)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


def plot_confusion_matrix(cm: list[list[int]], out_path: str) -> None:
    import matplotlib.pyplot as plt

    names = ["Boredom", "Confusion", "Engagement", "Frustration"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix (validation or test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Wrote", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training curves + optional confusion matrix")
    p.add_argument("--run_dir", default=None, help="Trainer output_dir (with checkpoint-* folders)")
    p.add_argument("--metrics_json", default=None, help="Path to evaluate.py metrics.json")
    p.add_argument(
        "--ipynb",
        default=None,
        help="Jupyter notebook with executed training cell stdout (loss / eval_loss dict lines)",
    )
    p.add_argument("--out_dir", default=None, help="Where to save PNGs (default: run_dir/plots or next to metrics.json)")
    args = p.parse_args()

    if not args.run_dir and not args.metrics_json and not args.ipynb:
        p.print_help()
        raise SystemExit(1)

    out_base = args.out_dir

    if args.ipynb:
        if not os.path.isfile(args.ipynb):
            print("Missing notebook:", args.ipynb)
        else:
            nb_stem = os.path.splitext(os.path.basename(args.ipynb))[0]
            ipynb_out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ipynb)), f"{nb_stem}_plots")
            os.makedirs(ipynb_out, exist_ok=True)
            runs = load_log_histories_from_ipynb(args.ipynb)
            if not runs:
                print(
                    "No training log dicts found. Ensure the notebook was executed and cells that run "
                    "src.train still contain stream output."
                )
            for cell_idx, hist in runs:
                safe = re.sub(r"[^\w\-]+", "_", nb_stem).strip("_")[:80] or "notebook"
                png = os.path.join(ipynb_out, f"training_curves_{safe}_cell_{cell_idx}.png")
                plot_training_curves(
                    hist,
                    png,
                    suptitle=f"Training curves (parsed notebook stdout, cell {cell_idx})",
                )

    if args.run_dir:
        out_base = args.out_dir or os.path.join(args.run_dir, "plots")
        os.makedirs(out_base, exist_ok=True)
        hist = load_log_history(args.run_dir)
        if not hist:
            print(
                "No checkpoint-*/trainer_state.json under --run_dir.\n"
                "Copy the full training output from Kaggle (all checkpoint-* folders), not only final/."
            )
        else:
            plot_training_curves(hist, os.path.join(out_base, "training_curves.png"))

    if args.metrics_json:
        if not os.path.isfile(args.metrics_json):
            print("Missing file:", args.metrics_json)
        else:
            mj_dir = os.path.dirname(os.path.abspath(args.metrics_json))
            m_out = args.out_dir or (out_base if args.run_dir else os.path.join(mj_dir, "plots"))
            os.makedirs(m_out, exist_ok=True)
            with open(args.metrics_json) as f:
                data = json.load(f)
            cm = data.get("confusion_matrix")
            if cm:
                plot_confusion_matrix(cm, os.path.join(m_out, "confusion_matrix.png"))
            else:
                print("metrics.json has no confusion_matrix key")


if __name__ == "__main__":
    main()
