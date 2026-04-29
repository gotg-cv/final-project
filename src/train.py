import argparse
import json
import os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import Trainer, TrainingArguments

from src.daisee_io import parse_daisee_csv
from src.data_loader import DaiseeDataset
from src.device_utils import device_name, get_torch_device
from src.metrics_utils import compute_metrics
from src.model_builder import get_daisee_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on DAiSEE")
    parser.add_argument("--data_root", required=True, help="DAiSEE root directory")
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--output_dir",
        default="outputs/daisee_videomae",
        help="Checkpoints and logs (use a persistent path when on Colab / Drive)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint-N folder to resume from",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate train set for quick runs (hyperparam probing)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate validation set for quick runs",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Local pick for device: {device_name(get_torch_device())} (HF Trainer will set the real device)")

    model = get_daisee_model(freeze_base=config["freeze_base"])

    print("Train CSV…")
    train_paths, train_labels = parse_daisee_csv(args.data_root, "Train")
    print("Validation CSV…")
    val_paths, val_labels = parse_daisee_csv(args.data_root, "Validation")

    if args.max_train_samples:
        train_paths = train_paths[: args.max_train_samples]
        train_labels = train_labels[: args.max_train_samples]
    if args.max_eval_samples:
        val_paths = val_paths[: args.max_eval_samples]
        val_labels = val_labels[: args.max_eval_samples]

    print(f"Samples — train: {len(train_paths)}, val: {len(val_paths)}")
    train_dataset = DaiseeDataset(video_paths=train_paths, labels=train_labels)
    eval_dataset = DaiseeDataset(video_paths=val_paths, labels=val_labels)

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=log_dir,
        logging_steps=10,
        remove_unused_columns=False,  # keep pixel_values through the collator
        save_total_limit=None,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print(f"Trainer device: {trainer.args.device}")

    if args.resume_from_checkpoint:
        print(f"Resuming from {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
