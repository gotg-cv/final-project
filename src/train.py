"""
train.py

This script orchestrates the fine-tuning process for the VideoMAE model using 
the Hugging Face Trainer. It parses the dataset directory, splits the data,
and configures optimized TrainingArguments for a Kaggle P100 GPU environment.
"""
import os
import argparse
import random
import json
from transformers import Trainer, TrainingArguments
from src.model_builder import get_daisee_model
from src.data_loader import DaiseeDataset

def get_video_paths_and_labels(data_dir):
    """
    Placeholder function to extract video paths and labels.
    In a real scenario, this would parse the DAiSEE Labels CSVs
    and map them to the .avi files in the data_dir.
    """
    print(f"Scanning {data_dir} for videos and labels...")
    # NOTE: Replace this with your actual DAiSEE dataset parsing logic!
    # Example:
    # df = pd.read_csv(os.path.join(data_dir, 'Labels', 'TrainLabels.csv'))
    # video_paths = [os.path.join(data_dir, 'DataSet', f) for f in df['ClipID']]
    # labels = df['AffectiveState'].tolist()
    
    # Returning 100 dummy entries for the sake of testing script structure
    dummy_paths = ["dummy_path.avi"] * 100
    dummy_labels = [random.randint(0, 3) for _ in range(100)]
    return dummy_paths, dummy_labels

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on DAiSEE")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root DAiSEE dataset")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config = json.load(f)
    
    print("Initializing model...")
    # Instantiate the model executing the ablation study baseline (training only the head)
    model = get_daisee_model(freeze_base=config["freeze_base"])
    
    train_path = os.path.join(args.data_root, "DataSet", "Train")
    val_path = os.path.join(args.data_root, "DataSet", "Validation")
    
    print("Loading data...")
    train_paths, train_labels = get_video_paths_and_labels(train_path)
    val_paths, val_labels = get_video_paths_and_labels(val_path)
    
    print(f"Creating datasets (Train: {len(train_paths)}, Val: {len(val_paths)})...")
    train_dataset = DaiseeDataset(video_paths=train_paths, labels=train_labels)
    eval_dataset = DaiseeDataset(video_paths=val_paths, labels=val_labels)
    
    # Configure TrainingArguments for P100 GPU on Kaggle
    training_args = TrainingArguments(
        output_dir="/kaggle/working/daisee_videomae_checkpoints",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir="/kaggle/working/logs",
        logging_steps=10,
        remove_unused_columns=False, # Essential for VideoMAE with custom pixel_values input
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training loop...")
    trainer.train()
    
    final_output_dir = "/kaggle/working/daisee_videomae_final"
    print(f"Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
