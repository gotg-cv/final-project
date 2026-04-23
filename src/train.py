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
import pandas as pd
from transformers import Trainer, TrainingArguments
from src.model_builder import get_daisee_model
from src.data_loader import DaiseeDataset

def parse_daisee_csv(data_root, split_name):
    """
    Parses the DAiSEE CSV and builds aligned lists of valid video paths and labels.
    split_name should be 'Train', 'Validation', or 'Test'.
    """
    csv_path = os.path.join(data_root, "Labels", f"{split_name}Labels.csv")
    df = pd.read_csv(csv_path)
    
    video_paths = []
    labels = []
    
    for _, row in df.iterrows():
        clip_id = str(row['ClipID']).replace('.avi', '').replace('.mp4', '')
        folder_id = clip_id[:6] # The first 6 digits determine the parent folder
        
        # Construct exact path: DataSet/Train/400023/4000231047/4000231047.avi
        video_path = os.path.join(data_root, "DataSet", split_name, folder_id, clip_id, f"{clip_id}.avi")
        
        if os.path.exists(video_path):
            # DAiSEE scores: 0=Boredom, 1=Confusion, 2=Engagement, 3=Frustration (to match our model head)
            scores = [row['Boredom'], row['Confusion'], row['Engagement'], row['Frustration']]
            dominant_label = scores.index(max(scores))
            
            video_paths.append(video_path)
            labels.append(dominant_label)
            
    return video_paths, labels

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
    
    # Parse the CSVs to get aligned paths and labels
    print("Parsing Train CSV...")
    train_paths, train_labels = parse_daisee_csv(args.data_root, "Train")
    print("Parsing Validation CSV...")
    val_paths, val_labels = parse_daisee_csv(args.data_root, "Validation")
    
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
