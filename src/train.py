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
from transformers.trainer_utils import get_last_checkpoint
from src.model_builder import get_daisee_model
from src.data_loader import DaiseeDataset

def parse_daisee_csv(data_root, split_name):
    csv_path = os.path.join(data_root, "Labels", f"{split_name}Labels.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # sanitize dirty headers
    
    video_paths, labels = [], []
    
    for _, row in df.iterrows():
        clip_id_ext = str(row['ClipID']).strip()
        clip_id = clip_id_ext.replace('.avi', '').replace('.mp4', '')
        folder_id = clip_id[:6]
        
        # exact daisee structure: dataset/train/110001/1100011002/1100011002.avi
        video_path = os.path.join(data_root, "DataSet", split_name, folder_id, clip_id, clip_id_ext)
        
        if os.path.exists(video_path):
            # explicitly map csv columns to our model's head indices
            scores = {
                0: row['Boredom'],
                1: row['Confusion'],
                2: row['Engagement'],
                3: row['Frustration']
            }
            dominant_label = max(scores, key=scores.get)
            video_paths.append(video_path)
            labels.append(dominant_label)
            
    return video_paths, labels

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on DAiSEE")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root DAiSEE dataset")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study (unfreeze last two layers)")
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config = json.load(f)
    
    print("Initializing model...")
    # instantiate the model executing the ablation study or baseline
    model = get_daisee_model(ablation=args.ablation)
    
    # parse the csvs to get aligned paths and labels
    print("Parsing Train CSV...")
    train_paths, train_labels = parse_daisee_csv(args.data_root, "Train")
    print("Parsing Validation CSV...")
    val_paths, val_labels = parse_daisee_csv(args.data_root, "Validation")
    
    print(f"Creating datasets (Train: {len(train_paths)}, Val: {len(val_paths)})...")
    train_dataset = DaiseeDataset(video_paths=train_paths, labels=train_labels)
    eval_dataset = DaiseeDataset(video_paths=val_paths, labels=val_labels)
    
    # configure trainingarguments for p100 gpu on kaggle
    if args.ablation:
        output_dir = "/kaggle/working/daisee_videomae_ablation_checkpoints"
        final_save_path = "/kaggle/working/daisee_videomae_ablation_final"
    else:
        output_dir = "/kaggle/working/daisee_videomae_checkpoints"
        final_save_path = "/kaggle/working/daisee_videomae_final"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir="/kaggle/working/logs",
        logging_steps=10,
        remove_unused_columns=False, # essential for videomae with custom pixel_values input
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training loop...")
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("No valid checkpoint found. Starting training from scratch.")
            
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"Saving final model to {final_save_path}...")
    trainer.save_model(final_save_path)
    print("Training complete!")

if __name__ == "__main__":
    main()
