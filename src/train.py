"""
train.py

This script orchestrates the fine-tuning process for the VideoMAE model using 
the Hugging Face Trainer. It parses the dataset directory, splits the data,
and configures optimized TrainingArguments for a Kaggle P100 GPU environment.
"""
import argparse
import random
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
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the DAiSEE dataset")
    args = parser.parse_args()
    
    print("Initializing model...")
    # Instantiate the model executing the ablation study baseline (training only the head)
    model = get_daisee_model(freeze_base=True)
    
    print("Loading data...")
    video_paths, labels = get_video_paths_and_labels(args.data_dir)
    
    # 80/20 Split
    split_idx = int(len(video_paths) * 0.8)
    train_paths, val_paths = video_paths[:split_idx], video_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"Creating datasets (Train: {len(train_paths)}, Val: {len(val_paths)})...")
    train_dataset = DaiseeDataset(train_paths, train_labels)
    val_dataset = DaiseeDataset(val_paths, val_labels)
    
    # Configure TrainingArguments for P100 GPU on Kaggle
    training_args = TrainingArguments(
        output_dir="/kaggle/working/daisee_videomae_checkpoints",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir="/kaggle/working/logs",
        logging_steps=10,
        remove_unused_columns=False, # Essential for VideoMAE with custom pixel_values input
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training loop...")
    trainer.train()
    
    final_output_dir = "/kaggle/working/daisee_videomae_final"
    print(f"Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
