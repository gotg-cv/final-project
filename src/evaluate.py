"""
evaluate.py

This script evaluates the fine-tuned VideoMAE model on the DAiSEE Test set
and generates quantitative classification metrics.
"""
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import VideoMAEForVideoClassification
from torch.utils.data import DataLoader
from src.data_loader import DaiseeDataset

def parse_test_csv(data_root):
    """
    Parses the DAiSEE Test CSV and builds aligned lists of valid video paths and labels.
    """
    csv_path = os.path.join(data_root, "Labels", "TestLabels.csv")
    df = pd.read_csv(csv_path)
    
    # CRITICAL: Sanitize headers
    df.columns = df.columns.str.strip()
    
    video_paths = []
    labels = []
    
    for _, row in df.iterrows():
        clip_id_ext = str(row['ClipID']).strip()
        clip_id = clip_id_ext.replace('.avi', '').replace('.mp4', '')
        folder_id = clip_id[:6]
        
        # Exact DAiSEE structure: DataSet/Test/110001/1100011002/1100011002.avi
        video_path = os.path.join(data_root, "DataSet", "Test", folder_id, clip_id, clip_id_ext)
        
        if os.path.exists(video_path):
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
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VideoMAE on DAiSEE Test Set")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the DAiSEE dataset root")
    parser.add_argument("--model_path", type=str, default="/kaggle/working/daisee_videomae_final", help="Path to the fine-tuned model")
    args = parser.parse_args()
    
    print("Parsing Test CSV...")
    video_paths, labels = parse_test_csv(args.data_root)
    
    print(f"Found {len(video_paths)} valid test videos. Initializing dataset...")
    test_dataset = DaiseeDataset(video_paths, labels)
    
    print("Initializing DataLoader (batch_size=8)...")
    # CRITICAL for Memory: Do not process all videos at once
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Loading model from {args.model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoMAEForVideoClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Starting inference loop on device: {device}...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            preds = logits.argmax(-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())
            
    print("\n--- Evaluation Results ---")
    print(classification_report(all_labels, all_preds, target_names=['Boredom', 'Confusion', 'Engagement', 'Frustration'], digits=4))

if __name__ == "__main__":
    main()
