"""
data_loader.py

This module handles video frame extraction, preprocessing, 
and the definition of PyTorch datasets/dataloaders for the DAiSEE dataset.
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from transformers import VideoMAEImageProcessor

class DaiseeDataset(Dataset):
    """
    PyTorch Dataset for the DAiSEE dataset.
    Extracts 16 uniform frames from a video and processes them for VideoMAE.
    Labels: 0=Boredom, 1=Confusion, 2=Engagement, 3=Frustration.
    """
    def __init__(self, video_paths, labels, processor_name="MCG-NJU/videomae-base-finetuned-kinetics"):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = VideoMAEImageProcessor.from_pretrained(processor_name)
        
    def __len__(self):
        return len(self.video_paths)
        
    def _extract_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            # Handle edge case where video cannot be read
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
            
        # Get uniformly distributed frame indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback if frame cannot be read
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract exactly 16 frames
        frames = self._extract_frames(video_path, num_frames=16)
        
        # Process frames using VideoMAE processor (normalization, resizing to 224x224)
        # The processor expects a list of 3D numpy arrays (H, W, C)
        inputs = self.processor(list(frames), return_tensors="pt")
        
        # inputs["pixel_values"] will have shape (1, num_frames, num_channels, height, width)
        # We squeeze the first dimension to get (num_frames, num_channels, height, width)
        video_tensor = inputs["pixel_values"].squeeze(0)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Return a dictionary instead of a tuple so the HF Trainer can parse it
        return {
            "pixel_values": video_tensor,
            "labels": label_tensor
        }
