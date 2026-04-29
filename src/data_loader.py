import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor

# Labels: 0 Boredom, 1 Confusion, 2 Engagement, 3 Frustration


class DaiseeDataset(Dataset):
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
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def __getitem__(self, idx):
        frames = self._extract_frames(self.video_paths[idx], num_frames=16)
        inputs = self.processor(list(frames), return_tensors="pt")
        video_tensor = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": video_tensor,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
