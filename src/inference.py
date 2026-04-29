import argparse

import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from src.device_utils import device_name, get_torch_device
from src.model_builder import MODEL_ID

CLASS_NAMES = {0: "Boredom", 1: "Confusion", 2: "Engagement", 3: "Frustration"}


def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 1, 0), num_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def predict_engagement(video_path: str, model_path: str = "outputs/daisee_videomae/final") -> str:
    device = get_torch_device()
    print(f"Device: {device_name(device)}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    frames = extract_frames(video_path)
    if len(frames) < 16:
        raise ValueError(f"Need ≥16 frames, got {len(frames)}: {video_path}")

    batch = processor(list(frames), return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1).item()

    name = CLASS_NAMES.get(pred, "Unknown")
    print(name)
    return name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", required=True)
    p.add_argument("--model_path", default="outputs/daisee_videomae/final")
    a = p.parse_args()
    predict_engagement(a.video_path, model_path=a.model_path)
