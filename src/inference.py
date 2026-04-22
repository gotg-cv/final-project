import torch
import cv2
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def predict_engagement(video_path, model_path="/kaggle/working/daisee_videomae_final"):
    # 1. Load the fine-tuned model and processor
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model.eval()
    
    # 2. Process the video
    frames = extract_frames(video_path)
    inputs = processor(list(frames), return_tensors="pt")
    
    # 3. Run Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
    # 4. Map to DAiSEE labels
    labels = {0: "Boredom", 1: "Confusion", 2: "Engagement", 3: "Frustration"}
    prediction = labels.get(predicted_class_idx, "Unknown")
    
    print(f"Predicted Affective State: {prediction}")
    return prediction

if __name__ == "__main__":
    # Replace with the path to a test video in your Kaggle input directory
    test_video = "/kaggle/input/daisee/DataSet/Test/Sample_Video.mp4" 
    predict_engagement(test_video)