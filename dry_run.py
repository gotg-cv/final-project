"""
dry_run.py

A standalone Python script to test the architecture and ensure the environment 
is correctly set up before proceeding to Phase 2.
"""
import torch
from src.model_builder import get_daisee_model

def main():
    print("Loading model and performing surgery (swapping head for 4 classes)...")
    # Using our new model builder
    model = get_daisee_model(freeze_base=True)
    
    # Generate a dummy video tensor: (batch_size, num_frames, num_channels, height, width)
    # The shape strictly follows the VideoMAE input requirements: (1, 16, 3, 224, 224)
    print("Generating dummy video tensor...")
    dummy_tensor = torch.randn(1, 16, 3, 224, 224)
    
    print(f"Input tensor shape: {dummy_tensor.shape}")
    
    print("Passing tensor through the surgically modified model...")
    with torch.no_grad():
        # The model expects pixel_values
        outputs = model(pixel_values=dummy_tensor)
        logits = outputs.logits
    
    print(f"Output logits shape: {logits.shape}")
    
    # Assert the output shape is strictly (1, 4)
    assert logits.shape == (1, 4), f"Expected shape (1, 4), but got {logits.shape}"
    
    print("Success! Model surgery successful. Output shape is exactly (1, 4).")

if __name__ == "__main__":
    main()
