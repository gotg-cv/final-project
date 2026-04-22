"""
model_builder.py

This module handles the instantiation of the VideoMAE architecture.
It loads the pre-trained `videomae-base-finetuned-kinetics` model 
and modifies its classification head to output the four target affective states:
Boredom, Confusion, Engagement, and Frustration.
"""
from transformers import VideoMAEForVideoClassification

def get_daisee_model(freeze_base=False):
    """
    Loads the VideoMAE model, swaps the classification head for 4 classes,
    and optionally freezes the base transformer layers.
    """
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    
    # Load model with a new 4-class head, discarding the original 400-class head
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    
    if freeze_base:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the final classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    return model
