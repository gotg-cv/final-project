"""
model_builder.py

This module handles the instantiation of the VideoMAE architecture.
It loads the pre-trained `videomae-base-finetuned-kinetics` model 
and modifies its classification head to output the four target affective states:
Boredom, Confusion, Engagement, and Frustration.
"""
from transformers import VideoMAEForVideoClassification

def get_daisee_model(ablation=False):
    """
    Loads the VideoMAE model, swaps the classification head for 4 classes,
    and handles parameter freezing based on the ablation flag.
    """
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    
    # load model with a new 4-class head, discarding the original 400-class head
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    
    for name, param in model.named_parameters():
        if ablation:
            # unfreeze the classifier and the last two transformer blocks (10 and 11)
            if "classifier" in name or "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            # default: unfreeze only the classifier head
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
    return model
