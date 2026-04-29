from transformers import VideoMAEForVideoClassification

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"


def get_daisee_model(freeze_base=False):
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model
