# Context-Aware Classroom Engagement Monitor

## Project Title & Overview
This project presents a Context-Aware Classroom Engagement Monitor designed to classify student affective states. It leverages a fine-tuned VideoMAE model (`videomae-base-finetuned-kinetics`) to analyze spatio-temporal dynamics from video clips. The model is trained on the DAiSEE dataset to detect four core affective states: Boredom, Confusion, Engagement, and Frustration.

## Environment Requirements
The project's dependencies are rigorously managed using a dedicated `requirements.txt` file. This ensures full reproducibility of the Python environment, including the required modules for PyTorch, Transformers, OpenCV, and scikit-learn.

## Setup Instructions
To replicate the environment locally or on a remote compute instance, execute the following bash commands:

```bash
git clone https://github.com/gotg-cv/final-project.git
cd final-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Instructions
This project utilizes the DAiSEE dataset. Due to ethical constraints and privacy regulations, the dataset is not publicly distributed in this repository. You must formally request access permission from the original DAiSEE authors.

Once acquired, extract the dataset and organize it in the project root exactly as follows:
```text
final-project/
├── DataSet/
│   ├── Train/
│   ├── Validation/
│   └── Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

## Reproducing Results
To run the training pipeline and reproduce our ablation study (unfreezing the final two transformer blocks and the classifier head), use the following command:

```bash
python src/train.py --config config.json --ablation
```

To run bulk inference on the Test set and generate the quantitative classification metrics, run the evaluation script:

```bash
python src/evaluate.py --data_root /path/to/data
```
*(Replace `/path/to/data` with the actual path to your DAiSEE dataset root).*

## Model Checkpoint & Live Demo
The final fine-tuned model weights and our interactive web UI are hosted on Hugging Face Spaces. You can interact with the live demo here:

[https://huggingface.co/spaces/naalamle/classroom-engagement-monitor](https://huggingface.co/spaces/naalamle/classroom-engagement-monitor)
