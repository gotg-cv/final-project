# Classroom Engagement Monitor

## Motivation
In educational environments, understanding student engagement is crucial for improving pedagogical approaches and educational outcomes. However, there is a significant lack of localized datasets and frameworks tailored to these environments. This project aims to bridge that gap by building a robust framework to monitor classroom engagement.

## Architecture
The system leverages the `videomae-base-finetuned-kinetics` model as its foundational architecture. VideoMAE uses masked autoencoders for self-supervised video pre-training, making it highly effective for video classification tasks. We will eventually fine-tune this model to classify four affective states: Boredom, Confusion, Engagement, and Frustration.

## Pipeline
1. **Data Preprocessing**: Extracting and organizing video frames from the DAiSEE dataset, ensuring appropriate formatting for VideoMAE inputs.
2. **Fine-Tuning**: Adapting the `videomae-base-finetuned-kinetics` model on the dataset to specialize in classifying the four affective states.
3. **Inference**: Running the fine-tuned model on live webcam streams or video files to predict real-time student engagement.
