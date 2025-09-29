# Wav2Vec2 Emotion Classifier

This project fine-tunes the pretrained Facebook Wav2Vec2-base model to classify spoken audio into seven emotions: Neutral, Happy, Sad, Angry, Fear, Surprise, and Disgust.

## Dataset
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

## Features
- Uses Hugging Face Transformers for model training and evaluation.
- Evaluates and saves checkpoints at the end of each epoch.
- Deploy trained model using Flask

## Output
The trained model predicts one of the seven emotion categories for any input speech sample
