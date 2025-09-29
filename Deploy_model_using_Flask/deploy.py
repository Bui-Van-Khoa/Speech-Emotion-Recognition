from flask import Flask, render_template, request
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

app = Flask(__name__) 

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=7
)
model.load_state_dict(torch.load("wav2vec2_finetuned.pth"))


@app.route('/')
def home():
    result = 'No result'
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])

def predict():
    inverse_label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
    file = request.files['audio']
    file.save("uploaded.wav")

    speech, sr = librosa.load('./uploaded.wav', sr=16000)

    # Pad or truncate
    if len(speech) > 32000:
        speech = speech[:32000]
    else:
        speech = np.pad(speech, (0, 32000 - len(speech)), 'constant')

    # Preprocess with the processor
    inputs =processor(
    speech,
    sampling_rate=16000,
    return_tensors='pt'
    )

    input_values = inputs.input_values.squeeze(1)

    with torch.no_grad():
        outputs = model(input_values)

    logits = outputs.logits
    predicted_class = logits.argmax(dim = -1).item()
    result = inverse_label_map[predicted_class]

    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
    