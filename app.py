from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os  # <-- Add this for Heroku port

app = Flask(__name__)

# ✅ Load the model and tokenizer directly from HuggingFace Hub
model = AutoModelForSequenceClassification.from_pretrained("Canthoughts/distilbert-emotion-api")
tokenizer = AutoTokenizer.from_pretrained("Canthoughts/distilbert-emotion-api")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return jsonify({'emotion': classes[pred]})

if __name__ == '__main__':
    # ✅ Important change for Heroku deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
