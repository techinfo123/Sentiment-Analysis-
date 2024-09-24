from flask import Flask, request, render_template
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model and tokenizer
model_path = 'C:/Users/Atishay/render-demo/model'  # Path to the model folder

# Verify the model folder exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model folder not found at: {model_path}")

# Load the model and tokenizer using from_pretrained
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    input_text = request.form['text']
    
    # Tokenize the input text
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
    # Predict using the trained model
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**encoded_input)
    
    # Get predicted class
    predictions = outputs.logits.argmax(dim=-1)
    
    # Interpret the result (adjust according to your model's output)
    output = 'Positive' if predictions.item() == 1 else 'Negative'

    return render_template('index.html', prediction_text='Sentiment: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
