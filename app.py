from flask import Flask, request, jsonify
import joblib
import os
import logging
from save_model import save_model

app = Flask(__name__)

MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model():
    try:
        # Run save_model function to create the model and vectorizer
        save_model()
    except Exception as e:
        logging.error(f"An error occurred while running save_model: {e}")
        raise

# Check if the model and vectorizer files exist, if not, create them
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    logging.info("Model or vectorizer file not found. Running save_model to create the model and vectorizer.")
    create_model()

# Load the model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    sentiment = 'Positive' if prediction[0] == 4 else 'Negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)