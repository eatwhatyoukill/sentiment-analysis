from flask import Flask, request, jsonify
import joblib
import os
import subprocess
import logging

app = Flask(__name__)

MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model():
    try:
        # Activate the virtual environment and run save_model.py to create the model and vectorizer
        venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')
        result = subprocess.run([venv_python, 'save_model.py'], check=True, capture_output=True, text=True)
        logging.info(result.stdout)
        logging.error(result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running save_model.py: {e}")
        logging.error(e.stdout)
        logging.error(e.stderr)
        raise

# Check if the model and vectorizer files exist, if not, create them
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    logging.info("Model or vectorizer file not found. Running save_model.py to create the model and vectorizer.")
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