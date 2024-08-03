# save_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import os
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model():
    # Log the start of execution
    logging.info("Execution Begins:")

    # Retrieve secrets from environment variables
    try:
        kaggle_username = os.environ['KAGGLE_USERNAME']
        kaggle_key = os.environ['KAGGLE_KEY']
        logging.info("Successfully retrieved Kaggle credentials from environment variables")
    except KeyError as e:
        logging.error(f"Environment variable not set: {e}")
        exit(1)

    # Define kaggle.json path
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Create the kaggle.json file programmatically if it doesn't exist
    if not os.path.exists(kaggle_json_path):
        os.makedirs(kaggle_dir, exist_ok=True)
        logging.info(f"Creating kaggle.json at {kaggle_json_path}")

        kaggle_config = {
            "username": kaggle_username,
            "key": kaggle_key
        }

        with open(kaggle_json_path, "w") as f:
            json.dump(kaggle_config, f)
        os.chmod(kaggle_json_path, 0o600)  # Set file permissions to read/write for owner only

        # Verify the kaggle.json file was created
        if not os.path.exists(kaggle_json_path):
            logging.error(f"Failed to create {kaggle_json_path}")
            exit(1)
        else:
            logging.info(f"kaggle.json created successfully at {kaggle_json_path}")
    else:
        logging.info(f"kaggle.json already exists at {kaggle_json_path}")

    # Authenticate Kaggle API
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    logging.info("Kaggle API authenticated successfully")

    # Function to download dataset with retries
    def download_dataset_with_retries(api, dataset, path, retries=3, delay=5):
        for attempt in range(retries):
            try:
                api.dataset_download_files(dataset, path=path, unzip=True)
                logging.info("Dataset downloaded successfully.")
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("All attempts to download the dataset failed.")
                    return False

    # Download dataset if not already downloaded
    dataset = 'kazanova/sentiment140'
    dest_folder = 'data'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    csv_file_path = os.path.join(dest_folder, 'training.1600000.processed.noemoticon.csv')
    if not os.path.exists(csv_file_path):
        logging.info("Dataset not found. Downloading.....")
        if not download_dataset_with_retries(api, dataset, dest_folder):
            exit(1)
    else:
        logging.info("Dataset already exists. Skipping download")

    # Load and prepare data
    data = pd.read_csv(csv_file_path, encoding='latin-1', names=["target", "ids", "date", "flag", "user", "text"])
    data = data[['target', 'text']]

    # Train model
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(data['text'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save model and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    logging.info("Model saved to sentiment_model.pkl")
    logging.info("Vectorizer saved to vectorizer.pkl")

if __name__ == "__main__":
    save_model()