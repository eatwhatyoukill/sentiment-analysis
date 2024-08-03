# Sentiment Analysis Tool

![Build Status](https://github.com/eatwhatyoukill/sentiment-analysis/actions/workflows/main_sentimentAnalysis30081996.yml/badge.svg)
![License](https://img.shields.io/github/license/eatwhatyoukill/sentiment-analysis)

This project is a Sentiment Analysis Tool that classifies text into positive or negative sentiment using machine learning. It provides a RESTful API built with Flask and uses scikit-learn for model training.

## Introduction

Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. This tool leverages machine learning to analyze the sentiment of text data.

## Prerequisites

- Python 3.10
- Virtual Environment (venv)
- Kaggle account with API credentials

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/eatwhatyoukill/sentiment-analysis.git
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # For Windows
    source venv/bin/activate  # For Linux/Mac
    ```

3. Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Kaggle API credentials:
    - Ensure you have your Kaggle API key saved in a `kaggle.json` file in the `.kaggle` directory within your home directory.

## Usage

To run the application, use the following command:

```bash
python app.py
```
You can send a POST request to the `/predict` endpoint with the text you want to analyze. For example, using curl:

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"I love this product!\"}" http://127.0.0.1:5000/predict
```

This will return a JSON response indicating whether the sentiment is positive or negative.

## Features

- **Text Classification**: Classifies text into positive or negative sentiment.
- **RESTful API**: Provides a RESTful API using Flask.
- **Model Training**: Trains a machine learning model using scikit-learn.

## Technologies Used

- **Python**: The primary programming language used for development.
- **Flask**: A lightweight WSGI web application framework used to create the API.
- **scikit-learn**: A machine learning library used to build and train the sentiment analysis model.
- **joblib**: Used for model serialization and deserialization.
- **Kaggle API**: Used to download the dataset from Kaggle.

## Azure Deployment

This project is configured to be deployed on Azure App Service using GitHub Actions.

### GitHub Actions Workflow

The GitHub Actions workflow file is located at `.github/workflows/main_sentimentAnalysis30081996.yml`. It defines the steps to build and deploy the application to Azure.

### Secrets

Ensure the following secrets are added to your GitHub repository:

- `AZUREAPPSERVICE_CLIENTID`
- `AZUREAPPSERVICE_TENANTID`
- `AZUREAPPSERVICE_SUBSCRIPTIONID`

### Deploying to Azure

1. Commit and push your changes to the `main` branch.
2. The GitHub Actions workflow will automatically trigger and deploy the application to Azure.

## Contributing
[Report an Issue or Request a Feature](https://github.com/eatwhatyoukill/sentiment-analysis/issues/new?assignees=&labels=&template=issue-template.md&title=)

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is sourced from Kaggle.
- Special thanks to the contributors of the libraries and tools used in this project.

## Contact

If you have any questions or suggestions, feel free to reach out.