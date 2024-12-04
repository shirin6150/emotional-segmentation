
# Hate Speech Detection using Machine Learning

This project aims to build a machine learning model to classify whether a given text contains hate speech or not. The model is trained using `Logistic Regression` and `CountVectorizer` for text vectorization. The trained model and vectorizer are saved as pickle files, which can be loaded later to make predictions on new, unseen text inputs.

## Project Overview

This project uses natural language processing (NLP) techniques to classify text as either `hate speech` or `non-hate speech`. The model is trained on a dataset consisting of labeled text data, and it outputs a binary classification (1 for hate speech, 0 for non-hate speech).

## Technologies Used

- Python
- Scikit-learn
- Pickle (for saving and loading models)
- CountVectorizer (for text vectorization)
- Logistic Regression (for classification)

## Installation

To run this project on your local machine, you need to have Python 3.x installed. You also need to install the required dependencies. You can install them by following these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/hate-speech-detection.git
   cd hate-speech-detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Requirements file (`requirements.txt`)**:
   You can create a `requirements.txt` with the following dependencies:
   ```
   scikit-learn==1.2.2
   numpy==1.23.0
   pickle5==0.0.11
   ```

## Model Training and Saving

The model is trained on a text dataset where each text sample is labeled as `hate speech` or `non-hate speech`. The training process involves:

1. **Text Vectorization**: The `CountVectorizer` is used to convert the input text into a numeric format that the machine learning model can process.
2. **Model Training**: A `Logistic Regression` model is trained on the vectorized data.
3. **Saving the Model and Vectorizer**: The trained model and vectorizer are saved using the `pickle` library, which allows you to reuse them for future predictions.

The model and vectorizer are saved in the following files:
- `hate_speech_model.pkl` (The trained machine learning model)
- `vectorizer.pkl` (The text vectorizer used during training)

## Prediction

Once the model and vectorizer are saved, you can load them and use them to make predictions on new text data. The model will classify the input text as either `hate speech` or `non-hate speech`.


