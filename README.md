# Hinglish Sentiment Analysis

## Introduction
This project analyzes text written in **Hinglish** (a mix of Hindi and English) to determine the sentiment of the text. It is built as a clean, modular project suitable for a college Data Mining and Techniques (DMT) course. 

The objective of this project is to classify Hinglish text into three categories:
* **Positive** (Label: 2)
* **Neutral** (Label: 1)
* **Negative** (Label: 0)

## Dataset
The project uses the `hinglish_sentiment.csv` dataset, which contains Hinglish text entries and their corresponding sentiment labels. 

## Methodology
The overall pipeline follows these steps:
`Dataset ↓ Preprocessing ↓ TF-IDF ↓ Logistic Regression ↓ Prediction`

### Preprocessing
Before training the model, the raw text goes through several cleaning steps in `preprocess.py`:
1. **Lowercase Conversion**: All text is converted to lowercase for consistency.
2. **URL & HTML Removal**: Links and HTML tags are stripped from the text.
3. **Punctuation Removal**: Special characters are removed while preserving words and spaces.
4. **Whitespace Cleaning**: Extra spaces are removed to keep the text clean.

### TF-IDF (Term Frequency-Inverse Document Frequency)
Text cannot be directly fed into a machine learning model, so it must be converted into numerical format. **TF-IDF** does this by calculating how important a word is to a specific sentence relative to the entire dataset. It highlights important words while downweighting common words like "hai", "ki", etc.

### Logistic Regression
We use **Logistic Regression**, a simple yet powerful classification algorithm. It learns the relationship between the numerical TF-IDF features and the sentiment labels to predict the sentiment of new, unseen text.

## Project Structure
The project is strictly separated by responsibilities:
* `dataset/`: Contains the training data (`hinglish_sentiment.csv`).
* `training/`: Contains all ML code (`preprocess.py`, `train.py`, `evaluate.py`).
* `model/`: Stores the trained ML model and TF-IDF vectorizer.
* `backend/`: Contains the Flask API (`app.py`) for making predictions.
* `frontend/`: Contains the Streamlit web interface (`app.py`).

## How to Run the Project

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
This will read the dataset, train the Logistic Regression model, and save it in the `model/` folder.
```bash
python training/train.py
```

### 3. Evaluate the Model
This evaluates the trained model on the test split and displays Accuracy, Precision, Recall, F1-score, and the Confusion Matrix.
```bash
python training/evaluate.py
```

### 4. Run the Backend (Flask API)
The backend loads the trained model and serves predictions. Leave this running in a terminal.
```bash
python backend/app.py
```

### 5. Run the Frontend (Streamlit UI)
The frontend provides a simple UI to test the model. Run this in a **new terminal**.
```bash
streamlit run frontend/app.py
```

## Example
If you enter: *"ye movie bahut acchi thi"* in the UI, the frontend sends it to the backend.
The backend returns a JSON response like this:
```json
{
  "text": "ye movie bahut acchi thi",
  "sentiment": "Positive",
  "confidence": 0.9123
}
```
The UI then displays **Positive** sentiment to the user.

## Limitations
While the model performs well on standard text, it has real-world limitations:
* **Sarcasm**: It struggles to detect sarcastic tones.
* **Slang & Spelling Variations**: Hinglish has no standard spelling (e.g., "acha", "accha", "achha"), which can confuse the model.
* **Context**: It evaluates sentences in isolation without knowing previous conversational context.
* **Ambiguous Expressions**: Mixed sentiments in a single sentence can result in lower confidence scores.
