from flask import Flask, request, jsonify
import joblib
import os
import sys
from pathlib import Path

# Setup robust paths
# backend/app.py is in the backend/ directory, so root is its parent.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

# Add PROJECT_ROOT to sys.path so we can import src as a module
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.preprocess import clean_hinglish_text
    from src.utils import ID2LABEL
except ImportError as e:
    # Fallback to prevent immediate crash if imports fail, though it should be fatal
    print(f"Error importing from src/: {e}")
    clean_hinglish_text = None
    ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

app = Flask(__name__)

# Load model and vectorizer at startup
model_path = MODELS_DIR / "logistic_regression.pkl"
vectorizer_path = MODELS_DIR / "vectorizers" / "tfidf_word_vectorizer.joblib"

MODEL = None
VECTORIZER = None

try:
    if model_path.exists() and vectorizer_path.exists():
        MODEL = joblib.load(model_path)
        VECTORIZER = joblib.load(vectorizer_path)
        print("Model and Vectorizer loaded successfully.")
    else:
        print(f"Warning: Model or Vectorizer not found at expected paths.")
        print(f"Model path: {model_path}")
        print(f"Vectorizer path: {vectorizer_path}")
except Exception as e:
    print(f"Error loading artifacts: {e}")

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "running",
        "message": "Hinglish Sentiment Analysis API is up and running."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None or VECTORIZER is None:
        return jsonify({"error": "Model artifacts failed to load on server startup."}), 500

    try:
        # Strict parsing of JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON."}), 400
            
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid request. Please provide 'text' in JSON body."}), 400
            
        text = data.get('text', '')
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Text cannot be empty."}), 400
            
        # 1. Clean the text (handle missing preprocessing gracefully)
        if clean_hinglish_text:
            cleaned_str = clean_hinglish_text(text)
        else:
            return jsonify({"error": "Preprocessing module unavailable."}), 500
        
        if not cleaned_str.strip():
             return jsonify({"error": "Text contains no valid words after preprocessing."}), 400
             
        # 2. Vectorize
        vectorized_text = VECTORIZER.transform([cleaned_str])
        
        # 3. Predict sentiment and confidence
        prediction_idx = int(MODEL.predict(vectorized_text)[0])
        probabilities = MODEL.predict_proba(vectorized_text)[0]
        confidence = float(max(probabilities))
        
        sentiment_label = ID2LABEL.get(prediction_idx, "Unknown")
        
        return jsonify({
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4)
        })
        
    except Exception as e:
        # Catch unexpected errors gracefully
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    # Listen on 0.0.0.0 for deployment
    app.run(host='0.0.0.0', port=5000, debug=False)
