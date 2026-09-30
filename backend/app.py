from flask import Flask, request, jsonify
import joblib
import os
import sys

# Add the project root to sys.path to import from training
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    from training.preprocess import clean_text
except ImportError:
    # Fallback in case of weird execution paths
    def clean_text(text):
        import re
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

app = Flask(__name__)

# Load model and vectorizer at startup
model_dir = os.path.join(project_root, 'model')
model_path = os.path.join(model_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

MODEL = None
VECTORIZER = None

try:
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        MODEL = joblib.load(model_path)
        VECTORIZER = joblib.load(vectorizer_path)
        print("Model and Vectorizer loaded successfully.")
    else:
        print("Warning: Model or Vectorizer not found. Prediction will fail.")
except Exception as e:
    print(f"Error loading model: {e}")

# Label mapping
SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "running", "message": "Hinglish Sentiment Analysis API is up and running."})

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None or VECTORIZER is None:
        return jsonify({"error": "Model or vectorizer not found. Please train the model first."}), 500

    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid request. Please provide 'text' in JSON body."}), 400
            
        text = data.get('text', '')
        
        if not isinstance(text, str) or text.strip() == '':
            return jsonify({"error": "Text cannot be empty."}), 400
            
        # 1. Clean the text
        cleaned_str = clean_text(text)
        
        if cleaned_str == '':
             return jsonify({"error": "Text contains no valid words after preprocessing."}), 400
             
        # 2. Vectorize
        vectorized_text = VECTORIZER.transform([cleaned_str])
        
        # 3. Predict sentiment and confidence
        prediction = MODEL.predict(vectorized_text)[0]
        probabilities = MODEL.predict_proba(vectorized_text)[0]
        confidence = float(max(probabilities))
        
        sentiment_label = SENTIMENT_MAP.get(int(prediction), "Unknown")
        
        return jsonify({
            "text": text,
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
