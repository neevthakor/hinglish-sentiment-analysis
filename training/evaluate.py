import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
from preprocess import clean_text

def evaluate_model():
    print("Loading dataset for evaluation...")
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hinglish_sentiment.csv')
    df = pd.read_csv(dataset_path)
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    
    print("Splitting dataset to retrieve test set...")
    # Must use the exact same random_state and stratify as in train.py
    _, X_test, _, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print("Loading model and vectorizer...")
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    try:
        model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    except FileNotFoundError:
        print("Model or vectorizer not found. Please run train.py first.")
        return

    print("Vectorizing test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test_tfidf)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    acc = accuracy_score(y_test, y_pred)
    # Use weighted or macro average for multiclass
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (Macro)")
    print(f"Recall:    {recall:.4f} (Macro)")
    print(f"F1-score:  {f1:.4f} (Macro)")
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    target_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(confusion_matrix(y_test, y_pred))
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
