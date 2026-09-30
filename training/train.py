import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

from preprocess import clean_text

def train_model():
    print("Loading dataset...")
    # Path is relative to the project root
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hinglish_sentiment.csv')
    df = pd.read_csv(dataset_path)
    
    # Ensure text and label columns exist (assuming 'text' and 'label' based on the dataset preview)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows with empty text after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Basic evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    print("Saving model and vectorizer...")
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'sentiment_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    
    print("Training complete. Model saved in the 'model' directory.")

if __name__ == "__main__":
    train_model()
