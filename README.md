# 🎭 Hinglish Sentiment Analysis

A machine learning project for **sentiment analysis of Hinglish (Hindi + English code-mixed) social-media text**.

The project follows an end-to-end pipeline covering data understanding, preprocessing, feature engineering, classical machine learning, deep learning, transformer-based modelling, model comparison, and Streamlit deployment.

## 🚀 Live Demo

**Try the deployed application:**

https://hinglish-sentiment-analysis-gf5xtjsj3umzoqpxe4eovk.streamlit.app/

Enter a Hinglish sentence and get:

- 😊 Positive
- 😐 Neutral
- 😞 Negative
- Confidence score
- Per-class probabilities
- Prediction/inference time

---

## 📌 Project Overview

Hinglish is a code-mixed language where Hindi and English are commonly used together, especially on social-media platforms.

For example:

```text
"Yaar ye movie ekdum zabardast thi!"
```

Traditional sentiment-analysis approaches can struggle with:

- Hindi-English code mixing
- Informal spelling
- Social-media language
- Slang
- Abbreviations
- Context-dependent expressions
- Informal punctuation and emojis

This project investigates different machine-learning approaches for classifying Hinglish text into three sentiment classes:

```text
Negative
Neutral
Positive
```

---

# 🧠 Project Pipeline

The project was developed in multiple phases:

```text
Phase 1
Data Understanding
        ↓
Phase 2
Text Preprocessing
        ↓
Phase 3
Feature Engineering
        ↓
Phase 4
Classical Machine Learning
        ↓
Phase 5
Deep Learning
        ↓
Phase 6
MuRIL Transformer
        ↓
Model Comparison
        ↓
Phase 7
Streamlit Deployment
```

---

# 📚 Project Phases

## Phase 1 — Data Understanding

The dataset was inspected to understand:

- Dataset structure
- Sentiment labels
- Class distribution
- Text characteristics
- Train/validation/test splits
- Code-mixed Hindi-English text

The three sentiment classes are:

| Label | Meaning  |
| ----- | -------- |
| `0`   | Negative |
| `1`   | Neutral  |
| `2`   | Positive |

---

## Phase 2 — Preprocessing

A dedicated preprocessing pipeline was developed in:

```text
src/preprocess.py
```

The same preprocessing function is reused during deployment to ensure that new user input follows the same processing pipeline used during model development.

The preprocessing pipeline handles operations such as:

- Lowercasing
- URL removal
- Mention handling
- HTML removal
- Hashtag handling
- Punctuation normalization
- Whitespace normalization
- Preservation of useful text information

Main function:

```python
clean_hinglish_text()
```

---

## Phase 3 — Feature Engineering

TF-IDF was used to convert cleaned text into numerical features.

The deployed vectorizer uses:

```text
Word-level TF-IDF
Unigrams + Bigrams
```

The trained vectorizer is stored as:

```text
models/vectorizers/tfidf_word_vectorizer.joblib
```

---

## Phase 4 — Classical Machine Learning

Several classical machine-learning approaches were evaluated.

The final production model selected for deployment is:

### 🏆 Logistic Regression + Word-level TF-IDF

Configuration:

```text
TF-IDF
├── Unigrams
└── Bigrams

        ↓

Logistic Regression
```

### Final Test Performance

**Macro F1: 0.6685**

The Logistic Regression model was selected because it achieved the best performance among the evaluated models on the held-out test split.

---

## Phase 5 — Deep Learning

Deep-learning approaches were also explored as part of the project.

The purpose of this phase was to compare traditional feature-based machine learning with neural approaches for Hinglish sentiment classification.

---

## Phase 6 — MuRIL Transformer

A transformer-based approach using **MuRIL** was evaluated for multilingual and Indian-language text understanding.

The transformer experiment was performed separately from the lightweight deployment pipeline.

After comparing the evaluated approaches, the Logistic Regression + TF-IDF model was selected for production deployment based on the final held-out evaluation.

---

# 🚀 Phase 7 — Deployment

The final model was deployed as an interactive **Streamlit web application**.

The deployment application does not retrain the model.

Instead, it loads the already-trained artifacts:

```text
User Input
    ↓
clean_hinglish_text()
    ↓
Saved TF-IDF Vectorizer
    ↓
Saved Logistic Regression Model
    ↓
Prediction
    ↓
Sentiment + Confidence + Probabilities
```

This keeps the deployment lightweight and avoids including the complete training environment or dataset.

---

# 📁 Deployment Project Structure

```text
hinglish-sentiment-analysis/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── logistic_regression.pkl
│   │
│   └── vectorizers/
│       └── tfidf_word_vectorizer.joblib
│
└── src/
    ├── preprocess.py
    └── utils.py
```

### Important

The deployment repository intentionally does **not** contain the original large training dataset or unnecessary training dependencies.

Only the artifacts required for inference are included.

---

# 🖥️ Application Features

The Streamlit application provides:

### ✍️ Hinglish Text Input

Users can enter any Hindi-English code-mixed sentence.

### 🎯 Sentiment Prediction

The application predicts:

```text
😊 Positive
😐 Neutral
😞 Negative
```

### 📊 Confidence

Displays the probability associated with the predicted class.

### 📈 Class Probabilities

Shows the model's probability distribution across:

- Negative
- Neutral
- Positive

### ⚡ Inference Time

Displays the time required to process the input and generate the prediction.

### 💡 Sample Inputs

The application includes predefined Hinglish examples for quick testing.

---

# 🛠️ Technology Stack

| Component              | Technology                |
| ----------------------- | -------------------------- |
| Programming Language   | Python                    |
| Machine Learning       | scikit-learn              |
| Feature Extraction     | TF-IDF                    |
| Classifier             | Logistic Regression       |
| Transformer Experiment | MuRIL                     |
| Web Framework          | Streamlit                 |
| Model Serialization    | Joblib                    |
| Data Processing        | Pandas                    |
| Deployment             | Streamlit Community Cloud |
| Version Control        | Git / GitHub              |

---

# ⚙️ Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/neevthakor/hinglish-sentiment-analysis.git
```

```bash
cd hinglish-sentiment-analysis
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Example Inputs

### Positive

```text
Yaar tu best hai, thank you so much for the help ❤️
```

### Negative

```text
Bahut bakwas service tha, bilkul bhi pasand nahi aaya.
```

### Neutral

```text
Kal office jaana hai, meeting hai 10 baje.
```

### Code-mixed

```text
Movie ka starting accha tha but ending bilkul boring thi.
```

---

# ⚠️ Model Limitations

The deployed model is a classical **TF-IDF + Logistic Regression** system.

Because TF-IDF primarily represents lexical patterns, the model has limitations when dealing with expressions where sentiment depends heavily on context.

Examples include:

- Friendly use of abusive/slang words
- Sarcasm
- Irony
- Context-dependent expressions
- Rare Hinglish phrases
- Emoji-dependent sentiment
- Informal code-mixed expressions

For example, a sentence containing a strongly negative word may still express positive sentiment depending on the surrounding context.

This is a known limitation of lexical feature-based sentiment classification and represents an opportunity for future improvement using richer contextual representations and better training data.

---

# 🔮 Future Improvements

Possible future improvements include:

- Character-level TF-IDF features
- Word + character feature combinations
- Better handling of emojis
- Improved Hinglish normalization
- More Hinglish-specific training data
- Data augmentation
- Class balancing
- Context-aware transformer models
- Fine-tuned multilingual/Indic-language transformers
- Error-driven dataset improvement
- Explainable prediction with feature-level analysis

---

# 📊 Final Model

The model deployed in production is:

```text
Word-level TF-IDF
        +
Unigrams + Bigrams
        ↓
Logistic Regression
```

### Macro F1 Score

```text
0.6685
```

The score was obtained on the held-out test split used for final model comparison.

---

# 👨‍💻 Author

**Neev Thakor**

Computer Science & Engineering — Data Science

GitHub:

[https://github.com/neevthakor](https://github.com/neevthakor)

---

# 🔗 Links

### GitHub Repository

[https://github.com/neevthakor/hinglish-sentiment-analysis](https://github.com/neevthakor/hinglish-sentiment-analysis)

### Live Streamlit Application

[https://hinglish-sentiment-analysis-gf5xtjsj3umzoqpxe4eovk.streamlit.app/](https://hinglish-sentiment-analysis-gf5xtjsj3umzoqpxe4eovk.streamlit.app/)

---

## 📌 Project Status

**Phase 7 — Deployment: ✅ Complete**

The trained Hinglish sentiment-analysis model is successfully deployed as a Streamlit web application.
