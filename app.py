"""
Phase 7 — Deployment (Streamlit web app)

This is a NEW file only. It does not touch Phases 1-6.
It imports and reuses the already-built artifacts instead of
recomputing or retraining anything.

Reused artifacts:
- src/preprocess.py
    -> clean_hinglish_text()       (Phase 2)

- models/vectorizers/tfidf_word_vectorizer.joblib
    -> TF-IDF vectorizer            (Phase 3)

- models/logistic_regression.pkl
    -> trained Logistic Regression  (Phase 4)

- src/utils.py
    -> ID2LABEL                     (label map)

Why Logistic Regression:
reports/final_model_comparison.md shows that Logistic Regression
with word-level TF-IDF (unigram + bigram) is the best-performing
model on the held-out test split.

Macro F1 = 0.6685

Run locally:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================================
# SAMPLE TEXT CALLBACK
# ============================================================================

def set_sample(sample):
    """
    Set the selected sample into the text input widget.

    Streamlit callbacks update session state before the widget
    is instantiated during the rerun.
    """
    st.session_state.text_input = sample


# ============================================================================
# PATH SETUP
# ============================================================================
# Make src/ importable without moving or duplicating project files.
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
MODELS_DIR = APP_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from preprocess import clean_hinglish_text  # noqa: E402
from utils import ID2LABEL  # noqa: E402


MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"

VECTORIZER_PATH = (
    MODELS_DIR
    / "vectorizers"
    / "tfidf_word_vectorizer.joblib"
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Hinglish Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# THEME / STYLING
# ============================================================================

SENTIMENT_STYLE = {
    "positive": {
        "emoji": "😊",
        "color": "#10B981",
        "glow": "rgba(16,185,129,0.35)",
        "label": "Positive",
    },
    "neutral": {
        "emoji": "😐",
        "color": "#F59E0B",
        "glow": "rgba(245,158,11,0.35)",
        "label": "Neutral",
    },
    "negative": {
        "emoji": "😞",
        "color": "#F43F5E",
        "glow": "rgba(244,63,94,0.35)",
        "label": "Negative",
    },
}


# NOTE: This is a pure <style> block (no Markdown syntax), so it is
# rendered with st.html() instead of st.markdown(). st.html() renders
# raw HTML/CSS directly and is not subject to CommonMark's rule that
# text indented 4+ spaces becomes a code block, which was the root
# cause of the literal-HTML-as-text bug.
st.html(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #7C3AED 0%,
            #06B6D4 100%
        );
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(124,58,237,0.25);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }

    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }

    .sample-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: rgba(124,58,237,0.12);
        color: #7C3AED;
        font-size: 0.85rem;
        margin: 0.15rem;
    }

    .metric-card {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        background: rgba(148,163,184,0.08);
        border: 1px solid rgba(148,163,184,0.18);
    }

    .footer-note {
        opacity: 0.6;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }

    </style>
    """
)


# ============================================================================
# CACHED MODEL / VECTORIZER LOADER
# ============================================================================

@st.cache_resource(show_spinner="Loading trained model & vectorizer...")
def load_artifacts():
    """
    Load the trained Logistic Regression model and TF-IDF vectorizer.

    Nothing is trained here.
    """

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():

        raise FileNotFoundError(
            f"Expected trained artifacts at:\n"
            f"  {MODEL_PATH}\n"
            f"  {VECTORIZER_PATH}\n\n"
            f"Make sure the trained model and vectorizer exist."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(text: str, model, vectorizer):
    """
    Complete inference pipeline:

        Raw text
            ↓
        Phase 2 preprocessing
            ↓
        Phase 3 TF-IDF transformation
            ↓
        Phase 4 Logistic Regression
            ↓
        Sentiment + probabilities
    """

    start = time.perf_counter()

    # ------------------------------------------------------------------------
    # Phase 2 preprocessing
    # ------------------------------------------------------------------------

    cleaned = clean_hinglish_text(text)

    # ------------------------------------------------------------------------
    # Phase 3 TF-IDF transformation
    # ------------------------------------------------------------------------

    X = vectorizer.transform([cleaned])

    # ------------------------------------------------------------------------
    # Phase 4 Logistic Regression prediction
    # ------------------------------------------------------------------------

    pred_id = int(model.predict(X)[0])

    # Probability for every class
    proba = model.predict_proba(X)[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    # ------------------------------------------------------------------------
    # Map model class IDs to sentiment labels
    # ------------------------------------------------------------------------

    class_probs = {
        ID2LABEL[int(class_id)]: float(probability)
        for class_id, probability in zip(model.classes_, proba)
    }

    predicted_label = ID2LABEL[pred_id]

    return {
        "cleaned_text": cleaned,
        "sentiment": predicted_label,
        "confidence": class_probs[predicted_label],
        "class_probs": class_probs,
        "elapsed_ms": elapsed_ms,
    }


# ============================================================================
# SAMPLE SENTENCES
# ============================================================================

SAMPLE_SENTENCES = [
    "Yeh movie bahut acchi thi, I loved it! 😍",

    "Bahut bakwas service tha, bilkul bhi pasand nahi aaya 😡",

    "Kal office jaana hai, meeting hai 10 baje",

    "Yaar tu best hai, thank you so much for the help ❤️",

    "Mujhe nahi pata yeh sahi decision hai ya galat",

    "Traffic itna zyada tha ki main office late pahunch gaya, bahut frustrating tha",
]


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("## 🎭 Hinglish Sentiment")

    st.markdown(
        "A code-mixed **Hindi + English** sentiment classifier, "
        "built end-to-end across 7 phases."
    )

    st.markdown("### 📌 Project phases")

    st.markdown(
        """
        - ✅ Phase 1 — Data Understanding
        - ✅ Phase 2 — Preprocessing
        - ✅ Phase 3 — Feature Engineering
        - ✅ Phase 4 — Classical ML
        - ✅ Phase 5 — Deep Learning
        - ✅ Phase 6 — MuRIL Transformer
        - 🚀 Phase 7 — Deployment
        """
    )

    st.markdown("### 🏆 Model in production")

    st.info(
        "**Logistic Regression** on word-level TF-IDF "
        "(unigram + bigram)\n\n"
        "Macro F1 = **0.6685** — best model on the "
        "held-out test split.",
        icon="🏆",
    )

    with st.expander("ℹ️ About this app"):

        st.markdown(
            """
            This is **Phase 7** of the project — a Streamlit
            deployment layer only.

            It does **not** retrain or modify anything from
            Phases 1-6.

            It loads:

            - `src/preprocess.py`
              → text cleaning

            - `models/vectorizers/tfidf_word_vectorizer.joblib`
              → TF-IDF features

            - `models/logistic_regression.pkl`
              → trained classifier

            The app returns:

            - Predicted sentiment
            - Confidence
            - Per-class probabilities
            - Inference time
            """
        )

    with st.expander("🧠 How prediction works"):

        st.markdown(
            """
            **1. Preprocessing**

            The input is cleaned using the exact Phase 2
            preprocessing function.

            **2. TF-IDF**

            The cleaned text is transformed using the saved
            Phase 3 word-level TF-IDF vectorizer.

            The vectorizer uses unigram + bigram features.

            **3. Classification**

            The Phase 4 Logistic Regression model predicts
            the sentiment.

            **4. Output**

            The application displays:

            - Negative probability
            - Neutral probability
            - Positive probability
            - Final predicted sentiment
            """
        )

    st.markdown("---")

    st.caption(
        "Built with Streamlit • scikit-learn • joblib"
    )


# ============================================================================
# HERO
# ============================================================================

# Pure HTML markup (no Markdown syntax) -> st.html() instead of st.markdown().
st.html(
    """
    <div class="hero">

        <h1>🎭 Hinglish Sentiment Analyzer</h1>

        <p>
            Type Hinglish (Hindi + English code-mixed) text and
            get instant sentiment predictions, powered by a
            Logistic Regression model trained on TF-IDF features. 🚀
        </p>

    </div>
    """
)


# ============================================================================
# LOAD MODEL
# ============================================================================

try:

    model, vectorizer = load_artifacts()

except FileNotFoundError as e:

    st.error(
        f"❌ Could not load model artifacts.\n\n{e}"
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ An unexpected error occurred while loading "
        f"the model artifacts:\n\n{e}"
    )

    st.stop()


# ============================================================================
# INITIALIZE TEXT INPUT STATE
# ============================================================================

if "text_input" not in st.session_state:
    st.session_state.text_input = ""


# ============================================================================
# INPUT AREA
# ============================================================================

left, right = st.columns(
    [2, 1],
    gap="large",
)


# ============================================================================
# LEFT COLUMN
# ============================================================================

with left:

    st.markdown("### ✍️ Enter Hinglish text")

    # ------------------------------------------------------------------------
    # TEXT AREA
    # ------------------------------------------------------------------------

    text_input = st.text_area(
        "Your text",
        key="text_input",
        height=140,
        placeholder="e.g. Yeh movie bahut acchi thi, I loved it! 😍",
        label_visibility="collapsed",
    )

    # ------------------------------------------------------------------------
    # SAMPLE BUTTONS
    # ------------------------------------------------------------------------

    st.markdown("**💡 Try a sample:**")

    sample_cols = st.columns(3)

    for i, sample in enumerate(SAMPLE_SENTENCES):

        col = sample_cols[i % 3]

        display_text = (
            sample[:28] + "…"
            if len(sample) > 28
            else sample
        )

        with col:

            st.button(
                display_text,
                key=f"sample_{i}",
                use_container_width=True,
                on_click=set_sample,
                args=(sample,),
            )

    # ------------------------------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------------------------------

    predict_clicked = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True,
    )


# ============================================================================
# RIGHT COLUMN
# ============================================================================

with right:

    st.markdown("### 📊 Class legend")

    for key, style in SENTIMENT_STYLE.items():

        # Pure HTML markup -> st.html()
        st.html(
            f"""
            <div style="
                padding: 0.7rem;
                margin-bottom: 0.5rem;
                border-radius: 10px;
                border: 1px solid {style['color']}55;
                background: {style['color']}12;
            ">

                <span style="font-size:1.4rem;">
                    {style['emoji']}
                </span>

                <span style="
                    font-weight:600;
                    color:{style['color']};
                    margin-left:0.5rem;
                ">
                    {style['label']}
                </span>

            </div>
            """
        )

    st.caption(
        "Trained on code-mixed Hindi-English social-media "
        "text (see Phase 1 report)."
    )


# ============================================================================
# PREDICTION + RESULTS
# ============================================================================

st.markdown("---")


if predict_clicked:

    # ------------------------------------------------------------------------
    # EMPTY INPUT CHECK
    # ------------------------------------------------------------------------

    if not text_input or not text_input.strip():

        st.warning(
            "⚠️ Please enter some text before analyzing — "
            "the input is empty."
        )

    # ------------------------------------------------------------------------
    # CLEANED TEXT EMPTY CHECK
    # ------------------------------------------------------------------------

    elif not clean_hinglish_text(text_input).strip():

        st.warning(
            "⚠️ After cleaning, this text contains no usable "
            "content (for example, only punctuation or links). "
            "Please try different text."
        )

    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------

    else:

        try:

            result = predict_sentiment(
                text_input,
                model,
                vectorizer,
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed.\n\n"
                f"{e}"
            )

            st.stop()

        sentiment_key = result["sentiment"].lower()

        if sentiment_key not in SENTIMENT_STYLE:

            st.error(
                f"❌ Unknown sentiment label returned by model: "
                f"`{result['sentiment']}`"
            )

            st.stop()

        style = SENTIMENT_STYLE[sentiment_key]

        # --------------------------------------------------------------------
        # RESULT HEADING
        # --------------------------------------------------------------------

        st.markdown("### 🎯 Result")

        card_col, metrics_col = st.columns(
            [1.3, 1],
            gap="large",
        )

        # --------------------------------------------------------------------
        # SENTIMENT CARD
        # --------------------------------------------------------------------

        with card_col:

            # Pure HTML markup -> st.html()
            st.html(
                f"""
                <div style="
                    border-radius: 18px;
                    padding: 1.6rem 1.8rem;
                    background: linear-gradient(
                        135deg,
                        {style['color']}22,
                        {style['color']}0D
                    );
                    border: 1px solid {style['color']}55;
                    box-shadow:
                        0 8px 24px {style['glow']};
                ">

                    <div style="font-size:2.4rem;">
                        {style['emoji']}
                    </div>

                    <div style="
                        font-family:'Poppins',sans-serif;
                        font-size:1.6rem;
                        font-weight:700;
                        color:{style['color']};
                    ">
                        {style['label']}
                    </div>

                    <div style="
                        opacity:0.75;
                        margin-top:0.3rem;
                    ">
                        Confidence:
                        <strong>
                            {result['confidence'] * 100:.2f}%
                        </strong>
                    </div>

                </div>
                """
            )

        # --------------------------------------------------------------------
        # METRICS
        # --------------------------------------------------------------------

        with metrics_col:

            m1, m2 = st.columns(2)

            with m1:
                # Pure HTML markup -> st.html()
                st.html(
                    f"""
                    <div class="metric-card">

                        <div style="
                            opacity:0.6;
                            font-size:0.8rem;
                        ">
                            CONFIDENCE
                        </div>

                        <div style="
                            font-size:1.4rem;
                            font-weight:700;
                        ">
                            {result['confidence'] * 100:.1f}%
                        </div>

                    </div>
                    """
                )

            with m2:
                # Pure HTML markup -> st.html()
                st.html(
                    f"""
                    <div class="metric-card">

                        <div style="
                            opacity:0.6;
                            font-size:0.8rem;
                        ">
                            PREDICTION TIME
                        </div>

                        <div style="
                            font-size:1.4rem;
                            font-weight:700;
                        ">
                            {result['elapsed_ms']:.2f} ms
                        </div>

                    </div>
                    """
                )

            st.caption(
                "Cleaned text used for prediction: "
                f"_{result['cleaned_text']}_"
            )

        # --------------------------------------------------------------------
        # PROBABILITY CHART
        # --------------------------------------------------------------------

        st.markdown("#### 📈 Probability by class")

        # Keep the display order consistent.
        display_order = [
            "negative",
            "neutral",
            "positive",
        ]

        probability_rows = []

        for sentiment in display_order:

            if sentiment in result["class_probs"]:

                probability_rows.append(
                    {
                        "Sentiment": SENTIMENT_STYLE[
                            sentiment
                        ]["label"],
                        "Probability": (
                            result["class_probs"][sentiment] * 100
                        ),
                    }
                )

        prob_df = pd.DataFrame(
            probability_rows
        ).set_index("Sentiment")

        st.bar_chart(
            prob_df,
            y="Probability",
            color="#7C3AED",
            use_container_width=True,
        )

        # --------------------------------------------------------------------
        # INDIVIDUAL CLASS PROBABILITIES
        # --------------------------------------------------------------------

        st.markdown("#### 📊 Class probabilities")

        prob_cols = st.columns(3)

        for i, sentiment in enumerate(display_order):

            if sentiment not in result["class_probs"]:
                continue

            probability = result["class_probs"][sentiment]

            sentiment_style = SENTIMENT_STYLE[sentiment]

            with prob_cols[i]:
                # Pure HTML markup -> st.html()
                st.html(
                    f"""
                    <div class="metric-card"
                         style="text-align:center;">

                        <div style="font-size:1.3rem;">
                            {sentiment_style['emoji']}
                        </div>

                        <div style="
                            font-weight:600;
                            color:{sentiment_style['color']};
                        ">
                            {sentiment_style['label']}
                        </div>

                        <div style="
                            font-size:1.2rem;
                            font-weight:700;
                        ">
                            {probability * 100:.2f}%
                        </div>

                    </div>
                    """
                )


# ============================================================================
# DEFAULT MESSAGE
# ============================================================================

else:

    st.info(
        "👆 Enter Hinglish text (or click a sample) and hit "
        "**Analyze Sentiment** to see results."
    )


# ============================================================================
# FOOTER
# ============================================================================

# Pure HTML markup -> st.html()
st.html(
    """
    <div class="footer-note">

        Phase 7 · Deployment layer ·
        Model: Logistic Regression (TF-IDF) ·
        Reuses Phases 1-6 artifacts without modification.

    </div>
    """
)