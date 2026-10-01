import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Hinglish Sentiment", page_icon="✨", layout="centered")

# Custom CSS for a more attractive UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }
    
    /* Header styling */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
        font-family: 'Inter', sans-serif;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E5E7EB !important;
        padding: 15px;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3);
    }
    
    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 16px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        color: white;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .positive { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .negative { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
    .neutral { background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%); }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-conf {
        font-size: 1.2rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">✨ Hinglish Sentiment</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Understand the emotion behind mixed Hindi-English text</p>', unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <h1 style="font-size: 3rem; margin: 0;">💬</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### 💡 Try Examples")
    st.markdown("Click on any sentence below to test the model.")
    
    samples = [
        "ye movie bahut acchi thi 😍",
        "bhai kya bakwas kar rahe ho 😡",
        "main theek hu, kal milte hain 👍",
        "love u bhaijan best pic ❤️",
        "tumhara deemagh kharab hai 🤦"
    ]
    
    for sample in samples:
        if st.button(sample, use_container_width=True):
            st.session_state['text_input'] = sample

# Main Input Section
user_text = st.text_area(
    "Enter your text here:", 
    value=st.session_state.get('text_input', ''), 
    height=140, 
    placeholder="Type something in Hinglish... (e.g. Aaj ka din bahut acha tha!)",
    label_visibility="collapsed"
)

import os

BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    st.error("🚨 Configuration Error: `BACKEND_URL` environment variable is not set. Please configure it in Streamlit Community Cloud.")
    st.stop()

API_URL = f"{BACKEND_URL.rstrip('/')}/predict"

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_btn = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyze. 📝")
    else:
        with st.spinner("🧠 Analyzing sentiment..."):
            try:
                response = requests.post(API_URL, json={"text": user_text}, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get("sentiment")
                    confidence = data.get("confidence", 0.0)
                    
                    # Determine styling based on sentiment
                    if sentiment == "Positive":
                        css_class = "positive"
                        emoji = "✨"
                    elif sentiment == "Negative":
                        css_class = "negative"
                        emoji = "⚠️"
                    else:
                        css_class = "neutral"
                        emoji = "⚖️"
                        
                    # Display results in a custom card
                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div class="result-title">{emoji} {sentiment}</div>
                        <div class="result-conf">Confidence Score: {confidence:.2f} / 1.0</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a spacer
                    st.write("")
                    
                    # Show progress bar for confidence
                    st.progress(float(confidence), text="Model Confidence")
                        
                else:
                    st.error(f"Error from backend: {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to the backend API. Please make sure the Flask server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Built with Flask & Streamlit ❤️</p>", unsafe_allow_html=True)
