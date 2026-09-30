import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Hinglish Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Hinglish Sentiment Analysis")
st.markdown("""
Welcome to the Hinglish Sentiment Analysis project! 
This tool analyzes text written in **Hinglish** (a mix of Hindi and English) and determines whether the sentiment is **Positive**, **Neutral**, or **Negative**.
""")

# Sample Sentences
st.sidebar.header("Try these examples:")
samples = [
    "ye movie bahut acchi thi",
    "bhai kya bakwas kar rahe ho",
    "main theek hu, kal milte hain",
    "love u bhaijan best pic",
    "tumhara deemagh kharab hai"
]

for sample in samples:
    if st.sidebar.button(sample):
        st.session_state['text_input'] = sample

st.subheader("Enter your Hinglish text below:")
# Get text from user
user_text = st.text_area(
    "", 
    value=st.session_state.get('text_input', ''), 
    height=100, 
    placeholder="Type something in Hinglish..."
)

API_URL = "http://localhost:5000/predict"

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Send request to Flask backend
                response = requests.post(
                    API_URL, 
                    json={"text": user_text},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get("sentiment")
                    confidence = data.get("confidence")
                    
                    # Display results with colors
                    st.markdown("### Result:")
                    
                    if sentiment == "Positive":
                        st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                    elif sentiment == "Negative":
                        st.error(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                    else:
                        st.info(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                        
                else:
                    st.error(f"Error from backend: {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Please make sure the Flask server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("DMT Project | Built with Flask & Streamlit")
