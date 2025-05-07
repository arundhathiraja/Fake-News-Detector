import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime


# ======================
# 1. TEXT PROCESSING MODULE
# ======================
def clean_text(text):
    """Preprocess text for analysis"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def contains_sensitive_keywords(text):
    """Check for controversial terms"""
    sensitive_words = [
        'fake', 'hoax', 'conspiracy', 'lie', 'false',
        'propaganda', 'deceive', 'mislead', 'scam'
    ]
    return [word for word in sensitive_words if word in text.lower()]


# ======================
# 2. NEWS FETCHER MODULE
# ======================
def get_current_news():
    """Fetch live news headlines from Google News"""
    try:
        url = "https://news.google.com/home?hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')][:15]  # Get top 15 headlines
        return headlines if headlines else ["Could not fetch live news. Please enter text manually."]
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return ["Could not fetch live news. Please enter text manually."]


# ======================
# 3. MODEL MODULE
# ======================
class FakeNewsModel:
    def __init__(self):
        """Initialize with a simple TF-IDF + Logistic Regression model"""
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)

        # Mock training data (REPLACE WITH YOUR ACTUAL TRAINING DATA)
        self.train_model()

    def train_model(self):
        """Train on sample data - REPLACE WITH YOUR DATASET"""
        X_train = [
            "This is a true news article about climate change",
            "The president made an official statement today",
            "Scientists confirm new discovery",
            "This is completely fake information",
            "Viral hoax spreads online",
            "Debunked conspiracy theory resurfaces"
        ]
        y_train = ['real', 'real', 'real', 'fake', 'fake', 'fake']

        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)

    def predict(self, text):
        """Make prediction on cleaned text"""
        try:
            cleaned = clean_text(text)
            X = self.vectorizer.transform([cleaned])
            pred = self.model.predict(X)[0]
            proba = np.max(self.model.predict_proba(X))
            return pred, float(proba)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return "error", 0.0


# ======================
# 4. STREAMLIT APP
# ======================
def main():
    st.set_page_config(page_title="Fake News Detector", layout="wide")

    # Initialize model (load or train)
    if 'model' not in st.session_state:
        st.session_state.model = FakeNewsModel()

    st.title("🔍 Fake News Detector")
    st.markdown("Analyze news articles or headlines for potential misinformation")

    # Input section
    news = st.text_area("Paste news article here:", height=150)
    use_live_news = st.checkbox("Or select from live news headlines")

    if use_live_news:
        current_news = get_current_news()
        selected_news = st.selectbox("Select headline:", current_news)
        news = selected_news if not news else news

    # Analysis button
    if st.button("Analyze Text", type="primary"):
        if not news.strip():
            st.warning("Please enter news text or select a headline")
            return

        with st.spinner("Analyzing..."):
            # Text processing
            cleaned_news = clean_text(news)

            # Model prediction
            label, confidence = st.session_state.model.predict(news)
            confidence_percent = round(confidence * 100, 2)

            # Sensitive keywords check
            sensitive_words = contains_sensitive_keywords(news)

            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                if label == 'real':
                    st.success(f"✅ Likely REAL News")
                else:
                    st.error(f"❌ Potential FAKE News")

                st.metric("Confidence", f"{confidence_percent}%")
                st.progress(confidence)

            with col2:
                if sensitive_words:
                    st.warning(f"⚠️ Contains sensitive terms: {', '.join(sensitive_words)}")

                if confidence < 0.7:
                    st.info("🔍 Medium confidence - verify with other sources")
                elif confidence < 0.5:
                    st.warning("⚠️ Low confidence - result may be unreliable")

            # Debug info (expandable)
            with st.expander("Analysis Details"):
                st.write("**Original Text:**", news)
                st.write("**Cleaned Text:**", cleaned_news)
                st.write(f"**Model Prediction:** {label} (confidence: {confidence:.2f})")

            # Feedback
            st.markdown("---")
            st.write("Was this analysis helpful?")
            feedback = st.radio("Feedback:", ("👍 Accurate", "🤔 Unsure", "👎 Inaccurate"), horizontal=True)

            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
                # (In production: store feedback in database)


if __name__ == "__main__":
    main()