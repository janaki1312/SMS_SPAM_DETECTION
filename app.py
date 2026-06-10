import streamlit as st
import pickle
import re

from scipy.sparse import hstack
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    return text


def capital_ratio(text):
    total_chars = len(text)

    if total_chars == 0:
        return 0

    capitals = sum(1 for c in text if c.isupper())

    return capitals / total_chars


def has_url(text):
    pattern = r'http\S+|www\S+|https\S+'

    return int(bool(re.search(pattern, text)))


def has_currency(text):
    symbols = ['£', '$', '₹']

    return int(any(symbol in text for symbol in symbols))

st.title("📩 SMS Spam Detection System")
st.caption("Built using Random Forest, TF-IDF and Feature Engineering")

message = st.text_area(
    "Enter a message:"
)
if st.button("Predict"):

    if not message.strip():
        st.warning("Please enter a message.")
        st.stop()

    clean_message = clean_text(message)

    message_length = len(clean_message)

    word_count = len(clean_message.split())

    exclamation_count = message.count("!")

    cap_ratio = capital_ratio(message)

    url_flag = has_url(message)

    currency_flag = has_currency(message)

    X_text = tfidf.transform([clean_message])

    numerical_features = [
        [
            message_length,
            word_count,
            exclamation_count,
            cap_ratio,
            url_flag,
            currency_flag
        ]
    ]

    X_final = hstack([
        X_text,
        numerical_features
    ])

    #prediction = model.predict(X_final)[0]

    probability = model.predict_proba(X_final)[0][1]

    st.subheader("Result")

    st.metric(
    "Spam Probability",
    f"{probability:.2%}"
)

    if probability >= 0.7:
        st.error(
            f"🚨 SPAM ({probability:.2%})"
        )

    elif probability <= 0.3:
        st.success(
            f"✅ HAM ({1 - probability:.2%} confidence)"
        )

    else:
        st.warning(
            f"⚠️ REVIEW REQUIRED ({probability:.2%})"
        )

