import streamlit as st
import joblib
import string
import tensorflow as tf

# ----------------- Load Model & Vectorizer -----------------
model = tf.keras.models.load_model("sentiment_model.h5")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Define punctuation remover (same as training)
def remove_punc(x):
    exclude = string.punctuation
    for i in exclude:
        x = x.replace(i, " ")
    return x

# ----------------- Streamlit UI -----------------
st.title("🍴 Restaurant Review Sentiment Analyzer")
st.write("Enter a restaurant review below and see if it’s **Positive** or **Negative**.")

# User input
review_text = st.text_area("Write your review here:")

if st.button("Analyze Review"):
    if review_text.strip() != "":
        stl = review_text.lower()
        strm = remove_punc(stl)
        sttf = tfidf.transform([strm]).toarray()
        pred = model.predict(sttf)[0][0]

        if pred < 0.5:
            st.error("😞 Negative Review")
        else:
            st.success("😊 Positive Review")
