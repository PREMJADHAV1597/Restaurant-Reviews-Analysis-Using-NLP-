import streamlit as st
import pandas as pd
import joblib
import string
import tensorflow as tf
import matplotlib.pyplot as plt

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
st.write("Analyze restaurant reviews as **Positive 😊** or **Negative 😞**")

# Tabs for Single Review / Bulk File
tab1, tab2 = st.tabs(["🔹 Single Review", "📂 Analyze Dataset"])

# ----------------- Single Review -----------------
with tab1:
    review_text = st.text_area("✍️ Write your review here:")
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

# ----------------- Bulk Dataset Analysis -----------------
with tab2:
    # Load file (your given file)
    data = pd.read_csv("C:\Users\jadha\Downloads\Restaurant_Reviews (1).tsv", sep="\t")

    if "Review" not in data.columns:
        st.error("❌ File must contain a column named 'Review'")
    else:
        # Preprocess reviews
        data['Cleaned'] = data['Review'].str.lower().apply(remove_punc)
        sttf = tfidf.transform(data['Cleaned']).toarray()
        preds = model.predict(sttf)

        # Convert predictions
        data['Sentiment'] = ["😊 Positive" if p >= 0.5 else "😞 Negative" for p in preds]

        st.write("### Sample Results")
        st.dataframe(data[['Review', 'Sentiment']].head(20))

        # Chart: distribution of sentiments
        st.write("### Sentiment Distribution")
        counts = data['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax, color=["green", "red"])
        plt.xticks(rotation=0)
        st.pyplot(fig)

        # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="restaurant_sentiment_results.csv",
            mime="text/csv"
        )

