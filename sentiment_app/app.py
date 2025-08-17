import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ----------------- Data Preprocessing -----------------
@st.cache_data
def load_and_prepare_data():
    # Load dataset (TSV file with columns: Review, Liked)
    data = pd.read_csv("Restaurant_Reviews(1).tsv", sep="\t")

    # Lowercasing
    data['Review'] = data['Review'].str.lower()

    # Remove punctuation
    exclude = string.punctuation
    def remove_punc(x):
        for i in exclude:
            x = x.replace(i, " ")
        return x

    data['Review'] = data['Review'].apply(remove_punc)
    return data

# Load data
data = load_and_prepare_data()

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['Review']).toarray()
y = data['Liked']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Logistic Regression for simplicity)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ----------------- Streamlit UI -----------------
st.title("🍴 Restaurant Review Sentiment Analyzer (NLP)")
st.write("This app analyzes restaurant reviews as **Positive 😊** or **Negative 😞**")

# Show accuracy
st.sidebar.write(f"📊 Model Accuracy: **{acc*100:.2f}%**")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔹 Single Review", "📂 Dataset Analysis", "📈 Model Insights"])

# ----------------- Single Review -----------------
with tab1:
    review_text = st.text_area("✍️ Write your review here:")
    if st.button("Analyze Review"):
        if review_text.strip() != "":
            processed = review_text.lower()
            for i in string.punctuation:
                processed = processed.replace(i, " ")
            vec = tfidf.transform([processed]).toarray()
            pred = model.predict(vec)[0]

            if pred == 1:
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

# ----------------- Bulk Dataset -----------------
with tab2:
    st.write("Using dataset: **Restaurant_Reviews.tsv**")
    preds = model.predict(X)
    data['Predicted Sentiment'] = ["😊 Positive" if p == 1 else "😞 Negative" for p in preds]

    st.write("### Sample Results")
    st.dataframe(data[['Review', 'Predicted Sentiment']].head(20))

    # Download option
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )

# ----------------- Model Insights -----------------
with tab3:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

