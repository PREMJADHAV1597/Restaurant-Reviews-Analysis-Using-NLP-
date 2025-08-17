import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------- Data Preprocessing -----------------
# Load dataset
a = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

# Lowercasing
a['Review'] = a['Review'].str.lower()

# Remove punctuation
exclude = string.punctuation
def remove_punc(x):
    for i in exclude:
        x = x.replace(i, " ")
    return x

a['Review'] = a['Review'].apply(remove_punc)

# TF-IDF Vectorization
tf = TfidfVectorizer()
X = tf.fit_transform(a['Review']).toarray()
y = a['Liked']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- Model -----------------
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

with st.spinner("Training model..."):
    hist = model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[es], verbose=0)

# ----------------- Streamlit UI -----------------
st.title("🍴 Restaurant Review Sentiment Analyzer")
st.write("Enter a restaurant review below and see if it’s **Positive** or **Negative**.")

# User input
review_text = st.text_area("Write your review here:")

if st.button("Analyze Review"):
    if review_text.strip() != "":
        stl = review_text.lower()
        strm = remove_punc(stl)
        sttf = tf.transform([strm]).toarray()
        pred = model.predict(sttf)[0][0]

        if pred < 0.5:
            st.error("😞 Negative Review")
        else:
            st.success("😊 Positive Review")

# ----------------- Model Performance -----------------
if st.checkbox("Show Model Performance"):
    st.write("### Confusion Matrix")
    yprob2 = model.predict(X_test)
    ypred_ts = [1 if i >= 0.5 else 0 for i in yprob2]

    cf = confusion_matrix(y_test, ypred_ts)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cf)
    disp.plot(ax=ax)
    st.pyplot(fig)
