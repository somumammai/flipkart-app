import streamlit as st
import joblib
from preprocess import clean_text

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("Flipkart Review Sentiment Analysis")

review = st.text_area("Enter Product Review")

if st.button("Predict Sentiment"):

    clean = clean_text(review)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😡")