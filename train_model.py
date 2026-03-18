import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/flipkart_reviews.csv")

# Use Review Text
df = df[['Review Text','Rating']]

# Create sentiment label
df['sentiment'] = df['Rating'].apply(lambda x: 1 if x >=3 else 0)

# Clean text
df['clean_review'] = df['Review Text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['clean_review'])

y = df['sentiment']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

# Train model
model = LogisticRegression()

model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# F1 Score
score = f1_score(y_test,y_pred)

print("F1 Score:",score)

# Save model
joblib.dump(model,"model/sentiment_model.pkl")
joblib.dump(tfidf,"model/tfidf_vectorizer.pkl")

print("Model Saved")