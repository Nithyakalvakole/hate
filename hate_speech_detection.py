
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
def load_data():
    # Replace with your Kaggle dataset path or API integration
    dataset_path = "path_to_your_kaggle_dataset.csv"
    data = pd.read_csv(dataset_path)
    return data

data = load_data()

# Data Preprocessing
def preprocess_data(data):
    data = data.dropna()  # Remove missing values
    data = data[data['text'].str.strip() != '']  # Remove empty texts
    data['label'] = data['label'].map({"hate": 1, "offensive": 1, "neutral": 0})  # Binary classification
    return data

data = preprocess_data(data)

# Split data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Training and evaluation
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

# Save model and vectorizer
joblib.dump(best_model, "best_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Deployment-ready function
def predict_hate_speech(text):
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Hate Speech" if prediction[0] == 1 else "Neutral"

# Example usage
example_text = "This is an example input text."
print(f"Prediction: {predict_hate_speech(example_text)}")
