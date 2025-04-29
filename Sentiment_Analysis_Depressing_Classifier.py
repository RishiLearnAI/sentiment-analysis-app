
# ===============================
# Sentiment Analysis - Depressing vs Not Depressing
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Sample Data (Depressing vs Not Depressing)
data = {
    'text': [
        "I feel so hopeless", 
        "It's a great day", 
        "I want to give up", 
        "I'm so happy right now", 
        "I can't stop crying", 
        "Life is beautiful", 
        "Everything feels so heavy", 
        "I'm excited for the weekend", 
        "I feel worthless", 
        "Today was amazing"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Depressing, 0 = Not Depressing
}

df = pd.DataFrame(data)

# Step 2: Preprocessing Text Data
X = df['text']
y = df['label']

# Step 3: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:
", confusion_matrix(y_test, y_pred))
print("Classification Report:
", classification_report(y_test, y_pred))

# Step 8: Make Predictions on New Data
new_statements = [
    "I feel so alone",
    "I'm looking forward to the future"
]
new_statements_vec = vectorizer.transform(new_statements)
new_predictions = model.predict(new_statements_vec)
print("New Predictions:", new_predictions)  # 1 = Depressing, 0 = Not Depressing
