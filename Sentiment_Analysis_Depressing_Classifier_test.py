
# ===============================
# Sentiment Analysis - Depressing vs Not Depressing
# ===============================

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load the dataset (update the filename if needed)
df = pd.read_csv('D:\PY\Combined Data.csv')

# Display the first few rows
print(df.head())
print(df.columns)

#rename the columns
df.rename(columns={'statement': 'text', 'status': 'sentiment'}, inplace=True)
print(df.columns)

#check the null values
print(df.isnull().sum())
df.dropna(inplace=True)  # or df.fillna('', inplace=True)
print(df.isnull().sum())

#Clean the Text

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove special characters and numbers
    text = text.lower()  # convert to lowercase
    text = text.strip()  # remove leading/trailing whitespace
    return text

df['clean_text'] = df['text'].apply(clean_text)

#no of unique sentiments extract
unique_values = df['sentiment'].unique()
print(unique_values)

#labeling the sentiments
df['sentiment'] = df['sentiment'].map({'Anxiety': 0, 'Normal': 1, 'Depression': 2, 'Suicidal': 3, 'Stress': 4, 'Bipolar': 5, 'Personality disorder': 6})  # adjust based on actual values

print(df['sentiment'].value_counts())

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform the clean text
X = vectorizer.fit_transform(df['clean_text'])

# Target labels
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")

# """ def predict_sentiment(text):
#     # Clean the input text using the same method as before
#     cleaned = clean_text(text)
    
#     # Transform using the trained vectorizer
#     vector = vectorizer.transform([cleaned])
    
#     # Predict
#     prediction = model.predict(vector)[0]
    
#     # Map back to label
#     label_mapping = {
#         0: 'Anxiety',
#         1: 'Normal',
#         2: 'Depression',
#         3: 'Suicidal',
#         4: 'Stress',
#         5: 'Bipolar',
#         6: 'Personality disorder'
#     }

#     # Return the sentiment label
#     return label_mapping.get(prediction, "Unknown") """

# Example: Try it
#user_input = input("Enter a statement: ")
#print("Predicted Sentiment:", predict_sentiment(user_input))
