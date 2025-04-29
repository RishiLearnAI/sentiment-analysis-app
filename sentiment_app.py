import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to clean the input text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove special characters and numbers
    text = text.lower()  # convert to lowercase
    text = text.strip()  # remove leading/trailing whitespace
    return text

# Function to predict sentiment
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    # Predict the sentiment category
    prediction = model.predict(vector)[0]
    
    # Map the prediction back to the sentiment label
    label_mapping = {
        0: 'Anxiety',
        1: 'Normal',
        2: 'Depression',
        3: 'Suicidal',
        4: 'Stress',
        5: 'Bipolar',
        6: 'Personality disorder'
    }
    
    # Return the sentiment label
    return label_mapping.get(prediction, "Unknown")

# Set up the Streamlit UI
st.title("Sentiment Analysis for Mental Health")
st.write("Enter a statement to check its sentiment category:")

user_input = st.text_area("Input Text")

if user_input:
    prediction = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: **{prediction}**")