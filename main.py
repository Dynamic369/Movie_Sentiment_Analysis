import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import mysql.connector


#Database connection
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user = 'root',
        password = "Pradum123",
        database ='sentiment_db'
    )

def save_to_db(review,sentiment,confidence):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO reviews (review_text, sentiment, confidence)
        VALUES (%s, %s, %s)
        """
    cursor.execute(sql,(review,sentiment,confidence))
    conn.commit()
    cursor.close()
    conn.close()

# Load the imdb dataset word_index
word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('best_model.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
VOCAB_SIZE = 10000  # must match your model

def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2) + 3
        # Clip to vocab range
        if index >= VOCAB_SIZE:
            index = 2  # unknown token
        encoded_review.append(index)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

#User Input
user_input =st.text_area("Movie Review")

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter a movie review first.")
    else:
        preprocessed_input = preprocess_text(user_input)

        
        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])
        sentiment = "Positive" if score > 0.5 else "Negative"


        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score}")

        # Save to MySQL
        save_to_db(user_input, sentiment, score)
       

else:
    st.write("Please enter a movie review")
