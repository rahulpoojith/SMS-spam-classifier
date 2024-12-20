import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download()


# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required files 'vectorizer.pkl' and 'model.pkl' not found. Please ensure they are in the project directory.")

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the SMS or Email content")
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")