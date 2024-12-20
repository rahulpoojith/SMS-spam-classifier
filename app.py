import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure necessary NLTK data is downloaded
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

# Load pre-trained vectorizer and model
try:
    tf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this script.")

st.title("SMS Spam Classifier")

# Input box for user message
input_sms = st.text_input("Enter the SMS or Email content")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # 1. Preprocess the text
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the input
        vector_input = tf.transform([transformed_sms])
        
        # 3. Predict the result
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")