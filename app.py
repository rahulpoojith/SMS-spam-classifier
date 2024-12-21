import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform input text
def transform_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    # Return the transformed text
    return " ".join(text)

# Load vectorizer and model
try:
    tf = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure this file exists
    model = pickle.load(open('model.pkl', 'rb'))    # Ensure this file exists
except FileNotFoundError:
    st.error("Required files 'vectorizer.pkl' and 'model.pkl' not found. Please ensure they are in the project directory.")
    st.stop()  # Stop execution if files are not found

# Streamlit app UI
st.title("Email/SMS Spam Classifier")

# Input from the user
input_sms = st.text_input("Enter the SMS or Email content")

# Prediction button
if st.button('Predict'):
    # Check if input is empty
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # Preprocess and transform the input text
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the transformed input
        vector_input = tf.transform([transformed_sms])
        
        # Predict using the loaded model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")