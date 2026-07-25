import streamlit as st
import pickle
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page Configuration
st.set_page_config(
    page_title="AI Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)


# Title
st.title("🧠 AI Next Word Prediction")
st.subheader("LSTM Based Language Model")


# Model Loading
@st.cache_resource
def load_resources():

    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"
    maxlen_path = "max_len.pkl"


    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    model = load_model(model_path, compile=False)


    with open(tokenizer_path,"rb") as f:
        tokenizer = pickle.load(f)


    with open(maxlen_path,"rb") as f:
        max_len = pickle.load(f)


    # Reverse mapping
    index_word = {
        index: word 
        for word,index in tokenizer.word_index.items()
    }


    return model, tokenizer, max_len, index_word



model, tokenizer, max_len, index_word = load_resources()



# Prediction Function
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences(
        [text]
    )[0]


    padded = pad_sequences(
        [sequence],
        maxlen=max_len-1,
        padding="pre"
    )


    prediction = model.predict(
        padded,
        verbose=0
    )


    predicted_index = np.argmax(prediction)


    return index_word.get(
        predicted_index,
        "Unknown"
    )



# Input UI

user_text = st.text_input(
    "Enter your sentence:",
    placeholder="Example: Machine learning is"
)



if st.button("🚀 Predict"):

    if user_text.strip():

        result = predict_next_word(user_text)


        st.success(
            f"Next Word Prediction: **{result}**"
        )

    else:
        st.warning(
            "Please enter text first."
        )



# Sidebar

with st.sidebar:

    st.header("Model Information")

    st.write(
        """
        🧠 Model: LSTM Neural Network

        📚 Task:
        Next Word Prediction

        ⚡ Framework:
        TensorFlow + Keras

        🎨 Interface:
        Streamlit
        """
    )



st.divider()

st.caption(
    "Built using Deep Learning (LSTM) and NLP"
)