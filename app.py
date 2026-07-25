import streamlit as st
import pickle
import numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False

st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")

if not TF_AVAILABLE:
    st.error("""
TensorFlow is not installed.

Current Python version does not have TensorFlow available.

Install Python 3.11 (recommended), create a virtual environment,
install TensorFlow, and run this app again.
""")
    st.stop()


@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")

    prediction = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Unknown"


st.write("Enter a sentence and the model will predict the next word.")

user_input = st.text_input(
    "Enter text:",
    placeholder="Type a sentence here..."
)

if st.button("Predict Next Word"):
    if user_input.strip():
        word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{word}**")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.caption("LSTM Next Word Prediction using Streamlit")