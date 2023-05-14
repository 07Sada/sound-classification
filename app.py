import streamlit as st
import numpy as np
import librosa
import pickle
import tensorflow as tf
from pydub import AudioSegment
from io import BytesIO
from IPython.display import Audio
import base64

# Loading the model and label Encoder
model = tf.keras.models.load_model("model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))

# Setting the title for application
st.title("Sound Classification")

# Adding Image
image_url = "https://img.freepik.com/premium-vector/girl-listening-music-with-headphone-laptop-cartoon-icon-illustration-people-music-icon-concept-isolated-flat-cartoon-style_138676-1722.jpg?w=2000"

# Display the image.
st.image(image_url, width=300)

# Create a file uploader
uploaded_file = st.file_uploader("Choose a wave file", type=["wav"])
file_1 = uploaded_file
file_2 = uploaded_file


def extract_feature(file_name):
    audio_data, sample_rate = librosa.load(file_name)
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T, axis=0)
    return np.array([scaled])


def print_prediction(file_name):
    pred_fea = extract_feature(file_name)
    pred_vector = np.argmax(model.predict(pred_fea), axis=-1)
    pred_class = le.inverse_transform(pred_vector)
    return pred_class[0]


if st.button("Submit"):
    result = print_prediction(file_2)
    st.header(f"The predicted class is: :blue[{result}]")


if uploaded_file is not None:
    audio_bytes = file_1.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    st.audio(audio_bytes)
