import gradio as gr
import librosa
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("model.h5")

# Emotion labels (edit as per your model)
emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Prediction function
def predict_emotion(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    predicted_emotion = emotions[np.argmax(prediction)]
    return predicted_emotion

# Gradio interface
app = gr.Interface(fn=predict_emotion, 
                   inputs=gr.Audio(type="filepath"), 
                   outputs="text",
                   title="🎙️ Speech Emotion Recognition",
                   description="Upload a voice clip and get the predicted emotion.")

app.launch()
