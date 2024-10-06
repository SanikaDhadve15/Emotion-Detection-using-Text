import streamlit as st
import pandas as pd
import numpy as np
import joblib
from deep_translator import GoogleTranslator

def emotion_detection():
    # Load the pre-trained emotion detection model
    model_path = "voting_emotion_classifier.pkl"  # Adjust this path as needed
    pipe_lr = joblib.load(model_path)

    emotions_emoji_dict = {
        "anger": "😠 (कडक)", 
        "disgust": "🤮 (घृणा)", 
        "fear": "😨😱 (भय)", 
        "happy": "🤗 (आनंद)", 
        "joy": "😂 (आनंद)", 
        "neutral": "😐 (तटस्थ)", 
        "sad": "😔 (दुखी)", 
        "sadness": "😔 (दुखी)", 
        "shame": "😳 (लाज)", 
        "surprise": "😮 (आश्चर्य)"
    }

    def predict_emotions(docx):
        results = pipe_lr.predict([docx])
        return results[0]

    def get_prediction_proba(docx):
        results = pipe_lr.predict_proba([docx])
        return results

    # Emotion Detection Form
    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type Here (English or Marathi)", height=150)
        language = st.radio("Select Language", ["English", "Marathi"])
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if raw_text:
            # Translate Marathi to English
            if language == "Marathi":
                translated_text = GoogleTranslator(source='mr', target='en').translate(raw_text)
            else:
                translated_text = raw_text

            prediction = predict_emotions(translated_text)
            probability = get_prediction_proba(translated_text)

            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")
        else:
            st.warning("Please enter some text.")
