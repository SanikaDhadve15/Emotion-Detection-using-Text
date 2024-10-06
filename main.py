import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from deep_translator import GoogleTranslator

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Project Details", "Emotion Detection", "Visualization"])

if page == "Project Details":
    st.title("EMOTION DETECTION USING TEXT.")
    st.markdown(""" 
    ### Emotion Detection System
    This application is designed to detect emotions from text inputs using natural language processing techniques. The model is pre-trained to recognize various emotions based on the input text.

    **Group Member Name:** SIDDHI AVHAD, SANIKA DHADVE, PURVAJA GANGURDE
    """)

elif page == "Emotion Detection":
    st.title("Emotion Detection")

    # Load the pre-trained emotion detection model
    model_path = "C:/Users/Sanika/OneDrive/Desktop/SEM 7/Natural Language Processing/Project NLP/voting_emotion_classifier.pkl"
    pipe_lr = joblib.load(model_path)

    emotions_emoji_dict = {
        "anger": "😠", "disgust": "🤮", "fear": "😨😱",
        "happy": "🤗", "joy": "😂", "neutral": "😐",
        "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
    }

    def predict_emotions(docx):
        results = pipe_lr.predict([docx])
        return results[0]

    def get_prediction_proba(docx):
        results = pipe_lr.predict_proba([docx])
        return results

    # CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff; 
            color: #333;  
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #333;  
            padding: 10px;
            border-radius: 5px;
        }
        .stTextInput, .stTextArea {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stButton {
            color: white;
            border-radius: 5px;
            padding: 10px 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Emotion Detection Form
    with st.form(key='my_form'):
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

elif page == "Visualization":
    st.title("Emotion Visualization")

    # Load the pre-trained emotion detection model
    model_path = "C:/Users/Sanika/OneDrive/Desktop/SEM 7/Natural Language Processing/Project NLP/voting_emotion_classifier.pkl"
    pipe_lr = joblib.load(model_path)

    def get_prediction_proba(docx):
        results = pipe_lr.predict_proba([docx])
        return results

    def visualize_emotion_probabilities(raw_text):
        probability = get_prediction_proba(raw_text)
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["Emotions", "Probability"]

        # Create a bar chart of emotion probabilities
        fig = alt.Chart(proba_df_clean).mark_bar().encode(
            x='Emotions',
            y='Probability',
            color='Emotions'
        ).properties(title='Emotion Detection Using Text')

        st.altair_chart(fig, use_container_width=True)

    # Visualization Form
    with st.form(key='visualize_form'):
        raw_text = st.text_area("Enter Text for Visualization (English or Marathi)", height=150)
        language = st.radio("Select Language", ["English", "Marathi"])
        submit_text = st.form_submit_button(label='Visualize')

    if submit_text:
        if raw_text:
            # Translate Marathi to English for visualization
            if language == "Marathi":
                translated_text = GoogleTranslator(source='mr', target='en').translate(raw_text)
            else:
                translated_text = raw_text

            visualize_emotion_probabilities(translated_text)
        else:
            st.warning("Please enter some text.")
