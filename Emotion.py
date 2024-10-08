import streamlit as st
from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Project Details", "Emotion Detection", "Visualization"])

if page == "Project Details":
    st.title("EMOTION DETECTION USING TEXT.")
    st.markdown(""" 
    ### Emotion Detection System
    The Emotion Detection Using Text application is a user-friendly web-based tool designed to analyze and identify emotions expressed in text inputs. Utilizing advanced natural language processing techniques, this application can detect various emotions from both English and Marathi text, providing users with immediate visualization.

    **Group Member Names:** SIDDHI AVHAD, SANIKA DHADVE, PURVAJA GANGURDE
    """)

elif page == "Emotion Detection":
    st.title("Emotion Detection")

    # Load the pre-trained emotion detection model
    model_path ="C:/Users/Sanika/OneDrive/Desktop/SEM 7/Natural Language Processing/Project NLP/emotion_classifier_pipe_lr.pkl"
    
    try:
        pipe_lr = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

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

    # CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff; 
            color: white;  
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            color: white;  
            padding: 10px;
            border-radius: 5px;
        }
        .stTextInput, .stTextArea {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stButton {
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
            emoji_icon = emotions_emoji_dict.get(prediction, "No emoji available")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")
        else:
            st.warning("Please enter some text.")

elif page == "Visualization":
    st.title("Emotion Visualization")

    # Load the pre-trained emotion detection model
    model_path = "C:/Users/Sanika/OneDrive/Desktop/SEM 7/Natural Language Processing/Project NLP/emotion_classifier_pipe_lr.pkl"
    
    try:
        pipe_lr = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

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
