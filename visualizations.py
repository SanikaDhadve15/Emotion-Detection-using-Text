import streamlit as st
import pandas as pd
import altair as alt
import joblib

def show_visualizations():
    st.title("Emotion Visualization")

    # Load the pre-trained emotion detection model
    model_path = "voting_emotion_classifier.pkl"  # Adjust this path as needed
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
