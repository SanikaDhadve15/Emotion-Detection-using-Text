import streamlit as st
from emotion_detection import emotion_detection
from visualizations import show_visualizations
from project_details import show_project_details

def main():
    st.set_page_config(page_title="Emotion Detection Using Text")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Project Details", "Emotion Detection", "Visualizations"])

    if page == "Project Details":
        show_project_details()
    elif page == "Emotion Detection":
        emotion_detection()
    elif page == "Visualizations":
        show_visualizations()

if __name__ == "__main__":
    main()
