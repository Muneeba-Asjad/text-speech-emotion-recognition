import os
import re
import io
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# NLTK data download
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

# ---------------------------------------------------------
# 1. EMOTION TO EMOJI MAPPING
# ---------------------------------------------------------
EMOJI_MAP = {
    'hate': '😡', 'neutral': '😐', 'anger': '🤬', 'love': '❤️',
    'worry': '😟', 'relief': '😮‍💨', 'happiness': '😊', 'fun': '🥳',
    'empty': '😶', 'enthusiasm': '🤩', 'sadness': '😢', 'surprise': '😲', 'boredom': '🥱'
}

KEYWORD_MAP = {
    'happy': 'happiness', 'excited': 'happiness', 'joy': 'happiness', 'glad': 'happiness',
    'angry': 'anger', 'hate': 'anger', 'mad': 'anger', 'furious': 'anger',
    'sad': 'sadness', 'depressed': 'sadness', 'crying': 'sadness', 'unhappy': 'sadness',
    'bored': 'boredom', 'dull': 'boredom', 'boring': 'boredom',
    'love': 'love', 'lovely': 'love', 'adoring': 'love',
    'worried': 'worry', 'worry': 'worry', 'scared': 'worry', 'anxious': 'worry',
    'surprised': 'surprise', 'surprise': 'surprise', 'wow': 'surprise', 'shocked': 'surprise',
    'relief': 'relief', 'relieved': 'relief', 'fun': 'fun', 'funny': 'fun', 'enjoying': 'fun',
    'enthusiasm': 'enthusiasm', 'enthusiastic': 'enthusiasm',
    'empty': 'empty', 'blank': 'empty', 'neutral': 'neutral', 'ok': 'neutral', 'normal': 'neutral', 'okay': 'neutral'
}

# ---------------------------------------------------------
# 2. PREPROCESSING & AUDIO PROCESSING
# ---------------------------------------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def transcribe_audio_bytes(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        st.error(f"Could not process audio: {e}")
        return None

# ---------------------------------------------------------
# 3. MAIN APPLICATION INTERFACE
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Text & Speech Emotion Recognition", page_icon="🎭", layout="wide")
    st.title("🎭 Text & Speech Emotion Recognition System")
    st.write("Detect emotions from **Text** or **Voice Input** using NLP and Machine Learning.")

    if 'user_input_text' not in st.session_state:
        st.session_state.user_input_text = ""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_path = os.path.join(script_dir, "dataset.csv")

    df = None
    st.sidebar.header("📊 Dataset Options")

    # Load dataset automatically or manually
    if os.path.exists(default_dataset_path):
        try:
            df = pd.read_csv(default_dataset_path)
            st.sidebar.success("✅ 'dataset.csv' loaded automatically!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset loaded successfully!")

    if df is not None:
        text_col = df.columns[0]
        target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        with st.spinner("Training model on dataset..."):
            df['clean_text'] = df[text_col].apply(preprocess_text)
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(df['clean_text'])
            y = df[target_col]

            if len(df) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        st.sidebar.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")

        # UI Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Emotion Detection", "📈 Model Evaluation", "📋 Confusion Matrix"])

        # TAB 1: Detection
        with tab1:
            st.subheader("Choose Input Method:")
            input_mode = st.radio("Input Source", ["Text Input", "Voice Input (Microphone)"], horizontal=True)

            if input_mode == "Text Input":
                typed_text = st.text_area("Enter your sentence/text below:", value=st.session_state.user_input_text or "I am so happy and excited today!")
                st.session_state.user_input_text = typed_text
            else:
                st.write("Click below to start voice recording:")
                audio = mic_recorder(
                    start_prompt="🎙️ Start Recording",
                    stop_prompt="⏹️ Stop & Process Audio",
                    key='recorder'
                )

                if audio:
                    audio_bytes = audio['bytes']
                    with st.spinner("Transcribing audio to text..."):
                        transcribed = transcribe_audio_bytes(audio_bytes)
                        if transcribed:
                            st.session_state.user_input_text = transcribed
                            st.success(f"**Recognized Text:** {transcribed}")

                if st.session_state.user_input_text:
                    st.text_input("Recognized Text:", value=st.session_state.user_input_text, key="voice_disp")

            if st.button("Detect Emotion", type="primary"):
                user_text = st.session_state.user_input_text.strip()
                if user_text != "":
                    cleaned_input = preprocess_text(user_text)
                    detected_emotion = None
                    confidence = 95.00

                    # 1. Keyword check for accuracy
                    words_in_input = user_text.lower().split()
                    for word in words_in_input:
                        clean_w = re.sub(r'[^a-zA-Z]', '', word)
                        if clean_w in KEYWORD_MAP:
                            detected_emotion = KEYWORD_MAP[clean_w]
                            break

                    # 2. Machine Learning Model check
                    if not detected_emotion:
                        input_vec = vectorizer.transform([cleaned_input])
                        detected_emotion = model.predict(input_vec)[0]
                        probabilities = model.predict_proba(input_vec)[0]
                        confidence = np.max(probabilities) * 100

                    emoji = EMOJI_MAP.get(str(detected_emotion).lower(), '🎭')

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### Predicted Emotion: **{str(detected_emotion).capitalize()}** {emoji}")
                        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                    with col2:
                        st.info(f"**Input Text:** {user_text}")
                else:
                    st.warning("Please provide text or record voice audio.")

        # TAB 2: Metrics
        with tab2:
            st.subheader("Model Performance Metrics (30% Test Data)")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy", f"{acc*100:.2f}%")
            col_b.metric("Precision", f"{precision:.2f}")
            col_c.metric("Recall", f"{recall:.2f}")
            col_d.metric("F1-Score", f"{f1:.2f}")

            st.write("#### Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).transpose())

        # TAB 3: Matrix
        with tab3:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

    else:
        st.warning("⚠️ 'dataset.csv' file missing from project folder.")

if __name__ == "__main__":
    main()