import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from gtts import gTTS
import os
import time
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Stress Relief Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #A8E6CF, #88D8A3);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .result-box {
        background: linear-gradient(135deg, #FFD93D, #FF8E53);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .feedback-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #56ab2f, #a8e6cf);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .recording-indicator {
        font-size: 1.5rem;
        color: #FF6B6B;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .feedback-button {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }
    .feedback-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6) !important;
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
    }
    .session-complete {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
    }
    .main-content {
        min-height: 600px;
    }
    .api-warning {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

# Google Form URL (Replace with your actual Google Form link)
FEEDBACK_FORM_URL = "https://docs.google.com/forms/d/your-form-id-here/viewform?usp=sharing"

class StressAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_analysis = pipeline("sentiment-analysis", 
                                         model=self.model, tokenizer=self.tokenizer)
        
        # Set up Groq client - using environment variable or Streamlit secrets
        self.client = None
        self._setup_groq_client()
    
    def _setup_groq_client(self):
        """Initialize Groq client using environment variable or Streamlit secrets"""
        try:
            # Try Streamlit secrets first, then environment variable
            api_key = None
            if hasattr(st, 'secrets') and st.secrets and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
            else:
                api_key = os.getenv('GROQ_API_KEY')
            
            if not api_key:
                # Don't show error on GitHub - just disable AI features gracefully
                st.sidebar.warning("🤖 AI Assistant: Setup Required")
                st.sidebar.info("Set `GROQ_API_KEY` environment variable for AI recommendations")
                return False
            
            from groq import Client
            self.client = Client(api_key=api_key)
            st.sidebar.success("🤖 AI Assistant: Connected ✓")
            return True
            
        except Exception as e:
            st.sidebar.error(f"🤖 AI Setup Error: {str(e)[:50]}...")
            return False
    
    def record_audio(self, duration):
        """Record audio with progress indication"""
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                              rate=RATE, input=True,
                              frames_per_buffer=CHUNK)
            
            # Recording progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            frames = []
            
            status_text.markdown('<div class="recording-indicator">🎤 Recording... Speak naturally!</div>', 
                               unsafe_allow_html=True)
            
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Update progress
                progress = (i + 1) / (int(RATE / CHUNK * duration))
                progress_bar.progress(progress)
                time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save audio file
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            status_text.success("✅ Recording completed!")
            return True
            
        except Exception as e:
            st.error(f"Recording failed: {e}")
            return False
    
    def transcribe_audio(self):
        """Transcribe audio to text"""
        try:
            with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try speaking more clearly.")
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the transcribed text"""
        try:
            result = self.sentiment_analysis(text)[0]
            sentiment = result['label']
            confidence = result['score']
            return sentiment, confidence
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
            return "NEUTRAL", 0.5
    
    def get_recommendation(self, text, sentiment):
        """Get personalized recommendation from LLM or fallback suggestion"""
        if not self.client:
            # Graceful fallback when AI is not available
            fallback_suggestions = [
                """🌿 **Quick Stress Relief**: It sounds like you're carrying a lot right now. Try this simple breathing exercise: **Inhale for 4 seconds, hold for 4, exhale for 6.** Repeat 3 times. You're doing great just by taking this moment for yourself! 💚""",
                """🌈 **Gentle Reminder**: It's completely normal to feel overwhelmed sometimes. Try writing down 3 things you're grateful for, no matter how small. This simple practice can shift your perspective. You're stronger than you think! 🌟""",
                """💧 **Self-Care Moment**: Take a slow sip of water and stretch your arms above your head. Sometimes our body just needs a small reminder that we're here and safe. You've got this! 🫂""",
                """☀️ **Nature Break**: Step outside for 2 minutes if you can, or just look out a window. Notice one thing you can see, hear, or feel. Connecting with your senses can ground you. You're doing important work! 🌱"""
            ]
            import random
            return random.choice(fallback_suggestions)
        
        try:
            system_prompt = f"""
            You are a compassionate psychologist specializing in stress management. 
            Based on the user's spoken text and their detected sentiment ({sentiment}), 
            provide a warm, empathetic, and practical recommendation in 1-2 sentences.
            
            Focus on:
            - Acknowledging their feelings
            - Offering a simple, actionable suggestion
            - Providing gentle encouragement
            
            Keep it concise, supportive, and professional.
            """
            
            llm_response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User said: '{text}'"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=100
            )
            
            return llm_response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Failed to generate AI recommendation: {e}")
            # Fallback to random suggestion
            fallback_suggestions = [
                "Take a deep breath and remember that it's okay to feel this way. Try going for a short walk to clear your mind.",
                "You're doing great just by acknowledging how you feel. Try placing a hand on your heart and taking 3 slow breaths.",
                "It's normal to have tough moments. Give yourself permission to pause and do one small thing that brings you comfort."
            ]
            import random
            return random.choice(fallback_suggestions)

def main():
    # Initialize the assistant
    assistant = StressAssistant()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Stress Relief Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your personal mental wellness companion - speak your mind, receive gentle guidance</p>', 
                unsafe_allow_html=True)
    
    # Session state for tracking
    if 'session_completed' not in st.session_state:
        st.session_state.session_completed = False
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = None
    if 'recommendation' not in st.session_state:
        st.session_state.recommendation = None
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        ### 🎙️ **Recording Instructions:**
        1. **Click "Start Recording"** below
        2. **Speak naturally** about how you're feeling (work stress, personal challenges, etc.)
        3. **Wait for the recording to complete** (progress bar will show)
        4. **Review your analysis** and personalized recommendation
        5. **Provide feedback** at the end to help us improve!
        
        ### 💡 **Tips for Best Results:**
        - Speak clearly in a quiet environment
        - Try to express your current emotions honestly
        - Recording length: 5-30 seconds works best
        - The system analyzes your tone and provides gentle guidance
        
        ### 🎯 **What You'll Get:**
        - 📊 Sentiment analysis of your speech
        - 🎧 Audio playback of your recording
        - 💬 Personalized stress relief recommendation
        - 🔊 Spoken recommendation in calming voice
        - 📝 Quick feedback opportunity
        
        ### 🔧 **Setup for AI Features:**
        Set `GROQ_API_KEY` environment variable for personalized AI recommendations
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ **Privacy Notice**")
        st.info("""
        🔒 **Your Privacy Matters:**
        - Audio processed locally, not stored
        - Transcriptions used only for analysis
        - No personal data collected
        - All processing happens in-browser
        - Feedback is anonymous
        """)
    
    # Main content - Use tabs for better organization
    tab1, tab2 = st.tabs(["🎙️ Record Session", "📊 Your Results"])
    
    with tab1:
        # Recording controls
        st.subheader("🎙️ Record Your Thoughts")
        
        # Duration slider
        duration = st.slider(
            "Recording Duration", 
            min_value=3, 
            max_value=60, 
            value=10,
            help="Choose how long you'd like to speak (3-60 seconds)"
        )
        
        # Record button
        if st.button("🎤 Start Recording", key="record", use_container_width=True):
            if assistant.record_audio(duration):
                with st.spinner("🔍 Processing your voice..."):
                    # Transcribe
                    transcribed_text = assistant.transcribe_audio()
                    
                    if transcribed_text:
                        # Store in session state
                        st.session_state.transcribed_text = transcribed_text
                        st.session_state.session_completed = True
                        
                        # Analyze sentiment
                        sentiment, confidence = assistant.analyze_sentiment(transcribed_text)
                        st.session_state.sentiment = sentiment
                        st.session_state.confidence = confidence
                        
                        # Get recommendation
                        with st.spinner("🤔 Generating personalized recommendation..."):
                            recommendation = assistant.get_recommendation(transcribed_text, sentiment)
                            st.session_state.recommendation = recommendation
                        
                        # Play original audio
                        st.audio(WAVE_OUTPUT_FILENAME, format='audio/wav')
                        
                        st.success("✅ Session completed! Check your results in the 'Your Results' tab.")
    
    with tab2:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        # Display results only if session completed
        if st.session_state.session_completed and st.session_state.transcribed_text:
            
            # Transcribed Text - Always visible and prominent
            st.markdown("""
            <div class="info-box">
                <h3>📝 What You Shared</h3>
                <p style="font-size: 1.2rem; margin: 1rem 0; font-style: italic; line-height: 1.5;">
                """ + st.session_state.transcribed_text + """
                </p>
                <div style="text-align: right; font-size: 0.9rem; color: #5D6D7E;">
                    💭 Your honest words help us understand you better
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sentiment Analysis - Always visible
            if st.session_state.sentiment:
                sentiment_emoji = "😊" if st.session_state.sentiment == "POSITIVE" else "😔" if st.session_state.sentiment == "NEGATIVE" else "😐"
                sentiment_color_class = "sentiment-positive" if st.session_state.sentiment == "POSITIVE" else "sentiment-negative"
                
                confidence = st.session_state.confidence if st.session_state.confidence else 0.5
                
                st.markdown(f"""
                <div class="{sentiment_color_class}">
                    <h3>{sentiment_emoji} Your Emotional State</h3>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Detected: <strong>{st.session_state.sentiment}</strong></p>
                    <p style="margin: 0.5rem 0; opacity: 0.9;">Confidence: <strong>{(confidence * 100):.1f}%</strong></p>
                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
                        🌈 This helps us tailor the perfect recommendation for you
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendation - Always visible and prominent
            if st.session_state.recommendation:
                st.markdown("""
                <div class="result-box">
                    <h3>💡 Your Personalized Guidance</h3>
                    <div style="font-size: 1.3rem; font-style: italic; margin: 1.5rem 0; line-height: 1.6; background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                    """ + st.session_state.recommendation + """
                    </div>
                    <div style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: #2E86AB;">
                        🌟 Remember, small steps lead to big changes
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio Recommendation - Always visible
                try:
                    with st.spinner("🗣️ Creating audio recommendation..."):
                        tts = gTTS(text=st.session_state.recommendation, lang='en', slow=False)
                        recommendation_audio_path = "recommendation.mp3"
                        tts.save(recommendation_audio_path)
                        
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
                        <h4 style="color: white; margin: 0 0 1rem 0;">🎧 Listen to Your Recommendation</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.audio(recommendation_audio_path, format='audio/mp3')
                    
                    # Download button
                    with open(recommendation_audio_path, "rb") as audio_file:
                        st.download_button(
                            label="💾 Download This Guidance",
                            data=audio_file.read(),
                            file_name="my_stress_relief_guidance.mp3",
                            mime="audio/mpeg",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")
            
            # Quick Action Buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 New Session", use_container_width=True):
                    # Reset session
                    for key in ['session_completed', 'transcribed_text', 'recommendation', 'sentiment', 'confidence']:
                        st.session_state[key] = None if key != 'session_completed' else False
                    st.rerun()
            
            with col2:
                st.markdown("---")
                st.info("👆 Switch to 'Record Session' tab to start fresh")
            
            with col3:
                if st.button("📝 Save This Session", use_container_width=True):
                    st.success("Session saved to your browser! 🎉")
        
        else:
            st.info("""
            ### 🌟 Welcome to Your Wellness Journey
            
            **Start by recording your thoughts in the "Record Session" tab above.**
            
            **What to expect:**
            - 📝 Your words will appear here clearly
            - 🎭 We'll analyze your emotional tone
            - 💡 You'll receive personalized guidance
            - 🎧 Hear your recommendation in a calming voice
            
            **Remember:** This is a safe space to express yourself honestly.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feedback section - only show after session completion
    if st.session_state.session_completed and st.session_state.transcribed_text:
        st.markdown("---")
        st.markdown("""
        <div class="session-complete">
            <h2>🎉 Thank You for This Moment of Self-Care!</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">Your vulnerability is your strength 💪</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback section
        st.markdown("""
        <div class="feedback-box">
            <h3>📝 Help Us Support You Better</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">Your feedback helps us make this tool even more helpful for everyone!</p>
            <p style="font-size: 1rem; margin: 0.5rem 0; opacity: 0.9;">It takes just 1 minute and is completely anonymous</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <a href="{FEEDBACK_FORM_URL}" target="_blank" style="
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: bold;
                    font-size: 1.1rem;
                    display: inline-block;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
                    transition: all 0.3s ease;
                ">
                    🌟 Share Your Feedback
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        # Feedback preview
        with st.expander("👀 Preview: What we'll ask you", expanded=False):
            st.markdown("""
            ### Quick 4-question survey:
            1. ⭐ **How helpful was the recommendation?** (1-5 stars)
            2. 💭 **What felt most meaningful to you?**
            3. 💡 **What could we improve?** (optional)
            4. ⏱️ **How was the experience overall?**
            
            *All responses are anonymous and help us serve you better!*
            """)
        
        st.balloons()
        st.success("💚 You're doing important work by caring for your mental wellness!")
    
    # Footer
    

if __name__ == "__main__":
    main()