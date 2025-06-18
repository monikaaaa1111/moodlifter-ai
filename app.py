import streamlit as st
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr
import random
import threading
import json
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize pyttsx3 (Text to Speech engine)
engine = pyttsx3.init()

# List of mood-based quotes (expanded collection)
quotes = {
    "happy": [
        "Keep spreading that positivity, the world needs more of it! 🌞",
        "You are capable of amazing things! 🌟",
        "Embrace the glorious mess that you are! 💫",
        "Today is a perfect day to start living your dreams! 🌈",
        "Your energy is contagious - keep shining bright! ✨"
    ],
    "sad": [
        "It's okay to feel down sometimes, take a deep breath and move forward. 💪",
        "You are not alone, tomorrow is a new day. 🌈",
        "Take it one step at a time, you'll get through this. 🌿",
        "Even the darkest night will end and the sun will rise. 🌅",
        "Your strength is greater than any temporary sadness. 🌟"
    ],
    "neutral": [
        "Everything will be okay, one step at a time. 🌱",
        "Breathe, relax, and know that you are doing your best. 🌸",
        "Stay calm, and keep going. You've got this. 💪",
        "Balance is the key to a centered life. 🧘‍♀️",
        "Sometimes the middle path is where wisdom lies. 🌿"
    ],
    "stressed": [
        "Stress is temporary. Take things one step at a time. 🌿",
        "Take a break, breathe deeply, and re-energize. 🌼",
        "Your pace is just fine. Relax, and breathe. 🌟",
        "You've overcome challenges before, and you will again. 💪",
        "Remember to pause and reset when things get overwhelming. 🧠"
    ],
    "angry": [
        "It's okay to feel angry, but remember to breathe before you act. 🧘",
        "Your emotions are valid, but they don't define you. 🌈",
        "Count to ten and remember what truly matters to you. 💫",
        "Channel that energy into something constructive. 🔨",
        "This feeling will pass, and clarity will return. 🌤️"
    ],
    "anxious": [
        "Anxiety is just a feeling, not a fact about the world. 🌎",
        "Focus on what you can control right now in this moment. ⏱️",
        "You are safe, you are capable, you will get through this. 🛡️",
        "Your mind might race, but your breath can bring you home. 🏡",
        "This moment of worry will pass like clouds in the sky. ☁️"
    ]
}

# Emotion-based AI suggestions expanded with more personalized advice
resources = {
    "happy": [
        "Try doing something creative like drawing or writing!",
        "Share your positivity with others!",
        "Listen to upbeat music to keep the vibes going.",
        "Start a gratitude journal to capture this positive energy.",
        "Reach out to someone who might need some cheer today."
    ],
    "sad": [
        "Maybe try talking to a friend or loved one.",
        "Here's a breathing exercise: Inhale deeply for 4 counts, hold for 4, and exhale for 4.",
        "Consider journaling your thoughts.",
        "Watch a comfort movie or show that always makes you feel better.",
        "Gentle movement like stretching can help shift your emotional state."
    ],
    "neutral": [
        "A walk outside can help reset your mind.",
        "Try a simple stretching routine to clear your head.",
        "Meditation or yoga can help bring a sense of calm.",
        "This is a good time to plan something you're looking forward to.",
        "Try learning something new while your mind is balanced."
    ],
    "stressed": [
        "Take a quick break and breathe deeply.",
        "Try listening to a podcast on stress relief.",
        "Consider doing some mindfulness exercises.",
        "Make a prioritized list to help organize your thoughts.",
        "Step away from screens for 15 minutes and stretch."
    ],
    "angry": [
        "Physical activity can help release tension - try a quick walk.",
        "Write down what's bothering you, then decide if it's worth your energy.",
        "Focus on your breath - inhale peace, exhale anger.",
        "Listen to calming music to shift your emotional state.",
        "Try counting to 20 before responding to anything that triggered you."
    ],
    "anxious": [
        "Ground yourself with the 5-4-3-2-1 technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
        "Progressive muscle relaxation: tense and release each muscle group.",
        "Try box breathing: inhale for 4, hold for 4, exhale for 4, hold for 4.",
        "Make a simple plan with small steps to address what's causing anxiety.",
        "Limit caffeine and stay hydrated with water or herbal tea."
    ]
}

# Training data for our simple mood classifier
training_data = {
    "happy": [
        "I feel great today!", "I'm so happy", "Everything is wonderful", 
        "I'm feeling fantastic", "Today is amazing", "I'm in such a good mood",
        "I feel blessed", "I'm thrilled about everything", "Life is beautiful"
    ],
    "sad": [
        "I feel down", "I'm so sad", "Everything is terrible", 
        "I'm feeling depressed", "Today is horrible", "I'm in such a bad mood",
        "I feel hopeless", "I'm upset about everything", "Nothing seems to be going right"
    ],
    "neutral": [
        "I'm okay", "Just normal today", "Nothing special", 
        "I feel alright", "Today is average", "I'm neither happy nor sad",
        "I feel balanced", "Things are fine", "Just another day"
    ],
    "stressed": [
        "I'm overwhelmed", "So much to do", "I can't handle this", 
        "I'm feeling pressured", "Everything is too much", "I'm stressed out",
        "I have too many deadlines", "I'm anxious about work", "There's not enough time"
    ],
    "angry": [
        "I'm furious", "This makes me so mad", "I can't believe this happened", 
        "I'm feeling rage", "I want to scream", "I'm so irritated",
        "This is infuriating", "I'm angry at everyone", "That's so frustrating"
    ],
    "anxious": [
        "I'm worried about everything", "I feel uneasy", "What if things go wrong", 
        "I'm feeling nervous", "I can't stop worrying", "I have butterflies in my stomach",
        "I'm dreading tomorrow", "My mind won't stop racing", "I feel on edge"
    ]
}

# Function for text-to-speech with voice selection
def speak_quote(quote, voice_index=None):
    def speak():
        engine.stop()  # Stop any previous speech
        voices = engine.getProperty('voices')
        
        # Set voice if specified and available
        if voice_index is not None and voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
            
        engine.setProperty('rate', 150)  # Set the speaking rate
        engine.say(quote)
        engine.runAndWait()

    thread = threading.Thread(target=speak)
    thread.start()

# Build a simple ML model for mood classification
def build_mood_classifier():
    # Prepare data
    texts = []
    labels = []
    
    for mood, phrases in training_data.items():
        texts.extend(phrases)
        labels.extend([mood] * len(phrases))
    
    # Create a training DataFrame
    df = pd.DataFrame({
        'text': texts,
        'mood': labels
    })
    
    # Create vectorizer and classifier
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    
    classifier = MultinomialNB()
    classifier.fit(X, df['mood'])
    
    return vectorizer, classifier

# Initialize the classifier
vectorizer, mood_classifier = build_mood_classifier()

# Enhanced function for analyzing mood with multiple techniques
def analyze_mood(text, use_ml=True):
    # Start with TextBlob sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Use our trained ML model if enabled
    if use_ml:
        try:
            # Transform input text
            text_transformed = vectorizer.transform([text])
            
            # Get predicted mood
            mood = mood_classifier.predict(text_transformed)[0]
            
            # Get prediction probabilities
            probs = mood_classifier.predict_proba(text_transformed)[0]
            max_prob = max(probs)
            
            # If confidence is high enough, return the ML prediction
            if max_prob > 0.4:
                return mood
        except Exception as e:
            st.write(f"ML mood detection error (falling back to basic analysis): {e}")
    
    # Fallback to rule-based sentiment if ML doesn't have high confidence
    if sentiment > 0.3:
        return "happy"
    elif sentiment < -0.3:
        return "sad"
    elif -0.3 <= sentiment <= 0.0:
        return "neutral"
    elif sentiment < 0.3 and "stress" in text.lower() or "overwhelm" in text.lower():
        return "stressed"
    elif "angry" in text.lower() or "mad" in text.lower() or "furious" in text.lower():
        return "angry"
    elif "anxious" in text.lower() or "worry" in text.lower() or "nervous" in text.lower():
        return "anxious"
    else:
        return "neutral"

# AI-based emotion detection (using external API)
def predict_emotion(text):
    # First try local analysis
    local_mood = analyze_mood(text)
    
    # Try external API if available and configured
    try:
        # Check if API key is configured
        if "HUGGINGFACE_API_KEY" in st.secrets:
            api_key = st.secrets["HUGGINGFACE_API_KEY"]
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": text},
                timeout=5  # Add timeout to prevent long waits
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Map the emotion labels to our categories
                emotion_mapping = {
                    "joy": "happy",
                    "sadness": "sad",
                    "neutral": "neutral",
                    "fear": "anxious",
                    "anger": "angry",
                    "surprise": "happy",
                    "disgust": "angry"
                }
                
                if isinstance(response_data, list) and len(response_data) > 0:
                    # Find highest scoring emotion
                    top_emotion = max(response_data[0], key=lambda x: x['score'])
                    mapped_mood = emotion_mapping.get(top_emotion['label'], local_mood)
                    return mapped_mood
    except Exception as e:
        # Just log the error and continue with local analysis
        print(f"External API error: {e}")
    
    # Fallback to local analysis
    return local_mood

# Improved function for voice input with better error handling
def listen_to_voice():
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.write("Listening for your mood... (Please speak now)")
            
            # Progress bar for visual feedback
            progress_bar = st.progress(0)
            
            # Adjusting for ambient noise before listening
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Animate the progress bar while listening
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.03)
            
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            try:
                # Recognize the speech and return the text
                text = recognizer.recognize_google(audio)
                st.write(f"Recognized speech: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand that. Please try again or use text input instead.")
                return None
            except sr.RequestError:
                st.error("Sorry, there was an issue with the speech service. Please try text input instead.")
                return None
    except Exception as e:
        st.error(f"Error accessing microphone: {str(e)}")
        st.info("Please check your microphone permissions or use text input instead.")
        return None

# Enhanced AI feedback with more personalized responses
def ai_feedback(mood, user_input="", mood_history=None):
    if mood_history is None or len(mood_history) < 2:
        # Basic feedback based on current mood
        if mood == 'happy':
            return "You're radiating positivity! Keep it going! 😊"
        elif mood == 'sad':
            return "I'm sorry you're feeling down. Remember that emotions are temporary, and it's okay to seek support. 🌸"
        elif mood == 'neutral':
            return "You seem balanced today. Sometimes, just being present is enough. 🌱"
        elif mood == 'stressed':
            return "I can sense your stress. Remember to take short breaks and practice deep breathing. 💪"
        elif mood == 'angry':
            return "I understand you're feeling frustrated. It's important to acknowledge your feelings without letting them control you. 🧠"
        elif mood == 'anxious':
            return "Anxiety can be challenging. Remember that your thoughts aren't always facts, and this feeling will pass. 🌈"
        else:
            return "You're doing amazing, even if you don't feel like it right now! Keep going! ✨"
    else:
        # More advanced feedback based on mood patterns
        if mood == mood_history[-2]:  # Same mood as before
            if mood == 'happy':
                return "You're maintaining your positive energy! That's wonderful to see. Keep nurturing what brings you joy! 🌟"
            elif mood == 'sad':
                return "I notice you're still feeling down. Would it help to talk to someone you trust about what's bothering you? 🌸"
            elif mood == 'stressed':
                return "Your stress seems to be persisting. Consider breaking down what's overwhelming you into smaller, manageable tasks. 📝"
            elif mood == 'angry':
                return "You're still feeling angry. Consider what's at the root of this emotion - sometimes anger masks other feelings like hurt or fear. 🧩"
            elif mood == 'anxious':
                return "Your anxiety is continuing. Try grounding yourself with the 5-4-3-2-1 technique to bring yourself back to the present moment. 🏡"
        else:  # Mood has changed
            if mood == 'happy' and mood_history[-2] in ['sad', 'angry', 'stressed', 'anxious']:
                return "What a wonderful change in your mood! Whatever you did to shift from feeling down to feeling good - keep doing more of that! 🌈"
            elif mood in ['sad', 'angry', 'stressed', 'anxious'] and mood_history[-2] == 'happy':
                return "I notice your mood has shifted downward. Remember that all emotions are temporary, and you've felt better before - you will again. 🌅"
            elif mood in ['happy', 'neutral'] and mood_history[-2] in ['sad', 'angry', 'stressed', 'anxious']:
                return "You're moving in a positive direction! That's a testament to your resilience and inner strength. 💪"
            
        # Default response if no specific pattern match
        return f"I notice your mood has shifted from {mood_history[-2]} to {mood}. How can I best support you right now? 🌱"

# Function to dynamically suggest resources based on mood and context
def suggest_resource(mood, user_input="", mood_history=None):
    # Get base suggestions for the mood
    mood_resources = resources.get(mood, resources["neutral"])
    
    # Special case handling based on input text
    lower_input = user_input.lower()
    
    if "work" in lower_input or "job" in lower_input or "boss" in lower_input:
        if mood == "stressed":
            return "Try the Pomodoro Technique: 25 minutes of focused work followed by a 5-minute break to manage work stress."
        elif mood == "angry":
            return "Before responding to a work situation, write out what you want to say first, then edit it before sending."
    
    if "sleep" in lower_input or "tired" in lower_input or "insomnia" in lower_input:
        if mood in ["anxious", "stressed"]:
            return "Try the 4-7-8 breathing technique before bed: inhale for 4, hold for 7, exhale for 8 counts."
    
    if "friend" in lower_input or "relationship" in lower_input or "family" in lower_input:
        if mood == "sad":
            return "Consider writing a letter expressing your feelings - you don't have to send it, but it helps process emotions."
        elif mood == "angry":
            return "When conflicts arise, try using 'I feel' statements rather than accusations to express your perspective."
    
    # Otherwise, return a random suggestion from the appropriate mood category
    return random.choice(mood_resources)

# Function to save and analyze dynamic mood history with AI insights
def save_dynamic_mood(mood, user_input):
    # Initialize session state variables if they don't exist
    if 'dynamic_mood' not in st.session_state:
        st.session_state['dynamic_mood'] = []
    
    if 'mood_timestamps' not in st.session_state:
        st.session_state['mood_timestamps'] = []
    
    if 'user_inputs' not in st.session_state:
        st.session_state['user_inputs'] = []
    
    # Add current data to history
    st.session_state['dynamic_mood'].append(mood)
    st.session_state['mood_timestamps'].append(datetime.now())
    st.session_state['user_inputs'].append(user_input)
    
    # Keep history to a reasonable length (last 10 entries)
    if len(st.session_state['dynamic_mood']) > 10:
        st.session_state['dynamic_mood'] = st.session_state['dynamic_mood'][-10:]
        st.session_state['mood_timestamps'] = st.session_state['mood_timestamps'][-10:]
        st.session_state['user_inputs'] = st.session_state['user_inputs'][-10:]
    
    # Generate insights if we have enough data
    if len(st.session_state['dynamic_mood']) >= 3:
        feedback = analyze_mood_patterns(
            st.session_state['dynamic_mood'],
            st.session_state['mood_timestamps'],
            st.session_state['user_inputs']
        )
        return feedback
    
    return None

# Enhanced function to analyze mood patterns with more sophisticated insights
def analyze_mood_patterns(mood_history, timestamps=None, inputs=None):
    # Basic frequency analysis
    mood_counts = {mood: mood_history.count(mood) for mood in set(mood_history)}
    most_frequent_mood = max(mood_counts, key=mood_counts.get)
    
    # Detect trends (improving or worsening)
    mood_values = {
        'happy': 5,
        'neutral': 3,
        'anxious': 2,
        'stressed': 2,
        'sad': 1,
        'angry': 1
    }
    
    # Convert moods to numerical values
    mood_numbers = [mood_values.get(mood, 3) for mood in mood_history[-5:]]
    
    # Check if there's a trend (simplified)
    improving = all(mood_numbers[i] <= mood_numbers[i+1] for i in range(len(mood_numbers)-1))
    worsening = all(mood_numbers[i] >= mood_numbers[i+1] for i in range(len(mood_numbers)-1))
    
    # Generate insights based on patterns
    if improving and mood_history[-1] in ['happy', 'neutral']:
        return "I notice a positive trend in your mood! Whatever you're doing seems to be working well. 🌟"
    
    elif worsening and mood_history[-1] in ['sad', 'angry', 'stressed', 'anxious']:
        return "I've noticed your mood has been declining. Consider reaching out to a trusted friend or professional if things continue to feel challenging. 🌸"
    
    # Frequent mood swings
    if len(set(mood_history[-4:])) >= 3:
        return "I notice your emotions have been varying quite a bit. Remember that it's normal to experience a range of feelings. Mindfulness practices might help you feel more centered. 🧘‍♀️"
    
    # Consistent mood
    if len(set(mood_history[-4:])) == 1:
        mood = mood_history[-1]
        if mood in ['sad', 'anxious', 'stressed']:
            return "You've been consistently feeling low lately. Consider small daily actions that bring you joy, even if they seem insignificant. 🌱"
        elif mood == 'happy':
            return "You've been consistently positive! You're in a great place to tackle challenges and spread joy to others. 💫"
    
    # Default insights based on most frequent mood
    if most_frequent_mood == 'happy':
        return "Overall, you've been mostly positive lately! That's wonderful to see. 😊"
    elif most_frequent_mood == 'sad':
        return "You've been feeling down often. Remember that seeking support is a sign of strength, not weakness. 🌸"
    elif most_frequent_mood == 'stressed':
        return "Stress has been a recurring theme. Your well-being matters - prioritize self-care activities that help you relax. 🌿"
    elif most_frequent_mood == 'anxious':
        return "Anxiety has been present for you lately. Remember that your mind sometimes overestimates threats and underestimates your ability to cope. 🌈"
    else:
        return "You're maintaining a balanced mood overall. That's a healthy place to be! 🌱"

# Function to visualize mood history
def show_mood_visualization():
    if 'dynamic_mood' not in st.session_state or len(st.session_state['dynamic_mood']) < 2:
        st.info("Not enough mood data to visualize yet. Share how you're feeling a few more times.")
        return
    
    # Create mood history data
    moods = st.session_state['dynamic_mood']
    
    # Create a numeric representation for visualization
    mood_values = {
        'happy': 5,
        'neutral': 3, 
        'anxious': 2,
        'stressed': 2,
        'sad': 1,
        'angry': 1
    }
    
    # Convert to numeric values
    y_values = [mood_values.get(mood, 3) for mood in moods]
    
    # Create labels
    labels = [f"Entry {i+1}" for i in range(len(moods))]
    
    # Create a dictionary for Streamlit to display
    chart_data = pd.DataFrame({
        'Entry': labels,
        'Mood Value': y_values,
        'Mood': moods
    })
    
    st.subheader("Your Mood History")
    st.line_chart(chart_data.set_index('Entry')['Mood Value'])
    
    # Display the mood labels
    st.write("Mood history: " + " → ".join(moods))

# Settings for the app - persistent preferences
def save_settings():
    # Initialize settings in session state if they don't exist
    if 'app_settings' not in st.session_state:
        st.session_state['app_settings'] = {
            'voice_enabled': True,
            'voice_index': 0,
            'advanced_analysis': True,
            'show_visualization': True,
            'theme': 'Light'
        }
    
    # Create a settings section
    st.sidebar.subheader("Settings")
    
    # Voice settings
    st.session_state['app_settings']['voice_enabled'] = st.sidebar.checkbox(
        "Enable voice feedback", 
        value=st.session_state['app_settings'].get('voice_enabled', True)
    )
    
    if st.session_state['app_settings']['voice_enabled']:
        # Get available voices
        voices = engine.getProperty('voices')
        voice_names = [f"Voice {i+1}" for i in range(len(voices))]
        
        st.session_state['app_settings']['voice_index'] = st.sidebar.selectbox(
            "Select voice",
            range(len(voice_names)),
            format_func=lambda x: voice_names[x],
            index=st.session_state['app_settings'].get('voice_index', 0)
        )
    
    # Analysis settings
    st.session_state['app_settings']['advanced_analysis'] = st.sidebar.checkbox(
        "Use advanced mood analysis", 
        value=st.session_state['app_settings'].get('advanced_analysis', True)
    )
    
    # Visualization settings
    st.session_state['app_settings']['show_visualization'] = st.sidebar.checkbox(
        "Show mood history visualization", 
        value=st.session_state['app_settings'].get('show_visualization', True)
    )
    
    # Theme selection
    themes = ["Light", "Dark", "Calm", "Energetic"]
    st.session_state['app_settings']['theme'] = st.sidebar.selectbox(
        "App theme",
        themes,
        index=themes.index(st.session_state['app_settings'].get('theme', 'Light'))
    )
    
    return st.session_state['app_settings']

# Main function to run the app
def main():
    # Apply theme based on settings
    settings = save_settings()
    
    # Custom CSS based on theme
    theme = settings.get('theme', 'Light')
    if theme == 'Dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #262730;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme == 'Calm':
        st.markdown("""
        <style>
        .stApp {
            background-color: #E8F4F8;
            color: #2E4057;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme == 'Energetic':
        st.markdown("""
        <style>
        .stApp {
            background-color: #FFF7ED;
            color: #7D3C98;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("MoodLifterBot AI 😊")
    
    # Introduction and instructions
    st.write("Welcome to MoodLifterBot AI! Share how you're feeling, and I'll help lift your mood with personalized insights and support.")
    
    # Ask user how they would like to input their mood
    option = st.radio("How would you like to input your mood?", ('Text', 'Voice'))
    
    user_input = None
    
    if option == 'Text':
        # Text input for mood
        user_input = st.text_input("Share how you're feeling (e.g., 'I'm happy!', 'I'm feeling down', etc.):")
    elif option == 'Voice':
        # Voice input button
        if st.button("Click to speak"):
            user_input = listen_to_voice()
    
    # Process input if provided
    if user_input:
        with st.spinner("Analyzing your mood..."):
            # Use advanced or basic analysis based on settings
            if settings.get('advanced_analysis', True):
                mood = predict_emotion(user_input)
            else:
                mood = analyze_mood(user_input, use_ml=False)
            
            # Display detected mood with appropriate emoji
            mood_emojis = {
                "happy": "😊", "sad": "😔", "neutral": "😐", 
                "stressed": "😰", "angry": "😠", "anxious": "😟"
            }
            
            st.write(f"Detected mood: {mood} {mood_emojis.get(mood, '')}")
            
            # Get a quote and display it
            quote = random.choice(quotes[mood])
            st.write(f"**Inspirational Quote:** {quote}")
            
            # Save mood and get insights
            insight = save_dynamic_mood(mood, user_input)
            
            # Get AI feedback
            ai_message = ai_feedback(
                mood, 
                user_input, 
                st.session_state.get('dynamic_mood', [])
            )
            st.write(f"**AI Insight:** {ai_message}")
            
            # Get suggested resource
            suggested_resource = suggest_resource(
                mood, 
                user_input, 
                st.session_state.get('dynamic_mood', [])
            )
            st.write(f"**Suggested Activity:** {suggested_resource}")
            
            # Display mood pattern analysis if available
            if insight:
                st.write(f"**Mood Pattern Analysis:** {insight}")
            
            # Speak the quote if voice is enabled
            if settings.get('voice_enabled', True):
                speak_quote(quote, settings.get('voice_index', 0))
    
    # Show visualization if enabled and we have data
    if settings.get('show_visualization', True) and 'dynamic_mood' in st.session_state:
        show_mood_visualization()
    
    # Footer with additional resources
    st.sidebar.markdown("---")
    st.sidebar.subheader("Additional Resources")
    st.sidebar.markdown("""
    - [Mental Health America](https://www.mhanational.org/)
    - [Mindfulness Exercises](https://www.mindful.org/category/meditation/daily-practices/)
    - [Crisis Text Line](https://www.crisistextline.org/) - Text HOME to 741741
    """)

if __name__ == "__main__":
    main()