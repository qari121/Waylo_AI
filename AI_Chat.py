import os
import wave
import openai
import sounddevice as sd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS
import io
import webrtcvad
import soundfile as sf
from scipy.io import wavfile
import tempfile
from elevenlabs import generate, play, set_api_key
from sentiment_tracker import SentimentTracker, Sentiment
from textblob import TextBlob
from api_client import WailoAPI
from time import sleep
from elevenlabs.api.error import APIError
from interest_tracker import InterestTracker, InterestField
from wailo_offline import WailoOfflineModel as OfflineModelHandler
import json
import pyttsx3
import sys
import logging
import threading
import queue
import pygame
import socket
import time
import subprocess
import random

# Import whisper recognition (only for offline mode)
from whisper_recognition import WhisperRecognition

# Load environment variables from .env file
load_dotenv()

print("Running updated AI_Chat.py")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("wailo")

# Check internet connection
def check_internet():
    try:
        # Try to connect to a well-known website
        socket.create_connection(("www.google.com", 80), timeout=2)
        return True
    except OSError:
        pass
    return False

# Initialize components based on internet availability
HAS_INTERNET = check_internet()
if HAS_INTERNET:
    # Initialize online components
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # client = OpenAI()
    # ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    # set_api_key(ELEVENLABS_API_KEY)
    print("Internet connection detected, using online services")
else:
    print("No internet connection detected, running in offline mode")
    client = None

# Add this import instead
try:
    from local_tts import LocalTTS
except ImportError:
    print("Local TTS not available, falling back to basic TTS") 
    LocalTTS = None

# Initialize TTS engine for offline mode (replace the existing TTS_ENGINE initialization)
def initialize_offline_tts():
    try:
        # Use cross-platform LocalTTS
        if LocalTTS is not None:
            print("Initializing Local TTS engine...")
            return LocalTTS()
        
        # Fallback to pyttsx3
        print("Falling back to pyttsx3...")
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        return tts_engine
    except Exception as e:
        print(f"Error initializing offline TTS: {e}")
        return None

TTS_ENGINE = None if HAS_INTERNET else initialize_offline_tts()

# Personality system messages
PERSONALITIES = {
    "normal": "You are Wailo, a friendly and caring AI pet who loves helping children learn. You explain things in simple terms, show enthusiasm for learning, and always encourage curiosity. Keep responses child-friendly, warm, and engaging. You should speak in a gentle, nurturing way that makes children feel safe and supported."
}

def record_with_silence_detection(max_duration=60, samplerate=16000, silence_threshold=2.0):
    """
    Records audio from the microphone until silence is detected.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # More aggressive VAD (0-3, higher = more aggressive)

    print("\nPreparing to record...")
    audio = []
    silence_counter = 0
    speech_detected = False
    min_speech_duration = 0.3  # Reduced minimum speech duration
    speech_duration = 0
    chunk_duration = 0.03
    chunk_samples = int(chunk_duration * samplerate)
    max_chunks = int(max_duration / chunk_duration)

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        # Clear initial buffer and wait
        for _ in range(3):
            stream.read(chunk_samples)
        
        print("🎤 Ready! Speak now...")
        sleep(0.5)
        
        consecutive_silence = 0  # Track consecutive silence chunks
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            audio.append(chunk)

            # Detect speech or silence
            is_speech = vad.is_speech(chunk.tobytes(), samplerate)
            
            if is_speech:
                if not speech_detected:
                    print("Recording...")
                speech_detected = True
                speech_duration += chunk_duration
                silence_counter = 0
                consecutive_silence = 0
            elif speech_detected and speech_duration > min_speech_duration:
                consecutive_silence += 1
                if consecutive_silence >= int(silence_threshold / chunk_duration):
                    print("Silence detected, processing...")
                    break

    if not speech_detected:
        print("No speech detected.")
        return None

    audio_data = np.concatenate(audio)
    return audio_data.tobytes()

def transcribe_audio_from_memory_online(audio_bytes, samplerate=16000):
    """Transcribes audio using OpenAI's Whisper API (online only)"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_bytes)
        
        try:
            with open(temp_wav.name, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            os.unlink(temp_wav.name)
            return transcript.text
        except Exception as e:
            print(f"Online transcription error: {e}")
            os.unlink(temp_wav.name)
            return ""

def ask_openai_online(question, personality="normal"):
    """Generate response using OpenAI's API (online only)"""
    print("Using online GPT model")
    system_prompt = PERSONALITIES.get(personality, PERSONALITIES["normal"])
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "I'm sorry, I'm having trouble thinking right now. Let's try again in a moment."

def speak_text_online(text):
    """Uses online TTS services (ElevenLabs/gTTS)"""
    print("Speaking as Wailo (online)...")
    
    # Try ElevenLabs first
    voice_id = "EXAVITQu4vr4xnSDxMaL"
    try:
        audio = generate(text=text, voice=voice_id)
        play(audio)
        return True
    except APIError as e:
        print("ElevenLabs unavailable, using gTTS fallback...")
        try:
            tts = gTTS(text=text, lang='en')
            temp_file = "temp_speech.mp3"
            tts.save(temp_file)
            os.system(f"afplay {temp_file}")
            os.remove(temp_file)
            return True
        except Exception as e:
            print(f"Speech synthesis failed: {e}")
            return False

def speak_text_offline(text, personality=None):
    """Use offline TTS to speak the given text"""
    if not text:
        return
        
    logger.info(f"Speaking as Wailo (offline)...")
    print("Speaking as Wailo (offline)...")
    
    if not TTS_ENGINE:
        return False
        
    try:
        # Check if we're using LocalTTS
        if isinstance(TTS_ENGINE, LocalTTS):
            # Use LocalTTS
            return TTS_ENGINE.say(text)
        else:
            # Using pyttsx3
            TTS_ENGINE.say(text)
            TTS_ENGINE.runAndWait()
            return True
    except Exception as e:
        logger.error(f"Offline TTS error: {e}")
        # Try emergency fallback if we have a critical error
        try:
            os.system(f'echo "{text}" | espeak')
            return True
        except:
            return False

def is_audio_valid(audio_bytes, threshold=300):
    """Check if the recorded audio has sufficient amplitude"""
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    amplitude = np.max(np.abs(audio_array))
    return amplitude > threshold

class WailoInteractionTracker:
    def __init__(self, mac_address: str):
        if not mac_address:
            raise ValueError("MAC address is required")
            
        self.mac_address = mac_address
        self.character = "friendly AI assistant"
        
        # Initialize offline model only in offline mode
        self.offline_model = None
        if not HAS_INTERNET:
            print("Attempting to initialize offline model...")
            retry_count = 0
            max_retries = 2
            
            while retry_count < max_retries and self.offline_model is None:
                try:
                    self.offline_model = OfflineModelHandler()
                    if self.offline_model.model is None:
                        print(f"Offline model initialization failed, attempt {retry_count + 1}/{max_retries}")
                        self.offline_model = None
                except Exception as e:
                    print(f"Error during model initialization (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                
            if self.offline_model is None:
                print("Warning: Failed to initialize offline model after all retries")
            
        self.is_offline = not HAS_INTERNET
        
        # Only initialize online components if we have internet
        if HAS_INTERNET:
            self.api = WailoAPI()
            self.toy_id = None
            self.user_id = None
            self.sentiment_tracker = SentimentTracker()
            self.interest_tracker = InterestTracker()
        
    def initialize(self):
        """Initialize tracker with toy info"""
        if self.is_offline:
            print("Operating in offline mode")
            return True
            
        try:
            toy_info = self.api.get_toy_info(self.mac_address)
            if toy_info and toy_info.get('data'):
                toy_data = toy_info['data']
                self.toy_id = toy_info['id']
                self.user_id = toy_data['user_id']
                self.character = toy_data['charecter']
                print(f"Initialized Wailo as a {self.character}")
                return toy_data['enabled']
        except Exception as e:
            print(f"Error connecting to online service: {e}")
            self.is_offline = True
            print("Switching to offline mode")
        return True

    def analyze_interaction(self, request_text: str, response_text: str, request_id: str, response_id: str):
        """Analyze the interaction for sentiment and interests (online only)"""
        if self.is_offline:
            return
            
        # Analyze sentiment
        sentiment_result = self.sentiment_tracker.analyze_sentiment(request_text)
        if sentiment_result is None or sentiment_result[0] is None:
            # Non-English/German detected, ask user to use English or German
            response = "I can understand English and German. Could you please speak to me in one of these languages?"
            print("\nLanguage Notice:", response)  # Print the notice
            self.api.log_response(response, self.mac_address)
            speak_text_online(response)  # Make sure to vocalize the response
            return
        
        sentiment, intensity = sentiment_result
        self.api.log_sentiment(
            sentiment.value,
            self.mac_address,
            request_id,
            response_id,
            intensity
        )

        # Analyze interests
        interests = self.interest_tracker.detect_interests(request_text)
        if interests is None:
            # Non-English/German detected - we already handled this above
            return
        
        # Log detected interests
        for topic, relevance in interests:
            self.api.log_interest(
                topic.lower(),  # Convert to lowercase for consistency
                self.mac_address,
                request_id,
                response_id,
                relevance
            )

def speak_text(text, personality=None):
    """Unified interface for speaking text using either online or offline TTS"""
    if HAS_INTERNET:
        speak_text_online(text)
    else:
        speak_text_offline(text, personality)

# Main chat function
def ai_chat():
    # Initialize components
    global HAS_INTERNET  # Add this to allow changing HAS_INTERNET
    
    print("Running updated ListAudioDevices.py")  # Keep this placeholder
    
    # Choose the right approach based on internet connectivity
    if HAS_INTERNET:
        # ONLINE MODE - use cloud services exclusively
        
        # Initial greeting using OpenAI
        initial_prompt = "Introduce yourself as a friendly AI assistant to a child who just started talking to you"
        response = ask_openai_online(initial_prompt)
        
        # Text to speech using online services
        speak_text_online(response)
        
        # Main conversation loop
        while True:
            try:
                print("\nListening for input (online mode)...")
                
                # Record and transcribe using OpenAI
                audio_bytes = record_with_silence_detection()
                if not audio_bytes or not is_audio_valid(audio_bytes):
                    print("No speech detected, please try again.")
                    continue
                
                user_input = transcribe_audio_from_memory_online(audio_bytes)
                if not user_input:
                    print("Could not transcribe speech, please try again.")
                    continue
                
                print(f"Transcribed input: {user_input}")
                    
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("Goodbye!")
                    break
                
                # Generate response using OpenAI GPT
                response = ask_openai_online(user_input)
                
                # Text to speech using online services
                speak_text_online(response)
                
            except KeyboardInterrupt:
                print("\nExiting upon user request.")
                break
            except Exception as e:
                logger.error(f"Error in online chat loop: {e}")
                print("Sorry, I encountered an error. Let's try again.")
    
    else:
        # OFFLINE MODE - use local models exclusively
        
        # Initialize local models
        print("Initializing offline models...")
        whisper = WhisperRecognition(model_name="openai/whisper-tiny.en", force_download=True)
        wailo_model = OfflineModelHandler()
        
        # Initial greeting using offline model
        initial_prompt = "Introduce yourself as a friendly AI assistant to a child who just started talking to you"
        response = wailo_model.generate_response(initial_prompt)
        
        # Text to speech using offline TTS
        speak_text_offline(response)
        
        # For tracking interaction attempts to suggest online mode
        interaction_attempts = 0
        suggest_online_after = random.randint(1, 5)
        
        # Main conversation loop
        while True:
            try:
                print("\n=== Waiting for your question ===")
                
                # Increment interaction attempts
                interaction_attempts += 1
                
                # Show online suggestion after random number of turns
                if interaction_attempts >= suggest_online_after:
                    # Speak the tip instead of just printing it
                    online_tip = "For better voice recognition and faster responses, try connecting to the internet! When online, I can use more powerful cloud-based services."
                    print("\n💡 TIP: " + online_tip)
                    speak_text_offline(online_tip)
                    
                    # Reset counter and get a new random threshold
                    interaction_attempts = 0
                    suggest_online_after = random.randint(1, 5)
                
                # Get speech input using local Whisper with improved dynamic recording
                user_input = whisper.transcribe()
                
                # Add stronger validation to make sure we have a real input
                if not user_input or len(user_input.strip()) < 2:  # At least 2 chars
                    print("\nCouldn't detect speech. Please try speaking louder or check your microphone.")
                    continue
                
                print(f"Transcribed input: {user_input}")
                    
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("Goodbye!")
                    break
                
                # Check if user wants to go online after hearing the tip
                if user_input.lower() in ["connect to internet", "go online", "connect online", "use internet", "switch to online", "yes connect", "yes go online"]:
                    print("Checking internet connection...")
                    if check_internet():
                        print("Internet connection detected! Switching to online mode...")
                        # Text to speech to confirm
                        speak_text_offline("Great! I'm switching to online mode for better performance.")
                        # Update internet status
                        HAS_INTERNET = True
                        # Restart in online mode
                        ai_chat()
                        return
                    else:
                        print("Still no internet connection available. Continuing in offline mode.")
                        speak_text_offline("I couldn't connect to the internet. Let's continue in offline mode.")
                        # Generate a response and continue
                        print("Thinking...")
                        response = wailo_model.generate_response("Let's continue our conversation. " + user_input)
                        speak_text_offline(response)
                        continue
                
                # Generate response using offline model
                print("Thinking...")
                response = wailo_model.generate_response(user_input)
                
                # Text to speech using offline TTS
                speak_text_offline(response)
                
            except KeyboardInterrupt:
                print("\nExiting upon user request.")
                break
            except Exception as e:
                logger.error(f"Error in offline chat loop: {e}")
                print("Sorry, I encountered an error. Let's try again.")
                # Add a delay here too to prevent rapid error loops
                time.sleep(2)

if __name__ == "__main__":
    ai_chat()