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

from whisper_recognition import WhisperRecognition

sd.default.device = (1, 1)  # or try (8, 8)

print("🔊 Available audio devices:")
print(sd.query_devices())

print("\n🎙️ Default Input Device:", sd.default.device[0])
print("🔈 Default Output Device:", sd.default.device[1])

# Load environment variables from .env file
load_dotenv()

print("Running AI_Chat.py in ONLINE-ONLY mode")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("wailo")

# Check internet connection
def check_internet():
    try:
        socket.create_connection(("www.google.com", 80), timeout=2)
        return True
    except OSError:
        return False

# Ensure internet is available
if not check_internet():
    print("❌ No internet connection. This script only supports ONLINE mode.")
    sys.exit(1)

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
set_api_key(os.getenv("ELEVENLABS_API_KEY"))
client = OpenAI()

# Personality system messages
PERSONALITIES = {
    "normal": "You are Wailo, a friendly and caring AI pet who loves helping children learn. You explain things in simple terms, show enthusiasm for learning, and always encourage curiosity. Keep responses child-friendly, warm, and engaging. You should speak in a gentle, nurturing way that makes children feel safe and supported."
}

def record_with_silence_detection(max_duration=60, samplerate=16000, silence_threshold=2.0):
    vad = webrtcvad.Vad()
    vad.set_mode(2)

    print("\nPreparing to record...")
    audio = []
    speech_detected = False
    speech_duration = 0
    chunk_duration = 0.03
    chunk_samples = int(chunk_duration * samplerate)
    max_chunks = int(max_duration / chunk_duration)

    with sd.InputStream(samplerate=samplerate, channels=2, dtype='int16') as stream:
        for _ in range(3):
            stream.read(chunk_samples)
        print("🎤 Ready! Speak now...")
        sleep(0.5)

        consecutive_silence = 0
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            audio.append(chunk)

            is_speech = vad.is_speech(chunk[:, 0].tobytes(), samplerate)
            if is_speech:
                if not speech_detected:
                    print("Recording...")
                speech_detected = True
                speech_duration += chunk_duration
                consecutive_silence = 0
            elif speech_detected and speech_duration > 0.3:
                consecutive_silence += 1
                if consecutive_silence >= int(silence_threshold / chunk_duration):
                    print("Silence detected, processing...")
                    break

    if not speech_detected:
        print("No speech detected.")
        return None

    return np.concatenate(audio).tobytes()

def transcribe_audio_from_memory_online(audio_bytes, samplerate=16000):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, 2)
            mono_audio = audio_array[:, 0]
            wf.writeframes(mono_audio.tobytes())

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



from pydub import AudioSegment
from pydub.playback import play

set_api_key(os.getenv("ELEVENLABS_API_KEY"))

def speak_text_online(text):
    print("Speaking as Wailo (online)...")
    try:
        audio_bytes = generate(text=text, voice="EXAVITQu4vr4xnSDxMaL")
        
        # Convert to audio segment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        
        # Play directly
        play(audio_segment)
        return True
    except Exception as e:
        print(f"ElevenLabs TTS failed: {e}")
        return False
    except Exception as e:
        print(f"Speech synthesis failed: {e}")
        return False


def is_audio_valid(audio_bytes, threshold=300):
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    amplitude = np.max(np.abs(audio_array))
    return amplitude > threshold

def ai_chat():
    print("\n=== Welcome to Wailo (Online Mode) ===")

    # Initial greeting
    initial_prompt = "Introduce yourself as a friendly AI assistant to a child who just started talking to you"
    response = ask_openai_online(initial_prompt)
    speak_text_online(response)

    while True:
        try:
            print("\nListening for input...")
            audio_bytes = record_with_silence_detection()
            if not audio_bytes or not is_audio_valid(audio_bytes):
                print("No speech detected, please try again.")
                continue

            user_input = transcribe_audio_from_memory_online(audio_bytes)
            if not user_input:
                print("Could not transcribe speech, please try again.")
                continue

            print(f"Transcribed input: {user_input}")

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Goodbye!")
                break

            response = ask_openai_online(user_input)
            speak_text_online(response)

        except KeyboardInterrupt:
            print("\nExiting upon user request.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print("An error occurred. Let's try again.")
            time.sleep(2)

if __name__ == "__main__":
    ai_chat()
