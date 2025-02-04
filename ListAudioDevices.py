import sounddevice as sd
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import numpy as np
import time
import io
import wave
import logging
import os
import threading
from queue import Queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def list_audio_devices():
    """List all available audio input and output devices"""
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']}")
    print("-" * 50)
    return devices

def initialize_speech_engine():
    """Initialize text-to-speech engine"""
    engine = pyttsx3.init()
    return engine

class VoiceChat:
    def __init__(self, api_key):
        self.client = OpenAI(api_key="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")
        self.logger = logging.getLogger(__name__)
        self.should_stop_speaking = False
        self.is_speaking = False
        self.command_queue = Queue()
        self.running = True

    def listen_for_commands(self, input_device):
        """Continuously listen for commands"""
        while self.running:
            try:
                # Use shorter duration for commands
                audio_data, samplerate = record_audio(input_device, duration=1)
                
                # Normalize and check audio level
                audio_data = np.nan_to_num(audio_data)
                rms = np.sqrt(np.mean(audio_data**2))
                
                if rms > 0.01:  # Adjust this threshold as needed
                    transcription = self.speech_to_text(audio_data, samplerate)
                    command = transcription.text.lower().strip()
                    
                    # Check if the command starts with 'wait' or 'stop'
                    if command.startswith('wait') or command.startswith('stop'):
                        command = command.split()[0]  # Take just the first word
                        print(f"\nCommand received: {command}")
                        if command == 'wait':
                            self.should_stop_speaking = True
                            time.sleep(0.5)  # Give time for the speech to stop
                        elif command == 'stop':
                            self.running = False
                            self.should_stop_speaking = True
                
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                self.logger.error(f"Command listening error: {str(e)}")
                continue

    def speak(self, text):
        """Convert text to speech using OpenAI's TTS"""
        try:
            self.is_speaking = True
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="mp3"
            )
            audio_bytes = response.content
            
            temp_file = "temp_audio.mp3"
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            
            if os.name == 'posix':  # For Unix/Linux/MacOS
                import subprocess
                process = subprocess.Popen(['afplay', temp_file])
                while process.poll() is None:  # While audio is playing
                    if self.should_stop_speaking:
                        process.terminate()
                        process.kill()  # Force kill if needed
                        self.should_stop_speaking = False
                        print("\nSpeech interrupted. Listening...")
                        break
                    time.sleep(0.1)  # Add small delay to prevent CPU overuse
            elif os.name == 'nt':  # For Windows
                import winsound
                winsound.PlaySound(temp_file, winsound.SND_FILENAME)
            
            try:
                os.remove(temp_file)
            except:
                pass
            
        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            self.is_speaking = False

    def speech_to_text(self, audio_data, samplerate, translate=False):
        """Convert speech to text using OpenAI's Whisper model"""
        try:
            # Normalize audio data to prevent overflow
            audio_data = np.nan_to_num(audio_data)  # Replace NaN with 0
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9  # Scale to avoid overflow
            
            # Create a temporary WAV file in memory
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(samplerate)
                wav_data = (audio_data * 32767).astype(np.int16).tobytes()
                wav_file.writeframes(wav_data)
            
            byte_io.seek(0)
            audio_file = ("audio.wav", byte_io, "audio/wav")
            
            # Add timeout parameter and handle different file sizes
            if translate:
                response = self.client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    timeout=30  # 30 seconds timeout
                )
            else:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    timeout=30  # 30 seconds timeout
                )
            
            # If we get here, we have a successful response
            return response

        except TimeoutError as e:
            self.logger.error("Request timed out")
            return type('TranscriptionError', (), {'text': "Sorry, the request timed out. Please try again."})
        except Exception as e:
            self.logger.error(f"Speech to text error: {str(e)}")
            return type('TranscriptionError', (), {'text': f"Error in transcription: {str(e)}"})

    def get_gpt_response(self, text):
        """Get response from GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT Error: {str(e)}")
            return "Sorry, I couldn't process that."

def record_audio(input_device, duration=5):
    """Record audio from specified input device"""
    try:
        samplerate = 16000  # Lower sample rate
        channels = 1
        
        # Configure the stream
        with sd.InputStream(samplerate=samplerate, 
                          device=input_device,
                          channels=channels, 
                          dtype=np.float32,
                          blocksize=1024,  # Smaller blocksize
                          latency='low') as stream:  # Request low latency
            
            frames = []
            for _ in range(int(samplerate * duration / 1024)):
                data, overflowed = stream.read(1024)
                if overflowed:
                    logger.warning("Audio buffer overflow")
                frames.append(data)
            
        recording = np.concatenate(frames)
        return recording.flatten(), samplerate
        
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        raise

def main():
    voice_chat = VoiceChat(api_key='your-api-key-here')
    devices = list_audio_devices()
    input_device = int(input("\nSelect input device number: "))
    
    print("\nOptions:")
    print("1. Regular conversation")
    print("2. Translate any language to English")
    mode = int(input("Select mode (1 or 2): "))
    
    print("\nVoice Chat Started!")
    print("Commands: 'wait' to interrupt, 'stop' to exit")
    
    # Start the command listening thread
    command_thread = threading.Thread(
        target=voice_chat.listen_for_commands, 
        args=(input_device,),
        daemon=True
    )
    command_thread.start()

    while voice_chat.running:
        try:
            print("\nListening for main input...")
            # Only record main input when not speaking
            if not voice_chat.is_speaking:
                audio_data, samplerate = record_audio(input_device, duration=5)
                transcription = voice_chat.speech_to_text(audio_data, samplerate, translate=(mode == 2))
                user_input = transcription.text.lower()
                print(f"You said: {user_input}")

                if user_input == "stop":
                    voice_chat.running = False
                    break
                
                # Get and speak GPT response
                gpt_response = voice_chat.get_gpt_response(user_input)
                print(f"ChatGPT: {gpt_response}")
                voice_chat.speak(gpt_response)
            
        except KeyboardInterrupt:
            voice_chat.running = False
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

    print("Shutting down...")
    voice_chat.running = False
    command_thread.join(timeout=1)

if __name__ == "__main__":
    main()


