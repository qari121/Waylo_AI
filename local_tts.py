import os
import torch
import tempfile
import pygame
import pyttsx3
import logging
import numpy as np
from urllib.request import urlretrieve
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class LocalTTS:
    """TTS implementation that uses Silero TTS with fallback to pyttsx3"""

    def __init__(self):
        """Initialize the TTS engine"""
        self.pygame_initialized = False
        self.pyttsx3_engine = None
        self.silero_model = None
        self.silero_lock = Lock()
        self.model_path = None
        self.sample_rate = 48000
        self.speaker = 'en_0'  # Female voice
        self.speech_rate = .9  # Slower speech rate (0.8 = 80% of normal speed)
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=4096)
            self.pygame_initialized = True
            logger.info("Pygame audio initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            
        # Create cache directory
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".wailo_tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to load Silero TTS model
        if self._ensure_silero_model():
            logger.info("Silero TTS is ready to use")
        else:
            logger.warning("Silero TTS failed to initialize, will use fallback TTS")
        
        # Initialize fallback pyttsx3 engine
        try:
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', 150)
            self.pyttsx3_engine.setProperty('volume', 0.9)
            logger.info("Fallback pyttsx3 engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
    
    def _ensure_silero_model(self):
        """Download or load the Silero TTS model"""
        try:
            model_path = os.path.join(self.cache_dir, "silero_model.pt")
            
            # Check if we already have the model
            if os.path.exists(model_path):
                logger.info(f"Loading Silero model from cache: {model_path}")
                # Load the model
                with self.silero_lock:
                    self.model_path = model_path
                    self.silero_model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
                    self.silero_model.to('cpu')  # Ensure it's on CPU for broader compatibility
                return True
            else:
                logger.warning("Silero model not found in cache and cannot be downloaded in offline mode")
                return False
        except Exception as e:
            logger.error(f"Error loading Silero TTS model: {e}")
            return False

    @staticmethod
    def download_model(cache_dir=None):
        """
        Download the Silero TTS model for later offline use.
        This should be called when internet is available.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".wailo_tts_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
        model_path = os.path.join(cache_dir, "silero_model.pt")
        
        # Check if we already have the model
        if os.path.exists(model_path):
            print(f"Silero model already downloaded at: {model_path}")
            return True
        
        # Download the model
        try:
            print("Downloading Silero TTS model (one-time operation)...")
            torch.hub.download_url_to_file(
                'https://models.silero.ai/models/tts/en/v3_en.pt',
                model_path,
                progress=True
            )
            print(f"Model successfully downloaded to {model_path}")
            return True
        except Exception as e:
            print(f"Error downloading Silero model: {e}")
            return False
    
    def say(self, text):
        """Speak the given text using the preferred TTS engine"""
        if not text:
            return False
        
        # Try using Silero if available
        if self.silero_model is not None and self.pygame_initialized:
            try:
                return self._speak_with_silero(text)
            except Exception as e:
                logger.error(f"Silero TTS failed: {e}")
                return self._speak_with_pyttsx3(text)
        else:
            return self._speak_with_pyttsx3(text)
    
    def _speak_with_silero(self, text):
        """Generate speech using Silero and play it with pygame"""
        # Create a temporary file for the audio output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate speech with Silero
            with self.silero_lock:
                logger.info(f"Generating speech with Silero TTS: '{text}'")
                audio = self.silero_model.apply_tts(
                    text=text,
                    speaker=self.speaker,
                    sample_rate=self.sample_rate
                )
            
            # Slow down the speech if needed
            if self.speech_rate < 0.99:
                # Calculate new length for slower speech
                new_length = int(audio.shape[0] / self.speech_rate)
                # Use interpolation to stretch the audio
                import torch.nn.functional as F
                audio = F.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                logger.info(f"Adjusted speech rate to {self.speech_rate:.2f}x")
            
            # Convert to numpy array and save as WAV
            audio_np = audio.numpy()
            
            # Use scipy to write the WAV file
            from scipy.io.wavfile import write as write_wav
            write_wav(temp_path, self.sample_rate, audio_np)
            
            # Play the generated audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return True
        except Exception as e:
            logger.error(f"Silero TTS error: {e}")
            # Clean up temporary file if there was an error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise
    
    def _speak_with_pyttsx3(self, text):
        """Speak text using pyttsx3 as fallback"""
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                return True
            except Exception as e:
                logger.error(f"pyttsx3 error: {e}")
        return False
    
    
    # These methods allow for compatibility with pyttsx3 API
    def runAndWait(self):
        """Compatibility with pyttsx3 API"""
        pass 