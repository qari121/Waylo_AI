import os
import torch
import logging
import datetime
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from rkllm_wrapper import RkllmWrapper

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"wailo_responses_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("wailo")

class WailoOfflineModel:
    """
    Handles offline AI responses for Wailo - a friendly AI pet for children.
    Uses TinyLlama with the proper chat format.
    """
    
    def __init__(self, model_path="./meta-llama_Llama-3.2-3B-rk3588.rkllm"):
        """
        Initialize the Wailo offline model with proper format.
        """
        self.model_path = model_path
        self.rkllm = None
        self.first_message = True
        
        # Load model
        logger.info(f"Initializing Wailo with RKLLM from: {model_path}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from the specified path."""
        try:
            logger.info(f"Loading RKLLM model from {self.model_path}...")
            self.rkllm = RkllmWrapper(self.model_path)
            logger.info("RKLLM model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading RKLLM model: {str(e)}")
            self.rkllm = None
    
    def generate_response(self, question: str) -> str:
        """Generate a response using the RKLLM model with proper format"""
        # Log the input
        logger.info(f"INPUT: {question}")
        
        if not self.is_ready():
            return "I'm having trouble thinking right now. Could we talk again in a little bit?"
        
        try:
            # Handle first message with a simple greeting
            if self.first_message:
                self.first_message = False
                first_greeting = "Hi! I'm Wailo, your friendly AI pet. What would you like to talk about today?"
                logger.info(f"FIRST MESSAGE: {first_greeting}")
                return first_greeting
            
            # Clean up the input
            question = question.strip()
            if not question:
                return "I'm listening! What would you like to talk about?"
            
            # Format for RKLLM
            system_message = "You are Wailo, a friendly AI pet for children. Keep your responses SHORT, simple and complete."
            prompt = f"<s>[INST] {system_message}\n\n{question} [/INST]"
            
            # Generate response
            logger.info("Starting generation...")
            generation_start = datetime.datetime.now()
            
            response = self.rkllm.generate_response(prompt)
            
            generation_time = (datetime.datetime.now() - generation_start).total_seconds()
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Clean up the response
            response = self._clean_response(response)
            
            logger.info(f"MODEL RESPONSE: {response}")
            logger.info(f"Generation time: {generation_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'd be happy to chat with you! What would you like to talk about?"
    
    def _clean_response(self, text: str, is_list: bool = False) -> str:
        """Clean the generated response to ensure completeness"""
        # Log raw response for debugging
        logger.debug(f"Raw response: {text}")
        
        # Cut off at known markers for the chat format
        markers = ["<|system|>", "<|user|>", "<|assistant|>"]
        for marker in markers:
            if marker in text:
                text = text.split(marker)[0]
        
        # Clean up extraneous newlines and spaces
        text = text.strip()
        
        # For list responses, ensure the list is properly completed
        if is_list and any(line.strip().startswith(str(i)+".") for line in text.split('\n') for i in range(1, 10)):
            # Count the number of list items
            list_items = [line for line in text.split('\n') if re.match(r'^\d+\.', line.strip())]
            
            # If we have an incomplete list (the last item is just a number)
            if list_items and re.match(r'^\d+\.\s*$', list_items[-1].strip()):
                # Remove the incomplete item
                text = '\n'.join(line for line in text.split('\n') 
                                if not re.match(r'^\d+\.\s*$', line.strip()))
        
        # Ensure we have a complete sentence by keeping text up to the last period, 
        # question mark, or exclamation point, if one exists
        sentences_end = [pos for pos, char in enumerate(text) if char in ['.', '!', '?']]
        if sentences_end:
            # Get the position of the last sentence-ending punctuation
            last_sentence_end = sentences_end[-1]
            # Only truncate if we're not cutting off more than half
            if last_sentence_end > len(text) * 0.5:  # Ensure we're not cutting off more than half
                text = text[:last_sentence_end + 1]
        
        # If we still don't have a sentence ending, add a period
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        # Ensure we have a response
        if not text or len(text) < 2:
            text = "I'd love to help! What would you like to talk about?"
            
        return text
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready to use."""
        return self.rkllm is not None


# For testing
if __name__ == "__main__":
    logger.info("Starting Wailo offline model in standalone mode")
    wailo = WailoOfflineModel()
    
    if wailo.is_ready():
        logger.info("Model loaded successfully and is ready for use.")
    else:
        logger.error("Model is not ready. Please check the logs for details.") 