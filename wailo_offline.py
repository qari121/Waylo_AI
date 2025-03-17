import os
import torch
import logging
import datetime
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    
    def __init__(self, model_path: str = "./tinyllama-chat"):
        """
        Initialize the Wailo offline model with proper format.
        """
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.first_message = True
        
        # Load model
        logger.info(f"Initializing Wailo with TinyLlama from: {model_path}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from the specified path."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Check if model file exists
            model_file = os.path.join(self.model_path, "model.safetensors")
            if not os.path.exists(model_file):
                logger.error(f"Model file not found at {model_file}")
                return
            
            # Load tokenizer with low memory usage
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with low memory usage and fp16 for better performance
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Put model in evaluation mode
            self.model.eval()
            logger.info(f"TinyLlama model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, question: str) -> str:
        """Generate a response using the TinyLlama model with proper format"""
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
            
            # Format messages for TinyLlama using the correct chat format
            system_message = "You are Wailo, a friendly AI pet for children. Keep your responses SHORT (1-2 sentences), simple and complete."
            
            # Format according to TinyLlama's expected format
            prompt = f"<|system|>\n{system_message}\n<|user|>\n{question}\n<|assistant|>\n"
            
            logger.info(f"Using prompt format: {prompt}")
            
            # Determine if this is likely a question that needs a list response
            needs_list = any(phrase in question.lower() for phrase in 
                             ["how to", "steps", "guide", "instructions", "help me", "ways to", "can you help"])
            
            # Use more tokens for list-type responses
            max_tokens = 150 if needs_list else 80
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            logger.info("Starting generation...")
            generation_start = datetime.datetime.now()
            
            # Generate with more tokens for list responses
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,  # Use more tokens for list responses
                    do_sample=True,  # Enable sampling for more varied responses
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3,  # Prevent repetition
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask  # Add attention mask
                )
            
            generation_time = (datetime.datetime.now() - generation_start).total_seconds()
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Extract the generated text
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up the response - extra cleanup for list responses
            response = self._clean_response(response, is_list=needs_list)
            
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
        return self.model is not None and self.tokenizer is not None


# For testing
if __name__ == "__main__":
    logger.info("Starting Wailo offline model in standalone mode")
    wailo = WailoOfflineModel()
    
    if wailo.is_ready():
        logger.info("Model loaded successfully and is ready for use.")
    else:
        logger.error("Model is not ready. Please check the logs for details.") 