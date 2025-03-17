from enum import Enum
from transformers import pipeline
import torch
import openai
from config import OPENAI_API_KEY
import langdetect
from typing import List, Tuple, Optional

class InterestField(Enum):
    # Academic & Learning
    SCIENCE = "science"
    MATH = "math"
    READING = "reading"
    
    # Technology & Vehicles
    COMPUTERS = "computers"
    VEHICLES = "vehicles"
    SPACE = "space"
    ROBOTICS = "robotics"
    
    # Nature & Animals
    ANIMALS = "animals"
    PLANTS = "plants"
    WEATHER = "weather"
    
    # Arts & Expression
    ARTS = "arts"
    MUSIC = "music"
    DANCE = "dance"
    
    # Social & Emotional
    FEELINGS = "feelings"
    FRIENDS = "friends"
    FAMILY = "family"
    
    # Activities
    SPORTS = "sports"
    GAMES = "games"
    STORIES = "stories"

class InterestTracker:
    def __init__(self):
        """Initialize with OpenAI API key from environment"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            # Use a smaller zero-shot model
            self.classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  # Smaller model
                device=-1  # Force CPU usage
            )
        except Exception as e:
            print(f"Warning: Could not load interest model: {e}")
            self.classifier = None
        
        # Define interest categories with child-friendly descriptions
        self.interest_descriptions = {
            InterestField.SCIENCE: "learning about how things work, doing experiments",
            InterestField.SPACE: "stars, planets, rockets, astronauts, space exploration",
            InterestField.VEHICLES: "cars, trucks, trains, airplanes, boats",
            InterestField.ANIMALS: "pets, wild animals, dinosaurs, zoo animals",
            InterestField.MUSIC: "songs, singing, musical instruments, dancing",
            InterestField.STORIES: "books, reading, telling stories, movies",
            InterestField.FEELINGS: "emotions, sharing feelings, talking about experiences",
            InterestField.FRIENDS: "playing with friends, making new friends",
            InterestField.FAMILY: "family activities, parents, siblings",
            InterestField.SPORTS: "physical activities, games, exercise",
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text with better handling of edge cases"""
        if not text or text.isspace():
            return "en"
            
        # Only attempt language detection if we have enough text
        if len(text.split()) < 3:  # Require at least 3 words
            return "en"
            
        try:
            # Add bias towards English for ambiguous cases
            lang = langdetect.detect_langs(text)
            # If English probability is close enough to the top language, prefer English
            if any(l.lang == 'en' and l.prob > 0.2 for l in lang):
                return "en"
            # For German, be more lenient as well
            if any(l.lang == 'de' and l.prob > 0.2 for l in lang):
                return "de"
            # Return the most probable language if it's very confident
            if lang[0].prob > 0.8:
                return lang[0].lang
            return "en"  # Default to English if unsure
        except:
            return "en"  # Default to English on any error

    def detect_interests(self, text: str) -> Optional[List[Tuple[str, float]]]:
        """Detect main topics/interests from text"""
        if not text:
            return []

        # Check language
        lang = self.detect_language(text)
        if lang not in ['en', 'de']:
            print(f"\nDetected non-English/German language: {lang}")
            print("Please use English or German for better interaction.")
            return None

        try:
            print("\n=== Interest Analysis ===")
            print(f"Analyzing text: {text}")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """Identify the main topics/interests in this text.
                    Consider topics like: learning, science, technology, arts, emotions, social interaction, etc.
                    Return a JSON array of objects containing:
                    - 'topic': the main topic or interest
                    - 'relevance': float 0-1 indicating how central this topic is
                    List only the most relevant topics (max 3)."""},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            results = eval(response.choices[0].message.content)
            print(f"OpenAI Response: {results}")
            
            interests = [(item['topic'], item['relevance'] * 10) for item in results]
            print(f"Detected interests: {interests}")
            return interests
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return []

    def get_interest_details(self, text: str) -> list[dict]:
        """Get detailed interest analysis including confidence and explanation"""
        try:
            result = self.classifier(
                text,
                list(self.interest_descriptions.values()),
                multi_label=True
            )
            
            details = []
            for label, score in zip(self.interest_descriptions.keys(), result['scores']):
                if score > 0.3:
                    details.append({
                        'interest': label.value,
                        'intensity': round(score * 10, 1),
                        'description': self.interest_descriptions[label]
                    })
            
            return sorted(details, key=lambda x: x['intensity'], reverse=True)
            
        except Exception as e:
            print(f"Error getting interest details: {e}")
            return []

    def _ml_detect_interests(self, text: str) -> list[tuple[InterestField, float]]:
        """ML-based interest detection"""
        # Skip empty or very short texts
        if not text or len(text.split()) < 2:
            return []

        # Get candidate labels and their descriptions
        candidate_labels = list(self.interest_descriptions.values())
        
        # Classify text against all interest categories
        result = self.classifier(
            text,
            candidate_labels,
            multi_label=True
        )
        
        # Map scores back to interest fields
        interests = []
        labels = list(self.interest_descriptions.keys())
        for label, score in zip(labels, result['scores']):
            if score > 0.3:  # Minimum confidence threshold
                intensity = round(score * 10, 1)
                interests.append((label, intensity))
        
        return sorted(interests, key=lambda x: x[1], reverse=True) 