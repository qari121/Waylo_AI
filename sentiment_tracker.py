from enum import Enum
from openai import OpenAI
from config import OPENAI_API_KEY
import langdetect
from textblob import TextBlob

class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class SentimentTracker:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text with better handling of edge cases"""
        if not text or text.isspace() or text.lower() == "session start":
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

    def analyze_sentiment(self, text: str) -> tuple[Sentiment, float]:
        """Analyze text sentiment using OpenAI"""
        if not text or text.lower() == "session start":
            return Sentiment.NEUTRAL, 5.0

        # Check language
        lang = self.detect_language(text)
        if lang not in ['en', 'de']:
            print(f"\nDetected non-English/German language: {lang}")
            print("Please use English or German for better interaction.")
            return None, None

        try:
            print(f"\n=== Sentiment Analysis ===")
            print(f"Analyzing text: {text}")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """Analyze the emotional tone of this text.
                    Return a JSON object with:
                    - 'sentiment': either 'positive', 'neutral', or 'negative'
                    - 'intensity': float 0-1 indicating strength of the emotion"""},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            result = eval(response.choices[0].message.content)
            print(f"OpenAI Response: {result}")
            
            sentiment = Sentiment(result['sentiment'])
            intensity = result['intensity'] * 10  # Convert to 0-10 scale
            
            final_result = (sentiment, round(intensity, 1))
            print(f"Final sentiment: {final_result}")
            return final_result
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return Sentiment.NEUTRAL, 5.0

    def get_all_emotions(self, text: str) -> list[tuple[Sentiment, float]]:
        """Get all detected emotions with their intensities using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an emotion analyzer. Analyze the emotional content of the text and respond with a JSON array of objects, each containing 'emotion' and 'intensity' (0-1). List up to 3 emotions present in the text."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            results = eval(response.choices[0].message.content)
            emotions = []
            
            for result in results:
                emotion = result['emotion'].lower()
                intensity = result['intensity'] * 10  # Convert to 0-10 scale
                sentiment = self.emotion_mapping.get(emotion, Sentiment.NEUTRAL)
                emotions.append((sentiment, round(intensity, 1)))
            
            return sorted(emotions, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return [(Sentiment.NEUTRAL, 5.0)]

    def calculate_intensity(self, similarity: float, polarity: float) -> float:
        """Calculate intensity based on semantic similarity and sentiment polarity"""
        # Base score from semantic similarity (0-6)
        base_score = similarity * 6
        
        # Add sentiment polarity contribution (0-4)
        polarity_score = abs(polarity) * 4
        
        # Combined score (0-10)
        total_score = base_score + polarity_score
        
        return min(max(total_score, 0.0), 10.0)

    def _map_emotions(self, emotions):
        # Placeholder for mapping emotions to our sentiment categories
        # This function needs to be implemented based on the output of the emotion classifier
        # For now, we'll use the first emotion in the list
        if emotions:
            return [(emotions[0]['label'], emotions[0]['score'])]
        else:
            return [(Sentiment.NEUTRAL, 5.0)]

    def _basic_sentiment_analysis(self, text: str) -> tuple[Sentiment, float]:
        """Improved fallback sentiment analysis"""
        text = text.lower()
        blob = TextBlob(text)
        
        # Enhanced emotion word weights
        emotion_words = {
            'happy': 2.0, 'joy': 2.0, 'excited': 2.0, 'love': 2.0, 'wonderful': 2.0,
            'sad': 2.0, 'angry': 2.0, 'scared': 2.0, 'worried': 2.0, 'upset': 2.0,
            'anxious': 2.0, 'nervous': 2.0, 'afraid': 2.0
        }
        
        # Calculate base score
        base_score = blob.sentiment.polarity * 5 + 5  # Convert -1,1 to 0,10
        
        # Add emotion word bonus
        emotion_multiplier = 1.0
        for word, weight in emotion_words.items():
            if word in text:
                emotion_multiplier += weight * 0.2
        
        final_score = min(base_score * emotion_multiplier, 10.0)
        
        # More nuanced sentiment mapping
        if 'scared' in text or 'worried' in text or 'anxious' in text:
            return Sentiment.ANXIOUS, final_score
        elif 'angry' in text or 'mad' in text or 'upset' in text:
            return Sentiment.ANGRY, final_score
        elif final_score > 7:
            return Sentiment.HAPPY, final_score
        elif final_score > 6:
            return Sentiment.EXCITED, final_score
        elif final_score < 4:
            return Sentiment.SAD, final_score
        
        return Sentiment.NEUTRAL, 5.0 