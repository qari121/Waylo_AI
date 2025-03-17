import requests
from typing import Optional, Dict

class WailoAPI:
    def __init__(self, base_url: str = "https://app.waylo.ai"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "accept": "*/*",
            "Authorization": "Bearer none"  # Replace with actual token if needed
        }
        
    def _log_response(self, endpoint: str, response) -> None:
        """Debug log API responses"""
        print(f"\n=== API Response [{endpoint}] ===")
        print(f"Status: {response.status_code}")
        try:
            data = response.json()
            print(f"Data: {data}")
        except:
            print(f"Raw response: {response.text}")
        print("=" * 40)

    def get_toy_info(self, mac_address: str) -> Optional[Dict]:
        """Get toy information by MAC address"""
        if not mac_address:
            raise ValueError("MAC address is required")
        
        url = f"{self.base_url}/toys/{mac_address}"
        response = requests.get(url, headers=self.headers)
        self._log_response("get_toy_info", response)
        if response.status_code == 200:
            return response.json()
        return None

    def log_request(self, message: str, mac_address: str) -> Optional[str]:
        """Log user's request and return request ID"""
        url = f"{self.base_url}/logs/addRequestLog"
        payload = {
            "message": message,
            "toy_mac_address": mac_address
        }
        response = requests.post(url, headers=self.headers, json=payload)
        self._log_response("log_request", response)
        if response.status_code == 200:
            data = response.json()
            return data.get("id")  # Get ID from response
        return None

    def log_response(self, message: str, mac_address: str) -> Optional[str]:
        """Log Wailo's response and return response ID"""
        url = f"{self.base_url}/logs/addResponseLog"
        payload = {
            "message": message,
            "toy_mac_address": mac_address
        }
        response = requests.post(url, headers=self.headers, json=payload)
        self._log_response("log_response", response)
        if response.status_code == 200:
            data = response.json()
            return data.get("id")  # Get ID from response
        return None

    def log_interest(self, interest: str, mac_address: str, 
                    request_id: str, response_id: str, intensity: float) -> bool:
        """Log user's interest in a topic"""
        url = f"{self.base_url}/interests/addInterestLog"
        payload = {
            "interest": interest,
            "toy_mac_address": mac_address,
            "request_id": request_id,
            "response_id": response_id,
            "intensity": intensity
        }
        response = requests.post(url, headers=self.headers, json=payload)
        self._log_response("log_interest", response)
        return response.status_code == 200 

    def log_sentiment(self, sentiment: str, mac_address: str, 
                     request_id: str, response_id: str, intensity: float) -> bool:
        """Log sentiment analysis"""
        url = f"{self.base_url}/sentiments/addSentimentLog"
        payload = {
            "sentiment": sentiment,
            "toy_mac_address": mac_address,
            "request_id": request_id,
            "response_id": response_id,
            "intensity": intensity
        }
        response = requests.post(url, headers=self.headers, json=payload)
        self._log_response("log_sentiment", response)
        return response.status_code == 200 