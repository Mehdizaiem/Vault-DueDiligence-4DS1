# Path: llm/groq_client.py
import os
import requests
import logging
import traceback
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env.local'))
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.error("GROQ_API_KEY not found. Please set it in your .env.local file or pass it as an argument.")
            raise ValueError("No API key provided for Groq.")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"GroqClient initialized for model: {self.model}")

    def generate(self, prompt: str, temperature: float = 0.5, max_tokens: int = 1500) -> str:
        messages = [{"role": "user", "content": prompt}]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False
        }

        try:
            logger.debug(f"Sending request to Groq API. Prompt length: {len(prompt)}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=90
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                token_usage = result.get("usage", {})
                logger.debug(f"Groq API success. Tokens used: {token_usage}")
                return answer
            else:
                error_message = f"Groq API error: {response.status_code} - {response.text}"
                logger.error(error_message)
                return f"Error: Failed to generate narrative. {error_message}"

        except requests.exceptions.Timeout:
            logger.error("Groq API request timed out.")
            return "Error: The request to the language model timed out."
        except Exception as e:
            logger.error(f"Error during Groq API call: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: An unexpected error occurred while generating narrative: {str(e)}"

    def is_available(self) -> bool:
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Groq API availability check failed: {e}")
            return False