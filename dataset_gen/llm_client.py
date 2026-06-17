import os
import time
from typing import Optional
from openai import OpenAI


class LLMClient:
    """Standalone OpenAI client for persona generation pipeline"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

    def call_gpt41(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        retry_attempts: int = 3
    ) -> str:
        """
        Call GPT-4.1 for fact generation (high quality)

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            retry_attempts: Number of retry attempts on failure

        Returns:
            Generated text response
        """
        return self._call_with_retry(
            model="gpt-4.1",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_attempts=retry_attempts
        )

    def _call_with_retry(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        retry_attempts: int
    ) -> str:
        """Internal method to call OpenAI API with retry logic"""
        last_exception = None

        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}/{retry_attempts}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {retry_attempts} attempts.")

        raise Exception(f"Failed to call {model} after {retry_attempts} attempts: {last_exception}")
