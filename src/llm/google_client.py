"""Google Gemini API client."""

import logging
import os
from typing import List, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of 2024)
GOOGLE_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}


class GoogleClient(BaseLLMClient):
    """Google Gemini API client with vision support."""
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize Google Gemini client.
        
        Args:
            model: Model name
            api_key: API key (defaults to GOOGLE_API_KEY env var)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided")
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
            
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text response."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        generation_config = self._genai.GenerationConfig(
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        
        response = self._client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        text = response.text if response.text else ""
        
        # Gemini doesn't always provide token counts
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self.estimate_cost(input_tokens, output_tokens),
            model=self._model,
            provider=self.provider_name,
        )
        
    def generate_with_images(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response with image inputs."""
        import base64
        from PIL import Image
        import io
        
        # Build content with images
        content = []
        
        # Add system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        # Convert base64 images to PIL
        for img_b64 in images:
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data))
            content.append(img)
            
        # Add text prompt
        content.append(full_prompt)
        
        generation_config = self._genai.GenerationConfig(
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        
        response = self._client.generate_content(
            content,
            generation_config=generation_config,
        )
        
        text = response.text if response.text else ""
        
        # Token counts
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self.estimate_cost(input_tokens, output_tokens),
            model=self._model,
            provider=self.provider_name,
        )
        
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = GOOGLE_PRICING.get(self._model, GOOGLE_PRICING["gemini-1.5-pro"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
        
    @property
    def model_name(self) -> str:
        return self._model
        
    @property
    def provider_name(self) -> str:
        return "google"

