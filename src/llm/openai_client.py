"""OpenAI API client."""

import logging
import os
from typing import List, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-vision-preview": {"input": 10.00, "output": 30.00},
}


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with vision support."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name
            api_key: API key (defaults to OPENAI_API_KEY env var)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
            
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
            
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
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
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Build content with images
        content = []
        
        # Add images first
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low",  # Use low detail for efficiency
                },
            })
            
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        messages.append({"role": "user", "content": content})
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
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
        pricing = OPENAI_PRICING.get(self._model, OPENAI_PRICING["gpt-4o"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
        
    @property
    def model_name(self) -> str:
        return self._model
        
    @property
    def provider_name(self) -> str:
        return "openai"

