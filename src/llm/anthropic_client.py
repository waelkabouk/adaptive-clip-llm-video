"""Anthropic API client."""

import logging
import os
from typing import List, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of 2024)
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client with vision support."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model: Model name
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
            
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
            
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text response."""
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self._client.messages.create(**kwargs)
        
        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
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
        # Build content with images
        content = []
        
        # Add images
        for img_b64 in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
            
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": content}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self._client.messages.create(**kwargs)
        
        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
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
        pricing = ANTHROPIC_PRICING.get(
            self._model, 
            ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]
        )
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
        
    @property
    def model_name(self) -> str:
        return self._model
        
    @property
    def provider_name(self) -> str:
        return "anthropic"

