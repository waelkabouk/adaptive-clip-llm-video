"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict


class LLMResponse(TypedDict):
    """Standard LLM response format."""
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str
    provider: str


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate text response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse with text and metadata
        """
        pass
    
    @abstractmethod
    def generate_with_images(
        self,
        prompt: str,
        images: List[str],  # Base64 encoded
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response with image inputs.
        
        Args:
            prompt: User prompt
            images: List of base64-encoded images
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse with text and metadata
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name."""
        pass
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost in USD.
        
        Override in subclasses with actual pricing.
        """
        return 0.0

