"""LLM client factory."""

import logging
from typing import Optional

from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient
from .local_client import LocalClient

logger = logging.getLogger(__name__)


def create_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: Provider name (openai, anthropic, google)
        model: Model name (uses provider default if not specified)
        api_key: API key (uses env var if not specified)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        
    Returns:
        BaseLLMClient instance
    """
    # Default models per provider
    default_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-2.0-flash",
        "local": "local-llm",
    }
    
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "local": LocalClient,
    }
    
    provider = provider.lower()
    
    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(providers.keys())}"
        )
        
    model = model or default_models[provider]
    
    logger.info(f"Creating LLM client: {provider}/{model}")
    
    return providers[provider](
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

