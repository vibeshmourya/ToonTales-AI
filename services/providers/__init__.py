"""
ToonTales AI — Provider Registry and Factory
Provides unified instantiation and discovery for all Story and Image AI providers.
"""

from typing import Dict, Any, List, Optional
from services.providers.base import (
    BaseProvider,
    BaseStoryProvider,
    BaseImageProvider,
    ProviderCapabilities,
    GenerationUsage,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    CapabilityError,
    NetworkError,
    InvalidRequestError,
    ContentPolicyError
)
from services.providers.gemini_provider import GeminiProvider
from services.providers.openai_provider import OpenAIProvider
from services.providers.anthropic_provider import AnthropicProvider
from services.providers.openrouter_provider import OpenRouterProvider
from services.providers.pollinations_provider import PollinationsProvider

# Registered providers
_PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
    "pollinations": PollinationsProvider
}


def list_story_providers() -> List[str]:
    """Returns provider keys that support text/story generation."""
    result = []
    for name, cls in _PROVIDERS.items():
        prov = cls()
        if prov.capabilities.supports_text:
            result.append(name)
    return result


def list_image_providers() -> List[str]:
    """Returns provider keys that support image generation."""
    result = []
    for name, cls in _PROVIDERS.items():
        prov = cls()
        if prov.capabilities.supports_images:
            result.append(name)
    return result


def get_provider_capabilities(name: str) -> ProviderCapabilities:
    """Returns capability metadata for a given provider key."""
    key = name.lower()
    if key not in _PROVIDERS:
        raise ProviderError(f"Unknown provider: '{name}'. Available: {list(_PROVIDERS.keys())}")
    return _PROVIDERS[key]().capabilities


def get_story_provider(name: str, api_key: Optional[str] = None) -> BaseStoryProvider:
    """Instantiates a story-generation provider adapter."""
    key = name.lower()
    if key not in _PROVIDERS:
        raise ProviderError(f"Unknown story provider: '{name}'. Available: {list_story_providers()}")
    prov = _PROVIDERS[key](api_key=api_key)
    if not prov.capabilities.supports_text:
        raise CapabilityError(
            f"Provider '{name}' does not support text/story generation.",
            provider=name
        )
    return prov


def get_image_provider(name: str, api_key: Optional[str] = None) -> BaseImageProvider:
    """Instantiates an image-generation provider adapter."""
    key = name.lower()
    if key not in _PROVIDERS:
        raise ProviderError(f"Unknown image provider: '{name}'. Available: {list_image_providers()}")
    prov = _PROVIDERS[key](api_key=api_key)
    if not prov.capabilities.supports_images:
        raise CapabilityError(
            f"Provider '{name}' does not support image generation.",
            provider=name
        )
    return prov
