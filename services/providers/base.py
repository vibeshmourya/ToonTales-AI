"""
ToonTales AI — Base Provider Interfaces and Normalized Exceptions
Defines abstract contracts for Story and Image generation providers,
model capabilities, and normalized exception hierarchy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# =====================================================================
# Normalized Exception Hierarchy
# =====================================================================

class ProviderError(Exception):
    """Base exception for all provider-related errors."""
    def __init__(self, message: str, provider: str = "unknown", raw_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.raw_error = raw_error

    def __str__(self):
        return f"[{self.provider.upper()}] {self.message}"


class AuthenticationError(ProviderError):
    """Raised when an API key is missing, invalid, or unauthorized (HTTP 401, 403)."""
    pass


class RateLimitError(ProviderError):
    """
    Raised when a rate limit or quota is reached (HTTP 429, RESOURCE_EXHAUSTED).
    Tracks retry delays and whether quota exhaustion is permanent vs temporary.
    """
    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retry_after: Optional[int] = None,
        is_quota_exhausted: bool = False,
        is_zero_quota: bool = False,
        raw_error: Optional[Exception] = None
    ):
        super().__init__(message, provider=provider, raw_error=raw_error)
        self.retry_after = retry_after
        self.is_quota_exhausted = is_quota_exhausted
        self.is_zero_quota = is_zero_quota


class QuotaExceededError(RateLimitError):
    """Raised specifically when account billing quota or free-tier allowance is exhausted."""
    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retry_after: Optional[int] = None,
        is_zero_quota: bool = False,
        raw_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            provider=provider,
            retry_after=retry_after,
            is_quota_exhausted=True,
            is_zero_quota=is_zero_quota,
            raw_error=raw_error
        )


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not found, deprecated, or unavailable (HTTP 404)."""
    pass


class CapabilityError(ModelNotFoundError):
    """Raised when a provider or model does not support the requested capability (e.g. Claude for images)."""
    pass


class NetworkError(ProviderError):
    """Raised for connection errors, timeouts, or DNS failures."""
    pass


class InvalidRequestError(ProviderError):
    """Raised when request parameters, prompts, or configurations are invalid (HTTP 400)."""
    pass


class ContentPolicyError(ProviderError):
    """Raised when request is rejected by provider's safety or content policy filters."""
    pass


# =====================================================================
# Provider Capabilities & Usage Metadata
# =====================================================================

@dataclass
class ProviderCapabilities:
    name: str
    display_name: str
    supports_text: bool
    supports_images: bool
    supports_image_edit: bool = False
    text_models: List[str] = field(default_factory=list)
    image_models: List[str] = field(default_factory=list)
    requires_api_key: bool = True
    api_key_env_var: str = ""
    default_text_model: str = ""
    default_image_model: str = ""
    pricing_info: str = ""


@dataclass
class GenerationUsage:
    provider: str
    model: str
    operation: str  # "story" or "image"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    images_generated: int = 0
    estimated_cost_usd: Optional[float] = None
    cost_type: str = "unknown"  # "reported", "estimated", "zero_cost", or "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def input_tokens(self) -> Optional[int]:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> Optional[int]:
        return self.completion_tokens


# =====================================================================
# Abstract Base Provider Contracts
# =====================================================================

class BaseProvider(ABC):
    """Common base for all AI providers."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Returns provider capabilities, supported models, and features."""
        pass

    @abstractmethod
    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validates API key credentials via a minimal authentic request.
        Does not claim validity merely because the string is non-empty.
        """
        pass


class BaseStoryProvider(BaseProvider):
    """Contract for text/story generation providers."""

    @abstractmethod
    def generate_story(
        self,
        story_idea: str,
        genre: str = "Adventure",
        target_age: str = "6–8 years",
        story_length: str = "Medium",
        art_style: str = "Colorful 2D cartoon",
        num_scenes: int = 5,
        ending_style: str = "Happy",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[GenerationUsage]]:
        """
        Generates a structured story dictionary conforming to the StoryModel schema:
        {
            "title": str,
            "summary": str,
            "genre": str,
            "art_style": str,
            "characters": [{"name": ..., "role": ..., "appearance": ..., "personality": ...}],
            "scenes": [{"scene_number": ..., "heading": ..., "narration": ..., "image_prompt": ..., "characters_present": ..., "environment": ...}]
        }
        """
        pass


class BaseImageProvider(BaseProvider):
    """Contract for image-generation providers."""

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        api_key: Optional[str] = None,
        reference_image_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, Optional[GenerationUsage]]:
        """
        Generates an illustration for a scene prompt.
        Returns raw verified image bytes and usage metadata.
        """
        pass
