"""
ToonTales AI — Google Gemini Provider Adapter
Supports:
- Story Generation: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash
- Image Generation: gemini-2.5-flash-image, gemini-3.1-flash-image
Uses the official google-genai SDK.
"""

import io
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from services.providers.base import (
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
from utils.helpers import get_config, mask_secret, validate_image_file

logger = logging.getLogger("ToonTalesAI.Providers.Gemini")


class GeminiProvider(BaseStoryProvider, BaseImageProvider):
    """Google Gemini adapter for both narrative generation and multimodal image generation."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_config("GEMINI_API_KEY")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="gemini",
            display_name="Google Gemini",
            supports_text=True,
            supports_images=True,
            supports_image_edit=False,
            text_models=[
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash"
            ],
            image_models=[
                "gemini-2.5-flash-image",
                "gemini-3.1-flash-image"
            ],
            requires_api_key=True,
            api_key_env_var="GEMINI_API_KEY",
            default_text_model="gemini-2.5-flash",
            default_image_model="gemini-2.5-flash-image",
            pricing_info="Text models offer free-tier quota. Image models require Pay-As-You-Go Google Cloud billing (0 free requests)."
        )

    def _get_client(self, api_key: Optional[str] = None) -> genai.Client:
        if hasattr(self, "_custom_get_client") and callable(self._custom_get_client):
            return self._custom_get_client(api_key)
        key = api_key or self._api_key or get_config("GEMINI_API_KEY")
        if not key:
            raise AuthenticationError(
                "Gemini API Key is missing. Please provide a key in settings or configure GEMINI_API_KEY.",
                provider="gemini"
            )
        factory = getattr(self, "_client_factory", genai.Client)
        return factory(api_key=key)

    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """Validates key by attempting a lightweight API call."""
        try:
            client = self._get_client(api_key)
            models = client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Gemini API key validation failed: {e}")
            return False

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
        """Generates structured story JSON using Gemini."""
        from services.story_generator import StoryModel

        client = self._get_client(api_key)
        selected_model = model or get_config("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

        system_instruction = (
            "You are a master children's cartoon storyteller, animation director, and creative writer for ToonTales AI.\n"
            "Your objective is to craft an original, captivating, emotionally resonant, and age-appropriate story.\n"
            "RULES:\n"
            f"1. Genre: {genre}\n"
            f"2. Target Audience: {target_age}\n"
            f"3. Narrative Length: {story_length}\n"
            f"4. Selected Art Style: {art_style}\n"
            f"5. Number of Scenes: Exactly {num_scenes} sequential scenes (from scene_number 1 to {num_scenes}).\n"
            f"6. Ending Style: {ending_style}\n"
            "7. Character Consistency: For every character, define immutable visual traits (species, colors, body shape, "
            "costume/clothing, distinctive markings). These traits MUST remain identical across all scenes.\n"
            "8. Scene Progression: Scene 1 introduces the protagonist and setting. Middle scenes present escalating wonder, "
            "fun challenges, or discoveries. The final scene delivers a deeply satisfying and conclusive resolution.\n"
            "9. Visual Detail: Each scene's 'image_prompt' must clearly describe the characters' exact visual appearance, "
            "their physical poses, emotional facial expressions, specific actions, background environment, and dynamic lighting, "
            f"matching the '{art_style}' visual style.\n"
            "10. Safety: Keep all content completely child-safe, positive, engaging, and free of violence or scary elements.\n"
            "11. Return strictly structured JSON matching the requested schema."
        )

        user_prompt = (
            f"Story Idea: {story_idea.strip()}\n\n"
            f"Please write a complete {num_scenes}-scene cartoon story based on this idea. "
            f"Ensure exactly {num_scenes} scenes are generated, with an engaging title, character profiles, "
            f"and detailed visual prompts for each scene."
        )

        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=StoryModel,
                    temperature=0.75,
                )
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise InvalidRequestError("Received empty response from Gemini API.", provider="gemini")

            try:
                story_model = StoryModel.model_validate_json(raw_text)
            except Exception:
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text.strip())
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                story_model = StoryModel.model_validate_json(cleaned)

            story_dict = story_model.model_dump()

            usage = GenerationUsage(
                provider="gemini",
                model=selected_model,
                operation="story",
                cost_type="estimated"
            )
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage.prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
                usage.completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
                usage.total_tokens = getattr(response.usage_metadata, "total_token_count", None)

            return story_dict, usage

        except ClientError as e:
            self._handle_client_error(e, selected_model)
        except APIError as e:
            raise ProviderError(f"Gemini API Error: {e}", provider="gemini", raw_error=e)
        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network error connecting to Gemini API: {e}", provider="gemini", raw_error=e)
        except Exception as e:
            if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError)):
                raise
            raise ProviderError(f"Gemini story generation failed: {e}", provider="gemini", raw_error=e)

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        api_key: Optional[str] = None,
        reference_image_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, Optional[GenerationUsage]]:
        """Generates cartoon illustration via Gemini multimodal Developer API endpoint."""
        client = self._get_client(api_key)
        selected_model = model or get_config("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

        if selected_model.startswith("imagen-"):
            raise CapabilityError(
                f"Model '{selected_model}' requires Gemini Enterprise Agent Platform mode and is not supported "
                "with standard Google AI Studio API keys in the Gemini Developer API. "
                "Please configure a Gemini Developer API image model such as 'gemini-2.5-flash-image'.",
                provider="gemini"
            )

        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=[prompt],
            )

            image_bytes = None
            if hasattr(response, "parts") and response.parts:
                for part in response.parts:
                    if getattr(part, "inline_data", None) is not None:
                        try:
                            img = part.as_image()
                            if img and getattr(img, "image_bytes", None):
                                image_bytes = img.image_bytes
                                break
                        except Exception:
                            pass
                        if getattr(part.inline_data, "data", None):
                            image_bytes = part.inline_data.data
                            break

            if not image_bytes and hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts:
                        for part in parts:
                            if getattr(part, "inline_data", None) is not None:
                                try:
                                    img = part.as_image()
                                    if img and getattr(img, "image_bytes", None):
                                        image_bytes = img.image_bytes
                                        break
                                except Exception:
                                    pass
                                if getattr(part.inline_data, "data", None):
                                    image_bytes = part.inline_data.data
                                    break

            if not image_bytes:
                text_content = getattr(response, "text", "") or "No image returned"
                raise InvalidRequestError(
                    f"Gemini model '{selected_model}' did not return an inline image. Response: {text_content[:200]}",
                    provider="gemini"
                )

            usage = GenerationUsage(
                provider="gemini",
                model=selected_model,
                operation="image",
                images_generated=1,
                cost_type="estimated"
            )

            return image_bytes, usage

        except ClientError as e:
            self._handle_client_error(e, selected_model)
        except APIError as e:
            raise ProviderError(f"Gemini API Error: {e}", provider="gemini", raw_error=e)
        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network error connecting to Gemini API: {e}", provider="gemini", raw_error=e)
        except Exception as e:
            if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError)):
                raise
            raise ProviderError(f"Gemini image generation failed: {e}", provider="gemini", raw_error=e)

    def _handle_client_error(self, e: ClientError, model: str):
        """Normalizes Google GenAI ClientError to unified typed exceptions."""
        err_msg = str(e)
        code = getattr(e, "code", None)

        if code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            is_zero_quota = "free-tier" in err_msg.lower() or "0 requests" in err_msg.lower() or "quota: 0" in err_msg.lower()
            retry_match = re.search(r'retry after (\d+)', err_msg.lower())
            retry_sec = int(retry_match.group(1)) if retry_match else None

            if is_zero_quota:
                raise QuotaExceededError(
                    f"Google AI Studio project has 0 free-tier quota allocated for model '{model}'. "
                    "Google Cloud billing (Pay-As-You-Go) is required, or switch provider to Pollinations AI.",
                    provider="gemini",
                    retry_after=None,
                    is_zero_quota=True,
                    raw_error=e
                )
            raise QuotaExceededError(
                f"Gemini rate limit exceeded for model '{model}'. Retry after {retry_sec or 30}s.",
                provider="gemini",
                retry_after=retry_sec or 30,
                is_zero_quota=False,
                raw_error=e
            )

        if code in (401, 403) or "PERMISSION_DENIED" in err_msg or "API_KEY_INVALID" in err_msg:
            raise AuthenticationError(
                "Gemini API Key is invalid, inactive, or lacks permissions for this model.",
                provider="gemini",
                raw_error=e
            )

        if code == 404 or "NOT_FOUND" in err_msg:
            raise ModelNotFoundError(
                f"Gemini model '{model}' was not found or is not accessible.",
                provider="gemini",
                raw_error=e
            )

        raise ProviderError(f"Gemini request failed: {err_msg}", provider="gemini", raw_error=e)
