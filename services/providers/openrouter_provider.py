"""
ToonTales AI — OpenRouter Provider Adapter
Supports:
- Story Generation: anthropic/claude-3.5-sonnet, openai/gpt-4o-mini, google/gemini-2.5-flash, meta-llama/llama-3.3-70b-instruct
- Image Generation: stabilityai/stable-diffusion-xl-base-1.0, google/imagen-3, or image-capable models
Validates model capabilities before execution.
"""

import base64
import io
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import requests
from PIL import Image

from services.providers.base import (
    BaseStoryProvider,
    BaseImageProvider,
    ProviderCapabilities,
    GenerationUsage,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    CapabilityError,
    NetworkError,
    InvalidRequestError,
    ContentPolicyError,
    ProviderError
)
from utils.helpers import get_config, mask_secret

logger = logging.getLogger("ToonTalesAI.Providers.OpenRouter")


class OpenRouterProvider(BaseStoryProvider, BaseImageProvider):
    """OpenRouter adapter with dynamic model discovery and capability validation."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_config("OPENROUTER_API_KEY")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="openrouter",
            display_name="OpenRouter",
            supports_text=True,
            supports_images=True,
            supports_image_edit=False,
            text_models=[
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o-mini",
                "google/gemini-2.5-flash",
                "meta-llama/llama-3.3-70b-instruct",
                "mistralai/mistral-large-2407"
            ],
            image_models=[
                "stabilityai/stable-diffusion-xl-base-1.0",
                "google/imagen-3",
                "openai/dall-e-3"
            ],
            requires_api_key=True,
            api_key_env_var="OPENROUTER_API_KEY",
            default_text_model="anthropic/claude-3.5-sonnet",
            default_image_model="stabilityai/stable-diffusion-xl-base-1.0",
            pricing_info="Aggregated routing with per-token and per-generation billing across top AI labs."
        )

    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        key = api_key or self._api_key or get_config("OPENROUTER_API_KEY")
        if not key:
            raise AuthenticationError(
                "OpenRouter API Key is missing. Please provide a key in settings or configure OPENROUTER_API_KEY.",
                provider="openrouter"
            )
        return key

    def _get_headers(self, key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://toontales.ai",
            "X-Title": "ToonTales AI Cartoon Storybook",
            "Content-Type": "application/json"
        }

    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """Validates key via OpenRouter auth endpoint."""
        try:
            key = self._get_api_key(api_key)
            resp = requests.get(
                f"{self.BASE_URL}/auth/key",
                headers=self._get_headers(key),
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter API key validation failed: {e}")
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
        """Generates structured story JSON using OpenRouter chat completions."""
        from services.story_generator import StoryModel

        key = self._get_api_key(api_key)
        selected_model = model or get_config("OPENROUTER_TEXT_MODEL", "anthropic/claude-3.5-sonnet")

        system_instruction = (
            "You are a master children's cartoon storyteller, animation director, and creative writer for ToonTales AI.\n"
            "Your objective is to craft an original, captivating, emotionally resonant, and age-appropriate story.\n"
            "Output must be strictly valid JSON conforming to the StoryModel schema:\n"
            "{\n"
            '  "title": "Creative Story Title",\n'
            '  "summary": "2-sentence synopsis",\n'
            f'  "genre": "{genre}",\n'
            f'  "art_style": "{art_style}",\n'
            '  "characters": [\n'
            '    {\n'
            '      "name": "Character Name",\n'
            '      "role": "Protagonist / Friend",\n'
            '      "appearance": "Detailed persistent cartoon appearance: species, body shape, costume, exact colors",\n'
            '      "personality": "Key traits and quirks"\n'
            '    }\n'
            '  ],\n'
            '  "scenes": [\n'
            '    {\n'
            '      "scene_number": 1,\n'
            '      "heading": "Scene Title",\n'
            '      "narration": "2-4 child-friendly sentences",\n'
            '      "image_prompt": "Exhaustive visual prompt for cartoon illustration: characters, actions, expressions, camera framing, lighting",\n'
            '      "characters_present": ["Character Name"],\n'
            '      "environment": "Backdrop setting description"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            f"RULES:\n"
            f"1. Generate exactly {num_scenes} sequential scenes (scene_number 1 to {num_scenes}).\n"
            f"2. Ending style: {ending_style}.\n"
            f"3. Target age group: {target_age}.\n"
            "4. Content must be 100% child-safe, engaging, and imaginative.\n"
            "5. Return ONLY the JSON object, nothing else."
        )

        user_prompt = (
            f"Story Idea: {story_idea.strip()}\n\n"
            f"Please write a complete {num_scenes}-scene cartoon story based on this idea. "
            f"Ensure exactly {num_scenes} scenes are created, with an engaging title, consistent character profiles, "
            f"and detailed visual prompts for each scene."
        )

        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(key),
                json={
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.75
                },
                timeout=55
            )

            if resp.status_code != 200:
                self._handle_http_error(resp, selected_model)

            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            usage_meta = data.get("usage", {})

            usage = GenerationUsage(
                provider="openrouter",
                model=selected_model,
                operation="story",
                prompt_tokens=usage_meta.get("prompt_tokens"),
                completion_tokens=usage_meta.get("completion_tokens"),
                total_tokens=usage_meta.get("total_tokens"),
                cost_type="estimated"
            )

            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text.strip())
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

            try:
                story_model = StoryModel.model_validate_json(cleaned)
            except Exception:
                match = re.search(r'(\{[\s\S]*\})', cleaned)
                if match:
                    story_model = StoryModel.model_validate_json(match.group(1))
                else:
                    raise InvalidRequestError("Failed to parse OpenRouter response into valid Story JSON.", provider="openrouter")

            story_dict = story_model.model_dump()
            return story_dict, usage

        except Exception as e:
            self._normalize_exception(e, selected_model)

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        api_key: Optional[str] = None,
        reference_image_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, Optional[GenerationUsage]]:
        """Generates cartoon illustration via OpenRouter image generation endpoints."""
        key = self._get_api_key(api_key)
        selected_model = model or get_config("OPENROUTER_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

        # Capability check: verify model is an image-capable model
        text_only_indicators = ["claude", "gpt-4", "gemini-flash", "llama-", "qwen", "mistral"]
        model_lower = selected_model.lower()
        if any(ind in model_lower for ind in text_only_indicators) and not any(img_ind in model_lower for img_ind in ["image", "imagen", "diffusion", "dall-e"]):
            raise CapabilityError(
                f"Selected OpenRouter model '{selected_model}' is a text/language model and does not support image generation. "
                "Please choose an image model such as 'stabilityai/stable-diffusion-xl-base-1.0', or switch to Pollinations AI.",
                provider="openrouter"
            )

        try:
            # Try images/generations endpoint first
            resp = requests.post(
                f"{self.BASE_URL}/images/generations",
                headers=self._get_headers(key),
                json={
                    "model": selected_model,
                    "prompt": prompt[:1000],
                    "size": f"{size[0]}x{size[1]}"
                },
                timeout=60
            )

            # If 404 or endpoint not supported, attempt multimodal chat image generation
            if resp.status_code == 404:
                resp = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(key),
                    json={
                        "model": selected_model,
                        "messages": [{"role": "user", "content": f"Generate an image: {prompt}"}],
                    },
                    timeout=60
                )

            if resp.status_code != 200:
                self._handle_http_error(resp, selected_model)

            data = resp.json()
            image_bytes = None

            # 1. Parse standard images/generations response
            if "data" in data and len(data["data"]) > 0:
                item = data["data"][0]
                if "b64_json" in item:
                    image_bytes = base64.b64decode(item["b64_json"])
                elif "url" in item:
                    img_resp = requests.get(item["url"], timeout=30)
                    image_bytes = img_resp.content

            # 2. Parse chat markdown / url response
            if not image_bytes and "choices" in data:
                content = data["choices"][0]["message"].get("content", "")
                url_match = re.search(r'https?://[^\s\)]+\.(?:png|jpg|jpeg|webp)', content)
                if url_match:
                    img_resp = requests.get(url_match.group(0), timeout=30)
                    image_bytes = img_resp.content

            if not image_bytes:
                raise InvalidRequestError(
                    f"OpenRouter model '{selected_model}' did not return valid image data.",
                    provider="openrouter"
                )

            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                pil_img.verify()

            usage = GenerationUsage(
                provider="openrouter",
                model=selected_model,
                operation="image",
                images_generated=1,
                cost_type="estimated"
            )

            return image_bytes, usage

        except Exception as e:
            self._normalize_exception(e, selected_model)

    def _handle_http_error(self, resp: requests.Response, model: str):
        """Translates OpenRouter HTTP errors to normalized exceptions."""
        code = resp.status_code
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                err_msg = err_json.get("error", {}).get("message", resp.text)
            else:
                err_msg = resp.text
        except Exception:
            err_msg = resp.text

        if code == 401:
            raise AuthenticationError("OpenRouter API Key is invalid or expired.", provider="openrouter")
        if code == 402:
            raise QuotaExceededError(
                f"OpenRouter credits depleted or payment required: {err_msg}",
                provider="openrouter"
            )
        if code == 429:
            retry_after = (
                resp.headers.get("retry-after") or
                resp.headers.get("Retry-After") or
                resp.headers.get("RETRY-AFTER")
            ) if hasattr(resp, "headers") and resp.headers else None
            sec = int(retry_after) if retry_after and str(retry_after).strip().isdigit() else 30
            raise RateLimitError(
                f"OpenRouter rate limit exceeded for '{model}': {err_msg}",
                provider="openrouter",
                retry_after=sec
            )
        if code == 404:
            raise ModelNotFoundError(f"OpenRouter model '{model}' not found: {err_msg}", provider="openrouter")

        raise InvalidRequestError(f"OpenRouter request failed (HTTP {code}): {err_msg}", provider="openrouter")

    def _normalize_exception(self, e: Exception, model: str):
        """Maps exceptions to normalized exceptions."""
        if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError, CapabilityError)):
            raise e
        err_str = str(e)
        if isinstance(e, (ConnectionError, TimeoutError, requests.exceptions.RequestException)):
            raise NetworkError(f"Network error connecting to OpenRouter API: {err_str}", provider="openrouter", raw_error=e)
        raise ProviderError(f"OpenRouter operation failed: {err_str}", provider="openrouter", raw_error=e)
