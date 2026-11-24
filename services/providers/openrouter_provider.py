"""
ToonTales AI — OpenRouter Provider Adapter
Supports:
- Story Generation: anthropic/claude-3.5-sonnet, openai/gpt-4o-mini, google/gemini-2.5-flash, meta-llama/llama-3.3-70b-instruct
- Image Generation: Dedicated OpenRouter Unified Image API (/api/v1/images) with dynamic model discovery (/api/v1/images/models)
Validates model capabilities before execution to prevent invalid model requests.
"""

import base64
import io
import json
import logging
import re
import time
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
from utils.helpers import get_config, mask_secret, validate_image_bytes

logger = logging.getLogger("ToonTalesAI.Providers.OpenRouter")


class OpenRouterProvider(BaseStoryProvider, BaseImageProvider):
    """OpenRouter adapter with dynamic model discovery and capability validation."""

    BASE_URL = "https://openrouter.ai/api/v1"
    IMAGES_API_URL = "https://openrouter.ai/api/v1/images"
    IMAGES_MODELS_URL = "https://openrouter.ai/api/v1/images/models"

    FALLBACK_IMAGE_MODELS: List[Dict[str, Any]] = [
        {
            "id": "recraft/recraft-v4.1-flash-v2:free",
            "name": "Recraft: Recraft v4.1 Flash (Free)",
            "description": "High-quality vector and raster cartoon illustrations."
        },
        {
            "id": "recraft/recraft-v4.1-flash-v2",
            "name": "Recraft: Recraft v4.1 Flash",
            "description": "Fast, high-fidelity design and cartoon generation."
        },
        {
            "id": "microsoft/mai-image-2.7-flash",
            "name": "Microsoft: MAI-Image-2.7 Flash",
            "description": "Low-latency, high-quality image generation."
        },
        {
            "id": "black-forest-labs/flux.2-flex",
            "name": "Black Forest Labs: FLUX.2 Flex",
            "description": "Next-generation diffusion model for detailed compositions."
        },
        {
            "id": "black-forest-labs/flux-3-image-klein",
            "name": "Black Forest Labs: FLUX.3 Image Klein",
            "description": "High-speed, compact image generation."
        },
        {
            "id": "openai/gpt-image-2.6-flare",
            "name": "OpenAI: GPT-Image-2.6 Flare",
            "description": "Advanced image generation with prompt precision."
        },
        {
            "id": "bytedance-seed/seedream-5-5",
            "name": "ByteDance: Seedream 5.5",
            "description": "High-aesthetic illustration model."
        }
    ]

    _cached_image_models: Optional[List[Dict[str, Any]]] = None
    _cache_timestamp: float = 0.0
    CACHE_TTL_SECONDS: float = 1800.0  # 30 minutes

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
                "recraft/recraft-v4.1-flash-v2:free",
                "recraft/recraft-v4.1-flash-v2",
                "microsoft/mai-image-2.7-flash",
                "black-forest-labs/flux.2-flex",
                "openai/gpt-image-2.6-flare"
            ],
            requires_api_key=True,
            api_key_env_var="OPENROUTER_API_KEY",
            default_text_model="anthropic/claude-3.5-sonnet",
            default_image_model="recraft/recraft-v4.1-flash-v2:free",
            pricing_info="Aggregated routing with per-token and per-generation billing across top AI labs."
        )

    @classmethod
    def get_available_image_models(
        cls,
        api_key: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Dynamically discovers supported image generation models from OpenRouter.
        Queries GET https://openrouter.ai/api/v1/images/models, with fallback to GET https://openrouter.ai/api/v1/models
        filtering models with output_modalities containing 'image'.
        Results are cached for CACHE_TTL_SECONDS.
        """
        now = time.time()
        if not force_refresh and cls._cached_image_models and (now - cls._cache_timestamp < cls.CACHE_TTL_SECONDS):
            return cls._cached_image_models

        headers = {
            "HTTP-Referer": "https://toontales.ai",
            "X-Title": "ToonTales AI Cartoon Storybook"
        }
        key = api_key or get_config("OPENROUTER_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        models_list: List[Dict[str, Any]] = []

        # 1. Primary Endpoint: Dedicated OpenRouter Image Models Catalog
        try:
            resp = requests.get(cls.IMAGES_MODELS_URL, headers=headers, timeout=10)
            if resp.status_code == 200:
                raw_data = resp.json()
                items = raw_data.get("data", [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and item.get("id"):
                            models_list.append({
                                "id": str(item["id"]).strip(),
                                "name": str(item.get("name") or item["id"]).strip(),
                                "description": str(item.get("description") or "").strip()
                            })
        except Exception as e:
            logger.warning(f"OpenRouter dedicated image models discovery failed: {e}")

        # 2. Secondary Fallback: General Models Catalog filtered for output modality 'image'
        if not models_list:
            try:
                resp = requests.get(f"{cls.BASE_URL}/models", headers=headers, timeout=10)
                if resp.status_code == 200:
                    raw_data = resp.json()
                    items = raw_data.get("data", [])
                    if isinstance(items, list):
                        for item in items:
                            if not isinstance(item, dict) or not item.get("id"):
                                continue
                            arch = item.get("architecture") or {}
                            out_mods = arch.get("output_modalities") or []
                            mod = arch.get("modality", "")
                            if "image" in out_mods or mod == "image":
                                models_list.append({
                                    "id": str(item["id"]).strip(),
                                    "name": str(item.get("name") or item["id"]).strip(),
                                    "description": str(item.get("description") or "").strip()
                                })
            except Exception as e:
                logger.warning(f"OpenRouter general models fallback discovery failed: {e}")

        # 3. Tertiary Fallback: Curated Verified Models (never empty or obsolete)
        if not models_list:
            logger.info("Using curated fallback OpenRouter image model catalog.")
            models_list = [dict(m) for m in cls.FALLBACK_IMAGE_MODELS]

        cls._cached_image_models = models_list
        cls._cache_timestamp = now
        logger.info(f"OpenRouter image models catalog loaded: {len(models_list)} models available.")
        return models_list

    def validate_image_model(self, model: str, api_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates the selected model before making an image generation request:
        - Rejects empty models
        - Rejects known text-only models
        - Rejects obsolete/dead IDs like 'stabilityai/stable-diffusion-xl-base-1.0'
        - Validates against active OpenRouter image models catalog
        """
        if not model or not str(model).strip():
            return False, "OpenRouter image model ID cannot be empty."

        clean_model = str(model).strip()

        # Check known obsolete IDs
        if "stabilityai/stable-diffusion" in clean_model.lower():
            return False, (
                f"Model '{clean_model}' is obsolete and no longer available on OpenRouter. "
                "Please select an active model from the catalog such as 'recraft/recraft-v4.1-flash-v2:free' or 'microsoft/mai-image-2.7-flash'."
            )

        # Check known text-only models
        model_lower = clean_model.lower()
        text_only = ["claude", "gpt-4", "gpt-3.5", "llama-", "mistral", "qwen-2", "gemini-flash", "gemini-pro"]
        if any(t in model_lower for t in text_only) and not any(i in model_lower for i in ["image", "imagen", "diffusion"]):
            return False, (
                f"Selected model '{clean_model}' is a text/language model and does not support image generation."
            )

        # Validate against discovered catalog
        try:
            available = self.get_available_image_models(api_key=api_key)
            valid_ids = {m["id"].lower() for m in available}
            if clean_model.lower() in valid_ids:
                return True, None
            # If not in discovered catalog and discovery fetched verified models
            if len(available) >= 5:
                return False, (
                    f"Model '{clean_model}' is not a valid OpenRouter image generation model ID. "
                    "Please select an active model from the dropdown."
                )
        except Exception as e:
            logger.warning(f"Error checking model against catalog: {e}")

        return True, None

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

            from services.json_repair import parse_and_repair_json
            parsed_data = parse_and_repair_json(raw_text)
            story_model = StoryModel.model_validate(parsed_data)
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
        """
        Generates cartoon illustration via OpenRouter Unified Image Generation API.
        Endpoints:
        1. Dedicated OpenRouter Images API: POST https://openrouter.ai/api/v1/images
        2. Compatibility Endpoint: POST https://openrouter.ai/api/v1/images/generations
        3. Multimodal Chat Endpoint: POST https://openrouter.ai/api/v1/chat/completions
        Parses signed URL, inline base64 (b64_json), and multimodal image content.
        Validates decoded image bytes with validate_image_bytes before saving or returning.
        """
        key = self._get_api_key(api_key)
        selected_model = (model or get_config("OPENROUTER_IMAGE_MODEL", "recraft/recraft-v4.1-flash-v2:free")).strip()

        # 1. Pre-validation: Stop BEFORE making an expensive or doomed API request
        is_valid, validation_err = self.validate_image_model(selected_model, api_key=key)
        if not is_valid:
            if "text/language model" in (validation_err or ""):
                raise CapabilityError(validation_err, provider="openrouter")
            raise ModelNotFoundError(validation_err, provider="openrouter", model=selected_model)

        image_bytes = None
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "response_format": "url"
        }

        try:
            # 2. Primary Route: Dedicated OpenRouter Unified Image Endpoint (/api/v1/images)
            resp = requests.post(
                self.IMAGES_API_URL,
                headers=self._get_headers(key),
                json=payload,
                timeout=60
            )

            # If 404 on /images, attempt compatibility endpoint (/images/generations)
            if resp.status_code == 404:
                logger.info(f"OpenRouter /images returned 404 for '{selected_model}', falling back to /images/generations")
                resp = requests.post(
                    f"{self.BASE_URL}/images/generations",
                    headers=self._get_headers(key),
                    json={
                        "model": selected_model,
                        "prompt": prompt,
                        "size": f"{size[0]}x{size[1]}"
                    },
                    timeout=60
                )

            # If still 404, attempt multimodal chat completions endpoint
            if resp.status_code == 404:
                logger.info(f"OpenRouter /images/generations returned 404 for '{selected_model}', falling back to /chat/completions")
                resp = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(key),
                    json={
                        "model": selected_model,
                        "messages": [{"role": "user", "content": f"Generate a high-quality cartoon illustration: {prompt}"}],
                    },
                    timeout=60
                )

            if resp.status_code != 200:
                self._handle_http_error(resp, selected_model)

            data = resp.json()

            # 3. Comprehensive Response Parsing
            # Format A: data array with url or b64_json (Standard OpenRouter /images response)
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                for item in data["data"]:
                    if not isinstance(item, dict):
                        continue
                    if item.get("url"):
                        img_resp = requests.get(item["url"], timeout=30)
                        if img_resp.status_code == 200:
                            image_bytes = img_resp.content
                            break
                    elif item.get("b64_json"):
                        image_bytes = base64.b64decode(item["b64_json"])
                        break

            # Format B: chat completion choices with message.images or content
            if not image_bytes and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {})
                # Format B1: message.images
                for img_obj in msg.get("images", []):
                    if isinstance(img_obj, dict):
                        if img_obj.get("url"):
                            img_resp = requests.get(img_obj["url"], timeout=30)
                            if img_resp.status_code == 200:
                                image_bytes = img_resp.content
                                break
                        elif img_obj.get("b64_json"):
                            image_bytes = base64.b64decode(img_obj["b64_json"])
                            break

                # Format B2: content with base64 data URI or HTTP image URL
                if not image_bytes:
                    content = msg.get("content", "")
                    data_uri_match = re.search(r'data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=]+)', content)
                    if data_uri_match:
                        image_bytes = base64.b64decode(data_uri_match.group(1))
                    else:
                        url_match = re.search(r'https?://[^\s\)\"\']+\.(?:png|jpg|jpeg|webp)', content)
                        if url_match:
                            img_resp = requests.get(url_match.group(0), timeout=30)
                            if img_resp.status_code == 200:
                                image_bytes = img_resp.content

            if not image_bytes:
                raise InvalidRequestError(
                    f"OpenRouter model '{selected_model}' did not return valid image data.",
                    provider="openrouter"
                )

            # 4. Byte-level & Pillow integrity validation
            is_valid_bytes, val_msg = validate_image_bytes(image_bytes)
            if not is_valid_bytes:
                raise ProviderError(
                    f"OpenRouter image failed byte-level validation: {val_msg}",
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
        """Translates OpenRouter HTTP errors to normalized typed exceptions."""
        code = resp.status_code
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                err_msg = err_json.get("error", {}).get("message", resp.text)
            else:
                err_msg = resp.text
        except Exception:
            err_msg = resp.text

        err_lower = str(err_msg).lower()

        if code == 400:
            if "not a valid model id" in err_lower or "is not a valid model" in err_lower or "unknown model" in err_lower:
                raise ModelNotFoundError(
                    f"OpenRouter model '{model}' is not a valid model ID: {err_msg}",
                    provider="openrouter",
                    model=model
                )
            raise InvalidRequestError(f"OpenRouter bad request (HTTP 400): {err_msg}", provider="openrouter")

        if code == 401:
            raise AuthenticationError("OpenRouter API Key is invalid, inactive, or expired.", provider="openrouter")

        if code == 402:
            raise QuotaExceededError(
                f"OpenRouter credits depleted or payment required: {err_msg}",
                provider="openrouter"
            )

        if code == 404:
            raise ModelNotFoundError(f"OpenRouter model '{model}' not found: {err_msg}", provider="openrouter", model=model)

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

        raise InvalidRequestError(f"OpenRouter request failed (HTTP {code}): {err_msg}", provider="openrouter")

    def _normalize_exception(self, e: Exception, model: str):
        """Maps exceptions to normalized exceptions."""
        if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError, CapabilityError)):
            raise e
        err_str = str(e)
        if isinstance(e, (ConnectionError, TimeoutError, requests.exceptions.RequestException)):
            raise NetworkError(f"Network error connecting to OpenRouter API: {err_str}", provider="openrouter", raw_error=e)
        raise ProviderError(f"OpenRouter operation failed: {err_str}", provider="openrouter", raw_error=e)
