"""
ToonTales AI — OpenAI Provider Adapter
Supports:
- Story Generation: gpt-4o, gpt-4o-mini, gpt-4-turbo
- Image Generation: dall-e-3, dall-e-2
Uses official openai SDK with HTTP fallback.
"""

import base64
import io
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import requests
from PIL import Image

try:
    from openai import OpenAI, APIError, AuthenticationError as OpenAIAuthError, RateLimitError as OpenAIRateLimitError, BadRequestError, NotFoundError
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

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
from utils.helpers import get_config, mask_secret, validate_image_file

logger = logging.getLogger("ToonTalesAI.Providers.OpenAI")


class OpenAIProvider(BaseStoryProvider, BaseImageProvider):
    """OpenAI adapter for narrative text and DALL-E image generation."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_config("OPENAI_API_KEY")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="openai",
            display_name="OpenAI",
            supports_text=True,
            supports_images=True,
            supports_image_edit=True,
            text_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo"
            ],
            image_models=[
                "dall-e-3",
                "dall-e-2"
            ],
            requires_api_key=True,
            api_key_env_var="OPENAI_API_KEY",
            default_text_model="gpt-4o-mini",
            default_image_model="dall-e-3",
            pricing_info="Pay-per-token for GPT text models and pay-per-image for DALL-E models."
        )

    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        key = api_key or self._api_key or get_config("OPENAI_API_KEY")
        if not key:
            raise AuthenticationError(
                "OpenAI API Key is missing. Please provide a key in settings or configure OPENAI_API_KEY.",
                provider="openai"
            )
        return key

    def _get_client(self, api_key: Optional[str] = None):
        key = self._get_api_key(api_key)
        return OpenAI(api_key=key, timeout=45.0)

    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """Validates key by querying OpenAI models endpoint."""
        try:
            if HAS_OPENAI_SDK:
                client = self._get_client(api_key)
                client.models.list()
                return True
            else:
                key = self._get_api_key(api_key)
                resp = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10
                )
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
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
        """Generates structured story JSON using OpenAI Chat Completions."""
        from services.story_generator import StoryModel

        key = self._get_api_key(api_key)
        selected_model = model or get_config("OPENAI_TEXT_MODEL", "gpt-4o-mini")

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
            if HAS_OPENAI_SDK:
                client = self._get_client(api_key)
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.75,
                )
                raw_text = response.choices[0].message.content
                usage_meta = response.usage
                usage = GenerationUsage(
                    provider="openai",
                    model=selected_model,
                    operation="story",
                    prompt_tokens=usage_meta.prompt_tokens if usage_meta else None,
                    completion_tokens=usage_meta.completion_tokens if usage_meta else None,
                    total_tokens=usage_meta.total_tokens if usage_meta else None,
                    cost_type="estimated"
                )
            else:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.75
                    },
                    timeout=45
                )
                if resp.status_code != 200:
                    self._handle_http_error(resp, selected_model)
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage_meta = data.get("usage", {})
                usage = GenerationUsage(
                    provider="openai",
                    model=selected_model,
                    operation="story",
                    prompt_tokens=usage_meta.get("prompt_tokens"),
                    completion_tokens=usage_meta.get("completion_tokens"),
                    total_tokens=usage_meta.get("total_tokens"),
                    cost_type="estimated"
                )

            if not raw_text:
                raise InvalidRequestError("Received empty response from OpenAI.", provider="openai")

            try:
                story_model = StoryModel.model_validate_json(raw_text)
            except Exception:
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text.strip())
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                story_model = StoryModel.model_validate_json(cleaned)

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
        """Generates cartoon illustration via OpenAI DALL-E endpoint."""
        key = self._get_api_key(api_key)
        selected_model = model or get_config("OPENAI_IMAGE_MODEL", "dall-e-3")

        # DALL-E 3 supports 1024x1024, 1024x1792, 1792x1024; DALL-E 2 supports 512x512, 1024x1024
        dim_str = "1024x1024" if selected_model == "dall-e-3" else "512x512"

        try:
            if HAS_OPENAI_SDK:
                client = self._get_client(api_key)
                kwargs = {
                    "model": selected_model,
                    "prompt": prompt[:1000],
                    "size": dim_str,
                    "n": 1,
                }
                # DALL-E 3 supports b64_json
                if selected_model in ("dall-e-3", "dall-e-2"):
                    kwargs["response_format"] = "b64_json"

                result = client.images.generate(**kwargs)
                item = result.data[0]
                if getattr(item, "b64_json", None):
                    image_bytes = base64.b64decode(item.b64_json)
                elif getattr(item, "url", None):
                    img_resp = requests.get(item.url, timeout=30)
                    image_bytes = img_resp.content
                else:
                    raise InvalidRequestError("OpenAI did not return image data or URL.", provider="openai")
            else:
                payload = {
                    "model": selected_model,
                    "prompt": prompt[:1000],
                    "size": dim_str,
                    "n": 1,
                    "response_format": "b64_json"
                }
                resp = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                if resp.status_code != 200:
                    self._handle_http_error(resp, selected_model)
                data = resp.json()
                item = data["data"][0]
                if "b64_json" in item:
                    image_bytes = base64.b64decode(item["b64_json"])
                elif "url" in item:
                    img_resp = requests.get(item["url"], timeout=30)
                    image_bytes = img_resp.content
                else:
                    raise InvalidRequestError("OpenAI did not return image data.", provider="openai")

            # Validate the bytes with Pillow
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                pil_img.verify()

            usage = GenerationUsage(
                provider="openai",
                model=selected_model,
                operation="image",
                images_generated=1,
                cost_type="estimated"
            )

            return image_bytes, usage

        except Exception as e:
            self._normalize_exception(e, selected_model)

    def _handle_http_error(self, resp: requests.Response, model: str):
        """Translates HTTP response to normalized exception."""
        code = resp.status_code
        try:
            err_body = resp.json().get("error", {})
            err_msg = err_body.get("message", resp.text)
        except Exception:
            err_msg = resp.text

        if code == 401:
            raise AuthenticationError("OpenAI API Key is invalid or expired.", provider="openai")
        if code == 429:
            retry_after = resp.headers.get("Retry-After")
            sec = int(retry_after) if retry_after and retry_after.isdigit() else 30
            is_quota = "quota" in err_msg.lower() or "billing" in err_msg.lower() or "insufficient_quota" in err_msg.lower()
            if is_quota:
                raise QuotaExceededError(
                    f"OpenAI account quota exceeded or insufficient billing credits: {err_msg}",
                    provider="openai",
                    retry_after=sec
                )
            raise RateLimitError(
                f"OpenAI rate limit exceeded for model '{model}': {err_msg}",
                provider="openai",
                retry_after=sec
            )
        if code == 404:
            raise ModelNotFoundError(f"OpenAI model '{model}' not found: {err_msg}", provider="openai")
        if code == 400 and ("safety" in err_msg.lower() or "policy" in err_msg.lower()):
            raise ContentPolicyError(f"OpenAI content policy rejection: {err_msg}", provider="openai")

        raise InvalidRequestError(f"OpenAI request failed (HTTP {code}): {err_msg}", provider="openai")

    def _normalize_exception(self, e: Exception, model: str):
        """Maps SDK exceptions to normalized exceptions."""
        if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError, ContentPolicyError)):
            raise e

        err_str = str(e)
        if "AuthenticationError" in type(e).__name__ or "invalid_api_key" in err_str or "incorrect api key" in err_str.lower():
            raise AuthenticationError("OpenAI API Key is invalid or unauthorized.", provider="openai", raw_error=e)
        if "RateLimitError" in type(e).__name__ or "insufficient_quota" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
            is_quota = "quota" in err_str.lower() or "billing" in err_str.lower()
            if is_quota:
                raise QuotaExceededError(f"OpenAI quota exhausted or insufficient credits: {err_str}", provider="openai", raw_error=e)
            raise RateLimitError(f"OpenAI rate limit reached: {err_str}", provider="openai", retry_after=30, raw_error=e)
        if "NotFoundError" in type(e).__name__ or "model_not_found" in err_str:
            raise ModelNotFoundError(f"OpenAI model '{model}' was not found: {err_str}", provider="openai", raw_error=e)
        if "content_policy_violation" in err_str or "safety" in err_str.lower():
            raise ContentPolicyError(f"OpenAI safety system rejected prompt: {err_str}", provider="openai", raw_error=e)
        if isinstance(e, (ConnectionError, TimeoutError, requests.exceptions.RequestException)):
            raise NetworkError(f"Network error connecting to OpenAI API: {err_str}", provider="openai", raw_error=e)

        raise ProviderError(f"OpenAI generation failed: {err_str}", provider="openai", raw_error=e)
