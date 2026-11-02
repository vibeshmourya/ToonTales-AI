"""
ToonTales AI — Anthropic Claude Provider Adapter
Supports:
- Story Generation: claude-3-7-sonnet-20250219, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-haiku-20240307, claude-3-opus-20240229
Important: Claude is strictly a story-generation text model. It raises CapabilityError if image generation is requested.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import requests

try:
    from anthropic import Anthropic, APIError as AnthropicAPIError, AuthenticationError as AnthropicAuthError, RateLimitError as AnthropicRateLimitError, NotFoundError as AnthropicNotFoundError
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False

from services.providers.base import (
    BaseStoryProvider,
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

logger = logging.getLogger("ToonTalesAI.Providers.Anthropic")


class AnthropicProvider(BaseStoryProvider):
    """Anthropic Claude adapter for story creation and scene structuring."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_config("ANTHROPIC_API_KEY")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="anthropic",
            display_name="Anthropic Claude",
            supports_text=True,
            supports_images=False,
            supports_image_edit=False,
            text_models=[
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229"
            ],
            image_models=[],
            requires_api_key=True,
            api_key_env_var="ANTHROPIC_API_KEY",
            default_text_model="claude-3-5-sonnet-20241022",
            default_image_model="",
            pricing_info="Pay-per-token for Claude text models. Claude does not generate images; pair with Pollinations, OpenAI, or Gemini for illustrations."
        )

    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        key = api_key or self._api_key or get_config("ANTHROPIC_API_KEY")
        if not key:
            raise AuthenticationError(
                "Anthropic API Key is missing. Please provide a key in settings or configure ANTHROPIC_API_KEY.",
                provider="anthropic"
            )
        return key

    def _get_client(self, api_key: Optional[str] = None):
        key = self._get_api_key(api_key)
        return Anthropic(api_key=key, timeout=50.0)

    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """Validates key via a minimal ping request."""
        try:
            if HAS_ANTHROPIC_SDK:
                client = self._get_client(api_key)
                client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "hi"}]
                )
                return True
            else:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "hi"}]
                    },
                    timeout=10
                )
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Anthropic API key validation failed: {e}")
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
        """Generates structured story JSON using Claude Messages API."""
        from services.story_generator import StoryModel

        key = self._get_api_key(api_key)
        selected_model = model or get_config("ANTHROPIC_TEXT_MODEL", "claude-3-5-sonnet-20241022")

        system_instruction = (
            "You are a master children's cartoon storyteller, animation director, and creative writer for ToonTales AI.\n"
            "Your objective is to craft an original, captivating, emotionally resonant, and age-appropriate story.\n"
            "You must return ONLY raw, valid JSON matching this exact StoryModel schema:\n"
            "{\n"
            '  "title": "Creative Story Title",\n'
            '  "summary": "2-sentence synopsis",\n'
            f'  "genre": "{genre}",\n'
            f'  "art_style": "{art_style}",\n'
            '  "characters": [\n'
            '    {\n'
            '      "name": "Character Name",\n'
            '      "role": "Protagonist / Friend",\n'
            '      "appearance": "Persistent cartoon appearance: species, body shape, costume, exact colors",\n'
            '      "personality": "Key traits and quirks"\n'
            '    }\n'
            '  ],\n'
            '  "scenes": [\n'
            '    {\n'
            '      "scene_number": 1,\n'
            '      "heading": "Scene Title",\n'
            '      "narration": "2-4 child-friendly sentences",\n'
            '      "image_prompt": "Detailed visual prompt for cartoon illustration: characters, actions, expressions, camera framing, lighting",\n'
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
            "5. Return ONLY the JSON object. Do not include markdown code block formatting (```json) or introductory commentary."
        )

        user_prompt = (
            f"Story Idea: {story_idea.strip()}\n\n"
            f"Please write a complete {num_scenes}-scene cartoon story based on this idea. "
            f"Ensure exactly {num_scenes} scenes are created, with an engaging title, consistent character profiles, "
            f"and detailed visual prompts for each scene."
        )

        try:
            if HAS_ANTHROPIC_SDK:
                client = self._get_client(api_key)
                response = client.messages.create(
                    model=selected_model,
                    max_tokens=4000,
                    system=system_instruction,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.75,
                )
                raw_text = response.content[0].text
                usage_meta = response.usage
                usage = GenerationUsage(
                    provider="anthropic",
                    model=selected_model,
                    operation="story",
                    prompt_tokens=usage_meta.input_tokens if usage_meta else None,
                    completion_tokens=usage_meta.output_tokens if usage_meta else None,
                    total_tokens=(usage_meta.input_tokens + usage_meta.output_tokens) if usage_meta else None,
                    cost_type="estimated"
                )
            else:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": selected_model,
                        "max_tokens": 4000,
                        "system": system_instruction,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "temperature": 0.75
                    },
                    timeout=50
                )
                if resp.status_code != 200:
                    self._handle_http_error(resp, selected_model)
                data = resp.json()
                raw_text = data["content"][0]["text"]
                usage_meta = data.get("usage", {})
                in_tok = usage_meta.get("input_tokens")
                out_tok = usage_meta.get("output_tokens")
                usage = GenerationUsage(
                    provider="anthropic",
                    model=selected_model,
                    operation="story",
                    prompt_tokens=in_tok,
                    completion_tokens=out_tok,
                    total_tokens=(in_tok + out_tok) if (in_tok is not None and out_tok is not None) else None,
                    cost_type="estimated"
                )

            if not raw_text:
                raise InvalidRequestError("Received empty response from Anthropic Claude.", provider="anthropic")

            # Extract JSON if enclosed in markdown
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text.strip())
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

            try:
                story_model = StoryModel.model_validate_json(cleaned)
            except Exception as parse_err:
                # Try finding outermost JSON curly braces
                match = re.search(r'(\{[\s\S]*\})', cleaned)
                if match:
                    story_model = StoryModel.model_validate_json(match.group(1))
                else:
                    raise InvalidRequestError(f"Failed to parse Claude output into valid Story JSON: {parse_err}", provider="anthropic")

            story_dict = story_model.model_dump()
            return story_dict, usage

        except Exception as e:
            self._normalize_exception(e, selected_model)

    def generate_image(self, *args, **kwargs):
        """Claude is a language model and does not support image generation."""
        raise CapabilityError(
            "Anthropic Claude is a language model and does not support native image generation. "
            "Please select an image provider such as Pollinations AI, OpenAI, or Google Gemini in the sidebar settings.",
            provider="anthropic"
        )

    def _handle_http_error(self, resp: requests.Response, model: str):
        """Translates Anthropic HTTP responses to normalized exceptions."""
        code = resp.status_code
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", resp.text)
        except Exception:
            err_msg = resp.text

        if code == 401:
            raise AuthenticationError("Anthropic API Key is invalid or expired.", provider="anthropic")
        if code == 429:
            retry_after = resp.headers.get("retry-after")
            sec = int(retry_after) if retry_after and retry_after.isdigit() else 30
            if "credit" in err_msg.lower() or "quota" in err_msg.lower():
                raise QuotaExceededError(f"Anthropic account quota or credit limit reached: {err_msg}", provider="anthropic", retry_after=sec)
            raise RateLimitError(f"Anthropic rate limit reached for model '{model}': {err_msg}", provider="anthropic", retry_after=sec)
        if code == 404:
            raise ModelNotFoundError(f"Anthropic model '{model}' not found: {err_msg}", provider="anthropic")

        raise InvalidRequestError(f"Anthropic request failed (HTTP {code}): {err_msg}", provider="anthropic")

    def _normalize_exception(self, e: Exception, model: str):
        """Maps exceptions to normalized exceptions."""
        if isinstance(e, (ProviderError, RateLimitError, QuotaExceededError, AuthenticationError, CapabilityError)):
            raise e

        err_str = str(e)
        if "AuthenticationError" in type(e).__name__ or "invalid_api_key" in err_str or "401" in err_str:
            raise AuthenticationError("Anthropic API Key is invalid or unauthorized.", provider="anthropic", raw_error=e)
        if "RateLimitError" in type(e).__name__ or "rate_limit" in err_str or "429" in err_str:
            if "credit" in err_str.lower() or "quota" in err_str.lower():
                raise QuotaExceededError(f"Anthropic credit balance depleted: {err_str}", provider="anthropic", raw_error=e)
            raise RateLimitError(f"Anthropic rate limit exceeded: {err_str}", provider="anthropic", retry_after=30, raw_error=e)
        if "NotFoundError" in type(e).__name__ or "not_found" in err_str or "404" in err_str:
            raise ModelNotFoundError(f"Anthropic model '{model}' not found: {err_str}", provider="anthropic", raw_error=e)
        if isinstance(e, (ConnectionError, TimeoutError, requests.exceptions.RequestException)):
            raise NetworkError(f"Network error connecting to Anthropic API: {err_str}", provider="anthropic", raw_error=e)

        raise ProviderError(f"Anthropic story generation failed: {err_str}", provider="anthropic", raw_error=e)
