"""
ToonTales AI — Pollinations AI Provider Adapter
Specialized cartoon image generation provider:
- Supports authenticated gen.pollinations.ai endpoint with secret key
- Supports open zero-cost tier on image.pollinations.ai
- Models: flux, turbo, flux-realism, flux-anime, flux-3d
"""

import io
import logging
import re
import time
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
import requests
from PIL import Image

from services.providers.base import (
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
    ProviderError
)
from utils.helpers import get_config, mask_secret

logger = logging.getLogger("ToonTalesAI.Providers.Pollinations")


class PollinationsProvider(BaseImageProvider):
    """Pollinations AI image generation adapter with rate limit management and dual endpoints."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_config("POLLINATIONS_API_KEY")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="pollinations",
            display_name="Pollinations AI",
            supports_text=False,
            supports_images=True,
            supports_image_edit=False,
            text_models=[],
            image_models=[
                "flux",
                "turbo",
                "flux-realism",
                "flux-anime",
                "flux-3d"
            ],
            requires_api_key=False,
            api_key_env_var="POLLINATIONS_API_KEY",
            default_text_model="",
            default_image_model="flux",
            pricing_info="Zero-cost open tier available. Optional secret key from enter.pollinations.ai/keys enables higher rate limits on gen.pollinations.ai."
        )

    def validate_key(self, api_key: Optional[str] = None) -> bool:
        """Validates key if provided, or returns True since public tier is open."""
        key = api_key or self._api_key or get_config("POLLINATIONS_API_KEY")
        if not key:
            return True  # Public tier does not require key
        try:
            resp = requests.get(
                "https://gen.pollinations.ai/image/test",
                headers={"Authorization": f"Bearer {key}", "User-Agent": "ToonTalesAI/1.0"},
                timeout=10
            )
            return resp.status_code != 401
        except Exception:
            return True  # Public fallback remains accessible

    def generate_story(self, *args, **kwargs):
        raise CapabilityError(
            "Pollinations AI is an image generation provider and does not support text/story generation. "
            "Please select a story provider such as Google Gemini, OpenAI, Claude, or OpenRouter.",
            provider="pollinations"
        )

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        api_key: Optional[str] = None,
        reference_image_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, Optional[GenerationUsage]]:
        """Generates cartoon illustration via Pollinations AI dual endpoint strategy."""
        key = api_key or self._api_key or get_config("POLLINATIONS_API_KEY")
        selected_model = model or get_config("POLLINATIONS_IMAGE_MODEL", "flux")

        encoded_prompt = urllib.parse.quote(prompt[:350])
        headers = {"User-Agent": "ToonTalesAI/1.0"}

        # Candidate URLs in priority order:
        # 1. gen.pollinations.ai with Bearer auth (if key present)
        # 2. image.pollinations.ai open tier
        urls_to_try = []
        if key:
            headers["Authorization"] = f"Bearer {key}"
            model_query = f"?model={urllib.parse.quote(selected_model)}&width={size[0]}&height={size[1]}&nologo=true"
            urls_to_try.append(("gen.pollinations.ai", f"https://gen.pollinations.ai/image/{encoded_prompt}{model_query}"))

        open_query = f"?width={size[0]}&height={size[1]}&nologo=true"
        if selected_model != "flux":
            open_query += f"&model={urllib.parse.quote(selected_model)}"
        urls_to_try.append(("image.pollinations.ai", f"https://image.pollinations.ai/prompt/{encoded_prompt}{open_query}"))

        last_error = None
        for endpoint_name, target_url in urls_to_try:
            for attempt in range(2):
                try:
                    logger.debug(f"Calling Pollinations ({endpoint_name}), attempt {attempt + 1}...")
                    response = requests.get(target_url, headers=headers, timeout=(10, 45))

                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "")
                        # Validate that it is binary image data
                        if (
                            "image" in content_type or
                            response.content.startswith(b"\x89PNG") or
                            response.content.startswith(b"\xff\xd8") or
                            len(response.content) >= 50
                        ):
                            # Pillow validation
                            with Image.open(io.BytesIO(response.content)) as pil_img:
                                pil_img.verify()

                            usage = GenerationUsage(
                                provider="pollinations",
                                model=selected_model,
                                operation="image",
                                images_generated=1,
                                cost_type="zero_cost" if not key else "reported"
                            )
                            return response.content, usage
                        elif "application/json" in content_type:
                            try:
                                err_data = response.json()
                                last_error = f"Pollinations returned error: {err_data}"
                            except Exception:
                                last_error = "Pollinations returned non-image JSON response."

                    elif response.status_code == 401:
                        if key:
                            raise AuthenticationError(
                                "Pollinations API Key is invalid or expired. Check your key at enter.pollinations.ai/keys.",
                                provider="pollinations"
                            )
                        # Without key, continue to public endpoint
                        last_error = "Pollinations authentication required on this endpoint."
                        break

                    elif response.status_code in (402, 429) or "free limit" in response.text.lower():
                        retry_match = re.search(r'retry after (\d+)', response.text.lower())
                        retry_sec = int(retry_match.group(1)) if retry_match else 45
                        raise QuotaExceededError(
                            f"Pollinations rate limit reached. Please wait {retry_sec} seconds before regenerating.",
                            provider="pollinations",
                            retry_after=retry_sec
                        )

                    elif response.status_code == 503:
                        time.sleep(2 * (attempt + 1))
                        last_error = "Pollinations service temporarily busy (HTTP 503)."
                        continue

                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:100]}"

                except (QuotaExceededError, AuthenticationError):
                    raise
                except requests.exceptions.Timeout as t_err:
                    last_error = f"Timeout connecting to Pollinations AI ({t_err})"
                except requests.exceptions.RequestException as req_err:
                    last_error = f"Network connection error: {req_err}"

                time.sleep(1.5 * (attempt + 1))

        if last_error and ("timeout" in str(last_error).lower() or "connection" in str(last_error).lower()):
            raise NetworkError(f"Pollinations AI connection failed: {last_error}", provider="pollinations")

        raise ProviderError(f"Pollinations AI image generation failed: {last_error}", provider="pollinations")
