"""
ToonTales AI - Cartoon Image Generation Engine
Supports:
1. Pollinations AI (Recommended default: accessible, high-quality cartoon diffusion, zero-cost open tier or authenticated via gen.pollinations.ai)
2. Google Gemini Developer API via official google-genai SDK ('gemini-2.5-flash-image' via generate_content for paid billing projects)

Features:
- Scene illustration state tracking: 'Pending', 'Generating', 'Generated', 'Failed'
- Specific typed exceptions for Quota, Auth, Model, and Network errors
- Early circuit-breaker on zero-quota detection to prevent hammering exhausted endpoints
- Enforces character visual consistency across scenes
- Applies chosen cartoon art style and vibrant color palettes
- Validates image files using Pillow before saving
- Atomic saving to prevent corrupting existing images during scene regeneration
"""

import io
import os
import re
import time
import logging
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import requests
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from utils.helpers import get_config, mask_secret, validate_image_file

logger = logging.getLogger("ToonTalesAI.ImageGenerator")


# =====================================================================
# Specific, Typed Exceptions for Granular Error Categorization
# =====================================================================

class ImageGenerationError(Exception):
    """Base exception for all image generation failures."""
    pass


class QuotaExceededError(ImageGenerationError):
    """
    Raised when API quota or rate limits (HTTP 429, RESOURCE_EXHAUSTED) are exceeded.
    Distinguishes zero free-tier quota (requires billing/upgrade) from temporary minute limits.
    """
    def __init__(self, message: str, retry_after: Optional[int] = None, is_zero_quota: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_zero_quota = is_zero_quota


class InvalidAPIKeyError(ImageGenerationError):
    """Raised when an API key is missing, invalid, or unauthorized (HTTP 401, 403)."""
    pass


class UnsupportedModelError(ImageGenerationError):
    """Raised when the requested model is not found, deprecated, or unsupported (HTTP 404)."""
    pass


class ImageNetworkError(ImageGenerationError):
    """Raised for network connectivity errors, connection timeouts, or socket failures."""
    pass


# =====================================================================
# Image Generator Service
# =====================================================================

class ImageGeneratorService:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        pollinations_api_key: Optional[str] = None,
        pollinations_model: Optional[str] = None
    ):
        """
        Initialize the Image Generator.
        :param provider: 'pollinations' (recommended) or 'gemini'
        :param api_key: Gemini API key (required if provider is 'gemini')
        :param model_name: Gemini image model (defaults to 'gemini-2.5-flash-image')
        :param pollinations_api_key: Optional secret key for Pollinations AI
        :param pollinations_model: Pollinations image model (defaults to 'flux')
        """
        self.provider = (provider or get_config("IMAGE_PROVIDER", "pollinations")).lower()
        self.api_key = api_key or get_config("GEMINI_API_KEY")
        self.model_name = model_name or get_config("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        self.pollinations_api_key = pollinations_api_key or get_config("POLLINATIONS_API_KEY")
        self.pollinations_model = pollinations_model or get_config("POLLINATIONS_IMAGE_MODEL", "flux")

        if self.provider == "gemini" and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(
                f"ImageGenerator initialized with provider='gemini', model='{self.model_name}' "
                f"(Key: {mask_secret(self.api_key)})"
            )
        else:
            self.client = None
            key_status = mask_secret(self.pollinations_api_key) if self.pollinations_api_key else "[None - Public Tier]"
            logger.info(
                f"ImageGenerator initialized with provider='pollinations', model='{self.pollinations_model}' "
                f"(Key: {key_status})"
            )

    def _get_gemini_client(self, api_key_override: Optional[str] = None) -> genai.Client:
        key = api_key_override or self.api_key
        if not key:
            raise InvalidAPIKeyError(
                "Gemini API Key is missing. Please provide a key in settings or switch provider to 'pollinations'."
            )
        return genai.Client(api_key=key)

    def construct_scene_prompt(
        self,
        scene: Dict[str, Any],
        characters: list,
        art_style: str = "Colorful 2D cartoon"
    ) -> str:
        """
        Constructs an optimized visual prompt for the image generation model.
        Incorporate art style, environment, persistent character traits, and scene action.
        """
        # If enriched prompt is already present, ensure it has character consistency
        if scene.get("enriched_image_prompt"):
            return scene["enriched_image_prompt"]

        chars_map = {c["name"].lower(): c for c in characters}
        scene_chars = scene.get("characters_present", [])
        
        char_descriptions = []
        for cname in scene_chars:
            matched = chars_map.get(cname.lower())
            if matched:
                char_descriptions.append(
                    f"{matched['name']}: {matched['appearance']}"
                )

        style_descriptors = {
            "Colorful 2D cartoon": "classic 2D animated cartoon, clean cel-shaded linework, rich flat colors, vibrant and whimsical",
            "Storybook illustration": "charming storybook illustration, soft textured brush strokes, warm pastel tones, golden lighting",
            "Anime-inspired illustration": "anime cartoon aesthetic, expressive eyes, vibrant cinematic lighting, crisp detailed artwork",
            "Watercolor illustration": "delicate watercolor cartoon painting, soft fluid pigment washes, gentle paper texture, dreamy ambience",
            "3D animated-film-inspired illustration": "3D animated feature film CGI style, smooth ambient lighting, tactile materials, depth of field"
        }

        chosen_style = style_descriptors.get(art_style, f"{art_style} style cartoon illustration")
        
        parts = [
            f"Children's cartoon illustration in {chosen_style}.",
            f"Scene: {scene.get('heading', '')}.",
            f"Environment & Setting: {scene.get('environment', '')}."
        ]
        
        if char_descriptions:
            parts.append("Characters: " + "; ".join(char_descriptions) + ".")
            
        parts.append(f"Visual Action: {scene.get('image_prompt', scene.get('narration', ''))}.")
        parts.append("Family-friendly, engaging expressions, vibrant colors, clear focal point, cinematic composition, no watermark, no text.")
        
        return " ".join(parts)

    def generate_single_scene_image(
        self,
        scene: Dict[str, Any],
        characters: list,
        art_style: str,
        output_filepath: str,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        pollinations_api_key_override: Optional[str] = None,
        pollinations_model_override: Optional[str] = None
    ) -> str:
        """
        Generates and saves an illustration for a single scene.
        Returns the absolute path to the saved image.
        Updates scene['illustration_status'] ('Generating' -> 'Generated' or 'Failed').
        Uses atomic file write to prevent corrupting existing images.
        """
        provider = (provider_override or self.provider).lower()
        prompt = self.construct_scene_prompt(scene, characters, art_style)
        scene_num = scene.get("scene_number", "?")
        
        logger.info(f"Generating image for Scene #{scene_num} using [{provider}]...")
        scene["illustration_status"] = "Generating"

        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = out_path.with_suffix(".tmp.png")

        try:
            if provider == "gemini":
                image_bytes = self._generate_with_gemini(prompt, api_key_override)
            elif provider == "pollinations":
                image_bytes = self._generate_with_pollinations(
                    prompt,
                    api_key_override=pollinations_api_key_override,
                    model_override=pollinations_model_override
                )
            else:
                raise UnsupportedModelError(f"Unsupported image provider: '{provider}'. Choose 'pollinations' or 'gemini'.")

            # Validate the bytes with Pillow before saving
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                if pil_img.mode in ("RGBA", "P"):
                    rgb_img = Image.new("RGB", pil_img.size, (255, 255, 255))
                    if pil_img.mode == "RGBA":
                        rgb_img.paste(pil_img, mask=pil_img.split()[3])
                    else:
                        rgb_img.paste(pil_img)
                    rgb_img.save(temp_path, format="PNG")
                else:
                    pil_img.save(temp_path, format="PNG")

            # Verify saved temp file
            if not validate_image_file(str(temp_path)):
                raise ImageGenerationError("Generated image failed Pillow integrity validation.")

            # Atomically replace destination file
            if out_path.exists():
                out_path.unlink()
            temp_path.rename(out_path)

            # Mark state as Generated only after confirmed validation and file write
            scene["illustration_status"] = "Generated"
            scene["image_path"] = str(out_path)
            scene["illustration_error"] = None

            logger.info(f"Scene #{scene_num} image saved successfully to {out_path.name}")
            return str(out_path)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            scene["illustration_status"] = "Failed"
            scene["illustration_error"] = str(e)
            logger.error(f"Image generation failed for Scene #{scene_num}: {e}")
            raise

    def _generate_with_gemini(self, prompt: str, api_key_override: Optional[str] = None) -> bytes:
        """
        Generate cartoon illustration using Google Gemini Developer API.
        Calls client.models.generate_content with supported multimodal image generation model
        (e.g. gemini-2.5-flash-image, gemini-3.1-flash-image).
        """
        if self.model_name.startswith("imagen-"):
            raise UnsupportedModelError(
                f"Model '{self.model_name}' requires Gemini Enterprise Agent Platform mode and is not supported "
                "with Google AI Studio API keys in the Gemini Developer API. "
                "Please configure a Gemini Developer API image model such as 'gemini-2.5-flash-image' or 'gemini-3.1-flash-image'."
            )

        client = self._get_gemini_client(api_key_override)
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
            )

            image_bytes = None

            # 1. Inspect response.parts directly
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

            # 2. Fallback inspection through response.candidates
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
                        if image_bytes:
                            break

            if not image_bytes:
                refusal_text = ""
                try:
                    if hasattr(response, "text") and response.text:
                        refusal_text = f" Model text response: '{response.text[:200]}'"
                except Exception:
                    pass
                raise ImageGenerationError(
                    f"Gemini API model '{self.model_name}' did not return an inline image in the response.{refusal_text} "
                    f"Please verify model '{self.model_name}' is enabled and supports image generation via generate_content."
                )

            return image_bytes

        except (APIError, ClientError, ServerError) as api_err:
            error_code = getattr(api_err, "code", 0)
            error_msg = getattr(api_err, "message", None) or str(api_err)
            error_status = getattr(api_err, "status", "") or ""
            error_details = str(getattr(api_err, "details", ""))
            full_error_str = f"{error_msg} {error_status} {error_details}".lower()
            
            logger.error(f"Gemini Image API Error ({error_code} {error_status}): {error_msg}")

            # Quota / Rate Limit detection
            if (
                error_code == 429
                or "resource_exhausted" in full_error_str
                or "quota" in full_error_str
                or "free-tier" in full_error_str
            ):
                is_zero_quota = (
                    "requests per minute: 0" in full_error_str
                    or "requests per day: 0" in full_error_str
                    or "tokens per minute: 0" in full_error_str
                    or "limit: 0" in full_error_str
                )
                
                retry_match = re.search(r'retry.*?(\d+)\s*(?:s|sec)', full_error_str)
                retry_after = int(retry_match.group(1)) if retry_match else None

                if is_zero_quota:
                    raise QuotaExceededError(
                        f"Google Gemini Image Quota Exhausted: Your Google Cloud/AI Studio project has 0 free-tier quota allocated "
                        f"for model '{self.model_name}' (Free-tier requests per minute/day: 0). "
                        "Native image generation in the Gemini API requires a project with Pay-As-You-Go billing enabled. "
                        "To generate cartoon illustrations without billing, switch the Image Generation Provider to 'Pollinations AI' in the sidebar.",
                        retry_after=retry_after,
                        is_zero_quota=True
                    )
                else:
                    delay_msg = f" Please retry in {retry_after} seconds." if retry_after else " Please wait before retrying."
                    raise QuotaExceededError(
                        f"Google Gemini API Rate Limit Exceeded (HTTP 429): Quota limit reached for model '{self.model_name}'.{delay_msg}",
                        retry_after=retry_after,
                        is_zero_quota=False
                    )

            # Authentication / API Key error detection
            if error_code in (401, 403) or "api_key" in full_error_str or "permission_denied" in full_error_str:
                raise InvalidAPIKeyError(
                    f"Authentication Failed ({error_code}): Your Gemini API key is invalid, inactive, or lacks permissions for model '{self.model_name}'. "
                    "Please verify your API key in Google AI Studio (https://aistudio.google.com/)."
                )

            # Model Not Found / Unsupported model detection
            if error_code == 404 or "not_found" in full_error_str or "not found" in full_error_str:
                raise UnsupportedModelError(
                    f"Model '{self.model_name}' was not found or is not accessible in the Gemini Developer API ({error_msg}). "
                    "Please select a supported Gemini image model such as 'gemini-2.5-flash-image'."
                )

            raise ImageGenerationError(f"Gemini API Error ({error_code}): {error_msg}")

        except (requests.exceptions.RequestException, ConnectionError, TimeoutError, OSError) as net_err:
            logger.error(f"Network error during Gemini image generation: {net_err}")
            raise ImageNetworkError(f"Network error connecting to Gemini API: {str(net_err)}")

        except ImageGenerationError:
            raise

        except Exception as e:
            logger.error(f"Unexpected image generation failure: {e}", exc_info=True)
            raise ImageGenerationError(f"Image generation failed: {str(e)}")

    def _generate_with_pollinations(
        self,
        prompt: str,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> bytes:
        """
        Generate cartoon illustration via Pollinations AI.
        Follows the official gen.pollinations.ai API documentation.
        - Supports authenticated requests to gen.pollinations.ai with POLLINATIONS_API_KEY.
        - Gracefully falls back to the open tier endpoint for zero-cost operation.
        """
        api_key = api_key_override or self.pollinations_api_key or get_config("POLLINATIONS_API_KEY")
        model = model_override or self.pollinations_model or get_config("POLLINATIONS_IMAGE_MODEL", "flux")
        
        encoded_prompt = urllib.parse.quote(prompt[:350])
        headers = {
            "User-Agent": "ToonTalesAI/1.0"
        }

        # Build candidate URLs in priority order:
        # 1. If API key exists, use official gen.pollinations.ai with Bearer auth
        # 2. Open tier endpoint on image.pollinations.ai
        urls_to_try = []
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            model_query = f"?model={urllib.parse.quote(model)}&width=512&height=512&nologo=true"
            urls_to_try.append(("gen.pollinations.ai", f"https://gen.pollinations.ai/image/{encoded_prompt}{model_query}"))
        
        # Always include the open tier endpoint
        urls_to_try.append(("image.pollinations.ai", f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"))

        last_error = None
        for endpoint_name, target_url in urls_to_try:
            for attempt in range(2):
                try:
                    logger.debug(f"Calling Pollinations ({endpoint_name}), attempt {attempt + 1}...")
                    response = requests.get(target_url, headers=headers, timeout=(10, 45))
                    
                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "")
                        if "image" in content_type or response.content.startswith(b"\x89PNG") or response.content.startswith(b"\xff\xd8") or len(response.content) >= 50:
                            return response.content
                        elif "application/json" in content_type:
                            try:
                                err_data = response.json()
                                last_error = f"Pollinations returned error: {err_data}"
                            except Exception:
                                last_error = f"Pollinations returned invalid non-image response"

                    elif response.status_code == 401:
                        if api_key:
                            raise InvalidAPIKeyError(
                                "Pollinations API Key is invalid or expired. Check your key at enter.pollinations.ai/keys."
                            )
                        # Without key, continue to next endpoint
                        last_error = "Pollinations authentication required on this endpoint."
                        break

                    elif response.status_code in (402, 429) or "free limit" in response.text.lower():
                        retry_match = re.search(r'retry after (\d+)', response.text.lower())
                        retry_sec = int(retry_match.group(1)) if retry_match else 45
                        raise QuotaExceededError(
                            f"Pollinations rate limit reached. Please wait {retry_sec} seconds before clicking 'Regenerate Illustration'.",
                            retry_after=retry_sec,
                            is_zero_quota=False
                        )

                    elif response.status_code == 503:
                        time.sleep(2 * (attempt + 1))
                        last_error = "Pollinations service temporarily busy (HTTP 503)."
                        continue

                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:100]}"

                except (QuotaExceededError, InvalidAPIKeyError):
                    raise
                except requests.exceptions.Timeout as t_err:
                    last_error = f"Timeout connecting to Pollinations AI ({t_err})"
                except requests.exceptions.RequestException as req_err:
                    last_error = f"Network connection error: {req_err}"

                time.sleep(1.5 * (attempt + 1))

        if last_error and ("timeout" in str(last_error).lower() or "connection" in str(last_error).lower()):
            raise ImageNetworkError(f"Pollinations AI connection failed: {last_error}")

        raise ImageGenerationError(f"Pollinations AI image generation failed: {last_error}")

    def generate_all_scenes(
        self,
        story_dict: Dict[str, Any],
        story_dir: str,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        pollinations_api_key_override: Optional[str] = None,
        pollinations_model_override: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Sequentially generates illustrations for all scenes in the story.
        Preserves successfully generated images if a later scene fails.
        Halts immediately on zero-quota error to avoid futile repeated API calls.
        """
        scenes = story_dict.get("scenes", [])
        characters = story_dict.get("characters", [])
        art_style = story_dict.get("art_style", "Colorful 2D cartoon")
        total_scenes = len(scenes)

        logger.info(f"Starting image generation for {total_scenes} scenes...")
        failures: List[Tuple[int, str]] = []
        zero_quota_encountered = False

        for idx, scene in enumerate(scenes):
            scene.setdefault("image_path", None)
            scene_num = scene.get("scene_number", idx + 1)
            filename = f"scene_{scene_num:02d}.png"
            dest_path = os.path.join(story_dir, filename)

            # If zero-quota error was hit previously, skip remaining scenes without hammering API
            if zero_quota_encountered:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = "Skipped: Gemini project has 0 free-tier image quota."
                failures.append((scene_num, "Skipped due to project zero quota exhaustion."))
                continue

            if progress_callback:
                progress_callback(
                    idx, total_scenes, f"Creating illustration for Scene {scene_num} of {total_scenes}..."
                )

            try:
                img_path = self.generate_single_scene_image(
                    scene=scene,
                    characters=characters,
                    art_style=art_style,
                    output_filepath=dest_path,
                    provider_override=provider_override,
                    api_key_override=api_key_override,
                    pollinations_api_key_override=pollinations_api_key_override,
                    pollinations_model_override=pollinations_model_override
                )
                scene["image_path"] = img_path
                scene["illustration_status"] = "Generated"
                scene["illustration_error"] = None

            except QuotaExceededError as q_err:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = str(q_err)
                failures.append((scene_num, str(q_err)))
                logger.warning(f"Scene {scene_num} quota exceeded: {q_err}")
                
                # If zero-quota (free tier 0 requests/day), stop hammering the API for remaining scenes
                if q_err.is_zero_quota:
                    logger.warning("Zero-quota detected. Skipping remaining scenes to avoid futile repeated requests.")
                    zero_quota_encountered = True

            except Exception as e:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = str(e)
                failures.append((scene_num, str(e)))
                logger.warning(f"Scene {scene_num} image generation failed: {e}")

        if progress_callback:
            progress_callback(total_scenes, total_scenes, "Image generation processing complete.")

        story_dict["image_generation_failures"] = failures
        if failures:
            logger.warning(f"Image generation completed with {len(failures)} failed scenes.")

        return story_dict
