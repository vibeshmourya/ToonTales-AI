"""
ToonTales AI — Multi-Provider Cartoon Image Generation Engine
Supports:
1. Pollinations AI (Recommended default: zero-cost cloud diffusion or authenticated gen.pollinations.ai)
2. Google Gemini Developer API (gemini-2.5-flash-image)
3. OpenAI DALL-E (dall-e-3, dall-e-2)
4. OpenRouter image-capable models (e.g. recraft/recraft-v4.1-flash-v2:free, microsoft/mai-image-2.7-flash)

Features:
- Independent provider and model configuration
- Scene illustration state tracking: 'Pending', 'Generating', 'Generated', 'Failed', 'Waiting for cooldown', 'Outdated'
- Sequential generation with resumption (skips already valid images by default)
- Duplicate request prevention via concurrency locking
- Cooldown and rate limit management with bounded backoff
- Enforces character visual consistency across scenes
- Pillow integrity validation before replacing images
- Atomic saving to preserve existing valid images if regeneration fails
"""

import io
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import requests
from PIL import Image

from services.providers import (
    get_image_provider,
    BaseImageProvider,
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
from services.rate_limit_service import RateLimitService
from services.usage_tracker import UsageTracker
from utils.helpers import get_config, mask_secret, validate_image_file, validate_image_bytes

logger = logging.getLogger("ToonTalesAI.ImageGenerator")


# =====================================================================
# Species Anatomy Guards
# =====================================================================

def get_species_guard(text: str) -> str:
    """
    Returns anatomical guard cues to prevent diffusion models from misrepresenting species.
    For example, ensures snakes remain limbless serpents and do not grow lizard legs.
    """
    t = text.lower()
    guards = []
    if any(k in t for k in ["snake", "serpent", "python", "cobra", "viper"]):
        guards.append("limbless serpent, long tubular snake body, smooth scales, strictly NO arms, strictly NO legs, strictly NO lizard limbs, NOT a lizard, NOT a crocodile, NOT a dragon")
    if any(k in t for k in ["squirrel", "chipmunk"]):
        guards.append("small woodland rodent, large bushy fluffy tail, tiny paws, small rounded ears")
    if any(k in t for k in ["rabbit", "bunny", "hare"]):
        guards.append("long upright bunny ears, soft fur, small twitchy nose, short fluffy cotton-tail")
    if any(k in t for k in ["bear"]):
        guards.append("chubby rounded bear body, round ears, short tail")
    if any(k in t for k in ["fox"]):
        guards.append("slender canid, pointed upright ears, sharp snout, large bushy white-tipped tail")
    if any(k in t for k in ["bird", "owl", "sparrow", "eagle", "parrot"]):
        guards.append("feathered avian body, two wings, beak, two bird feet, strictly NO four legs")
    if any(k in t for k in ["fish"]):
        guards.append("aquatic fish body, fins, gills, scales, fish tail, strictly NO legs, strictly NO arms")
    return "; ".join(guards)


# =====================================================================
# Backwards-Compatible Exception Aliases
# =====================================================================

ImageGenerationError = ProviderError
InvalidAPIKeyError = AuthenticationError
UnsupportedModelError = ModelNotFoundError
ImageNetworkError = NetworkError


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
        Initialize the Image Generator with provider and model settings.
        :param provider: 'pollinations', 'gemini', 'openai', 'openrouter'
        :param api_key: Provider API key
        :param model_name: Image model identifier
        :param pollinations_api_key: Optional key specifically for Pollinations AI
        :param pollinations_model: Optional model for Pollinations AI
        """
        self.provider_name = (provider or get_config("IMAGE_PROVIDER", "gemini")).lower()
        self.api_key = api_key
        self.model_name = model_name
        self.pollinations_api_key = pollinations_api_key or get_config("POLLINATIONS_API_KEY")
        self.pollinations_model = pollinations_model or get_config("POLLINATIONS_IMAGE_MODEL", "flux")

        logger.info(f"ImageGenerator initialized with provider='{self.provider_name}', model='{self.model_name}'")

    def _get_provider_adapter(
        self,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None
    ) -> BaseImageProvider:
        p_name = (provider_override or self.provider_name).lower()
        key = api_key_override or self.api_key
        if p_name == "pollinations" and not key:
            key = self.pollinations_api_key

        return get_image_provider(p_name, api_key=key)

    def _get_gemini_client(self, api_key_override: Optional[str] = None):
        prov = self._get_provider_adapter("gemini", api_key_override)
        return prov._get_client(api_key_override)

    def _generate_with_gemini(self, prompt: str, api_key_override: Optional[str] = None) -> bytes:
        prov = self._get_provider_adapter("gemini", api_key_override)
        prov._custom_get_client = lambda key: self._get_gemini_client(key)
        bytes_data, _ = prov.generate_image(prompt, model=self.model_name, api_key=api_key_override)
        return bytes_data

    def _generate_with_pollinations(
        self,
        prompt: str,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> bytes:
        import services.providers.pollinations_provider as pp
        pp.requests = requests
        prov = self._get_provider_adapter("pollinations", api_key_override)
        bytes_data, _ = prov.generate_image(
            prompt,
            model=model_override or self.pollinations_model,
            api_key=api_key_override
        )
        return bytes_data

    def construct_scene_prompt(
        self,
        scene: Dict[str, Any],
        characters: list,
        art_style: str = "Colorful 2D cartoon"
    ) -> str:
        """
        Constructs an optimized, character-first visual prompt for the image generation model.
        Enforces character primacy: characters and their exact actions lead the prompt to ensure
        diffusion attention prioritizes the subjects rather than creating an empty landscape.
        Embeds species anatomy guards (e.g. limbless serpent for snakes, no lizard legs).
        """
        # If prompt was already enriched with character primacy and scene was not edited, reuse it
        existing_prompt = scene.get("enriched_image_prompt")
        if (
            existing_prompt
            and scene.get("illustration_status") not in ("Pending", "Outdated")
            and not existing_prompt.startswith("Children's cartoon storybook illustration in")
            and not existing_prompt.startswith("Cartoon storybook illustration in")
        ):
            return existing_prompt

        # Safely map characters (defensive against strings, dicts, or missing keys)
        chars_map = {}
        for c in (characters or []):
            if isinstance(c, dict) and "name" in c and c["name"]:
                chars_map[str(c["name"]).strip().lower()] = c
            elif isinstance(c, str) and c.strip():
                chars_map[c.strip().lower()] = {
                    "name": c.strip(),
                    "role": "Character",
                    "appearance": f"cartoon character named {c.strip()}",
                    "species": c.strip()
                }

        raw_chars = scene.get("characters_present", [])
        if isinstance(raw_chars, str):
            scene_chars = [c.strip() for c in raw_chars.split(",") if c.strip()]
        elif isinstance(raw_chars, list):
            scene_chars = [str(c).strip() for c in raw_chars if c]
        else:
            scene_chars = []

        # FALLBACK 1: If scene_chars is empty, scan narration, heading, and action for character names
        if not scene_chars:
            scene_search_text = f"{scene.get('heading', '')} {scene.get('narration', '')} {scene.get('character_actions', '')} {scene.get('image_prompt', '')}".lower()
            for cname, cdata in chars_map.items():
                if cname in scene_search_text:
                    scene_chars.append(cdata.get("name", cname))

        # FALLBACK 2: If still empty, default to the protagonist (first character)
        if not scene_chars and characters:
            first_c = characters[0]
            if isinstance(first_c, dict):
                scene_chars.append(first_c.get("name", "Protagonist"))
            elif isinstance(first_c, str):
                scene_chars.append(first_c)

        style_descriptors = {
            "Colorful 2D cartoon": "classic 2D animated cartoon, clean cel-shaded linework, rich flat colors, vibrant and whimsical, storybook framing",
            "Storybook illustration": "charming storybook illustration, children's picture-book style, soft textured brush strokes, warm pastel tones, golden ambient lighting",
            "Anime-inspired illustration": "vibrant anime cartoon aesthetic, expressive eyes, dynamic cinematic lighting, crisp detailed artwork",
            "Watercolor illustration": "delicate watercolor cartoon painting, soft fluid pigment washes, gentle paper texture, dreamy whimsical ambience",
            "3D animated-film-inspired illustration": "3D animated feature film CGI style, smooth ambient lighting, tactile materials, depth of field"
        }
        chosen_style = style_descriptors.get(art_style, f"{art_style} style cartoon illustration")

        heading = scene.get("heading") or scene.get("scene_title", "")
        env = scene.get("environment") or scene.get("setting", "storybook setting")
        action = scene.get("character_actions") or scene.get("image_prompt") or scene.get("narration", "exploring")
        tone = scene.get("emotional_tone") or "cheerful and adventurous"
        lighting = scene.get("lighting") or "bright whimsical daylight"

        hero_char_summaries = []
        char_detail_blocks = []
        for cname in scene_chars:
            matched = chars_map.get(cname.lower())
            if not matched:
                for k, v in chars_map.items():
                    if k in cname.lower() or cname.lower() in k:
                        matched = v
                        break
            if matched and isinstance(matched, dict):
                c_name = matched.get("name", cname)
                c_app = matched.get("appearance", "cartoon character")
                c_role = matched.get("role", "character")
                c_species = matched.get("species", "")

                species_guard = get_species_guard(f"{c_name} {c_app} {c_species}")
                guard_str = f" [{species_guard}]" if species_guard else ""

                hero_char_summaries.append(f"{c_name} ({c_app})")
                char_detail_blocks.append(f"{c_name} ({c_role}): {c_app}{guard_str}")
            else:
                species_guard = get_species_guard(cname)
                guard_str = f" [{species_guard}]" if species_guard else ""
                hero_char_summaries.append(f"{cname}")
                char_detail_blocks.append(f"{cname} (character): cartoon {cname}{guard_str}")

        char_hero_text = " and ".join(hero_char_summaries) if hero_char_summaries else "the story characters"

        # Construct Character-Primacy Prompt Template
        prompt_parts = [
            # 1. HERO OPENING: Characters and action FIRST!
            f"{chosen_style} children's picture-book illustration of {char_hero_text} {action}.",
            # 2. PRIMARY CHARACTERS & ANATOMY GUARDS
            f"PRIMARY CHARACTERS: {'; '.join(char_detail_blocks)}.",
            # 3. SCENE ACTION & INTERACTION
            f"SCENE ACTION & POSE: Characters actively engaged in: {action}. Facial expressions: {tone}.",
            # 4. ENVIRONMENT & BACKGROUND (after characters)
            f"ENVIRONMENT & SETTING: Background environment is {env}. Scene title context: {heading}. Lighting: {lighting}.",
        ]

        if scene.get("important_objects"):
            objs = scene["important_objects"]
            if isinstance(objs, list) and objs:
                prompt_parts.append(f"KEY OBJECTS: {', '.join(str(o) for o in objs)}.")

        prompt_parts.extend([
            # 5. COMPOSITION & FRAMING
            "COMPOSITION: Characters are large, centered, and prominent in the foreground and midground with clear focal point. Storybook picture-book framing, not an empty landscape.",
            # 6. NEGATIVE PROMPT / EXCLUSIONS
            "EXCLUSIONS: No empty landscape without characters, no missing characters, no incorrect animal anatomy, no lizard legs on snake, no dinosaur, no text, no captions, no watermarks, no logos, no artist signatures, no borders."
        ])

        prompt = " ".join(prompt_parts)
        scene["enriched_image_prompt"] = prompt
        return prompt

    def generate_single_scene_image(
        self,
        scene: Dict[str, Any],
        characters: list,
        art_style: str,
        output_filepath: str,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        pollinations_api_key_override: Optional[str] = None,
        pollinations_model_override: Optional[str] = None,
        story_id: str = "default_story"
    ) -> str:
        """
        Generates and saves an illustration for a single scene.
        Returns the absolute path to the saved image.
        Updates scene['illustration_status'] ('Generating' -> 'Generated' or 'Failed' / 'Waiting for cooldown').
        Uses atomic file write to prevent corrupting existing images.
        """
        provider_key = (provider_override or self.provider_name).lower()
        scene_num = scene.get("scene_number", 1)

        # 1. Check duplicate generation concurrency lock
        if not RateLimitService.acquire_scene_lock(story_id, scene_num):
            logger.warning(f"Scene #{scene_num} is already being generated. Skipping duplicate call.")
            return scene.get("image_path", output_filepath)

        # 2. Check provider cooldown
        cooldown_sec = RateLimitService.get_cooldown_remaining(provider_key)
        if cooldown_sec > 0:
            scene["illustration_status"] = "Waiting for cooldown"
            scene["illustration_error"] = f"Provider '{provider_key}' is in cooldown. Retry in {cooldown_sec}s."
            RateLimitService.release_scene_lock(story_id, scene_num)
            raise RateLimitError(
                f"Provider '{provider_key}' is currently cooling down. Please wait {cooldown_sec} seconds.",
                provider=provider_key,
                retry_after=cooldown_sec
            )

        prompt = self.construct_scene_prompt(scene, characters, art_style)
        scene["illustration_status"] = "Generating"

        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = out_path.with_suffix(".tmp.png")

        usage = None
        try:
            if provider_key == "gemini":
                image_bytes = self._generate_with_gemini(prompt, api_key_override)
                chosen_model = model_override or self.model_name or "gemini-2.5-flash-image"
                usage = GenerationUsage(
                    provider="gemini",
                    model=chosen_model,
                    operation="image",
                    images_generated=1,
                    cost_type="estimated"
                )
            elif provider_key == "pollinations":
                image_bytes = self._generate_with_pollinations(
                    prompt,
                    api_key_override=pollinations_api_key_override or api_key_override,
                    model_override=pollinations_model_override or model_override
                )
                chosen_model = pollinations_model_override or model_override or self.pollinations_model
                key_present = bool(pollinations_api_key_override or api_key_override or self.pollinations_api_key)
                usage = GenerationUsage(
                    provider="pollinations",
                    model=chosen_model,
                    operation="image",
                    images_generated=1,
                    cost_type="reported" if key_present else "zero_cost"
                )
            else:
                adapter = self._get_provider_adapter(provider_key, api_key_override)
                chosen_model = model_override or self.model_name
                image_bytes, usage = adapter.generate_image(
                    prompt=prompt,
                    model=chosen_model,
                    size=(512, 512),
                    api_key=api_key_override or self.api_key
                )

            # Validate decoded image bytes before processing or saving
            is_valid, val_err = validate_image_bytes(image_bytes)
            if not is_valid:
                raise ProviderError(
                    f"Generated image failed byte-level integrity validation: {val_err}",
                    provider=provider_key
                )

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
                raise ProviderError("Generated image failed Pillow integrity validation.", provider=provider_key)

            # Atomically replace destination file
            if out_path.exists():
                out_path.unlink()
            temp_path.rename(out_path)

            # Mark state as Generated only after confirmed validation and file write
            scene["illustration_status"] = "Generated"
            scene["image_path"] = str(out_path)
            scene["illustration_error"] = None
            scene["image_provider"] = provider_key
            scene["image_model"] = chosen_model

            # Record successful usage ONLY after verified Pillow validation and atomic disk write
            if usage:
                UsageTracker.record_usage_obj(usage, status="success")
            else:
                UsageTracker.record(
                    provider=provider_key,
                    model=chosen_model or "unknown",
                    operation="image",
                    status="success",
                    images_count=1,
                    estimated_cost=0.04 if provider_key == "openai" else 0.0,
                    cost_type="zero_cost" if provider_key == "pollinations" else "estimated"
                )

            logger.info(f"Scene #{scene_num} image saved successfully to {out_path.name}")
            return str(out_path)

        except RateLimitError as r_err:
            if temp_path.exists():
                temp_path.unlink()
            if r_err.retry_after:
                RateLimitService.record_cooldown(provider_key, r_err.retry_after)
            scene["illustration_status"] = "Waiting for cooldown" if not getattr(r_err, "is_quota_exhausted", False) else "Failed"
            scene["illustration_error"] = str(r_err)
            UsageTracker.record(provider=provider_key, model=self.model_name or "default", operation="image", status="rate_limited")
            logger.warning(f"Rate limit for Scene #{scene_num} on [{provider_key}]: {r_err}")
            raise

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            scene["illustration_status"] = "Failed"
            scene["illustration_error"] = str(e)
            UsageTracker.record(provider=provider_key, model=self.model_name or "default", operation="image", status="failed")
            logger.error(f"Image generation failed for Scene #{scene_num} on [{provider_key}]: {e}")
            raise

        finally:
            RateLimitService.release_scene_lock(story_id, scene_num)

    def generate_all_scenes(
        self,
        story_dict: Dict[str, Any],
        story_dir: str,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        pollinations_api_key_override: Optional[str] = None,
        pollinations_model_override: Optional[str] = None,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Sequentially generates illustrations for all scenes in the story.
        - Identifies and skips scenes that already have valid illustrations.
        - Preserves successfully generated images if a later scene fails.
        - Halts immediately on zero-quota error or enters cooldown to avoid futile repeated requests.
        - Allows generation to resume later without restarting completed scenes.
        """
        scenes = story_dict.get("scenes", [])
        characters = story_dict.get("characters", [])
        art_style = story_dict.get("art_style", "Colorful 2D cartoon")
        total_scenes = len(scenes)
        story_id = story_dict.get("story_id", Path(story_dir).name)

        logger.info(f"Starting sequential image generation for {total_scenes} scenes...")
        failures: List[Tuple[int, str]] = []
        zero_quota_encountered = False
        cooldown_encountered = False

        for idx, scene in enumerate(scenes):
            scene.setdefault("image_path", None)
            scene_num = scene.get("scene_number", idx + 1)
            filename = f"scene_{scene_num:02d}.png"
            dest_path = os.path.join(story_dir, filename)

            # Check if valid illustration already exists on disk
            existing_path = scene.get("image_path") or dest_path
            if skip_existing and os.path.isfile(existing_path) and validate_image_file(existing_path):
                scene["image_path"] = existing_path
                if scene.get("illustration_status") != "Outdated":
                    scene["illustration_status"] = "Generated"
                    logger.info(f"Scene #{scene_num} already has valid illustration. Skipping.")
                    continue

            # Circuit breaker: zero quota or active cooldown halts repeated futile calls
            if zero_quota_encountered:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = "Skipped: Provider zero quota exhaustion."
                failures.append((scene_num, "Skipped due to zero quota exhaustion."))
                continue
            if cooldown_encountered:
                scene["illustration_status"] = "Waiting for cooldown"
                scene["illustration_error"] = "Skipped: Provider cooling down."
                failures.append((scene_num, "Skipped due to cooldown."))
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
                    model_override=model_override,
                    pollinations_api_key_override=pollinations_api_key_override,
                    pollinations_model_override=pollinations_model_override,
                    story_id=story_id
                )
                scene["image_path"] = img_path
                scene["illustration_status"] = "Generated"
                scene["illustration_error"] = None

            except QuotaExceededError as q_err:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = str(q_err)
                failures.append((scene_num, str(q_err)))
                logger.warning(f"Scene {scene_num} quota exceeded: {q_err}")
                
                if getattr(q_err, "is_quota_exhausted", False) or getattr(q_err, "is_zero_quota", False):
                    logger.warning("Zero-quota or quota exhaustion detected. Stopping sequential generation.")
                    zero_quota_encountered = True

            except RateLimitError as r_err:
                scene["illustration_status"] = "Waiting for cooldown"
                scene["illustration_error"] = str(r_err)
                failures.append((scene_num, str(r_err)))
                logger.warning(f"Scene {scene_num} rate limited: {r_err}")
                cooldown_encountered = True

            except Exception as e:
                scene["illustration_status"] = "Failed"
                scene["illustration_error"] = str(e)
                failures.append((scene_num, str(e)))
                logger.warning(f"Scene {scene_num} image generation failed: {e}")

        if progress_callback:
            progress_callback(total_scenes, total_scenes, "Image generation processing complete.")

        story_dict["image_generation_failures"] = failures
        if failures:
            logger.warning(f"Image generation completed with {len(failures)} failed/skipped scenes.")

        return story_dict
