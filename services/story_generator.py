"""
ToonTales AI — Multi-Provider Story Generation Service
Orchestrates text generation across Google Gemini, OpenAI, Anthropic Claude, and OpenRouter.
Enforces character consistency, Pydantic schema validation, local JSON repair,
request deduplication, and caching.
"""

import json
import logging
import re
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, model_validator

from google import genai

from services.providers import (
    get_story_provider,
    BaseStoryProvider,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    CapabilityError,
    NetworkError,
    InvalidRequestError
)
from services.usage_tracker import UsageTracker
from services.cache_service import RequestCache
from services.json_repair import parse_and_repair_json
from utils.helpers import get_config, mask_secret

logger = logging.getLogger("ToonTalesAI.StoryGenerator")


# =====================================================================
# Pydantic Schemas for Structured JSON Output
# =====================================================================

class CharacterModel(BaseModel):
    name: str = Field(..., description="Character's name")
    role: str = Field("Protagonist", description="Role in story, e.g., 'Protagonist', 'Best Friend', 'Mentor'")
    appearance: str = Field(
        ...,
        description=(
            "Exhaustive visual description for cartoon illustration: species or type, "
            "body shape, primary colors, eyes, clothing, accessories, and distinctive features."
        )
    )
    personality: str = Field("Friendly and brave", description="Key personality traits and behavioral quirks")
    species: Optional[str] = Field(None, description="Character species or physical type (e.g. dragon, rabbit, boy)")

    @model_validator(mode="before")
    @classmethod
    def normalize_char(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Safe defaults if appearance or name are missing
            data.setdefault("role", "Protagonist")
            data.setdefault("personality", "Friendly and brave")
            if "appearance" not in data or not data["appearance"]:
                data["appearance"] = f"A charming cartoon {data.get('species') or 'character'}."
            return data
        elif isinstance(data, str):
            return {
                "name": data,
                "role": "Character",
                "appearance": f"A colorful cartoon character named {data}.",
                "personality": "Friendly."
            }
        return {
            "name": "Character",
            "role": "Character",
            "appearance": "A colorful cartoon character.",
            "personality": "Friendly."
        }


class SceneModel(BaseModel):
    scene_id: Optional[str] = Field(None, description="Stable unique scene identifier")
    scene_number: int = Field(..., description="Sequential scene index starting at 1")
    heading: str = Field(..., description="Catchy, short scene title, e.g., 'A Dream Amongst the Stars'")
    scene_title: Optional[str] = Field(None, description="Alias for heading")
    narration: str = Field(..., description="Engaging story narration for this scene (2-4 child-friendly sentences)")
    narrative_text: Optional[str] = Field(None, description="Alias for narration")
    visual_description: Optional[str] = Field(None, description="Visual description of scene composition")
    image_prompt: str = Field(
        ...,
        description=(
            "Detailed cartoon visual scene description describing character actions, "
            "emotions, environment, camera angle, and vibrant lighting."
        )
    )
    illustration_prompt: Optional[str] = Field(None, description="Alias for image_prompt")
    characters_present: List[str] = Field(default_factory=list, description="Names of characters visible in this scene")
    character_actions: Optional[str] = Field(None, description="Specific actions performed by characters in this scene")
    emotional_tone: Optional[str] = Field(None, description="Emotional atmosphere of the scene (e.g. playful, curious, triumphant)")
    environment: str = Field(..., description="Setting/backdrop description, e.g., 'A cozy bedroom lit by starlight'")
    setting: Optional[str] = Field(None, description="Alias for environment")
    important_objects: Optional[List[str]] = Field(default_factory=list, description="Key objects in the scene")
    lighting: Optional[str] = Field(None, description="Lighting and atmosphere (e.g. warm golden afternoon, twilight)")
    illustration_status: str = Field("Pending", description="Status of the scene illustration")
    image_path: Optional[str] = Field(None, description="Path to generated illustration image")
    illustration_error: Optional[str] = Field(None, description="Error message if generation failed")
    image_provider: Optional[str] = Field(None, description="Provider used to generate illustration")
    image_model: Optional[str] = Field(None, description="Model used to generate illustration")
    enriched_image_prompt: Optional[str] = Field(None, description="Enriched visual prompt with character details")

    @model_validator(mode="before")
    @classmethod
    def normalize_scene(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {
                "scene_number": 1,
                "heading": "Scene 1",
                "narration": str(data),
                "image_prompt": f"A cartoon illustration of {data}",
                "environment": "Enchanted setting",
                "characters_present": []
            }
        
        # Sync heading and scene_title
        heading = data.get("heading") or data.get("scene_title") or f"Scene {data.get('scene_number', 1)}"
        data["heading"] = heading
        data["scene_title"] = heading

        # Sync narration and narrative_text
        narration = data.get("narration") or data.get("narrative_text") or ""
        data["narration"] = narration
        data["narrative_text"] = narration

        # Sync image_prompt and illustration_prompt
        img_prompt = data.get("image_prompt") or data.get("illustration_prompt") or data.get("visual_description") or narration
        data["image_prompt"] = img_prompt
        data["illustration_prompt"] = img_prompt

        # Sync environment and setting
        env = data.get("environment") or data.get("setting") or "Enchanted setting"
        data["environment"] = env
        data["setting"] = env

        # Ensure scene_id exists
        s_num = data.get("scene_number", 1)
        if not data.get("scene_id"):
            data["scene_id"] = f"scene_{s_num:02d}"

        # Clean characters_present to list of strings
        cp = data.get("characters_present", [])
        if isinstance(cp, str):
            data["characters_present"] = [c.strip() for c in cp.split(",") if c.strip()]
        elif isinstance(cp, list):
            data["characters_present"] = [str(c).strip() for c in cp if c]
        else:
            data["characters_present"] = []

        return data


class StoryModel(BaseModel):
    title: str = Field(..., description="Creative, whimsical, and memorable story title")
    summary: str = Field(..., description="Engaging 2-sentence story synopsis")
    genre: str = Field("Adventure", description="Selected story genre")
    art_style: str = Field("Colorful 2D cartoon", description="Selected cartoon art style")
    characters: List[CharacterModel] = Field(..., min_length=1, description="List of story characters with persistent visuals")
    scenes: List[SceneModel] = Field(..., min_length=1, description="Chronological sequence of story scenes")
    story_id: Optional[str] = Field(None, description="Unique identifier for story")
    story_provider: Optional[str] = Field(None, description="Provider used to craft story")
    story_model: Optional[str] = Field(None, description="Model used to craft story")


# =====================================================================
# Story Generator Service Implementation
# =====================================================================

class StoryGeneratorService:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the multi-provider story generator.
        :param provider: 'gemini', 'openai', 'anthropic', 'openrouter'
        :param api_key: Optional API key override
        :param model_name: Optional model override
        """
        self.provider_name = (provider or get_config("STORY_PROVIDER", "gemini")).lower()
        self.api_key = api_key
        self.model_name = model_name
        self._provider_instance: Optional[BaseStoryProvider] = None

        logger.info(f"StoryGenerator initialized with provider='{self.provider_name}', model='{self.model_name}'")

    def _get_provider(self, provider_override: Optional[str] = None, api_key_override: Optional[str] = None) -> BaseStoryProvider:
        p_name = (provider_override or self.provider_name).lower()
        key = api_key_override or self.api_key
        prov = get_story_provider(p_name, api_key=key)
        if p_name == "gemini":
            prov._client_factory = genai.Client
        return prov

    def generate_story(
        self,
        story_idea: str,
        genre: str = "Adventure",
        target_age: str = "6–8 years",
        story_length: str = "Medium",
        art_style: str = "Colorful 2D cartoon",
        num_scenes: int = 5,
        ending_style: str = "Happy",
        api_key_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Generates a complete illustrated cartoon story using the configured provider.
        Enforces character visual consistency, strict structured JSON schema,
        request deduplication, and caching.
        """
        if not story_idea or not story_idea.strip():
            raise ValueError("Story idea cannot be empty. Please provide an imaginative concept.")

        p_name = (provider_override or self.provider_name).lower()
        chosen_model = model_override or self.model_name
        if not chosen_model:
            from services.providers import get_provider_capabilities
            try:
                chosen_model = get_provider_capabilities(p_name).default_text_model
            except Exception:
                chosen_model = "default"

        # 1. Check request cache to prevent duplicate token consumption
        cache_key = RequestCache.compute_story_key(
            story_idea=story_idea,
            genre=genre,
            target_age=target_age,
            story_length=story_length,
            art_style=art_style,
            num_scenes=num_scenes,
            ending_style=ending_style,
            provider=p_name,
            model=chosen_model
        )

        if not force_regenerate:
            cached_story = RequestCache.get_cached_story(cache_key)
            if cached_story:
                logger.info(f"Reusing cached story for '{cached_story.get('title')}'")
                return cached_story

        provider = self._get_provider(provider_override, api_key_override)

        logger.info(
            f"Generating story using [{provider.capabilities.name}] | Model: {chosen_model} | "
            f"Genre: {genre} | Age: {target_age} | Scenes: {num_scenes}"
        )

        try:
            story_dict, usage = provider.generate_story(
                story_idea=story_idea,
                genre=genre,
                target_age=target_age,
                story_length=story_length,
                art_style=art_style,
                num_scenes=num_scenes,
                ending_style=ending_style,
                model=chosen_model,
                api_key=api_key_override or self.api_key
            )

            # Record usage
            UsageTracker.record_usage_obj(usage, status="success")

            # Post-process: ensure character consistency references are embedded into all image prompts
            story_dict = self.enrich_character_consistency(story_dict)

            # Validate scene count
            if len(story_dict.get("scenes", [])) == 0:
                raise ValueError("Generated story does not contain any scenes.")

            # Record generation metadata
            story_dict["story_provider"] = provider.capabilities.name
            story_dict["story_model"] = chosen_model

            # Cache the completed story
            RequestCache.cache_story(cache_key, story_dict)

            logger.info(f"Story '{story_dict.get('title')}' successfully generated with {len(story_dict['scenes'])} scenes.")
            return story_dict

        except Exception as e:
            UsageTracker.record(
                provider=provider.capabilities.name,
                model=chosen_model or provider.capabilities.default_text_model,
                operation="story",
                status="failed"
            )
            logger.error(f"Story generation failure with provider {provider.capabilities.name}: {e}")
            raise

    @staticmethod
    def enrich_character_consistency(story_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhances scene image prompts by explicitly prefixing consistent character
        visual descriptions for every character participating in that scene.
        Guarantees safety against non-dict character inputs or corrupt structures.
        """
        if not isinstance(story_dict, dict):
            return story_dict

        # Safely map characters (handling dicts, strings, ints, or missing names)
        raw_characters = story_dict.get("characters", [])
        characters_map = {}
        if isinstance(raw_characters, list):
            for c in raw_characters:
                if isinstance(c, dict) and "name" in c and c["name"]:
                    characters_map[str(c["name"]).strip().lower()] = c
                elif isinstance(c, str) and c.strip():
                    characters_map[c.strip().lower()] = {
                        "name": c.strip(),
                        "role": "Character",
                        "appearance": f"A cartoon character named {c.strip()}.",
                        "personality": "Friendly."
                    }

        art_style = story_dict.get("art_style", "Colorful 2D cartoon")

        scenes = story_dict.get("scenes", [])
        if not isinstance(scenes, list):
            return story_dict

        for idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue

            scene_num = scene.get("scene_number", idx + 1)
            scene.setdefault("scene_id", f"scene_{scene_num:02d}")
            scene.setdefault("scene_number", scene_num)
            scene.setdefault("illustration_status", "Pending")

            base_prompt = scene.get("image_prompt") or scene.get("illustration_prompt") or scene.get("narration", "")
            raw_present = scene.get("characters_present", [])
            if isinstance(raw_present, str):
                present_chars = [c.strip() for c in raw_present.split(",") if c.strip()]
            elif isinstance(raw_present, list):
                present_chars = [str(c).strip() for c in raw_present if c]
            else:
                present_chars = []

            # FALLBACK 1: If characters_present is empty, scan scene text for character names
            if not present_chars:
                search_text = f"{scene.get('heading', '')} {scene.get('narration', '')} {scene.get('character_actions', '')} {base_prompt}".lower()
                for cname, cdata in characters_map.items():
                    if cname in search_text:
                        present_chars.append(cdata.get("name", cname))

            # FALLBACK 2: If still empty, default to the story protagonist
            if not present_chars and raw_characters:
                first_c = raw_characters[0]
                if isinstance(first_c, dict):
                    present_chars.append(first_c.get("name", "Protagonist"))
                elif isinstance(first_c, str):
                    present_chars.append(first_c)

            # Persist normalized characters_present onto scene dict
            scene["characters_present"] = present_chars

            # Collect visual descriptions and hero summaries for present characters
            hero_char_summaries = []
            char_detail_blocks = []
            for char_name in present_chars:
                clean_name = char_name.strip().lower()
                matched = characters_map.get(clean_name)
                if not matched:
                    for k, v in characters_map.items():
                        if k in clean_name or clean_name in k:
                            matched = v
                            break
                if matched and isinstance(matched, dict):
                    c_app = matched.get("appearance", "cartoon character")
                    c_role = matched.get("role", "character")
                    hero_char_summaries.append(f"{matched.get('name', char_name)} ({c_app})")
                    char_detail_blocks.append(f"{matched.get('name', char_name)} ({c_role}): {c_app}")
                else:
                    hero_char_summaries.append(char_name)
                    char_detail_blocks.append(f"{char_name} (character)")

            char_hero_text = " and ".join(hero_char_summaries) if hero_char_summaries else "the story characters"
            action = scene.get("character_actions") or base_prompt or "exploring the scene"
            env = scene.get("environment") or scene.get("setting", "storybook meadow")
            heading = scene.get("heading") or f"Scene {scene_num}"
            tone = scene.get("emotional_tone") or "cheerful and adventurous"
            lighting = scene.get("lighting") or "bright whimsical daylight"

            # Build Character-Primacy Enriched Image Prompt
            enriched_parts = [
                # 1. HERO OPENING: Characters and action FIRST
                f"{art_style} children's picture-book illustration of {char_hero_text} {action}.",
                # 2. PRIMARY CHARACTERS
                f"PRIMARY CHARACTERS: {'; '.join(char_detail_blocks)}.",
                # 3. SCENE ACTION & POSE
                f"SCENE ACTION & POSE: Characters actively engaged in: {action}. Facial expressions: {tone}.",
                # 4. ENVIRONMENT & SETTING
                f"ENVIRONMENT & SETTING: Background environment is {env}. Scene title context: {heading}. Lighting: {lighting}.",
            ]

            if scene.get("important_objects"):
                objs = scene["important_objects"]
                if isinstance(objs, list) and objs:
                    enriched_parts.append(f"KEY OBJECTS: {', '.join(str(o) for o in objs)}.")

            enriched_parts.extend([
                # 5. COMPOSITION & FRAMING
                "COMPOSITION: Characters are large, centered, and prominent in the foreground and midground with clear focal point. Storybook picture-book framing, not an empty landscape.",
                # 6. NEGATIVE PROMPT / EXCLUSIONS
                "EXCLUSIONS: No empty landscape without characters, no missing characters, no incorrect animal anatomy, no lizard legs on snake, no dinosaur, no text, no captions, no watermarks, no logos, no artist signatures, no borders."
            ])

            # Store the enriched prompt for the image generator
            scene["enriched_image_prompt"] = " ".join(enriched_parts)

        return story_dict
