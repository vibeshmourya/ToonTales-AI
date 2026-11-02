"""
ToonTales AI — Multi-Provider Story Generation Service
Orchestrates text generation across Google Gemini, OpenAI, Anthropic Claude, and OpenRouter.
Enforces character consistency, Pydantic schema validation, and safe error normalization.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

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
from utils.helpers import get_config, mask_secret

logger = logging.getLogger("ToonTalesAI.StoryGenerator")



# =====================================================================
# Pydantic Schemas for Structured JSON Output
# =====================================================================

class CharacterModel(BaseModel):
    name: str = Field(..., description="Character's name")
    role: str = Field(..., description="Role in story, e.g., 'Protagonist', 'Best Friend', 'Mentor'")
    appearance: str = Field(
        ...,
        description=(
            "Exhaustive visual description for cartoon illustration: species or type, "
            "body shape, primary colors, eyes, clothing, accessories, and distinctive features."
        )
    )
    personality: str = Field(..., description="Key personality traits and behavioral quirks")


class SceneModel(BaseModel):
    scene_number: int = Field(..., description="Sequential scene index starting at 1")
    heading: str = Field(..., description="Catchy, short scene title, e.g., 'A Dream Amongst the Stars'")
    narration: str = Field(..., description="Engaging story narration for this scene (2-4 child-friendly sentences)")
    image_prompt: str = Field(
        ...,
        description=(
            "Detailed cartoon visual scene description describing character actions, "
            "emotions, environment, camera angle, and vibrant lighting."
        )
    )
    characters_present: List[str] = Field(default_factory=list, description="Names of characters visible in this scene")
    environment: str = Field(..., description="Setting/backdrop description, e.g., 'A cozy bedroom lit by starlight'")


class StoryModel(BaseModel):
    title: str = Field(..., description="Creative, whimsical, and memorable story title")
    summary: str = Field(..., description="Engaging 2-sentence story synopsis")
    genre: str = Field(..., description="Selected story genre")
    art_style: str = Field(..., description="Selected cartoon art style")
    characters: List[CharacterModel] = Field(..., min_length=1, description="List of story characters with persistent visuals")
    scenes: List[SceneModel] = Field(..., min_length=3, description="Chronological sequence of story scenes")


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
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a complete illustrated cartoon story using the configured provider.
        Enforces character visual consistency and strict structured JSON schema.
        """
        if not story_idea or not story_idea.strip():
            raise ValueError("Story idea cannot be empty. Please provide an imaginative concept.")

        provider = self._get_provider(provider_override, api_key_override)
        chosen_model = model_override or self.model_name

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
            story_dict["story_model"] = chosen_model or provider.capabilities.default_text_model

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
        Enhances scene image prompts by explicitly prefixing the consistent character
        visual descriptions for every character participating in that scene.
        """
        characters_map = {c["name"].lower(): c for c in story_dict.get("characters", [])}
        art_style = story_dict.get("art_style", "Colorful 2D cartoon")

        for scene in story_dict.get("scenes", []):
            base_prompt = scene.get("image_prompt", "")
            present_chars = scene.get("characters_present", [])
            
            # Collect visual descriptions for present characters
            char_visual_snippets = []
            for char_name in present_chars:
                clean_name = char_name.strip().lower()
                matched = characters_map.get(clean_name)
                if not matched:
                    for k, v in characters_map.items():
                        if k in clean_name or clean_name in k:
                            matched = v
                            break
                if matched:
                    char_visual_snippets.append(
                        f"{matched['name']} (Appearance: {matched['appearance']})"
                    )

            # Build enriched image prompt
            enriched_parts = [
                f"Cartoon storybook illustration in {art_style} style.",
                f"Scene setting: {scene.get('environment', '')}.",
            ]
            if char_visual_snippets:
                enriched_parts.append(
                    "Characters with persistent design: " + "; ".join(char_visual_snippets) + "."
                )
            enriched_parts.append(f"Scene action: {base_prompt}")
            enriched_parts.append(
                "High quality cartoon art, vivid expressive faces, vibrant palette, clean lines, charming storybook composition."
            )

            # Store the enriched prompt for the image generator
            scene["enriched_image_prompt"] = " ".join(enriched_parts)

        return story_dict
