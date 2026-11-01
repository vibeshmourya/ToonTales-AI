"""
ToonTales AI - Story Generation Service
Powered by Google Gemini API via official google-genai SDK.
Produces structured, consistent cartoon stories with divided scenes and character profiles.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai.errors import APIError

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
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the story generator with API credentials and model configuration.
        """
        self.api_key = api_key or get_config("GEMINI_API_KEY")
        self.model_name = model_name or get_config("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
        
        if not self.api_key:
            logger.warning("StoryGenerator initialized without API Key. API calls will require explicit key.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"StoryGenerator initialized with model '{self.model_name}' (Key: {mask_secret(self.api_key)})")

    def _get_client(self, override_key: Optional[str] = None) -> genai.Client:
        key = override_key or self.api_key
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please configure GEMINI_API_KEY in .env or enter it in the app settings."
            )
        return genai.Client(api_key=key)

    def generate_story(
        self,
        story_idea: str,
        genre: str = "Adventure",
        target_age: str = "6–8 years",
        story_length: str = "Medium",
        art_style: str = "Colorful 2D cartoon",
        num_scenes: int = 5,
        ending_style: str = "Happy",
        api_key_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete illustrated cartoon story using Google Gemini API.
        Enforces character visual consistency and strict structured JSON schema.
        """
        if not story_idea or not story_idea.strip():
            raise ValueError("Story idea cannot be empty. Please provide an imaginative concept.")

        client = self._get_client(api_key_override)
        
        logger.info(
            f"Generating story: '{story_idea[:40]}...' | Genre: {genre} | Age: {target_age} | "
            f"Scenes: {num_scenes} | Art Style: {art_style}"
        )

        # Craft system instructions for child-friendly storytelling and character consistency
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
            # We configure Gemini using the official google-genai SDK
            response = client.models.generate_content(
                model=self.model_name,
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
                raise ValueError("Received empty response from Gemini API.")

            # Parse and validate with Pydantic
            try:
                story_model = StoryModel.model_validate_json(raw_text)
            except Exception as parse_err:
                logger.warning(f"Standard JSON validation error: {parse_err}. Attempting sanitization...")
                cleaned = self._clean_json_markdown(raw_text)
                story_model = StoryModel.model_validate_json(cleaned)

            story_dict = story_model.model_dump()
            
            # Post-process: ensure character consistency references are embedded into all image prompts
            story_dict = self.enrich_character_consistency(story_dict)

            # Validate scene count
            if len(story_dict.get("scenes", [])) == 0:
                raise ValueError("Generated story does not contain any scenes.")

            logger.info(f"Story '{story_dict.get('title')}' successfully generated with {len(story_dict['scenes'])} scenes.")
            return story_dict

        except APIError as api_err:
            logger.error(f"Gemini API Error: {api_err}")
            raise RuntimeError(f"Google Gemini API error: {api_err.message if hasattr(api_err, 'message') else str(api_err)}")
        except Exception as e:
            logger.error(f"Story generation failure: {e}", exc_info=True)
            raise

    @staticmethod
    def _clean_json_markdown(raw_str: str) -> str:
        """
        Strip markdown formatting (e.g. ```json ... ```) if present.
        """
        text = raw_str.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

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
                    # Try partial match
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
