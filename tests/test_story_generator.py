"""
Unit tests for Story Generator Service.
Validates Pydantic schemas, character consistency injection, and mocked API responses.
"""

import pytest
from unittest.mock import MagicMock, patch
from services.story_generator import (
    StoryGeneratorService,
    StoryModel,
    CharacterModel,
    SceneModel
)


@pytest.fixture
def sample_story_dict():
    return {
        "title": "Ruby's Lunar Adventure",
        "summary": "A brave rabbit travels to the moon in her purple astronaut suit.",
        "genre": "Adventure",
        "art_style": "Colorful 2D cartoon",
        "characters": [
            {
                "name": "Ruby",
                "role": "Protagonist",
                "appearance": "Small white rabbit with long ears, purple astronaut suit, and silver helmet.",
                "personality": "Inquisitive, daring, and friendly."
            }
        ],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "The Backyard Telescope",
                "narration": "Ruby peered through her telescope, marveling at the bright silver moon.",
                "image_prompt": "Ruby gazes up at the sparkling night sky beside a wooden telescope.",
                "characters_present": ["Ruby"],
                "environment": "Grassy backyard under a starlit indigo sky"
            },
            {
                "scene_number": 2,
                "heading": "Touchdown on the Moon",
                "narration": "With a gentle bounce, Ruby took her very first steps on the dusty lunar surface.",
                "image_prompt": "Ruby floating gently above the moon crater, smiling joyfully.",
                "characters_present": ["Ruby"],
                "environment": "Luminescent lunar crater with Earth glowing in the background"
            },
            {
                "scene_number": 3,
                "heading": "Return with Stardust",
                "narration": "Ruby returned home with a jar of glittering moonbeams in her paws.",
                "image_prompt": "Ruby holding a glowing jar of stardust inside her spaceship.",
                "characters_present": ["Ruby"],
                "environment": "Spaceship cabin filled with warm buttons and dials"
            }
        ]
    }


def test_story_model_validation(sample_story_dict):
    """Verify that a valid story structure matches the Pydantic StoryModel schema."""
    model = StoryModel(**sample_story_dict)
    assert model.title == "Ruby's Lunar Adventure"
    assert len(model.characters) == 1
    assert len(model.scenes) == 3
    assert model.characters[0].name == "Ruby"


def test_story_model_invalid_missing_fields():
    """Verify validation fails if required fields are missing."""
    invalid_data = {
        "title": "Incomplete Tale"
        # missing summary, characters, scenes, etc.
    }
    with pytest.raises(Exception):
        StoryModel(**invalid_data)


def test_character_consistency_enrichment(sample_story_dict):
    """Test that character visual descriptions are automatically injected into scene prompts."""
    enriched = StoryGeneratorService.enrich_character_consistency(sample_story_dict)
    scene_1 = enriched["scenes"][0]
    
    assert "enriched_image_prompt" in scene_1
    prompt = scene_1["enriched_image_prompt"]
    # Verify character appearance is present in prompt
    assert "purple astronaut suit" in prompt
    assert "Colorful 2D cartoon" in prompt
    assert "Ruby" in prompt


@patch("services.story_generator.genai.Client")
def test_story_generator_mocked_gemini(mock_client_class, sample_story_dict):
    """Test full story generation workflow with mocked Gemini API client."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    # Mock generate_content response
    mock_response = MagicMock()
    mock_response.text = StoryModel(**sample_story_dict).model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    service = StoryGeneratorService(api_key="test_fake_key")
    result = service.generate_story(
        story_idea="A rabbit astronaut",
        num_scenes=3
    )

    assert result["title"] == "Ruby's Lunar Adventure"
    assert len(result["scenes"]) == 3
    assert "enriched_image_prompt" in result["scenes"][0]
    mock_client.models.generate_content.assert_called_once()
