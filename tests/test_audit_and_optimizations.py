"""
ToonTales AI — System Audit, Optimization & Reliability Regression Test Suite
Validates:
1. Local JSON repair under diverse malformed LLM responses.
2. Request deduplication and caching (Phase 2).
3. Structured scene & character model normalization with aliases (Phase 3).
4. Character visual consistency injection and scene prompt overhaul (Phase 4).
5. Watermark-free parameter enforcement and prompt negative keywords (Phase 6).
6. Resilience against previous fatal errors (KeyError: 'total_images', AttributeError: 'int' object has no attribute 'get').
7. Corrupted image handling in PDF generation.
8. Usage tracker resilience with non-numeric/missing fields and cache hits.
"""

import io
import os
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.json_repair import parse_and_repair_json, repair_json_string
from services.cache_service import RequestCache
from services.story_generator import (
    StoryModel,
    CharacterModel,
    SceneModel,
    StoryGeneratorService
)
from services.image_generator import ImageGeneratorService
from services.pdf_generator import PDFStorybookGenerator
from services.usage_tracker import UsageTracker, create_usage_summary, safe_int, safe_float
from services.rate_limit_service import RateLimitService
from services.providers.base import GenerationUsage


def _create_fake_png_bytes(color: str = "blue", size: tuple = (256, 256)) -> bytes:
    img = Image.new("RGB", size, color=color)
    img.putpixel((0, 0), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =====================================================================
# 1. LOCAL JSON REPAIR TESTS
# =====================================================================

def test_json_repair_with_markdown_fences():
    raw = '```json\n{"title": "Test Story", "summary": "A cute story."}\n```'
    parsed = parse_and_repair_json(raw)
    assert parsed["title"] == "Test Story"
    assert parsed["summary"] == "A cute story."


def test_json_repair_with_trailing_commas():
    raw = '{"title": "Story with Commas", "scenes": [{"scene_number": 1, }, ], }'
    parsed = parse_and_repair_json(raw)
    assert parsed["title"] == "Story with Commas"
    assert len(parsed["scenes"]) == 1
    assert parsed["scenes"][0]["scene_number"] == 1


def test_json_repair_with_smart_quotes_and_control_chars():
    raw = '{\x00"title": \x08“Smart Quoted Story”\x1f, "summary": "Text with \x02 control chars"}'
    parsed = parse_and_repair_json(raw)
    assert "Smart Quoted Story" in parsed["title"]


def test_json_repair_extra_text_around_json():
    raw = 'Here is your storybook JSON object:\n{"title": "Surrounded Story", "summary": "Surrounded"}\nHope you like it!'
    parsed = parse_and_repair_json(raw)
    assert parsed["title"] == "Surrounded Story"


def test_json_repair_empty_fails_gracefully():
    with pytest.raises(ValueError) as exc:
        parse_and_repair_json("")
    assert "empty" in str(exc.value).lower()


# =====================================================================
# 2. REQUEST DEDUPLICATION & CACHING TESTS
# =====================================================================

def test_request_cache_story_deduplication():
    RequestCache.reset()
    key = RequestCache.compute_story_key(
        story_idea="A space dog",
        genre="Adventure",
        target_age="6-8 years",
        story_length="Medium",
        art_style="Colorful 2D cartoon",
        num_scenes=5,
        ending_style="Happy",
        provider="gemini",
        model="gemini-2.5-flash"
    )

    # Initial check: empty
    assert RequestCache.get_cached_story(key) is None

    # Store story
    fake_story = {"title": "Astro Pup", "scenes": [{"scene_number": 1, "heading": "Launch"}]}
    RequestCache.cache_story(key, fake_story)

    # Second check: retrieved from cache
    cached = RequestCache.get_cached_story(key)
    assert cached is not None
    assert cached["title"] == "Astro Pup"

    stats = RequestCache.get_cache_stats()
    assert stats["cached_stories"] == 1
    assert stats["story_hits"] == 1
    assert stats["total_hits"] == 1
    RequestCache.reset()


def test_story_generator_uses_cache():
    RequestCache.reset()
    service = StoryGeneratorService(provider="gemini", api_key="fake_key")

    fake_story_dict = {
        "title": "Cached Bunny",
        "summary": "Pip visits space.",
        "genre": "Adventure",
        "art_style": "Colorful 2D cartoon",
        "characters": [{"name": "Pip", "role": "Hero", "appearance": "Blue bunny", "personality": "Brave"}],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Start",
                "narration": "Pip starts.",
                "image_prompt": "Pip jumping.",
                "environment": "Meadow",
                "characters_present": ["Pip"]
            }
        ]
    }

    cache_key = RequestCache.compute_story_key(
        story_idea="Pip the bunny",
        genre="Adventure",
        target_age="6–8 years",
        story_length="Medium",
        art_style="Colorful 2D cartoon",
        num_scenes=1,
        ending_style="Happy",
        provider="gemini",
        model="gemini-2.5-flash"
    )
    RequestCache.cache_story(cache_key, fake_story_dict)

    # Calling generate_story should return the cached story WITHOUT calling provider
    with patch.object(service, "_get_provider") as mock_prov:
        result = service.generate_story(
            story_idea="Pip the bunny",
            genre="Adventure",
            target_age="6–8 years",
            story_length="Medium",
            art_style="Colorful 2D cartoon",
            num_scenes=1,
            ending_style="Happy",
            model_override="gemini-2.5-flash"
        )
        mock_prov.assert_not_called()
        assert result["title"] == "Cached Bunny"

    RequestCache.reset()


# =====================================================================
# 3. STRUCTURED SCENE & CHARACTER MODEL NORMALIZATION
# =====================================================================

def test_scene_model_alias_synchronization():
    # Scene with scene_title instead of heading, narrative_text instead of narration, setting instead of environment
    data = {
        "scene_number": 2,
        "scene_title": "The Crystal Cave",
        "narrative_text": "They stepped inside the glowing cavern.",
        "illustration_prompt": "Characters surrounded by amethyst.",
        "setting": "Luminescent cave",
        "character_actions": "Walking cautiously",
        "emotional_tone": "Awe and wonder",
        "lighting": "Soft purple glow"
    }
    model = SceneModel(**data)
    assert model.heading == "The Crystal Cave"
    assert model.scene_title == "The Crystal Cave"
    assert model.narration == "They stepped inside the glowing cavern."
    assert model.narrative_text == "They stepped inside the glowing cavern."
    assert model.image_prompt == "Characters surrounded by amethyst."
    assert model.illustration_prompt == "Characters surrounded by amethyst."
    assert model.environment == "Luminescent cave"
    assert model.setting == "Luminescent cave"
    assert model.scene_id == "scene_02"
    assert model.illustration_status == "Pending"


def test_character_model_string_fallback():
    # If a character is provided as a plain string
    model = CharacterModel.model_validate("Sparky")
    assert model.name == "Sparky"
    assert "Sparky" in model.appearance
    assert model.role == "Character"


# =====================================================================
# 4. ERROR REGRESSION: ATTRIBUTEERROR 'INT' OBJECT HAS NO ATTRIBUTE 'GET'
# =====================================================================

def test_enrich_character_consistency_with_non_dict_characters():
    """Ensure enrich_character_consistency does not crash if characters are ints or strings."""
    story = {
        "art_style": "Colorful 2D cartoon",
        "characters": [1, 2, "Barnaby", {"name": "Ruby", "appearance": "white rabbit in purple suit"}],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Intro",
                "narration": "Ruby waved.",
                "image_prompt": "Ruby waving happily.",
                "characters_present": ["Ruby", "Barnaby", "1"],
                "environment": "Green meadow"
            }
        ]
    }
    enriched = StoryGeneratorService.enrich_character_consistency(story)
    scene_1 = enriched["scenes"][0]
    assert "enriched_image_prompt" in scene_1
    assert "white rabbit in purple suit" in scene_1["enriched_image_prompt"]
    assert "Ruby" in scene_1["enriched_image_prompt"]


def test_construct_scene_prompt_with_non_dict_characters():
    """Ensure construct_scene_prompt does not crash if characters list contains ints."""
    service = ImageGeneratorService(provider="pollinations")
    scene = {
        "scene_number": 1,
        "heading": "The Meadow",
        "narration": "Ruby and Barnaby played.",
        "image_prompt": "Ruby running beside Barnaby.",
        "characters_present": ["Ruby", "1"],
        "environment": "Sunny meadow",
        "illustration_status": "Pending"
    }
    characters = [1, 2, {"name": "Ruby", "appearance": "small white rabbit in purple suit"}]
    prompt = service.construct_scene_prompt(scene, characters, art_style="Colorful 2D cartoon")
    assert "white rabbit in purple suit" in prompt
    assert "no watermark" in prompt.lower()


def test_pdf_generator_with_non_dict_characters_and_scenes(tmp_path):
    """Ensure PDF generator does not crash when characters or scenes contain non-dict elements."""
    story_data = {
        "title": "Resilient Tale",
        "summary": "Testing resilience against corrupt scene structures.",
        "genre": "Fantasy",
        "art_style": "Colorful 2D cartoon",
        "characters": [1, "Sparky", {"name": "Pip", "role": "Hero", "appearance": "Tiny mouse", "personality": "Brave"}],
        "scenes": [
            1,  # int scene
            "Plain text scene description",  # string scene
            {
                "scene_number": 3,
                "heading": "Normal Scene",
                "narration": "All is well.",
                "image_path": None,
                "environment": "Cozy room"
            }
        ]
    }

    pdf_out = str(tmp_path / "resilient.pdf")
    gen = PDFStorybookGenerator()
    result = gen.generate_pdf(story_data, pdf_out)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 500


def test_pdf_generator_empty_output_path_fails():
    gen = PDFStorybookGenerator()
    with pytest.raises(ValueError):
        gen.generate_pdf({"title": "Test"}, "")


def test_pdf_generator_corrupted_image_file_handling(tmp_path):
    """Ensure corrupted, unparseable, or empty image files on disk do not crash PDF export."""
    corrupt_file = str(tmp_path / "corrupt.png")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT A REAL IMAGE DATA JUST CORRUPT TEXT")

    story_data = {
        "title": "Corrupt Image Tale",
        "summary": "Testing graceful placeholder insertion.",
        "genre": "Fantasy",
        "art_style": "Colorful 2D cartoon",
        "characters": [],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Scene with Corrupted Image",
                "narration": "Narration text here.",
                "image_path": corrupt_file,
                "environment": "Enchanted woods"
            }
        ]
    }

    pdf_out = str(tmp_path / "corrupt_img_test.pdf")
    gen = PDFStorybookGenerator()
    result = gen.generate_pdf(story_data, pdf_out)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 500


# =====================================================================
# 5. WATERMARK SUPPRESSION & PROMPT NEGATIVE KEYWORDS
# =====================================================================

def test_pollinations_provider_includes_nologo_and_nofeed():
    """Verify Pollinations provider passes &nologo=true and &nofeed=true on both endpoints."""
    from services.providers.pollinations_provider import PollinationsProvider
    prov = PollinationsProvider(api_key="mock_key")
    fake_png = _create_fake_png_bytes("teal")

    with patch("services.providers.pollinations_provider.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_png
        mock_resp.headers = {"content-type": "image/png"}
        mock_get.return_value = mock_resp

        prov.generate_image("A cute kitten in a box", model="flux")

        called_url = mock_get.call_args[0][0]
        assert "nologo=true" in called_url
        assert "nofeed=true" in called_url


# =====================================================================
# 6. USAGE TRACKER RESILIENCE & KEYERROR 'TOTAL_IMAGES' REGRESSION
# =====================================================================

def test_usage_tracker_resilience_to_malformed_inputs():
    UsageTracker.reset()
    # Record non-numeric tokens, non-numeric images, non-numeric cost
    UsageTracker.record(
        provider="gemini",
        model="gemini-2.5-flash",
        operation="story",
        status="success",
        tokens="not_a_number",  # type: ignore
        images_count="invalid",  # type: ignore
        estimated_cost="free"    # type: ignore
    )

    summary = UsageTracker.get_summary()
    assert summary["total_requests"] == 1
    assert summary["total_tokens"] == 0
    assert summary["total_images"] == 0
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["successful_stories"] == 1
    assert "total_images" in summary
    assert "successful_images" in summary
    assert "failed_images" in summary

    UsageTracker.reset()


def test_safe_conversions():
    assert safe_int("42") == 42
    assert safe_int("invalid", default=5) == 5
    assert safe_int(None, default=0) == 0
    assert safe_int(True, default=0) == 0

    assert safe_float("3.14") == 3.14
    assert safe_float("invalid", default=1.0) == 1.0
    assert safe_float(None, default=0.0) == 0.0

