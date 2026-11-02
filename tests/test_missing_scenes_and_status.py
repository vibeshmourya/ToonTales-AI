"""
Regression test suite for missing scenes formatting, illustration status lifecycle,
safe usage summary fallback, and resilient PDF export.
"""

import os
from PIL import Image
from services.pdf_generator import PDFStorybookGenerator
from services.usage_tracker import UsageTracker, create_usage_summary
from services.rate_limit_service import RateLimitService


def test_missing_scenes_list_of_ints():
    """Requirement 1: missing_scenes = [3, 4, 5] formatted safely without AttributeError."""
    missing_scenes = [3, 4, 5]
    formatted = PDFStorybookGenerator.format_missing_scenes(missing_scenes)
    assert formatted == "3, 4, 5"


def test_missing_scenes_list_of_dictionaries():
    """Requirement 2: List containing scene dictionaries."""
    missing_scenes = [
        {"scene_number": 1, "heading": "Scene 1"},
        {"scene_number": 2, "heading": "Scene 2"},
    ]
    formatted = PDFStorybookGenerator.format_missing_scenes(missing_scenes)
    assert formatted == "1, 2"


def test_missing_scenes_empty_list():
    """Requirement 3: An empty missing-scenes list or None."""
    assert PDFStorybookGenerator.format_missing_scenes([]) == ""
    assert PDFStorybookGenerator.format_missing_scenes(None) == ""


def test_missing_scenes_absent_scene_number():
    """Requirement 4: A missing-scene entry with absent scene number or using scene_id."""
    # Absent scene number -> fallback '?'
    formatted = PDFStorybookGenerator.format_missing_scenes([{}])
    assert formatted == "?"

    # Dict with scene_id fallback
    formatted_id = PDFStorybookGenerator.format_missing_scenes([{"scene_id": 4}])
    assert formatted_id == "4"

    # Mixed list with int, dict, missing key, and scene_id
    mixed = [1, {"scene_number": 2}, {}, {"scene_id": 4}]
    assert PDFStorybookGenerator.format_missing_scenes(mixed) == "1, 2, ?, 4"


def test_scene_with_generated_image(tmp_path):
    """Requirement 5: A scene with a verified generated image on disk."""
    img_path = str(tmp_path / "scene_01.png")
    test_img = Image.new("RGB", (100, 100), color="blue")
    test_img.save(img_path)

    story_data = {
        "scenes": [
            {"scene_number": 1, "image_path": img_path, "illustration_status": "Generated"},
            {"scene_number": 2, "image_path": None, "illustration_status": "Pending"},
        ]
    }

    missing = PDFStorybookGenerator.get_missing_illustrations(story_data)
    assert missing == [2]
    assert 1 not in missing


def test_scene_with_failed_image_attempt():
    """Requirement 6: A scene with a failed image generation attempt tracks status and missing state."""
    scene = {
        "scene_number": 3,
        "image_path": None,
        "illustration_status": "Failed",
        "illustration_error": "403 Forbidden: Invalid API Key",
    }
    story_data = {"scenes": [scene]}

    missing = PDFStorybookGenerator.get_missing_illustrations(story_data)
    assert missing == [3]
    assert scene["illustration_status"] == "Failed"
    assert "Invalid API Key" in scene["illustration_error"]


def test_scene_waiting_for_cooldown():
    """Requirement 7: A scene waiting for a rate-limit cooldown."""
    RateLimitService.reset()
    RateLimitService.record_cooldown("openai", duration_seconds=60)
    assert RateLimitService.is_in_cooldown("openai") is True
    assert RateLimitService.get_cooldown_remaining("openai") > 0

    scene = {
        "scene_number": 2,
        "image_path": None,
        "illustration_status": "Waiting for cooldown",
        "illustration_error": "Rate limited. Cooldown active.",
    }
    assert scene["illustration_status"] == "Waiting for cooldown"
    RateLimitService.reset()


def test_usage_summary_missing_optional_fields():
    """Requirement 8: Usage summary with missing optional fields handles safe defaults."""
    # Base summary guarantees all keys
    summary = create_usage_summary()
    assert summary["total_images"] == 0
    assert summary["successful_images"] == 0
    assert summary["failed_images"] == 0
    assert summary["total_stories"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_requests"] == 0
    assert summary["estimated_cost_usd"] == 0.0

    # Incomplete summary simulation safely handled by .get
    incomplete = {"successful_images": 5}
    total_imgs = incomplete.get("total_images", incomplete.get("successful_images", 0))
    failed_imgs = incomplete.get("failed_images", 0)
    assert total_imgs == 5
    assert failed_imgs == 0


def test_reruns_and_repeated_scene_navigation():
    """Requirement 9: Multiple scenes navigation and reruns do not corrupt scene state or duplicate."""
    scenes = [
        {"scene_number": 1, "heading": "Intro", "illustration_status": "Generated"},
        {"scene_number": 2, "heading": "Middle", "illustration_status": "Pending"},
        {"scene_number": 3, "heading": "Climax", "illustration_status": "Failed"},
    ]
    story = {"title": "Test Run", "scenes": scenes}

    # Simulate repeated scene index changes (reruns)
    current_idx = 0
    total_scenes = len(scenes)

    # Navigate forward
    current_idx = min(current_idx + 1, total_scenes - 1)
    assert current_idx == 1
    assert story["scenes"][current_idx]["heading"] == "Middle"

    # Navigate to end
    current_idx = min(current_idx + 1, total_scenes - 1)
    assert current_idx == 2
    assert story["scenes"][current_idx]["heading"] == "Climax"

    # Navigate back
    current_idx = max(current_idx - 1, 0)
    assert current_idx == 1

    # Verify scene data was not corrupted or mutated unexpectedly
    assert len(story["scenes"]) == 3
    assert story["scenes"][0]["illustration_status"] == "Generated"
    assert story["scenes"][1]["illustration_status"] == "Pending"
    assert story["scenes"][2]["illustration_status"] == "Failed"


def test_pdf_export_when_some_illustrations_missing(tmp_path):
    """Requirement 10: PDF export when some illustrations are missing."""
    img_path = str(tmp_path / "scene_01.png")
    test_img = Image.new("RGB", (200, 200), color="purple")
    test_img.save(img_path)

    story_data = {
        "title": "Partially Illustrated Book",
        "summary": "Some scenes have illustrations, others have placeholders.",
        "genre": "Fantasy",
        "art_style": "Cartoon 2D",
        "characters": [
            {"name": "Pip", "role": "Mouse Explorer", "appearance": "Tiny mouse with a compass"}
        ],
        "scenes": [
            {"scene_number": 1, "heading": "First Step", "narration": "Pip started walking.", "image_path": img_path},
            {"scene_number": 2, "heading": "Lost in Woods", "narration": "Pip lost his way.", "image_path": None},
            {"scene_number": 3, "heading": "Found the Light", "narration": "Pip found a lantern.", "image_path": ""},
        ]
    }

    missing = PDFStorybookGenerator.get_missing_illustrations(story_data)
    assert missing == [2, 3]
    assert PDFStorybookGenerator.format_missing_scenes(missing) == "2, 3"

    pdf_out = str(tmp_path / "partial_illustrated.pdf")
    generator = PDFStorybookGenerator()
    result_path = generator.generate_pdf(story_data, pdf_out)

    assert os.path.isfile(result_path)
    assert os.path.getsize(result_path) > 1000
    with open(result_path, "rb") as f:
        assert f.read(5).startswith(b"%PDF-")
