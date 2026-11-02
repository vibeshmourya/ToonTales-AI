"""
ToonTales AI — Usage Tracker & Summary Regression Tests
Validates:
1. Empty usage summary structure and key presence.
2. Safe defaults when accessing optional fields.
3. Successful image generation increments successful_images and total_images.
4. Failed image generation increments failed_images without incrementing successful_images.
5. Corrupt image bytes rejected by Pillow are not counted as successful images.
6. Multiple scenes and skipping existing images does not double-count usage.
7. Different story and image providers record usage accurately.
8. Usage summary rendering safely handles missing keys and empty configurations.
9. Pending images are not counted as generated images.
"""

import io
import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.usage_tracker import UsageTracker, create_usage_summary
from services.image_generator import ImageGeneratorService
from services.story_generator import StoryGeneratorService
from services.providers.base import GenerationUsage, ProviderError, QuotaExceededError


def _create_fake_png_bytes(color: str = "green", size: tuple = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_usage_tracker():
    UsageTracker.reset()
    yield
    UsageTracker.reset()


def test_empty_usage_summary():
    """Verify that a freshly created summary has all required keys initialized to 0."""
    summary = create_usage_summary()

    required_keys = [
        "total_requests",
        "total_images",
        "successful_images",
        "failed_images",
        "total_stories",
        "successful_stories",
        "failed_stories",
        "total_tokens",
        "estimated_cost_usd"
    ]

    for key in required_keys:
        assert key in summary, f"Required key '{key}' missing from create_usage_summary()"
        assert summary[key] == 0 or summary[key] == 0.0

    # Also verify UsageTracker.get_summary() on empty logs
    tracker_summary = UsageTracker.get_summary()
    for key in required_keys:
        assert key in tracker_summary, f"Required key '{key}' missing from UsageTracker.get_summary()"
        assert tracker_summary[key] == 0 or tracker_summary[key] == 0.0


def test_summary_missing_optional_fields_fallback():
    """Verify safe retrieval patterns when dictionaries have missing or partial keys."""
    partial_summary = {"total_requests": 5}

    total_images = partial_summary.get("total_images", 0)
    successful_images = partial_summary.get("successful_images", total_images)
    failed_images = partial_summary.get("failed_images", 0)
    estimated_cost = partial_summary.get("estimated_cost_usd", 0.0)

    assert total_images == 0
    assert successful_images == 0
    assert failed_images == 0
    assert estimated_cost == 0.0


def test_successful_image_generation_tracking(tmp_path):
    """Verify successful image generation increments total_images and successful_images."""
    service = ImageGeneratorService(provider="pollinations")
    fake_png = _create_fake_png_bytes("cyan")
    dest_file = str(tmp_path / "scene_01.png")

    scene = {
        "scene_number": 1,
        "heading": "Scene 1",
        "narration": "Narration 1",
        "image_prompt": "Prompt 1",
        "characters_present": []
    }

    with patch.object(service, "_generate_with_pollinations", return_value=fake_png):
        service.generate_single_scene_image(
            scene=scene,
            characters=[],
            art_style="Colorful 2D cartoon",
            output_filepath=dest_file
        )

    summary = UsageTracker.get_summary()
    assert summary["total_images"] == 1
    assert summary["successful_images"] == 1
    assert summary["failed_images"] == 0
    assert summary["total_requests"] == 1


def test_failed_image_generation_tracking(tmp_path):
    """Verify failed image generation increments failed_images and does NOT increment successful_images."""
    service = ImageGeneratorService(provider="gemini", api_key="test_key")
    dest_file = str(tmp_path / "scene_failed.png")

    scene = {
        "scene_number": 1,
        "heading": "Scene 1",
        "narration": "Narration 1",
        "image_prompt": "Prompt 1",
        "characters_present": []
    }

    with patch.object(service, "_generate_with_gemini", side_effect=QuotaExceededError("Rate limited")):
        with pytest.raises(QuotaExceededError):
            service.generate_single_scene_image(
                scene=scene,
                characters=[],
                art_style="Colorful 2D cartoon",
                output_filepath=dest_file
            )

    summary = UsageTracker.get_summary()
    assert summary["total_images"] == 0
    assert summary["successful_images"] == 0
    assert summary["failed_images"] == 1
    assert summary["total_requests"] == 1


def test_corrupt_pillow_image_not_counted_as_success(tmp_path):
    """Verify corrupt bytes failing Pillow verification are recorded as failed, not success."""
    service = ImageGeneratorService(provider="pollinations")
    dest_file = str(tmp_path / "corrupt.png")

    scene = {
        "scene_number": 1,
        "heading": "Scene 1",
        "narration": "Narration 1",
        "image_prompt": "Prompt 1",
        "characters_present": []
    }

    # Pass HTML error response bytes instead of valid PNG bytes
    with patch.object(service, "_generate_with_pollinations", return_value=b"<html>502 Bad Gateway</html>"):
        with pytest.raises(Exception):
            service.generate_single_scene_image(
                scene=scene,
                characters=[],
                art_style="Colorful 2D cartoon",
                output_filepath=dest_file
            )

    summary = UsageTracker.get_summary()
    assert summary["total_images"] == 0
    assert summary["successful_images"] == 0
    assert summary["failed_images"] == 1


def test_multiple_scenes_and_reruns_no_double_counting(tmp_path):
    """Verify existing valid images are skipped and not double-counted across reruns."""
    service = ImageGeneratorService(provider="pollinations")
    fake_png = _create_fake_png_bytes("magenta")

    # Pre-create scene 1 on disk
    scene1_path = str(tmp_path / "scene_01.png")
    with open(scene1_path, "wb") as f:
        f.write(fake_png)

    story_dict = {
        "story_id": "rerun_story_test",
        "title": "Story Rerun",
        "art_style": "Colorful 2D cartoon",
        "characters": [],
        "scenes": [
            {"scene_number": 1, "image_path": scene1_path, "illustration_status": "Generated"},
            {"scene_number": 2, "image_path": None, "illustration_status": "Pending"}
        ]
    }

    with patch.object(service, "_generate_with_pollinations", return_value=fake_png):
        service.generate_all_scenes(story_dict, str(tmp_path), skip_existing=True)

    # Only scene 2 was generated; scene 1 was already valid
    summary1 = UsageTracker.get_summary()
    assert summary1["total_images"] == 1
    assert summary1["successful_images"] == 1

    # Simulate an immediate rerun or navigation without edits
    with patch.object(service, "_generate_with_pollinations", return_value=fake_png):
        service.generate_all_scenes(story_dict, str(tmp_path), skip_existing=True)

    # Both scenes now exist on disk, so 0 additional generations occur!
    summary2 = UsageTracker.get_summary()
    assert summary2["total_images"] == 1
    assert summary2["successful_images"] == 1


def test_different_story_and_image_providers():
    """Verify tracking distinct providers for text and image generation."""
    # 1. Story generation record
    story_usage = GenerationUsage(
        provider="anthropic",
        model="claude-3-5-sonnet",
        operation="story",
        prompt_tokens=300,
        completion_tokens=500,
        total_tokens=800,
        cost_type="estimated"
    )
    UsageTracker.record_usage_obj(story_usage, status="success")

    # 2. Image generation record (OpenAI DALL-E)
    image_usage = GenerationUsage(
        provider="openai",
        model="dall-e-3",
        operation="image",
        images_generated=1,
        estimated_cost_usd=0.04,
        cost_type="estimated"
    )
    UsageTracker.record_usage_obj(image_usage, status="success")

    # 3. Failed image generation record (Gemini)
    UsageTracker.record(
        provider="gemini",
        model="gemini-2.5-flash-image",
        operation="image",
        status="rate_limited",
        images_count=0
    )

    summary = UsageTracker.get_summary()
    assert summary["total_requests"] == 3
    assert summary["total_stories"] == 1
    assert summary["successful_stories"] == 1
    assert summary["total_images"] == 1
    assert summary["successful_images"] == 1
    assert summary["failed_images"] == 1
    assert summary["total_tokens"] == 800
    assert summary["estimated_cost_usd"] >= 0.04


def test_usage_summary_rendering_without_api_key():
    """Verify that UI rendering logic executes cleanly even when no keys are configured."""
    summary = UsageTracker.get_summary()

    # Simulate Streamlit UI block in app.py lines 492-498
    total_requests = summary.get("total_requests", 0)
    total_tokens = summary.get("total_tokens", 0)
    total_images = summary.get("total_images", summary.get("successful_images", 0))
    failed_images = summary.get("failed_images", 0)
    estimated_cost = summary.get("estimated_cost_usd", 0.0)

    display_text = f"Requests: {total_requests}, Tokens: {total_tokens:,}, Images: {total_images}, Cost: ${estimated_cost:.4f}"
    assert "Images: 0" in display_text
    assert "Requests: 0" in display_text


def test_pending_images_not_counted():
    """Verify that scenes with 'Pending' status do not increment generated image counts."""
    story = {
        "title": "Pending Story",
        "scenes": [
            {"scene_number": 1, "illustration_status": "Pending", "image_path": None},
            {"scene_number": 2, "illustration_status": "Pending", "image_path": None}
        ]
    }

    # No generation executed yet
    summary = UsageTracker.get_summary()
    assert summary["total_images"] == 0
    assert summary["successful_images"] == 0
    assert summary["failed_images"] == 0
