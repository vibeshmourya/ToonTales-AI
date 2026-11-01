"""
Unit tests for Cartoon Image Generator Service.
Validates prompt construction, Gemini Developer API generate_content image extraction,
Pillow validation, error categorization (Quota, Auth, Model, Network), rate limits,
circuit-breaker behavior on zero-quota, and regeneration preservation.
"""

import os
import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.image_generator import (
    ImageGeneratorService,
    QuotaExceededError,
    InvalidAPIKeyError,
    UnsupportedModelError,
    ImageNetworkError,
    ImageGenerationError
)
from google.genai.errors import APIError, ClientError
from utils.helpers import validate_image_file


@pytest.fixture
def sample_character():
    return [{
        "name": "Ruby",
        "role": "Protagonist",
        "appearance": "Small white rabbit with long ears, purple astronaut suit, and silver helmet.",
        "personality": "Inquisitive"
    }]


@pytest.fixture
def sample_scene():
    return {
        "scene_number": 1,
        "heading": "The Launchpad",
        "narration": "Ruby stood beside her carrot-shaped rocket.",
        "image_prompt": "Ruby standing proudly in front of a glowing orange rocket ship.",
        "characters_present": ["Ruby"],
        "environment": "Emerald green launch meadow"
    }


def _create_fake_png_bytes(color: str = "purple", size: tuple = (100, 100)) -> bytes:
    """Helper to create valid PNG bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_prompt_construction(sample_scene, sample_character):
    """Test that construct_scene_prompt embeds the art style, character traits, and scene action."""
    service = ImageGeneratorService(provider="gemini", model_name="gemini-2.5-flash-image")
    prompt = service.construct_scene_prompt(
        scene=sample_scene,
        characters=sample_character,
        art_style="Storybook illustration"
    )

    assert "storybook illustration" in prompt.lower()
    assert "purple astronaut suit" in prompt
    assert "Ruby standing proudly" in prompt
    assert "Emerald green launch meadow" in prompt


def test_image_extraction_generate_content(sample_scene, sample_character):
    """Test extracting image bytes from generate_content response with inline_data."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")
    fake_bytes = _create_fake_png_bytes()

    mock_client = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.data = fake_bytes
    mock_img_obj = MagicMock()
    mock_img_obj.image_bytes = fake_bytes
    mock_part.as_image.return_value = mock_img_obj

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        extracted_bytes = service._generate_with_gemini("Draw a cute cartoon rabbit")
        assert extracted_bytes == fake_bytes
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash-image",
            contents=["Draw a cute cartoon rabbit"]
        )


def test_missing_image_data_raises_error(sample_scene, sample_character):
    """Test error raised when generate_content returns text but no inline_data."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")

    mock_client = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = None
    mock_part.as_image.return_value = None

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_response.text = "I cannot generate that image."
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(ImageGenerationError) as excinfo:
            service._generate_with_gemini("Draw something")
        assert "did not return an inline image" in str(excinfo.value)


def test_imagen_model_name_rejects_enterprise():
    """Test that specifying an 'imagen-' model raises UnsupportedModelError pointing to Gemini Developer API models."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="imagen-3.0-generate-002")
    with pytest.raises(UnsupportedModelError) as excinfo:
        service._generate_with_gemini("Draw something")
    assert "Gemini Enterprise Agent Platform mode" in str(excinfo.value)
    assert "gemini-2.5-flash-image" in str(excinfo.value)


def test_quota_exhausted_zero_quota(sample_scene, sample_character):
    """Test detection of 429 quota exhaustion on free tier with 0 requests/day."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")
    
    # Simulate Gemini 429 ClientError with 0 quota details
    error_payload = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Resource has been exhausted (e.g. check quota). Free-tier requests per minute: 0, Free-tier requests per day: 0"
        }
    }
    client_err = ClientError(429, error_payload)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = client_err

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(QuotaExceededError) as excinfo:
            service._generate_with_gemini("Draw dragon")
        assert excinfo.value.is_zero_quota is True
        assert "0 free-tier quota" in str(excinfo.value)
        assert "Pollinations AI" in str(excinfo.value)


def test_quota_exhausted_rate_limit_with_retry_delay(sample_scene, sample_character):
    """Test extraction of retry delay from rate limit response."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")

    error_payload = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Rate limit exceeded. Please retry after 25s."
        }
    }
    client_err = ClientError(429, error_payload)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = client_err

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(QuotaExceededError) as excinfo:
            service._generate_with_gemini("Draw rocket")
        assert excinfo.value.is_zero_quota is False
        assert excinfo.value.retry_after == 25


def test_invalid_api_key_error(sample_scene, sample_character):
    """Test InvalidAPIKeyError raised on HTTP 401/403 permission denied."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")

    error_payload = {
        "error": {
            "code": 403,
            "status": "PERMISSION_DENIED",
            "message": "API_KEY_INVALID: The provided API key is expired or invalid."
        }
    }
    client_err = ClientError(403, error_payload)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = client_err

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(InvalidAPIKeyError) as excinfo:
            service._generate_with_gemini("Draw star")
        assert "invalid, inactive, or lacks permissions" in str(excinfo.value)


def test_unsupported_model_error(sample_scene, sample_character):
    """Test UnsupportedModelError raised on HTTP 404 NOT_FOUND."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="non-existent-image-model")

    error_payload = {
        "error": {
            "code": 404,
            "status": "NOT_FOUND",
            "message": "models/non-existent-image-model is not found for API version v1beta."
        }
    }
    client_err = ClientError(404, error_payload)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = client_err

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(UnsupportedModelError) as excinfo:
            service._generate_with_gemini("Draw scene")
        assert "not found or is not accessible" in str(excinfo.value)


def test_network_failure_error(sample_scene, sample_character):
    """Test ImageNetworkError raised on connection failure."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("Connection refused by peer")

    with patch.object(service, "_get_gemini_client", return_value=mock_client):
        with pytest.raises(ImageNetworkError) as excinfo:
            service._generate_with_gemini("Draw scene")
        assert "Network error connecting to Gemini API" in str(excinfo.value)


def test_circuit_breaker_halts_on_zero_quota(tmp_path, sample_character):
    """Test that generate_all_scenes halts immediately after zero-quota error to avoid futile requests."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")

    story_dict = {
        "title": "Three Scene Story",
        "art_style": "Colorful 2D cartoon",
        "characters": sample_character,
        "scenes": [
            {"scene_number": 1, "heading": "S1", "narration": "N1", "image_prompt": "P1"},
            {"scene_number": 2, "heading": "S2", "narration": "N2", "image_prompt": "P2"},
            {"scene_number": 3, "heading": "S3", "narration": "N3", "image_prompt": "P3"},
        ]
    }

    call_count = 0
    def mock_single_image(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise QuotaExceededError("0 free-tier quota", is_zero_quota=True)

    with patch.object(service, "generate_single_scene_image", side_effect=mock_single_image):
        result = service.generate_all_scenes(story_dict, str(tmp_path))
        # Crucial: generate_single_scene_image should only be called ONCE, not 3 times!
        assert call_count == 1
        # All scenes should have image_path = None
        for sc in result["scenes"]:
            assert sc["image_path"] is None
        assert len(result["image_generation_failures"]) == 3
        assert "zero quota" in result["image_generation_failures"][1][1].lower()


def test_generate_single_scene_image_with_mock(tmp_path, sample_scene, sample_character):
    """Test single scene image generation with mock bytes and Pillow validation."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")
    fake_bytes = _create_fake_png_bytes("purple")
    out_file = str(tmp_path / "test_scene_01.png")

    with patch.object(service, "_generate_with_gemini", return_value=fake_bytes) as mock_gemini:
        res_path = service.generate_single_scene_image(
            scene=sample_scene,
            characters=sample_character,
            art_style="Colorful 2D cartoon",
            output_filepath=out_file
        )

        assert os.path.isfile(res_path)
        assert validate_image_file(res_path) is True
        mock_gemini.assert_called_once()


def test_preservation_of_previous_image_on_regeneration_failure(tmp_path, sample_scene, sample_character):
    """Test that if regeneration fails, the original image is preserved and not deleted or corrupted."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")
    out_file = str(tmp_path / "scene_01.png")

    # Save original image
    original_bytes = _create_fake_png_bytes("green")
    with open(out_file, "wb") as f:
        f.write(original_bytes)
    assert os.path.isfile(out_file)

    # Attempt regeneration that throws QuotaExceededError
    with patch.object(service, "_generate_with_gemini", side_effect=QuotaExceededError("Rate limited")):
        with pytest.raises(QuotaExceededError):
            service.generate_single_scene_image(
                scene=sample_scene,
                characters=sample_character,
                art_style="Colorful 2D cartoon",
                output_filepath=out_file
            )

    # Ensure original file is still intact and valid
    assert os.path.isfile(out_file)
    with open(out_file, "rb") as f:
        assert f.read() == original_bytes


def test_scene_to_image_association(tmp_path, sample_character):
    """Test generate_all_scenes correctly associates image paths with each scene."""
    service = ImageGeneratorService(provider="gemini", api_key="fake_key", model_name="gemini-2.5-flash-image")
    fake_bytes = _create_fake_png_bytes("blue")

    story_dict = {
        "title": "Test Story",
        "art_style": "Colorful 2D cartoon",
        "characters": sample_character,
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Scene 1",
                "narration": "Narration 1",
                "image_prompt": "Prompt 1",
                "characters_present": ["Ruby"]
            },
            {
                "scene_number": 2,
                "heading": "Scene 2",
                "narration": "Narration 2",
                "image_prompt": "Prompt 2",
                "characters_present": ["Ruby"]
            }
        ]
    }

    with patch.object(service, "_generate_with_gemini", return_value=fake_bytes):
        updated_story = service.generate_all_scenes(story_dict, str(tmp_path))
        for sc in updated_story["scenes"]:
            assert sc.get("image_path") is not None
            assert os.path.isfile(sc["image_path"])


def test_generate_with_pollinations_mock(tmp_path, sample_scene, sample_character):
    """Test pollinations provider dispatch with mocked requests response."""
    service = ImageGeneratorService(provider="pollinations")
    fake_bytes = _create_fake_png_bytes("blue", (64, 64))
    out_file = str(tmp_path / "test_poll_01.png")

    with patch("services.image_generator.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_get.return_value = mock_response

        res_path = service.generate_single_scene_image(
            scene=sample_scene,
            characters=sample_character,
            art_style="Watercolor illustration",
            output_filepath=out_file
        )

        assert os.path.isfile(res_path)
        assert validate_image_file(res_path) is True
        assert sample_scene["illustration_status"] == "Generated"
        assert sample_scene["illustration_error"] is None


def test_pollinations_authenticated_bearer_header(sample_scene, sample_character):
    """Test that pollinations uses Authorization Bearer header when API key is provided."""
    service = ImageGeneratorService(provider="pollinations", pollinations_api_key="secret_test_key_123")
    fake_bytes = _create_fake_png_bytes("green", (64, 64))

    with patch("services.image_generator.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_get.return_value = mock_response

        service._generate_with_pollinations("A flying dragon")

        # Verify that requests.get was called with Bearer auth header
        args, kwargs = mock_get.call_args
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer secret_test_key_123"
        assert "gen.pollinations.ai" in args[0]


def test_pollinations_invalid_api_key_raises_error():
    """Test InvalidAPIKeyError raised on HTTP 401 when using pollinations API key."""
    service = ImageGeneratorService(provider="pollinations", pollinations_api_key="invalid_key")

    with patch("services.image_generator.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        with pytest.raises(InvalidAPIKeyError) as excinfo:
            service._generate_with_pollinations("A magical fairy")
        assert "invalid or expired" in str(excinfo.value).lower()


def test_pollinations_http_429_quota_exceeded():
    """Test QuotaExceededError raised when Pollinations returns HTTP 429 with retry header."""
    service = ImageGeneratorService(provider="pollinations")

    with patch("services.image_generator.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit reached. Please retry after 30 seconds."
        mock_get.return_value = mock_response

        with pytest.raises(QuotaExceededError) as excinfo:
            service._generate_with_pollinations("A magical fairy")
        assert excinfo.value.retry_after == 30
        assert "rate limit reached" in str(excinfo.value).lower()


def test_pollinations_network_timeout():
    """Test ImageNetworkError raised on connection timeout."""
    import requests
    service = ImageGeneratorService(provider="pollinations")

    with patch("services.image_generator.requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with pytest.raises(ImageNetworkError) as excinfo:
            service._generate_with_pollinations("A starry galaxy")
        assert "connection failed" in str(excinfo.value).lower() or "timeout" in str(excinfo.value).lower()


def test_invalid_image_data_fails_validation(tmp_path, sample_scene, sample_character):
    """Test ImageGenerationError raised when provider returns corrupt non-image bytes."""
    service = ImageGeneratorService(provider="pollinations")
    # Non-image HTML error response bytes
    corrupt_bytes = b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>" * 10
    out_file = str(tmp_path / "corrupt_scene.png")

    with patch.object(service, "_generate_with_pollinations", return_value=corrupt_bytes):
        with pytest.raises(Exception) as excinfo:
            service.generate_single_scene_image(
                scene=sample_scene,
                characters=sample_character,
                art_style="Colorful 2D cartoon",
                output_filepath=out_file
            )
        assert sample_scene["illustration_status"] == "Failed"
        assert sample_scene["illustration_error"] is not None


def test_regeneration_of_single_scene_workflow(tmp_path, sample_scene, sample_character):
    """Test regeneration lifecycle: Pending -> Generating -> Generated."""
    service = ImageGeneratorService(provider="pollinations")
    fake_bytes = _create_fake_png_bytes("gold", (80, 80))
    out_file = str(tmp_path / "regen_scene.png")

    sample_scene["illustration_status"] = "Pending"
    sample_scene["image_path"] = None

    with patch.object(service, "_generate_with_pollinations", return_value=fake_bytes):
        saved_path = service.generate_single_scene_image(
            scene=sample_scene,
            characters=sample_character,
            art_style="Colorful 2D cartoon",
            output_filepath=out_file
        )

        assert sample_scene["illustration_status"] == "Generated"
        assert sample_scene["image_path"] == saved_path
        assert os.path.isfile(saved_path)
        assert validate_image_file(saved_path) is True


def test_scene_description_editing_marks_pending(tmp_path, sample_scene, sample_character):
    """Test when scene description is edited or pasted, status is set to Pending and can regenerate."""
    service = ImageGeneratorService(provider="pollinations")
    fake_bytes_1 = _create_fake_png_bytes("blue", (60, 60))
    fake_bytes_2 = _create_fake_png_bytes("red", (60, 60))
    out_file = str(tmp_path / "edited_scene.png")

    # Initial generation
    with patch.object(service, "_generate_with_pollinations", return_value=fake_bytes_1):
        service.generate_single_scene_image(
            scene=sample_scene,
            characters=sample_character,
            art_style="Colorful 2D cartoon",
            output_filepath=out_file
        )
        assert sample_scene["illustration_status"] == "Generated"

    # User edits/pastes a new description
    sample_scene["heading"] = "The Hidden Crystal Cave"
    sample_scene["narration"] = "Ruby walked through glowing amethyst crystals inside the cave."
    sample_scene["environment"] = "Luminescent crystalline cavern"
    sample_scene["illustration_status"] = "Pending"
    sample_scene["illustration_error"] = None

    assert sample_scene["illustration_status"] == "Pending"

    # Now regenerate the scene illustration with the new description
    with patch.object(service, "_generate_with_pollinations", return_value=fake_bytes_2):
        new_path = service.generate_single_scene_image(
            scene=sample_scene,
            characters=sample_character,
            art_style="Colorful 2D cartoon",
            output_filepath=out_file
        )
        assert sample_scene["illustration_status"] == "Generated"
        assert sample_scene["image_path"] == new_path
        assert os.path.isfile(new_path)

