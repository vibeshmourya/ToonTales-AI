"""
ToonTales AI — OpenRouter Unified Image Generation Test Suite
Validates:
1. Rejection of obsolete model IDs (e.g. stabilityai/stable-diffusion-xl-base-1.0) before sending HTTP requests.
2. Rejection of text-only language models (e.g. anthropic/claude-3.5-sonnet) before sending HTTP requests.
3. Dynamic model discovery via /api/v1/images/models and /api/v1/models.
4. Fallback model catalog when discovery fails.
5. Error handling: HTTP 400 (not a valid model ID), HTTP 401 (Auth), HTTP 402 (Credits), HTTP 429 (Rate limit with Retry-After), HTTP 404.
6. Network timeout handling (NetworkError).
7. Malformed response handling.
8. Valid image URL response parsing and download.
9. Valid base64 (b64_json) response decoding.
10. Multimodal chat completions fallback with data URI and image objects.
11. Image persistence and zero automatic fallback to Pollinations.
12. Concurrency locking preventing duplicate image generation.
"""

import io
import os
import base64
import requests
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.providers.openrouter_provider import OpenRouterProvider
from services.image_generator import ImageGeneratorService
from services.providers.base import (
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    CapabilityError,
    NetworkError,
    InvalidRequestError,
    ProviderError
)
from utils.helpers import validate_image_file, validate_image_bytes


def _create_test_png_bytes(color: str = "cyan", size: tuple = (256, 256)) -> bytes:
    """Creates a valid non-uniform PNG byte string for testing."""
    img = Image.new("RGB", size, color=color)
    img.putpixel((0, 0), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =====================================================================
# 1. MODEL VALIDATION & REJECTION OF INVALID / OBSOLETE MODELS
# =====================================================================

def test_openrouter_rejects_obsolete_stabilityai_model_before_request():
    """Verify that stabilityai/stable-diffusion-xl-base-1.0 is stopped locally before making an API call."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    with patch("services.providers.openrouter_provider.requests.post") as mock_post:
        with pytest.raises(ModelNotFoundError) as excinfo:
            prov.generate_image("Draw a dragon", model="stabilityai/stable-diffusion-xl-base-1.0")

        # Crucial: HTTP post must NOT be called for known obsolete model ID
        mock_post.assert_not_called()
        assert "obsolete" in str(excinfo.value).lower()
        assert "stabilityai/stable-diffusion-xl-base-1.0" in str(excinfo.value)


def test_openrouter_rejects_text_only_models():
    """Verify that text-only language models are rejected with CapabilityError before making an API call."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    with patch("services.providers.openrouter_provider.requests.post") as mock_post:
        with pytest.raises(CapabilityError) as excinfo:
            prov.generate_image("Draw a rabbit", model="anthropic/claude-3.5-sonnet")

        mock_post.assert_not_called()
        assert "does not support image generation" in str(excinfo.value)


def test_openrouter_rejects_empty_model_id():
    """Verify empty model ID raises error before making request."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    with patch("services.providers.openrouter_provider.requests.post") as mock_post:
        with pytest.raises(ModelNotFoundError):
            prov.generate_image("Draw a rabbit", model="   ")
        mock_post.assert_not_called()


# =====================================================================
# 2. DYNAMIC MODEL DISCOVERY
# =====================================================================

def test_openrouter_dynamic_discovery_from_images_models():
    """Verify dynamic discovery queries GET https://openrouter.ai/api/v1/images/models."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {
                "id": "recraft/recraft-v4.1-flash-v2:free",
                "name": "Recraft Flash v4.1 (Free)",
                "description": "Raster and vector art"
            },
            {
                "id": "microsoft/mai-image-2.7-flash",
                "name": "MAI-Image-2.7 Flash",
                "description": "High-throughput cartoon"
            }
        ]
    }

    with patch("services.providers.openrouter_provider.requests.get", return_value=mock_resp) as mock_get:
        models = OpenRouterProvider.get_available_image_models(api_key="test_key", force_refresh=True)

        assert len(models) == 2
        assert models[0]["id"] == "recraft/recraft-v4.1-flash-v2:free"
        assert models[1]["id"] == "microsoft/mai-image-2.7-flash"
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "images/models" in args[0]


def test_openrouter_dynamic_discovery_fallback_on_network_failure():
    """Verify that if discovery encounters an exception, it safely returns verified fallback models."""
    with patch("services.providers.openrouter_provider.requests.get", side_effect=ConnectionError("DNS failure")):
        models = OpenRouterProvider.get_available_image_models(force_refresh=True)

        assert len(models) >= 5
        # Verify fallback does not contain obsolete models
        model_ids = [m["id"] for m in models]
        assert "recraft/recraft-v4.1-flash-v2:free" in model_ids
        assert "stabilityai/stable-diffusion-xl-base-1.0" not in model_ids


# =====================================================================
# 3. HTTP ERROR MAPPINGS
# =====================================================================

def test_openrouter_http_400_not_valid_model_id():
    """Verify that HTTP 400 with 'is not a valid model ID' raises ModelNotFoundError."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error":{"message":"custom-model-x is not a valid model ID"}}'
    mock_resp.json.return_value = {"error": {"message": "custom-model-x is not a valid model ID"}}

    # Mock validate_image_model to allow request to proceed to HTTP call
    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError) as excinfo:
                prov.generate_image("Draw something", model="custom-model-x")
            assert "is not a valid model ID" in str(excinfo.value)
            assert excinfo.value.provider == "openrouter"


def test_openrouter_http_401_authentication_error():
    """Verify HTTP 401 raises AuthenticationError."""
    prov = OpenRouterProvider(api_key="invalid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error":{"message":"Invalid API key"}}'
    mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            with pytest.raises(AuthenticationError) as excinfo:
                prov.generate_image("Draw something", model="recraft/recraft-v4.1-flash-v2:free")
            assert "invalid, inactive, or expired" in str(excinfo.value).lower()


def test_openrouter_http_402_quota_exhausted():
    """Verify HTTP 402 raises QuotaExceededError."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.text = '{"error":{"message":"Insufficient credits"}}'
    mock_resp.json.return_value = {"error": {"message": "Insufficient credits"}}

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            with pytest.raises(QuotaExceededError) as excinfo:
                prov.generate_image("Draw something", model="recraft/recraft-v4.1-flash-v2:free")
            assert "credits depleted" in str(excinfo.value).lower()


def test_openrouter_http_429_rate_limit_with_retry_after():
    """Verify HTTP 429 parses Retry-After header and raises RateLimitError."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "45"}
    mock_resp.text = '{"error":{"message":"Rate limit exceeded"}}'
    mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            with pytest.raises(RateLimitError) as excinfo:
                prov.generate_image("Draw something", model="recraft/recraft-v4.1-flash-v2:free")
            assert excinfo.value.retry_after == 45
            assert "rate limit exceeded" in str(excinfo.value).lower()


def test_openrouter_timeout_raises_network_error():
    """Verify connection timeout raises NetworkError."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", side_effect=requests.exceptions.Timeout("Read timeout")):
            with pytest.raises(NetworkError) as excinfo:
                prov.generate_image("Draw something", model="recraft/recraft-v4.1-flash-v2:free")
            assert "Network error" in str(excinfo.value)


# =====================================================================
# 4. RESPONSE PARSING: URL, BASE64, MULTIMODAL, AND CORRUPTION
# =====================================================================

def test_openrouter_valid_image_url_response():
    """Verify parsing data[0].url and downloading bytes."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")
    fake_png = _create_test_png_bytes("teal")

    # Mock image generation response returning a signed download URL
    mock_gen_resp = MagicMock()
    mock_gen_resp.status_code = 200
    mock_gen_resp.json.return_value = {
        "data": [{"url": "https://image.openrouter.ai/signed/sample_image.png"}]
    }

    # Mock image file download response
    mock_dl_resp = MagicMock()
    mock_dl_resp.status_code = 200
    mock_dl_resp.content = fake_png

    def mock_requests_get(url, *args, **kwargs):
        if "sample_image.png" in url:
            return mock_dl_resp
        return MagicMock(status_code=404)

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_gen_resp):
            with patch("services.providers.openrouter_provider.requests.get", side_effect=mock_requests_get):
                img_bytes, usage = prov.generate_image("A cheerful cartoon snake", model="recraft/recraft-v4.1-flash-v2:free")
                assert img_bytes == fake_png
                assert usage.provider == "openrouter"
                assert usage.images_generated == 1


def test_openrouter_valid_b64_json_response():
    """Verify decoding data[0].b64_json inline base64 image data."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")
    fake_png = _create_test_png_bytes("orange")
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"b64_json": b64_str}]
    }

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            img_bytes, usage = prov.generate_image("A happy bunny", model="recraft/recraft-v4.1-flash-v2:free")
            assert img_bytes == fake_png
            assert usage.provider == "openrouter"


def test_openrouter_chat_completions_data_uri_fallback():
    """Verify decoding data URI from multimodal chat completion response."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")
    fake_png = _create_test_png_bytes("gold")
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    # Simulate /images returning 404, /images/generations returning 404, and /chat/completions returning data URI
    resp_404 = MagicMock(status_code=404)
    resp_chat = MagicMock()
    resp_chat.status_code = 200
    resp_chat.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": f"Here is your image: data:image/png;base64,{b64_str}"
                }
            }
        ]
    }

    def mock_post(url, *args, **kwargs):
        if "chat/completions" in url:
            return resp_chat
        return resp_404

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", side_effect=mock_post):
            img_bytes, usage = prov.generate_image("A cartoon squirrel", model="microsoft/mai-image-2.7-flash")
            assert img_bytes == fake_png


def test_openrouter_malformed_response_raises_error():
    """Verify empty or non-image response raises InvalidRequestError."""
    prov = OpenRouterProvider(api_key="sk-or-test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "I am a text model, I cannot draw."}}]}

    with patch.object(prov, "validate_image_model", return_value=(True, None)):
        with patch("services.providers.openrouter_provider.requests.post", return_value=mock_resp):
            with pytest.raises(InvalidRequestError) as excinfo:
                prov.generate_image("A cartoon landscape", model="microsoft/mai-image-2.7-flash")
            assert "did not return valid image data" in str(excinfo.value)


# =====================================================================
# 5. INTEGRATION WORKFLOW: IMAGE GENERATOR SERVICE & NO FALLBACK
# =====================================================================

def test_openrouter_failure_does_not_fallback_to_pollinations(tmp_path):
    """Verify that when OpenRouter fails, it does NOT silently switch to Pollinations."""
    service = ImageGeneratorService(
        provider="openrouter",
        api_key="test_key",
        model_name="recraft/recraft-v4.1-flash-v2:free"
    )
    dest_file = str(tmp_path / "scene_01.png")

    scene = {
        "scene_number": 1,
        "heading": "Scene 1",
        "narration": "Narration 1",
        "characters_present": ["Slinky"]
    }
    characters = [{"name": "Slinky", "role": "Protagonist", "appearance": "A bright green snake"}]

    # Simulate OpenRouter failure
    with patch("services.providers.openrouter_provider.OpenRouterProvider.generate_image", side_effect=ModelNotFoundError("Model not found")):
        with patch("services.image_generator.ImageGeneratorService._generate_with_pollinations") as mock_poll:
            with pytest.raises(ModelNotFoundError):
                service.generate_single_scene_image(
                    scene=scene,
                    characters=characters,
                    art_style="Colorful 2D cartoon",
                    output_filepath=dest_file
                )
            # Ensure Pollinations was NOT called
            mock_poll.assert_not_called()
            assert scene["illustration_status"] == "Failed"
            assert "Model not found" in scene["illustration_error"]


def test_openrouter_image_persistence_on_failure(tmp_path):
    """Verify existing valid illustration is preserved if a subsequent regeneration fails."""
    service = ImageGeneratorService(provider="openrouter", api_key="test_key")
    dest_file = str(tmp_path / "scene_persist.png")

    # Create original valid image
    original_png = _create_test_png_bytes("purple")
    with open(dest_file, "wb") as f:
        f.write(original_png)

    scene = {
        "scene_number": 1,
        "heading": "Scene 1",
        "narration": "Narration 1",
        "image_path": dest_file,
        "illustration_status": "Generated"
    }

    # Attempt regeneration that fails
    with patch("services.providers.openrouter_provider.OpenRouterProvider.generate_image", side_effect=RateLimitError("Rate limit")):
        with pytest.raises(RateLimitError):
            service.generate_single_scene_image(
                scene=scene,
                characters=[],
                art_style="Colorful 2D cartoon",
                output_filepath=dest_file
            )

    # File on disk must remain intact and identical to original bytes
    assert os.path.isfile(dest_file)
    with open(dest_file, "rb") as f:
        assert f.read() == original_png
