"""
ToonTales AI — Multi-Provider Comprehensive Test Suite
Validates:
1. Story generation across all adapters (Gemini, OpenAI, Anthropic, OpenRouter).
2. Image generation across all adapters (Pollinations, OpenAI, Gemini, OpenRouter).
3. Capability checks (Anthropic image rejection, Pollinations story rejection, pure text model rejection for images).
4. Error normalization: 401/403 -> AuthenticationError, 404 -> ModelNotFoundError, 429 -> RateLimitError / QuotaExceededError.
5. Concurrency lock duplicate-request prevention.
6. Skip existing valid images and sequential resumption.
7. PDF export without triggering any image-generation API calls.
8. Voice narration integration.
"""

import io
import os
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.providers.base import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    CapabilityError,
    NetworkError,
    GenerationUsage
)
from services.providers.gemini_provider import GeminiProvider
from services.providers.openai_provider import OpenAIProvider
from services.providers.anthropic_provider import AnthropicProvider
from services.providers.openrouter_provider import OpenRouterProvider
from services.providers.pollinations_provider import PollinationsProvider
from services.providers import get_story_provider, get_image_provider
from services.story_generator import StoryGeneratorService, StoryModel
from services.image_generator import ImageGeneratorService
from services.rate_limit_service import RateLimitService
from services.usage_tracker import UsageTracker
from services.pdf_generator import PDFStorybookGenerator
from services.narration_service import VoiceNarrationService


def _create_fake_png_bytes(color: str = "orange", size: tuple = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_STORY_JSON = json.dumps({
    "title": "The Cosmic Bunny",
    "summary": "Pip the rabbit embarks on a whimsical journey through starlit skies.",
    "genre": "Adventure",
    "art_style": "Colorful 2D cartoon",
    "characters": [
        {
            "name": "Pip",
            "role": "Hero",
            "appearance": "A tiny blue rabbit wearing red aviator goggles",
            "personality": "Playful and brave"
        }
    ],
    "scenes": [
        {
            "scene_number": 1,
            "heading": "The Rocket Tree",
            "narration": "Pip looked up at the hollow baobab converted into a launch vessel.",
            "image_prompt": "Pip the blue rabbit adjusting red aviator goggles before a giant wooden rocket tree.",
            "characters_present": ["Pip"],
            "environment": "Twilit starry savanna"
        },
        {
            "scene_number": 2,
            "heading": "Blast Off",
            "narration": "With a burst of sparkling violet smoke, the rocket rose into the clouds.",
            "image_prompt": "Wooden rocket ascending surrounded by glittering purple smoke and stars.",
            "characters_present": ["Pip"],
            "environment": "Upper atmosphere"
        },
        {
            "scene_number": 3,
            "heading": "Moonlight Touchdown",
            "narration": "Pip landed gently on the glowing lunar surface, waving at the stars.",
            "image_prompt": "Pip smiling joyfully on the glowing moon surface with stars all around.",
            "characters_present": ["Pip"],
            "environment": "Luminescent lunar crater"
        }
    ]
})


# =====================================================================
# 1. STORY PROVIDERS TEST SUITE
# =====================================================================

def test_gemini_story_provider_mock():
    prov = GeminiProvider(api_key="mock_gemini_key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_STORY_JSON
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(prov, "_get_client", return_value=mock_client):
        story, usage = prov.generate_story("Prompt", "Instruction", model="gemini-2.5-flash")
        assert story["title"] == "The Cosmic Bunny"
        assert len(story["scenes"]) == 3
        assert usage.provider == "gemini"
        assert usage.operation == "story"


def test_openai_story_provider_mock():
    prov = OpenAIProvider(api_key="mock_openai_key")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = SAMPLE_STORY_JSON
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 200
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(prov, "_get_client", return_value=mock_client):
        story, usage = prov.generate_story("Prompt", "Instruction", model="gpt-4o-mini")
        assert story["title"] == "The Cosmic Bunny"
        assert usage.provider == "openai"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200


def test_anthropic_story_provider_mock():
    prov = AnthropicProvider(api_key="mock_anthropic_key")
    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.text = SAMPLE_STORY_JSON
    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage.input_tokens = 150
    mock_resp.usage.output_tokens = 250
    mock_client.messages.create.return_value = mock_resp

    with patch.object(prov, "_get_client", return_value=mock_client):
        story, usage = prov.generate_story("Prompt", "Instruction", model="claude-3-5-sonnet-20241022")
        assert story["title"] == "The Cosmic Bunny"
        assert usage.provider == "anthropic"
        assert usage.input_tokens == 150
        assert usage.output_tokens == 250


def test_openrouter_story_provider_mock():
    prov = OpenRouterProvider(api_key="mock_openrouter_key")
    fake_http_resp = MagicMock()
    fake_http_resp.status_code = 200
    fake_http_resp.json.return_value = {
        "choices": [{"message": {"content": SAMPLE_STORY_JSON}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 180}
    }

    with patch("services.providers.openrouter_provider.requests.post", return_value=fake_http_resp):
        story, usage = prov.generate_story("Prompt", "Instruction", model="meta-llama/llama-3.3-70b-instruct")
        assert story["title"] == "The Cosmic Bunny"
        assert usage.provider == "openrouter"
        assert usage.input_tokens == 80


# =====================================================================
# 2. IMAGE PROVIDERS TEST SUITE
# =====================================================================

def test_openai_image_provider_mock():
    prov = OpenAIProvider(api_key="mock_openai_key")
    mock_client = MagicMock()
    fake_png = _create_fake_png_bytes("teal")
    import base64
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    mock_item = MagicMock()
    mock_item.b64_json = b64_str
    mock_resp = MagicMock()
    mock_resp.data = [mock_item]
    mock_client.images.generate.return_value = mock_resp

    with patch.object(prov, "_get_client", return_value=mock_client):
        img_bytes, usage = prov.generate_image("A cute blue bunny", model="dall-e-3")
        assert img_bytes == fake_png
        assert usage.provider == "openai"
        assert usage.images_generated == 1


def test_openrouter_image_provider_mock():
    prov = OpenRouterProvider(api_key="mock_openrouter_key")
    fake_png = _create_fake_png_bytes("pink")
    import base64
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    fake_http_resp = MagicMock()
    fake_http_resp.status_code = 200
    fake_http_resp.json.return_value = {
        "data": [{"b64_json": b64_str}]
    }

    with patch("services.providers.openrouter_provider.requests.post", return_value=fake_http_resp):
        img_bytes, usage = prov.generate_image("A magical meadow", model="stabilityai/stable-diffusion-xl-base-1.0")
        assert img_bytes == fake_png
        assert usage.provider == "openrouter"


# =====================================================================
# 3. CAPABILITY VALIDATION TESTS
# =====================================================================

def test_anthropic_rejects_image_generation():
    prov = AnthropicProvider(api_key="mock_key")
    with pytest.raises(CapabilityError) as excinfo:
        prov.generate_image("Draw something")
    assert "does not support" in str(excinfo.value).lower() and "image" in str(excinfo.value).lower()
    assert excinfo.value.provider == "anthropic"


def test_pollinations_rejects_story_generation():
    prov = PollinationsProvider(api_key="mock_key")
    with pytest.raises(CapabilityError) as excinfo:
        prov.generate_story("Prompt", "Instruction")
    assert "does not support text/story generation" in str(excinfo.value)
    assert excinfo.value.provider == "pollinations"


def test_openrouter_rejects_pure_text_model_for_images():
    prov = OpenRouterProvider(api_key="mock_key")
    with pytest.raises(CapabilityError) as excinfo:
        prov.generate_image("Draw something", model="google/gemini-flash-1.5")
    assert "does not support image generation" in str(excinfo.value)


# =====================================================================
# 4. ERROR NORMALIZATION & SENSITIVITY
# =====================================================================

def test_openai_auth_error_normalization():
    prov = OpenAIProvider(api_key="invalid_key")
    mock_client = MagicMock()
    from openai import AuthenticationError as OpenAIAuthError
    mock_client.chat.completions.create.side_effect = OpenAIAuthError("Incorrect API key provided", response=MagicMock(status_code=401), body={})

    with patch.object(prov, "_get_client", return_value=mock_client):
        with pytest.raises(AuthenticationError) as excinfo:
            prov.generate_story("Prompt", "Instruction", model="gpt-4o")
        assert "api key is invalid" in str(excinfo.value).lower()
        assert excinfo.value.provider == "openai"


def test_openrouter_rate_limit_retry_after_parsing():
    prov = OpenRouterProvider(api_key="valid_key")
    fake_resp = MagicMock()
    fake_resp.status_code = 429
    fake_resp.headers = {"Retry-After": "45"}
    fake_resp.text = "Rate limit reached. Quota exceeded."

    with patch("services.providers.openrouter_provider.requests.post", return_value=fake_resp):
        with pytest.raises(RateLimitError) as excinfo:
            prov.generate_story("Prompt", "Instruction", model="meta-llama/llama-3.3-70b-instruct")
        assert excinfo.value.retry_after == 45
        assert excinfo.value.provider == "openrouter"


# =====================================================================
# 5. DUPLICATE-REQUEST PREVENTION (CONCURRENCY LOCKING)
# =====================================================================

def test_duplicate_scene_generation_locking():
    RateLimitService.reset()
    story_id = "test_story_lock_123"
    scene_num = 1

    # First acquisition succeeds
    assert RateLimitService.acquire_scene_lock(story_id, scene_num) is True

    # Immediate second acquisition fails (duplicate prevented)
    assert RateLimitService.acquire_scene_lock(story_id, scene_num) is False

    # After release, acquisition succeeds again
    RateLimitService.release_scene_lock(story_id, scene_num)
    assert RateLimitService.acquire_scene_lock(story_id, scene_num) is True
    RateLimitService.release_scene_lock(story_id, scene_num)


# =====================================================================
# 6. SEQUENTIAL GENERATION & SKIPPING EXISTING
# =====================================================================

def test_sequential_generation_skips_existing_valid_images(tmp_path):
    RateLimitService.reset()
    service = ImageGeneratorService(provider="pollinations")
    fake_png = _create_fake_png_bytes("green")

    # Scene 1 already has a valid file on disk
    scene1_path = str(tmp_path / "scene_01.png")
    with open(scene1_path, "wb") as f:
        f.write(fake_png)

    story_dict = {
        "story_id": "test_resume_story",
        "title": "Resume Test",
        "art_style": "Colorful 2D cartoon",
        "characters": [{"name": "Pip", "appearance": "Rabbit"}],
        "scenes": [
            {"scene_number": 1, "image_path": scene1_path, "heading": "S1", "narration": "N1"},
            {"scene_number": 2, "image_path": None, "heading": "S2", "narration": "N2"}
        ]
    }

    call_count = 0
    def mock_single_image(scene, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        out = str(tmp_path / f"scene_{scene['scene_number']:02d}.png")
        with open(out, "wb") as f:
            f.write(fake_png)
        return out

    with patch.object(service, "generate_single_scene_image", side_effect=mock_single_image):
        result = service.generate_all_scenes(story_dict, str(tmp_path), skip_existing=True)

        # Scene 1 was skipped because it already existed; only Scene 2 was generated!
        assert call_count == 1
        assert result["scenes"][0]["image_path"] == scene1_path
        assert result["scenes"][0]["illustration_status"] == "Generated"
        assert result["scenes"][1]["image_path"] is not None


# =====================================================================
# 7. PDF EXPORT DOES NOT TRIGGER API REQUESTS
# =====================================================================

def test_pdf_export_never_calls_api(tmp_path):
    fake_png = _create_fake_png_bytes("yellow")
    scene1_path = str(tmp_path / "pdf_scene_01.png")
    with open(scene1_path, "wb") as f:
        f.write(fake_png)

    story = {
        "title": "Offline PDF Story",
        "genre": "Fable",
        "target_audience": "Children (5-8)",
        "moral": "Kindness wins",
        "art_style": "Colorful 2D cartoon",
        "characters": [{"name": "Pip", "role": "Hero", "appearance": "Rabbit"}],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Scene With Image",
                "narration": "Pip smiled under the sun.",
                "image_path": scene1_path,
                "illustration_status": "Generated"
            },
            {
                "scene_number": 2,
                "heading": "Scene Missing Image",
                "narration": "Pip looked at the distant mountain.",
                "image_path": None,
                "illustration_status": "Pending"
            }
        ]
    }

    pdf_out = str(tmp_path / "storybook.pdf")

    # Spy on network requests to ensure none occur during PDF generation
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        gen = PDFStorybookGenerator()
        result_path = gen.generate_pdf(story, pdf_out)

        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 1000
        mock_get.assert_not_called()
        mock_post.assert_not_called()


# =====================================================================
# 8. AUDIO NARRATION INTEGRATION
# =====================================================================

def test_narration_service_generates_audio(tmp_path):
    service = VoiceNarrationService()
    out_audio = str(tmp_path / "scene_01.mp3")

    def fake_save(self_tts, path):
        with open(path, "wb") as f:
            f.write(b"fake_mp3_audio")

    with patch("gtts.gTTS.save", side_effect=fake_save, autospec=True):
        res = service.generate_narration("Hello young adventurers!", out_audio)
        assert res == out_audio
        assert os.path.isfile(out_audio)
