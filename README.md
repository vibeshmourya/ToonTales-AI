# ToonTales AI - Multi-Provider Generative AI Cartoon Story & Illustration Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI](https://img.shields.io/badge/Provider-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![OpenAI](https://img.shields.io/badge/Provider-OpenAI%20GPT--4o%20%26%20DALL--E-10a37f.svg)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Provider-Anthropic%20Claude-d97757.svg)](https://console.anthropic.com/)
[![OpenRouter](https://img.shields.io/badge/Provider-OpenRouter-6366f1.svg)](https://openrouter.ai/)
[![Pollinations](https://img.shields.io/badge/Provider-Pollinations%20AI-ec4899.svg)](https://pollinations.ai/)
[![Tests](https://img.shields.io/badge/Tests-42%2F42%20Passing-brightgreen.svg)]()
[![Academic Project](https://img.shields.io/badge/MCA-Generative%20AI%20Project-purple.svg)]()

> **ToonTales AI** is an enterprise-grade, modular, multi-provider AI storytelling and cartoon illustration engine. It transforms story concepts into fully illustrated digital storybooks with episodic scenes, character design persistence, voice narration, and publication-ready PDF exports.

---

## 1. Multi-Provider Architecture

ToonTales AI decouples Story Generation from Image Generation, allowing users to configure any supported text provider independently from any supported image provider.

```
                    ┌───────────────────────────────────────────────┐
                    │               ToonTales AI                    │
                    │               Streamlit UI                    │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
             ┌──────────────▼─────────────┐   ┌─────────────▼──────────────┐
             │   Story Generator Service  │   │   Image Generator Service  │
             └──────────────┬─────────────┘   └─────────────┬──────────────┘
                            │                               │
      ┌─────────────────────┼────────────────────┐          │
      ▼                     ▼                    ▼          ▼
[Google Gemini]       [OpenAI GPT-4o]    [Anthropic Claude] [OpenRouter]
(gemini-2.5-flash)    (gpt-4o, mini)     (claude-3-5-sonnet) (Llama-3.3, etc.)
                                                 │
                                                 │ (Claude: Text-only adapter)
                                                 │ (Rejects image generation)
                            ┌────────────────────┼────────────────────┐
                            ▼                    ▼                    ▼
                    [Pollinations AI]     [OpenAI DALL-E]     [Google Gemini]
                    (flux, turbo, etc.)   (dall-e-3, dall-e-2) (2.5-flash-image)
```

### Supported Providers & Capabilities

| Provider | Story Generation | Image Generation | Models Supported | Authentication |
| :--- | :---: | :---: | :--- | :--- |
| **Google Gemini** | Yes | Yes | Text: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`<br/>Images: `gemini-2.5-flash-image`, `gemini-3.1-flash-image` | Google AI Studio Key (`GEMINI_API_KEY`) |
| **OpenAI** | Yes | Yes | Text: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`<br/>Images: `dall-e-3`, `dall-e-2` | OpenAI API Key (`OPENAI_API_KEY`) |
| **Anthropic Claude** | Yes | **No** (Enforced) | Text: `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet`, `claude-3-5-haiku`, `claude-3-haiku`, `claude-3-opus` | Anthropic API Key (`ANTHROPIC_API_KEY`) |
| **OpenRouter** | Yes | Yes (Validated) | Text: `meta-llama/llama-3.3-70b-instruct`, `anthropic/claude-3.5-sonnet`, etc.<br/>Images: `stabilityai/stable-diffusion-xl-base-1.0`, etc. | OpenRouter Key (`OPENROUTER_API_KEY`) |
| **Pollinations AI** | **No** (Enforced) | Yes | Images: `flux` (recommended), `turbo`, `flux-realism`, `flux-anime`, `flux-3d` | Zero-cost open tier or optional `POLLINATIONS_API_KEY` |

---

## 2. Secure API Key Management

* **Password-Masked Inputs:** All secret inputs in the Streamlit sidebar use `type="password"`.
* **Zero Secret Leakage:** Keys are never rendered in URLs, story metadata, image prompts, PDF headers, or logs. Secrets are masked (e.g. `sk-proj-****abcd`) in UI feedback.
* **Session State & Environment Precedence:** Keys are loaded from `.env` / `os.environ` / Streamlit secrets by default, and user entry in the sidebar overrides for that interactive session without modifying files on disk.

---

## 3. Rate Limiting, Retries & Concurrency Locking

To prevent rate-limit exhaustion and UI stutter:

1. **Scene-Level Concurrency Locks:** `RateLimitService.acquire_scene_lock(story_id, scene_num)` blocks duplicate clicks and Streamlit reruns from dispatching simultaneous API requests for the same scene.
2. **Dynamic `Retry-After` Header Parsing:** Automatically extracts cooldown durations from provider response headers (e.g. HTTP 429) and enforces a bounded backoff cooldown before retrying.
3. **Sequential Scene Generation:** Scenes are generated sequentially one-by-one with granular progress bars.
4. **Resumable Generation:** Scenes that already have valid illustrations on disk are identified and skipped by default. If generation is halted due to a rate limit or quota issue, completed scenes are preserved and can be resumed later.
5. **Atomic Image Preservation:** Images are verified using Pillow in a temporary file (`.tmp.png`) before destination replacement. A failed regeneration attempt preserves the existing valid image without corrupting or deleting it.

---

## 4. Usage & Cost Transparency

A built-in thread-safe `UsageTracker` monitors:
- Provider, model, and operation (story vs image)
- Input and output tokens consumed
- Image generation counts
- Reported vs estimated costs (e.g. DALL-E 3 $0.04/image, GPT-4o-mini token rates)
- Zero-cost distinctions (e.g. Pollinations free tier)

View metrics at any time inside the **📊 AI Usage & Cost Transparency** expander in the sidebar.

---

## 5. Offline PDF Storybook Export

* **Zero External Requests:** The PDF generator operates entirely offline using locally cached and validated scene illustrations. It never makes external API requests during compilation.
* **Missing Image Resilience:** Scenes missing illustrations render clean placeholder callouts without crashing the document compilation.
* **Pre-Export Warning:** Displays a friendly notice before download if any illustrations are missing, while still allowing the user to export immediately.

---

## 6. Installation & Setup

### 6.1 Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `conda`)

### 6.2 Installation
```bash
git clone https://github.com/vibeshmourya/ToonTales-AI.git
cd ToonTales-AI

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Environment Configuration
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Fill in the provider API keys you wish to use:
```ini
# Story Provider ('gemini', 'openai', 'anthropic', 'openrouter')
STORY_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# Image Provider ('pollinations', 'gemini', 'openai', 'openrouter')
IMAGE_PROVIDER=pollinations
POLLINATIONS_IMAGE_MODEL=flux
```

---

## 7. Running the Application

```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 8. Automated Testing

Run the full automated test suite (42 tests, 100% mocked, no paid API keys required):

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_image_generator.py::test_prompt_construction PASSED
tests/test_image_generator.py::test_image_extraction_generate_content PASSED
...
tests/test_multi_provider.py::test_gemini_story_provider_mock PASSED
tests/test_multi_provider.py::test_openai_story_provider_mock PASSED
tests/test_multi_provider.py::test_anthropic_story_provider_mock PASSED
tests/test_multi_provider.py::test_openrouter_story_provider_mock PASSED
tests/test_multi_provider.py::test_openai_image_provider_mock PASSED
tests/test_multi_provider.py::test_openrouter_image_provider_mock PASSED
tests/test_multi_provider.py::test_anthropic_rejects_image_generation PASSED
tests/test_multi_provider.py::test_pollinations_rejects_story_generation PASSED
tests/test_multi_provider.py::test_duplicate_scene_generation_locking PASSED
tests/test_multi_provider.py::test_sequential_generation_skips_existing_valid_images PASSED
tests/test_multi_provider.py::test_pdf_export_never_calls_api PASSED
tests/test_pdf_generator.py::test_pdf_generation_complete PASSED
tests/test_story_generator.py::test_story_generator_mocked_gemini PASSED

============================= 42 passed in 20.03s =============================
```

---

## 9. Academic Attribution
Developed for the MCA Generative Artificial Intelligence Curriculum.
Licensed for academic evaluation and research.
