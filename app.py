"""
ToonTales AI — Generative AI-Powered Cartoon Storytelling Platform
MCA Academic Project

A complete, end-to-end web application that transforms arbitrary story ideas into
illustrated digital cartoon storybooks using Google Gemini API and state-of-the-art
image generation.
"""

import os
import sys
import json
import logging
from pathlib import Path
import streamlit as st
from PIL import Image

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.story_generator import StoryGeneratorService
from services.image_generator import (
    ImageGeneratorService,
    QuotaExceededError,
    InvalidAPIKeyError,
    UnsupportedModelError,
    ImageNetworkError,
    ImageGenerationError
)
from services.pdf_generator import PDFStorybookGenerator
from services.narration import NarrationService
from utils.helpers import (
    create_story_workspace,
    save_story_json,
    load_story_json,
    get_config,
    mask_secret,
    validate_image_file
)

# Set page configuration
st.set_page_config(
    page_title="ToonTales AI - Cartoon Storytelling Platform",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger("ToonTalesAI.App")

# =====================================================================
# Custom CSS for Modern, Child-Friendly, and Professional Aesthetics
# =====================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary-indigo: #4f46e5;
    --primary-purple: #7c3aed;
    --accent-amber: #f59e0b;
    --accent-coral: #f43f5e;
    --bg-soft-blue: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #1e1b4b;
    --text-secondary: #475569;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

/* Header & Brand */
.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    margin-bottom: 20px;
}

.brand-title {
    font-family: 'Fredoka', cursive, sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}

.brand-badge {
    background: #e0e7ff;
    color: #4338ca;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #e0e7ff 50%, #fdf2f8 100%);
    border-radius: 20px;
    padding: 36px 30px;
    margin-bottom: 25px;
    border: 1px solid rgba(226, 232, 240, 0.8);
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.08);
}

.hero-heading {
    font-family: 'Fredoka', cursive, sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e1b4b;
    line-height: 1.2;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #475569;
    max-width: 750px;
    line-height: 1.6;
}

/* Cards */
.story-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    margin-bottom: 20px;
}

.character-card {
    background: #f8fafc;
    border-radius: 14px;
    padding: 16px;
    border-left: 5px solid #4f46e5;
    margin-bottom: 12px;
    transition: transform 0.2s ease;
}

.character-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.1);
}

.character-name {
    font-family: 'Fredoka', cursive, sans-serif;
    font-size: 1.15rem;
    color: #1e1b4b;
    font-weight: 600;
}

.character-role {
    font-size: 0.85rem;
    color: #6366f1;
    font-weight: 600;
    text-transform: uppercase;
}

/* Scene Navigation & Counter */
.scene-progress-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #eef2ff;
    padding: 10px 18px;
    border-radius: 12px;
    margin-bottom: 16px;
}

.scene-counter-text {
    font-family: 'Fredoka', cursive, sans-serif;
    font-size: 1.1rem;
    color: #3730a3;
    font-weight: 600;
}

.scene-title-badge {
    font-family: 'Fredoka', cursive, sans-serif;
    font-size: 1.7rem;
    color: #1e1b4b;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 6px;
}

.narration-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    font-size: 1.2rem;
    line-height: 1.7;
    color: #1e293b;
    margin-top: 15px;
    margin-bottom: 15px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
}

.scene-env-tag {
    font-size: 0.88rem;
    color: #64748b;
    font-style: italic;
    background: #f1f5f9;
    padding: 6px 12px;
    border-radius: 8px;
    display: inline-block;
}

/* Button Customizations */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

/* Streamlit audio widget clean margin */
audio {
    border-radius: 10px;
    width: 100%;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Session State Initialization
# =====================================================================

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Home"

if "story_data" not in st.session_state:
    st.session_state.story_data = None

if "current_scene_idx" not in st.session_state:
    st.session_state.current_scene_idx = 0

if "story_id" not in st.session_state:
    st.session_state.story_id = None

if "story_dir" not in st.session_state:
    st.session_state.story_dir = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "audio_paths" not in st.session_state:
    st.session_state.audio_paths = {}

if "is_generating_image" not in st.session_state:
    st.session_state.is_generating_image = False

if "image_provider_choice" not in st.session_state:
    st.session_state.image_provider_choice = get_config("IMAGE_PROVIDER", "pollinations").lower()

if "last_quota_error" not in st.session_state:
    st.session_state.last_quota_error = None


# =====================================================================
# Sidebar: Configuration & Model Settings
# =====================================================================

with st.sidebar:
    st.markdown("### ⚙️ ToonTales AI Settings")
    st.caption("Configure Generative AI models & keys")

    # API Key Handling
    env_gemini_key = get_config("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_gemini_key,
        type="password",
        help="Get your key at https://aistudio.google.com/. Can also be set in .env file."
    )
    active_api_key = api_key_input.strip() if api_key_input else None

    if active_api_key:
        st.success(f"Key configured: {mask_secret(active_api_key)}")
    else:
        st.warning("⚠️ No Gemini API key detected. Story generation requires Gemini.")

    st.markdown("---")
    st.markdown("#### 🤖 Model Configuration")

    # Gemini Text Model selector
    text_model_options = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    env_text_model = get_config("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
    default_text_idx = text_model_options.index(env_text_model) if env_text_model in text_model_options else 0
    selected_text_model = st.selectbox(
        "Gemini Story Model",
        options=text_model_options,
        index=default_text_idx,
        help="Model used for narrative creation, character design, and scene division."
    )

    # Image Provider Selector
    img_provider_options = ["pollinations", "gemini"]
    current_choice = st.session_state.image_provider_choice
    default_img_idx = img_provider_options.index(current_choice) if current_choice in img_provider_options else 0

    selected_image_provider = st.selectbox(
        "Image Generation Provider",
        options=img_provider_options,
        index=default_img_idx,
        format_func=lambda x: "Pollinations AI (Recommended - No Billing Required)" if x == "pollinations" else "Google Gemini (Requires Google Cloud Billing)",
        help=(
            "Pollinations AI provides free cloud diffusion and works reliably without requiring Google Cloud billing. "
            "Google Gemini image models require a Pay-As-You-Go Google Cloud project (free-tier image quota is 0)."
        )
    )
    st.session_state.image_provider_choice = selected_image_provider

    selected_image_model = "flux"
    selected_pollinations_model = "flux"
    active_pollinations_key = None

    if selected_image_provider == "pollinations":
        pol_models = ["flux", "turbo", "flux-realism", "flux-anime", "flux-3d"]
        env_pol_model = get_config("POLLINATIONS_IMAGE_MODEL", "flux")
        default_pol_idx = pol_models.index(env_pol_model) if env_pol_model in pol_models else 0
        selected_pollinations_model = st.selectbox(
            "Pollinations Model",
            options=pol_models,
            index=default_pol_idx,
            help="Select the AI diffusion model. 'flux' produces the highest quality cartoon artwork."
        )
        selected_image_model = selected_pollinations_model

        env_pol_key = get_config("POLLINATIONS_API_KEY", "")
        pol_key_input = st.text_input(
            "Pollinations API Key (Optional)",
            value=env_pol_key,
            type="password",
            help="Optional API key from enter.pollinations.ai/keys. Leave empty to use the zero-cost public tier."
        )
        active_pollinations_key = pol_key_input.strip() if pol_key_input else None
        if active_pollinations_key:
            st.success(f"Key active: {mask_secret(active_pollinations_key)}")
        else:
            st.caption("ℹ️ Using zero-cost open tier (no key required).")

    else:
        gemini_img_models = [
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image",
            "Custom..."
        ]
        env_img_model = get_config("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        default_img_model_idx = gemini_img_models.index(env_img_model) if env_img_model in gemini_img_models else 0
        
        chosen_model_preset = st.selectbox(
            "Gemini Image Model",
            options=gemini_img_models,
            index=default_img_model_idx,
            help="Select a supported Gemini multimodal image generation model."
        )
        if chosen_model_preset == "Custom...":
            selected_image_model = st.text_input(
                "Enter Custom Gemini Image Model Identifier",
                value=env_img_model if env_img_model not in ["gemini-2.5-flash-image", "gemini-3.1-flash-image"] else "",
                placeholder="e.g. gemini-2.5-flash-image"
            ).strip()
        else:
            selected_image_model = chosen_model_preset

        st.warning(
            "⚠️ **Gemini Image Quota Notice:** In Google AI Studio, native image models have 0 free requests. "
            "A Pay-As-You-Go Google Cloud billing project is required. Switch to 'Pollinations AI' above for free generation."
        )

    st.markdown("---")
    st.markdown("#### 🔊 Audio Narration")
    enable_audio = st.checkbox(
        "Enable Voice Narration",
        value=True,
        help="Converts scene narration into child-friendly speech audio using gTTS."
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b; line-height: 1.4;'>"
        "<b>ToonTales AI</b><br/>"
        "MCA Generative AI Project<br/>"
        "Powered by Google Gemini Developer API"
        "</div>",
        unsafe_allow_html=True
    )


# =====================================================================
# Top Header & Brand Bar
# =====================================================================

col_h1, col_h2 = st.columns([7, 3])
with col_h1:
    st.markdown(
        """
        <div class="brand-container">
            <span style="font-size: 2.2rem;">🎨</span>
            <h1 class="brand-title">ToonTales AI</h1>
            <span class="brand-badge">Generative AI Storybook</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_h2:
    nav_selection = st.radio(
        "Navigation",
        ["Home", "Create Story", "About"],
        horizontal=True,
        label_visibility="collapsed"
    )


# =====================================================================
# VIEW: ABOUT PAGE
# =====================================================================

if nav_selection == "About":
    st.markdown(
        """
        <div class="story-card">
            <h2>📖 About ToonTales AI</h2>
            <p>
                <b>ToonTales AI</b> is an end-to-end Generative AI platform engineered for children's cartoon storytelling.
                It is developed as an academic project for Master of Computer Applications (MCA) in Generative Artificial Intelligence.
            </p>
            <h3>🌟 Key Generative AI Capabilities</h3>
            <ul>
                <li><b>Prompt Engineering & Structured JSON:</b> Orchestrates Google Gemini to generate validated narrative JSON schemas with character profiles and chronological scenes.</li>
                <li><b>Persistent Character Consistency:</b> Visual signatures (species, costume, body shape, markings) are automatically extracted and injected into each scene's visual prompt.</li>
                <li><b>Multimodal Image Generation:</b> Supports Google Imagen 3 (via official <code>google-genai</code> SDK) and high-performance cartoon diffusion with zero-cost fallback.</li>
                <li><b>Illustrated PDF Publication:</b> Programmatically formats and exports full storybooks with cover pages, typography, and embedded illustrations using ReportLab.</li>
                <li><b>Voice Narration:</b> Integrated Text-to-Speech (gTTS) for scene-by-scene audio playback.</li>
            </ul>
            <h3>🎓 Architecture & Technologies</h3>
            <ul>
                <li><b>Language & Frontend:</b> Python 3.11+, Streamlit, Custom Responsive CSS</li>
                <li><b>AI Engine:</b> Google Gemini API (<code>google-genai</code> SDK 2.25+)</li>
                <li><b>Document Engineering:</b> ReportLab 4.5+ with dynamic Numbered Canvas</li>
                <li><b>Image Verification:</b> Pillow (PIL) for image integrity checking and RGB color-space normalization</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================================
# VIEW: HOME / HERO PAGE
# =====================================================================

elif nav_selection == "Home" and st.session_state.story_data is None:
    st.markdown(
        """
        <div class="hero-box">
            <h1 class="hero-heading">Turn Your Imagination Into a Cartoon Story 🚀</h1>
            <p class="hero-subtitle">
                Enter any whimsical idea and watch Generative AI bring it to life! 
                ToonTales AI writes an original adventure, designs consistent cartoon characters, 
                generates scene-by-scene illustrations, creates voice narration, and publishes a downloadable illustrated PDF.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🌟 Example Stories You Can Create")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="story-card">
                <h4>🐰 Ruby's Moon Launch</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    A brave little rabbit dreams of traveling to the moon in her purple astronaut suit.
                </p>
                <span class="brand-badge">Space &bull; 6–8 years</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="story-card">
                <h4>🐉 Barnaby The Tea Dragon</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    A tiny green dragon who breathes fragrant jasmine tea steam instead of fire helps a sad village smile.
                </p>
                <span class="brand-badge">Fantasy &bull; 3–5 years</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="story-card">
                <h4>🤖 Pip The Clockwork Squirrel</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    A tiny brass mechanical squirrel discovers the ancient secret of glowing solar acorns in an enchanted forest.
                </p>
                <span class="brand-badge">Adventure &bull; 9–12 years</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================================
# VIEW: CREATE STORY FORM (When no story is active or user clicked Create)
# =====================================================================

    st.markdown("### ✨ Create Your Cartoon Story")
    
    # Quick Demo Loader Card
    demo_json_path = os.path.join(BASE_DIR, "generated_stories", "demo_story", "story.json")
    if os.path.isfile(demo_json_path):
        with st.expander("💡 Want a quick demonstration? Load our sample storybook", expanded=False):
            st.info("You can explore the interactive viewer, character cards, voice narration, and illustrated PDF download instantly.")
            if st.button("🚀 Load Sample Storybook (Ruby's Moon Adventure)", use_container_width=True):
                demo_data = load_story_json(demo_json_path)
                if demo_data:
                    demo_story_dir = os.path.join(BASE_DIR, "generated_stories", "demo_story")
                    st.session_state.story_data = demo_data
                    st.session_state.story_id = "demo_story"
                    st.session_state.story_dir = demo_story_dir
                    st.session_state.pdf_path = os.path.join(demo_story_dir, "story.pdf")
                    # Check audio files
                    audio_map = {}
                    for sc in demo_data.get("scenes", []):
                        sn = sc.get("scene_number", 1)
                        af = os.path.join(demo_story_dir, f"scene_{sn:02d}.mp3")
                        if os.path.isfile(af):
                            audio_map[sn] = af
                    st.session_state.audio_paths = audio_map
                    st.session_state.current_scene_idx = 0
                    st.success("Loaded demo story!")
                    st.rerun()

    with st.form("story_creation_form"):
        story_idea_input = st.text_area(
            "What is your story idea?",
            placeholder="e.g. A little rabbit dreams of becoming an astronaut and traveling to the moon.",
            height=120,
            help="Describe your characters, setting, or the adventure they embark upon."
        )

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            genre_choice = st.selectbox(
                "Genre",
                [
                    "Adventure", "Fantasy", "Comedy", "Friendship",
                    "Mystery", "Educational", "Science Fiction", "Moral Story"
                ],
                index=0
            )
            target_age_choice = st.selectbox(
                "Target Age Group",
                ["3–5 years", "6–8 years", "9–12 years", "Teenagers", "General Audience"],
                index=1
            )

        with f_col2:
            art_style_choice = st.selectbox(
                "Cartoon Art Style",
                [
                    "Colorful 2D cartoon",
                    "Storybook illustration",
                    "Anime-inspired illustration",
                    "Watercolor illustration",
                    "3D animated-film-inspired illustration"
                ],
                index=0
            )
            story_length_choice = st.selectbox(
                "Story Length",
                ["Short", "Medium", "Long"],
                index=1
            )

        with f_col3:
            num_scenes_choice = st.selectbox(
                "Number of Scenes",
                [5, 6, 8],
                index=0,
                help="Choose how many illustrated scenes will be generated."
            )
            ending_style_choice = st.selectbox(
                "Ending Style",
                ["Happy", "Funny", "Inspirational", "Educational", "Surprise ending"],
                index=0
            )

        st.markdown(
            f"<div style='font-size: 0.85rem; color: #64748b; margin-top: 5px; margin-bottom: 12px;'>"
            f"⚡ <i>Will generate 1 original narrative with persistent characters, {num_scenes_choice} scenes, "
            f"and {num_scenes_choice} individual cartoon illustrations using <b>{selected_image_provider.capitalize()}</b>.</i>"
            f"</div>",
            unsafe_allow_html=True
        )

        submit_button = st.form_submit_button(
            "🚀 Generate My Illustrated Story",
            type="primary",
            use_container_width=True
        )

    # Process Story Generation Submission
    if submit_button:
        if not story_idea_input.strip():
            st.error("Please enter a story idea before proceeding.")
        elif selected_image_provider == "gemini" and not active_api_key:
            st.error("Gemini API Key is required. Please enter your API key in the sidebar.")
        else:
            # 1. Initialize services
            story_gen = StoryGeneratorService(
                api_key=active_api_key,
                model_name=selected_text_model
            )
            image_gen = ImageGeneratorService(
                provider=selected_image_provider,
                api_key=active_api_key,
                model_name=selected_image_model,
                pollinations_api_key=active_pollinations_key,
                pollinations_model=selected_pollinations_model
            )
            pdf_gen = PDFStorybookGenerator()
            tts_gen = NarrationService()

            # 2. Setup storage directory
            workspace = create_story_workspace(story_idea_input[:20])
            story_dir = workspace["story_dir"]
            story_id = workspace["story_id"]

            progress_placeholder = st.empty()
            status_box = st.status("🔮 Bringing your cartoon story to life...", expanded=True)

            try:
                # Step A: Narrative Generation
                status_box.write("✨ Contacting Google Gemini to craft the original story and scenes...")
                generated_story = story_gen.generate_story(
                    story_idea=story_idea_input,
                    genre=genre_choice,
                    target_age=target_age_choice,
                    story_length=story_length_choice,
                    art_style=art_style_choice,
                    num_scenes=num_scenes_choice,
                    ending_style=ending_style_choice,
                    api_key_override=active_api_key
                )
                status_box.write(f"✅ Story crafted: **{generated_story.get('title')}**")
                status_box.write(f"👥 Characters designed: {len(generated_story.get('characters', []))}")
                status_box.write(f"🎬 Scenes mapped: {len(generated_story.get('scenes', []))}")

                # Step B: Cartoon Illustrations Generation
                status_box.write(f"🎨 Generating cartoon illustrations via {selected_image_provider.upper()}...")
                
                prog_bar = status_box.progress(0)
                def on_image_progress(current, total, msg):
                    ratio = current / max(1, total)
                    prog_bar.progress(min(1.0, ratio))
                    status_box.write(f"🖼️ {msg}")

                generated_story = image_gen.generate_all_scenes(
                    story_dict=generated_story,
                    story_dir=story_dir,
                    provider_override=selected_image_provider,
                    api_key_override=active_api_key,
                    pollinations_api_key_override=active_pollinations_key,
                    pollinations_model_override=selected_pollinations_model,
                    progress_callback=on_image_progress
                )

                image_failures = generated_story.get("image_generation_failures", [])
                if image_failures:
                    zero_quota = any("zero quota" in str(msg).lower() or "free-tier" in str(msg).lower() for _, msg in image_failures)
                    if zero_quota:
                        status_box.write("⚠️ **Image Quota Notice:** Configured Gemini project has 0 free-tier quota for image models. The story text and audio are ready! You can switch the Image Provider in the sidebar to 'Pollinations AI' to generate illustrations without any quota.")
                    else:
                        status_box.write(f"⚠️ {len(image_failures)} scene illustration(s) skipped or rate-limited. They can be retried individually using '🎨 Regenerate Illustration'.")
                else:
                    status_box.write("✅ All scene illustrations generated successfully.")

                # Step C: Optional Voice Narration Generation
                audio_map = {}
                if enable_audio:
                    status_box.write("🎙️ Synthesizing voice narration for scenes...")
                    for sc in generated_story.get("scenes", []):
                        s_num = sc.get("scene_number", 1)
                        audio_filename = f"scene_{s_num:02d}.mp3"
                        audio_file_path = os.path.join(story_dir, audio_filename)
                        narr_path = tts_gen.generate_narration(sc.get("narration", ""), audio_file_path)
                        if narr_path:
                            audio_map[s_num] = narr_path
                    status_box.write(f"✅ Generated {len(audio_map)} narration audio files.")

                # Step D: PDF Storybook Export
                status_box.write("📑 Formatting downloadable PDF storybook...")
                pdf_file_path = os.path.join(story_dir, "story.pdf")
                pdf_gen.generate_pdf(generated_story, pdf_file_path)
                status_box.write("✅ Illustrated PDF created successfully.")

                # Save JSON to disk
                json_file_path = os.path.join(story_dir, "story.json")
                save_story_json(generated_story, json_file_path)

                # Update Streamlit Session State
                st.session_state.story_data = generated_story
                st.session_state.story_id = story_id
                st.session_state.story_dir = story_dir
                st.session_state.pdf_path = pdf_file_path
                st.session_state.audio_paths = audio_map
                st.session_state.current_scene_idx = 0

                status_box.update(label="🎉 Storybook successfully generated!", state="complete", expanded=False)
                st.success("Your storybook is ready! Scroll down to explore.")
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Generation encountered an issue", state="error", expanded=True)
                st.error(f"Error during story generation: {str(e)}")
                logger.error("Story generation failed", exc_info=True)


# =====================================================================
# VIEW: INTERACTIVE STORYBOOK VIEWER
# =====================================================================

if st.session_state.story_data is not None:
    story = st.session_state.story_data
    scenes = story.get("scenes", [])
    total_scenes = len(scenes)
    curr_idx = st.session_state.current_scene_idx

    # Clamp index safely
    if curr_idx >= total_scenes:
        curr_idx = total_scenes - 1
        st.session_state.current_scene_idx = curr_idx
    if curr_idx < 0:
        curr_idx = 0
        st.session_state.current_scene_idx = 0

    curr_scene = scenes[curr_idx]

    # --- Storybook Header Bar ---
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #fae8ff 100%); padding: 20px; border-radius: 16px; margin-bottom: 20px; border: 1px solid #cbd5e1;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #4338ca; text-transform: uppercase; letter-spacing: 1px;">
                ★ {story.get('genre', 'Adventure').upper()} &bull; {story.get('art_style', 'Cartoon')} ★
            </div>
            <h1 style="font-family: 'Fredoka', cursive; font-size: 2.3rem; color: #1e1b4b; margin: 4px 0 8px 0;">
                {story.get('title', 'A Magical Tale')}
            </h1>
            <p style="font-size: 1.05rem; color: #334155; margin: 0; line-height: 1.5;">
                <i>&ldquo;{story.get('summary', '')}&rdquo;</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Top Action Toolbar: Reset & PDF Download ---
    t_col1, t_col2, t_col3 = st.columns([4, 3, 3])
    with t_col1:
        st.write("") # spacer
    with t_col2:
        # Download PDF Button
        pdf_path = st.session_state.pdf_path
        if pdf_path and os.path.isfile(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Illustrated PDF",
                data=pdf_bytes,
                file_name=f"{story.get('title', 'story').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    with t_col3:
        if st.button("✨ Start New Story", use_container_width=True):
            st.session_state.story_data = None
            st.session_state.current_scene_idx = 0
            st.session_state.story_id = None
            st.session_state.pdf_path = None
            st.session_state.audio_paths = {}
            st.rerun()

    # --- Main Interactive Scene Display ---
    main_col, side_col = st.columns([7, 3])

    with main_col:
        # Scene Progress Navigation Bar
        img_status = curr_scene.get("illustration_status")
        if not img_status:
            img_status = "Generated" if (curr_scene.get("image_path") and os.path.isfile(curr_scene.get("image_path", ""))) else "Pending"
            curr_scene["illustration_status"] = img_status

        status_badge_map = {
            "Generated": ("#dcfce7", "#15803d", "✅ Illustration Generated"),
            "Pending": ("#fef3c7", "#b45309", "⏳ Illustration Pending"),
            "Generating": ("#e0e7ff", "#4338ca", "🔄 Generating..."),
            "Failed": ("#fee2e2", "#b91c1c", "❌ Illustration Failed"),
        }
        bg_c, txt_c, badge_lbl = status_badge_map.get(img_status, ("#f1f5f9", "#475569", img_status))

        st.markdown(
            f"""
            <div class="scene-progress-bar">
                <span class="scene-counter-text">📖 Scene {curr_idx + 1} of {total_scenes}</span>
                <span style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: {bg_c}; color: {txt_c}; padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">{badge_lbl}</span>
                    <span style="font-size: 0.9rem; color: #6366f1; font-weight: 600;">{story.get('art_style')}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='scene-title-badge'>{curr_scene.get('heading', f'Scene {curr_idx + 1}')}</div>",
            unsafe_allow_html=True
        )

        # Scene Image Display
        img_path = curr_scene.get("image_path")
        if img_path and os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to render image: {e}")
        else:
            st.info(
                f"🎨 Illustration for Scene {curr_idx + 1} is pending or failed. "
                "Click **'🎨 Regenerate Illustration'** below to generate it."
            )

        # Display failure reason if failed
        if img_status == "Failed" and curr_scene.get("illustration_error"):
            st.error(f"**Illustration Error:** {curr_scene.get('illustration_error')}")

        # Gemini Quota Exhaustion Helper & One-Click Recovery
        if img_status == "Failed" and curr_scene.get("illustration_error") and ("quota" in curr_scene.get("illustration_error", "").lower() or "free-tier" in curr_scene.get("illustration_error", "").lower()):
            st.warning(
                "💡 **Google Gemini Quota Notice:** Google AI Studio allocates 0 free-tier quota for image models. "
                "You can switch to **Pollinations AI** for zero-cost cloud diffusion."
            )
            if st.button("👉 Switch to Pollinations AI & Generate Illustration Now", type="primary", key=f"switch_to_poll_{curr_idx}", use_container_width=True):
                st.session_state.image_provider_choice = "pollinations"
                with st.spinner("Generating illustration via Pollinations AI..."):
                    try:
                        st.session_state.is_generating_image = True
                        curr_scene["illustration_status"] = "Generating"
                        img_service = ImageGeneratorService(
                            provider="pollinations",
                            pollinations_api_key=active_pollinations_key,
                            pollinations_model=selected_pollinations_model
                        )
                        scene_num = curr_scene.get("scene_number", curr_idx + 1)
                        dest_file = os.path.join(
                            st.session_state.story_dir, f"scene_{scene_num:02d}.png"
                        )
                        new_img_path = img_service.generate_single_scene_image(
                            scene=curr_scene,
                            characters=story.get("characters", []),
                            art_style=story.get("art_style", "Colorful 2D cartoon"),
                            output_filepath=dest_file,
                            provider_override="pollinations",
                            pollinations_api_key_override=active_pollinations_key,
                            pollinations_model_override=selected_pollinations_model
                        )
                        curr_scene["image_path"] = new_img_path
                        curr_scene["illustration_status"] = "Generated"
                        curr_scene["illustration_error"] = None
                        
                        pdf_svc = PDFStorybookGenerator()
                        pdf_svc.generate_pdf(story, st.session_state.pdf_path)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.success(f"Scene {curr_idx + 1} illustration generated via Pollinations AI!")
                        st.rerun()
                    except Exception as poll_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(poll_err)
                        st.error(f"Pollinations generation failed: {poll_err}")
                    finally:
                        st.session_state.is_generating_image = False

        # Narration Box
        st.markdown(
            f"""
            <div class="narration-box">
                {curr_scene.get('narration', '')}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Voice Narration Audio Player
        scene_num = curr_scene.get("scene_number", curr_idx + 1)
        audio_file = st.session_state.audio_paths.get(scene_num)
        if audio_file and os.path.isfile(audio_file):
            st.audio(audio_file, format="audio/mp3")

        # Environment & Setting info tag
        if curr_scene.get("environment"):
            st.markdown(
                f"<div class='scene-env-tag'>📍 <b>Setting:</b> {curr_scene.get('environment')}</div>",
                unsafe_allow_html=True
            )

        # Scene Description Editor / Paster (Requirement 3)
        with st.expander("✏️ Edit or Paste Scene Description", expanded=False):
            st.caption("Edit scene narration or paste new descriptions. Saving sets status to 'Pending' so you can generate an illustration.")
            with st.form(key=f"scene_editor_form_{curr_idx}"):
                edit_heading = st.text_input("Scene Heading", value=curr_scene.get("heading", f"Scene {curr_idx + 1}"))
                edit_narration = st.text_area("Scene Narration / Description", value=curr_scene.get("narration", ""), height=100)
                edit_env = st.text_input("Scene Setting / Environment", value=curr_scene.get("environment", ""))

                save_scene_btn = st.form_submit_button("💾 Save Description & Mark Pending")
                if save_scene_btn:
                    curr_scene["heading"] = edit_heading.strip()
                    curr_scene["narration"] = edit_narration.strip()
                    curr_scene["environment"] = edit_env.strip()
                    curr_scene["image_prompt"] = f"{edit_narration.strip()[:200]}, set in {edit_env.strip()}."
                    curr_scene["illustration_status"] = "Pending"
                    curr_scene["illustration_error"] = None
                    if st.session_state.story_dir:
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                    st.success(f"Scene {curr_idx + 1} updated and marked as 'Pending'!")
                    st.rerun()

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Navigation Buttons (Previous, Regenerate, Next)
        b_col1, b_col2, b_col3 = st.columns([3, 4, 3])

        with b_col1:
            if st.button("⬅️ Previous Scene", disabled=(curr_idx == 0), use_container_width=True):
                st.session_state.current_scene_idx -= 1
                st.rerun()

        with b_col2:
            is_busy = st.session_state.get("is_generating_image", False)
            regen_btn = st.button(
                "🎨 Regenerate Illustration",
                disabled=is_busy,
                use_container_width=True,
                help="Generate illustration for this scene using the configured provider."
            )
            if regen_btn:
                st.session_state.is_generating_image = True
                curr_scene["illustration_status"] = "Generating"
                with st.spinner(f"Generating illustration for Scene {curr_idx + 1} via {selected_image_provider.upper()}..."):
                    try:
                        img_service = ImageGeneratorService(
                            provider=selected_image_provider,
                            api_key=active_api_key,
                            model_name=selected_image_model,
                            pollinations_api_key=active_pollinations_key,
                            pollinations_model=selected_pollinations_model
                        )
                        dest_file = os.path.join(
                            st.session_state.story_dir, f"scene_{scene_num:02d}.png"
                        )
                        new_img_path = img_service.generate_single_scene_image(
                            scene=curr_scene,
                            characters=story.get("characters", []),
                            art_style=story.get("art_style", "Colorful 2D cartoon"),
                            output_filepath=dest_file,
                            provider_override=selected_image_provider,
                            api_key_override=active_api_key,
                            pollinations_api_key_override=active_pollinations_key,
                            pollinations_model_override=selected_pollinations_model
                        )
                        # Update current scene image and status
                        curr_scene["image_path"] = new_img_path
                        curr_scene["illustration_status"] = "Generated"
                        curr_scene["illustration_error"] = None
                        
                        # Re-generate PDF to incorporate new image
                        pdf_svc = PDFStorybookGenerator()
                        pdf_svc.generate_pdf(story, st.session_state.pdf_path)
                        # Save updated story JSON
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.session_state.is_generating_image = False
                        st.success(f"Scene {curr_idx + 1} illustration updated!")
                        st.rerun()
                    except QuotaExceededError as q_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(q_err)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.error(f"⚠️ Quota Exceeded: {str(q_err)}")
                        if getattr(q_err, "is_zero_quota", False):
                            st.info(
                                "💡 **How to resolve:** Google AI Studio requires a linked Google Cloud Billing (Pay-As-You-Go) "
                                "project to use native Gemini image models. Switch the Image Generation Provider to "
                                "**'Pollinations AI'** in the sidebar settings (or click the button above) for zero-cost cartoon generation."
                            )
                        logger.warning(f"Regeneration quota error: {q_err}")
                    except InvalidAPIKeyError as key_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(key_err)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.error(f"🔑 Authentication Error: {str(key_err)}")
                        logger.warning(f"Regeneration auth error: {key_err}")
                    except UnsupportedModelError as model_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(model_err)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.error(f"🚫 Model Configuration Error: {str(model_err)}")
                        logger.warning(f"Regeneration model error: {model_err}")
                    except ImageNetworkError as net_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(net_err)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.error(f"🌐 Network Error: {str(net_err)}")
                        logger.warning(f"Regeneration network error: {net_err}")
                    except Exception as regen_err:
                        curr_scene["illustration_status"] = "Failed"
                        curr_scene["illustration_error"] = str(regen_err)
                        save_story_json(story, os.path.join(st.session_state.story_dir, "story.json"))
                        st.error(f"Failed to regenerate scene: {regen_err}")
                        logger.error(f"Regeneration error: {regen_err}")
                    finally:
                        st.session_state.is_generating_image = False

        with b_col3:
            if st.button("Next Scene ➡️", disabled=(curr_idx == total_scenes - 1), use_container_width=True):
                st.session_state.current_scene_idx += 1
                st.rerun()

        # Quick Jump Carousel / Thumbnail Row
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 📑 Jump to Scene")
        jump_cols = st.columns(total_scenes)
        for i, col in enumerate(jump_cols):
            with col:
                btn_type = "primary" if i == curr_idx else "secondary"
                if st.button(f"{i + 1}", key=f"jump_btn_{i}", type=btn_type, use_container_width=True):
                    st.session_state.current_scene_idx = i
                    st.rerun()

    # --- Right Sidebar / Column: Persistent Characters & Story Details ---
    with side_col:
        st.markdown("### 👥 Characters")
        st.caption("Visual attributes preserved across all scenes")

        for ch in story.get("characters", []):
            st.markdown(
                f"""
                <div class="character-card">
                    <div class="character-name">{ch.get('name', 'Character')}</div>
                    <div class="character-role">{ch.get('role', 'Character')}</div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 6px;">
                        <b>Visuals:</b> {ch.get('appearance', '')}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
                        <b>Traits:</b> {ch.get('personality', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 🎬 Scene Visual Prompt")
        with st.expander("Inspect AI Prompt", expanded=False):
            st.code(
                curr_scene.get("enriched_image_prompt", curr_scene.get("image_prompt", "")),
                language="text"
            )
            st.caption("Characters present: " + ", ".join(curr_scene.get("characters_present", ["None"])))
