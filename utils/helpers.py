"""
ToonTales AI - Helper Utilities
Handles configuration, file paths, image validation, story persistence, and sanitization.
"""

import os
import re
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "generated_stories"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ToonTalesAI")


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve configuration from environment variables or Streamlit secrets.
    Never logs or exposes the secret value.
    """
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        return val.strip()

    # Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            sec_val = st.secrets[key]
            if sec_val is not None and str(sec_val).strip() != "":
                return str(sec_val).strip()
    except Exception:
        pass

    return default


def mask_secret(secret: Optional[str]) -> str:
    """
    Mask an API key for safe display/logging (e.g. AIza...4xQ).
    """
    if not secret:
        return "[NOT SET]"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"


def sanitize_filename(name: str) -> str:
    """
    Sanitize text to create clean, filesystem-safe directory/file names.
    """
    s = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip().replace(' ', '_')
    return s[:40] if s else "story"


def create_story_workspace(story_title: str = "story") -> Dict[str, Any]:
    """
    Creates a unique directory for a new story session.
    Returns directory path and unique story ID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    clean_title = sanitize_filename(story_title)
    story_id = f"story_{timestamp}_{clean_title}_{unique_suffix}"

    story_dir = GENERATED_DIR / story_id
    story_dir.mkdir(parents=True, exist_ok=True)

    return {
        "story_id": story_id,
        "story_dir": str(story_dir),
        "json_path": str(story_dir / "story.json"),
        "pdf_path": str(story_dir / "story.pdf")
    }


def save_story_json(story_data: Dict[str, Any], filepath: str) -> None:
    """
    Saves the structured story data to disk as JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(story_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Story data saved to {path.name}")


def load_story_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads story data from JSON file if it exists.
    """
    path = Path(filepath)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load story JSON from {filepath}: {e}")
        return None


def validate_image_file(image_path: str) -> bool:
    """
    Validates that a generated image file exists, is readable, and is a valid image.
    """
    p = Path(image_path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        with Image.open(p) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Image validation failed for {image_path}: {e}")
        return False
