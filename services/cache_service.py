"""
ToonTales AI — Request Cache & Deduplication Service
Prevents redundant API calls, saves token costs, and handles in-flight request deduplication.
"""

import hashlib
import json
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("ToonTalesAI.CacheService")

_LOCK = threading.Lock()
_STORY_CACHE: Dict[str, Dict[str, Any]] = {}
_IMAGE_CACHE: Dict[str, str] = {}  # prompt_hash -> image_file_path
_CACHE_HITS: Dict[str, int] = {"story_hits": 0, "image_hits": 0}


class RequestCache:
    """Thread-safe request-level cache and deduplication service."""

    @staticmethod
    def compute_story_key(
        story_idea: str,
        genre: str,
        target_age: str,
        story_length: str,
        art_style: str,
        num_scenes: int,
        ending_style: str,
        provider: str,
        model: Optional[str] = None
    ) -> str:
        """Computes a deterministic SHA-256 fingerprint for story generation inputs."""
        raw = f"{story_idea.strip().lower()}|{genre.lower()}|{target_age.lower()}|{story_length.lower()}|{art_style.lower()}|{num_scenes}|{ending_style.lower()}|{provider.lower()}|{(model or '').lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_image_key(prompt: str, art_style: str, provider: str, model: Optional[str] = None) -> str:
        """Computes a deterministic SHA-256 fingerprint for image generation inputs."""
        raw = f"{prompt.strip().lower()}|{art_style.lower()}|{provider.lower()}|{(model or '').lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def get_cached_story(key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached story if available, making a deep copy to prevent mutation."""
        with _LOCK:
            if key in _STORY_CACHE:
                _CACHE_HITS["story_hits"] += 1
                logger.info(f"Cache hit: Reusing existing story generation for key {key[:8]}...")
                # Return deep copy via json serialize/deserialize
                return json.loads(json.dumps(_STORY_CACHE[key]))
        return None

    @staticmethod
    def cache_story(key: str, story_dict: Dict[str, Any]) -> None:
        """Stores a successfully generated story in cache."""
        with _LOCK:
            _STORY_CACHE[key] = json.loads(json.dumps(story_dict))
            logger.info(f"Story cached under key {key[:8]} (total cached stories: {len(_STORY_CACHE)})")

    @staticmethod
    def get_cached_image(key: str) -> Optional[str]:
        """Retrieves path to previously generated image for identical prompt/style/provider."""
        import os
        with _LOCK:
            path = _IMAGE_CACHE.get(key)
            if path and os.path.isfile(path) and os.path.getsize(path) > 0:
                _CACHE_HITS["image_hits"] += 1
                logger.info(f"Cache hit: Reusing existing image at {path} for key {key[:8]}...")
                return path
        return None

    @staticmethod
    def cache_image(key: str, image_path: str) -> None:
        """Stores a generated image path under its prompt hash."""
        with _LOCK:
            _IMAGE_CACHE[key] = image_path

    @staticmethod
    def get_cache_stats() -> Dict[str, int]:
        """Returns statistics on cache usage and hits."""
        with _LOCK:
            return {
                "cached_stories": len(_STORY_CACHE),
                "cached_images": len(_IMAGE_CACHE),
                "story_hits": _CACHE_HITS["story_hits"],
                "image_hits": _CACHE_HITS["image_hits"],
                "total_hits": _CACHE_HITS["story_hits"] + _CACHE_HITS["image_hits"]
            }

    @staticmethod
    def reset() -> None:
        """Clears all caches (used during testing)."""
        with _LOCK:
            _STORY_CACHE.clear()
            _IMAGE_CACHE.clear()
            _CACHE_HITS["story_hits"] = 0
            _CACHE_HITS["image_hits"] = 0

