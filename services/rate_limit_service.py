"""
ToonTales AI — Rate Limiting, Cooldown & Concurrency Lock Service
Manages provider cooldowns, parses Retry-After headers, and prevents
duplicate scene generation requests.
"""

import time
import threading
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("ToonTalesAI.RateLimitService")

_LOCK = threading.Lock()
_COOLDOWNS: Dict[str, float] = {}  # provider -> timestamp when cooldown expires
_ACTIVE_GENERATIONS: Dict[Tuple[str, int], float] = {}  # (story_id, scene_num) -> timestamp started


class RateLimitService:
    """Thread-safe rate limit, cooldown, and duplicate request manager."""

    @staticmethod
    def record_cooldown(provider: str, duration_seconds: int = 30) -> None:
        """Records a cooldown duration for a specific provider."""
        expiry = time.time() + max(1, duration_seconds)
        with _LOCK:
            _COOLDOWNS[provider.lower()] = expiry
        logger.info(f"Recorded cooldown for provider '{provider}': {duration_seconds}s (expires in {duration_seconds}s)")

    @staticmethod
    def get_cooldown_remaining(provider: str) -> int:
        """Returns the number of seconds remaining on a cooldown, or 0 if clear."""
        with _LOCK:
            expiry = _COOLDOWNS.get(provider.lower(), 0)
        remaining = int(expiry - time.time())
        return max(0, remaining)

    @staticmethod
    def is_in_cooldown(provider: str) -> bool:
        """Checks if a provider is currently in cooldown."""
        return RateLimitService.get_cooldown_remaining(provider) > 0

    @staticmethod
    def acquire_scene_lock(story_id: str, scene_number: int, timeout_seconds: int = 120) -> bool:
        """
        Attempts to acquire a generation lock for a specific scene.
        Returns True if acquired, False if generation for this scene is already active.
        Auto-releases stale locks older than timeout_seconds.
        """
        key = (story_id, scene_number)
        now = time.time()
        with _LOCK:
            if key in _ACTIVE_GENERATIONS:
                started_at = _ACTIVE_GENERATIONS[key]
                if now - started_at < timeout_seconds:
                    logger.warning(f"Duplicate request blocked: Scene #{scene_number} of {story_id} is already generating.")
                    return False
                else:
                    logger.warning(f"Releasing stale lock for Scene #{scene_number} of {story_id} (> {timeout_seconds}s old).")
            _ACTIVE_GENERATIONS[key] = now
            return True

    @staticmethod
    def release_scene_lock(story_id: str, scene_number: int) -> None:
        """Releases the generation lock for a scene."""
        key = (story_id, scene_number)
        with _LOCK:
            _ACTIVE_GENERATIONS.pop(key, None)

    @staticmethod
    def is_scene_generating(story_id: str, scene_number: int) -> bool:
        """Checks if a scene generation is currently active."""
        key = (story_id, scene_number)
        with _LOCK:
            return key in _ACTIVE_GENERATIONS

    @staticmethod
    def reset() -> None:
        """Resets all cooldowns and active locks (for testing and session clear)."""
        with _LOCK:
            _COOLDOWNS.clear()
            _ACTIVE_GENERATIONS.clear()
