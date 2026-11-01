"""
ToonTales AI - Voice Narration Service
Provides optional text-to-speech narration for cartoon story scenes.
Uses gTTS (Google Text-to-Speech) with graceful error handling and offline fallback.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ToonTalesAI.Narration")


class NarrationService:
    def __init__(self):
        self._gtts_available = False
        try:
            from gtts import gTTS
            self._gtts_available = True
            logger.info("Voice narration service initialized with gTTS engine.")
        except ImportError:
            logger.warning("gTTS not installed. Voice narration will be disabled unless installed.")

    def generate_narration(self, text: str, output_filepath: str, lang: str = "en") -> Optional[str]:
        """
        Converts scene narration text into an audio MP3 file.
        Non-blocking and completely failsafe: if network or audio generation fails,
        returns None and logs warning without disrupting the story.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to narration service.")
            return None

        if not self._gtts_available:
            logger.warning("Voice narration skipped: gTTS engine unavailable.")
            return None

        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from gtts import gTTS
            logger.info(f"Generating audio narration for '{text[:30]}...' -> {out_path.name}")
            tts = gTTS(text=text.strip(), lang=lang, slow=False)
            tts.save(str(out_path))

            if out_path.is_file() and out_path.stat().st_size > 0:
                logger.info(f"Audio narration saved successfully to {out_path.name}")
                return str(out_path)
            else:
                logger.warning(f"Audio file created but appears empty: {out_path.name}")
                return None

        except Exception as e:
            logger.warning(f"Voice narration generation failed (non-fatal): {e}")
            return None
