"""
ToonTales AI — Narration Service Architecture Adapter
Re-exports NarrationService as VoiceNarrationService for modular provider architecture.
"""

from services.narration import NarrationService

VoiceNarrationService = NarrationService

__all__ = ["NarrationService", "VoiceNarrationService"]
