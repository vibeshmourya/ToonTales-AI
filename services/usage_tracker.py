"""
ToonTales AI — Cost and Usage Tracking Service
Transparently logs provider calls, token usage, image counts, and reported/estimated costs.
Never displays misleading 'free' labels. Resilient to missing fields and non-numeric values.
"""

import threading
import logging
from typing import List, Dict, Any, Optional
from services.providers.base import GenerationUsage

logger = logging.getLogger("ToonTalesAI.UsageTracker")

_LOCK = threading.Lock()
_USAGE_LOGS: List[Dict[str, Any]] = []


def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts an arbitrary value to int, avoiding ValueErrors."""
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts an arbitrary value to float, avoiding ValueErrors."""
    if val is None or isinstance(val, bool):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def create_usage_summary() -> Dict[str, Any]:
    """
    Creates and returns a standardized, zero-initialized usage summary dictionary.
    Guarantees that all required metrics exist with predictable types and safe defaults.
    """
    return {
        "total_requests": 0,
        "total_images": 0,
        "successful_images": 0,
        "failed_images": 0,
        "total_stories": 0,
        "successful_stories": 0,
        "failed_stories": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cache_hits": 0,
    }


class UsageTracker:
    """Lightweight thread-safe session-level usage and cost tracking service."""

    create_usage_summary = staticmethod(create_usage_summary)

    @staticmethod
    def record(
        provider: str,
        model: str,
        operation: str,
        status: str = "success",
        tokens: Optional[int] = None,
        images_count: int = 0,
        estimated_cost: Optional[float] = None,
        cost_type: str = "unknown"
    ) -> None:
        from datetime import datetime

        prov_clean = (provider or "UNKNOWN").upper()
        op_clean = (operation or "UNKNOWN").capitalize()
        status_clean = (status or "success").capitalize()

        clean_tokens = safe_int(tokens, default=0)
        clean_images = safe_int(images_count, default=0)
        clean_cost = safe_float(estimated_cost, default=-1.0)
        if clean_cost < 0:
            clean_cost = None

        # Automatic cost estimation fallback if not explicitly provided
        final_cost = clean_cost
        final_cost_type = cost_type
        if final_cost is None:
            if prov_clean == "POLLINATIONS":
                final_cost = 0.0
                final_cost_type = "zero_cost"
            elif prov_clean == "OPENAI" and op_clean == "Image":
                final_cost = 0.04 * clean_images if "dall-e-3" in str(model).lower() else 0.02 * clean_images
                final_cost_type = "estimated"
            elif clean_tokens > 0:
                # Approximate 0.50 USD per 1M tokens as conservative default if unknown
                final_cost = round(clean_tokens * 0.0000005, 6)
                final_cost_type = "estimated"
            else:
                final_cost = 0.0

        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "provider": prov_clean,
            "model": model or "default",
            "operation": op_clean,
            "status": status_clean,
            "tokens": clean_tokens,
            "images": clean_images,
            "estimated_cost": final_cost,
            "cost_type": final_cost_type
        }
        with _LOCK:
            _USAGE_LOGS.append(entry)
        logger.info(f"Usage logged: [{entry['provider']}] {entry['operation']} with {entry['model']} (Status: {entry['status']})")

    @staticmethod
    def record_usage_obj(usage: Optional[GenerationUsage], status: str = "success") -> None:
        if not usage:
            return
        tokens_val = usage.total_tokens
        if tokens_val is None:
            tokens_val = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        UsageTracker.record(
            provider=usage.provider,
            model=usage.model,
            operation=usage.operation,
            status=status,
            tokens=tokens_val,
            images_count=usage.images_generated,
            estimated_cost=usage.estimated_cost_usd,
            cost_type=usage.cost_type
        )

    @staticmethod
    def get_logs() -> List[Dict[str, Any]]:
        with _LOCK:
            return list(_USAGE_LOGS)

    @staticmethod
    def reset() -> None:
        """Clears all logged usage entries (primarily for testing and session isolation)."""
        with _LOCK:
            _USAGE_LOGS.clear()

    @staticmethod
    def get_summary() -> Dict[str, Any]:
        """
        Calculates and returns the aggregated usage summary.
        Guarantees presence of all standard fields:
        total_requests, total_images, successful_images, failed_images,
        total_stories, successful_stories, failed_stories, total_tokens, estimated_cost_usd, cache_hits.
        """
        with _LOCK:
            summary = create_usage_summary()
            summary["total_requests"] = len(_USAGE_LOGS)

            try:
                from services.cache_service import RequestCache
                summary["cache_hits"] = RequestCache.get_cache_stats().get("total_hits", 0)
            except Exception:
                summary["cache_hits"] = 0

            for entry in _USAGE_LOGS:
                if not isinstance(entry, dict):
                    continue

                op = str(entry.get("operation", "")).lower()
                status = str(entry.get("status", "")).lower()
                tokens = safe_int(entry.get("tokens"), 0)
                images = safe_int(entry.get("images"), 0)
                cost = safe_float(entry.get("estimated_cost"), 0.0)

                summary["total_tokens"] += tokens
                summary["estimated_cost_usd"] += cost

                if op == "story":
                    summary["total_stories"] += 1
                    if status == "success":
                        summary["successful_stories"] += 1
                    else:
                        summary["failed_stories"] += 1
                elif op == "image":
                    if status == "success":
                        summary["successful_images"] += images
                        summary["total_images"] += images
                    else:
                        summary["failed_images"] += 1

            return summary


__all__ = ["UsageTracker", "create_usage_summary", "safe_int", "safe_float"]
