"""
ToonTales AI — Cost and Usage Tracking Service
Transparently logs provider calls, token usage, image counts, and reported/estimated costs.
Never displays misleading 'free' labels.
"""

import threading
import logging
from typing import List, Dict, Any, Optional
from services.providers.base import GenerationUsage

logger = logging.getLogger("ToonTalesAI.UsageTracker")

_LOCK = threading.Lock()
_USAGE_LOGS: List[Dict[str, Any]] = []


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

        # Safely handle non-numeric or mock inputs
        is_num_tokens = isinstance(tokens, (int, float)) and not isinstance(tokens, bool)
        clean_tokens = int(tokens) if is_num_tokens else 0

        is_num_images = isinstance(images_count, (int, float)) and not isinstance(images_count, bool)
        clean_images = int(images_count) if is_num_images else 0

        is_num_cost = isinstance(estimated_cost, (int, float)) and not isinstance(estimated_cost, bool)
        clean_cost = float(estimated_cost) if is_num_cost else None

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
            elif is_num_tokens and clean_tokens > 0:
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
        UsageTracker.record(
            provider=usage.provider,
            model=usage.model,
            operation=usage.operation,
            status=status,
            tokens=usage.total_tokens or ((usage.prompt_tokens or 0) + (usage.completion_tokens or 0)),
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
        total_stories, successful_stories, failed_stories, total_tokens, estimated_cost_usd.
        """
        with _LOCK:
            summary = create_usage_summary()
            summary["total_requests"] = len(_USAGE_LOGS)

            for entry in _USAGE_LOGS:
                op = str(entry.get("operation", "")).lower()
                status = str(entry.get("status", "")).lower()
                tokens = int(entry.get("tokens") or 0)
                images = int(entry.get("images") or 0)
                cost = float(entry.get("estimated_cost") or 0.0)

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


__all__ = ["UsageTracker", "create_usage_summary"]
