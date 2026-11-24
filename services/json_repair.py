"""
ToonTales AI — Local JSON Repair Utility
Cleans, normalizes, and repairs LLM-generated JSON strings locally without
re-calling the provider API for minor syntax flaws or formatting issues.
"""

import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ToonTalesAI.JSONRepair")


def repair_json_string(raw_text: str) -> str:
    """
    Cleans and repairs common syntax issues in JSON strings returned by LLMs:
    - Strips markdown code blocks (```json ... ```)
    - Extracts the outermost JSON object between first '{' and last '}'
    - Removes trailing commas before '}' or ']'
    - Cleans non-printable ASCII control characters
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot parse empty JSON text.")

    text = raw_text.strip()

    # 1. Strip leading and trailing markdown code block formatting
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 2. Extract outermost JSON object { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    else:
        # Check if wrapped in array [ ... ]
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            text = text[first_bracket:last_bracket + 1]

    # 3. Remove trailing commas before closing braces/brackets
    # e.g., {"a": 1, } -> {"a": 1} or [1, 2, ] -> [1, 2]
    text = re.sub(r",\s*([\}\]])", r"\1", text)

    # 4. Clean stray control characters (keep \r, \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text


def parse_and_repair_json(raw_text: str) -> Dict[str, Any]:
    """
    Parses a raw text string into a Python dictionary with local fault tolerance.
    Attempts standard json.loads first, falling back to local repair rules.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Received empty response from AI provider.")

    # 1. Direct parse attempt
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Repaired string attempt
    cleaned = repair_json_string(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        logger.warning(f"Standard JSON repair failed: {err}. Attempting aggressive normalization...")

    # 3. Aggressive repair: fix unescaped double quotes inside values, newlines, etc.
    try:
        # Normalize smart quotes to standard quotes
        normalized = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # Remove trailing commas again
        normalized = re.sub(r",\s*([\}\]])", r"\1", normalized)
        return json.loads(normalized)
    except json.JSONDecodeError as final_err:
        logger.error(f"Failed to parse JSON even after aggressive repair: {final_err}")
        raise ValueError(f"Invalid JSON returned by provider: {final_err}") from final_err

