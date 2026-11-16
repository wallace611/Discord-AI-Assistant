"""Helpers for cleaning up raw local-model output."""

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks that reasoning models may emit before the answer."""
    return _THINK_BLOCK.sub("", text).strip()


def extract_json_object(raw: str) -> dict[str, Any]:
    """Return the JSON object embedded in ``raw``.

    Tolerates reasoning blocks, code fences and surrounding prose. Raises
    ``ValueError`` when no valid object is present.
    """
    text = strip_reasoning(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON value is not an object")
    return data
