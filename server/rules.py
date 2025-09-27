from __future__ import annotations

from typing import Any, Dict, Tuple


def apply_sync_rules(evt: Dict[str, Any]) -> Tuple[str | None, str]:
    """Return (mutated_text, reason). None means NOOP.

    Keep this fast (<200ms). Extend your own logic here.
    """
    payload = evt.get("payload") or {}
    text = payload.get("text") or ""
    if not text:
        return None, "NO_TEXT"

    # Example: normalize whitespace
    mutated = " ".join(text.split())

    if mutated != text:
        return mutated, "normalize-whitespace"

    return None, "NOOP"


