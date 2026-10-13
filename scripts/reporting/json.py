"""Shared JSON artifact serialization used by experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
    trailing_newline: bool = True,
) -> Path:
    """Write a JSON artifact while preserving the caller's existing layout."""
    output = Path(path)
    text = json.dumps(value, indent=indent)
    if trailing_newline:
        text += "\n"
    output.write_text(text, encoding="utf-8")
    return output
