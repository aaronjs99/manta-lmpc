"""Shared JSON artifact serialization used by experiment runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
    trailing_newline: bool = True,
    atomic: bool = False,
) -> Path:
    """Write a JSON artifact while preserving the caller's existing layout."""
    output = Path(path)
    text = json.dumps(value, indent=indent)
    if trailing_newline:
        text += "\n"
    if not atomic:
        output.write_text(text, encoding="utf-8")
        return output
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
