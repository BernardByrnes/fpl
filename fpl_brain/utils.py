"""Small shared helpers with no FPL-specific knowledge."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def safe_get(
    value: Any,
    key: str,
    default: T | None = None,
    cast: Callable[[Any], T] | None = None,
) -> T | None:
    """Read a mapping defensively and optionally coerce its value.

    Missing and explicit null values both return ``default``.  A failed cast
    returns ``None`` and is logged at DEBUG so malformed upstream values never
    take down an ingest.
    """

    if not isinstance(value, dict) or key not in value or value[key] is None:
        return default
    result = value[key]
    if cast is None:
        return result  # type: ignore[return-value]
    try:
        return cast(result)
    except (TypeError, ValueError, OverflowError) as exc:
        LOGGER.debug("Could not cast key %s value %r: %s", key, result, exc)
        return None


def to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(float(value))


def to_float(value: Any) -> float:
    return float(value)


def to_text(value: Any) -> str:
    return str(value)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def flag(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if to_bool(value) else 0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def subtract_days(value: str, days: int | float) -> str:
    parsed = parse_utc(value) or datetime.now(timezone.utc)
    return (parsed - timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalise_name(value: str | None) -> str:
    """Accent-strip, casefold, collapse whitespace, and remove punctuation."""

    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = without_marks.casefold()
    no_punctuation = re.sub(r"[^\w\s]", " ", folded, flags=re.UNICODE)
    return " ".join(no_punctuation.split())


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(raw_dir: str | Path, verbose: bool = False, quiet: bool = False) -> None:
    """Configure stderr logging and the required append-only fetch log."""

    ensure_directory(raw_dir)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO))
    root.addHandler(stderr_handler)

    file_handler = logging.FileHandler(Path(raw_dir) / "fetch.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

