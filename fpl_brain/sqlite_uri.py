"""Safe construction of read-only SQLite file URIs."""

from __future__ import annotations

from pathlib import Path


def readonly_sqlite_uri(path: str | Path) -> str:
    """Return an absolute SQLite file URI with its path component escaped.

    ``Path.as_uri`` percent-encodes URI-significant filename characters before
    the query string is appended, so characters such as ``#`` cannot turn
    ``mode=ro`` into a fragment or otherwise change the selected filename.
    """

    return f"{Path(path).expanduser().resolve().as_uri()}?mode=ro"
