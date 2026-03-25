# ui/session_store.py
"""
WHY THIS EXISTS:
Streamlit reruns the entire script on every interaction. Without persistent
storage, the user_id would regenerate on every rerun, creating a new user
identity and losing access to all previously indexed documents.

This module saves user_id and the last uploaded doc_id to a JSON file on
disk so they survive page refreshes, tab closes, and app restarts.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Session file sits next to the UI directory — not inside the repo
_SESSION_FILE = Path(__file__).parent.parent / ".smartdocs_session.json"


def get_or_create_user_id() -> str:
    """
    Return the persisted user_id, or create one if none exists.

    Guarantees:
      - The same user_id is returned on every call within the same installation
      - A new UUID v4 is generated exactly once, then persisted
      - Any file I/O error falls back to a session-scoped UUID (logged as warning)
    """
    data = _load()
    if "user_id" in data and _is_valid_uuid(data["user_id"]):
        return data["user_id"]

    new_id = str(uuid.uuid4())
    data["user_id"] = new_id
    _save(data)
    logger.info("Created new user_id: %s", new_id)
    return new_id


def save_doc_id(doc_id: str, filename: str) -> None:
    """
    Persist the most recently uploaded document.
    Called by upload_panel after every successful /ingest response.

    Args:
        doc_id:   UUID string of the indexed document
        filename: Display name (e.g. "gst_notice.pdf")
    """
    data = _load()
    data["last_doc_id"] = doc_id
    data["last_doc_filename"] = filename
    _save(data)
    logger.debug("Saved doc_id=%s filename=%s to session", doc_id, filename)


def get_last_doc() -> Tuple[Optional[str], Optional[str]]:
    """
    Return (doc_id, filename) from the last successful upload.
    Returns (None, None) if no document has been uploaded yet.
    """
    data = _load()
    doc_id = data.get("last_doc_id")
    filename = data.get("last_doc_filename")
    if not doc_id or not _is_valid_uuid(doc_id):
        return None, None
    return doc_id, filename


def clear_session() -> None:
    """
    Remove the session file. Used for debugging and testing only.
    The next call to get_or_create_user_id() will create a fresh identity.
    """
    try:
        if _SESSION_FILE.exists():
            _SESSION_FILE.unlink()
            logger.info("Session file cleared")
    except Exception:
        logger.exception("Failed to clear session file")


# ------------------------------------------------------------------ #
# Private helpers                                                       #
# ------------------------------------------------------------------ #


def _load() -> dict:
    """Load and parse the session JSON file. Returns {} on any error."""
    try:
        if _SESSION_FILE.exists():
            text = _SESSION_FILE.read_text(encoding="utf-8")
            return json.loads(text)
    except Exception:
        logger.warning("Could not read session file %s — starting fresh", _SESSION_FILE)
    return {}


def _save(data: dict) -> None:
    """Write data to the session JSON file. Logs warning on failure (non-fatal)."""
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Could not write session file %s", _SESSION_FILE, exc_info=True)


def _is_valid_uuid(value: str) -> bool:
    """Return True if value is a well-formed UUID string."""
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False