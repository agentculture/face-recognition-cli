"""face-recognition-cli — agent-first CLI for an AgentCulture mesh agent."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from face_recognition_cli.api import build_face_recognition

try:
    __version__ = _pkg_version("face-recognition-cli")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = ["__version__", "build_face_recognition"]
