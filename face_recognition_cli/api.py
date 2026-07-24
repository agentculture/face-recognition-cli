"""Stable public consumer API — the surface reachy-mini-cli imports.

:mod:`face_recognition_cli` exposes exactly one thing at its root beyond
``__version__``: :func:`build_face_recognition`. It exists because
reachy-mini-cli's ``reachy/behavior/face_sense.py::build_face_recognition``
(the function this repo's engine and store were extracted *from* — see
``reachy/vision/face.py`` + ``face_store.py``) is going to delegate to this
one instead of constructing ``FaceEngine``/``FaceStore`` itself. The call
shape is identical by design: the same keyword-only ``models_dir=`` /
``store_base_dir=`` pair, the same ``tuple[engine, store] | None`` return, the
same "probe before import, never raise" contract. A caller that already knows
reachy's ``build_face_recognition`` needs to learn nothing new to call this
one.

Degradation contract, mirrored from reachy's ``_VISION_WARNED`` pattern:

* Probe FIRST, via :func:`_cv2_available` (``importlib.util.find_spec``) —
  before any lazy import of the engine/store construction path. Merely
  importing this module never imports ``cv2``.
* ``cv2`` absent: log exactly ONE process-wide warning naming the fix
  (``pip install 'face-recognition-cli[cpu]'``) and return ``None``. Never
  raise. The warning is gated by a module-level :data:`_WARNED` latch —
  module-level, not per-call or per-instance, because the extra's absence is
  a property of the *process*, not of any one caller.
* ``cv2`` present but construction fails (a broken vision stack — missing
  model files, a corrupt install, ...): the same one-warning-then-``None``
  contract applies. A broken stack disables the feature; it is not this
  function's job to raise for it.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never imports cv2
    from face_recognition_cli.engine import FaceEngine
    from face_recognition_cli.store import FaceStore

logger = logging.getLogger(__name__)

#: Process-wide latch for the missing-``[cpu]``-extra (or broken-stack)
#: warning. Module-level rather than per-call because the extra's absence (or
#: a broken vision stack) is a property of the process, not of any one
#: caller — mirrors reachy-mini-cli's ``face_sense._VISION_WARNED``.
_WARNED = False


def _cv2_available() -> bool:
    """Whether ``cv2`` (opencv) is importable, without importing it.

    The probe half of the degradation contract — a pure
    ``importlib.util.find_spec`` check, no side effect. Exposed as its own
    function (rather than inlined) so tests can monkeypatch this single seam
    to simulate ``[cpu]``/``[gpu]`` being installed without needing a real
    opencv wheel.
    """
    return importlib.util.find_spec("cv2") is not None


def build_face_recognition(
    *,
    models_dir: Path | None = None,
    store_base_dir: Path | None = None,
) -> tuple["FaceEngine", "FaceStore"] | None:
    """Build ``(engine, store)``, or ``None`` when the vision stack is unavailable.

    Returns ``None`` — after **exactly one** process-wide logged warning —
    when the ``[cpu]``/``[gpu]`` extra (opencv) is absent, which is the
    default state of a bare install. A ``None`` return is not an error: a
    caller is expected to keep running with the feature disabled.

    The imports are lazy and happen only after the probe succeeds, so this
    module — and merely calling this function on a bare install — never
    imports ``cv2``. ``models_dir`` / ``store_base_dir`` are passed through to
    :class:`~face_recognition_cli.engine.FaceEngine` /
    :class:`~face_recognition_cli.store.FaceStore` only when given, so their
    own defaults apply otherwise.
    """
    global _WARNED  # one process-wide warning latch, by design

    if not _cv2_available():
        if not _WARNED:
            _WARNED = True
            logger.warning(
                "face recognition needs the [cpu] (or [gpu]) extra (opencv); the feature "
                "stays unavailable (install: pip install 'face-recognition-cli[cpu]')"
            )
        return None

    try:
        from face_recognition_cli.engine import FaceEngine
        from face_recognition_cli.store import FaceStore

        engine = FaceEngine(models_dir=models_dir) if models_dir is not None else FaceEngine()
        store = FaceStore(base_dir=store_base_dir) if store_base_dir is not None else FaceStore()
    except Exception:  # a broken vision stack disables the feature, nothing more
        if not _WARNED:
            _WARNED = True
            logger.warning(
                "face recognition unavailable; the feature stays disabled", exc_info=True
            )
        return None
    return (engine, store)
