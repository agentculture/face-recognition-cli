"""State-dir + per-consumer bank layout.

This module owns exactly one thing: where on disk face-recognition-cli's
mutable state lives, and how that space is partitioned. It is pure standard
library — no ``numpy``, no ``cv2``, nothing third-party — so it stays
importable on a bare install with neither the ``[cpu]`` nor the ``[gpu]``
extra. The engine and the store build on top of this module; they do not
duplicate its resolution logic.

Layout on disk::

    <state dir>/
        banks/
            default/      # faces.json + embeddings/*.npy for the default bank
            reachy/        # a different consumer's identities, fully isolated
            colleague/     # another consumer, also fully isolated
        models/            # shared ONNX models (YuNet + SFace), bank-independent

**Why banks exist.** Every consumer of this package — reachy-mini-cli, a
mesh colleague agent, an interactive operator — enrolls its own set of faces.
Those identities must never leak across consumers: an id enrolled for reachy
must be invisible to ``list``/``match``/``forget`` run against colleague's
bank, and vice versa. ``bank_dir()`` gives each consumer its own subtree under
``banks/`` so the store (which owns writing ``faces.json`` and the embedding
``.npy`` files) can treat each bank as a completely separate index with zero
extra logic — bank isolation is pure path composition above the store, not a
property the store itself has to implement.

**Why models are bank-independent.** The YuNet detector and SFace embedder
ONNX files are identical for every bank — banks partition *identities*, not
*models*. The SFace file alone is ~37 MB; duplicating it per bank would waste
disk for no isolation benefit, since two banks embedding through the same
model still cannot compare across each other (the store, not the model, is
what keeps them apart). ``models_dir()`` therefore lives one level above
``banks/``, shared by every bank.

**Resolution chain**, mirroring the pattern reachy-mini-cli's own
``reachy/daemon.py:state_dir()`` uses for its daemon bookkeeping (this module
does not import that one — reachy-mini-cli is a sibling repo, not a
dependency — it only follows the same convention so operators who know one
tool already know the other):

1. ``$FACE_RECOGNITION_STATE_DIR`` — overrides everything; tests use this for
   isolation from the real home directory.
2. ``$XDG_STATE_HOME/face-recognition-cli``.
3. ``~/.local/state/face-recognition-cli`` — the XDG default when neither
   override is set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from face_recognition_cli.cli._errors import EXIT_USER_ERROR, CliError

#: The shape a bank name must take. Bank names become path segments under
#: ``<state dir>/banks/``, so anything that could traverse out of that tree
#: (separators, ``..``, absolute paths, drive prefixes) is rejected outright —
#: the leading alphanumeric also rules out dot-led names like ``..``.
_BANK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The bank name used when no explicit bank is requested and no
#: ``$FACE_RECOGNITION_BANK`` override is set.
DEFAULT_BANK = "default"

_STATE_DIR_ENV = "FACE_RECOGNITION_STATE_DIR"
_XDG_STATE_HOME_ENV = "XDG_STATE_HOME"
_BANK_ENV = "FACE_RECOGNITION_BANK"
_APP_DIR_NAME = "face-recognition-cli"


def state_dir() -> Path:
    """Return (and create) the per-user state dir for this package.

    ``$FACE_RECOGNITION_STATE_DIR`` overrides everything (tests use it for
    isolation); otherwise ``$XDG_STATE_HOME/face-recognition-cli`` or
    ``~/.local/state/face-recognition-cli``. The directory is created
    (``mkdir(parents=True, exist_ok=True)``) before it is returned, matching
    reachy-mini-cli's ``daemon.py:state_dir()`` convention.
    """
    override = os.environ.get(_STATE_DIR_ENV)
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get(_XDG_STATE_HOME_ENV)
        base = (
            Path(xdg) / _APP_DIR_NAME if xdg else Path.home() / ".local" / "state" / _APP_DIR_NAME
        )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _validate_bank_name(name: str) -> str:
    """Return *name* if it is a safe bank name; raise a clean user error if not.

    A bank name is a single path segment under ``banks/`` — validating the
    shape here (rather than resolving-and-containing later) is what keeps
    :func:`bank_dir` unable to escape the state tree.
    """
    if not _BANK_NAME_RE.fullmatch(name):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"invalid bank name: {name!r}",
            remediation=(
                "bank names are single path segments: letters, digits, '.', '_' or '-', "
                "starting with a letter or digit (no separators, no '..')"
            ),
        )
    return name


def resolve_bank(name: str | None = None) -> str:
    """Resolve which bank to use.

    Precedence: an explicit ``name`` argument wins; otherwise
    ``$FACE_RECOGNITION_BANK``; otherwise :data:`DEFAULT_BANK`. Whichever
    source supplied the name, it is validated — a bank name is a path segment
    under ``banks/``, so traversal shapes are a clean exit-1 error, never a
    path.
    """
    if name:
        return _validate_bank_name(name)
    env = os.environ.get(_BANK_ENV)
    if env:
        return _validate_bank_name(env)
    return DEFAULT_BANK


def bank_dir(name: str | None = None) -> Path:
    """Return the directory for a bank, without creating it.

    The bank is resolved via :func:`resolve_bank`. Unlike :func:`state_dir`,
    this does **not** create the directory — the store owns building its own
    tree (``faces.json`` + ``embeddings/``) the first time it writes.
    """
    return state_dir() / "banks" / resolve_bank(name)


def models_dir() -> Path:
    """Return the shared, bank-independent directory for ONNX models.

    Every bank embeds through the same YuNet + SFace models, so this path
    never varies with the current bank — it lives directly under
    :func:`state_dir`, one level above ``banks/``.
    """
    return state_dir() / "models"
