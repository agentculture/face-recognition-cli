"""Frame loading shared by the image-consuming verbs (``enroll`` / ``match``).

One helper, :func:`load_frame`, turns the ``--image`` argument both verbs take
into the BGR ``numpy`` array :meth:`face_recognition_cli.engine.FaceEngine.detect`
expects. Two input forms, per the frame decision recorded in the spec:

* a **path** to an encoded image file, or
* ``-`` — **raw encoded bytes on stdin**.

The ``-`` form is what keeps this tool out of the capture business. It owns no
camera and opens no device; a producer writing an encoded still to stdout —
webcam-cli / media-cli style — composes with it over a plain pipe::

    webcam-cli capture --format png - | face-recognition enroll --name ada --image -

so capture (their lane) and interpretation (this lane) stay separate processes
with a byte stream between them.

**Diagnosis order is deliberate.** The bytes are read, and checked for
emptiness, *before* ``cv2`` is probed. A typo'd path or an empty pipe is a user
error on every install, so it reports as exit 1 ("check the path") rather than
exit 2 ("install the [cpu] extra") on a box that merely happens to lack
OpenCV — the exit-2 answer would be true but useless, and would send the
operator installing a 60 MB wheel to fix a typo. Once there ARE bytes to
decode, a missing extra is the honest blocker and surfaces as the engine's own
exit-2 :class:`~face_recognition_cli.cli._errors.CliError`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from face_recognition_cli.cli._errors import EXIT_USER_ERROR, CliError

# Private to the engine module, but this is the same package: the cv2 probe is
# deliberately SHARED rather than reimplemented, so a missing [cpu]/[gpu] extra
# produces exactly one error shape (exit 2, naming the extra) no matter which
# surface reaches it first — the CLI's decode step or the engine's model load.
from face_recognition_cli.engine import _import_cv2

#: The ``--image`` value that means "read encoded bytes from stdin".
STDIN_SENTINEL = "-"

#: Refuse image payloads beyond this many bytes, from stdin or a file. Real
#: encoded stills are megabytes at most; the cap exists so a runaway producer
#: piped into ``--image -`` (or a mistakenly named huge file) fails fast with
#: a clean error instead of exhausting memory. Generous by design.
MAX_IMAGE_BYTES = 64 * 1024 * 1024

_TOO_LARGE_HINT = (
    "the input exceeds the 64 MiB image limit; pass a single encoded still, "
    "not a stream or an unrelated large file"
)

_SUPPORTED = "png, jpeg, bmp, webp — anything the local OpenCV build decodes"

_STDIN_HINT = f"pipe an encoded image ({_SUPPORTED}) on stdin, or pass --image <path>"
_PATH_HINT = "check the path exists and is a readable file, or pass --image - to read stdin"


def _read_stdin_bytes() -> bytes:
    """Read every byte available on stdin, as bytes (never text)."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="stdin is not available as a byte stream",
            remediation=_STDIN_HINT,
        )
    try:
        data = buffer.read(MAX_IMAGE_BYTES + 1)
    except OSError as err:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="could not read image bytes from stdin",
            remediation=_STDIN_HINT,
        ) from err
    if len(data) > MAX_IMAGE_BYTES:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="image data on stdin is too large",
            remediation=_TOO_LARGE_HINT,
        )
    return data


def _read_file_bytes(image: str) -> bytes:
    """Read an encoded image file whole; a bad path is a user error."""
    path = Path(image)
    try:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"image file is too large: {image}",
                remediation=_TOO_LARGE_HINT,
            )
        return path.read_bytes()
    except OSError as err:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"could not read image file: {image}",
            remediation=_PATH_HINT,
        ) from err


def load_frame(image: str) -> np.ndarray:
    """Load *image* (a path, or ``-`` for stdin) into a decoded BGR frame.

    Raises :class:`CliError` with code 1 for anything the caller can fix
    (missing/unreadable path, empty stdin, bytes that are not an image) and
    code 2 — the engine's own error — when the ``[cpu]``/``[gpu]`` extra is
    not installed.
    """
    data = _read_stdin_bytes() if image == STDIN_SENTINEL else _read_file_bytes(image)
    if not data:
        source = "no bytes arrived on stdin" if image == STDIN_SENTINEL else f"{image} is empty"
        raise CliError(
            code=EXIT_USER_ERROR,
            message="could not decode image data",
            remediation=f"{source}; expected an encoded image ({_SUPPORTED})",
        )

    cv2 = _import_cv2()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="could not decode image data",
            remediation=f"expected an encoded image ({_SUPPORTED}), not raw pixels or text",
        )
    return frame
