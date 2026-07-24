"""``face-recognition-cli match`` — who is the face in this image?

Detects the largest face in ``--image`` (a path, or ``-`` for encoded bytes on
stdin), embeds it, and compares that embedding by cosine similarity against
every identity enrolled in the selected bank. Read-only: nothing is written,
and an unmatched face is never enrolled as a side effect.

**A no-match is a result, not an error.** "I do not recognise this person" is
the correct, expected answer for any face that was never enrolled, so it exits
0 and reports ``no match`` / ``{"match": null}``. Reserving non-zero for
genuine failures (unreadable image, no face in the frame, missing extra) keeps
``match`` usable in a shell conditional without a stderr dance.

``--threshold`` overrides the store's default cosine floor (0.5) for this call
only. Raise it to demand a closer match, lower it to accept a looser one; the
score is reported either way so a caller can apply its own policy.
"""

from __future__ import annotations

import argparse

from face_recognition_cli.cli._commands._frames import load_frame
from face_recognition_cli.cli._errors import EXIT_USER_ERROR, CliError
from face_recognition_cli.cli._output import emit_result
from face_recognition_cli.engine import FaceEngine
from face_recognition_cli.state import bank_dir
from face_recognition_cli.store import FaceStore

_NO_FACE_HINT = (
    "use an image where one face is clearly visible, closer to the camera and "
    "facing it; try better lighting or a higher-resolution still"
)


def cmd_match(args: argparse.Namespace) -> None:
    json_mode = bool(getattr(args, "json", False))

    frame = load_frame(args.image)
    detection = FaceEngine().detect(frame)
    if detection is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="no face detected in the image",
            remediation=_NO_FACE_HINT,
        )

    store = FaceStore(base_dir=bank_dir(args.bank))
    # threshold=None hands the decision to the store's own default (0.5).
    hit = store.match(detection.embedding, threshold=args.threshold)

    if hit is None:
        emit_result({"match": None} if json_mode else "no match", json_mode=json_mode)
        return

    if json_mode:
        emit_result(
            {"match": {"face_id": hit.face_id, "name": hit.name, "score": hit.score}},
            json_mode=True,
        )
        return

    text = f"face_id: {hit.face_id}\n" f"name: {hit.name}\n" f"score: {hit.score:.4f}"
    emit_result(text, json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "match",
        help="Identify the face in an image against a bank; a no-match exits 0.",
    )
    p.add_argument(
        "--image",
        required=True,
        help="Path to an encoded image file, or '-' to read image bytes from stdin.",
    )
    p.add_argument(
        "--bank",
        default=None,
        help="Face bank to search (default: $FACE_RECOGNITION_BANK, else 'default').",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity to call it a match (default: 0.5).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_match)
