"""``face-recognition-cli enroll`` — give the face in an image a name.

Detects the largest face in ``--image`` (a path, or ``-`` for encoded bytes on
stdin), embeds it, and files that embedding under ``--name`` in the selected
face bank. The generated 4-char **id** is the durable handle: names are for
humans and may repeat, the id is what ``match`` reports back and what
``forget`` takes.

Bank selection follows :func:`face_recognition_cli.state.resolve_bank` —
explicit ``--bank`` wins, else ``$FACE_RECOGNITION_BANK``, else ``default`` —
so every consumer (reachy, a colleague agent, an operator) keeps its
identities in a separate, non-overlapping index.

Ordering inside the handler is load-frame → detect → store, so the two cheap
failures (an unreadable path, no face in the frame) are reported before
anything is written to disk, and an enrolment either lands completely or not
at all.
"""

from __future__ import annotations

import argparse

from face_recognition_cli.cli._commands._frames import load_frame
from face_recognition_cli.cli._errors import EXIT_USER_ERROR, CliError
from face_recognition_cli.cli._output import emit_result
from face_recognition_cli.engine import FaceEngine
from face_recognition_cli.state import bank_dir, resolve_bank
from face_recognition_cli.store import FaceStore

_NO_FACE_HINT = (
    "use an image where one face is clearly visible, closer to the camera and "
    "facing it; try better lighting or a higher-resolution still"
)


def cmd_enroll(args: argparse.Namespace) -> None:
    json_mode = bool(getattr(args, "json", False))

    frame = load_frame(args.image)
    detection = FaceEngine().detect(frame)
    if detection is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="no face detected in the image",
            remediation=_NO_FACE_HINT,
        )

    bank = resolve_bank(args.bank)
    store = FaceStore(base_dir=bank_dir(args.bank))
    face_id = store.enroll(args.name, detection.embedding)

    # Read the count back off the record rather than hard-coding 1: a fresh
    # enrolment writes exactly one embedding today, and this stays honest if
    # multi-angle enrolment ever lands.
    num_embeddings = next(
        (face["num_embeddings"] for face in store.list_faces() if face["id"] == face_id),
        1,
    )

    if json_mode:
        emit_result(
            {
                "face_id": face_id,
                "name": args.name,
                "bank": bank,
                "num_embeddings": num_embeddings,
            },
            json_mode=True,
        )
        return

    text = (
        f"face_id: {face_id}\n"
        f"name: {args.name}\n"
        f"bank: {bank}\n"
        f"num_embeddings: {num_embeddings}"
    )
    emit_result(text, json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "enroll",
        help="Enroll the face in an image under a name; returns its generated id.",
    )
    p.add_argument(
        "--name",
        required=True,
        help="Human name to file this identity under.",
    )
    p.add_argument(
        "--image",
        required=True,
        help="Path to an encoded image file, or '-' to read image bytes from stdin.",
    )
    p.add_argument(
        "--bank",
        default=None,
        help="Face bank to enroll into (default: $FACE_RECOGNITION_BANK, else 'default').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_enroll)
