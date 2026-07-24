"""``face-recognition-cli forget`` — delete one identity from a bank.

Wraps :meth:`~face_recognition_cli.store.FaceStore.forget`, which removes a
single permanent face record and its ``.npy`` embedding file(s). Unlike
``forget-all`` this verb is not the destructive-by-default kind covered by the
mesh write-verb rule — it names exactly one id, so there is nothing to preview:
either that id exists in the selected bank and is deleted, or it doesn't and
the command fails with a structured error.
"""

from __future__ import annotations

import argparse

from face_recognition_cli.cli._errors import EXIT_USER_ERROR, CliError
from face_recognition_cli.cli._output import emit_result
from face_recognition_cli.state import bank_dir, resolve_bank
from face_recognition_cli.store import FaceStore


def cmd_forget(args: argparse.Namespace) -> None:
    bank = resolve_bank(getattr(args, "bank", None))
    store = FaceStore(base_dir=bank_dir(args.bank))
    json_mode = bool(getattr(args, "json", False))

    removed = store.forget(args.face_id)
    if not removed:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"no face with id '{args.face_id}' in bank '{bank}'",
            remediation=f"run 'face-recognition-cli list --bank {bank}' to see enrolled ids",
        )

    if json_mode:
        emit_result({"forgotten": args.face_id, "bank": bank}, json_mode=True)
        return
    emit_result(f"forgot {args.face_id} from bank '{bank}'", json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "forget",
        help="Delete one identity from a bank by id.",
    )
    p.add_argument("face_id", help="The id of the face to forget (see 'list').")
    p.add_argument(
        "--bank",
        default=None,
        help="Bank to operate on (default: $FACE_RECOGNITION_BANK or 'default').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_forget)
