"""``face-recognition-cli list`` — inventory of enrolled identities in a bank.

Read-only. Builds a :class:`~face_recognition_cli.store.FaceStore` against the
selected bank (``--bank``, default resolved by
:func:`face_recognition_cli.state.resolve_bank`) and reports
:meth:`~face_recognition_cli.store.FaceStore.list_faces` verbatim in JSON mode,
or one line per identity in text mode.

Named ``list_faces`` (not ``list``) to avoid shadowing the builtin at import
time; the registered verb name is still ``list``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from face_recognition_cli.cli._output import emit_result
from face_recognition_cli.state import bank_dir, resolve_bank
from face_recognition_cli.store import FaceStore


def _iso_date(created: float) -> str:
    """Render a store ``created`` epoch timestamp as an ISO-8601 date (UTC)."""
    return datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()


def cmd_list(args: argparse.Namespace) -> None:
    bank = resolve_bank(getattr(args, "bank", None))
    store = FaceStore(base_dir=bank_dir(args.bank))
    faces = store.list_faces()
    json_mode = bool(getattr(args, "json", False))

    if json_mode:
        emit_result({"bank": bank, "faces": faces, "count": len(faces)}, json_mode=True)
        return

    if not faces:
        emit_result("no faces enrolled", json_mode=False)
        return

    lines = [
        f"{face['id']}  {face['name']}  "
        f"created={_iso_date(face['created'])}  embeddings={face['num_embeddings']}"
        for face in faces
    ]
    emit_result("\n".join(lines), json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "list",
        help="List enrolled face identities in a bank.",
    )
    p.add_argument(
        "--bank",
        default=None,
        help="Bank to list (default: $FACE_RECOGNITION_BANK or 'default').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_list)
