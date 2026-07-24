"""``face-recognition-cli forget-all`` — destroy every identity in a bank.

The one requirement :class:`~face_recognition_cli.store.FaceStore` does not
already meet on its own (it only ever deletes one identity at a time via
``forget``). Wiping a whole bank is destructive and irreversible, so this verb
obeys the mesh write-verb rule: **dry-run by default, ``--apply`` commits.**
Without ``--apply`` nothing is deleted — the command only reports what *would*
be deleted (count, ids, names) for the selected bank. With ``--apply`` it
iterates :meth:`~face_recognition_cli.store.FaceStore.list_faces` and calls
:meth:`~face_recognition_cli.store.FaceStore.forget` on each id, scoped to the
one bank selected via ``--bank`` — every other bank is untouched.
"""

from __future__ import annotations

import argparse

from face_recognition_cli.cli._output import emit_result
from face_recognition_cli.state import bank_dir, resolve_bank
from face_recognition_cli.store import FaceStore


def cmd_forget_all(args: argparse.Namespace) -> None:
    bank = resolve_bank(getattr(args, "bank", None))
    store = FaceStore(base_dir=bank_dir(args.bank))
    apply = bool(getattr(args, "apply", False))
    json_mode = bool(getattr(args, "json", False))

    targets = [{"id": face["id"], "name": face["name"]} for face in store.list_faces()]

    if apply:
        for target in targets:
            store.forget(target["id"])

    payload = {
        "bank": bank,
        "dry_run": not apply,
        "count": len(targets),
        "faces": targets,
    }

    if json_mode:
        emit_result(payload, json_mode=True)
        return

    if not targets:
        verb = "forgot" if apply else "would forget"
        emit_result(f"{verb} 0 faces from bank '{bank}'", json_mode=False)
        return

    if apply:
        lines = [f"forgot {t['id']}  {t['name']}" for t in targets]
        lines.append(f"forgot {len(targets)} face(s) from bank '{bank}'")
        emit_result("\n".join(lines), json_mode=False)
        return

    lines = [f"would forget {len(targets)} face(s) from bank '{bank}':"]
    lines.extend(f"  {t['id']}  {t['name']}" for t in targets)
    lines.append("nothing deleted — re-run with --apply to commit")
    emit_result("\n".join(lines), json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "forget-all",
        help="Delete every identity in a bank (dry-run by default; --apply commits).",
    )
    p.add_argument(
        "--bank",
        default=None,
        help="Bank to operate on (default: $FACE_RECOGNITION_BANK or 'default').",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Commit the deletion. Without this flag, nothing is deleted (dry run).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_forget_all)
