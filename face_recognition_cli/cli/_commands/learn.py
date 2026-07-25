"""``face-recognition-cli learn`` — the learnability affordance.

Prints a structured self-teaching prompt. Must satisfy the agent-first rubric:
>=200 chars and mention purpose, command map, exit codes, --json, and explain.
"""

from __future__ import annotations

import argparse

from face_recognition_cli import __version__
from face_recognition_cli.cli._output import emit_result

_TEXT = """\
face-recognition-cli — face identity for the AgentCulture mesh.

Purpose
-------
Detect the face in an image, enroll it under a human name, match a new face
against what is already enrolled, and manage those identities. Embeddings come
from OpenCV's YuNet detector + SFace 128-dim embedder (the [cpu] / [gpu]
extras); identities live in a local per-consumer bank on disk. Nothing leaves
the machine — the only network access is the one-time model download.

This tool owns no camera and opens no device: feed it an image path, or encoded
image bytes on stdin (--image -), and it interprets the frame.

Face commands
-------------
  face-recognition-cli enroll --name N --image P  File a face under a name.
  face-recognition-cli match --image P            Who is this? No match = exit 0.
  face-recognition-cli list                       Identities enrolled in a bank.
  face-recognition-cli forget <id>                Delete one identity.
  face-recognition-cli forget-all [--apply]       Wipe a bank (dry-run default).

Introspection commands
----------------------
  face-recognition-cli whoami             Identity from culture.yaml.
  face-recognition-cli learn              This self-teaching prompt.
  face-recognition-cli explain <path>...  Markdown docs for any noun/verb path.
  face-recognition-cli overview           Descriptive snapshot of the agent.
  face-recognition-cli doctor             Check the agent-identity invariants.
  face-recognition-cli cli overview       Describe the CLI surface itself.

Face banks
----------
Every store-touching verb takes --bank <name>. Identities are partitioned per
consumer: one enrolled in a given bank is invisible to list/match/forget run
against another. Default: $FACE_RECOGNITION_BANK, else 'default'.
  face-recognition-cli explain banks

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr never mix.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg, no face in the frame)
  2 environment / setup error (notably: the [cpu] extra is not installed)
  3+ reserved

More detail
-----------
  face-recognition-cli explain face-recognition-cli
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "face-recognition-cli",
        "version": __version__,
        "purpose": (
            "Face identity: detect the face in an image, enroll it under a name, "
            "match it against a bank, and manage the stored identities."
        ),
        "commands": [
            {"path": ["enroll"], "summary": "File a face under a name; prints its id."},
            {"path": ["match"], "summary": "Identify a face; a no-match exits 0."},
            {"path": ["list"], "summary": "Inventory of the identities in a bank."},
            {"path": ["forget"], "summary": "Delete one identity by id."},
            {
                "path": ["forget-all"],
                "summary": "Delete every identity in a bank (dry-run; --apply commits).",
            },
            {"path": ["whoami"], "summary": "Identity probe from culture.yaml."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by path."},
            {"path": ["overview"], "summary": "Descriptive snapshot of the agent."},
            {"path": ["doctor"], "summary": "Check the agent-identity invariants."},
            {"path": ["cli", "overview"], "summary": "Describe the CLI surface."},
        ],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "face-recognition-cli explain <path>",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
