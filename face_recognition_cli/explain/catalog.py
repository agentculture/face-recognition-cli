"""Markdown catalog for ``face-recognition-cli explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("face-recognition-cli",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# face-recognition-cli

Face identity for the AgentCulture mesh: detect the face in an image, enroll it
under a human name, match a new face against what is already enrolled, and
manage those identities. Embeddings come from OpenCV's YuNet detector and SFace
128-dim embedder (the `[cpu]` / `[gpu]` extras); identities live in a local
per-consumer **bank** on disk.

Nothing leaves the machine: the only network access anywhere in this tool is
the one-time download of the two ONNX model files.

## Face verbs

- `face-recognition-cli enroll --name <name> --image <path|->` — file a face
  under a name; prints its generated id.
- `face-recognition-cli match --image <path|->` — who is this? A no-match is a
  result (exit 0), not an error.
- `face-recognition-cli list` — inventory of the identities in a bank.
- `face-recognition-cli forget <id>` — delete one identity.
- `face-recognition-cli forget-all` — delete every identity in a bank
  (dry-run by default; `--apply` commits).

## Introspection verbs

- `face-recognition-cli whoami` — identity probe from `culture.yaml`.
- `face-recognition-cli learn` — structured self-teaching prompt.
- `face-recognition-cli explain <path>` — markdown docs for any noun/verb.
- `face-recognition-cli overview` — descriptive snapshot of the agent.
- `face-recognition-cli doctor` — check the agent-identity invariants.
- `face-recognition-cli cli overview` — describe the CLI surface.

## Concepts

- `face-recognition-cli explain banks` — per-consumer face banks: how identities
  are partitioned, how a bank is selected, and where state lives on disk.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error (notably: the `[cpu]` extra is not installed)
- `3+` reserved

## See also

- `face-recognition-cli explain enroll`
- `face-recognition-cli explain banks`
- `face-recognition-cli explain doctor`
"""

_WHOAMI = """\
# face-recognition-cli whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    face-recognition-cli whoami
    face-recognition-cli whoami --json
"""

_LEARN = """\
# face-recognition-cli learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    face-recognition-cli learn
    face-recognition-cli learn --json
"""

_EXPLAIN = """\
# face-recognition-cli explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    face-recognition-cli explain face-recognition-cli
    face-recognition-cli explain whoami
    face-recognition-cli explain --json <path>
"""

_OVERVIEW = """\
# face-recognition-cli overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    face-recognition-cli overview
    face-recognition-cli overview --json
"""

_DOCTOR = """\
# face-recognition-cli doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`colleague` → `AGENTS.colleague.md`), plus a
skills-present check. Exits 1 when unhealthy.

## Usage

    face-recognition-cli doctor
    face-recognition-cli doctor --json
"""

_CLI = """\
# face-recognition-cli cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    face-recognition-cli cli overview
    face-recognition-cli cli overview --json
"""


_ENROLL = """\
# face-recognition-cli enroll

Detects the largest face in an image, embeds it, and files that embedding under
a human name in the selected bank. Prints the generated 4-char **id**: names are
for humans and may repeat, the id is the durable handle `match` reports back and
`forget` takes.

## Usage

    face-recognition-cli enroll --name ada --image face.png
    face-recognition-cli enroll --name ada --image face.png --bank reachy
    face-recognition-cli enroll --name ada --image - < face.png
    face-recognition-cli enroll --name ada --image face.png --json

## Input

`--image` takes a path to an encoded image (png, jpeg, bmp, webp — whatever the
local OpenCV build decodes), or `-` to read encoded image bytes from **stdin**.
The `-` form is what keeps this tool out of the capture business: it owns no
camera and opens no device, so a capture tool pipes into it instead.

    webcam-cli capture --format png - | face-recognition enroll --name ada --image -

## Notes

- `--bank <name>` selects the face bank (default `$FACE_RECOGNITION_BANK`, else
  `default`). See `face-recognition-cli explain banks`.
- JSON: `{"face_id", "name", "bank", "num_embeddings"}`.
- Exits `1` when the image cannot be read or decoded, or when no face is in the
  frame. Exits `2` when neither the `[cpu]` nor the `[gpu]` extra is installed.
- Nothing is written unless a face is actually detected: an enrolment lands
  completely or not at all.
"""

_MATCH = """\
# face-recognition-cli match

Identifies the largest face in an image against every identity enrolled in the
selected bank, by cosine similarity. Read-only: nothing is written, and an
unmatched face is never enrolled as a side effect.

**A no-match is a result, not an error.** "I do not recognise this person" is
the correct answer for anyone never enrolled, so it exits `0` and reports
`no match` / `{"bank": ..., "match": null}` — which keeps `match` usable in a shell
conditional. Non-zero is reserved for genuine failures: an unreadable image, no
face in the frame, a missing extra.

## Usage

    face-recognition-cli match --image face.png
    face-recognition-cli match --image - < face.png
    face-recognition-cli match --image face.png --bank reachy
    face-recognition-cli match --image face.png --threshold 0.7 --json

## Notes

- `--threshold` overrides the store's default cosine floor (`0.5`) for this call
  only. Raise it to demand a closer match, lower it to accept a looser one; the
  score is reported either way so a caller can apply its own policy.
- `--bank <name>` selects the bank searched (default `$FACE_RECOGNITION_BANK`,
  else `default`). See `face-recognition-cli explain banks`.
- JSON: `{"bank": <bank>, "match": null}`, or
  `{"bank": <bank>, "match": {"face_id", "name", "score"}}`.
- Exits `2` when neither the `[cpu]` nor the `[gpu]` extra is installed.
"""

_LIST = """\
# face-recognition-cli list

Read-only inventory of the identities enrolled in one bank: id, name, creation
date, and how many embeddings back each identity. Needs no OpenCV — the store is
plain JSON plus numpy arrays — so `list` works on a bare install with neither
extra.

## Usage

    face-recognition-cli list
    face-recognition-cli list --bank reachy
    face-recognition-cli list --json

## Notes

- JSON: `{"bank", "count", "faces": [{"id", "name", "created", "num_embeddings"}]}`.
- Text mode prints one line per identity, or `no faces enrolled`.
- An empty (or never-created) bank is not an error: it lists nothing and exits 0.
- The `id` column is what `forget` takes.
"""

_FORGET = """\
# face-recognition-cli forget <id>

Deletes exactly one identity from a bank — its record in the index and its
embedding file(s). Takes the 4-char id from `list` or `enroll`, not a name
(names may repeat; ids do not). Needs no OpenCV.

Unlike `forget-all`, this verb is not dry-run-by-default: it names one id, so
there is nothing to preview. Either that id exists in the selected bank and is
deleted, or the command fails with a structured error.

## Usage

    face-recognition-cli forget a3f9
    face-recognition-cli forget a3f9 --bank reachy
    face-recognition-cli forget a3f9 --json

## Notes

- JSON: `{"forgotten": <id>, "bank": <bank>}`.
- Deletion is scoped to one bank. An id enrolled in a *different* bank is not
  visible here and exits `1` — see `face-recognition-cli explain banks`.
"""

_FORGET_ALL = """\
# face-recognition-cli forget-all

Deletes **every** identity in one bank. This is destructive and irreversible, so
it follows the mesh write-verb rule: **dry-run by default, `--apply` commits.**
Without `--apply` nothing is deleted — the command only reports what *would* go
(count, ids, names). Needs no OpenCV.

Scope is a single bank: `--bank` selects it, and every other bank is untouched.

## Usage

    face-recognition-cli forget-all                        # dry run: what would go
    face-recognition-cli forget-all --bank reachy
    face-recognition-cli forget-all --bank reachy --apply  # commits
    face-recognition-cli forget-all --json

## Notes

- JSON: `{"bank", "dry_run", "count", "faces": [{"id", "name"}]}`. `dry_run` is
  `true` unless `--apply` was passed.
- This is the deletion half of the privacy story: enrolled embeddings are
  biometric identifiers, and every identity in a bank is removable with one
  command.
"""

_BANKS = """\
# banks — per-consumer face identity stores

A **bank** is one self-contained set of enrolled identities: its own index and
its own embedding files. Every consumer of this tool — reachy-mini-cli, a mesh
colleague agent, an operator at a terminal — gets its own bank, so an identity
enrolled by one is invisible to `list`, `match`, and `forget` run against
another. The isolation is total: there is no cross-bank search and no merge.

## Selecting a bank

Every store-touching verb (`enroll`, `match`, `list`, `forget`, `forget-all`)
takes `--bank <name>`. Resolution order:

1. an explicit `--bank <name>`;
2. `$FACE_RECOGNITION_BANK`;
3. the bank named `default`.

## On disk

    <state dir>/
        banks/
            default/       # faces.json + embeddings/*.npy
            reachy/        # another consumer, fully isolated
            colleague/     # and another
        models/            # YuNet + SFace ONNX — shared by every bank

`<state dir>` resolves as `$FACE_RECOGNITION_STATE_DIR`, else
`$XDG_STATE_HOME/face-recognition-cli`, else
`~/.local/state/face-recognition-cli`.

Banks partition *identities*, not *models*: every bank embeds through the same
YuNet + SFace pair, so `models/` sits one level above `banks/` and is downloaded
once (SFace alone is ~37 MB). It is the store, not the model, that keeps banks
apart — which also means an embedding stays comparable across machines, and a
face enrolled on a GPU host matches on a CPU one.

## Usage

    face-recognition-cli list --bank reachy
    FACE_RECOGNITION_BANK=reachy face-recognition-cli list

## See also

- `face-recognition-cli explain forget-all` — wiping one bank.
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("face-recognition-cli",): _ROOT,
    ("face-recognition",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
    ("enroll",): _ENROLL,
    ("match",): _MATCH,
    ("list",): _LIST,
    ("forget",): _FORGET,
    ("forget-all",): _FORGET_ALL,
    # A concept, not a verb: `banks` has no parser entry, but it is the one
    # piece of shared vocabulary every store-touching verb's --bank flag needs.
    ("banks",): _BANKS,
}
