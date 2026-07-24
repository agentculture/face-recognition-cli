"""Markdown catalog for ``face-recognition-cli explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("face-recognition-cli",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# face-recognition-cli

A clonable template for AgentCulture mesh agents. It carries an agent-first CLI
(cited from the teken `python-cli` reference), a mesh identity (`culture.yaml` +
`CLAUDE.md`), the canonical guildmaster skill kit under `.claude/skills/`, and a
buildable/deployable package baseline. Clone it, rename the package, edit
`culture.yaml`, and you have a new agent.

## Verbs

- `face-recognition-cli whoami` — identity probe from `culture.yaml`.
- `face-recognition-cli learn` — structured self-teaching prompt.
- `face-recognition-cli explain <path>` — markdown docs for any noun/verb.
- `face-recognition-cli overview` — descriptive snapshot of the agent.
- `face-recognition-cli doctor` — check the agent-identity invariants.
- `face-recognition-cli cli overview` — describe the CLI surface.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `face-recognition-cli explain whoami`
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
}
