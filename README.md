# face-recognition-cli

Face recognition and identity management: detect faces, collect embeddings per
identity, and manage them by generated **id** and human **name** — enroll,
match, list, forget one, and forget all at once.

The engine is OpenCV **YuNet** (detection) + **SFace** (128-dim embeddings),
extracted from [`reachy-mini-cli`](https://github.com/agentculture/reachy-mini-cli)
so Reachy Mini can depend on this package instead of carrying its own copy.

> **Status: scaffold.** The agent-first CLI, packaging, and CI baseline are in
> place. The face engine, the store, the `enroll` / `match` / `list` / `forget` /
> `forget-all` verbs, and the `[cpu]` / `[gpu]` extras are **not implemented
> yet** — they are described under [Roadmap](#roadmap) and tracked in
> [issue #1](https://github.com/agentculture/face-recognition-cli/issues/1).
> Sections below say plainly which is which.

## Install

```bash
pip install face-recognition-cli
```

The installed console script is **`face-recognition`**. The import package is
`face_recognition_cli` — deliberately *not* `face_recognition`, which is the
import name of the unrelated dlib-based `face-recognition` distribution on PyPI.

Planned compute-class extras (not published yet):

| Extra | Target | Backend |
|-------|--------|---------|
| `[cpu]` | Raspberry-Pi-class boxes, incl. the Reachy Mini robot | `opencv-python-headless` — the default path |
| `[gpu]` | DGX Spark, Jetson, RTX-class hosts | GPU-accelerated execution of the *same* ONNX models |

Both extras will run **the same ONNX models and produce the same embedding
space** — a face enrolled on a Jetson matches on a Pi and vice versa. The GPU
extra accelerates execution only; it never swaps in a different model, because
that would silently fork the embedding space and make stored faces unmatchable
across machines. A bare install with neither extra stays importable and exits
with a clean error naming the extra you need.

## Quickstart

```bash
uv sync
uv run pytest -n auto                       # run the test suite
uv run face-recognition whoami              # identity from culture.yaml
uv run face-recognition learn               # self-teaching prompt (add --json)
uv run teken cli doctor . --strict          # the agent-first rubric gate CI runs
```

## CLI

Available today — the agent-first introspection surface:

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path. |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `cli overview` | Describe the CLI surface itself. |

Every command supports `--json`. Results go to stdout, errors and diagnostics to
stderr (never mixed). Exit codes: `0` success, `1` user error, `2` environment
error, `3+` reserved.

## Roadmap

Planned verbs (see [issue #1](https://github.com/agentculture/face-recognition-cli/issues/1)):

| Verb | What it will do |
|------|-----------------|
| `enroll` | Create a permanent identity from a detected face — returns its generated id. |
| `match` | Find the best-matching enrolled identity for a face (cosine similarity, default threshold 0.5). |
| `list` | Inventory of enrolled identities: id, name, created, embedding count. |
| `forget <id>` | Delete one identity and its embeddings. |
| `forget-all` | Delete **every** identity. Destructive and irreversible, so it is **dry-run by default** and reports what would be deleted (count, ids, names); `--apply` commits. |

Also planned: a stable public API (`FaceEngine` + `FaceStore`) for
`reachy-mini-cli` to import, and the `[cpu]` / `[gpu]` extras above.

Still undecided, and deliberately not guessed here: where the store lives by
default (and how it reads faces already enrolled under Reachy's state dir),
whether the CLI grows a continuous watch mode or stays one-shot, whether frames
come from a path you pass in or from a camera this tool opens itself, and
whether the store's temporary/TTL tier survives.

## Privacy

This tool handles **biometric identifiers of real people**. The design
commitments:

- **Everything stays on the machine.** Embeddings and the identity index are
  written to local state only. Nothing is uploaded, and there is no telemetry.
- **The only network access is a one-time model download** — the YuNet and SFace
  ONNX files from the OpenCV model zoo, cached locally after the first run.
- **Deletion is a first-class verb, not a cleanup chore.** `forget` removes one
  identity and its embedding files; `forget-all` removes every identity in one
  command.
- **What is stored is an embedding, not a photograph** — a 128-dim vector per
  enrolled face, plus the name you chose and a generated 4-character id.

If you enroll other people, that is their biometric data on your disk. Get their
consent, and use `forget` / `forget-all` when they ask.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the architecture, the naming rules, the
conventions (version-bump-every-PR, the `cicd` PR lane), and the open design
questions. Skill provenance is tracked in
[`docs/skill-sources.md`](docs/skill-sources.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
