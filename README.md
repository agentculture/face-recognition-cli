# face-recognition-cli

Face recognition and identity management: detect faces, collect embeddings per
identity, and manage them by generated **id** and human **name** — enroll,
match, list, forget one, and forget all at once.

The engine is OpenCV **YuNet** (detection) + **SFace** (128-dim embeddings),
extracted from [`reachy-mini-cli`](https://github.com/agentculture/reachy-mini-cli)
so Reachy Mini can depend on this package instead of carrying its own copy.
The engine, the store, per-consumer face **banks**, and all five domain verbs
are implemented and covered below.

## Install

```bash
pip install 'face-recognition-cli[cpu]'
```

The installed console script is **`face-recognition`**. The import package is
`face_recognition_cli` — deliberately *not* `face_recognition`, which is the
import name of the unrelated dlib-based `face-recognition` distribution on PyPI.

Compute-class extras — both install the **same** `opencv-python-headless`
package and load the **same** two ONNX models, so the choice between them is
about *where inference runs*, never *which model runs*:

| Extra | Target | What it buys |
|-------|--------|---------------|
| `[cpu]` | Raspberry-Pi-class boxes, incl. the Reachy Mini robot | The default path. Detection and embedding both run on CPU. |
| `[gpu]` | DGX Spark, Jetson, RTX-class hosts | Opportunistic acceleration: if the locally installed OpenCV build carries a CUDA DNN backend, the **detector** runs through it. The PyPI `opencv-python-headless` wheel ships no CUDA DNN, so this only activates on a locally built OpenCV (Jetson/DGX images typically carry one) — on a plain `pip install`, `[gpu]` behaves identically to `[cpu]`. |

The **recognizer deliberately stays on CPU on both extras**, so the 128-dim
embedding it produces is bit-identical no matter which host produced it. This
is the hard invariant: **both extras run the same ONNX models and produce the
same embedding space** — a face enrolled on a Jetson matches on a Pi, and vice
versa. `[gpu]` accelerates execution only; it never swaps in a different
model, because that would silently fork the embedding space and make stored
faces unmatchable across machines.

A **bare install** (neither extra) stays importable — the introspection verbs
(`whoami`, `learn`, `explain`, `overview`, `doctor`, `cli overview`) and the
read-only `list` verb all work with no OpenCV present. The verbs that touch an
image (`enroll`, `match`) exit `2` with a message naming the extra:

```text
error: the opencv face-recognition engine is not installed
hint: install the cpu extra: pip install 'face-recognition-cli[cpu]'
```

## Quickstart

```bash
uv sync                                     # or: pip install 'face-recognition-cli[cpu]'
uv run face-recognition whoami              # identity from culture.yaml
uv run face-recognition learn               # self-teaching prompt (add --json)
uv run face-recognition list --json         # inventory of the default bank
```

## CLI

### Face verbs

Every verb below takes `--bank <name>` (see [Banks](#banks)) and `--json`.
Results go to stdout, errors and diagnostics to stderr (never mixed). Exit
codes: `0` success, `1` user error, `2` environment error (missing extra),
`3+` reserved.

**`enroll`** — detect the largest face in an image, embed it, and file that
embedding under a name. Prints the generated 4-char id.

```bash
face-recognition enroll --name ada --image face.jpg
face-recognition enroll --name ada --image face.jpg --json
# {"face_id": "a3f9", "name": "ada", "bank": "default", "num_embeddings": 1}
```

`--image` also accepts `-` to read encoded image bytes from stdin, so a
capture tool can pipe straight in without this tool ever opening a camera:

```bash
some-producer | face-recognition enroll --name ada --image -
```

**`match`** — identify the face in an image against everything enrolled in a
bank, by cosine similarity. A no-match is a *result*, not an error — it exits
`0`, so `match` is safe to use in a shell conditional.

```bash
face-recognition match --image face.jpg --json
# hit:      {"bank": "default", "match": {"face_id": "a3f9", "name": "ada", "score": 0.87}}
# no match: {"bank": "default", "match": null}
```

**`list`** — inventory of the identities enrolled in a bank. Needs no OpenCV.

```bash
face-recognition list
# a3f9  ada  created=2026-07-24  embeddings=1
face-recognition list --json
# {"bank": "default", "faces": [{"id": "a3f9", "name": "ada", "created": ..., "num_embeddings": 1}], "count": 1}
```

**`forget <id>`** — delete one identity (its index record and its `.npy`
embedding file). Takes the id, not the name — names may repeat, ids don't.

```bash
face-recognition forget a3f9
```

**`forget-all`** — delete **every** identity in a bank. Destructive and
irreversible, so it is **dry-run by default**: without `--apply` nothing is
deleted, and the command only reports what *would* go.

```bash
face-recognition forget-all
# would forget 2 face(s) from bank 'default':
#   a3f9  ada
#   b7k2  grace
# nothing deleted — re-run with --apply to commit

face-recognition forget-all --apply
# forgot a3f9  ada
# forgot b7k2  grace
# forgot 2 face(s) from bank 'default'
```

### Introspection verbs

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path (including the `banks` concept). |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `cli overview` | Describe the CLI surface itself. |

## Banks

Every consumer of this tool — `reachy-mini-cli`, a mesh colleague agent, an
interactive operator — enrolls its own set of faces into its own **bank**, a
self-contained index plus its own embedding files. Identities never leak
across banks: an id enrolled for `reachy` is invisible to `list` / `match` /
`forget` run against `colleague`'s bank, and there is no cross-bank search.

Bank selection follows this precedence: an explicit `--bank <name>`, else
`$FACE_RECOGNITION_BANK`, else the bank named `default`.

On disk, under the state dir:

```text
<state dir>/
    banks/
        default/       # faces.json + embeddings/*.npy
        reachy/        # reachy-mini-cli's identities, fully isolated
        colleague/     # a mesh colleague agent's identities, fully isolated
    models/            # YuNet + SFace ONNX — shared by every bank, downloaded once
```

`<state dir>` resolves as `$FACE_RECOGNITION_STATE_DIR`, else
`$XDG_STATE_HOME/face-recognition-cli`, else `~/.local/state/face-recognition-cli`.

The ONNX models are bank-independent — every bank embeds through the same
YuNet + SFace pair, so `models/` lives once, one level above `banks/`, rather
than being duplicated per bank (SFace alone is ~37 MB).

```bash
face-recognition list --bank reachy
FACE_RECOGNITION_BANK=reachy face-recognition list
```

## Library API

`face_recognition_cli` exposes one function at its root, meant for
`reachy-mini-cli` (and any other in-process consumer) to import directly:

```python
from face_recognition_cli import build_face_recognition

result = build_face_recognition(models_dir=None, store_base_dir=None)
if result is None:
    # opencv (the [cpu]/[gpu] extra) is not installed, or the vision stack
    # is broken — one warning was logged, and the feature stays disabled.
    ...
else:
    engine, store = result
```

`build_face_recognition` probes for `cv2` with `importlib.util.find_spec`
*before* importing anything — merely calling it on a bare install never
imports `cv2` — and returns `None` (after exactly one process-wide logged
warning) rather than raising when the extra is missing or construction fails.
`models_dir=` / `store_base_dir=` are optional overrides; omit them to use the
shared models directory and the default bank.

## Privacy & consent

This tool stores **biometric identifiers of real people**: a 128-dim face
embedding plus a name, per enrolled identity. Treat it accordingly.

- **Enroll people only with their consent.** An embedding is derived from
  someone's face; get their agreement before you run `enroll` on their image.
- **Everything stays on the machine.** Embeddings and the identity index are
  written under the local state dir only (see [Banks](#banks)). Nothing is
  ever uploaded, there is no telemetry, and nothing phones home.
- **The only network access anywhere in this package** is the one-time
  download of the two OpenCV model-zoo files — `face_detection_yunet_2023mar.onnx`
  and `face_recognition_sface_2021dec.onnx` — performed by `_ensure_model` in
  [`face_recognition_cli/engine.py`](face_recognition_cli/engine.py) the first
  time an image-consuming verb runs, and cached under `<state dir>/models/`
  afterward. No other module in this package touches the network.
- **Deletion is one verb away.** `forget <id>` removes a single identity and
  its embedding file; `forget-all --apply` removes every identity in a bank in
  one command; deleting the state dir removes everything the tool has ever
  written.
- **Retention is entirely operator-controlled.** Nothing in the permanent
  tier expires on its own — a record persists until something calls `forget`
  or `forget-all`. The temporary tier (an unused "who are you?" seam, not
  wired to any CLI verb today) lives only in process memory and holds nothing
  across process restarts.
- **Per-bank isolation keeps different consumers' data apart.** Reachy's
  identities, a colleague agent's identities, and an operator's identities
  each live in their own bank on disk, with no cross-bank search or merge.

If you enroll other people, that is their biometric data on your disk. Get
their consent, and use `forget` / `forget-all` when they ask.

## Development

```bash
uv sync                                              # create .venv, install (dev deps incl. teken)
uv run face-recognition whoami                       # run the CLI
uv run pytest -n auto                                # full suite (parallel)
uv run pytest --cov=face_recognition_cli --cov-report=term   # coverage (CI gate: fail_under=60)
uv run teken cli doctor . --strict                   # the agent-first rubric gate CI enforces
```

See [`CLAUDE.md`](CLAUDE.md) for the architecture, the naming rules, the
conventions (version-bump-every-PR, the `cicd` PR lane), and the CI/release
setup. Skill provenance is tracked in
[`docs/skill-sources.md`](docs/skill-sources.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
