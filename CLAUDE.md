# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`face-recognition-cli` is an **AgentCulture mesh agent** whose domain is *face
identity*: detect faces, collect embeddings **per identity**, and manage those
identities by a generated **id** and a human **name** — enroll, match, list,
forget one, and forget all at once.

It exists to take over the OpenCV **YuNet + SFace** engine that
[`reachy-mini-cli`](https://github.com/agentculture/reachy-mini-cli) carries in
`reachy/vision/face.py` + `reachy/vision/face_store.py`, so the robot depends on
this package instead of shipping its own copy. The build brief is
[issue #1](https://github.com/agentculture/face-recognition-cli/issues/1) — it
records *why* the engine is extracted rather than swapped for dlib. The frame
and plan derived from it are checked in under [`docs/specs/`](docs/specs/) and
[`docs/plans/`](docs/plans/).

**State of the repo today.** The domain work is **on disk**. Alongside the
scaffold's agent-first CLI (the introspection verbs `whoami`, `learn`,
`explain`, `overview`, `doctor`, `cli overview`), the vendored skill kit and the
CI/publish baseline, the package now carries:

- `face_recognition_cli/engine.py` — the ported `FaceEngine`: OpenCV YuNet
  detector + SFace 128-dim embedder, `cv2` imported lazily.
- `face_recognition_cli/store.py` — the ported `FaceStore`: cosine matching over
  a `faces.json` index plus one `.npy` per embedding.
- `face_recognition_cli/state.py` — the state dir and the per-consumer **bank**
  layout (`banks/<name>/`, shared `models/`). Pure stdlib.
- `face_recognition_cli/api.py` — `build_face_recognition()`, the stable import
  surface `reachy-mini-cli` consumes, re-exported from the package root.
- five domain verbs under `cli/_commands/` — `enroll`, `match`, `list`,
  `forget`, `forget-all` — wired into `_build_parser()`, each with an `explain`
  catalog entry (plus a `banks` concept entry).
- `[cpu]` / `[gpu]` extras in `pyproject.toml`, and `numpy` as the single base
  dependency.

What has **not** happened yet: the first real release (task `t12` in the plan)
and the cross-repo migration issue on `reachy-mini-cli` (`t13`). See
[The domain work](#the-domain-work-landed) for what shipped and
[Decisions (recorded)](#decisions-recorded) for the questions the build settled.
Keep the boundary honest in both directions: describe what is actually on disk,
and mark anything that runs ahead of it as planned.

Siblings worth knowing: `face-cli` is the *expressive output* side of a face (a
rendered face on a screen) — you are the *perceptual input* side; you share a
subject and nothing else. `webcam-cli` (capture) and `media-cli` (device plane)
own camera access.

## Naming — deliberate, do not "fix"

| Thing | Value |
|-------|-------|
| Installed console script | **`face-recognition`** (`face_recognition_cli.cli:main`) |
| Import package | **`face_recognition_cli`** |
| Distribution / PyPI name | `face-recognition-cli` |
| `prog=` and every help / `learn` / `explain` / README string | `face-recognition-cli` |

The import package is **deliberately decoupled** from the console script. The
obvious name `face_recognition` is the *import* name of the unrelated
`face-recognition` distribution on PyPI (the dlib one); taking it would shadow
that package for anyone who installs both. Do not rename the package to match
the command.

Note the two spellings in use: `prog=` is `face-recognition-cli` while the
installed script is `face-recognition`. Both `explain face-recognition-cli` and
`explain face-recognition` resolve to the root catalog entry
(`explain/catalog.py`). If you ever change either name, do it as one deliberate
pass — `pyproject.toml` (`name`, `[project.scripts]`), `prog=`, every string
under `cli/_commands/` and `explain/catalog.py`, `sonar-project.properties`,
`README.md`, and the test assertions — never piecemeal:

```bash
git grep -nF -e 'face-recognition-cli' -e 'face_recognition_cli' -e 'face-recognition'
```

## Identity and the backend mismatch

`culture.yaml` declares:

```yaml
agents:
- suffix: face-recognition-cli
  backend: colleague
  model: sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
```

`backend: colleague` fixes the mesh resident prompt file to
**`AGENTS.colleague.md`** (present on disk); this `CLAUDE.md` is the Claude Code
guidance file, not the mesh runtime prompt. Together they satisfy the two
invariants `steward doctor` checks — **prompt-file-present** and
**backend-consistency** — and the CLI's own `doctor` re-implements the same
checks locally.

**Known inconsistency:** the build brief (issue #1) lists the backend as
`claude` "as scaffolded", and the seed `CLAUDE.md` this file replaced claimed
the same. The checked-in reality is `colleague`, inherited verbatim from the
template. The brief asks you to *reconcile `culture.yaml` against the backend
you actually run*. That reconciliation has **not** been done.

**It is already breaking a tool.** `devex pr reply` rejects the declaration
outright:

```text
devex: culture.yaml agent 'face-recognition-cli' has unknown backend 'colleague'
hint: expected one of claude (= claude-code), codex, copilot, acp
```

So the `cicd` skill's reply lane only works with an explicit
`--agent claude-code` override (`devex pr reply <PR> --agent claude-code`).
`lint` / `open` / `read` / `status` / `await` are unaffected — they don't
resolve the backend.

The mesh evidence points the same way: `devague`'s `CLAUDE.md` calls `claude`
"the mesh standard", and of the siblings in this workspace only `steward` (and
this repo, via the template) declares `colleague` — `guildmaster`, `devague`,
`colleague`, and `reachy-mini-cli` all declare `claude`. If you flip it, three
things move together or `doctor` and CI break:

1. `culture.yaml` (`backend:`),
2. the test assertions in `tests/test_cli.py` — `test_whoami_text` and
   `test_whoami_json` assert `backend == "colleague"` and will fail,
3. the prose: `doctor`'s own check would still pass (`_PROMPT_FILE` maps
   `claude → CLAUDE.md`, which exists), but `AGENTS.colleague.md` becomes a dead
   file and `explain/catalog.py`'s `_DOCTOR` entry — which names
   `colleague → AGENTS.colleague.md` explicitly — goes stale.

The engine-extraction build **deliberately deferred** this: the backend stays
`colleague`, because flipping it is an identity change with its own test and
prose fallout and had no business riding along inside a domain-code extraction.
It is still owed — see [Decisions (recorded)](#decisions-recorded).

## Commands

```bash
uv sync                                              # create .venv, install (dev deps incl. teken)
uv run face-recognition whoami                       # run the CLI
uv run face-recognition list --json                  # a domain verb; needs no OpenCV
uv sync --extra cpu                                  # add OpenCV, for enroll/match
uv run pytest -n auto                                # full suite (parallel)
uv run pytest tests/test_cli.py::test_whoami_text    # a single test
uv run pytest --cov=face_recognition_cli --cov-report=term   # coverage (CI gate: fail_under=60)
uv run teken cli doctor . --strict                   # the agent-first rubric gate CI enforces
```

Lint stack (the CI `lint` job runs all of these; line length is 100 everywhere):

```bash
uv run black --check face_recognition_cli tests
uv run isort --check-only face_recognition_cli tests
uv run flake8 face_recognition_cli tests
uv run bandit -c pyproject.toml -r face_recognition_cli    # B101/B404/B603 skipped in pyproject
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken" "#.venv" "#.devague"
```

The rubric gate currently reports `healthy: 26/26`. It runs the *installed* CLI
end to end (help, `learn`, `explain`, `overview`, `doctor`, JSON parseability,
error shape), so a broken verb fails CI even when pytest is green — run it
before you push CLI changes.

## Architecture: the agent-first CLI

Everything routes through `face_recognition_cli.cli.main()` → `_build_parser()`
(`face_recognition_cli/cli/__init__.py`). The design exists to satisfy the
**teken agent-first rubric** (`teken cli doctor . --strict`), which gates CI —
keep it green when you touch the CLI.

- **Adding a verb:** create `face_recognition_cli/cli/_commands/<verb>.py`
  exposing `register(sub)` (add a `--json` flag, `set_defaults(func=...)`), then
  import it and call `<verb>.register(sub)` inside `_build_parser()` — there is
  a marked "Register your own noun groups here" comment at the call site. That
  is the only wiring step. `whoami.py` is the canonical example.
- **Noun groups** (a subcommand with its own sub-verbs, like `cli`): when you
  call `p.add_subparsers(...)`, pass `parser_class=type(p)` so nested parse
  errors keep the structured error contract instead of falling back to
  argparse's default stderr/exit-2. A noun that has action-verbs **must** also
  expose an `overview` verb (rubric requirement) — see
  `cli/_commands/cli.py`, which exists purely to satisfy that check.
- **Error contract** (`cli/_errors.py`, `cli/_output.py`): every failure raises
  `CliError(code, message, remediation)`; `_dispatch` catches it and wraps *any*
  other exception so no Python traceback ever leaks. `main()` pre-scans argv for
  `--json` into `_CliArgumentParser._json_hint` so even argparse parse-time
  errors (which fire before `args.json` exists) render as JSON when asked. Text
  errors are always two lines: `error: …` then `hint: …` (the `hint:` prefix is
  rubric-required). Exit policy: `0` success, `1` user error, `2` environment
  error, `3+` reserved.
- **Output split** (`cli/_output.py`): results → stdout, errors and diagnostics
  → stderr, **never mixed**, in both text and JSON modes. Every verb takes
  `--json`.
- **`explain` catalog** (`face_recognition_cli/explain/`): markdown keyed by
  command-path tuples in `catalog.py`'s `ENTRIES`.
  `test_every_catalog_path_resolves` verifies every catalog entry resolves — but
  nothing fails if you add a verb *without* a catalog entry, so add the `ENTRIES`
  key yourself when you add a verb.
- **`whoami` / `doctor`:** `whoami` hand-parses `culture.yaml` with a line
  scanner — no YAML library — and walks up from `__file__` so it reports *this
  agent's* identity, not whatever `culture.yaml` sits in the caller's cwd. The
  template's *zero* third-party runtime dependencies became **no dependencies
  beyond `numpy`** when the engine landed (the engine's arrays and the store's
  `.npy` files force it; `pyproject.toml` records the trade). That is still a
  property worth keeping: everything the CLI plumbing itself needs is standard
  library, so the introspection verbs — and `list` / `forget` / `forget-all`,
  which touch the store but never the engine — run on a bare install with
  neither extra.

The CLI is cited (cite-don't-import) from teken's `python-cli` reference;
`teken` is a dev dependency only.

## The domain work (landed)

What follows describes code that is on disk. The rationale is kept because the
invariants are load-bearing: they are *why* the extraction was worth doing, and
quietly breaking one invalidates faces real people already enrolled.

### The invariant that outranks everything

`engine.py` and `store.py` are ports of `reachy/vision/face.py` and
`reachy/vision/face_store.py`, **keeping the embedding space and the on-disk
layout compatible**. That compatibility is the whole reason extraction beat
adopting dlib:

1. SFace embeddings and dlib `face_recognition` encodings are both 128-dim but
   are *different vector spaces* — switching engines invalidates every face
   already enrolled on the robot and forces real people to re-enrol.
2. dlib needs a cmake/C++ toolchain build on the Raspberry-Pi-class robot box;
   `opencv-python-headless` ships wheels.

The claim is tested, not asserted: `tests/fixtures/reachy_store/` is a
`faces.json` + `embeddings/*.npy` tree written by reachy's own
`FaceStore.enroll()`, and `tests/test_store.py` loads it through the ported
store. The two model URL/filename literals at the top of `engine.py` *are* the
embedding-space contract — editing them forks the vector space.

### `FaceEngine` (`face_recognition_cli/engine.py`)

- OpenCV **YuNet** detector + **SFace** 128-dim embedder, largest-face
  selection (by bbox area), `alignCrop` + `feature`. `EMBEDDING_DIM = 128`.
- **`cv2` is imported lazily, inside functions only** (`_import_cv2`) — never at
  module import time, so the module stays importable on a bare install with
  neither extra. A missing extra surfaces as a clean exit-2 `CliError` naming
  `[cpu]`, not an `ImportError` traceback. `cli/_commands/_frames.py` reuses
  that same probe rather than reimplementing it, so there is exactly one
  missing-OpenCV error shape no matter which surface hits it first.
- ONNX models are pulled from the OpenCV model zoo
  (`face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`)
  into `<state dir>/models/` — shared across banks — downloaded to a `.part`
  sibling and renamed into place only after clearing a **size sanity floor**
  (`YUNET_MIN_BYTES = 100_000`, `SFACE_MIN_BYTES = 20_000_000`). A truncated
  download or a captive-portal HTML page is rejected rather than written to
  disk, where it would "exist" on the next run and fail cryptically inside
  OpenCV.
- `detect(frame)` is **synchronous and stateless-per-call**, returning
  `FaceDetection(bbox_norm, embedding) | None`. Nova's daemon thread and 500 ms
  interval were deliberately *not* ported: loop ownership belongs to the caller,
  and that is a recorded non-goal. Do not re-add a background loop inside the
  engine.

### `FaceStore` (`face_recognition_cli/store.py`)

Ported faithfully; the surface is what reachy already had:

| Member | Behaviour |
|---|---|
| `enroll(name, embedding) -> face_id` | Permanent record from a fresh embedding, in one call |
| `match(embedding, *, threshold=None) -> FaceMatch \| None` | Cosine match; `FaceMatch(face_id, name, score)` |
| `forget(face_id) -> bool` | Delete **one** identity |
| `list_faces() -> list[dict]` | Inventory (`id`, `name`, `created`, `num_embeddings`) |
| `get_unique_id(name)` / `permanent_count` | Lookups |
| `remember_temporary` / `get_temporary` / `cleanup_expired` / `temporary_count` | Temporary tier with TTL |
| `load` / `save` | One JSON index (`faces.json`) + one `.npy` per embedding under `embeddings/`, in the bank dir |

Constants: `DEFAULT_MATCH_THRESHOLD = 0.5` (cosine), `DEFAULT_TEMP_TTL = 15 * 60`
(900 s). Ids are **4-char lowercase alphanumeric** generated with `secrets` — a
CSPRNG chosen to sidestep bandit's insecure-random lint, not for
confidentiality. A corrupt or missing index degrades to "start fresh", never
raises; `save` is write-then-replace. Every time-sensitive method takes `now=`,
and the constructor takes `base_dir=` and `clock=`, so it is deterministic under
test — **keep those seams**, they are why the store is testable without mocking
time.

The **one deliberate deviation from reachy** is the default storage root:
`default_base_dir()` resolves to `state.bank_dir()` (`<state dir>/banks/<bank>`)
instead of reachy's `<state dir>/faces`. Format unchanged; only where an
*unconfigured* store looks.

Two **hardenings past what reachy shipped**, both failure-path only and both
recorded in the module docstring — do not "simplify" either back out:

- `load` also catches `UnicodeDecodeError`, so an index that is not valid UTF-8
  (non-text garbage bytes, not just malformed JSON) degrades to "start fresh"
  instead of raising.
- `save` brings its `mkdir` inside the `try`, so an unwritable parent is logged
  like every other persistence failure instead of propagating.

### State and banks (`face_recognition_cli/state.py`)

Pure standard library — no `numpy`, no `cv2` — so it is importable anywhere.
`state_dir()` resolves `$FACE_RECOGNITION_STATE_DIR`, else
`$XDG_STATE_HOME/face-recognition-cli`, else
`~/.local/state/face-recognition-cli`. Under it:

- `banks/<name>/` — one self-contained index per consumer. `resolve_bank()`
  takes an explicit name, else `$FACE_RECOGNITION_BANK`, else `default`. Bank
  isolation is pure path composition *above* the store; the store implements
  nothing for it.
- `models/` — the ONNX pair, **bank-independent**. Banks partition identities,
  not models, and SFace alone is ~37 MB.

### The CLI surface (`face_recognition_cli/cli/_commands/`)

Five verbs, each `--json`, each with an `explain/catalog.py` `ENTRIES` key
(plus a `("banks",)` concept entry — vocabulary the `--bank` flag needs, with no
parser of its own):

- **`enroll`** / **`match`** take `--image <path>` or `-` for encoded bytes on
  stdin, decoded by `_frames.load_frame`. Its diagnosis order is deliberate:
  bytes are read and checked for emptiness *before* `cv2` is probed, so a typo'd
  path is exit 1 ("check the path") rather than exit 2 ("install a 60 MB
  wheel") on a box that merely lacks OpenCV.
- **`match`** treats a no-match as a *result*, not an error — exit 0 with
  `no match` / `{"bank": …, "match": null}` — so it works in a shell
  conditional. `--threshold` overrides the store's 0.5 floor for one call.
- **`list`** (module `list_faces.py`, named to avoid shadowing the builtin) and
  **`forget <face_id>`** are the inventory and single-delete verbs; `forget` on
  an unknown id is a structured user error pointing at `list`.
- **`forget-all`** is destructive and irreversible, so it obeys the mesh
  write-verb rule: **dry-run by default, `--apply` commits.** Without `--apply`
  nothing is deleted and it reports the would-delete set (count, ids, names);
  with `--apply` it forgets each id in the selected bank only. Every other bank
  is untouched either way.
- Every store-touching verb takes `--bank` and echoes the resolved bank back in
  its `--json` payload — including `match` on a miss, since a null result is
  only meaningful next to the bank it was searched against.

### The consumer API (`face_recognition_cli/api.py`)

`build_face_recognition(*, models_dir=None, store_base_dir=None) ->
tuple[FaceEngine, FaceStore] | None`, re-exported from the package root. It is
deliberately the same call shape `reachy/behavior/face_sense.py` already uses:
probe for opencv with `importlib.util.find_spec` **before** any lazy import,
return `None` (never raise) when the stack is unavailable, keep `models_dir=` /
`store_base_dir=` injectable. The missing-extra warning is emitted exactly once
per process via a module-level `_WARNED` latch — the extra's absence is a
property of the process, not of a caller. A broken-but-present stack degrades
the same way. This is a published contract with an external consumer; do not
change its shape unilaterally.

### `[cpu]` and `[gpu]` — the hard invariant

Both extras install the **same** `opencv-python-headless>=4.9,<5` (the bound
`reachy-mini-cli`'s `[vision]` uses) and load the **same** two ONNX files. They
are compute-class extras, not model-class extras:

- **`[cpu]`** — the Raspberry-Pi-class robot box, the default documented path.
- **`[gpu]`** — DGX Spark / Jetson / RTX-class hosts, for throughput.

**Non-negotiable: both paths produce the same embedding space.** A face enrolled
on a Jetson must match on the Pi and vice versa. The v1 mechanism is
`engine._try_enable_cuda`, and it is deliberately tiny — it changes *where*
inference executes, never *which model* executes:

- It probes `cv2.cuda.getCudaEnabledDeviceCount()` and, when positive, selects
  `DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA`. Every branch degrades to plain CPU;
  a GPU that cannot be used is not an error.
- **Only the detector is offered to the GPU.** The recognizer stays on the CPU
  on purpose: it emits the stored embedding, so keeping it on one backend takes
  even float-noise off the table for the same-space invariant, while bounding
  boxes are consumed as pixel geometry and are insensitive to a last-ulp
  difference. Detection also runs on every frame while embedding runs once per
  enrol/match, so that is where the throughput is anyway.
- **The PyPI wheels ship no CUDA DNN**, so this is a silent no-op for a normal
  pip install; the capability comes from a locally built OpenCV, which
  Jetson/DGX images typically carry. The `[gpu]` extra therefore declares an
  intent, not a guarantee — the build backing it decides.

Never swap in a different model on the GPU path: that silently forks the
embedding space and every stored face becomes unmatchable on the other machine.
onnxruntime-gpu over the *identical* ONNX files is the documented future path if
the DNN backend stops being enough. If a second model ever becomes necessary,
the store must record which model produced each embedding and refuse
cross-model matches. Say so explicitly rather than letting it happen quietly.

The lazy-import discipline holds across both: a bare install with **neither**
extra stays importable, the introspection and store-only verbs still work, and
the engine verbs fail with a clean exit-2 naming the right extra.

## Decisions (recorded)

Issue #1 left five questions open. All five are settled and shipped; the
reasoning is in [`docs/specs/`](docs/specs/) and the frame under `.devague/`.

1. **State location — this package's own XDG state dir, partitioned into
   per-consumer banks.** Not one store both tools read: identities live in
   `<state dir>/banks/<name>/`, models are shared at `<state dir>/models/`, and
   selection is `--bank` → `$FACE_RECOGNITION_BANK` → `default`
   (`$FACE_RECOGNITION_STATE_DIR` overrides the root, which is how tests stay
   off a real home directory). **Already-enrolled faces keep working** because
   the format did not change and `base_dir=` is injectable — `reachy-mini-cli`
   points it at its existing `<state dir>/faces` tree and reads exactly what it
   wrote.
2. **Loop ownership stays with the caller.** No watch mode; `detect` is
   synchronous and the CLI is one-shot. This was already the non-goal and it did
   not move.
3. **Frames come in; cameras stay out.** `--image <path>` or `-` for encoded
   bytes on stdin. `webcam-cli` (capture) and `media-cli` (device plane) keep
   their lane, and the composition between lanes is a plain pipe. **Known gap:**
   `webcam-cli` has no single-still capture verb to feed this yet — that belongs
   in the cross-repo coordination issue, not in an assumption made here.
4. **The temporary tier is kept, ported as-is.** No consumer uses it — reachy
   never called it either, and the tier is memory-only so the on-disk index is
   identical either way. It is the enrolment half of a "who are you?" flow;
   carrying ~30 lines of dead-but-tested surface beats re-deriving it. Its
   unused status is recorded in `store.py`'s docstring so nobody mistakes it for
   a live code path.
5. **Consent and privacy — a README section, written against the shipped code.**
   Embeddings and the identity index are local only, with no telemetry; the sole
   network path is the one-time model-zoo download in `engine._download`;
   deletion is a first-class verb (`forget` / `forget-all`, per bank); and what
   is stored is a 128-dim vector plus a name and a 4-char id, not a photograph.

Two further decisions the build forced:

- **`numpy` is a base dependency** (`dependencies = ["numpy>=1.26,<3"]`). The
  engine's arrays and the store's `.npy` files both need it, so it cannot sit
  behind an extra; the template's zero-runtime-deps property was deliberately
  given up for it and nothing else.
- **The backend stays `colleague`.** The reconciliation issue #1 asks for was
  deferred rather than folded into a domain-code extraction — see
  [Identity and the backend mismatch](#identity-and-the-backend-mismatch) for
  the three-part move it still needs.

Still genuinely open, and blocking the release rather than the implementation:
whether a PyPI / TestPyPI **Trusted Publisher** is registered for this project
(see [CI / release](#ci--release)) — unverifiable from the repo alone.

## The `reachy-mini-cli` migration (next)

`reachy-mini-cli` is the first consumer and the reason this repo exists. **This
package's side is done** — the engine, the store, the extras and
`build_face_recognition()` in the call shape `face_sense.py` already uses are
all on disk. Two steps remain, in this order: **publish a usable release**
(task `t12`), then **open the cross-repo issue** (`t13`).

Reachy's side, for that issue: delete `reachy/vision/face.py` + `face_store.py`,
depend on this package, rewire `reachy/behavior/face_sense.py`
(`build_face_recognition`) and the `[vision]` extra, and keep already-enrolled
faces under its `state_dir()/faces` working by passing that path as
`store_base_dir=`. The `webcam-cli` single-still feeder gap belongs in the same
conversation.

**Do not push changes into `reachy-mini-cli` yourself.** Open an issue on that
repo (the `communicate` skill does cross-repo issues), agree the API and the
state-dir/migration question with its agent, and let it do its own side.
Sequence it so reachy is never broken: **publish a usable release first, migrate
second.**

## CI / release

Two workflows in `.github/workflows/`:

- **`tests.yml`** — `test` (pytest + coverage → SonarCloud, gated on
  `env.SONAR_TOKEN != ''` so fork PRs stay green), `lint` (the stack above plus
  `teken cli doctor . --strict`), and `version-check`.
- **`publish.yml`** — pushing to `main` publishes to PyPI via Trusted
  Publishing; PRs do a TestPyPI dry-run (`<version>.dev<run_number>`). Both are
  path-filtered to `pyproject.toml` and `face_recognition_cli/**`. Fork PRs skip
  the publish job (no OIDC context).

**Every PR bumps the version — even docs/config/CI.** `version-check` compares
`pyproject.toml` against `origin/main` and comments on the PR when they match.
Use the `version-bump` skill; write a real `CHANGELOG.md` entry (the script
leaves the sections blank and nothing fails on an empty one).

**Before the first real release:** verify a PyPI / TestPyPI **Trusted Publisher**
is registered for `face-recognition-cli` and the `pypi` / `testpypi` GitHub
environments exist — `guild create` configures the GitHub side only. This has
not been confirmed yet.

Note that `CHANGELOG.md` entries at **0.6.1 and below** are the
`culture-agent-template` history inherited by the scaffold, not this agent's;
`0.7.0` is the first entry that describes `face-recognition-cli` itself. The
engine extraction has not been released yet — `t12` is the version bump and
changelog entry that covers it.

## Skills (`.claude/skills/`)

The canonical guildmaster kit, vendored **cite-don't-import**. Provenance and
the re-sync procedure live in [`docs/skill-sources.md`](docs/skill-sources.md) —
including the tracked local divergences (`agex` → `devex`, `outsource` →
`ask-colleague`, and four devague-origin skills vendored directly from devague).
**Do not reformat or edit the vendored scripts** — re-sync instead; if a change
is needed, lift it upstream into guildmaster first.

Tooling prerequisites: **`devex`** on PATH (the `cicd` skill delegates the PR
lifecycle to `devex pr`) and **`agtag`** on PATH (the `communicate` skill wraps
`agtag issue`). **`colleague`** on PATH is optional — `ask-colleague` exits with
an install hint if absent.

The ones you will actually reach for here: `cicd` (PRs + SonarCloud gating),
`version-bump`, `run-tests`, `communicate` (the cross-repo issue to
`reachy-mini-cli`), `ask-colleague`, and the devague chain
(`scope` → `think` → `challenge` → `spec-to-plan` → `assign-to-workforce` →
`summarize-delivery`, with `deviate` for mid-run divergence).

## Conventions and workflow

- **PRs** go through the `cicd` skill. Sign online posts as
  `- face-recognition-cli (Claude)` — the `cicd` / `communicate` scripts resolve
  the nick from `culture.yaml` automatically, so don't add the signature by hand
  in bodies those scripts author.
- **Reach for `ask-colleague` reflexively** — its value is a *second,
  independent mind* (a different backend/model), not a stronger one. Before
  presenting or opening a PR on a non-trivial committed diff, run `review`; for
  a fresh read of an unfamiliar area, run `explore`. Both are read-only
  (throwaway worktree, zero side effects). `write --apply` / `write --pr` needs
  the user's go-ahead. Its output is a second opinion to verify and own, never
  authority.
- **Git worktrees you create live in
  `../.worktrees.face-recognition-cli/<name>/`** — one repo-named directory
  beside the checkout, one subfolder per worktree:

  ```bash
  git worktree add ../.worktrees.face-recognition-cli/<name> -b <branch>
  ```

  Never a shared `../worktrees/`: this workspace holds many sibling projects, so
  a generic folder accumulates orphaned trees from several repos with nothing
  indicating ownership, and a sweep-up cannot tell a live lane from junk. Use a
  branch prefix scoped to the work (`engine/extract`, not `agent/t2`) — plain
  `agent/*` collides with leftovers from earlier fan-outs and `git worktree add
  -b` fails on an existing branch. The vendored `assign-to-workforce` skill's
  fan-out example uses *both* the shared path and `agent/<task-id>`; it is cited
  verbatim and must not be edited, so override both when following it. Remove a
  worktree with `git worktree remove <path>` (`prune` only clears metadata for
  directories already gone). Exception: `ask-colleague`'s read-only verbs create
  their own throwaway worktree under `${TMPDIR:-/tmp}` and reap it on an EXIT
  trap — outside this rule, not a violation of it.
- **Memory discipline — recall before, remember after.** `/recall` before a
  non-trivial task (prior decisions, gotchas, "have we done this before?");
  `/remember` when a non-obvious decision, constraint, or costly gotcha
  surfaces, as it happens. The wrappers default to this agent's scope with
  `--visibility public`, which routes to `<repo-root>/.eidetic/memory`
  (committed, shared with mesh peers on both backends); `--visibility private`
  routes to `$HOME/.eidetic/memory` and is never committed. `/recall` reads both
  and merges. Don't store what the repo already records — store what you would
  otherwise re-derive.

This file describes the repository **as it exists on disk today**, with planned
work explicitly marked. When you edit it, keep claims grounded in checked-in
reality; if a section drifts ahead of reality, mark it `(planned)` — and when a
planned piece lands, move it into
[The domain work](#the-domain-work-landed) and describe what is actually there.
