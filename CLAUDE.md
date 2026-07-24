# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`face-recognition-cli` is an **AgentCulture mesh agent** whose domain is *face
identity*: detect faces, collect embeddings **per identity**, and manage those
identities by a generated **id** and a human **name** — enroll, match, list,
forget one, and forget all at once.

It exists to take over the OpenCV **YuNet + SFace** engine that
[`reachy-mini-cli`](https://github.com/agentculture/reachy-mini-cli) currently
carries in `reachy/vision/face.py` + `reachy/vision/face_store.py`, so the robot
depends on this package instead of shipping its own copy. The full build brief
is [issue #1](https://github.com/agentculture/face-recognition-cli/issues/1) —
read it before starting domain work; it records *why* the engine is extracted
rather than swapped for dlib, and which decisions are still open.

**State of the repo today.** This is still the `culture-agent-template` scaffold
plus this file: an agent-first CLI with the introspection verbs (`whoami`,
`learn`, `explain`, `overview`, `doctor`, `cli overview`), the vendored skill
kit, and the CI/publish baseline. **There is no face-recognition code checked in
yet** — no `FaceEngine`, no `FaceStore`, no `[cpu]`/`[gpu]` extras, no
`enroll`/`match`/`forget` verbs. Everything under
[The domain work](#the-domain-work-planned) is *planned*, and is marked as such.
Keep that boundary honest: when you land a piece, move it out of the planned
section and describe what is actually on disk.

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

## Commands

```bash
uv sync                                              # create .venv, install (dev deps incl. teken)
uv run face-recognition whoami                       # run the CLI
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
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken"
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
  scanner (no YAML library — the runtime package has **zero third-party
  dependencies**, and that is a property worth keeping) and walks up from
  `__file__` so it reports *this agent's* identity, not whatever `culture.yaml`
  sits in the caller's cwd.

The CLI is cited (cite-don't-import) from teken's `python-cli` reference;
`teken` is a dev dependency only.

## The domain work (planned)

None of this is on disk yet. Sources: issue #1 and the sibling checkout at
`../reachy-mini-cli`.

### What to extract, and the one invariant that outranks everything

Take over `reachy/vision/face.py` (engine) and `reachy/vision/face_store.py`
(store) from `reachy-mini-cli`, **keeping the embedding space and the on-disk
layout compatible**. That compatibility is the whole reason extraction beat
adopting dlib:

1. SFace embeddings and dlib `face_recognition` encodings are both 128-dim but
   are *different vector spaces* — switching engines invalidates every face
   already enrolled on the robot and forces real people to re-enrol.
2. dlib needs a cmake/C++ toolchain build on the Raspberry-Pi-class robot box;
   `opencv-python-headless` ships wheels.

### `FaceEngine` (from `reachy/vision/face.py`)

- OpenCV **YuNet** detector + **SFace** 128-dim embedder, largest-face
  selection, `alignCrop` + `feature`. `EMBEDDING_DIM = 128`.
- **`cv2` is imported lazily, inside functions only** — never at module import
  time. The module must stay importable on a bare install with neither extra; a
  missing extra surfaces as a clean exit-2 `CliError` pointing at the right
  extra, not an `ImportError` traceback.
- ONNX models are pulled from the OpenCV model zoo
  (`face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`)
  into `<state dir>/models/`, downloaded to a `.part` sibling and renamed into
  place only after clearing a **size sanity floor** (100 KB / 20 MB). A
  truncated download or a captive-portal HTML page is rejected rather than
  written to disk, where it would "exist" on the next run and fail cryptically
  inside OpenCV.
- `detect(frame)` is **synchronous and stateless-per-call**, returning
  `FaceDetection(bbox_norm, embedding) | None`. Nova's daemon thread and 500 ms
  interval were deliberately *not* ported: loop ownership belongs to the caller.
  Do not re-add a background loop inside the engine.

### `FaceStore` (from `reachy/vision/face_store.py`)

It already satisfies most of the brief. Port it as-is:

| Member | Behaviour |
|---|---|
| `enroll(name, embedding) -> face_id` | Permanent record from a fresh embedding, in one call |
| `match(embedding, *, threshold=None) -> FaceMatch \| None` | Cosine match; `FaceMatch(face_id, name, score)` |
| `forget(face_id) -> bool` | Delete **one** identity |
| `list_faces() -> list[dict]` | Inventory (`id`, `name`, `created`, `num_embeddings`) |
| `get_unique_id(name)` / `permanent_count` | Lookups |
| `remember_temporary` / `get_temporary` / `cleanup_expired` / `temporary_count` | Temporary tier with TTL |
| `load` / `save` | One JSON index (`faces.json`) + one `.npy` per embedding, under `<state dir>/faces` |

Constants: `DEFAULT_MATCH_THRESHOLD = 0.5` (cosine), `DEFAULT_TEMP_TTL = 900`.
Ids are **4-char lowercase alphanumeric** generated with `secrets` — a CSPRNG
chosen to sidestep bandit's insecure-random lint, not for confidentiality.
A corrupt or missing index degrades to "start fresh", never raises; `save` is
write-then-replace. Every time-sensitive method takes `now=`, and the
constructor takes `base_dir=` and `clock=`, so it is deterministic under test —
**keep those seams**, they are why the store is testable without mocking time.

### The genuinely new work

1. **`forget-all`** — the one requirement `FaceStore` does not already meet.
   It is destructive and irreversible, so it obeys the mesh write-verb rule:
   **dry-run by default, `--apply` commits.** Report what *would* be deleted
   (count, ids, names) first.
2. **A real CLI surface.** Inside reachy this was a library with no verbs of its
   own. At minimum `enroll` / `match` / `list` / `forget` / `forget-all`,
   alongside the template's introspection verbs. Every verb takes `--json`, and
   every new verb needs an `explain/catalog.py` entry.
3. **The `[cpu]` / `[gpu]` split** (below).
4. **Packaging a stable public API** that `reachy-mini-cli` can import.
   `reachy/behavior/face_sense.py`'s `build_face_recognition(*, models_dir=None,
   store_base_dir=None) -> tuple[engine, store] | None` is the shape its
   consumer expects — it probes for opencv with `importlib.util.find_spec`,
   imports lazily only after the probe, and returns `None` (never raises) when
   the stack is unavailable. Keep `models_dir=` / `store_base_dir=` injectable.

### `[cpu]` and `[gpu]` — the hard invariant

Follow the extras convention `reachy-mini-cli` already uses (its `[cpu]`/`[gpu]`
are generic compute-class extras; its face engine sits behind `[vision]` =
`opencv-python-headless>=4.9,<5`):

- **`[cpu]`** — the Raspberry-Pi-class robot box. `opencv-python-headless`, the
  default path.
- **`[gpu]`** — DGX Spark / Jetson / RTX-class hosts, for throughput.

**Non-negotiable: both paths run the same ONNX models and produce the same
embedding space.** A face enrolled on a Jetson must match on the Pi and vice
versa. Accelerate the *execution* — OpenCV DNN CUDA backend, or onnxruntime-gpu
over the identical ONNX files — and never swap in a different model on the GPU
path: that silently forks the embedding space and every stored face becomes
unmatchable on the other machine. If a second model ever becomes necessary, the
store must record which model produced each embedding and refuse cross-model
matches. Say so explicitly rather than letting it happen quietly.

Keep the lazy-import discipline across both: a bare install with **neither**
extra must stay importable and fail with a clean exit-2 naming the right extra.

### Open questions — decide, then record the decision here

From the brief; none are settled. Record each answer in this file (and the
README where operator-facing) as you land it:

1. **State location.** Reachy stores faces under its own per-user state dir
   (`$REACHY_STATE_DIR`, else `$XDG_STATE_HOME/reachy`, else the XDG-default
   state directory under the user's home — see `reachy/daemon.py`'s
   `state_dir()`), with faces at `<state dir>/faces`. A standalone tool needs
   its own default — one store both read, or a documented import path?
   Already-enrolled faces must keep working either way.
2. **Whose loop?** `detect` is deliberately synchronous and loop-free. Does the
   CLI grow a continuous watch mode, or stay one-shot and leave loops to callers?
3. **Where do frames come from?** Take a frame/image path and stay engine-only,
   or open cameras? Engine-only (fed by `webcam-cli` / `media-cli`) is the
   cleaner lane split — but agree it with those agents, don't assume it.
4. **Temporary tier.** Keep the 15-min TTL tier (it exists for a "who are you?"
   enrolment flow that was never built), or drop it as unused surface?
5. **Consent and privacy.** This repo is **public** and the tool stores
   biometric identifiers of real people. Decide retention, how easy deletion is,
   and whether embeddings ever leave the machine — then write it in the README.
   Nothing here phones home; the only network access in the extracted engine is
   the one-time model-zoo download.

### The `reachy-mini-cli` migration

`reachy-mini-cli` is the first consumer and the reason this repo exists. Its
side: delete `reachy/vision/face.py` + `face_store.py`, depend on this package,
rewire `reachy/behavior/face_sense.py` (`build_face_recognition`) and the
`[vision]` extra, and keep already-enrolled faces under its `state_dir()/faces`
working.

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

Note that `CHANGELOG.md` currently carries the **template's** history (up to
0.6.1), inherited by the scaffold — entries below the first
`face-recognition-cli` release describe `culture-agent-template`, not this
agent.

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
reality; if a section drifts ahead of reality, mark it `(planned)` or move it
under [The domain work](#the-domain-work-planned).
