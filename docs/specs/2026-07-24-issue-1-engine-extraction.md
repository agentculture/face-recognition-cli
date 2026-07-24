# issue-1 engine extraction

> face-recognition-cli ships the extracted YuNet+SFace engine and identity store from reachy-mini-cli: enroll, match, list, forget, forget-all verbs, cpu/gpu extras over the same ONNX models, and a stable API reachy-mini-cli can import
> instruction: Implement as the extraction PR series described by the frame: port the two modules, add banks + verbs + extras, publish, then open the cross-repo migration issue — publish-first, migrate-second

## Audience

- reachy-mini-cli (the first consumer, importing build_face_recognition), and mesh agents/operators driving the CLI directly — each consumer with its own isolated face bank (reachy's, colleague's, ...)

## Before → After

- Before: the engine exists only inside reachy-mini-cli as a library with no CLI verbs of its own; this repo is the culture-agent-template scaffold with zero face-recognition code on disk (CLAUDE.md marks all domain work as planned)
- After: face-recognition-cli is published on PyPI with the ported FaceEngine + FaceStore, per-consumer banks, enroll/match/list/forget/forget-all verbs (all --json), [cpu]/[gpu] extras over the same ONNX models, and a stable build_face_recognition API — so reachy-mini-cli can delete its vision/face copies and depend on this package

## Requirements

- Port reachy/vision/face.py (FaceEngine, 240 lines) and reachy/vision/face_store.py (FaceStore, 327 lines) keeping the SFace 128-dim embedding space and the on-disk layout (faces.json index + one .npy per embedding) byte-compatible — the compatibility invariant is the reason extraction beat dlib per issue #1
  - instruction: Port face.py -> face_recognition_cli/engine.py and face_store.py -> face_recognition_cli/store.py (names free; layout/content faithful), preserving faces.json schema, embeddings/<id>.npy layout, DEFAULT_MATCH_THRESHOLD=0.5, DEFAULT_TEMP_TTL=900, 4-char secrets ids, size floors, .part-rename downloads
  - honesty: Compatibility is provable: a faces.json + embeddings/*.npy tree written by reachy's face_store.py loads through the ported FaceStore with identical match results, and the ported engine downloads the exact same two model-zoo URLs (no model change anywhere in the diff)
- The port must rehome exactly two reachy-internal imports: reachy.cli._errors.CliError/EXIT_ENV_ERROR (face.py:44) maps to the identical face_recognition_cli/cli/_errors.py contract already on disk, and reachy.daemon.state_dir (face.py:103, face_store.py:52) needs a new state-dir helper in this package; remediation strings change from 'reachy-mini-cli[vision]' to this package's extras
  - instruction: Swap reachy.cli._errors -> face_recognition_cli.cli._errors (same contract, already on disk); replace reachy.daemon.state_dir with a new face_recognition_cli state-dir helper implementing the XDG chain + banks layout from c12
  - honesty: After the port, grep for 'from reachy' / 'import reachy' over face_recognition_cli/ returns nothing, and the missing-extra CliError remediation names a face-recognition-cli extra, not reachy's [vision]
- Expose a stable consumer API shaped like reachy/behavior/face_sense.py's build_face_recognition(*, models_dir=None, store_base_dir=None) -> tuple[engine, store] | None: probe opencv via importlib.util.find_spec before any lazy import, return None (never raise) when the stack is absent, keep models_dir=/store_base_dir= injectable — this is the exact call shape the first consumer already uses
  - instruction: Expose build_face_recognition(*, models_dir=None, store_base_dir=None) at a stable import path (package root __init__), probing cv2 via importlib.util.find_spec before any lazy import
  - honesty: On an install without opencv, build_face_recognition() returns None (no raise, one process-wide warning); with [cpu] installed it returns (engine, store) honouring models_dir=/store_base_dir= — matching the exact call shape face_sense.py uses today
- New CLI verbs enroll / match / list / forget / forget-all register through the existing pattern: one module per verb under cli/_commands/ exposing register(sub) with --json and set_defaults(func=...), wired at the marked call site in cli/__init__.py's _build_parser(); forget-all is dry-run by default with --apply to commit (mesh write-verb rule), reporting count, ids, and names before deletion
  - instruction: One module per verb under cli/_commands/ following whoami.py: register(sub), --json, set_defaults(func=...); wire at cli/__init__.py:91; enroll/match take --image <path> or '-' for stdin bytes (cv2.imdecode), --bank on every store-touching verb
  - honesty: All five verbs are registered and pass the structured-error contract (no traceback ever, hint: line present, --json parseable); forget-all without --apply deletes nothing and reports the would-delete set (count, ids, names); with --apply it empties the selected bank only
- Every new verb gets an ENTRIES key in explain/catalog.py — test_every_catalog_path_resolves only verifies existing entries resolve, nothing fails when a verb lacks an entry, so the catalog additions are a deliberate checklist item; the teken rubric gate (26/26, runs the installed CLI end-to-end) must stay green
  - instruction: Add ENTRIES keys for enroll/match/list/forget/forget-all (and the bank concept) in explain/catalog.py in the same PR that adds the verbs; run the rubric gate locally before pushing
  - honesty: face-recognition explain <verb> resolves for every new verb and teken cli doctor . --strict reports healthy after the CLI changes
- [cpu] pins opencv-python-headless>=4.9,<5 (the same bound reachy-mini-cli's [vision] uses) and is the default documented path; [gpu] accelerates execution of the identical ONNX files only (OpenCV DNN CUDA or onnxruntime-gpu) and never swaps models — a bare install with neither extra stays importable and every engine verb fails with a clean exit-2 CliError naming the right extra
  - instruction: [project.optional-dependencies]: cpu = [opencv-python-headless>=4.9,<5]; gpu accelerates execution of the identical ONNX files (opencv contrib CUDA or onnxruntime-gpu — pick at implementation time and document); numpy joins base dependencies per q1
  - honesty: A bare install (pip install face-recognition-cli, no extras) imports and runs every introspection verb; enroll/match exit 2 naming [cpu]; the codebase contains exactly one pair of model URLs/filenames so the gpu path cannot silently reference different models
- README gains a consent/privacy section (brief open question 5): the repo is public and stores biometric identifiers — record retention policy, deletion ease (forget/forget-all), and that embeddings never leave the machine (only network access is the one-time model-zoo download, already true of the ported engine)
  - instruction: Write the section against the code actually landed (cite the download function as the only network path); include per-bank isolation as part of the privacy story
  - honesty: The README privacy section states: embeddings never leave the machine, the only network access is the one-time model-zoo download, deletion is one verb away (forget/forget-all per bank) — and each statement is true of the shipped code
- State layout is per-consumer face banks: <state dir>/banks/<consumer>/ (faces.json + embeddings/) so reachy has its own bank, colleague its own, fully separate — selected via a --bank flag (env-overridable) with a sane default bank; ONNX models stay shared at <state dir>/models/ since both banks embed in the same space and the 37MB SFace file need not be duplicated
  - instruction: Resolve bank paths above the ported store: default_state_dir()/banks/<bank>/ passed as base_dir=, models always default_state_dir()/models; --bank flag on every store-touching verb, FACE_RECOGNITION_BANK env override, default bank name 'default'. Verify: enroll into bank A, then list/match in bank B sees nothing; models dir contains exactly one copy of each ONNX after use from two banks
  - honesty: Two banks are fully isolated (an identity enrolled in one is invisible to list/match/forget in the other) while both share one models/ directory — verified by a test that enrolls in two banks and cross-checks

## Honesty conditions

- Every element of the announcement exists on disk at release — engine, store, banks, the five verbs, extras, build_face_recognition — checkable via the c16 success signals
- No commit in this work touches ../reachy-mini-cli; the only cross-repo artifact is the migration issue, opened only after a usable release is published
- The shipped surface serves both audiences: reachy imports build_face_recognition unchanged in call shape, and operators/agents get isolated per-consumer banks via --bank
- The after-state is reached exactly when publish.yml succeeds on main for the release version and the published wheel satisfies reachy's face_sense call shape
- Grounded in checked-in reality: git ls-files shows no engine/store module today and CLAUDE.md marks all domain work as planned
- Each listed signal is mechanically checkable — CI jobs, exit codes, an enroll-then-match roundtrip test, and a reachy-written-store fixture load test — none requires judgment

## Success signals

- pytest green + teken cli doctor --strict green in CI; a bare install stays importable with engine verbs failing exit-2 naming the extra; an end-to-end enroll-then-match roundtrip works from the CLI on a sample image via [cpu]; and a faces store written by reachy's face_store.py loads unchanged through the ported store (base_dir= pointed at it)

## Scope / boundaries

- reachy-mini-cli's side of the migration (deleting its face.py/face_store.py, rewiring face_sense.py and [vision]) is out of scope here — issue #1 says do not push changes into that repo; the deliverable toward it is a cross-repo issue via the communicate skill, sequenced publish-first-migrate-second

## Non-goals

- No camera access, no watch loop, no threading inside the engine: detect(frame) stays synchronous and stateless-per-call (nova's daemon thread was deliberately not ported — face.py's docstring records why), and face-cli (expressive output) / webcam-cli (capture) / media-cli (device plane) own their lanes — this tool does not absorb them

## Assumptions

- reachy-mini-cli's tests/test_vision_face.py (539 lines) is the porting reference for engine+store tests — the determinism seams (base_dir=, clock=, now=, models_dir=) exist specifically so those tests run without mocking time or network; coverage gate here is fail_under=60 and CI paths (pyproject.toml + face_recognition_cli/**) already cover the new modules with no workflow edits

## Scope exploration

- `s1` — `../reachy-mini-cli/reachy/vision/face.py + face_store.py`: both files read in full; FaceEngine is lazy-cv2 YuNet+SFace with .part-then-rename model downloads behind size floors (100KB/20MB); FaceStore already implements enroll/match/forget/list_faces/get_unique_id plus temp tier; on-disk layout is faces.json + embeddings/<id>.npy
  - seeds: `c2`
- `s2` — `import graph of the two ported modules`: grep of face.py/face_store.py shows only two reachy.* imports (CliError and state_dir); face_recognition_cli/cli/_errors.py already defines the same CliError(code,message,remediation) + EXIT_ENV_ERROR=2 shape, so the engine's error contract transplants unchanged
  - seeds: `c3`
- `s3` — `../reachy-mini-cli/reachy/behavior/face_sense.py (build_face_recognition, lines 232-274)`: read in full: probes vision_unavailable_reason() first, lazy-imports FaceEngine/FaceStore only after the probe, one process-wide warning, returns (engine, store) or None; face_sense calls only engine.detect and store.match — nothing else from the store's surface
  - seeds: `c4`
- `s4` — `face_recognition_cli/cli/__init__.py + cli/_commands/ + _errors.py + _output.py`: read the parser plumbing: _CliArgumentParser json-hint pre-scan, _dispatch wrapping, parser_class propagation, and the 'Register your own noun groups here' call site at cli/__init__.py:91 — new verbs slot in with no plumbing changes; whoami.py is the canonical example
  - seeds: `c5`
- `s5` — `face_recognition_cli/explain/catalog.py + teken rubric gate`: catalog is a dict keyed by command-path tuples with both 'face-recognition-cli' and 'face-recognition' resolving to root; CI's lint job runs 'teken cli doctor . --strict' against the installed CLI, so a broken verb fails CI even when pytest is green
  - seeds: `c6`
- `s6` — `../reachy-mini-cli/pyproject.toml [project.optional-dependencies]`: read the extras block: [vision]=opencv-python-headless>=4.9,<5 with the lazy-import rationale in comments; [cpu]/[gpu] exist there as generic empty compute-class extras — this repo's [cpu]/[gpu] carry the actual engine deps instead, per the brief's steer
  - seeds: `c7`
- `s7` — `pyproject.toml (this repo, dependencies = [])`: confirmed dependencies=[] and no [project.optional-dependencies] section exists yet; reachy-mini-cli carries numpy as a base dep, so the port forces an explicit decision here
- `s8` — `../reachy-mini-cli/reachy/daemon.py state_dir() (lines 43-56)`: read the resolution chain verbatim; both ported constructors already take base_dir=/models_dir= overrides, so the injection seam for reachy's existing data exists regardless of which default this package picks
- `s9` — `grep -rn 'remember_temporary|get_temporary|cleanup_expired|temporary_count' ../reachy-mini-cli/reachy/`: empty result outside face_store.py: the temporary tier is unused by the only consumer, so keeping it is a fidelity choice, not a compatibility requirement — the on-disk index is untouched either way (temp tier is memory-only, never persisted)
- `s10` — `../webcam-cli/README.md (capture surface + lane statement)`: read the README: webcam-cli owns USB capture and 'does not own interpreting what is in a frame (a vision model's job)' — the lane split is already stated from their side; but the single-still 'capture' verb this tool would want as a feeder does not exist yet
- `s11` — `culture.yaml + tests/test_cli.py:44,54 + CLAUDE.md 'Identity and the backend mismatch'`: verified the assertions exist at those lines ('backend: colleague' text and payload['backend']=='colleague' JSON); CLAUDE.md documents the mesh evidence (only steward + this repo declare colleague) and the exact three-part move if flipped
- `s12` — `issue #1 'The reachy-mini-cli migration' section`: the brief is explicit: 'Do not push changes into reachy-mini-cli yourself. Open an issue on that repo... publish a usable release first, migrate second'
  - seeds: `c8`
- `s13` — `face.py module docstring (no-threading deviation) + issue #1 'Your lane'`: face.py's docstring documents 'No threading / no background dispatch loop' as a deliberate deviation from nova with loop ownership assigned to the caller; the brief assigns perceptual-input only, with face-cli explicitly 'share a subject and nothing else'
  - seeds: `c9`
- `s14` — `../reachy-mini-cli/tests/test_vision_face.py + .github/workflows/{tests,publish}.yml`: test file located and sized; publish.yml path filter read — face_recognition_cli/** is already included so domain code triggers publish automatically; version-check job enforces the every-PR bump
  - seeds: `c10`
- `s15` — `issue #1 'Open questions' 5 + face.py download path`: verified in face.py that the only network code is the model-zoo _download with fixed https URLs; nothing else in either ported module touches the network, so the never-phones-home claim is grounded in the code being ported
  - seeds: `c11`
- `s16` — `state-layout refinement (user decision on q2)`: per-consumer banks under <state>/banks/<consumer>/, shared <state>/models/; grounded in s8 (both constructors already take base_dir=/models_dir= so the bank resolver is pure path composition above the ported store, no store changes needed)
  - seeds: `c12`
