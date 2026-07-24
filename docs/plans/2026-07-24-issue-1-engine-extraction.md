# Build Plan — issue-1 engine extraction

slug: `issue-1-engine-extraction` · status: `exported` · from frame: `issue-1-engine-extraction`

> face-recognition-cli ships the extracted YuNet+SFace engine and identity store from reachy-mini-cli: enroll, match, list, forget, forget-all verbs, cpu/gpu extras over the same ONNX models, and a stable API reachy-mini-cli can import

## Tasks

### t1 — State-dir + bank layout module (face_recognition_cli/state.py): XDG chain FACE_RECOGNITION_STATE_DIR -> XDG_STATE_HOME/face-recognition-cli -> ~/.local/state/face-recognition-cli; bank_dir(name) -> <state>/banks/<name>; shared models_dir() -> <state>/models; default bank 'default'; FACE_RECOGNITION_BANK env override

- covers: c12
- acceptance:
  - test_state.py proves the three-step resolution chain with env overrides, bank_dir('reachy') != bank_dir('default'), and models_dir() is bank-independent
  - pure stdlib module: no numpy, no cv2, importable on a bare install

### t2 — Packaging: numpy joins base dependencies; [cpu] = opencv-python-headless>=4.9,<5; [gpu] over the identical ONNX models (mechanism decided here and documented)

- covers: c7
- acceptance:
  - uv sync succeeds; 'import face_recognition_cli' works in a venv with base deps only
  - pyproject diff shows numpy in dependencies and both extras; no second model URL/filename introduced anywhere

### t3 — Port FaceStore -> face_recognition_cli/store.py faithful to reachy's face_store.py: two tiers, cosine match, secrets 4-char ids, write-then-replace save, corrupt-index degrade, base_dir=/clock=/now= seams; default base_dir = default bank via state.py

- depends on: t1, t2
- covers: c2, h2
- acceptance:
  - a fixture tree written by reachy's face_store.py (checked into tests/fixtures/) loads through the ported store with identical list_faces and match results
  - enroll/match/forget/list/get_unique_id/temp-tier behaviour matches reachy's test_vision_face.py store cases, ported and green

### t4 — Port FaceEngine -> face_recognition_cli/engine.py faithful to reachy's face.py: lazy cv2, YuNet+SFace, largest-face, .part-rename downloads behind size floors; CliError rehomed to face_recognition_cli.cli._errors naming the [cpu] extra; models default to state.models_dir()

- depends on: t1, t2
- covers: c2, c3, h3
- acceptance:
  - grep 'reachy' over face_recognition_cli/ returns nothing; missing-cv2 path raises CliError exit 2 whose remediation names face-recognition-cli[cpu]
  - engine unit tests run with a faked cv2 (no network, no real models): URL constants unchanged from reachy's, size-floor rejection and .part cleanup proven
  - EMBEDDING_DIM=128 and both model URLs/filenames byte-identical to reachy's face.py

### t5 — Public consumer API: build_face_recognition(*, models_dir=None, store_base_dir=None) -> (engine, store) | None at package root, find_spec probe before lazy import, one process-wide warning, never raises

- depends on: t3, t4
- covers: c4, h4
- acceptance:
  - without cv2 installed: returns None, warns exactly once per process, raises nothing
  - with cv2 faked/present: returns (FaceEngine, FaceStore) honouring models_dir=/store_base_dir= — call-shape identical to reachy's face_sense.py usage

### t6 — Verbs enroll + match (cli/_commands/enroll.py, match.py only — no shared-file edits): --image <path> or '-' reads image bytes from stdin (cv2.imdecode); --bank selects the store; enroll takes --name; match reports face_id/name/score or no-match

- depends on: t3, t4
- covers: c5
- acceptance:
  - enroll from a file path and from stdin both create a record in the selected bank and print the new id (text + --json)
  - match honours --threshold and returns distinct exit-0 no-match shape vs errors; missing [cpu] yields exit-2 CliError with hint

### t7 — Verbs list + forget + forget-all (cli/_commands/list.py, forget.py, forget_all.py only — no shared-file edits): forget-all dry-runs by default reporting count/ids/names, --apply commits, scoped to the selected bank

- depends on: t3
- covers: c5, h5
- acceptance:
  - forget-all without --apply deletes nothing (store unchanged on disk) and reports the exact would-delete set; with --apply the selected bank is emptied and other banks untouched
  - list and forget operate per-bank with --json shapes; forget of an unknown id exits 1 with hint

### t8 — Wire + catalog + gates: register all five verbs at cli/__init__.py:91, add explain/catalog.py ENTRIES (five verbs + banks concept), CLI-level two-bank isolation test, bare-install behaviour test, single-model-URL-pair grep test; teken cli doctor . --strict green

- depends on: t5, t6, t7
- covers: c5, h5, c6, h6, h1, h7
- acceptance:
  - explain resolves for enroll/match/list/forget/forget-all; teken cli doctor . --strict reports healthy
  - integration test: enroll into --bank a, then list/match/forget in --bank b see nothing; models dir is shared (one copy of each ONNX filename)
  - without cv2: every introspection verb still works and store-touching verbs exit 2 naming [cpu]; test asserts exactly one pair of model URLs exists in the codebase

### t9 — End-to-end roundtrip test (tests/test_e2e_roundtrip.py): CLI enroll-then-match on a sample image via [cpu], auto-skipped when cv2 or models are unavailable so CI never downloads the 37MB model

- depends on: t8
- covers: c16, h14, h9, c1
- acceptance:
  - with [cpu] + models present locally: enroll a face image then match the same image -> same face_id, score >= threshold, via the installed CLI
  - in CI (no cv2): the test skips cleanly; no test in the suite performs a network download

### t10 — README: install + extras table made real, the five verbs with examples (--image/stdin, --bank), per-consumer banks, and the consent/privacy section (retention, one-verb deletion per bank, embeddings never leave the machine, sole network access = model-zoo download)

- depends on: t8
- covers: c11, h8, c13, h11
- acceptance:
  - every privacy statement is true of the shipped code (the download function is the only network path) and both audiences are documented: import API for reachy, CLI + banks for agents/operators
  - markdownlint-cli2 green; Roadmap/status blocks no longer claim the engine is unimplemented

### t11 — CLAUDE.md realignment: move landed work out of 'The domain work (planned)', record the five resolved decisions (numpy base dep, XDG state dir + banks, temp tier kept, image path + stdin input, backend stays colleague), update the zero-deps claim

- depends on: t8
- covers: c15, h13
- acceptance:
  - CLAUDE.md describes the repo as it exists on disk after the merge — no planned-marked section describes shipped code, and the decision log matches the devague frame

### t12 — Release: version-bump skill (minor), real CHANGELOG entry, PR via cicd skill, merge -> publish.yml Trusted-Publishing to PyPI; verify the published wheel installs and drives build_face_recognition

- depends on: t9, t10, t11
- covers: c14, h12, c1
- acceptance:
  - version-check green; publish.yml succeeds on main for the release version
  - pip install face-recognition-cli[cpu] in a fresh venv: whoami works, build_face_recognition importable with reachy's exact call shape

### t13 — Cross-repo migration issue on reachy-mini-cli via the communicate skill: propose the swap (delete face.py/face_store.py, depend on this package, rewire face_sense + [vision]), state-dir/bank options for its existing store, and the webcam-cli feeder gap; publish-first sequencing explicit

- depends on: t12
- covers: c8, h10
- acceptance:
  - issue exists on agentculture/reachy-mini-cli, signed - face-recognition-cli (Claude), opened only after the PyPI release is live; no commit in this plan touches ../reachy-mini-cli

## Risks

- [unknown_nonblocking] PyPI/TestPyPI Trusted Publisher registration for face-recognition-cli is unverified (guild create configured the GitHub side only) — blocks t12's publish, not the implementation (task t12)
- [unknown_nonblocking] [gpu] mechanism open: PyPI opencv wheels ship no CUDA DNN backend, so the extra likely lands as onnxruntime-gpu over the same ONNX files or as a documented locally-built-cv2 path — decided in t2/t4; the same-models invariant holds regardless (task t2)
- [unknown_nonblocking] Engine tests must never download models in CI (37MB SFace); all unit tests fake cv2/models, only the skippable t9 e2e touches real ones (task t9)
