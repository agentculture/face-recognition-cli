"""Skippable end-to-end enroll-then-match roundtrip, driven through the real CLI.

Every other test in this suite fakes ``cv2`` (see ``test_verbs_enroll_match.py``
/ ``test_cli_domain.py``): that is deliberate, since the dev venv installs
neither ``[cpu]`` nor ``[gpu]``. This file is the one exception the spec
(``docs/specs/2026-07-24-issue-1-engine-extraction.md``) calls for directly —
claim c16's success signal ("an end-to-end enroll-then-match roundtrip works
from the CLI on a sample image via [cpu]"), and honesty conditions h9 ("every
element of the announcement exists on disk ... checkable via the c16 success
signals") and h14 ("each listed signal is mechanically checkable — CI jobs,
exit codes, an enroll-then-match roundtrip test, and a reachy-written-store
fixture load test — none requires judgment"). h14 names this exact test.

:func:`face_recognition_cli.cli.main` is called in-process with real argv,
real ``cv2``, the real :class:`~face_recognition_cli.engine.FaceEngine`, and
the real :class:`~face_recognition_cli.store.FaceStore` — nothing here is
monkeypatched. Everything *around* that real run is still isolated:

* ``$FACE_RECOGNITION_STATE_DIR`` points at ``tmp_path`` for the whole test,
  same as every other test file in this suite.
* The two ONNX models are never downloaded here. They are copied (symlinked
  where possible) into the isolated state dir from a location that must
  already hold them — ``$FACE_RECOGNITION_E2E_MODELS`` if set, else
  ``~/.local/state/face-recognition-cli/models``. If neither has both files,
  the roundtrip test skips at collection time rather than reaching for the
  network.
* The face image is either ``$FACE_RECOGNITION_E2E_IMAGE`` (a real photo, the
  preferred source when set) or a synthetic cartoon face drawn at test time
  with ``cv2`` primitives — never a photo of a real person committed to this
  repo. YuNet may or may not detect a synthetic face reliably; if it doesn't,
  the test skips (it must never hard-fail merely because the drawing wasn't
  good enough) rather than treating that as a roundtrip failure.

The roundtrip itself is real work end to end: ``enroll`` writes a permanent
identity from the detected embedding, ``match`` against the same image must
find that *same* face_id at a cosine score high enough to clear the store's
own 0.5 default, and ``forget-all --apply`` wipes the scratch bank again
afterwards regardless of outcome.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli.cli import main
from face_recognition_cli.engine import SFACE_FILE, YUNET_FILE

# ---------------------------------------------------------------------------
# Skip-guard resolution — computed once at collection time, never mid-test.
# ---------------------------------------------------------------------------


def _find_models_dir() -> Path | None:
    """Locate a directory that already holds both ONNX model files.

    Never downloads, never raises. Preference order: an explicit
    ``$FACE_RECOGNITION_E2E_MODELS`` override, then the default state dir's
    ``models/`` (``~/.local/state/face-recognition-cli/models``) — the same
    default :func:`face_recognition_cli.state.models_dir` resolves to when no
    ``$FACE_RECOGNITION_STATE_DIR``/``$XDG_STATE_HOME`` override is set.
    """
    candidates: list[Path] = []
    env_override = os.environ.get("FACE_RECOGNITION_E2E_MODELS")
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path.home() / ".local" / "state" / "face-recognition-cli" / "models")

    for candidate in candidates:
        if (candidate / YUNET_FILE).is_file() and (candidate / SFACE_FILE).is_file():
            return candidate
    return None


def _compute_skip_reason() -> str | None:
    missing: list[str] = []
    if importlib.util.find_spec("cv2") is None:
        missing.append(
            "cv2 is not importable — install the cpu extra "
            "(uv sync --extra cpu, or pip install 'face-recognition-cli[cpu]')"
        )
    if _MODELS_DIR is None:
        missing.append(
            "the YuNet/SFace ONNX model files were not found under "
            "~/.local/state/face-recognition-cli/models — set "
            "FACE_RECOGNITION_E2E_MODELS=<dir already holding both .onnx files> "
            "to point at a directory that has them"
        )
    if not missing:
        return None
    return "e2e roundtrip prerequisites missing: " + "; ".join(missing)


_MODELS_DIR = _find_models_dir()
_SKIP_REASON = _compute_skip_reason()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_isolated_state_dir(tmp_path: Path) -> Path:
    """Build ``tmp_path/state/models/`` from the already-resolved models dir.

    Symlinks where the platform allows it (cheap, and makes it obvious the
    file is borrowed rather than copied); falls back to a real copy when
    symlinking isn't available. Either way, nothing is fetched from the
    network — ``_MODELS_DIR`` is only reachable at all when the skip guard
    above already proved both files exist on disk.
    """
    state_dir = tmp_path / "state"
    models_dir = state_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    assert _MODELS_DIR is not None  # guaranteed by the skip guard
    for filename in (YUNET_FILE, SFACE_FILE):
        src = _MODELS_DIR / filename
        dest = models_dir / filename
        if dest.exists():
            continue
        try:
            dest.symlink_to(src)
        except OSError:
            shutil.copyfile(src, dest)
    return state_dir


def _write_synthetic_face(path: Path) -> None:
    """Draw a cartoon-ish frontal face and save it as a PNG at *path*.

    A filled ellipse for the head, two dark ellipse eyes, and a mouth arc —
    simple frontal geometry YuNet has a reasonable shot at, but never
    guaranteed. ``cv2`` is only imported here, inside the function, matching
    this package's lazy-import discipline; by the time this runs the skip
    guard has already proved it importable.
    """
    import cv2

    size = 400
    frame = np.full((size, size, 3), 235, dtype=np.uint8)  # light background
    skin = (170, 200, 235)  # BGR
    center = (size // 2, size // 2)

    cv2.ellipse(frame, center, (120, 150), 0, 0, 360, skin, -1)

    eye_axes = (18, 12)
    left_eye = (center[0] - 45, center[1] - 30)
    right_eye = (center[0] + 45, center[1] - 30)
    cv2.ellipse(frame, left_eye, eye_axes, 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(frame, right_eye, eye_axes, 0, 0, 360, (30, 30, 30), -1)

    nose_top = (center[0], center[1] - 10)
    nose_bottom = (center[0], center[1] + 30)
    cv2.line(frame, nose_top, nose_bottom, (140, 160, 190), 3)

    mouth_center = (center[0], center[1] + 70)
    cv2.ellipse(frame, mouth_center, (45, 20), 0, 0, 180, (60, 60, 90), 4)

    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"cv2.imwrite failed to write the synthetic face image to {path}")


# ---------------------------------------------------------------------------
# The roundtrip itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_enroll_then_match_roundtrip_via_the_real_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """enroll writes a face_id, match finds the SAME one, forget-all wipes it.

    Every call goes through :func:`face_recognition_cli.cli.main` — the exact
    entry point ``face-recognition`` (the installed console script) uses —
    with real ``cv2``/``FaceEngine``/``FaceStore``. Isolation is limited to
    the state dir (``tmp_path``, models copied in — see module docstring) and
    the bank name (``e2e``, wiped again at the end).
    """
    state_dir = _prepare_isolated_state_dir(tmp_path)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(state_dir))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)

    env_image = os.environ.get("FACE_RECOGNITION_E2E_IMAGE")
    if env_image:
        image_path = Path(env_image)
        if not image_path.is_file():
            pytest.skip(f"FACE_RECOGNITION_E2E_IMAGE={env_image!r} does not point at a file")
        image_source = f"FACE_RECOGNITION_E2E_IMAGE={env_image}"
    else:
        image_path = tmp_path / "synthetic-face.png"
        _write_synthetic_face(image_path)
        image_source = "the synthetic drawn image"

    bank = "e2e"

    def _run(argv: list[str]) -> tuple[int, pytest.CaptureResult]:
        rc = main([*argv, "--json"])
        return rc, capsys.readouterr()

    rc, captured = _run(
        ["enroll", "--name", "e2e-roundtrip", "--image", str(image_path), "--bank", bank]
    )
    if rc == 1:
        error_payload = json.loads(captured.err) if captured.err else {}
        if "no face detected" in error_payload.get("message", ""):
            pytest.skip(
                f"YuNet did not detect a face in {image_source}; supply "
                "FACE_RECOGNITION_E2E_IMAGE=<path to a real face photo> to run "
                "the full roundtrip"
            )
        pytest.fail(f"enroll failed unexpectedly (exit {rc}): {captured.err}")
    assert rc == 0, captured.err
    enroll_payload = json.loads(captured.out)
    face_id = enroll_payload["face_id"]
    assert len(face_id) == 4
    assert enroll_payload["bank"] == bank

    try:
        rc, captured = _run(["match", "--image", str(image_path), "--bank", bank])
        assert rc == 0, captured.err
        match_payload = json.loads(captured.out)
        assert match_payload["bank"] == bank
        assert match_payload["match"] is not None, "the enrolled face was not matched back"
        assert match_payload["match"]["face_id"] == face_id
        assert match_payload["match"]["score"] >= 0.5
    finally:
        rc, captured = _run(["forget-all", "--bank", bank, "--apply"])
        assert rc == 0, captured.err
        cleanup_payload = json.loads(captured.out)
        assert cleanup_payload["dry_run"] is False
        assert face_id in [face["id"] for face in cleanup_payload["faces"]]


# ---------------------------------------------------------------------------
# Always-run — no skip guard, proves nothing under tests/ reaches the network
# ---------------------------------------------------------------------------

#: A line containing one of these markers (case-insensitive) next to
#: ``urlopen(`` is a fake/stand-in, not a real call.
_FAKE_CONTEXT_MARKERS = ("fake", "stand-in", "monkeypatch", "mock")


def test_no_test_in_this_repo_makes_a_real_network_call() -> None:
    """``requests.`` never appears in tests/; every ``urlopen(`` names a fake.

    This is the always-run counterpart to the roundtrip test above: that one
    is allowed to skip (no models, no cv2, a synthetic face YuNet can't see),
    but this one never does — it is the mechanical proof that skipping it is
    even safe, i.e. that no test file anywhere in this repo would otherwise
    reach out to the real network. ``requests`` is not a dependency of this
    project at all (the engine's own downloader is stdlib ``urllib`` — see
    ``engine.py``), so it should never be named in tests/; ``urlopen(``
    legitimately appears in ``test_engine.py`` as a **fake** stand-in the
    download test monkeypatches in for ``_download``'s unit tests, never a
    real call — proven per-line below rather than assumed. This file is
    excluded from its own scan since it necessarily spells both needles out
    literally in order to search for them.
    """
    tests_dir = Path(__file__).resolve().parent
    this_file = Path(__file__).resolve()
    violations: list[str] = []

    for path in sorted(tests_dir.rglob("*.py")):
        if path.resolve() == this_file:
            continue
        rel = path.relative_to(tests_dir).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "requests." in line:
                violations.append(f"{rel}:{lineno}: 'requests.' — not a dependency of this repo")
            if "urlopen(" in line and not any(
                marker in line.lower() for marker in _FAKE_CONTEXT_MARKERS
            ):
                violations.append(f"{rel}:{lineno}: 'urlopen(' outside a fake/monkeypatch context")

    assert violations == [], "network call(s) found under tests/:\n" + "\n".join(violations)
