"""Tests for ``face_recognition_cli.api`` — ``build_face_recognition`` (task t5).

TDD — written before ``face_recognition_cli/api.py`` existed, mirroring
reachy-mini-cli's ``reachy/behavior/face_sense.py::build_face_recognition``
(lines 232-274): probe ``cv2`` availability first, never import it merely to
check, log exactly one process-wide warning when the vision stack is
unavailable, and never raise.

The dev venv this suite runs in has **no cv2 installed** — the "without cv2"
tests below exercise the real absence rather than faking it. The "with cv2"
tests monkeypatch the probe seam (``api._cv2_available``) rather than trying
to fake a ``cv2`` module in ``sys.modules``, because construction of
``FaceEngine``/``FaceStore`` never touches ``cv2`` itself (it is imported
lazily, inside ``FaceEngine.detect()``'s first call) — so a real engine/store
pair can be built and inspected with no opencv present at all, as long as the
probe is told to say yes.

Every test that relies on ``FaceStore``'s *default* ``base_dir`` isolates
``$FACE_RECOGNITION_STATE_DIR`` into ``tmp_path`` first, matching
``tests/test_store.py``'s convention — otherwise a default-args call would
create directories under the real home directory.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

import face_recognition_cli
from face_recognition_cli import api
from face_recognition_cli.engine import FaceEngine
from face_recognition_cli.store import FaceStore


@pytest.fixture(autouse=True)
def _reset_warned_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-wide warning latch before every test.

    Without this, whichever test runs first "spends" the one warning and every
    test after it would see zero log records — the latch is module-level by
    design (mirrors reachy's ``_VISION_WARNED``), so tests must reach in and
    reset it themselves rather than relying on import order.
    """
    monkeypatch.setattr(api, "_WARNED", False)


@pytest.fixture(autouse=True)
def _isolate_state_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep default-``base_dir``/``models_dir`` resolution off the real home dir."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


def test_no_cv2_installed_in_this_dev_venv() -> None:
    """Sanity check for the suite's own premise: this venv has no cv2."""
    assert importlib.util.find_spec("cv2") is None


class TestPackageExport:
    def test_package_root_exposes_build_face_recognition(self) -> None:
        assert face_recognition_cli.build_face_recognition is api.build_face_recognition

    def test_package_root_still_exposes_version(self) -> None:
        assert hasattr(face_recognition_cli, "__version__")

    def test_dunder_all_names_both(self) -> None:
        assert "build_face_recognition" in face_recognition_cli.__all__
        assert "__version__" in face_recognition_cli.__all__


class TestWithoutCv2:
    """Exercised against the real, cv2-less dev venv — no monkeypatching of the probe."""

    def test_returns_none(self) -> None:
        assert api.build_face_recognition() is None

    def test_never_raises_with_explicit_dirs(self, tmp_path: Path) -> None:
        result = api.build_face_recognition(
            models_dir=tmp_path / "models", store_base_dir=tmp_path / "store"
        )
        assert result is None

    def test_logs_exactly_one_warning_across_two_calls(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="face_recognition_cli.api"):
            first = api.build_face_recognition()
            second = api.build_face_recognition()

        assert first is None
        assert second is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_warning_names_the_cpu_extra(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="face_recognition_cli.api"):
            api.build_face_recognition()

        assert len(caplog.records) == 1
        assert "face-recognition-cli[cpu]" in caplog.records[0].message


class TestWithCv2Available:
    """``api._cv2_available`` monkeypatched True — the probe seam, not a fake cv2 module."""

    @pytest.fixture(autouse=True)
    def _cv2_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api, "_cv2_available", lambda: True)

    def test_returns_engine_and_store_pair(self) -> None:
        result = api.build_face_recognition()
        assert result is not None
        engine, store = result
        assert isinstance(engine, FaceEngine)
        assert isinstance(store, FaceStore)

    def test_passes_through_models_dir_and_store_base_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        store_base_dir = tmp_path / "store"

        result = api.build_face_recognition(models_dir=models_dir, store_base_dir=store_base_dir)

        assert result is not None
        engine, store = result
        assert engine._models_dir == models_dir
        assert store.base_dir == store_base_dir

    def test_defaults_apply_when_both_none(self) -> None:
        result = api.build_face_recognition()

        assert result is not None
        engine, store = result
        # No explicit models_dir: the engine keeps the "unresolved" None and
        # only resolves lazily on first use (see FaceEngine._resolve_models_dir).
        assert engine._models_dir is None
        # FaceStore resolves eagerly in __init__, so base_dir is always a Path.
        assert isinstance(store.base_dir, Path)

    def test_no_warning_logged_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="face_recognition_cli.api"):
            api.build_face_recognition()

        assert caplog.records == []


class TestConstructionFailure:
    """A broken vision stack (cv2 importable, construction still fails) disables the feature."""

    @pytest.fixture(autouse=True)
    def _cv2_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api, "_cv2_available", lambda: True)

    def test_returns_none_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*args: object, **kwargs: object) -> FaceEngine:
            raise RuntimeError("simulated broken vision stack")

        monkeypatch.setattr("face_recognition_cli.engine.FaceEngine", _boom)

        result = api.build_face_recognition()

        assert result is None

    def test_logs_exactly_one_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> FaceEngine:
            raise RuntimeError("simulated broken vision stack")

        monkeypatch.setattr("face_recognition_cli.engine.FaceEngine", _boom)

        with caplog.at_level(logging.WARNING, logger="face_recognition_cli.api"):
            first = api.build_face_recognition()
            second = api.build_face_recognition()

        assert first is None
        assert second is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
