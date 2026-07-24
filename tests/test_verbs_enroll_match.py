"""Tests for the ``enroll`` and ``match`` verbs — written test-first (TDD).

These tests define the contract the two image-consuming verbs must satisfy
before either module exists. Four properties shape every test here:

* **No opencv.** The dev environment installs neither ``[cpu]`` nor ``[gpu]``,
  which is exactly the bare-install case the verbs must survive. The
  missing-extra path is proven with ``monkeypatch.setitem(sys.modules, "cv2",
  None)`` and the happy path with a **fake** ``cv2`` module injected the same
  way (the seam ``tests/test_engine.py`` already uses), so every assertion
  holds whether or not opencv happens to be installed.
* **No network, no models.** ``FaceEngine.detect`` is monkeypatched, so the
  lazy model load (and its download) is never reached.
* **No real home.** ``$FACE_RECOGNITION_STATE_DIR`` is pinned into ``tmp_path``
  by an autouse fixture, so enrolments land in a throwaway bank tree.
* **No main() dependency.** The verbs are not wired into
  ``cli/__init__.py`` yet (task t8 owns registration + the explain catalog), so
  each test builds its own ``argparse`` parser, calls ``register(sub)`` on it,
  parses a real argv list, and invokes ``args.func(args)``. That exercises the
  whole verb — flag surface, handler, output — without the unwired main CLI.

Handlers return ``None`` on success; ``cli.__init__._dispatch`` maps that to
exit 0, so "exit 0" is asserted here as "the handler returned None and raised
nothing" (notably for the no-match case, which is a *result*, not an error).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli.cli._commands import _frames
from face_recognition_cli.cli._commands import enroll as enroll_cmd
from face_recognition_cli.cli._commands import match as match_cmd
from face_recognition_cli.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from face_recognition_cli.engine import EMBEDDING_DIM, FaceDetection, FaceEngine

# A byte prefix the fake cv2 treats as "decodable". Real PNG magic, so the
# fixture files on disk are plausible input rather than arbitrary noise.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pin the state dir into a throwaway directory — never the real home."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


def _make_fake_cv2() -> types.ModuleType:
    """A fake ``cv2`` with just the decode surface :func:`load_frame` uses."""
    module = types.ModuleType("cv2")
    module.IMREAD_COLOR = 1  # type: ignore[attr-defined]
    calls: list[tuple[bytes, int]] = []

    def imdecode(buf, flags):  # noqa: ANN001 - fake
        data = np.asarray(buf, dtype=np.uint8).tobytes()
        calls.append((data, flags))
        if data.startswith(_PNG_MAGIC):
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return None  # real cv2 returns None for undecodable bytes

    module.imdecode = imdecode  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _make_fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


@pytest.fixture
def no_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sys.modules[name] = None`` makes ``import name`` raise ImportError."""
    monkeypatch.setitem(sys.modules, "cv2", None)  # type: ignore[arg-type]


def _image_bytes() -> bytes:
    return _PNG_MAGIC + b"pretend-this-is-an-encoded-still"


def _write_image(tmp_path, name: str = "face.png") -> Path:
    path = tmp_path / name
    path.write_bytes(_image_bytes())
    return path


def _feed_stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(data)))


def _unit(axis: int) -> np.ndarray:
    """A 128-dim unit vector along one axis — cosine geometry made obvious."""
    vec = np.zeros(EMBEDDING_DIM, dtype=float)
    vec[axis] = 1.0
    return vec


def _tilted(cosine: float) -> np.ndarray:
    """A unit vector whose cosine against ``_unit(0)`` is exactly *cosine*."""
    vec = np.zeros(EMBEDDING_DIM, dtype=float)
    vec[0] = cosine
    vec[1] = float(np.sqrt(1.0 - cosine**2))
    return vec


def _detection(embedding: np.ndarray) -> FaceDetection:
    return FaceDetection(bbox_norm=(0.1, 0.1, 0.5, 0.5), embedding=embedding)


def _patch_detect(monkeypatch: pytest.MonkeyPatch, result: FaceDetection | None) -> list:
    """Make ``FaceEngine.detect`` return *result*; collect the frames it saw."""
    seen: list = []

    def _detect(self, frame):  # noqa: ANN001 - stand-in
        seen.append(frame)
        return result

    monkeypatch.setattr(FaceEngine, "detect", _detect)
    return seen


def _parser() -> argparse.ArgumentParser:
    """A real parser with both verbs registered — the main CLI is not wired."""
    parser = argparse.ArgumentParser(prog="face-recognition-cli")
    sub = parser.add_subparsers(dest="command")
    enroll_cmd.register(sub)
    match_cmd.register(sub)
    return parser


def _parse(argv: list[str]) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _run(argv: list[str]):
    args = _parse(argv)
    return args.func(args)


# ---------------------------------------------------------------------------
# Argument surface
# ---------------------------------------------------------------------------


class TestArgumentSurface:
    def test_enroll_requires_name(self, tmp_path) -> None:
        image = _write_image(tmp_path)
        with pytest.raises(SystemExit):
            _parse(["enroll", "--image", str(image)])

    def test_enroll_requires_image(self) -> None:
        with pytest.raises(SystemExit):
            _parse(["enroll", "--name", "ada"])

    def test_match_requires_image(self) -> None:
        with pytest.raises(SystemExit):
            _parse(["match"])

    def test_enroll_defaults(self, tmp_path) -> None:
        args = _parse(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path))])
        assert args.bank is None
        assert args.json is False
        assert args.name == "ada"

    def test_match_defaults(self, tmp_path) -> None:
        args = _parse(["match", "--image", str(_write_image(tmp_path))])
        assert args.bank is None
        assert args.threshold is None  # None => the store's own 0.5 default
        assert args.json is False

    def test_match_threshold_parses_as_float(self, tmp_path) -> None:
        args = _parse(["match", "--image", str(_write_image(tmp_path)), "--threshold", "0.83"])
        assert isinstance(args.threshold, float)
        assert args.threshold == pytest.approx(0.83)

    def test_bank_flag_is_accepted_by_both_store_touching_verbs(self, tmp_path) -> None:
        image = str(_write_image(tmp_path))
        assert _parse(["enroll", "--name", "a", "--image", image, "--bank", "reachy"]).bank == (
            "reachy"
        )
        assert _parse(["match", "--image", image, "--bank", "reachy"]).bank == "reachy"

    def test_both_verbs_bind_a_handler(self, tmp_path) -> None:
        image = str(_write_image(tmp_path))
        assert callable(_parse(["enroll", "--name", "a", "--image", image]).func)
        assert callable(_parse(["match", "--image", image]).func)


# ---------------------------------------------------------------------------
# The missing-extra path — one clean exit-2 error shape, shared with the engine
# ---------------------------------------------------------------------------


class TestMissingCv2:
    def test_enroll_without_cv2_raises_exit2_naming_the_cpu_extra(
        self, no_cv2, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = _write_image(tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(0)))

        with pytest.raises(CliError) as excinfo:
            _run(["enroll", "--name", "ada", "--image", str(image)])

        assert excinfo.value.code == EXIT_ENV_ERROR
        assert excinfo.value.code == 2
        assert "face-recognition-cli[cpu]" in excinfo.value.remediation

    def test_match_without_cv2_raises_exit2_naming_the_cpu_extra(
        self, no_cv2, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = _write_image(tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(0)))

        with pytest.raises(CliError) as excinfo:
            _run(["match", "--image", str(image)])

        assert excinfo.value.code == EXIT_ENV_ERROR
        assert "face-recognition-cli[cpu]" in excinfo.value.remediation

    def test_missing_extra_never_leaks_an_import_traceback(self, no_cv2, tmp_path) -> None:
        """The failure is a CliError, not a bare ImportError."""
        with pytest.raises(CliError):
            _frames.load_frame(str(_write_image(tmp_path)))

    def test_a_bad_path_is_diagnosed_before_the_cv2_probe(self, no_cv2, tmp_path) -> None:
        """A typo'd path must not be reported as a missing opencv install."""
        with pytest.raises(CliError) as excinfo:
            _run(["enroll", "--name", "ada", "--image", str(tmp_path / "nope.png")])

        assert excinfo.value.code == EXIT_USER_ERROR
        assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Frame loading — path, stdin, and the two decode failures
# ---------------------------------------------------------------------------


class TestFrameLoading:
    def test_loads_a_file_path_into_a_decoded_frame(self, fake_cv2, tmp_path) -> None:
        frame = _frames.load_frame(str(_write_image(tmp_path)))

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (8, 8, 3)
        assert fake_cv2.calls[0][0] == _image_bytes()
        assert fake_cv2.calls[0][1] == fake_cv2.IMREAD_COLOR

    def test_dash_reads_raw_bytes_from_stdin(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed_stdin(monkeypatch, _image_bytes())

        frame = _frames.load_frame("-")

        assert frame.shape == (8, 8, 3)
        assert fake_cv2.calls[0][0] == _image_bytes()

    def test_missing_file_is_a_user_error(self, fake_cv2, tmp_path) -> None:
        with pytest.raises(CliError) as excinfo:
            _frames.load_frame(str(tmp_path / "absent.png"))

        assert excinfo.value.code == EXIT_USER_ERROR
        assert "path" in excinfo.value.remediation.lower()

    def test_a_directory_path_is_a_user_error(self, fake_cv2, tmp_path) -> None:
        with pytest.raises(CliError) as excinfo:
            _frames.load_frame(str(tmp_path))

        assert excinfo.value.code == EXIT_USER_ERROR

    def test_undecodable_bytes_on_disk_are_a_user_error(self, fake_cv2, tmp_path) -> None:
        garbage = tmp_path / "notes.txt"
        garbage.write_bytes(b"this is definitely not an encoded image")

        with pytest.raises(CliError) as excinfo:
            _frames.load_frame(str(garbage))

        assert excinfo.value.code == EXIT_USER_ERROR
        assert "decode" in excinfo.value.message
        assert "png" in excinfo.value.remediation.lower()

    def test_empty_stdin_is_a_user_error(self, fake_cv2, monkeypatch: pytest.MonkeyPatch) -> None:
        _feed_stdin(monkeypatch, b"")

        with pytest.raises(CliError) as excinfo:
            _frames.load_frame("-")

        assert excinfo.value.code == EXIT_USER_ERROR
        assert "decode" in excinfo.value.message

    def test_empty_stdin_is_diagnosed_without_opencv(
        self, no_cv2, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No bytes is a user error on any install, extra present or not."""
        _feed_stdin(monkeypatch, b"")

        with pytest.raises(CliError) as excinfo:
            _frames.load_frame("-")

        assert excinfo.value.code == EXIT_USER_ERROR

    def test_stdin_without_a_byte_buffer_is_a_user_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace())

        with pytest.raises(CliError) as excinfo:
            _frames.load_frame("-")

        assert excinfo.value.code == EXIT_USER_ERROR

    def test_an_unreadable_stdin_is_a_user_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            def read(self) -> bytes:
                raise OSError("reading from stdin while output is captured")

        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=_Boom()))

        with pytest.raises(CliError) as excinfo:
            _frames.load_frame("-")

        assert excinfo.value.code == EXIT_USER_ERROR


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------


class TestEnroll:
    def test_writes_faces_json_into_the_named_bank(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(
            [
                "enroll",
                "--name",
                "ada",
                "--image",
                str(_write_image(tmp_path)),
                "--bank",
                "reachy",
                "--json",
            ]
        )

        index = tmp_path / "state" / "banks" / "reachy" / "faces.json"
        assert index.exists()
        payload = json.loads(capsys.readouterr().out)
        record = json.loads(index.read_text(encoding="utf-8"))
        assert list(record) == [payload["face_id"]]
        assert record[payload["face_id"]]["name"] == "ada"

    def test_defaults_to_the_default_bank(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path)), "--json"])

        assert (tmp_path / "state" / "banks" / "default" / "faces.json").exists()
        assert json.loads(capsys.readouterr().out)["bank"] == "default"

    def test_bank_env_override_is_honoured(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        monkeypatch.setenv("FACE_RECOGNITION_BANK", "colleague")
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path)), "--json"])

        assert (tmp_path / "state" / "banks" / "colleague" / "faces.json").exists()
        assert json.loads(capsys.readouterr().out)["bank"] == "colleague"

    def test_json_shape(self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path)), "--json"])

        out = capsys.readouterr()
        payload = json.loads(out.out)
        assert set(payload) == {"face_id", "name", "bank", "num_embeddings"}
        assert payload["name"] == "ada"
        assert payload["bank"] == "default"
        assert payload["num_embeddings"] == 1
        assert len(payload["face_id"]) == 4
        assert out.err == ""

    def test_text_output_reports_the_id_and_bank_on_stdout_only(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path)), "--bank", "b1"])

        out = capsys.readouterr()
        assert out.err == ""
        assert "bank: b1" in out.out
        assert "name: ada" in out.out
        # The generated id appears verbatim in the text rendering.
        index = json.loads(
            (tmp_path / "state" / "banks" / "b1" / "faces.json").read_text(encoding="utf-8")
        )
        (face_id,) = list(index)
        assert f"face_id: {face_id}" in out.out

    def test_no_face_detected_is_a_user_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _patch_detect(monkeypatch, None)

        with pytest.raises(CliError) as excinfo:
            _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path))])

        assert excinfo.value.code == EXIT_USER_ERROR
        assert excinfo.value.message == "no face detected in the image"
        assert excinfo.value.remediation

    def test_no_face_detected_writes_nothing(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _patch_detect(monkeypatch, None)

        with pytest.raises(CliError):
            _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path))])

        assert not (tmp_path / "state" / "banks" / "default" / "faces.json").exists()

    def test_the_decoded_frame_is_what_reaches_the_engine(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        seen = _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", str(_write_image(tmp_path)), "--json"])

        assert len(seen) == 1
        assert seen[0].shape == (8, 8, 3)

    def test_enrol_from_stdin(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _feed_stdin(monkeypatch, _image_bytes())
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["enroll", "--name", "ada", "--image", "-", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "ada"
        assert (tmp_path / "state" / "banks" / "default" / "embeddings").is_dir()

    def test_two_enrolments_get_distinct_ids(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        image = str(_write_image(tmp_path))
        _patch_detect(monkeypatch, _detection(_unit(0)))
        _run(["enroll", "--name", "ada", "--image", image, "--json"])
        first = json.loads(capsys.readouterr().out)["face_id"]
        _run(["enroll", "--name", "grace", "--image", image, "--json"])
        second = json.loads(capsys.readouterr().out)["face_id"]

        assert first != second


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------


class TestMatch:
    def _enrol(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
        tmp_path,
        *,
        name: str = "ada",
        bank: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> str:
        image = str(_write_image(tmp_path, name=f"{name}-{bank or 'default'}.png"))
        _patch_detect(monkeypatch, _detection(embedding if embedding is not None else _unit(0)))
        argv = ["enroll", "--name", name, "--image", image, "--json"]
        if bank is not None:
            argv += ["--bank", bank]
        _run(argv)
        return json.loads(capsys.readouterr().out)["face_id"]

    def test_matches_the_enrolled_identity(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        face_id = self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(0)))

        rc = _run(["match", "--image", str(_write_image(tmp_path)), "--json"])

        assert rc is None  # None => exit 0 via _dispatch
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"bank", "match"}
        assert payload["bank"] == "default"
        assert set(payload["match"]) == {"face_id", "name", "score"}
        assert payload["match"]["face_id"] == face_id
        assert payload["match"]["name"] == "ada"
        assert payload["match"]["score"] == pytest.approx(1.0)

    def test_match_text_output_carries_id_name_and_a_4dp_score(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        face_id = self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_tilted(0.75)))

        _run(["match", "--image", str(_write_image(tmp_path))])

        out = capsys.readouterr()
        assert out.err == ""
        assert f"face_id: {face_id}" in out.out
        assert "name: ada" in out.out
        assert "score: 0.7500" in out.out

    def test_a_no_match_is_not_an_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(1)))  # orthogonal => cosine 0

        rc = _run(["match", "--image", str(_write_image(tmp_path)), "--json"])

        assert rc is None
        out = capsys.readouterr()
        assert json.loads(out.out) == {"bank": "default", "match": None}
        assert out.err == ""

    def test_a_no_match_prints_no_match_in_text_mode(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(1)))

        _run(["match", "--image", str(_write_image(tmp_path))])

        out = capsys.readouterr()
        assert out.out.strip() == "no match"
        assert out.err == ""

    def test_an_empty_bank_is_a_no_match_not_an_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _patch_detect(monkeypatch, _detection(_unit(0)))

        rc = _run(["match", "--image", str(_write_image(tmp_path)), "--json"])

        assert rc is None
        assert json.loads(capsys.readouterr().out) == {"bank": "default", "match": None}

    def test_threshold_override_can_reject_a_default_hit(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_tilted(0.75)))
        image = str(_write_image(tmp_path))

        # 0.75 clears the store's 0.5 default ...
        _run(["match", "--image", image, "--json"])
        assert json.loads(capsys.readouterr().out)["match"]["score"] == pytest.approx(0.75)

        # ... but not an explicit 0.9.
        _run(["match", "--image", image, "--threshold", "0.9", "--json"])
        assert json.loads(capsys.readouterr().out) == {"bank": "default", "match": None}

    def test_threshold_override_can_accept_a_default_miss(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_tilted(0.3)))  # below the 0.5 default
        image = str(_write_image(tmp_path))

        _run(["match", "--image", image, "--json"])
        assert json.loads(capsys.readouterr().out) == {"bank": "default", "match": None}

        _run(["match", "--image", image, "--threshold", "0.2", "--json"])
        assert json.loads(capsys.readouterr().out)["match"]["name"] == "ada"

    def test_banks_are_isolated(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path, name="ada", bank="alpha")
        _patch_detect(monkeypatch, _detection(_unit(0)))

        _run(["match", "--image", str(_write_image(tmp_path)), "--bank", "beta", "--json"])
        assert json.loads(capsys.readouterr().out) == {"bank": "beta", "match": None}

        _run(["match", "--image", str(_write_image(tmp_path)), "--bank", "alpha", "--json"])
        assert json.loads(capsys.readouterr().out)["match"]["name"] == "ada"

    def test_match_from_stdin(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        self._enrol(monkeypatch, capsys, tmp_path)
        _patch_detect(monkeypatch, _detection(_unit(0)))
        _feed_stdin(monkeypatch, _image_bytes())

        _run(["match", "--image", "-", "--json"])

        assert json.loads(capsys.readouterr().out)["match"]["name"] == "ada"

    def test_empty_stdin_is_a_user_error(self, fake_cv2, monkeypatch: pytest.MonkeyPatch) -> None:
        _feed_stdin(monkeypatch, b"")

        with pytest.raises(CliError) as excinfo:
            _run(["match", "--image", "-"])

        assert excinfo.value.code == EXIT_USER_ERROR

    def test_missing_image_file_is_a_user_error(self, fake_cv2, tmp_path) -> None:
        with pytest.raises(CliError) as excinfo:
            _run(["match", "--image", str(tmp_path / "gone.png")])

        assert excinfo.value.code == EXIT_USER_ERROR

    def test_undecodable_file_is_a_user_error(self, fake_cv2, tmp_path) -> None:
        garbage = tmp_path / "garbage.png"
        garbage.write_bytes(b"\x00\x01\x02 not an image")

        with pytest.raises(CliError) as excinfo:
            _run(["match", "--image", str(garbage)])

        assert excinfo.value.code == EXIT_USER_ERROR
        assert "decode" in excinfo.value.message

    def test_no_face_detected_is_a_user_error(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _patch_detect(monkeypatch, None)

        with pytest.raises(CliError) as excinfo:
            _run(["match", "--image", str(_write_image(tmp_path))])

        assert excinfo.value.code == EXIT_USER_ERROR
        assert excinfo.value.message == "no face detected in the image"
