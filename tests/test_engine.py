"""Tests for ``face_recognition_cli.engine`` — the ported YuNet + SFace engine.

Written test-first (TDD): these tests define the contract the port must
satisfy. Three properties shape every test here:

* **No network.** Nothing downloads a model. ``_ensure_model``'s download seam
  is monkeypatched, and the ``detect()`` tests pre-create stub model files so
  ``_ensure_model`` short-circuits on ``path.exists()``.
* **No real models.** The stub files are a few bytes; the engine never reads
  their contents because ``cv2`` itself is faked.
* **No opencv.** The dev environment installs neither ``[cpu]`` nor ``[gpu]``,
  which is exactly the bare-install case the port must survive. The
  missing-extra path is proven with ``monkeypatch.setitem(sys.modules, "cv2",
  None)`` (the seam reachy-mini-cli's ``tests/test_vision_face.py`` uses), so
  the assertions hold whether or not ``cv2`` happens to be installed; the
  happy path is proven with a **fake** ``cv2`` module injected the same way.

The constant assertions (URLs, filenames, size floors, ``EMBEDDING_DIM``) are
the unit-level proof of the same-embedding-space invariant: the ported engine
must fetch byte-identical model files to the ones reachy-mini-cli already
enrolled faces with. Changing any of those literals forks the vector space and
invalidates every stored face, so they are asserted as literals here rather
than compared against the module's own names.
"""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv, no shell, test-only import probe
import sys
import types
import urllib.error
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli import engine, state
from face_recognition_cli.cli._errors import EXIT_ENV_ERROR, CliError

_REPO_ROOT = Path(__file__).parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "face_recognition_cli"


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pin the state dir into a throwaway directory — never the real home."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)


# ---------------------------------------------------------------------------
# A fake cv2: enough surface for FaceEngine, nothing more
# ---------------------------------------------------------------------------


class _FakeDetectorNoSetters:
    """A YuNet stand-in *without* setPreferableBackend/Target.

    Older ``cv2`` builds expose no DNN-backend setters on ``FaceDetectorYN``;
    the CUDA probe must ``hasattr``-guard against exactly this shape.
    """

    def __init__(self, faces: np.ndarray | None) -> None:
        self._faces = faces
        self.input_sizes: list[tuple[int, int]] = []
        self.detect_calls = 0

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 - cv2 API name
        self.input_sizes.append(size)

    def detect(self, frame: np.ndarray) -> tuple[int, np.ndarray | None]:
        self.detect_calls += 1
        return 1, self._faces


class _FakeDetector(_FakeDetectorNoSetters):
    """A YuNet stand-in that *does* expose the DNN-backend setters."""

    def __init__(self, faces: np.ndarray | None) -> None:
        super().__init__(faces)
        self.backend: object | None = None
        self.target: object | None = None

    def setPreferableBackend(self, backend: object) -> None:  # noqa: N802 - cv2 API name
        self.backend = backend

    def setPreferableTarget(self, target: object) -> None:  # noqa: N802 - cv2 API name
        self.target = target


class _FakeDetectorRaisingSetters(_FakeDetectorNoSetters):
    """A YuNet stand-in whose DNN-backend setters blow up.

    The shape of a box that *reports* CUDA devices but cannot actually use
    them (driver/toolkit mismatch) — must degrade to CPU, never propagate.
    """

    def __init__(self, faces: np.ndarray | None) -> None:
        super().__init__(faces)
        self.backend: object | None = None
        self.target: object | None = None

    def setPreferableBackend(self, backend: object) -> None:  # noqa: N802 - cv2 API name
        raise RuntimeError("cuda backend is not available in this build")

    def setPreferableTarget(self, target: object) -> None:  # noqa: N802 - cv2 API name
        raise RuntimeError("cuda target is not available in this build")


class _FakeRecognizer:
    """An SFace stand-in: alignCrop echoes, feature returns a 1x128 row."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding
        self.align_calls: list[np.ndarray] = []

    def alignCrop(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:  # noqa: N802
        self.align_calls.append(np.asarray(face))
        return frame

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        # Real SFace returns a (1, 128) row — the port must flatten it.
        return self._embedding.reshape(1, -1)


def _make_fake_cv2(
    *,
    faces: np.ndarray | None,
    embedding: np.ndarray | None = None,
    cuda_devices: int | None = None,
    cuda_raises: bool = False,
    detector_cls: type[_FakeDetectorNoSetters] = _FakeDetector,
) -> types.ModuleType:
    """Build a fake ``cv2`` module.

    ``cuda_devices=None`` omits the ``cv2.cuda`` namespace entirely (the shape
    of a plain PyPI ``opencv-python-headless`` wheel); an int provides it and
    reports that many CUDA devices.
    """
    module = types.ModuleType("cv2")
    emb = embedding if embedding is not None else np.arange(128, dtype=np.float32)

    created: dict[str, object] = {}

    class _FaceDetectorYN:
        @staticmethod
        def create(model, config, input_size, **kwargs):  # noqa: ANN001 - fake
            detector = detector_cls(faces)
            created["detector"] = detector
            created["detector_args"] = (model, config, input_size, kwargs)
            created["detector_creates"] = int(created.get("detector_creates", 0)) + 1
            return detector

    class _FaceRecognizerSF:
        @staticmethod
        def create(model, config):  # noqa: ANN001 - fake
            recognizer = _FakeRecognizer(emb)
            created["recognizer"] = recognizer
            created["recognizer_args"] = (model, config)
            return recognizer

    module.FaceDetectorYN = _FaceDetectorYN  # type: ignore[attr-defined]
    module.FaceRecognizerSF = _FaceRecognizerSF  # type: ignore[attr-defined]
    module.created = created  # type: ignore[attr-defined]

    dnn = types.SimpleNamespace(DNN_BACKEND_CUDA=5, DNN_TARGET_CUDA=6)
    module.dnn = dnn  # type: ignore[attr-defined]

    if cuda_raises:

        def _boom() -> int:
            raise RuntimeError("no cuda runtime on this box")

        module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            getCudaEnabledDeviceCount=_boom
        )
    elif cuda_devices is not None:
        module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            getCudaEnabledDeviceCount=lambda: cuda_devices
        )

    return module


def _stub_models(models_dir: Path) -> Path:
    """Pre-create both model files so ``_ensure_model`` never downloads.

    ``_ensure_model`` returns early on ``path.exists()`` without a size check,
    so a few stub bytes stand in for the real ~230 KB / ~37 MB ONNX files.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / engine.YUNET_FILE).write_bytes(b"stub-yunet")
    (models_dir / engine.SFACE_FILE).write_bytes(b"stub-sface")
    return models_dir


def _faces(*boxes: tuple[float, float, float, float]) -> np.ndarray:
    """Build a YuNet-shaped detection array (N x 15) from (x, y, w, h) boxes."""
    out = np.zeros((len(boxes), 15), dtype=np.float32)
    for row, box in enumerate(boxes):
        out[row, :4] = box
        out[row, 14] = 0.99  # score column
    return out


# ---------------------------------------------------------------------------
# Import boundary — the module loads with no opencv anywhere
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def test_engine_module_imports_without_cv2_installed(self) -> None:
        """The import at the top of this file already proves it; assert the surface."""
        assert hasattr(engine, "FaceEngine")
        assert hasattr(engine, "FaceDetection")

    def test_engine_module_does_not_import_cv2_at_module_load(self) -> None:
        """Run in a fresh subprocess so the probe cannot see this test's fakes."""
        code = "import sys; import face_recognition_cli.engine; print('cv2' in sys.modules)"
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=True,
        )
        assert proc.stdout.strip() == "False"

    def test_package_source_contains_no_reachy_import(self) -> None:
        """The port must rehome both reachy imports — nothing may reference reachy."""
        offenders: list[str] = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith(("from reachy", "import reachy")):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {stripped}")
        assert offenders == []


# ---------------------------------------------------------------------------
# Constants — the same-embedding-space invariant, asserted as literals
# ---------------------------------------------------------------------------


class TestModelConstants:
    def test_yunet_url_is_byte_identical_to_reachys(self) -> None:
        assert engine.YUNET_URL == (
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx"
        )

    def test_sface_url_is_byte_identical_to_reachys(self) -> None:
        assert engine.SFACE_URL == (
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
            "face_recognition_sface_2021dec.onnx"
        )

    def test_model_filenames_are_byte_identical_to_reachys(self) -> None:
        assert engine.YUNET_FILE == "face_detection_yunet_2023mar.onnx"
        assert engine.SFACE_FILE == "face_recognition_sface_2021dec.onnx"

    def test_size_floors_match_reachys(self) -> None:
        assert engine.YUNET_MIN_BYTES == 100_000
        assert engine.SFACE_MIN_BYTES == 20_000_000

    def test_detector_defaults_match_reachys(self) -> None:
        assert engine.DEFAULT_SCORE_THRESHOLD == 0.6
        assert engine.DEFAULT_NMS_THRESHOLD == 0.3
        assert engine.DEFAULT_TOP_K == 5
        assert engine.DEFAULT_INPUT_SIZE == (320, 320)

    def test_embedding_dim_is_128(self) -> None:
        assert engine.EMBEDDING_DIM == 128

    def test_real_model_constants_clear_their_own_sanity_floors(self) -> None:
        """Guard against the floors and the real files drifting apart silently."""
        assert engine.YUNET_MIN_BYTES < 300_000  # real file is ~230KB
        assert engine.SFACE_MIN_BYTES < 39_000_000  # real file is ~37MB

    def test_exactly_one_model_url_pair_in_the_package(self) -> None:
        """No second model may hide anywhere — one pair, or the space forks."""
        hits = 0
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            hits += path.read_text(encoding="utf-8").count("opencv_zoo/raw/main/models/")
        assert hits == 2


# ---------------------------------------------------------------------------
# The missing-extra path — clean exit-2 CliError naming THIS package's extra
# ---------------------------------------------------------------------------


class TestMissingCv2:
    def test_import_cv2_raises_clean_exit2_cli_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``sys.modules[name] = None`` makes ``import name`` raise ImportError."""
        monkeypatch.setitem(sys.modules, "cv2", None)  # type: ignore[arg-type]

        with pytest.raises(CliError) as excinfo:
            engine._import_cv2()

        assert excinfo.value.code == EXIT_ENV_ERROR
        assert excinfo.value.code == 2
        assert "opencv" in excinfo.value.message.lower()

    def test_remediation_names_this_packages_cpu_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cv2", None)  # type: ignore[arg-type]

        with pytest.raises(CliError) as excinfo:
            engine._import_cv2()

        remediation = excinfo.value.remediation
        assert "face-recognition-cli[cpu]" in remediation
        assert "reachy" not in remediation.lower()
        assert "[vision]" not in remediation

    def test_detect_surfaces_the_same_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setitem(sys.modules, "cv2", None)  # type: ignore[arg-type]

        eng = engine.FaceEngine(models_dir=tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(CliError) as excinfo:
            eng.detect(frame)

        assert excinfo.value.code == EXIT_ENV_ERROR
        assert "face-recognition-cli[cpu]" in excinfo.value.remediation

    def test_detect_without_cv2_downloads_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The cv2 probe comes first — a missing extra must not hit the network."""
        monkeypatch.setitem(sys.modules, "cv2", None)  # type: ignore[arg-type]

        def _fail(*_a, **_k):
            raise AssertionError("must not download when cv2 is missing")

        monkeypatch.setattr(engine, "_download", _fail)

        eng = engine.FaceEngine(models_dir=tmp_path)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(CliError):
            eng.detect(frame)

        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# _ensure_model — .part staging, size floor, cleanup (no network)
# ---------------------------------------------------------------------------


class TestEnsureModel:
    def test_downloads_when_missing_and_renames_into_place(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, Path]] = []

        def _fake_download(url, dest, *, timeout=30.0):
            calls.append((url, dest))
            # Staged, never written in place — with a per-process unique
            # suffix so concurrent downloaders cannot share a temp file.
            assert ".part." in dest.name
            assert dest.name.startswith("model.onnx.part.")
            dest.write_bytes(b"x" * 150_000)

        monkeypatch.setattr(engine, "_download", _fake_download)

        path = engine._ensure_model(
            "model.onnx",
            "https://example.invalid/model.onnx",
            min_bytes=100_000,
            models_dir=tmp_path / "models",
        )

        assert path == tmp_path / "models" / "model.onnx"
        assert path.stat().st_size == 150_000
        assert len(calls) == 1
        assert list((tmp_path / "models").glob("model.onnx.part*")) == []

    def test_skips_download_when_already_present(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = tmp_path / "model.onnx"
        existing.write_bytes(b"y" * 500_000)

        def _fail(*_a, **_k):
            raise AssertionError("must not download when the model file already exists")

        monkeypatch.setattr(engine, "_download", _fail)

        path = engine._ensure_model(
            "model.onnx",
            "https://example.invalid/model.onnx",
            min_bytes=100_000,
            models_dir=tmp_path,
        )

        assert path == existing
        assert path.stat().st_size == 500_000

    def test_rejects_a_truncated_download(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_download(url, dest, *, timeout=30.0):
            dest.write_bytes(b"x" * 10)  # far below any sane floor

        monkeypatch.setattr(engine, "_download", _fake_download)

        with pytest.raises(CliError) as excinfo:
            engine._ensure_model(
                "model.onnx",
                "https://example.invalid/model.onnx",
                min_bytes=100_000,
                models_dir=tmp_path,
            )

        assert excinfo.value.code == EXIT_ENV_ERROR
        assert "truncated" in excinfo.value.message
        assert not (tmp_path / "model.onnx").exists()
        assert list(tmp_path.glob("model.onnx.part*")) == []

    def test_rejects_download_with_network_error_and_cleans_up_partial(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_download(url, dest, *, timeout=30.0):
            dest.write_bytes(b"partial")
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(engine, "_download", _raising_download)

        with pytest.raises(urllib.error.URLError):
            engine._ensure_model(
                "model.onnx",
                "https://example.invalid/model.onnx",
                min_bytes=100_000,
                models_dir=tmp_path,
            )

        assert list(tmp_path.glob("model.onnx.part*")) == []
        assert not (tmp_path / "model.onnx").exists()

    def test_creates_the_models_dir_when_absent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            engine,
            "_download",
            lambda url, dest, *, timeout=30.0: dest.write_bytes(b"z" * 200_000),
        )
        nested = tmp_path / "a" / "b" / "models"

        path = engine._ensure_model(
            "model.onnx",
            "https://example.invalid/model.onnx",
            min_bytes=100_000,
            models_dir=nested,
        )

        assert path.exists()
        assert path.parent == nested


# ---------------------------------------------------------------------------
# _download — chunked stdlib transfer, this package's User-Agent (no network)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """A minimal urlopen() stand-in: a context manager with chunked read()."""

    def __init__(self, payload: bytes) -> None:
        self._remaining = payload
        self.read_calls = 0

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        chunk, self._remaining = self._remaining[:size], self._remaining[size:]
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class TestDownload:
    def test_streams_chunks_to_dest_with_this_packages_user_agent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"m" * 150_000  # larger than one 64 KiB chunk
        captured: dict[str, object] = {}
        response = _FakeResponse(payload)

        def _fake_urlopen(req, timeout=None):
            captured["req"] = req
            captured["timeout"] = timeout
            return response

        monkeypatch.setattr(engine.urllib.request, "urlopen", _fake_urlopen)
        dest = tmp_path / "model.onnx.part"

        engine._download("https://example.invalid/model.onnx", dest)

        assert dest.read_bytes() == payload
        assert response.read_calls >= 3  # chunked, not one slurp
        req = captured["req"]
        assert req.full_url == "https://example.invalid/model.onnx"
        # The UA is rehomed from reachy-mini-cli's to this package's name.
        assert req.get_header("User-agent") == "face-recognition-cli"
        assert captured["timeout"] == 30.0


# ---------------------------------------------------------------------------
# Models dir default — shared across banks, via state.models_dir()
# ---------------------------------------------------------------------------


class TestModelsDirDefault:
    def test_default_models_dir_is_state_models_dir(self, tmp_path) -> None:
        # FACE_RECOGNITION_STATE_DIR is pinned by the autouse fixture.
        assert engine._default_models_dir() == state.models_dir()
        assert engine._default_models_dir() == tmp_path / "state" / "models"

    def test_default_models_dir_is_not_under_a_bank(self, tmp_path) -> None:
        """Models are bank-independent — one copy shared by every bank."""
        resolved = engine._default_models_dir()
        assert "banks" not in resolved.parts
        assert resolved.parent == state.state_dir()

    def test_engine_resolves_the_default_when_no_models_dir_injected(self, tmp_path) -> None:
        eng = engine.FaceEngine()
        assert eng._resolve_models_dir() == state.models_dir()

    def test_injected_models_dir_wins(self, tmp_path) -> None:
        injected = tmp_path / "injected"
        eng = engine.FaceEngine(models_dir=injected)
        assert eng._resolve_models_dir() == injected


# ---------------------------------------------------------------------------
# detect() — the happy path, with a FAKE cv2 (no opencv, no models, no network)
# ---------------------------------------------------------------------------


class TestDetectWithFakeCv2:
    def test_largest_face_is_selected_and_bbox_normalised(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        models_dir = _stub_models(tmp_path / "models")
        # Two faces: the second is much larger (60*80 vs 20*10).
        faces = _faces((5.0, 7.0, 20.0, 10.0), (40.0, 50.0, 60.0, 80.0))
        embedding = np.arange(128, dtype=np.float32)
        fake = _make_fake_cv2(faces=faces, embedding=embedding)
        monkeypatch.setitem(sys.modules, "cv2", fake)

        frame = np.zeros((200, 400, 3), dtype=np.uint8)  # H=200, W=400
        result = engine.FaceEngine(models_dir=models_dir).detect(frame)

        assert isinstance(result, engine.FaceDetection)
        assert result.bbox_norm == pytest.approx((40.0 / 400, 50.0 / 200, 100.0 / 400, 130.0 / 200))
        assert result.embedding.shape == (engine.EMBEDDING_DIM,)
        assert np.array_equal(result.embedding, embedding)

        # The largest row — not the first — was handed to alignCrop.
        recognizer = fake.created["recognizer"]
        assert recognizer.align_calls[0][:4] == pytest.approx([40.0, 50.0, 60.0, 80.0])

    def test_input_size_is_set_from_the_frame_as_width_height(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=_faces((1.0, 2.0, 3.0, 4.0)))
        monkeypatch.setitem(sys.modules, "cv2", fake)

        frame = np.zeros((120, 160, 3), dtype=np.uint8)  # H=120, W=160
        engine.FaceEngine(models_dir=models_dir).detect(frame)

        assert fake.created["detector"].input_sizes == [(160, 120)]

    def test_no_faces_returns_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=None)
        monkeypatch.setitem(sys.modules, "cv2", fake)

        result = engine.FaceEngine(models_dir=models_dir).detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        assert result is None

    def test_empty_face_array_returns_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=np.zeros((0, 15), dtype=np.float32))
        monkeypatch.setitem(sys.modules, "cv2", fake)

        result = engine.FaceEngine(models_dir=models_dir).detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        assert result is None

    def test_detector_and_recognizer_are_built_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)))
        monkeypatch.setitem(sys.modules, "cv2", fake)

        eng = engine.FaceEngine(models_dir=models_dir)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        eng.detect(frame)
        eng.detect(frame)

        assert fake.created["detector_creates"] == 1
        assert fake.created["detector"].detect_calls == 2

    def test_create_receives_the_model_paths_and_tuned_thresholds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)))
        monkeypatch.setitem(sys.modules, "cv2", fake)

        eng = engine.FaceEngine(
            models_dir=models_dir, score_threshold=0.9, nms_threshold=0.1, top_k=2
        )
        eng.detect(np.zeros((10, 10, 3), dtype=np.uint8))

        model, config, input_size, kwargs = fake.created["detector_args"]
        assert model == str(models_dir / engine.YUNET_FILE)
        assert config == ""
        assert input_size == engine.DEFAULT_INPUT_SIZE
        assert kwargs == {"score_threshold": 0.9, "nms_threshold": 0.1, "top_k": 2}
        assert fake.created["recognizer_args"] == (str(models_dir / engine.SFACE_FILE), "")

    def test_detect_never_touches_the_network(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        models_dir = _stub_models(tmp_path / "models")
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)))
        monkeypatch.setitem(sys.modules, "cv2", fake)

        def _fail(*_a, **_k):
            raise AssertionError("models already on disk — must not download")

        monkeypatch.setattr(engine, "_download", _fail)

        assert (
            engine.FaceEngine(models_dir=models_dir).detect(np.zeros((10, 10, 3), dtype=np.uint8))
            is not None
        )

    def test_face_detection_is_frozen(self) -> None:
        detection = engine.FaceDetection(bbox_norm=(0.0, 0.0, 1.0, 1.0), embedding=np.zeros(128))
        with pytest.raises(FrozenInstanceError):
            detection.bbox_norm = (1.0, 1.0, 1.0, 1.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The CUDA DNN probe — opportunistic, silent, never fatal
# ---------------------------------------------------------------------------


class TestCudaProbe:
    def _detect_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path, fake) -> object:
        models_dir = _stub_models(tmp_path / "models")
        monkeypatch.setitem(sys.modules, "cv2", fake)
        result = engine.FaceEngine(models_dir=models_dir).detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )
        assert result is not None  # the probe must never break detection
        return fake.created["detector"]

    def test_cv2_without_a_cuda_namespace_falls_back_silently(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The PyPI opencv-python-headless wheel shape: no cv2.cuda at all."""
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)), cuda_devices=None)
        assert not hasattr(fake, "cuda")

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert detector.backend is None
        assert detector.target is None

    def test_zero_cuda_devices_leaves_the_default_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)), cuda_devices=0)

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert detector.backend is None
        assert detector.target is None

    def test_cuda_device_present_selects_the_cuda_dnn_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)), cuda_devices=1)

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert detector.backend == fake.dnn.DNN_BACKEND_CUDA
        assert detector.target == fake.dnn.DNN_TARGET_CUDA

    def test_detector_without_the_setters_is_skipped_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Older cv2: FaceDetectorYN has no setPreferableBackend — hasattr guard."""
        fake = _make_fake_cv2(
            faces=_faces((1.0, 1.0, 2.0, 2.0)),
            cuda_devices=1,
            detector_cls=_FakeDetectorNoSetters,
        )

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert not hasattr(detector, "setPreferableBackend")

    def test_a_raising_backend_setter_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """CUDA is reported but unusable — detection must still succeed on CPU."""
        fake = _make_fake_cv2(
            faces=_faces((1.0, 1.0, 2.0, 2.0)),
            cuda_devices=1,
            detector_cls=_FakeDetectorRaisingSetters,
        )

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert detector.backend is None
        assert detector.target is None

    def test_a_raising_cuda_probe_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A broken CUDA runtime must degrade to CPU, never surface to the caller."""
        fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)), cuda_raises=True)

        detector = self._detect_once(monkeypatch, tmp_path, fake)

        assert detector.backend is None
        assert detector.target is None

    def test_the_cuda_path_never_changes_which_models_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Acceleration is execution-only: the same two ONNX files either way."""
        models_dir = _stub_models(tmp_path / "models")
        seen = []
        for devices in (0, 1):
            fake = _make_fake_cv2(faces=_faces((1.0, 1.0, 2.0, 2.0)), cuda_devices=devices)
            monkeypatch.setitem(sys.modules, "cv2", fake)
            engine.FaceEngine(models_dir=models_dir).detect(np.zeros((10, 10, 3), dtype=np.uint8))
            seen.append((fake.created["detector_args"][0], fake.created["recognizer_args"][0]))

        assert seen[0] == seen[1]
        assert seen[0] == (
            str(models_dir / engine.YUNET_FILE),
            str(models_dir / engine.SFACE_FILE),
        )


# ---------------------------------------------------------------------------
# No loop, no threads — the recorded non-goal
# ---------------------------------------------------------------------------


class TestNoBackgroundLoop:
    def test_engine_source_has_no_threading_or_sleep(self) -> None:
        source = (_PACKAGE_ROOT / "engine.py").read_text(encoding="utf-8")
        assert "import threading" not in source
        assert "Thread(" not in source
        assert "time.sleep" not in source

    def test_detect_is_a_plain_synchronous_method(self) -> None:
        import inspect

        assert not inspect.iscoroutinefunction(engine.FaceEngine.detect)
        assert not inspect.isgeneratorfunction(engine.FaceEngine.detect)
