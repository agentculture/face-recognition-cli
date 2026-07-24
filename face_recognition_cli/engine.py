"""Face detection + embedding engine — OpenCV YuNet + SFace.

Provenance: cited (cite-don't-import) from ``reachy_nova.face_recognition``
into ``reachy-mini-cli``'s ``reachy/vision/face.py``, and **extracted here**
so the robot depends on this package instead of carrying its own copy. The
models, the URLs they are fetched from, and the 128-dim SFace embedding space
are byte-identical to what reachy-mini-cli already enrolled faces with —
that compatibility is the whole reason extraction beat swapping in dlib, and
changing any model constant below silently forks the vector space and
invalidates every face already enrolled on a robot.

Deviations carried over from the nova original, and why:

* **``cv2`` is imported lazily, inside functions only — never at module import
  time.** This keeps :mod:`face_recognition_cli.engine` (and anything that
  merely imports it) loadable on a bare install with neither the ``[cpu]`` nor
  the ``[gpu]`` extra present. A missing OpenCV surfaces as this repo's
  established clean exit-2
  :class:`~face_recognition_cli.cli._errors.CliError` naming the ``[cpu]``
  extra, not an ``ImportError`` traceback.
* **No threading / no background dispatch loop.** Nova's ``FaceRecognition``
  owns a daemon thread, a busy flag, and a fixed 500 ms detect interval.
  That loop-owning responsibility belongs to the caller — a recorded non-goal
  for this package. :class:`FaceEngine` is a synchronous, stateless-per-call
  ``detect(frame)``; whoever owns the frame source owns the tick.
* **Downloads get a size sanity check.** Nova trusts ``urlretrieve`` blindly,
  but a truncated download, a dropped connection, or an HTML error page
  swapped in by a captive portal would otherwise be written to disk, "exist"
  on the next run, and then fail cryptically inside OpenCV instead of at
  download time. A transfer lands in a ``.part`` sibling and is renamed into
  place only after it clears the expected floor.

Two things are new here, neither of which touches the embedding space:

* **Models live in one shared, bank-independent directory.** This package
  partitions *identities* into per-consumer banks (see
  :mod:`face_recognition_cli.state`), but every bank embeds through the same
  YuNet + SFace pair, so the ONNX files sit at ``<state dir>/models/`` — one
  level above ``banks/`` — rather than being duplicated (~37 MB for SFace
  alone) per bank.
* **An opportunistic CUDA DNN backend.** :meth:`FaceEngine._load` probes for a
  CUDA-capable local OpenCV build and, when it finds one, asks the detector to
  run through cv2's CUDA DNN backend. It is a silent no-op on the PyPI
  ``opencv-python-headless`` wheels (which ship no CUDA DNN), and only
  Jetson/DGX-class *locally built* OpenCV carries the capability. This is what
  the ``[gpu]`` extra buys: **acceleration of execution only** — the models
  never change, so a face enrolled on a Jetson still matches on a Pi.
"""

from __future__ import annotations

import logging
import os
import secrets
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_recognition_cli import state
from face_recognition_cli.cli._errors import EXIT_ENV_ERROR, CliError

logger = logging.getLogger(__name__)

# --- OpenCV model zoo — the SAME models/URLs reachy-mini-cli (and nova) use --
# Byte-identical on purpose: these two literals ARE the embedding-space
# contract. Editing them forks the vector space (see the module docstring).
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

# Sanity floors for the size check below. A genuine download is ~230 KB
# (YuNet) / ~37 MB (SFace); a truncated download or a swapped-in HTML error
# page is nowhere near these sizes, so a generous floor well below the real
# size still catches a corrupt download without being fragile to the model
# zoo's file growing a little across revisions.
YUNET_MIN_BYTES = 100_000
SFACE_MIN_BYTES = 20_000_000

MODELS_DIRNAME = "models"

DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
DEFAULT_INPUT_SIZE = (320, 320)

#: 128 — the fixed embedding width SFace produces (documented for callers that
#: want to validate a stored/incoming vector's shape without loading cv2).
EMBEDDING_DIM = 128


@dataclass(frozen=True)
class FaceDetection:
    """One detected face: its normalised bounding box and 128-dim embedding."""

    #: ``(x1, y1, x2, y2)`` in ``[0, 1]``, normalised by frame width/height.
    bbox_norm: tuple[float, float, float, float]
    embedding: np.ndarray


def _import_cv2():  # type: ignore[no-untyped-def]
    """Lazily import ``cv2``; raise a clean exit-2 CliError when it's absent."""
    try:
        import cv2
    except ImportError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message="the opencv face-recognition engine is not installed",
            remediation="install the cpu extra: pip install 'face-recognition-cli[cpu]'",
        ) from err
    return cv2


def _default_models_dir() -> Path:
    """Where ONNX models live by default: ``<state dir>/models``.

    Deliberately *not* per-bank — banks partition identities, not models, and
    every bank embeds through the same files (see
    :func:`face_recognition_cli.state.models_dir`).
    """
    return state.models_dir()


def _download(url: str, dest: Path, *, timeout: float = 30.0) -> None:
    """Stream *url* to *dest* (stdlib ``urllib`` only, chunked write)."""
    req = urllib.request.Request(url, headers={"User-Agent": "face-recognition-cli"})
    with urllib.request.urlopen(  # nosec B310 - fixed https model-zoo URL, not user input
        req, timeout=timeout
    ) as resp:
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)


def _ensure_model(filename: str, url: str, *, min_bytes: int, models_dir: Path) -> Path:
    """Download *filename* from *url* into *models_dir* if not already present.

    Downloads to a ``.part.*`` sibling first and only renames it into place
    once the transfer completes AND clears the ``min_bytes`` sanity floor, so
    a partial/failed/corrupt download can never masquerade as a usable model
    file on a later call. The staging name carries a per-process unique
    suffix so concurrent downloaders (two CLI invocations racing on a fresh
    state dir) cannot truncate or interleave each other's temp file; the
    atomic ``replace`` publish means the last completed download wins, and
    every winner is the same bytes.
    """
    path = models_dir / filename
    if path.exists():
        return path

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = models_dir / f"{filename}.part.{os.getpid()}-{secrets.token_hex(4)}"
    logger.info("downloading %s ...", filename)
    try:
        _download(url, tmp_path)
        size = tmp_path.stat().st_size
        if size < min_bytes:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"downloaded {filename} looks truncated ({size} bytes)",
                remediation="check network connectivity and retry",
            )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    logger.info("downloaded %s (%d bytes)", filename, path.stat().st_size)
    return path


def _try_enable_cuda(cv2, *targets) -> bool:  # type: ignore[no-untyped-def]
    """Opportunistically route *targets* through cv2's CUDA DNN backend.

    This is the entire mechanism behind the ``[gpu]`` extra, and it is
    deliberately tiny: it changes **where** inference executes, never **which
    model** executes. Both extras install the same ``opencv-python-headless``
    and load the same two ONNX files, so the embedding space is identical on a
    Pi and on a Jetson by construction — a face enrolled on one matches on the
    other. If a second inference stack ever becomes necessary, the store must
    start recording which model produced each embedding; do not quietly swap
    models here.

    Every branch degrades to plain CPU rather than failing, because none of
    these capabilities is guaranteed:

    * ``cv2.cuda`` is absent entirely on the PyPI wheels (both extras pull
      those), so this is a **no-op for a normal pip install** — the capability
      comes from a locally built OpenCV, which Jetson/DGX images typically
      carry.
    * ``FaceDetectorYN`` only gained ``setPreferableBackend`` /
      ``setPreferableTarget`` in newer OpenCV releases, hence the ``hasattr``
      guard rather than a version check.
    * A machine can report CUDA devices and still raise from the CUDA runtime
      (driver/toolkit mismatch), hence the broad ``except``.

    Returns ``True`` only when at least one target was switched over; the
    result is informational (logged at debug level) and never surfaces to the
    caller — a GPU that cannot be used is not an error.
    """
    try:
        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return False
        backend = cv2.dnn.DNN_BACKEND_CUDA
        target = cv2.dnn.DNN_TARGET_CUDA
    except Exception as err:
        logger.debug("cuda probe failed, staying on the cpu backend: %s", err)
        return False

    enabled = False
    for obj in targets:
        if not (hasattr(obj, "setPreferableBackend") and hasattr(obj, "setPreferableTarget")):
            continue
        try:
            obj.setPreferableBackend(backend)
            obj.setPreferableTarget(target)
            enabled = True
        except Exception as err:
            logger.debug("could not select the cuda dnn backend, staying on cpu: %s", err)
    if enabled:
        logger.debug("cuda dnn backend selected (same models, accelerated execution)")
    return enabled


class FaceEngine:
    """Detect the largest face in a frame and extract its 128-dim embedding.

    Lazily loads the YuNet detector + SFace recognizer (via :func:`_import_cv2`
    and :func:`_ensure_model`) on the first :meth:`detect` call; a missing
    ``[cpu]``/``[gpu]`` extra surfaces as a clean exit-2 :class:`CliError` at
    that point, not at construction or import time.

    Parameters
    ----------
    models_dir:
        Directory model files are downloaded into/read from. Defaults to
        ``<state dir>/models`` (shared by every bank); tests inject an isolated
        ``tmp_path``.
    score_threshold, nms_threshold, top_k:
        Passed straight through to ``cv2.FaceDetectorYN.create`` — see the
        OpenCV docs. Defaults match reachy-mini-cli's (and nova's).
    """

    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._models_dir = models_dir
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._detector = None
        self._recognizer = None

    def _resolve_models_dir(self) -> Path:
        return self._models_dir if self._models_dir is not None else _default_models_dir()

    def _load(self) -> None:
        """Lazily construct the YuNet detector + SFace recognizer (once)."""
        if self._detector is not None and self._recognizer is not None:
            return

        cv2 = _import_cv2()
        models_dir = self._resolve_models_dir()
        yunet_path = _ensure_model(
            YUNET_FILE, YUNET_URL, min_bytes=YUNET_MIN_BYTES, models_dir=models_dir
        )
        sface_path = _ensure_model(
            SFACE_FILE, SFACE_URL, min_bytes=SFACE_MIN_BYTES, models_dir=models_dir
        )

        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            DEFAULT_INPUT_SIZE,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

        # Acceleration only, never a model change — see _try_enable_cuda.
        # Silent and best-effort: a build without CUDA DNN (every PyPI wheel)
        # simply stays on the default CPU backend.
        #
        # Only the DETECTOR is offered to the GPU, not the recognizer. The
        # detector emits bounding boxes, which are consumed as pixel geometry
        # and are insensitive to a last-ulp difference between backends; the
        # recognizer emits the stored embedding itself, so leaving it on the
        # CPU keeps the vector path bit-for-bit identical on every host and
        # takes even float-noise off the table for the same-embedding-space
        # invariant. Detection also runs on every frame while embedding runs
        # once per enrol/match, so this is where the throughput is anyway.
        # _try_enable_cuda takes varargs — adding the recognizer later is a
        # one-argument change, but it needs a cross-host match check first.
        _try_enable_cuda(cv2, self._detector)

    def detect(self, frame: np.ndarray) -> FaceDetection | None:
        """Detect the largest face in *frame* (BGR ``H x W x 3``) and embed it.

        Synchronous and stateless-per-call by design: no thread, no interval,
        no background dispatch. Returns ``None`` when no face is found. Raises
        :class:`CliError` (exit 2) the first time this is called without the
        ``[cpu]``/``[gpu]`` extra installed.
        """
        self._load()

        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None

        # Largest-face selection (width * height), mirrors nova.
        areas = faces[:, 2] * faces[:, 3]
        best_idx = int(np.argmax(areas))
        best_face = faces[best_idx]

        bx, by, bw, bh = (float(best_face[i]) for i in range(4))
        bbox_norm = (bx / w, by / h, (bx + bw) / w, (by + bh) / h)

        aligned = self._recognizer.alignCrop(frame, best_face)
        embedding = self._recognizer.feature(aligned).flatten()

        return FaceDetection(bbox_norm=bbox_norm, embedding=embedding)
