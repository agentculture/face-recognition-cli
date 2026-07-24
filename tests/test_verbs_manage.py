"""Tests for the ``list`` / ``forget`` / ``forget-all`` verbs (task t7).

TDD — written before the corresponding ``cli/_commands/list_faces.py``,
``cli/_commands/forget.py``, and ``cli/_commands/forget_all.py`` modules exist.

These verbs are pure :class:`~face_recognition_cli.store.FaceStore` operations
— no ``cv2``, no network, no camera. They are **not** wired into
``face_recognition_cli.cli.main`` yet (task t8 owns registration in
``cli/__init__.py`` and the ``explain`` catalog), so tests build a standalone
argparse parser, call each module's ``register()``, parse real argv, and
invoke ``args.func(args)`` directly — mirroring exactly what ``main()`` will
eventually do once wired.

``forget-all`` is the one genuinely new/destructive verb (spec claim c5/h5):
it must obey the mesh write-verb rule — dry-run by default, ``--apply``
commits — and the dry-run path must be *provably* side-effect-free, not just
"reports it wouldn't delete anything" while secretly touching disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli.cli._commands import forget, forget_all, list_faces
from face_recognition_cli.cli._errors import CliError
from face_recognition_cli.state import bank_dir
from face_recognition_cli.store import FaceStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin state-dir resolution into a throwaway dir; never touch ~/.local/state."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


def _embedding(seed: int) -> np.ndarray:
    """A deterministic, non-degenerate 128-dim synthetic embedding."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=128).astype(float)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test-verbs")
    sub = parser.add_subparsers(dest="command")
    list_faces.register(sub)
    forget.register(sub)
    forget_all.register(sub)
    return parser


def _run(argv: list[str]) -> argparse.Namespace:
    """Parse *argv* with a fresh standalone parser and invoke the handler."""
    parser = _make_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return args


def _seed_bank(bank: str, names: list[str], seed_start: int = 0) -> dict[str, str]:
    """Enroll each name into *bank* via a real ``FaceStore``. Returns name -> id."""
    store = FaceStore(base_dir=bank_dir(bank))
    ids: dict[str, str] = {}
    for offset, name in enumerate(names):
        face_id = store.enroll(name, _embedding(seed_start + offset))
        ids[name] = face_id
    return ids


def _embeddings_dir(bank: str) -> Path:
    return bank_dir(bank) / "embeddings"


def _index_path(bank: str) -> Path:
    return bank_dir(bank) / "faces.json"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_empty_bank_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run(["list"])
        out = capsys.readouterr()
        assert out.out.strip() == "no faces enrolled"
        assert out.err == ""

    def test_empty_bank_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run(["list", "--json"])
        out = capsys.readouterr()
        payload = json.loads(out.out)
        assert payload == {"bank": "default", "faces": [], "count": 0}
        assert out.err == ""

    def test_populated_bank_lists_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        ids = _seed_bank("x", ["Alice", "Bob"])
        _run(["list", "--bank", "x", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert payload["bank"] == "x"
        assert payload["count"] == 2
        got_names_by_id = {f["id"]: f["name"] for f in payload["faces"]}
        assert got_names_by_id == {v: k for k, v in ids.items()}
        for face in payload["faces"]:
            assert face["num_embeddings"] == 1
            assert "created" in face

    def test_populated_bank_text_shows_id_and_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ids = _seed_bank("x", ["Alice"])
        _run(["list", "--bank", "x"])
        out = capsys.readouterr()
        assert ids["Alice"] in out.out
        assert "Alice" in out.out
        assert out.err == ""

    def test_bank_flag_selects_disjoint_contents(self, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_bank("a", ["Alice"])
        _seed_bank("b", ["Bob"])

        _run(["list", "--bank", "a", "--json"])
        payload_a = json.loads(capsys.readouterr().out)
        _run(["list", "--bank", "b", "--json"])
        payload_b = json.loads(capsys.readouterr().out)

        names_a = {f["name"] for f in payload_a["faces"]}
        names_b = {f["name"] for f in payload_b["faces"]}
        assert names_a == {"Alice"}
        assert names_b == {"Bob"}
        assert names_a.isdisjoint(names_b)


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_removes_exactly_one_identity(self, capsys: pytest.CaptureFixture[str]) -> None:
        ids = _seed_bank("x", ["Alice", "Bob"])
        alice_id, bob_id = ids["Alice"], ids["Bob"]

        _run(["forget", alice_id, "--bank", "x", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload == {"forgotten": alice_id, "bank": "x"}

        store = FaceStore(base_dir=bank_dir("x"))
        remaining = {f["id"] for f in store.list_faces()}
        assert remaining == {bob_id}

        index = json.loads(_index_path("x").read_text())
        assert alice_id not in index
        assert bob_id in index
        assert not (_embeddings_dir("x") / f"{alice_id}.npy").exists()
        assert (_embeddings_dir("x") / f"{bob_id}.npy").exists()

    def test_unknown_id_raises_cli_error_code_1_with_list_remediation(self) -> None:
        _seed_bank("x", ["Alice"])
        parser = _make_parser()
        args = parser.parse_args(["forget", "zzzz", "--bank", "x"])

        with pytest.raises(CliError) as excinfo:
            args.func(args)

        err = excinfo.value
        assert err.code == 1
        assert "zzzz" in err.message
        assert "list" in err.remediation

    def test_bank_scoped_leaves_other_bank_intact(self, capsys: pytest.CaptureFixture[str]) -> None:
        ids_a = _seed_bank("a", ["Alice"])
        ids_b = _seed_bank("b", ["Alice"])  # same name, different bank -> different id

        _run(["forget", ids_a["Alice"], "--bank", "a"])
        capsys.readouterr()

        store_b = FaceStore(base_dir=bank_dir("b"))
        remaining_b = store_b.list_faces()
        assert len(remaining_b) == 1
        assert remaining_b[0]["id"] == ids_b["Alice"]

    def test_text_confirmation_mentions_id_stdout_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ids = _seed_bank("x", ["Alice"])
        _run(["forget", ids["Alice"], "--bank", "x"])
        out = capsys.readouterr()
        assert ids["Alice"] in out.out
        assert out.err == ""


# ---------------------------------------------------------------------------
# forget-all — dry run (default, must be side-effect-free)
# ---------------------------------------------------------------------------


class TestForgetAllDryRun:
    def test_dry_run_leaves_store_byte_identical(self, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_bank("x", ["Alice", "Bob"])
        before_index = _index_path("x").read_bytes()
        before_files = sorted(p.name for p in _embeddings_dir("x").iterdir())

        _run(["forget-all", "--bank", "x", "--json"])
        payload = json.loads(capsys.readouterr().out)

        after_index = _index_path("x").read_bytes()
        after_files = sorted(p.name for p in _embeddings_dir("x").iterdir())

        assert before_index == after_index
        assert before_files == after_files

        assert payload["bank"] == "x"
        assert payload["dry_run"] is True
        assert payload["count"] == 2
        names = {f["name"] for f in payload["faces"]}
        assert names == {"Alice", "Bob"}
        assert all(set(f) == {"id", "name"} for f in payload["faces"])

    def test_dry_run_text_reports_would_delete_set_and_apply_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ids = _seed_bank("x", ["Alice"])
        _run(["forget-all", "--bank", "x"])
        out = capsys.readouterr()

        assert ids["Alice"] in out.out
        assert "Alice" in out.out
        assert "--apply" in out.out
        assert out.err == ""

        store = FaceStore(base_dir=bank_dir("x"))
        assert store.permanent_count == 1  # nothing deleted by the dry run

    def test_dry_run_empty_bank_exits_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run(["forget-all", "--bank", "empty", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 0
        assert payload["dry_run"] is True
        assert payload["faces"] == []


# ---------------------------------------------------------------------------
# forget-all --apply
# ---------------------------------------------------------------------------


class TestForgetAllApply:
    def test_apply_empties_selected_bank_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_bank("a", ["Alice", "Carol"])
        _seed_bank("b", ["Bob"])

        _run(["forget-all", "--bank", "a", "--apply", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert payload["bank"] == "a"
        assert payload["dry_run"] is False
        assert payload["count"] == 2
        names = {f["name"] for f in payload["faces"]}
        assert names == {"Alice", "Carol"}

        store_a = FaceStore(base_dir=bank_dir("a"))
        assert store_a.list_faces() == []
        index_a = json.loads(_index_path("a").read_text())
        assert index_a == {}
        assert list(_embeddings_dir("a").iterdir()) == []

        store_b = FaceStore(base_dir=bank_dir("b"))
        remaining_b = store_b.list_faces()
        assert len(remaining_b) == 1
        assert remaining_b[0]["name"] == "Bob"

    def test_apply_text_confirms_and_is_stdout_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ids = _seed_bank("x", ["Alice"])
        _run(["forget-all", "--bank", "x", "--apply"])
        out = capsys.readouterr()
        assert ids["Alice"] in out.out
        assert out.err == ""

        store = FaceStore(base_dir=bank_dir("x"))
        assert store.permanent_count == 0

    def test_apply_empty_bank_exits_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run(["forget-all", "--bank", "empty", "--apply", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 0
        assert payload["dry_run"] is False
        assert payload["faces"] == []


# ---------------------------------------------------------------------------
# cross-cutting: JSON always parses, text never leaks to stderr
# ---------------------------------------------------------------------------


def test_all_json_shapes_parse(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_bank("x", ["Alice", "Bob"])

    _run(["list", "--bank", "x", "--json"])
    json.loads(capsys.readouterr().out)

    _run(["forget-all", "--bank", "x", "--json"])  # dry run, no mutation
    json.loads(capsys.readouterr().out)

    ids = _seed_bank("y", ["Carol"])
    _run(["forget", ids["Carol"], "--bank", "y", "--json"])
    json.loads(capsys.readouterr().out)

    _run(["forget-all", "--bank", "x", "--apply", "--json"])
    json.loads(capsys.readouterr().out)
