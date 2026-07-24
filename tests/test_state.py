"""Tests for ``face_recognition_cli.state`` — the state-dir + bank layout.

Every test isolates itself via ``monkeypatch`` on the resolution-chain env vars
and ``tmp_path`` for any filesystem writes — never the real home directory, so
this suite is safe to run on a developer machine or CI without side effects.
"""

from __future__ import annotations

import pytest

from face_recognition_cli import state


def _clear_state_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACE_RECOGNITION_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


# --- state_dir() resolution chain ------------------------------------------


def test_state_dir_uses_face_recognition_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_state_env(monkeypatch)
    override = tmp_path / "override-state"
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(override))
    assert state.state_dir() == override


def test_state_dir_falls_back_to_xdg_state_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_state_env(monkeypatch)
    xdg = tmp_path / "xdg-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    assert state.state_dir() == xdg / "face-recognition-cli"


def test_state_dir_falls_back_to_home_when_neither_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setattr(state.Path, "home", classmethod(lambda cls: tmp_path))
    assert state.state_dir() == tmp_path / ".local" / "state" / "face-recognition-cli"


def test_state_dir_override_beats_xdg_state_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``$FACE_RECOGNITION_STATE_DIR`` wins even when ``$XDG_STATE_HOME`` is also set."""
    _clear_state_env(monkeypatch)
    override = tmp_path / "override-wins"
    xdg = tmp_path / "xdg-state"
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(override))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    resolved = state.state_dir()
    assert resolved == override
    assert resolved != xdg / "face-recognition-cli"


def test_state_dir_creates_the_directory(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_state_env(monkeypatch)
    override = tmp_path / "created-state"
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(override))
    assert not override.exists()
    resolved = state.state_dir()
    assert resolved.is_dir()


# --- resolve_bank() ----------------------------------------------------------


def test_resolve_bank_defaults_to_default_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_state_env(monkeypatch)
    assert state.resolve_bank() == state.DEFAULT_BANK
    assert state.DEFAULT_BANK == "default"


def test_resolve_bank_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_BANK", "reachy")
    assert state.resolve_bank() == "reachy"


def test_resolve_bank_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_BANK", "reachy")
    assert state.resolve_bank("colleague") == "colleague"


def test_resolve_bank_explicit_arg_beats_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_state_env(monkeypatch)
    assert state.resolve_bank("colleague") == "colleague"


def test_resolve_bank_none_falls_through_to_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_state_env(monkeypatch)
    assert state.resolve_bank(None) == state.DEFAULT_BANK
    monkeypatch.setenv("FACE_RECOGNITION_BANK", "reachy")
    assert state.resolve_bank(None) == "reachy"


# --- bank_dir() ---------------------------------------------------------------


def test_bank_dir_is_under_state_dir_banks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    assert state.bank_dir("reachy") == tmp_path / "banks" / "reachy"


def test_bank_dir_differs_per_bank_name(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    reachy_dir = state.bank_dir("reachy")
    default_dir = state.bank_dir("default")
    assert reachy_dir != default_dir
    assert reachy_dir.parent == default_dir.parent
    assert reachy_dir.parent == tmp_path / "banks"


def test_bank_dir_uses_resolve_bank_when_name_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("FACE_RECOGNITION_BANK", "colleague")
    assert state.bank_dir() == tmp_path / "banks" / "colleague"


def test_bank_dir_does_not_create_the_directory(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The store owns creating its own tree — bank_dir() only computes a path."""
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    reachy_dir = state.bank_dir("reachy")
    assert not reachy_dir.exists()


# --- models_dir() — deliberately bank-independent ----------------------------


def test_models_dir_is_under_state_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    assert state.models_dir() == tmp_path / "models"


def test_models_dir_is_identical_regardless_of_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))

    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)
    no_bank_env = state.models_dir()

    monkeypatch.setenv("FACE_RECOGNITION_BANK", "reachy")
    reachy_bank_env = state.models_dir()

    monkeypatch.setenv("FACE_RECOGNITION_BANK", "colleague")
    colleague_bank_env = state.models_dir()

    assert no_bank_env == reachy_bank_env == colleague_bank_env == tmp_path / "models"


def test_models_dir_unaffected_by_explicit_bank_dir_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Calling ``bank_dir()`` for different banks must never move ``models_dir()``."""
    _clear_state_env(monkeypatch)
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path))
    state.bank_dir("reachy")
    state.bank_dir("colleague")
    assert state.models_dir() == tmp_path / "models"


def test_state_dir_env_var_name_is_exact(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Sanity check the exact env var name so a typo can't silently pass other tests."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "exact-name-check"))
    assert state.state_dir() == tmp_path / "exact-name-check"
