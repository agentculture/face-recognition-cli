"""Cross-cutting integration tests for the wired domain verbs (task t8).

Every test here drives the **real installed CLI surface** —
:func:`face_recognition_cli.cli.main` with an argv list — rather than building
a throwaway parser around one verb module. That is the point of this file: the
per-verb suites (``test_verbs_enroll_match.py``, ``test_verbs_manage.py``)
prove each handler in isolation against a parser they construct themselves, so
nothing there would notice if a verb were never registered in
``cli/__init__.py``. These tests fail if the wiring is missing.

Four properties are proven end to end:

* **Reachability** — all five verbs parse, appear in the top-level help, and
  resolve through ``explain``; and *every* registered verb has a catalog entry
  (``test_every_catalog_path_resolves`` in ``test_cli.py`` only checks the
  converse — that existing entries resolve — so a verb added without an entry
  would otherwise slip through silently).
* **Bank isolation (h1)** — two banks seeded directly through
  :class:`~face_recognition_cli.store.FaceStore`, then cross-checked through
  the CLI: ``list``/``forget-all`` on one bank never observe or touch the
  other, while ``models/`` stays shared and bank-independent.
* **Bare-install behaviour (h7)** — with ``cv2`` absent, the introspection
  verbs and the three store-only verbs work fully, while ``enroll``/``match``
  fail with a clean exit-2 naming the ``[cpu]`` extra. ``cv2`` is blocked with
  ``sys.modules["cv2"] = None`` (the seam the rest of the suite uses), so the
  proof holds whether or not opencv happens to be installed locally.
* **One model pair (h7)** — the ``[gpu]`` path cannot silently reference
  different ONNX files, enforced by scanning the package source.

Seeding goes through ``FaceStore`` directly rather than through ``enroll``
because ``enroll`` needs a detector: the store and the numpy embeddings need
no opencv at all, so these tests never touch cv2, a model file, or the network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli import engine as engine_mod
from face_recognition_cli import state
from face_recognition_cli.cli import _build_parser, main
from face_recognition_cli.explain import known_paths
from face_recognition_cli.store import FaceStore

#: The five domain verbs task t8 wires in.
DOMAIN_VERBS = ("enroll", "match", "list", "forget", "forget-all")

#: The template's introspection verbs, which must keep working untouched.
INTROSPECTION_VERBS = ("whoami", "learn", "explain", "overview", "doctor")

#: Verbs that touch only the store — no detector, no opencv, ever.
STORE_ONLY_VERBS = ("list", "forget", "forget-all")

#: Directory holding the installed package source (for the source scans).
PACKAGE_ROOT = Path(engine_mod.__file__).resolve().parent

# Real PNG magic so the fixture files below are plausible input rather than
# arbitrary noise; nothing ever decodes them (cv2 is blocked first).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin the state dir into a throwaway tree — never the real home."""
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


@pytest.fixture
def no_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block opencv: ``sys.modules[name] = None`` makes ``import name`` raise."""
    monkeypatch.setitem(sys.modules, "cv2", None)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    """A real, non-empty file on disk — enough to get past the byte read."""
    path = tmp_path / "frame.png"
    path.write_bytes(_PNG_MAGIC + b"\x00" * 64)
    return path


def _unit(axis: int) -> np.ndarray:
    """A unit vector along *axis* — orthogonal to every other ``_unit``."""
    vec = np.zeros(engine_mod.EMBEDDING_DIM, dtype=float)
    vec[axis] = 1.0
    return vec


def _seed_bank(bank: str, names: dict[str, int]) -> dict[str, str]:
    """Enroll ``{name: axis}`` straight into *bank*; return ``{name: face_id}``."""
    store = FaceStore(base_dir=state.bank_dir(bank))
    return {name: store.enroll(name, _unit(axis)) for name, axis in names.items()}


def _snapshot(bank: str) -> dict[str, bytes]:
    """Byte-exact snapshot of every file under a bank's directory."""
    root = state.bank_dir(bank)
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_json(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    """Run *argv* with ``--json``, assert exit 0 + clean stderr, return stdout."""
    rc = main([*argv, "--json"])
    captured = capsys.readouterr()
    assert rc == 0, f"{argv} exited {rc}: {captured.err}"
    assert captured.err == ""
    return json.loads(captured.out)


def _registered_verbs() -> list[str]:
    """Top-level subcommand names the real parser actually registers."""
    parser = _build_parser()
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        if action.choices:
            return list(action.choices)
    raise AssertionError("no subparsers action found on the top-level parser")


# ---------------------------------------------------------------------------
# Reachability — the wiring itself
# ---------------------------------------------------------------------------


class TestVerbsAreReachable:
    """The five verbs are registered on the real top-level parser."""

    @pytest.mark.parametrize("verb", DOMAIN_VERBS)
    def test_verb_help_exits_zero(self, verb: str, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main([verb, "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert f"face-recognition-cli {verb}" in captured.out
        assert "--json" in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize("verb", DOMAIN_VERBS)
    def test_verb_is_listed_in_top_level_help(
        self, verb: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 0
        help_text = capsys.readouterr().out
        # Match the subcommand table line ("  <verb>   <help>"), not an
        # incidental mention inside another verb's help string.
        assert re.search(rf"^\s+{re.escape(verb)}\s", help_text, re.M), help_text

    def test_the_introspection_verbs_survive_the_wiring(self) -> None:
        registered = _registered_verbs()
        for verb in (*INTROSPECTION_VERBS, "cli", *DOMAIN_VERBS):
            assert verb in registered, f"{verb} is not registered"

    @pytest.mark.parametrize("verb", DOMAIN_VERBS)
    def test_verb_carries_a_bank_flag(self, verb: str, capsys: pytest.CaptureFixture[str]) -> None:
        """Every store-touching verb selects its bank — c12's CLI half."""
        with pytest.raises(SystemExit):
            main([verb, "--help"])
        assert "--bank" in capsys.readouterr().out


class TestExplainCoversTheDomain:
    """``explain`` resolves for every new verb, plus the banks concept."""

    @pytest.mark.parametrize("topic", [*DOMAIN_VERBS, "banks"])
    def test_explain_resolves(self, topic: str, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["explain", topic])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert captured.out.startswith("#")
        assert "face-recognition-cli" in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize("topic", [*DOMAIN_VERBS, "banks"])
    def test_explain_json_shape(self, topic: str, capsys: pytest.CaptureFixture[str]) -> None:
        payload = _run_json(["explain", topic], capsys)
        assert payload["path"] == [topic]
        assert payload["markdown"].startswith("#")

    def test_every_registered_verb_has_a_catalog_entry(self) -> None:
        """The converse of ``test_every_catalog_path_resolves``.

        That test walks the catalog and checks each entry resolves; nothing
        fails when a *verb* lacks an entry. This walks the parser instead, so
        a future verb wired without a catalog key fails here.
        """
        paths = set(known_paths())
        missing = [verb for verb in _registered_verbs() if (verb,) not in paths]
        assert missing == []

    def test_root_entry_lists_the_domain_verbs(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["explain", "face-recognition-cli"]) == 0
        root = capsys.readouterr().out
        for verb in DOMAIN_VERBS:
            assert f"face-recognition-cli {verb}" in root
        assert "banks" in root

    def test_banks_entry_documents_the_layout_and_the_env_override(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["explain", "banks"]) == 0
        body = capsys.readouterr().out
        assert "banks/" in body
        assert "FACE_RECOGNITION_BANK" in body
        assert "models/" in body


# ---------------------------------------------------------------------------
# h1 — two banks, fully isolated, one shared models dir
# ---------------------------------------------------------------------------


class TestTwoBankIsolation:
    """An identity in one bank is invisible — and untouchable — from another."""

    @pytest.fixture
    def banks(self) -> dict[str, dict[str, str]]:
        """Seed bank 'reachy' (2 identities) and bank 'colleague' (1)."""
        return {
            "reachy": _seed_bank("reachy", {"ada": 0, "grace": 1}),
            "colleague": _seed_bank("colleague", {"linus": 2}),
        }

    def test_list_sees_only_its_own_bank(
        self, banks: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        reachy = _run_json(["list", "--bank", "reachy"], capsys)
        assert reachy["bank"] == "reachy"
        assert reachy["count"] == 2
        assert {face["name"] for face in reachy["faces"]} == {"ada", "grace"}
        assert {face["id"] for face in reachy["faces"]} == set(banks["reachy"].values())

        colleague = _run_json(["list", "--bank", "colleague"], capsys)
        assert colleague["bank"] == "colleague"
        assert {face["name"] for face in colleague["faces"]} == {"linus"}

    def test_the_default_bank_sees_neither(
        self, banks: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = _run_json(["list"], capsys)
        assert payload == {"bank": "default", "faces": [], "count": 0}

    def test_forget_all_dry_run_deletes_nothing(
        self, banks: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        before = _snapshot("reachy")

        payload = _run_json(["forget-all", "--bank", "reachy"], capsys)

        assert payload["dry_run"] is True
        assert payload["count"] == 2
        assert {face["name"] for face in payload["faces"]} == {"ada", "grace"}
        assert _snapshot("reachy") == before
        assert _run_json(["list", "--bank", "reachy"], capsys)["count"] == 2

    def test_forget_all_apply_empties_one_bank_only(
        self, banks: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        colleague_before = _snapshot("colleague")

        payload = _run_json(["forget-all", "--bank", "reachy", "--apply"], capsys)

        assert payload["dry_run"] is False
        assert payload["count"] == 2

        # reachy is empty ...
        assert _run_json(["list", "--bank", "reachy"], capsys)["count"] == 0
        # ... and colleague is byte-for-byte what it was, on disk.
        assert _snapshot("colleague") == colleague_before
        after = _run_json(["list", "--bank", "colleague"], capsys)
        assert after["count"] == 1
        assert after["faces"][0]["name"] == "linus"
        assert after["faces"][0]["id"] == banks["colleague"]["linus"]

    def test_forget_cannot_reach_across_banks(
        self, banks: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        foreign = banks["colleague"]["linus"]

        rc = main(["forget", foreign, "--bank", "reachy"])

        captured = capsys.readouterr()
        assert rc == 1
        assert captured.err.startswith("error:")
        assert "hint:" in captured.err
        assert _snapshot("colleague") == _snapshot("colleague")
        assert _run_json(["list", "--bank", "colleague"], capsys)["count"] == 1

    def test_the_bank_env_override_drives_the_cli(
        self,
        banks: dict[str, dict[str, str]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("FACE_RECOGNITION_BANK", "colleague")

        payload = _run_json(["list"], capsys)

        assert payload["bank"] == "colleague"
        assert {face["name"] for face in payload["faces"]} == {"linus"}

    def test_models_dir_is_shared_and_carries_no_bank_segment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Banks partition identities, not models (state.py's rationale)."""
        shared = state.models_dir()
        assert shared == state.state_dir() / "models"
        assert "banks" not in shared.parts

        for bank in ("reachy", "colleague"):
            monkeypatch.setenv("FACE_RECOGNITION_BANK", bank)
            assert state.models_dir() == shared, "models dir moved with the bank"
            assert bank not in shared.parts
            # The bank dir, by contrast, does move.
            assert state.bank_dir().name == bank
            assert state.bank_dir().parent == state.state_dir() / "banks"


# ---------------------------------------------------------------------------
# h7 — a bare install (neither [cpu] nor [gpu]) stays usable
# ---------------------------------------------------------------------------


class TestBareInstall:
    """No opencv: introspection + store verbs work; engine verbs exit 2."""

    def test_the_dev_venv_installs_neither_extra(self) -> None:
        """The documented dev/CI baseline: no ``[cpu]``, no ``[gpu]``.

        CI installs the dev group only, so this holds there. If you install
        ``[cpu]`` locally (for the skippable e2e roundtrip), this is the one
        test that notices — every behavioural proof below blocks ``cv2``
        through ``sys.modules`` and stays green either way.
        """
        assert importlib.util.find_spec("cv2") is None

    @pytest.mark.parametrize(
        "argv",
        [
            ["whoami"],
            ["learn"],
            ["explain", "face-recognition-cli"],
            ["overview"],
            ["doctor"],
            ["cli", "overview"],
        ],
    )
    def test_introspection_verbs_work_without_opencv(
        self, argv: list[str], no_cv2: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(argv)
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert captured.out.strip()
        assert "Traceback" not in captured.err

    def test_store_only_verbs_work_fully_without_opencv(
        self, no_cv2: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ids = _seed_bank("solo", {"ada": 0, "grace": 1})

        listed = _run_json(["list", "--bank", "solo"], capsys)
        assert listed["count"] == 2

        forgotten = _run_json(["forget", ids["ada"], "--bank", "solo"], capsys)
        assert forgotten == {"forgotten": ids["ada"], "bank": "solo"}

        wiped = _run_json(["forget-all", "--bank", "solo", "--apply"], capsys)
        assert wiped["count"] == 1
        assert _run_json(["list", "--bank", "solo"], capsys)["count"] == 0

    @pytest.mark.parametrize("verb", ["enroll", "match"])
    def test_engine_verbs_exit_two_naming_the_cpu_extra(
        self,
        verb: str,
        no_cv2: None,
        image: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        argv = [verb, "--image", str(image)]
        if verb == "enroll":
            argv += ["--name", "ada"]

        rc = main(argv)

        captured = capsys.readouterr()
        assert rc == 2, captured.err
        assert captured.out == ""
        assert captured.err.startswith("error:")
        assert "hint:" in captured.err
        assert "face-recognition-cli[cpu]" in captured.err
        assert "Traceback" not in captured.err

    def test_the_missing_extra_error_is_json_in_json_mode(
        self, no_cv2: None, image: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["match", "--image", str(image), "--json"])

        captured = capsys.readouterr()
        assert rc == 2
        assert captured.out == ""
        payload = json.loads(captured.err)
        assert payload["code"] == 2
        assert "[cpu]" in payload["remediation"]


class TestSingleModelPair:
    """The package names exactly one YuNet + SFace pair — h7's hard invariant."""

    @staticmethod
    def _sources() -> list[Path]:
        return sorted(PACKAGE_ROOT.rglob("*.py"))

    def test_the_model_zoo_is_referenced_only_by_engine(self) -> None:
        hits = {
            path.relative_to(PACKAGE_ROOT).as_posix(): path.read_text().count("opencv_zoo")
            for path in self._sources()
            if "opencv_zoo" in path.read_text()
        }
        # Exactly two: the YuNet URL and the SFace URL, both in engine.py.
        assert hits == {"engine.py": 2}

    @pytest.mark.parametrize("attr", ["YUNET_FILE", "SFACE_FILE"])
    def test_model_filenames_are_constants_in_engine_only(self, attr: str) -> None:
        filename = getattr(engine_mod, attr)
        assert filename.endswith(".onnx")
        holders = [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in self._sources()
            if filename in path.read_text()
        ]
        assert holders == ["engine.py"], f"{attr} leaked outside engine.py"

    def test_both_urls_point_at_the_same_zoo_revision(self) -> None:
        for url in (engine_mod.YUNET_URL, engine_mod.SFACE_URL):
            assert url.startswith("https://github.com/opencv/opencv_zoo/raw/main/models/")
        assert engine_mod.YUNET_URL.endswith(engine_mod.YUNET_FILE)
        assert engine_mod.SFACE_URL.endswith(engine_mod.SFACE_FILE)


# ---------------------------------------------------------------------------
# The structured-error contract, through the wired verbs
# ---------------------------------------------------------------------------


class TestErrorContract:
    """No traceback, ever; ``hint:`` in text mode, parseable JSON in JSON mode."""

    def test_missing_required_args_is_a_two_line_user_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["enroll"])

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error:")
        assert "hint:" in captured.err
        assert "Traceback" not in captured.err

    def test_missing_required_args_renders_json_when_asked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Parse-time errors fire before ``args.json`` exists — the argv
        pre-scan in ``main()`` is what keeps them JSON."""
        with pytest.raises(SystemExit) as exc:
            main(["enroll", "--json"])

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        payload = json.loads(captured.err)
        assert set(payload) == {"code", "message", "remediation"}
        assert payload["code"] == 1

    @pytest.mark.parametrize("verb", STORE_ONLY_VERBS)
    def test_an_unknown_flag_stays_structured(
        self, verb: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            main([verb, "--bogus"])

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.err.startswith("error:")
        assert "hint:" in captured.err
        assert "Traceback" not in captured.err

    def test_forgetting_an_unknown_id_is_a_user_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["forget", "zzzz"])

        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == ""
        assert captured.err.startswith("error:")
        assert "hint:" in captured.err
        assert "Traceback" not in captured.err

    def test_forgetting_an_unknown_id_renders_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["forget", "zzzz", "--json"])

        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == ""
        payload = json.loads(captured.err)
        assert payload["code"] == 1
        assert payload["remediation"]

    @pytest.mark.parametrize("verb", DOMAIN_VERBS)
    def test_no_verb_ever_leaks_a_traceback(
        self, verb: str, no_cv2: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every failure mode of every verb routes through ``_dispatch``."""
        try:
            main([verb, "--image", "/no/such/file.png", "--name", "x", "zzzz"])
        except SystemExit as exc:  # argparse rejected the argv shape
            assert exc.code == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "Traceback" not in captured.out
