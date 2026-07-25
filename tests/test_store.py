"""Tests for ``face_recognition_cli.store`` — the ported ``FaceStore`` (task t3).

TDD — these tests were written before ``face_recognition_cli/store.py`` existed
and define the contract the port must satisfy.

The store's matching math (cosine threshold, temporary-vs-permanent tiers, TTL
expiry) and its on-disk persistence are pure ``numpy`` + stdlib: every test here
runs with **no cv2**, no network, and no camera, using synthetic 128-dim
embeddings. Nothing touches the real home directory — filesystem writes go to
``tmp_path`` and :func:`default_base_dir` is exercised with
``$FACE_RECOGNITION_STATE_DIR`` pinned into ``tmp_path``.

Compatibility fixture (claim h2)
--------------------------------
``tests/fixtures/reachy_store/`` is **genuinely reachy-written**: it was produced
by importing ``reachy.vision.face_store.FaceStore`` from the sibling
``reachy-mini-cli`` checkout and calling its real ``enroll()`` with
``base_dir=`` pointed at that directory, so the committed ``faces.json`` and
``embeddings/*.npy`` bytes came out of reachy's own write path rather than a
hand-rolled imitation of it. The generator was a throwaway script — no test in
this file imports ``reachy.*``, and this repo does not depend on it.
``TestReachyFixtureCompatibility`` below loads that tree through the *ported*
store and proves the embedding space and on-disk layout survived extraction.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from face_recognition_cli.store import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_TEMP_TTL,
    FaceMatch,
    FaceStore,
    cosine_similarity,
    default_base_dir,
)

#: The committed, reachy-written compatibility fixture.
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reachy_store"

#: The identities the fixture was enrolled with, via reachy's own ``enroll()``.
FIXTURE_NAMES = {"Alice Example", "Bob Example", "Carol Example"}

_ID_ALPHABET = set("abcdefghijklmnopqrstuvwxyz0123456789")


def _embedding(seed: int) -> np.ndarray:
    """A deterministic, non-degenerate 128-dim synthetic embedding."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=128).astype(float)


def _assert_valid_face_id(face_id: str) -> None:
    """A face id is 4 lowercase-alnum characters (reachy's ``_generate_id``)."""
    assert len(face_id) == 4
    assert set(face_id) <= _ID_ALPHABET


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin the state-dir resolution chain into a throwaway dir.

    Guards the default-path tests (and any accidental default construction)
    against ever touching the developer's real ``~/.local/state``.
    """
    monkeypatch.setenv("FACE_RECOGNITION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FACE_RECOGNITION_BANK", raising=False)


# ---------------------------------------------------------------------------
# cosine_similarity — pure math
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = _embedding(1)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        v = _embedding(2)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_near_zero_vector_degrades_to_zero(self) -> None:
        """A (near) zero vector is defined as similarity 0.0, never a ZeroDivision."""
        v = _embedding(3)
        zero = np.zeros(128)
        assert cosine_similarity(v, zero) == 0.0
        assert cosine_similarity(zero, zero) == 0.0
        assert cosine_similarity(v, np.full(128, 1e-12)) == 0.0


# ---------------------------------------------------------------------------
# Compatibility with a reachy-written store (claim h2)
# ---------------------------------------------------------------------------


class TestReachyFixtureCompatibility:
    """The whole reason extraction beat adopting dlib: the store stays readable."""

    def _raw_index(self) -> dict:
        return json.loads((FIXTURE_DIR / "faces.json").read_text(encoding="utf-8"))

    def test_fixture_is_in_reachys_on_disk_format(self) -> None:
        """Guards the fixture itself: index filename, record keys, .npy dtype/shape."""
        raw = self._raw_index()
        assert len(raw) == 3
        for face_id, record in raw.items():
            _assert_valid_face_id(face_id)
            assert set(record) == {"name", "created", "embedding_files"}
            assert record["embedding_files"] == [f"{face_id}.npy"]
            stored = np.load(FIXTURE_DIR / "embeddings" / f"{face_id}.npy")
            assert stored.shape == (128,)
            assert stored.dtype == np.float64

    def test_reachy_written_index_loads_through_the_ported_store(self) -> None:
        store = FaceStore(base_dir=FIXTURE_DIR)
        store.load()  # must not raise

        raw = self._raw_index()
        assert store.permanent_count == len(raw)

        listed = {entry["id"]: entry for entry in store.list_faces()}
        assert set(listed) == set(raw)
        assert {entry["name"] for entry in listed.values()} == FIXTURE_NAMES
        for face_id, entry in listed.items():
            assert entry["name"] == raw[face_id]["name"]
            assert entry["created"] == raw[face_id]["created"]
            assert entry["num_embeddings"] == 1

    def test_reachy_written_embedding_matches_its_own_identity(self) -> None:
        """A query equal to a stored embedding resolves to that stored identity."""
        store = FaceStore(base_dir=FIXTURE_DIR)
        raw = self._raw_index()

        for face_id, record in raw.items():
            stored = np.load(FIXTURE_DIR / "embeddings" / f"{face_id}.npy")
            match = store.match(stored)
            assert match is not None
            assert match.face_id == face_id
            assert match.name == record["name"]
            assert match.score == pytest.approx(1.0)

    def test_reachy_written_embedding_matches_a_nearby_query(self) -> None:
        """Not just exact equality — a perturbed query still lands on the identity."""
        store = FaceStore(base_dir=FIXTURE_DIR)
        raw = self._raw_index()
        face_id = sorted(raw)[0]
        stored = np.load(FIXTURE_DIR / "embeddings" / f"{face_id}.npy")
        noisy = stored + np.random.default_rng(4242).normal(scale=0.05, size=stored.shape)

        match = store.match(noisy)
        assert match is not None
        assert match.face_id == face_id
        assert match.name == raw[face_id]["name"]
        assert match.score > 0.9

    def test_reachy_written_names_resolve_case_insensitively(self) -> None:
        store = FaceStore(base_dir=FIXTURE_DIR)
        raw = self._raw_index()
        for face_id, record in raw.items():
            assert store.get_unique_id(record["name"].upper()) == face_id

    def test_ported_store_can_extend_a_reachy_written_tree(self, tmp_path: Path) -> None:
        """Round trip the other way: enrol into reachy's tree, keep its records."""
        work_dir = tmp_path / "reachy_store"
        shutil.copytree(FIXTURE_DIR, work_dir)
        raw_before = json.loads((work_dir / "faces.json").read_text(encoding="utf-8"))

        store = FaceStore(base_dir=work_dir)
        new_id = store.enroll("Dave Example", _embedding(500))

        raw_after = json.loads((work_dir / "faces.json").read_text(encoding="utf-8"))
        assert set(raw_after) == set(raw_before) | {new_id}
        for face_id, record in raw_before.items():
            assert raw_after[face_id] == record
        assert set(raw_after[new_id]) == {"name", "created", "embedding_files"}

    def test_fixture_directory_is_not_mutated_by_read_only_use(self) -> None:
        """Reading the committed fixture must leave the checked-in bytes alone."""
        before = (FIXTURE_DIR / "faces.json").read_bytes()
        store = FaceStore(base_dir=FIXTURE_DIR)
        store.list_faces()
        store.match(_embedding(7))
        assert (FIXTURE_DIR / "faces.json").read_bytes() == before
        assert not (FIXTURE_DIR / "faces.json.tmp").exists()


# ---------------------------------------------------------------------------
# Permanent tier — enroll / match / forget / list
# ---------------------------------------------------------------------------


class TestEnroll:
    def test_enroll_returns_a_four_char_lowercase_alnum_id(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Ada Lovelace", _embedding(10))
        _assert_valid_face_id(face_id)

    def test_enroll_ids_are_unique_across_calls(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        ids = {store.enroll(f"Person {i}", _embedding(200 + i)) for i in range(10)}
        assert len(ids) == 10
        assert store.permanent_count == 10

    def test_enroll_persists_index_and_embedding_in_reachys_layout(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Heidi", _embedding(101), now=1234.5)

        index_path = tmp_path / "faces.json"
        emb_path = tmp_path / "embeddings" / f"{face_id}.npy"
        assert index_path.exists()
        assert emb_path.exists()

        body = json.loads(index_path.read_text(encoding="utf-8"))
        assert set(body[face_id]) == {"name", "created", "embedding_files"}
        assert body[face_id]["name"] == "Heidi"
        assert body[face_id]["created"] == 1234.5
        assert body[face_id]["embedding_files"] == [f"{face_id}.npy"]

        stored = np.load(emb_path)
        assert stored.shape == (128,)
        assert np.allclose(stored, _embedding(101))

    def test_enroll_shows_up_in_list_faces(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Dana", _embedding(50), now=99.0)

        listed = store.list_faces()
        assert listed == [{"id": face_id, "name": "Dana", "created": 99.0, "num_embeddings": 1}]

    def test_enroll_survives_a_fresh_store_instance(self, tmp_path: Path) -> None:
        face_id = FaceStore(base_dir=tmp_path).enroll("Grace", _embedding(100))

        reopened = FaceStore(base_dir=tmp_path)
        match = reopened.match(_embedding(100))
        assert match is not None
        assert match.face_id == face_id
        assert match.name == "Grace"

    def test_enroll_defaults_created_to_the_injected_clock(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path, clock=lambda: 777.0)
        store.enroll("Ivy", _embedding(103))
        assert store.list_faces()[0]["created"] == 777.0


class TestMatch:
    def test_default_threshold_is_one_half(self, tmp_path: Path) -> None:
        assert DEFAULT_MATCH_THRESHOLD == 0.5
        assert FaceStore(base_dir=tmp_path).threshold == 0.5

    def test_match_returns_a_facematch_for_the_enrolled_embedding(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        emb = _embedding(10)
        face_id = store.enroll("Ada Lovelace", emb)

        match = store.match(emb)
        assert isinstance(match, FaceMatch)
        assert match.face_id == face_id
        assert match.name == "Ada Lovelace"
        assert match.score == pytest.approx(1.0)

    def test_match_returns_none_for_an_unrelated_embedding(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.enroll("Ada Lovelace", _embedding(11))
        assert store.match(_embedding(12)) is None

    def test_match_returns_none_on_an_empty_store(self, tmp_path: Path) -> None:
        assert FaceStore(base_dir=tmp_path).match(_embedding(13)) is None

    def test_per_call_threshold_overrides_the_default(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path, threshold=0.5)
        emb = _embedding(20)
        near = emb + np.random.default_rng(21).normal(scale=0.3, size=emb.shape)
        store.enroll("Grace Hopper", emb)

        baseline = store.match(near)
        assert baseline is not None
        score = baseline.score
        assert store.match(near, threshold=score + 0.01) is None
        assert store.match(near, threshold=score - 0.01) is not None

    def test_constructor_threshold_overrides_the_module_default(self, tmp_path: Path) -> None:
        emb = _embedding(30)
        near = emb + np.random.default_rng(31).normal(scale=3.0, size=emb.shape)

        permissive = FaceStore(base_dir=tmp_path, threshold=0.0)
        permissive.enroll("Mary Jackson", emb)
        strict = FaceStore(base_dir=tmp_path, threshold=0.99)

        assert permissive.match(near) is not None
        assert strict.match(near) is None

    def test_best_of_multiple_candidates_wins(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        emb_a = _embedding(40)
        emb_b = _embedding(41)
        id_a = store.enroll("Alice", emb_a)
        store.enroll("Bob", emb_b)

        match = store.match(emb_a)
        assert match is not None
        assert match.face_id == id_a
        assert match.name == "Alice"

    def test_missing_embedding_file_is_skipped_not_raised(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Ivy", _embedding(102))
        (tmp_path / "embeddings" / f"{face_id}.npy").unlink()

        assert store.match(_embedding(102)) is None

    def test_corrupt_embedding_file_is_skipped_not_raised(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        good_id = store.enroll("Good", _embedding(110))
        bad_id = store.enroll("Bad", _embedding(111))
        (tmp_path / "embeddings" / f"{bad_id}.npy").write_bytes(b"not a npy file at all")

        match = store.match(_embedding(110))
        assert match is not None
        assert match.face_id == good_id


class TestForget:
    def test_forget_removes_the_record_and_its_embedding_file(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        emb = _embedding(60)
        face_id = store.enroll("Carol", emb)
        emb_path = tmp_path / "embeddings" / f"{face_id}.npy"
        assert emb_path.exists()

        assert store.forget(face_id) is True
        assert not emb_path.exists()
        assert store.match(emb) is None
        assert store.permanent_count == 0
        assert store.list_faces() == []

        body = json.loads((tmp_path / "faces.json").read_text(encoding="utf-8"))
        assert face_id not in body

    def test_forget_unknown_id_returns_false(self, tmp_path: Path) -> None:
        assert FaceStore(base_dir=tmp_path).forget("nope") is False

    def test_forget_leaves_other_identities_intact(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        doomed = store.enroll("Doomed", _embedding(61))
        keeper = store.enroll("Keeper", _embedding(62))

        assert store.forget(doomed) is True
        assert [entry["id"] for entry in store.list_faces()] == [keeper]
        assert (tmp_path / "embeddings" / f"{keeper}.npy").exists()

    def test_forget_tolerates_an_already_missing_embedding_file(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Ghost", _embedding(63))
        (tmp_path / "embeddings" / f"{face_id}.npy").unlink()

        assert store.forget(face_id) is True
        assert store.permanent_count == 0


class TestLookups:
    def test_get_unique_id_is_case_and_whitespace_insensitive(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Erin Malone", _embedding(70))

        assert store.get_unique_id("Erin Malone") == face_id
        assert store.get_unique_id("erin malone") == face_id
        assert store.get_unique_id("  ERIN MALONE  ") == face_id

    def test_get_unique_id_unknown_name_returns_none(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.enroll("Erin Malone", _embedding(71))
        assert store.get_unique_id("nobody") is None

    def test_permanent_count_tracks_enrolments(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        assert store.permanent_count == 0
        store.enroll("Fay", _embedding(72))
        assert store.permanent_count == 1
        store.enroll("Gil", _embedding(73))
        assert store.permanent_count == 2

    def test_list_faces_is_empty_on_a_fresh_store(self, tmp_path: Path) -> None:
        assert FaceStore(base_dir=tmp_path).list_faces() == []


# ---------------------------------------------------------------------------
# Temporary tier — TTL, injected clock, never matched
# ---------------------------------------------------------------------------


class TestTemporaryTier:
    def test_remember_temporary_returns_a_tmp_prefixed_id(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        temp_id = store.remember_temporary(_embedding(80))
        assert temp_id.startswith("tmp_")
        _assert_valid_face_id(temp_id.removeprefix("tmp_"))
        assert store.temporary_count == 1

    def test_get_temporary_round_trips_the_embedding(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        emb = _embedding(81)
        temp_id = store.remember_temporary(emb)

        got = store.get_temporary(temp_id)
        assert got is not None
        assert np.allclose(got, emb)

    def test_get_temporary_returns_a_copy_not_the_original(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        temp_id = store.remember_temporary(_embedding(82))

        got = store.get_temporary(temp_id)
        assert got is not None
        got[0] = 999.0
        got_again = store.get_temporary(temp_id)
        assert got_again is not None
        assert got_again[0] != 999.0

    def test_get_temporary_missing_id_returns_none(self, tmp_path: Path) -> None:
        assert FaceStore(base_dir=tmp_path).get_temporary("tmp_nope") is None

    def test_temporary_tier_is_never_matched(self, tmp_path: Path) -> None:
        """match() searches the permanent tier only — a temp embedding never hits."""
        store = FaceStore(base_dir=tmp_path)
        emb = _embedding(83)
        store.remember_temporary(emb)
        assert store.match(emb) is None

    def test_temporary_tier_is_not_persisted(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.remember_temporary(_embedding(84))
        store.save()
        assert FaceStore(base_dir=tmp_path).temporary_count == 0

    def test_default_ttl_is_fifteen_minutes(self, tmp_path: Path) -> None:
        assert DEFAULT_TEMP_TTL == 15 * 60
        assert FaceStore(base_dir=tmp_path).ttl == DEFAULT_TEMP_TTL

    def test_cleanup_expired_evicts_only_entries_past_ttl(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path, ttl=DEFAULT_TEMP_TTL)

        old_id = store.remember_temporary(_embedding(90), now=1_000.0)
        fresh_id = store.remember_temporary(_embedding(91), now=1_000.0 + DEFAULT_TEMP_TTL - 1)

        removed = store.cleanup_expired(now=1_000.0 + DEFAULT_TEMP_TTL + 1)

        assert removed == 1
        assert store.get_temporary(old_id) is None
        assert store.get_temporary(fresh_id) is not None
        assert store.temporary_count == 1

    def test_cleanup_expired_with_nothing_expired_returns_zero(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.remember_temporary(_embedding(92), now=0.0)
        assert store.cleanup_expired(now=1.0) == 0
        assert store.temporary_count == 1

    def test_cleanup_expired_boundary_is_strictly_greater_than_ttl(self, tmp_path: Path) -> None:
        """Exactly ttl seconds old is still alive; one second past it is not."""
        store = FaceStore(base_dir=tmp_path, ttl=10.0)
        store.remember_temporary(_embedding(94), now=0.0)
        assert store.cleanup_expired(now=10.0) == 0
        assert store.cleanup_expired(now=10.001) == 1

    def test_injected_clock_drives_default_now(self, tmp_path: Path) -> None:
        """Omitting now= falls back to the injected clock, not wall-clock time."""
        fake_time = {"t": 500.0}
        store = FaceStore(base_dir=tmp_path, clock=lambda: fake_time["t"])

        temp_id = store.remember_temporary(_embedding(93))
        fake_time["t"] = 500.0 + DEFAULT_TEMP_TTL + 1
        removed = store.cleanup_expired()

        assert removed == 1
        assert store.get_temporary(temp_id) is None
        assert store.temporary_count == 0


# ---------------------------------------------------------------------------
# Persistence — load robustness + write-then-replace
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_missing_index_loads_as_empty(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path / "never-written")
        store.load()  # must not raise
        assert store.permanent_count == 0
        assert store.list_faces() == []

    def test_corrupt_index_degrades_to_empty_without_raising(self, tmp_path: Path) -> None:
        (tmp_path / "faces.json").write_text("{ not valid json", encoding="utf-8")

        store = FaceStore(base_dir=tmp_path)
        store.load()  # must not raise

        assert store.permanent_count == 0
        assert store.list_faces() == []

    def test_binary_garbage_index_degrades_to_empty(self, tmp_path: Path) -> None:
        (tmp_path / "faces.json").write_bytes(b"\x00\x01\x02\xff\xfe garbage \xde\xad\xbe\xef")

        store = FaceStore(base_dir=tmp_path)
        store.load()  # must not raise
        assert store.permanent_count == 0

    def test_non_object_index_degrades_to_empty(self, tmp_path: Path) -> None:
        """Valid JSON of the wrong shape (a list) is still 'start fresh'."""
        (tmp_path / "faces.json").write_text("[1, 2, 3]", encoding="utf-8")

        store = FaceStore(base_dir=tmp_path)
        store.load()
        assert store.permanent_count == 0

    def test_a_corrupt_index_can_be_enrolled_over(self, tmp_path: Path) -> None:
        (tmp_path / "faces.json").write_text("}{", encoding="utf-8")

        store = FaceStore(base_dir=tmp_path)
        face_id = store.enroll("Fresh Start", _embedding(120))

        body = json.loads((tmp_path / "faces.json").read_text(encoding="utf-8"))
        assert list(body) == [face_id]

    def test_load_creates_the_base_and_embeddings_directories(self, tmp_path: Path) -> None:
        base = tmp_path / "brand-new"
        FaceStore(base_dir=base).load()
        assert base.is_dir()
        assert (base / "embeddings").is_dir()

    def test_save_is_write_then_replace_leaving_no_tmp_behind(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.enroll("Atomic", _embedding(130))
        store.save()

        assert (tmp_path / "faces.json").exists()
        assert not (tmp_path / "faces.json.tmp").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["embeddings", "faces.json"]

    def test_save_writes_indented_json_readable_by_reachy(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path)
        store.enroll("Indented", _embedding(131))

        text = (tmp_path / "faces.json").read_text(encoding="utf-8")
        assert "\n  " in text  # json.dumps(..., indent=2)
        assert json.loads(text)

    def test_save_on_an_unwritable_path_does_not_raise(self, tmp_path: Path) -> None:
        """Persistence failures are logged, never propagated (reachy's contract)."""
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")

        store = FaceStore(base_dir=blocker / "store")
        store.save()  # must not raise


# ---------------------------------------------------------------------------
# default_base_dir() — the one deliberate deviation from reachy
# ---------------------------------------------------------------------------


class TestDefaultBaseDir:
    def test_default_base_dir_resolves_under_banks_default(self, tmp_path: Path) -> None:
        """The deviation: reachy's <state>/faces becomes <state>/banks/default here."""
        state_root = tmp_path / "state"
        assert default_base_dir() == state_root / "banks" / "default"

    def test_default_base_dir_honours_the_bank_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("FACE_RECOGNITION_BANK", "reachy")
        assert default_base_dir() == tmp_path / "state" / "banks" / "reachy"

    def test_store_without_base_dir_uses_the_default_bank(self, tmp_path: Path) -> None:
        store = FaceStore()
        assert store.base_dir == tmp_path / "state" / "banks" / "default"

    def test_banks_are_isolated_from_each_other(self, tmp_path: Path) -> None:
        """Two banks are two independent indexes — pure path composition."""
        alice_bank = FaceStore(base_dir=tmp_path / "banks" / "a")
        bob_bank = FaceStore(base_dir=tmp_path / "banks" / "b")

        emb = _embedding(140)
        alice_bank.enroll("Alice", emb)

        assert alice_bank.match(emb) is not None
        assert bob_bank.match(emb) is None
        assert bob_bank.list_faces() == []


# ---------------------------------------------------------------------------
# PR #3 review hardenings — traversal containment + persistence honesty
# ---------------------------------------------------------------------------


class TestEmbeddingPathContainment:
    """``embedding_files`` entries are data; they must stay in embeddings/."""

    def _store_with_crafted_index(self, base: Path, emb_file: str) -> FaceStore:
        base.mkdir(parents=True, exist_ok=True)
        (base / "faces.json").write_text(
            json.dumps(
                {"ab12": {"name": "mallory", "created": 1.0, "embedding_files": [emb_file]}}
            ),
            encoding="utf-8",
        )
        return FaceStore(base_dir=base)

    def test_match_skips_traversal_entries(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.npy"
        np.save(outside, _embedding(7))
        store = self._store_with_crafted_index(tmp_path / "bank", "../../outside.npy")

        assert store.match(_embedding(7)) is None  # never np.load()ed from outside

    def test_forget_never_unlinks_outside_the_embeddings_dir(self, tmp_path: Path) -> None:
        outside = tmp_path / "precious.npy"
        np.save(outside, _embedding(8))
        store = self._store_with_crafted_index(tmp_path / "bank", "../../precious.npy")

        assert store.forget("ab12") is True  # the record itself is removed
        assert outside.exists()  # the decoy outside the bank survives

    def test_absolute_paths_are_skipped_too(self, tmp_path: Path) -> None:
        outside = tmp_path / "abs.npy"
        np.save(outside, _embedding(9))
        store = self._store_with_crafted_index(tmp_path / "bank", str(outside))

        assert store.match(_embedding(9)) is None
        assert store.forget("ab12") is True
        assert outside.exists()

    def test_ordinary_filenames_still_work(self, tmp_path: Path) -> None:
        base = tmp_path / "bank"
        store = FaceStore(base_dir=base)
        face_id = store.enroll("Alice", _embedding(10))

        reloaded = FaceStore(base_dir=base)
        hit = reloaded.match(_embedding(10))
        assert hit is not None and hit.face_id == face_id


class TestEnrollPersistenceHonesty:
    """``enroll`` must not report success when the index never reached disk."""

    def test_save_returns_true_on_success(self, tmp_path: Path) -> None:
        store = FaceStore(base_dir=tmp_path / "bank")
        store.enroll("Alice", _embedding(11))
        assert store.save() is True

    def test_save_returns_false_when_the_index_cannot_land(self, tmp_path: Path) -> None:
        base = tmp_path / "bank"
        store = FaceStore(base_dir=base)
        store.load()
        # Make the index path unlandable: faces.json as a *directory* makes
        # the atomic replace fail with an OSError (IsADirectoryError).
        (base / "faces.json").mkdir(parents=True, exist_ok=True)
        assert store.save() is False

    def test_enroll_rolls_back_and_raises_when_save_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = tmp_path / "bank"
        store = FaceStore(base_dir=base)
        store.load()
        monkeypatch.setattr(store, "save", lambda: False)
        embedding = _embedding(12)

        with pytest.raises(OSError):
            store.enroll("Alice", embedding)

        assert store.permanent_count == 0  # in-memory record rolled back
        assert list((base / "embeddings").glob("*.npy")) == []  # orphan removed
