import os
import tempfile
import pytest
import numpy as np

from webinar_processor.services.speaker_database import SpeakerDatabase, EMBEDDING_DIM, EMBEDDING_DTYPE


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    database = SpeakerDatabase(db_path=db_path)
    yield database
    os.unlink(db_path)


def _make_embedding(seed=0):
    """Create a deterministic embedding for testing."""
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(EMBEDDING_DTYPE)


class TestAddSpeaker:
    def test_add_speaker_basic(self, db):
        emb = _make_embedding(1)
        assert db.add_speaker("spk_test0001", emb) is True

    def test_add_speaker_with_all_fields(self, db):
        emb = _make_embedding(2)
        assert db.add_speaker(
            "spk_test0002", emb,
            inferred_name="Иванов",
            confirmed_name="Проф. Иванов",
            gender="male",
            confidence_score=0.95,
            num_samples=5,
            notes="Лектор",
        ) is True
        speaker = db.get_speaker("spk_test0002")
        assert speaker is not None
        assert speaker['confirmed_name'] == "Проф. Иванов"
        assert speaker['inferred_name'] == "Иванов"
        assert speaker['gender'] == "male"
        assert speaker['confidence_score'] == 0.95
        assert speaker['num_samples'] == 5
        assert speaker['notes'] == "Лектор"

    def test_add_duplicate_fails(self, db):
        emb = _make_embedding(3)
        assert db.add_speaker("spk_dup", emb) is True
        assert db.add_speaker("spk_dup", emb) is False

    def test_add_wrong_dim_fails(self, db):
        bad_emb = np.zeros(128, dtype=EMBEDDING_DTYPE)
        assert db.add_speaker("spk_bad", bad_emb) is False

    def test_add_non_array_fails(self, db):
        assert db.add_speaker("spk_bad2", "not an array") is False


class TestGetSpeaker:
    def test_get_existing(self, db):
        emb = _make_embedding(10)
        db.add_speaker("spk_get1", emb, inferred_name="Test")
        speaker = db.get_speaker("spk_get1")
        assert speaker is not None
        assert speaker['speaker_id'] == "spk_get1"
        assert speaker['inferred_name'] == "Test"
        np.testing.assert_array_almost_equal(speaker['voice_embedding'], emb, decimal=5)

    def test_get_nonexistent(self, db):
        assert db.get_speaker("spk_nope") is None


class TestUpdateSpeaker:
    def test_update_name(self, db):
        emb = _make_embedding(20)
        db.add_speaker("spk_upd1", emb)
        assert db.update_speaker("spk_upd1", confirmed_name="Новое имя") is True
        speaker = db.get_speaker("spk_upd1")
        assert speaker['confirmed_name'] == "Новое имя"

    def test_update_notes(self, db):
        emb = _make_embedding(21)
        db.add_speaker("spk_upd2", emb)
        assert db.update_speaker("spk_upd2", notes="Заметка") is True
        speaker = db.get_speaker("spk_upd2")
        assert speaker['notes'] == "Заметка"

    def test_update_num_samples(self, db):
        emb = _make_embedding(22)
        db.add_speaker("spk_upd3", emb)
        assert db.update_speaker("spk_upd3", num_samples=10) is True
        speaker = db.get_speaker("spk_upd3")
        assert speaker['num_samples'] == 10

    def test_update_embedding(self, db):
        emb1 = _make_embedding(23)
        emb2 = _make_embedding(24)
        db.add_speaker("spk_upd4", emb1)
        assert db.update_speaker("spk_upd4", voice_embedding=emb2) is True
        speaker = db.get_speaker("spk_upd4")
        np.testing.assert_array_almost_equal(speaker['voice_embedding'], emb2, decimal=5)

    def test_update_nonexistent(self, db):
        assert db.update_speaker("spk_nope", confirmed_name="X") is False

    def test_update_no_fields(self, db):
        emb = _make_embedding(25)
        db.add_speaker("spk_upd5", emb)
        assert db.update_speaker("spk_upd5") is False


class TestFindMatchingSpeaker:
    def test_match_above_threshold(self, db):
        emb = _make_embedding(30)
        db.add_speaker("spk_match1", emb)
        # Same embedding should match perfectly
        result = db.find_matching_speaker(emb, threshold=0.7)
        assert result is not None
        assert result[0] == "spk_match1"
        assert result[1] > 0.99

    def test_no_match_below_threshold(self, db):
        emb1 = _make_embedding(31)
        emb2 = _make_embedding(32)
        db.add_speaker("spk_match2", emb1)
        # Different random embedding unlikely to match
        result = db.find_matching_speaker(emb2, threshold=0.99)
        assert result is None

    def test_no_speakers_in_db(self, db):
        emb = _make_embedding(33)
        result = db.find_matching_speaker(emb, threshold=0.5)
        assert result is None

    def test_best_match_selected(self, db):
        emb1 = _make_embedding(34)
        emb2 = _make_embedding(35)
        # Add a slightly perturbed version of emb1
        emb1_close = emb1 + np.random.RandomState(99).randn(EMBEDDING_DIM).astype(EMBEDDING_DTYPE) * 0.01
        db.add_speaker("spk_far", emb2)
        db.add_speaker("spk_close", emb1_close)
        result = db.find_matching_speaker(emb1, threshold=0.5)
        assert result is not None
        assert result[0] == "spk_close"


class TestDeleteSpeaker:
    def test_delete_existing(self, db):
        emb = _make_embedding(40)
        db.add_speaker("spk_del1", emb)
        assert db.delete_speaker("spk_del1") is True
        assert db.get_speaker("spk_del1") is None

    def test_delete_nonexistent(self, db):
        assert db.delete_speaker("spk_nope") is False

    def test_delete_cascades_appearances(self, db):
        emb = _make_embedding(41)
        db.add_speaker("spk_del2", emb)
        db.add_appearance("spk_del2", "/path/transcript.json", "/path/audio.wav", "SPEAKER_00")
        assert len(db.get_appearances("spk_del2")) == 1
        db.delete_speaker("spk_del2")
        assert len(db.get_appearances("spk_del2")) == 0


class TestMergeSpeakers:
    def test_merge_basic(self, db):
        emb1 = _make_embedding(50)
        emb2 = _make_embedding(51)
        db.add_speaker("spk_src", emb1, num_samples=3)
        db.add_speaker("spk_tgt", emb2, num_samples=2)

        assert db.merge_speakers("spk_src", "spk_tgt") is True
        assert db.get_speaker("spk_src") is None  # source deleted

        target = db.get_speaker("spk_tgt")
        assert target is not None
        assert target['num_samples'] == 5  # 3 + 2

        # Verify weighted average
        expected = (emb1 * 3 + emb2 * 2) / 5
        np.testing.assert_array_almost_equal(target['voice_embedding'], expected, decimal=5)

    def test_merge_name_inheritance(self, db):
        emb1 = _make_embedding(52)
        emb2 = _make_embedding(53)
        db.add_speaker("spk_src2", emb1, confirmed_name="Source Name")
        db.add_speaker("spk_tgt2", emb2)

        db.merge_speakers("spk_src2", "spk_tgt2")
        target = db.get_speaker("spk_tgt2")
        assert target['confirmed_name'] == "Source Name"

    def test_merge_target_name_preserved(self, db):
        emb1 = _make_embedding(54)
        emb2 = _make_embedding(55)
        db.add_speaker("spk_src3", emb1, confirmed_name="Source")
        db.add_speaker("spk_tgt3", emb2, confirmed_name="Target")

        db.merge_speakers("spk_src3", "spk_tgt3")
        target = db.get_speaker("spk_tgt3")
        assert target['confirmed_name'] == "Target"  # target keeps own name

    def test_merge_transfers_appearances(self, db):
        emb1 = _make_embedding(56)
        emb2 = _make_embedding(57)
        db.add_speaker("spk_src4", emb1)
        db.add_speaker("spk_tgt4", emb2)
        db.add_appearance("spk_src4", "/path/t1.json")
        db.add_appearance("spk_tgt4", "/path/t2.json")

        db.merge_speakers("spk_src4", "spk_tgt4")
        appearances = db.get_appearances("spk_tgt4")
        paths = [a['transcript_path'] for a in appearances]
        assert "/path/t1.json" in paths
        assert "/path/t2.json" in paths

    def test_merge_nonexistent_source(self, db):
        emb = _make_embedding(58)
        db.add_speaker("spk_tgt5", emb)
        assert db.merge_speakers("spk_nope", "spk_tgt5") is False

    def test_merge_nonexistent_target(self, db):
        emb = _make_embedding(59)
        db.add_speaker("spk_src5", emb)
        assert db.merge_speakers("spk_src5", "spk_nope") is False

    def test_merge_confidence_max(self, db):
        emb1 = _make_embedding(60)
        emb2 = _make_embedding(61)
        db.add_speaker("spk_src6", emb1, confidence_score=0.9)
        db.add_speaker("spk_tgt6", emb2, confidence_score=0.5)

        db.merge_speakers("spk_src6", "spk_tgt6")
        target = db.get_speaker("spk_tgt6")
        assert target['confidence_score'] == 0.9


class TestAppearances:
    def test_add_appearance(self, db):
        emb = _make_embedding(70)
        db.add_speaker("spk_app1", emb)
        assert db.add_appearance("spk_app1", "/path/t.json", "/path/a.wav", "SPEAKER_00") is True
        appearances = db.get_appearances("spk_app1")
        assert len(appearances) == 1
        assert appearances[0]['transcript_path'] == "/path/t.json"
        assert appearances[0]['audio_path'] == "/path/a.wav"
        assert appearances[0]['original_label'] == "SPEAKER_00"

    def test_duplicate_appearance_ignored(self, db):
        emb = _make_embedding(71)
        db.add_speaker("spk_app2", emb)
        db.add_appearance("spk_app2", "/path/t.json")
        db.add_appearance("spk_app2", "/path/t.json")  # duplicate
        appearances = db.get_appearances("spk_app2")
        assert len(appearances) == 1

    def test_multiple_appearances(self, db):
        emb = _make_embedding(72)
        db.add_speaker("spk_app3", emb)
        db.add_appearance("spk_app3", "/path/t1.json")
        db.add_appearance("spk_app3", "/path/t2.json")
        db.add_appearance("spk_app3", "/path/t3.json")
        appearances = db.get_appearances("spk_app3")
        assert len(appearances) == 3

    def test_get_appearances_empty(self, db):
        emb = _make_embedding(73)
        db.add_speaker("spk_app4", emb)
        assert db.get_appearances("spk_app4") == []


class TestGetAllSpeakers:
    def test_empty_db(self, db):
        assert db.get_all_speakers() == []

    def test_returns_all(self, db):
        for i in range(3):
            db.add_speaker(f"spk_all{i}", _make_embedding(80 + i))
        result = db.get_all_speakers()
        assert len(result) == 3

    def test_includes_appearance_count(self, db):
        emb = _make_embedding(90)
        db.add_speaker("spk_cnt1", emb)
        db.add_appearance("spk_cnt1", "/t1.json")
        db.add_appearance("spk_cnt1", "/t2.json")

        result = db.get_all_speakers()
        speaker = [s for s in result if s['speaker_id'] == 'spk_cnt1'][0]
        assert speaker['appearance_count'] == 2

    def test_zero_appearances(self, db):
        emb = _make_embedding(91)
        db.add_speaker("spk_cnt2", emb)
        result = db.get_all_speakers()
        speaker = [s for s in result if s['speaker_id'] == 'spk_cnt2'][0]
        assert speaker['appearance_count'] == 0
