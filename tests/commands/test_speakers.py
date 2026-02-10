import json
import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from webinar_processor.commands.cmd_speakers import speakers, _generate_speaker_id
from webinar_processor.services.speaker_database import SpeakerDatabase, EMBEDDING_DIM, EMBEDDING_DTYPE

# The CLI commands use lazy imports inside function bodies:
#   from webinar_processor.services.speaker_database import SpeakerDatabase
# To patch these, we patch the class at its source module.
DB_PATCH = 'webinar_processor.services.speaker_database.SpeakerDatabase'
VOICE_PATCH = 'webinar_processor.services.voice_embedding_service.VoiceEmbeddingService'
LLM_PATCH = 'webinar_processor.commands.cmd_speakers.LLMClient'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    database = SpeakerDatabase(db_path=db_path)
    yield database
    os.unlink(db_path)


def _make_embedding(seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(EMBEDDING_DTYPE)


class TestGenerateSpeakerId:
    def test_format(self):
        sid = _generate_speaker_id()
        assert sid.startswith("spk_")
        assert len(sid) == 12  # "spk_" + 8 hex chars

    def test_unique(self):
        ids = {_generate_speaker_id() for _ in range(100)}
        assert len(ids) == 100


class TestListCommand:
    def test_list_empty(self, runner, db):
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['list'])
        assert result.exit_code == 0
        assert "No speakers in database" in result.output

    def test_list_with_speakers(self, runner, db):
        emb = _make_embedding(1)
        db.add_speaker("spk_list0001", emb, confirmed_name="Иванов", gender="male")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['list'])
        assert result.exit_code == 0
        assert "spk_list0001" in result.output
        assert "Иванов" in result.output

    def test_list_json(self, runner, db):
        emb = _make_embedding(2)
        db.add_speaker("spk_json0001", emb, confirmed_name="Test")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['list', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]['speaker_id'] == "spk_json0001"
        assert data[0]['confirmed_name'] == "Test"

    def test_list_name_priority(self, runner, db):
        emb1 = _make_embedding(3)
        emb2 = _make_embedding(4)
        db.add_speaker("spk_namec", emb1, confirmed_name="Confirmed", inferred_name="Inferred")
        db.add_speaker("spk_namei", emb2, inferred_name="OnlyInferred")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['list'])
        assert "Confirmed (C)" in result.output
        assert "OnlyInferred (I)" in result.output


class TestInfoCommand:
    def test_info_existing(self, runner, db):
        emb = _make_embedding(10)
        db.add_speaker("spk_info0001", emb, confirmed_name="Проф. Тест", notes="Заметка")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['info', 'spk_info0001'])
        assert result.exit_code == 0
        assert "spk_info0001" in result.output
        assert "Проф. Тест" in result.output
        assert "Заметка" in result.output
        assert "No recorded appearances" in result.output

    def test_info_not_found(self, runner, db):
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['info', 'spk_nope'])
        assert "not found" in result.output

    def test_info_with_appearances(self, runner, db):
        emb = _make_embedding(11)
        db.add_speaker("spk_info0002", emb)
        db.add_appearance("spk_info0002", "/path/t.json", "/path/a.wav", "SPEAKER_00")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['info', 'spk_info0002'])
        assert "Appearances (1)" in result.output
        assert "/path/t.json" in result.output
        assert "SPEAKER_00" in result.output


class TestUpdateCommand:
    def test_update_name(self, runner, db):
        emb = _make_embedding(20)
        db.add_speaker("spk_upd0001", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['update', 'spk_upd0001', '--name', 'Новое имя'])
        assert result.exit_code == 0
        assert "Successfully updated" in result.output
        speaker = db.get_speaker("spk_upd0001")
        assert speaker['confirmed_name'] == "Новое имя"

    def test_update_notes(self, runner, db):
        emb = _make_embedding(21)
        db.add_speaker("spk_upd0002", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['update', 'spk_upd0002', '--notes', 'Test note'])
        assert "Successfully updated" in result.output
        speaker = db.get_speaker("spk_upd0002")
        assert speaker['notes'] == "Test note"

    def test_update_no_args(self, runner, db):
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['update', 'spk_x'])
        assert "Please provide at least one update" in result.output

    def test_update_nonexistent(self, runner, db):
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['update', 'spk_nope', '--name', 'X'])
        assert "Failed to update" in result.output


class TestDeleteCommand:
    def test_delete_with_yes(self, runner, db):
        emb = _make_embedding(30)
        db.add_speaker("spk_del0001", emb, confirmed_name="ToDelete")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['delete', 'spk_del0001', '--yes'])
        assert result.exit_code == 0
        assert "Deleted speaker" in result.output
        assert db.get_speaker("spk_del0001") is None

    def test_delete_with_confirmation(self, runner, db):
        emb = _make_embedding(31)
        db.add_speaker("spk_del0002", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['delete', 'spk_del0002'], input='y\n')
        assert "Deleted speaker" in result.output

    def test_delete_cancelled(self, runner, db):
        emb = _make_embedding(32)
        db.add_speaker("spk_del0003", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['delete', 'spk_del0003'], input='n\n')
        assert db.get_speaker("spk_del0003") is not None

    def test_delete_not_found(self, runner, db):
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['delete', 'spk_nope', '--yes'])
        assert "not found" in result.output


class TestMergeCommand:
    def test_merge_basic(self, runner, db):
        emb1 = _make_embedding(40)
        emb2 = _make_embedding(41)
        db.add_speaker("spk_msrc", emb1, confirmed_name="Source")
        db.add_speaker("spk_mtgt", emb2, confirmed_name="Target")
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['merge', 'spk_msrc', 'spk_mtgt'])
        assert result.exit_code == 0
        assert "Successfully merged" in result.output
        assert db.get_speaker("spk_msrc") is None
        assert db.get_speaker("spk_mtgt") is not None

    def test_merge_source_not_found(self, runner, db):
        emb = _make_embedding(42)
        db.add_speaker("spk_mtgt2", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['merge', 'spk_nope', 'spk_mtgt2'])
        assert "not found" in result.output

    def test_merge_target_not_found(self, runner, db):
        emb = _make_embedding(43)
        db.add_speaker("spk_msrc2", emb)
        with patch(DB_PATCH, return_value=db):
            result = runner.invoke(speakers, ['merge', 'spk_msrc2', 'spk_nope'])
        assert "not found" in result.output


class TestEnrollCommand:
    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_enroll_success(self, mock_db_cls, mock_voice_cls, runner):
        mock_voice = MagicMock()
        mock_voice.extract_single_speaker_embedding.return_value = _make_embedding(50)
        mock_voice_cls.return_value = mock_voice

        mock_db = MagicMock()
        mock_db.find_matching_speaker.return_value = None
        mock_db.add_speaker.return_value = True
        mock_db_cls.return_value = mock_db

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name
        try:
            result = runner.invoke(speakers, ['enroll', '--name', 'Test', '--audio', audio_path])
            assert result.exit_code == 0
            assert "Enrolled speaker: Test" in result.output
            mock_db.add_speaker.assert_called_once()
            call_kwargs = mock_db.add_speaker.call_args
            assert call_kwargs[1]['confirmed_name'] == 'Test'
            assert call_kwargs[1]['confidence_score'] == 1.0
        finally:
            os.unlink(audio_path)

    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_enroll_extraction_fails(self, mock_db_cls, mock_voice_cls, runner):
        mock_voice = MagicMock()
        mock_voice.extract_single_speaker_embedding.return_value = None
        mock_voice_cls.return_value = mock_voice

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name
        try:
            result = runner.invoke(speakers, ['enroll', '--name', 'Test', '--audio', audio_path])
            assert "Failed to extract" in result.output
        finally:
            os.unlink(audio_path)

    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_enroll_similar_exists_abort(self, mock_db_cls, mock_voice_cls, runner):
        mock_voice = MagicMock()
        mock_voice.extract_single_speaker_embedding.return_value = _make_embedding(51)
        mock_voice_cls.return_value = mock_voice

        mock_db = MagicMock()
        mock_db.find_matching_speaker.return_value = ("spk_existing", 0.85)
        mock_db.get_speaker.return_value = {'confirmed_name': 'Existing', 'inferred_name': None}
        mock_db_cls.return_value = mock_db

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name
        try:
            result = runner.invoke(speakers, ['enroll', '--name', 'Test', '--audio', audio_path], input='n\n')
            assert "Similar speaker found" in result.output
            mock_db.add_speaker.assert_not_called()
        finally:
            os.unlink(audio_path)


class TestRelabelCommand:
    @patch(LLM_PATCH)
    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_relabel_creates_new_speakers(self, mock_db_cls, mock_voice_cls, mock_llm_cls, runner):
        transcript = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00", "text": "Hello"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01", "text": "World"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(transcript, f)
            transcript_path = f.name

        mock_voice = MagicMock()
        emb0 = _make_embedding(60)
        emb1 = _make_embedding(61)
        mock_voice.process_audio_file.return_value = {
            "SPEAKER_00": emb0,
            "SPEAKER_01": emb1,
        }
        mock_voice_cls.return_value = mock_voice

        mock_db = MagicMock()
        mock_db.get_all_speakers.return_value = []
        mock_db.find_matching_speaker.return_value = None
        mock_db.add_speaker.return_value = True
        mock_db.get_speaker.return_value = None
        mock_db.add_appearance.return_value = True
        mock_db_cls.return_value = mock_db

        mock_llm = MagicMock()
        mock_llm.extract_speaker_name.return_value = None
        mock_llm_cls.return_value = mock_llm

        try:
            result = runner.invoke(speakers, ['relabel', transcript_path, '/fake/audio.wav'])
            assert result.exit_code == 0
            assert mock_db.add_speaker.call_count == 2
            for call in mock_db.add_speaker.call_args_list:
                sid = call[1]['speaker_id']
                assert sid.startswith("spk_")
                assert len(sid) == 12
        finally:
            os.unlink(transcript_path)
            output = transcript_path + '.relabeled'
            if os.path.exists(output):
                os.unlink(output)

    @patch(LLM_PATCH)
    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_relabel_output_option(self, mock_db_cls, mock_voice_cls, mock_llm_cls, runner):
        transcript = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00", "text": "Hello"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(transcript, f)
            transcript_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name

        mock_voice = MagicMock()
        emb = _make_embedding(70)
        mock_voice.process_audio_file.return_value = {"SPEAKER_00": emb}
        mock_voice_cls.return_value = mock_voice

        mock_db = MagicMock()
        mock_db.get_all_speakers.return_value = []
        mock_db.find_matching_speaker.return_value = None
        mock_db.add_speaker.return_value = True
        mock_db.get_speaker.return_value = {'confirmed_name': 'Named', 'inferred_name': None}
        mock_db.add_appearance.return_value = True
        mock_db_cls.return_value = mock_db

        mock_llm = MagicMock()
        mock_llm.extract_speaker_name.return_value = None
        mock_llm_cls.return_value = mock_llm

        try:
            result = runner.invoke(speakers, ['relabel', transcript_path, '/fake/audio.wav', '-o', output_path])
            assert result.exit_code == 0
            assert f"Updated transcript saved to {output_path}" in result.output
            with open(output_path) as f:
                data = json.load(f)
            assert data[0]['speaker'] == 'Named'
        finally:
            os.unlink(transcript_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch(LLM_PATCH)
    @patch(VOICE_PATCH)
    @patch(DB_PATCH)
    def test_relabel_no_embeddings(self, mock_db_cls, mock_voice_cls, mock_llm_cls, runner):
        transcript = [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hi"}]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(transcript, f)
            transcript_path = f.name

        mock_voice = MagicMock()
        mock_voice.process_audio_file.return_value = {}
        mock_voice_cls.return_value = mock_voice

        mock_db = MagicMock()
        mock_db.get_all_speakers.return_value = []
        mock_db_cls.return_value = mock_db

        mock_llm_cls.return_value = MagicMock()

        try:
            result = runner.invoke(speakers, ['relabel', transcript_path, '/fake/audio.wav'])
            assert "No valid voice embeddings" in result.output
        finally:
            os.unlink(transcript_path)
