import json
import os
import tempfile
import pytest
from unittest.mock import patch
from click.testing import CliRunner

from webinar_processor.commands.cmd_transcript_verify import transcript_verify


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_transcript():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01", "text": "Нормальный текст."},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "спасибо за внимание " * 5},
        {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01", "text": "Ещё нормальный текст."},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def media_file():
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'\x00')  # Dummy file
        path = f.name
    yield path
    os.unlink(path)


class TestTranscriptVerifyCLI:
    def test_no_llm_succeeds(self, runner, valid_transcript, media_file):
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as rf:
            report_path = rf.name

        try:
            result = runner.invoke(transcript_verify, [
                valid_transcript,
                '--media', media_file,
                '--no-llm',
                '--report', report_path,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert os.path.exists(report_path)
            with open(report_path, 'r') as f:
                content = f.read()
            assert "Transcript Verification Report" in content
            assert "```transcript-issue" in content
        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)

    def test_invalid_transcript(self, runner, media_file):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json{{{")
            bad_path = f.name

        try:
            result = runner.invoke(transcript_verify, [
                bad_path,
                '--media', media_file,
                '--no-llm',
            ])
            assert result.exit_code != 0
        finally:
            os.unlink(bad_path)

    def test_empty_transcript(self, runner, media_file):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump([], f)
            empty_path = f.name

        try:
            result = runner.invoke(transcript_verify, [
                empty_path,
                '--media', media_file,
                '--no-llm',
            ])
            assert result.exit_code != 0
        finally:
            os.unlink(empty_path)
