import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from webinar_processor.commands.cmd_transcript_fix import transcript_fix


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def transcript_and_report():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01", "text": "Нормальный текст."},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "сломанный текст"},
        {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01", "text": "Ещё нормальный текст."},
    ]
    report_content = """# Transcript Verification Report

```transcript-issue
{
  "issue_id": "ISS-001",
  "status": "accepted",
  "severity": "high",
  "rule_id": "repetition_loop",
  "segment_indices": [1],
  "time_range": {"start": 5.0, "end": 10.0},
  "speaker_ids": ["SPEAKER_01"],
  "left_valid_index": 0,
  "right_valid_index": 2,
  "evidence": {"repeated_phrase": "test", "repeat_count": 5}
}
```
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False)
        transcript_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(report_content)
        report_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'\x00')
        media_path = f.name

    yield transcript_path, report_path, media_path

    for p in (transcript_path, report_path, media_path):
        if os.path.exists(p):
            os.unlink(p)


class TestTranscriptFixCLI:
    @patch('webinar_processor.services.transcript_fixer_service.get_completion')
    @patch('webinar_processor.services.transcript_fixer_service.extract_audio_slice')
    @patch('webinar_processor.services.transcript_fixer_service.RetranscriptionService')
    def test_fix_succeeds(self, mock_retrans_cls, mock_extract, mock_completion,
                          runner, transcript_and_report):
        transcript_path, report_path, media_path = transcript_and_report

        mock_retrans = MagicMock()
        mock_retrans.transcribe_whisper.return_value = "исправленный текст whisper"
        mock_retrans.transcribe_qwen3.return_value = "исправленный текст qwen3"
        mock_retrans_cls.return_value = mock_retrans

        mock_completion.return_value = json.dumps({
            "has_problem": True,
            "corrected_text": "Исправленный текст.",
            "source": "qwen3",
            "reasoning": "Test fix",
        })

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as of:
            out_path = of.name
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as rf:
            fix_report_path = rf.name

        try:
            result = runner.invoke(transcript_fix, [
                transcript_path,
                '--media', media_path,
                '--report', report_path,
                '--out', out_path,
                '--fix-report', fix_report_path,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert os.path.exists(out_path)
            assert os.path.exists(fix_report_path)

            with open(out_path, 'r') as f:
                fixed = json.load(f)
            assert fixed[1]["text"] == "Исправленный текст."
        finally:
            for p in (out_path, fix_report_path):
                if os.path.exists(p):
                    os.unlink(p)
