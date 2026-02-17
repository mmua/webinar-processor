import json
import pytest
from unittest.mock import patch, MagicMock

from webinar_processor.services.transcript_verifier_service import (
    validate_transcript,
    _check_repetition_loop,
    _check_long_text_no_sentence_markers,
    _check_out_of_alphabet,
    run_heuristics,
    run_llm_verification,
    generate_report,
    verify_transcript,
    TranscriptIssue,
)


# --- Helpers ---

def _seg(start, end, speaker="SPEAKER_01", text="Нормальный текст."):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


# --- validate_transcript ---

class TestValidateTranscript:
    def test_valid(self):
        segments = [_seg(0, 1), _seg(1, 2), _seg(2, 3)]
        validate_transcript(segments)  # should not raise

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_transcript([])

    def test_not_list(self):
        with pytest.raises(ValueError, match="list"):
            validate_transcript({"text": "foo"})

    def test_missing_keys(self):
        with pytest.raises(ValueError, match="missing keys"):
            validate_transcript([{"start": 0, "end": 1}])

    def test_start_greater_than_end(self):
        with pytest.raises(ValueError, match="start > end"):
            validate_transcript([_seg(5, 3)])

    def test_non_monotonic(self):
        with pytest.raises(ValueError, match="non-monotonic"):
            validate_transcript([_seg(1, 2), _seg(0, 0.5)])

    def test_non_numeric_timestamps(self):
        with pytest.raises(ValueError, match="non-numeric"):
            validate_transcript([{"start": "0", "end": 1, "speaker": "S", "text": "t"}])


# --- H1: repetition_loop ---

class TestRepetitionLoop:
    def test_ngram_repeat(self):
        text = "спасибо за внимание " * 5
        segments = [_seg(0, 10, text=text)]
        result = _check_repetition_loop(0, segments)
        assert result is not None
        assert "repeated_phrase" in result["evidence"]
        assert result["evidence"]["repeat_count"] >= 3

    def test_no_repeat(self):
        segments = [_seg(0, 5, text="Это нормальный текст без повторений и проблем.")]
        result = _check_repetition_loop(0, segments)
        assert result is None

    def test_cross_segment_overlap(self):
        text = "одинаковый текст слово слово слово"
        segments = [
            _seg(0, 5, speaker="S1", text=text),
            _seg(5, 10, speaker="S1", text=text),
        ]
        result = _check_repetition_loop(1, segments)
        assert result is not None
        assert "overlap_ratio" in result["evidence"]

    def test_cross_segment_different_speakers(self):
        text = "одинаковый текст слово слово слово"
        segments = [
            _seg(0, 5, speaker="S1", text=text),
            _seg(5, 10, speaker="S2", text=text),
        ]
        result = _check_repetition_loop(1, segments)
        # Different speakers -> no cross-segment check (only n-gram within)
        # The text itself doesn't have 3+ n-gram repeats either
        assert result is None


# --- H2: long_text_no_sentence_markers ---

class TestLongTextNoSentenceMarkers:
    def test_long_no_punctuation(self):
        text = "слово " * 80  # 480 chars, no sentence-end punctuation
        segments = [_seg(0, 60, text=text)]
        result = _check_long_text_no_sentence_markers(0, segments)
        assert result is not None
        assert result["evidence"]["text_length"] > 200

    def test_short_text(self):
        segments = [_seg(0, 5, text="Короткий текст")]
        result = _check_long_text_no_sentence_markers(0, segments)
        assert result is None

    def test_long_with_punctuation(self):
        text = "Это предложение. " * 20  # Long but well-punctuated
        segments = [_seg(0, 60, text=text)]
        result = _check_long_text_no_sentence_markers(0, segments)
        assert result is None


# --- H3: out_of_alphabet ---

class TestOutOfAlphabet:
    def test_many_unexpected(self):
        # 50% unexpected chars
        text = "Нормальный" + "♦♣♠♥★☆" * 10
        segments = [_seg(0, 5, text=text)]
        result = _check_out_of_alphabet(0, segments)
        assert result is not None
        assert result["evidence"]["symbol_ratio"] > 0.05

    def test_clean_text(self):
        segments = [_seg(0, 5, text="Привет, мир! Hello world. Тест 123 — «цитата»")]
        result = _check_out_of_alphabet(0, segments)
        assert result is None

    def test_empty_text(self):
        segments = [_seg(0, 5, text="")]
        result = _check_out_of_alphabet(0, segments)
        assert result is None


# --- run_heuristics ---

class TestRunHeuristics:
    def test_mixed_issues(self):
        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 10, text="спасибо за внимание " * 5),
            _seg(10, 15, text="Ещё нормальный текст."),
        ]
        issues = run_heuristics(segments)
        assert len(issues) >= 1
        assert all(isinstance(i, TranscriptIssue) for i in issues)
        assert issues[0].issue_id.startswith("ISS-")

    def test_clean_transcript(self):
        segments = [_seg(i, i+1, text=f"Предложение номер {i}.") for i in range(5)]
        issues = run_heuristics(segments)
        assert len(issues) == 0

    def test_multiple_heuristics_on_same_segment(self):
        # Text that triggers both repetition_loop AND long_text_no_sentence_markers
        text = "спасибо за внимание " * 40  # 800 chars, repeated, no sentence-end punct
        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 60, text=text),
            _seg(60, 65, text="Ещё нормальный текст."),
        ]
        issues = run_heuristics(segments)
        seg1_issues = [i for i in issues if 1 in i.segment_indices]
        rule_ids = {i.rule_id for i in seg1_issues}
        assert "repetition_loop" in rule_ids
        assert "long_text_no_sentence_markers" in rule_ids
        assert len(seg1_issues) >= 2


# --- generate_report ---

class TestGenerateReport:
    def test_report_contains_fenced_blocks(self):
        issues = [
            TranscriptIssue(
                issue_id="ISS-001",
                status="open",
                severity="high",
                rule_id="repetition_loop",
                segment_indices=[1],
                time_range={"start": 5.0, "end": 10.0},
                speaker_ids=["SPEAKER_01"],
                left_valid_index=0,
                right_valid_index=2,
                evidence={"repeated_phrase": "тест", "repeat_count": 5},
            ),
        ]
        report = generate_report("test.json", "test.mp4", [_seg(0, 1)] * 3, issues)
        assert "```transcript-issue" in report
        assert '"ISS-001"' in report
        assert "Transcript Verification Report" in report

    def test_empty_issues(self):
        report = generate_report("test.json", "test.mp4", [_seg(0, 1)], [])
        assert "0 flagged" in report


# --- verify_transcript (integration with mock LLM) ---

class TestVerifyTranscript:
    def test_no_llm_mode(self):
        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 10, text="спасибо за внимание " * 5),
            _seg(10, 15, text="Ещё нормальный текст."),
        ]
        report = verify_transcript(
            segments, "test.json", "test.mp4",
            model="gpt-5-mini", no_llm=True,
        )
        assert "```transcript-issue" in report
        # No LLM verdict in no-llm mode
        assert "llm_verdict" not in report

    @patch('webinar_processor.services.transcript_verifier_service.get_completion')
    def test_with_llm(self, mock_completion):
        mock_completion.return_value = json.dumps({
            "decision": "problem",
            "confidence": 0.9,
            "reason": "Test reason",
        })

        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 10, text="спасибо за внимание " * 5),
            _seg(10, 15, text="Ещё нормальный текст."),
        ]
        report = verify_transcript(
            segments, "test.json", "test.mp4",
            model="gpt-5-mini", no_llm=False,
        )
        assert "llm_verdict" in report
        assert mock_completion.called
        # LLM-confirmed problems should be auto-accepted
        assert '"accepted"' in report

    @patch('webinar_processor.services.transcript_verifier_service.get_completion')
    def test_llm_no_problem_stays_open(self, mock_completion):
        mock_completion.return_value = json.dumps({
            "decision": "no_problem",
            "confidence": 0.8,
            "reason": "Looks fine actually",
        })

        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 10, text="спасибо за внимание " * 5),
            _seg(10, 15, text="Ещё нормальный текст."),
        ]
        report = verify_transcript(
            segments, "test.json", "test.mp4",
            model="gpt-5-mini", no_llm=False,
        )
        assert "```transcript-issue" in report
        assert '"no_problem"' in report
        # LLM says no_problem -> stays open, not auto-accepted
        assert '"open"' in report

    @patch('webinar_processor.services.transcript_verifier_service.get_completion')
    def test_llm_unparseable_response_defaults_to_problem(self, mock_completion):
        mock_completion.return_value = "not valid json at all"

        segments = [
            _seg(0, 5, text="Нормальный текст."),
            _seg(5, 10, text="спасибо за внимание " * 5),
            _seg(10, 15, text="Ещё нормальный текст."),
        ]
        report = verify_transcript(
            segments, "test.json", "test.mp4",
            model="gpt-5-mini", no_llm=False,
        )
        # Should default to "problem" on parse failure
        assert '"problem"' in report
        assert "parse error" in report.lower()

    def test_invalid_transcript(self):
        with pytest.raises(ValueError):
            verify_transcript([], "test.json", "test.mp4", model="gpt-5-mini")

    def test_empty_dict_raises_list_error(self):
        with pytest.raises(ValueError, match="list"):
            validate_transcript({})
