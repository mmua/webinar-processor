import json
import pytest
from unittest.mock import patch

from webinar_processor.services.transcript_fixer_service import (
    parse_report,
    merge_windows,
    apply_fixes,
    _generate_fix_report,
    ParsedIssue,
    FixResult,
)


# --- Helpers ---

def _seg(start, end, speaker="SPEAKER_01", text="Нормальный текст."):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def _make_report(*issue_dicts):
    """Build a minimal report with transcript-issue blocks."""
    blocks = []
    for d in issue_dicts:
        blocks.append(f"```transcript-issue\n{json.dumps(d, ensure_ascii=False)}\n```")
    return "\n\n".join(blocks)


def _issue_dict(issue_id="ISS-001", status="accepted", segment_indices=None,
                start=5.0, end=10.0, left=0, right=2):
    return {
        "issue_id": issue_id,
        "status": status,
        "segment_indices": segment_indices or [1],
        "time_range": {"start": start, "end": end},
        "left_valid_index": left,
        "right_valid_index": right,
        "severity": "high",
        "rule_id": "repetition_loop",
    }


# --- parse_report ---

class TestParseReport:
    def test_accepted_only(self):
        report = _make_report(
            _issue_dict("ISS-001", status="accepted"),
            _issue_dict("ISS-002", status="open"),
            _issue_dict("ISS-003", status="ignored"),
        )
        issues = parse_report(report, include_open=False)
        assert len(issues) == 1
        assert issues[0].issue_id == "ISS-001"

    def test_include_open(self):
        report = _make_report(
            _issue_dict("ISS-001", status="accepted"),
            _issue_dict("ISS-002", status="open"),
            _issue_dict("ISS-003", status="ignored"),
        )
        issues = parse_report(report, include_open=True)
        assert len(issues) == 2
        ids = {i.issue_id for i in issues}
        assert ids == {"ISS-001", "ISS-002"}

    def test_empty_report(self):
        issues = parse_report("# Empty report\nNo issues found.", include_open=True)
        assert len(issues) == 0

    def test_malformed_json(self):
        report = "```transcript-issue\n{bad json\n```"
        issues = parse_report(report)
        assert len(issues) == 0

    def test_missing_required_fields(self):
        report = _make_report({"issue_id": "ISS-001", "status": "accepted"})
        issues = parse_report(report)
        assert len(issues) == 0

    def test_sorted_by_start_time(self):
        report = _make_report(
            _issue_dict("ISS-002", status="accepted", start=20.0, end=25.0),
            _issue_dict("ISS-001", status="accepted", start=5.0, end=10.0),
        )
        issues = parse_report(report)
        assert issues[0].issue_id == "ISS-001"
        assert issues[1].issue_id == "ISS-002"


# --- merge_windows ---

class TestMergeWindows:
    def test_non_overlapping(self):
        segments = [_seg(i, i+1) for i in range(10)]
        issues = [
            ParsedIssue("ISS-001", "accepted", [2], {"start": 2, "end": 3}, 1, 3),
            ParsedIssue("ISS-002", "accepted", [7], {"start": 7, "end": 8}, 6, 8),
        ]
        windows = merge_windows(issues, segments)
        assert len(windows) == 2
        assert windows[0].issues[0].issue_id == "ISS-001"
        assert windows[1].issues[0].issue_id == "ISS-002"

    def test_overlapping_merges(self):
        segments = [_seg(i, i+1) for i in range(10)]
        issues = [
            ParsedIssue("ISS-001", "accepted", [2], {"start": 2, "end": 3}, 1, 4),
            ParsedIssue("ISS-002", "accepted", [3], {"start": 3, "end": 4}, 2, 5),
        ]
        windows = merge_windows(issues, segments)
        assert len(windows) == 1
        assert len(windows[0].issues) == 2

    def test_empty(self):
        windows = merge_windows([], [])
        assert len(windows) == 0

    def test_null_left_index(self):
        segments = [_seg(i, i+1) for i in range(5)]
        issues = [
            ParsedIssue("ISS-001", "accepted", [0], {"start": 0, "end": 1}, None, 1),
        ]
        windows = merge_windows(issues, segments)
        assert len(windows) == 1
        assert windows[0].start_time == 0.0

    def test_null_right_index(self):
        """Last segment garbled: end_time should be None (extends to end of file)."""
        segments = [_seg(i, i+1) for i in range(5)]
        issues = [
            ParsedIssue("ISS-001", "accepted", [4], {"start": 4, "end": 5}, 3, None),
        ]
        windows = merge_windows(issues, segments)
        assert len(windows) == 1
        assert windows[0].end_time is None

    def test_merge_with_null_right_preserves_none(self):
        """When merging and one window has None end_time, result should be None."""
        segments = [_seg(i, i+1) for i in range(5)]
        issues = [
            ParsedIssue("ISS-001", "accepted", [3], {"start": 3, "end": 4}, 2, 4),
            ParsedIssue("ISS-002", "accepted", [4], {"start": 4, "end": 5}, 3, None),
        ]
        windows = merge_windows(issues, segments)
        assert len(windows) == 1
        assert windows[0].end_time is None


# --- apply_fixes ---

class TestApplyFixes:
    def test_fixed_text_goes_to_first_segment(self):
        segments = [
            _seg(0, 1, text="ok"),
            _seg(1, 2, text="broken1"),
            _seg(2, 3, text="broken2"),
            _seg(3, 4, text="ok2"),
        ]
        issue = ParsedIssue("ISS-001", "accepted", [1, 2], {"start": 1, "end": 3}, 0, 3)
        results = [FixResult("ISS-001", "fixed", "broken1 broken2", corrected_text="corrected")]
        issue_map = {"ISS-001": issue}

        fixed = apply_fixes(segments, results, issue_map)
        assert fixed[1]["text"] == "corrected"
        assert fixed[2]["text"] == ""
        # Originals untouched
        assert fixed[0]["text"] == "ok"
        assert fixed[3]["text"] == "ok2"

    def test_kept_original_unchanged(self):
        segments = [_seg(0, 1, text="original")]
        issue = ParsedIssue("ISS-001", "accepted", [0], {"start": 0, "end": 1}, None, None)
        results = [FixResult("ISS-001", "kept_original", "original")]
        issue_map = {"ISS-001": issue}

        fixed = apply_fixes(segments, results, issue_map)
        assert fixed[0]["text"] == "original"

    def test_failed_outcome_unchanged(self):
        segments = [_seg(0, 1, text="original")]
        issue = ParsedIssue("ISS-001", "accepted", [0], {"start": 0, "end": 1}, None, None)
        results = [FixResult("ISS-001", "failed", "original", reasoning="error")]
        issue_map = {"ISS-001": issue}

        fixed = apply_fixes(segments, results, issue_map)
        assert fixed[0]["text"] == "original"

    def test_does_not_mutate_input(self):
        segments = [_seg(0, 1, text="original")]
        issue = ParsedIssue("ISS-001", "accepted", [0], {"start": 0, "end": 1}, None, None)
        results = [FixResult("ISS-001", "fixed", "original", corrected_text="new")]
        issue_map = {"ISS-001": issue}

        fixed = apply_fixes(segments, results, issue_map)
        assert fixed[0]["text"] == "new"
        assert segments[0]["text"] == "original"


# --- _generate_fix_report ---

class TestGenerateFixReport:
    def test_report_format(self):
        results = [
            FixResult("ISS-001", "fixed", "broken text", corrected_text="good text", source="qwen3"),
            FixResult("ISS-002", "kept_original", "fine text", reasoning="No issue"),
            FixResult("ISS-003", "failed", "bad text", reasoning="LLM error"),
        ]
        report = _generate_fix_report("t.json", "report.md", 100, 3, 3, results)
        assert "# Transcript Fix Report" in report
        assert "1 fixed" in report
        assert "1 kept_original" in report
        assert "1 failed" in report
        assert "ISS-001: fixed" in report
        assert "ISS-002: kept_original" in report

    def test_empty_results(self):
        report = _generate_fix_report("t.json", "report.md", 100, 0, 0, [])
        assert "0 fixed" in report
        assert "0 failed" in report
