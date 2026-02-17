"""Transcript fixer service: parse report, retranscribe, judge, apply fixes."""

import copy
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from webinar_processor.services.retranscription_service import RetranscriptionService
from webinar_processor.utils.completion import get_completion
from webinar_processor.utils.ffmpeg import extract_audio_slice
from webinar_processor.utils.package import get_config_path

logger = logging.getLogger(__name__)


# --- Dataclasses ---

@dataclass
class ParsedIssue:
    issue_id: str
    status: str
    segment_indices: List[int]
    time_range: Dict[str, float]
    left_valid_index: Optional[int]
    right_valid_index: Optional[int]
    severity: str = ""
    rule_id: str = ""
    speaker_ids: List[str] = None
    evidence: Dict = None


@dataclass
class RetranscriptionWindow:
    start_time: float
    end_time: float
    issues: List[ParsedIssue]
    segment_indices: List[int]
    left_valid_index: Optional[int]
    right_valid_index: Optional[int]


@dataclass
class FixResult:
    issue_id: str
    outcome: str  # "fixed", "kept_original", "failed"
    original_text: str
    corrected_text: Optional[str] = None
    source: Optional[str] = None
    reasoning: Optional[str] = None


# --- Report parsing ---

_ISSUE_BLOCK_RE = re.compile(
    r"```transcript-issue\s*\n(.*?)\n```",
    re.DOTALL,
)


def parse_report(report_text: str, include_open: bool = False) -> List[ParsedIssue]:
    blocks = _ISSUE_BLOCK_RE.findall(report_text)
    issues = []

    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            logger.warning("Failed to parse issue block: %s...", block[:80])
            continue

        required = {"issue_id", "status", "segment_indices", "time_range"}
        if not required.issubset(data.keys()):
            logger.warning("Issue block missing required fields: %s", data.get("issue_id", "unknown"))
            continue

        status = data["status"]
        if status == "accepted" or (include_open and status == "open"):
            issues.append(ParsedIssue(
                issue_id=data["issue_id"],
                status=status,
                segment_indices=data["segment_indices"],
                time_range=data["time_range"],
                left_valid_index=data.get("left_valid_index"),
                right_valid_index=data.get("right_valid_index"),
                severity=data.get("severity", ""),
                rule_id=data.get("rule_id", ""),
                speaker_ids=data.get("speaker_ids", []),
                evidence=data.get("evidence", {}),
            ))

    issues.sort(key=lambda i: i.time_range["start"])
    return issues


# --- Window merging ---

def merge_windows(issues: List[ParsedIssue], segments: list) -> List[RetranscriptionWindow]:
    if not issues:
        return []

    windows = []
    for issue in issues:
        left_idx = issue.left_valid_index
        right_idx = issue.right_valid_index

        start_time = segments[left_idx]["start"] if left_idx is not None else 0.0
        end_time = segments[right_idx]["end"] if right_idx is not None else None

        windows.append(RetranscriptionWindow(
            start_time=start_time,
            end_time=end_time,
            issues=[issue],
            segment_indices=list(issue.segment_indices),
            left_valid_index=left_idx,
            right_valid_index=right_idx,
        ))

    # Merge overlapping windows
    merged = [windows[0]]
    for win in windows[1:]:
        prev = merged[-1]
        prev_end = prev.end_time if prev.end_time is not None else float('inf')
        if win.start_time <= prev_end:
            # Merge: if either end_time is None (extends to end of file), result is None
            if prev.end_time is None or win.end_time is None:
                prev.end_time = None
            else:
                prev.end_time = max(prev.end_time, win.end_time)
            prev.issues.extend(win.issues)
            prev.segment_indices = sorted(set(prev.segment_indices + win.segment_indices))
            if win.right_valid_index is not None:
                if prev.right_valid_index is None or win.right_valid_index > prev.right_valid_index:
                    prev.right_valid_index = win.right_valid_index
        else:
            merged.append(win)

    return merged


# --- LLM judge + reconstruct ---

def _load_reconstruct_prompt() -> str:
    path = get_config_path("transcript-fix-reconstruct-prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_segment_text(segments: list, indices: List[int]) -> str:
    return " ".join(segments[i]["text"].strip() for i in indices if i < len(segments))


def _get_context_text(segments: list, anchor_idx: Optional[int], direction: str, count: int = 3) -> str:
    if anchor_idx is None:
        return "(начало/конец транскрипта)"
    lines = []
    if direction == "left":
        start = max(0, anchor_idx - count + 1)
        for i in range(start, anchor_idx + 1):
            lines.append(segments[i]["text"].strip())
    else:
        end = min(len(segments), anchor_idx + count)
        for i in range(anchor_idx, end):
            lines.append(segments[i]["text"].strip())
    return "\n".join(lines)


def _judge_and_reconstruct(
    original_text: str,
    context_left: str,
    context_right: str,
    whisper_text: str,
    qwen3_text: str,
    model: str,
    prompt_template: str,
) -> Dict:
    prompt = prompt_template.format(
        original_text=original_text,
        context_left=context_left,
        context_right=context_right,
        whisper_text=whisper_text,
        qwen3_text=qwen3_text,
    )
    response = get_completion(prompt, model=model)
    return json.loads(response.strip())


# --- Apply fixes ---

def apply_fixes(segments: list, results: List[FixResult], issue_map: Dict[str, ParsedIssue]) -> list:
    fixed = copy.deepcopy(segments)
    for result in results:
        if result.outcome != "fixed" or result.corrected_text is None:
            continue
        issue = issue_map[result.issue_id]
        indices = issue.segment_indices
        if indices:
            fixed[indices[0]]["text"] = result.corrected_text
            for idx in indices[1:]:
                fixed[idx]["text"] = ""
    return fixed


# --- Fix report ---

def _generate_fix_report(
    transcript_path: str,
    report_path: str,
    total_segments: int,
    total_issues: int,
    processed: int,
    results: List[FixResult],
) -> str:
    fixed_count = sum(1 for r in results if r.outcome == "fixed")
    kept_count = sum(1 for r in results if r.outcome == "kept_original")
    failed_count = sum(1 for r in results if r.outcome == "failed")

    lines = [
        "# Transcript Fix Report",
        "",
        f"- **Input**: {transcript_path} ({total_segments} segments)",
        f"- **Report**: {report_path} ({total_issues} issues)",
        f"- **Processed**: {processed} issues",
        f"- **Results**: {fixed_count} fixed, {kept_count} kept_original, {failed_count} failed",
        "",
        "---",
        "",
    ]

    for result in results:
        lines.append(f"## {result.issue_id}: {result.outcome}")
        lines.append("")
        original_preview = result.original_text[:120] + ("..." if len(result.original_text) > 120 else "")
        lines.append(f"Original: \"{original_preview}\"")
        if result.corrected_text:
            corrected_preview = result.corrected_text[:120] + ("..." if len(result.corrected_text) > 120 else "")
            lines.append(f"Corrected: \"{corrected_preview}\"")
        if result.source:
            lines.append(f"Source: {result.source}")
        if result.reasoning:
            lines.append(f"Reasoning: {result.reasoning}")
        lines.append("")

    return "\n".join(lines)


# --- Public orchestrator ---

def fix_transcript(
    segments: list,
    transcript_path: str,
    media_path: str,
    report_text: str,
    model: str,
    language: str = "ru",
    include_open: bool = False,
) -> tuple:
    """
    Returns (fixed_segments, fix_report_text).
    """
    issues = parse_report(report_text, include_open=include_open)

    if not issues:
        logger.info("No processable issues found in report")
        report = _generate_fix_report(transcript_path, "(report)", len(segments), 0, 0, [])
        return copy.deepcopy(segments), report

    issue_map = {i.issue_id: i for i in issues}
    windows = merge_windows(issues, segments)

    retranscription = RetranscriptionService(language=language)
    prompt_template = _load_reconstruct_prompt()
    results: List[FixResult] = []

    for window in windows:
        # Determine end time
        if window.end_time is None:
            # Use last segment end as approximation
            end_time = segments[-1]["end"]
        else:
            end_time = window.end_time

        duration = end_time - window.start_time
        if duration <= 0:
            for issue in window.issues:
                results.append(FixResult(
                    issue_id=issue.issue_id,
                    outcome="failed",
                    original_text=_get_segment_text(segments, issue.segment_indices),
                    reasoning="Invalid window duration",
                ))
            continue

        # Extract audio slice
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            extract_audio_slice(media_path, wav_path, window.start_time, duration)

            # Retranscribe
            logger.info("Retranscribing window %.1f-%.1f...", window.start_time, end_time)
            whisper_text = retranscription.transcribe_whisper(wav_path)
            qwen3_text = retranscription.transcribe_qwen3(wav_path)

            # Judge each issue in this window
            for issue in window.issues:
                original_text = _get_segment_text(segments, issue.segment_indices)
                context_left = _get_context_text(segments, issue.left_valid_index, "left")
                context_right = _get_context_text(segments, issue.right_valid_index, "right")

                try:
                    verdict = _judge_and_reconstruct(
                        original_text=original_text,
                        context_left=context_left,
                        context_right=context_right,
                        whisper_text=whisper_text,
                        qwen3_text=qwen3_text,
                        model=model,
                        prompt_template=prompt_template,
                    )

                    if verdict.get("has_problem"):
                        results.append(FixResult(
                            issue_id=issue.issue_id,
                            outcome="fixed",
                            original_text=original_text,
                            corrected_text=verdict.get("corrected_text", ""),
                            source=verdict.get("source", ""),
                            reasoning=verdict.get("reasoning", ""),
                        ))
                    else:
                        results.append(FixResult(
                            issue_id=issue.issue_id,
                            outcome="kept_original",
                            original_text=original_text,
                            reasoning=verdict.get("reasoning", ""),
                        ))
                except Exception as e:
                    logger.error("LLM judge failed for %s: %s", issue.issue_id, e)
                    raise
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    fixed_segments = apply_fixes(segments, results, issue_map)
    fix_report = _generate_fix_report(
        transcript_path, "(report)", len(segments), len(issues), len(issues), results,
    )

    return fixed_segments, fix_report
