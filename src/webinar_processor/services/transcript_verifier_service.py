"""Transcript verification service: heuristic detection + optional LLM confirmation."""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional

from webinar_processor.utils.completion import get_completion
from webinar_processor.utils.package import get_config_path

logger = logging.getLogger(__name__)

# --- Dataclasses ---

@dataclass
class LLMVerdict:
    decision: str  # "problem" or "no_problem"
    confidence: float
    reason: str


@dataclass
class TranscriptIssue:
    issue_id: str
    status: str  # "open"
    severity: str  # "low", "medium", "high"
    rule_id: str
    segment_indices: List[int]
    time_range: Dict[str, float]
    speaker_ids: List[str]
    left_valid_index: Optional[int]
    right_valid_index: Optional[int]
    evidence: Dict
    llm_verdict: Optional[LLMVerdict] = None


# --- Validation ---

def validate_transcript(segments: list) -> None:
    if not isinstance(segments, list):
        raise ValueError("Transcript must be a list of segments")
    if not segments:
        raise ValueError("Transcript is empty")

    required_keys = {"start", "end", "speaker", "text"}
    prev_start = -1.0
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"Segment {i} is not a dict")
        missing = required_keys - set(seg.keys())
        if missing:
            raise ValueError(f"Segment {i} missing keys: {missing}")
        if not isinstance(seg["start"], (int, float)) or not isinstance(seg["end"], (int, float)):
            raise ValueError(f"Segment {i} has non-numeric timestamps")
        if seg["start"] > seg["end"]:
            raise ValueError(f"Segment {i} has start > end ({seg['start']} > {seg['end']})")
        if seg["start"] < prev_start:
            raise ValueError(f"Segment {i} has non-monotonic start ({seg['start']} < {prev_start})")
        prev_start = seg["start"]


# --- Heuristics ---

def _check_repetition_loop(seg_index: int, segments: list) -> Optional[Dict]:
    """H1: Detect repeating phrase patterns."""
    seg = segments[seg_index]
    text = seg["text"].strip()
    words = text.split()

    # Check n-gram repetition within segment
    for n in range(3, 6):
        if len(words) < n:
            continue
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        for phrase, count in counts.most_common(1):
            if count >= 3:
                return {
                    "severity": "high" if count >= 5 else "medium",
                    "evidence": {"repeated_phrase": phrase, "repeat_count": count},
                }

    # Check cross-segment overlap with previous same-speaker segment
    if seg_index > 0:
        prev = segments[seg_index - 1]
        if prev["speaker"] == seg["speaker"]:
            prev_text = prev["text"].strip().lower()
            cur_text = text.lower()
            if prev_text and cur_text:
                # Jaccard-like word overlap
                prev_words = set(prev_text.split())
                cur_words = set(cur_text.split())
                if prev_words and cur_words:
                    overlap = len(prev_words & cur_words) / max(len(prev_words | cur_words), 1)
                    if overlap > 0.70:
                        return {
                            "severity": "high" if overlap > 0.85 else "medium",
                            "evidence": {"overlap_ratio": round(overlap, 3)},
                        }
    return None


def _check_long_text_no_sentence_markers(seg_index: int, segments: list) -> Optional[Dict]:
    """H2: Long text with weak punctuation/capitalization."""
    text = segments[seg_index]["text"].strip()
    if len(text) <= 200:
        return None

    sentence_ends = sum(1 for c in text if c in ".!?")
    punct_ratio = sentence_ends / len(text) if text else 0

    # Count uppercase after whitespace (sentence starts)
    upper_after_ws = 0
    ws_count = 0
    for i, c in enumerate(text):
        if c in " \t\n":
            ws_count += 1
            if i + 1 < len(text) and text[i + 1].isupper():
                upper_after_ws += 1
    cap_ratio = upper_after_ws / max(ws_count, 1)

    # Thresholds: normal text has ~2-4% sentence-end punctuation and ~5-15% capitalization after ws
    if punct_ratio < 0.005 and cap_ratio < 0.03:
        severity = "high" if punct_ratio == 0 else "medium"
        return {
            "severity": severity,
            "evidence": {
                "text_length": len(text),
                "punctuation_ratio": round(punct_ratio, 4),
                "capitalization_ratio": round(cap_ratio, 4),
            },
        }
    return None


_ALLOWED_CHARS_RE = re.compile(
    r'[а-яА-ЯёЁa-zA-Z0-9\s'
    r'.,!?;:\-—–()\[\]"\'«»…]'
)


def _check_out_of_alphabet(seg_index: int, segments: list) -> Optional[Dict]:
    """H3: High ratio of unexpected symbols."""
    text = segments[seg_index]["text"].strip()
    if not text:
        return None

    unexpected = [c for c in text if not _ALLOWED_CHARS_RE.match(c)]
    ratio = len(unexpected) / len(text)

    if ratio > 0.05:
        sample = "".join(set(unexpected))[:20]
        severity = "high" if ratio > 0.15 else ("medium" if ratio > 0.10 else "low")
        return {
            "severity": severity,
            "evidence": {
                "symbol_ratio": round(ratio, 4),
                "unexpected_chars_sample": sample,
            },
        }
    return None


# --- Heuristic orchestrator ---

_HEURISTICS = [
    ("repetition_loop", _check_repetition_loop),
    ("long_text_no_sentence_markers", _check_long_text_no_sentence_markers),
    ("out_of_alphabet_symbols", _check_out_of_alphabet),
]


def run_heuristics(segments: list) -> List[TranscriptIssue]:
    issues: List[TranscriptIssue] = []
    issue_counter = 0

    for seg_idx in range(len(segments)):
        for rule_id, check_fn in _HEURISTICS:
            result = check_fn(seg_idx, segments)
            if result is not None:
                issue_counter += 1
                seg = segments[seg_idx]

                left_valid = seg_idx - 1 if seg_idx > 0 else None
                right_valid = seg_idx + 1 if seg_idx < len(segments) - 1 else None

                issues.append(TranscriptIssue(
                    issue_id=f"ISS-{issue_counter:03d}",
                    status="open",
                    severity=result["severity"],
                    rule_id=rule_id,
                    segment_indices=[seg_idx],
                    time_range={"start": seg["start"], "end": seg["end"]},
                    speaker_ids=[seg["speaker"]],
                    left_valid_index=left_valid,
                    right_valid_index=right_valid,
                    evidence=result["evidence"],
                ))
    return issues


# --- LLM verification ---

def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _get_context(segments: list, seg_idx: int, direction: str, count: int = 3) -> str:
    lines = []
    if direction == "left":
        start = max(0, seg_idx - count)
        for i in range(start, seg_idx):
            lines.append(segments[i]["text"].strip())
    else:
        end = min(len(segments), seg_idx + count + 1)
        for i in range(seg_idx + 1, end):
            lines.append(segments[i]["text"].strip())
    return "\n".join(lines)


def run_llm_verification(segments: list, candidates: List[TranscriptIssue], model: str) -> List[TranscriptIssue]:
    prompt_template = _load_verify_prompt()

    for issue in candidates:
        seg_idx = issue.segment_indices[0]
        flagged_text = segments[seg_idx]["text"].strip()
        context_left = _get_context(segments, seg_idx, "left")
        context_right = _get_context(segments, seg_idx, "right")

        prompt = prompt_template.format(
            context_left=context_left or "(начало транскрипта)",
            flagged_text=flagged_text,
            context_right=context_right or "(конец транскрипта)",
            rule_id=issue.rule_id,
            evidence=json.dumps(issue.evidence, ensure_ascii=False),
        )

        try:
            response = get_completion(prompt, model=model)
            verdict_data = json.loads(response.strip())
            issue.llm_verdict = LLMVerdict(
                decision=verdict_data.get("decision", "no_problem"),
                confidence=float(verdict_data.get("confidence", 0.0)),
                reason=verdict_data.get("reason", ""),
            )
            logger.info("LLM verdict for %s: %s (%.2f)",
                        issue.issue_id, issue.llm_verdict.decision, issue.llm_verdict.confidence)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse LLM verdict for %s: %s", issue.issue_id, e)
            issue.llm_verdict = LLMVerdict(
                decision="problem",
                confidence=0.0,
                reason=f"LLM response parse error: {e}",
            )

    return candidates


def _load_verify_prompt() -> str:
    path = get_config_path("transcript-verify-judge-prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Report generation ---

def generate_report(
    transcript_path: str,
    media_path: str,
    segments: list,
    issues: List[TranscriptIssue],
) -> str:
    lines = [
        "# Transcript Verification Report",
        "",
        f"- **Transcript**: {transcript_path}",
        f"- **Media**: {media_path}",
        f"- **Date**: {date.today().isoformat()}",
        f"- **Segments**: {len(segments)} total, {len(issues)} flagged",
        "",
        "---",
        "",
    ]

    for issue in issues:
        time_str = f"{_format_time(issue.time_range['start'])}-{_format_time(issue.time_range['end'])}"
        lines.append(f"## {issue.issue_id}: {issue.rule_id.replace('_', ' ').title()} at {time_str}")
        lines.append("")

        # Human-readable summary
        if issue.rule_id == "repetition_loop":
            if "repeated_phrase" in issue.evidence:
                lines.append(
                    f"Segment {issue.segment_indices[0]} contains repeated phrase "
                    f"\"{issue.evidence['repeated_phrase']}\" ({issue.evidence['repeat_count']} times)."
                )
            elif "overlap_ratio" in issue.evidence:
                lines.append(
                    f"Segments {issue.segment_indices} have {issue.evidence['overlap_ratio']*100:.0f}% text overlap."
                )
        elif issue.rule_id == "long_text_no_sentence_markers":
            lines.append(
                f"Segment {issue.segment_indices[0]}: {issue.evidence['text_length']} chars with "
                f"punctuation ratio {issue.evidence['punctuation_ratio']:.4f}."
            )
        elif issue.rule_id == "out_of_alphabet_symbols":
            lines.append(
                f"Segment {issue.segment_indices[0]}: {issue.evidence['symbol_ratio']*100:.1f}% "
                f"unexpected symbols."
            )

        if issue.llm_verdict:
            if issue.llm_verdict.decision == "problem":
                lines.append(f"LLM confirms this is likely a problem ({issue.llm_verdict.confidence:.2f}).")
            else:
                lines.append(f"LLM says no problem ({issue.llm_verdict.confidence:.2f}): {issue.llm_verdict.reason}")
        lines.append("")

        lines.append(f"**Status**: {issue.status}")
        lines.append("**Action**: Review and set to `accepted` or `ignored`.")
        lines.append("")

        # Fenced JSON block
        issue_dict = {
            "issue_id": issue.issue_id,
            "status": issue.status,
            "severity": issue.severity,
            "rule_id": issue.rule_id,
            "segment_indices": issue.segment_indices,
            "time_range": issue.time_range,
            "speaker_ids": issue.speaker_ids,
            "left_valid_index": issue.left_valid_index,
            "right_valid_index": issue.right_valid_index,
            "evidence": issue.evidence,
        }
        if issue.llm_verdict:
            issue_dict["llm_verdict"] = asdict(issue.llm_verdict)

        lines.append("```transcript-issue")
        lines.append(json.dumps(issue_dict, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# --- Public orchestrator ---

def verify_transcript(
    segments: list,
    transcript_path: str,
    media_path: str,
    model: str,
    no_llm: bool = False,
) -> str:
    validate_transcript(segments)

    logger.info("Running heuristics on %d segments...", len(segments))
    issues = run_heuristics(segments)
    logger.info("Found %d heuristic issues", len(issues))

    if issues and not no_llm:
        logger.info("Running LLM verification on %d candidates...", len(issues))
        issues = run_llm_verification(segments, issues, model)

    report = generate_report(transcript_path, media_path, segments, issues)
    return report
