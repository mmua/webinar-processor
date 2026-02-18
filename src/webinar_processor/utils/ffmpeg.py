import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_wav_filename(input_path: str, output_dir: str) -> str:
    """
    Generate a WAV filename from an input audio/video file path.

    Args:
        input_path: Path to the input audio/video file
        output_dir: Directory where the WAV file should be placed

    Returns:
        Path to the output WAV file
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    wav_filename = f"{base_name}.wav"
    return os.path.join(output_dir, wav_filename)


def extract_audio_slice(
    input_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = 16000,
):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"extract_audio_slice failed: {e.stderr.decode() if e.stderr else e}"
        )
        raise


def convert_mp4_to_wav(input_path: str, output_path: str, sample_rate: int = 16000):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"convert_mp4_to_wav failed: {e.stderr.decode() if e.stderr else e}"
        )
        raise


def normalize_audio_file(input_path: str, output_path: str, sample_rate: int = 16000):
    """Normalize loudness and dynamics for more stable ASR quality.

    This is intentionally conservative so timing stays stable for diarization.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11,acompressor=threshold=-21dB:ratio=3:attack=5:release=50",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"normalize_audio_file failed: {e.stderr.decode() if e.stderr else e}"
        )
        raise


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds as float

    Raises:
        subprocess.CalledProcessError: If ffprobe fails
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def detect_silence_intervals(input_path: str) -> Optional[List[float]]:
    """Detect silence intervals >= 7 seconds using FFmpeg silencedetect filter.

    Short pauses (< 7s) are ignored and kept in the video.

    Args:
        input_path: Path to video file

    Returns:
        Flat list of silence boundaries [start1, end1, start2, end2, ...]
        or None if no qualifying silence detected.
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        "silencedetect=noise=-35dB:d=7",
        "-nostats",
        "-f",
        "null",
        "-",
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg silence detection error: {e.stderr or e}")
        return None

    pattern = r"silence_(start|end): ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    matches = re.findall(pattern, output)
    times = [float(t) for _, t in matches]

    return times if times else None


def compute_non_silence_intervals(
    silence_times: List[float], video_duration: float
) -> Optional[List[Tuple[float, float]]]:
    """Transform silence boundaries into non-silence content intervals.

    Args:
        silence_times: Flat list of silence boundaries [start1, end1, ...]
        video_duration: Total video duration in seconds

    Returns:
        List of (start, end) tuples representing content segments
    """
    if not silence_times:
        return None

    times = silence_times.copy()

    # Handle video starting with silence
    if times[0] <= 0:
        times.pop(0)
    else:
        times.insert(0, 0)

    # Handle video ending without silence
    if len(times) % 2 != 0:
        times.append(video_duration)

    # Convert to (start, end) pairs
    intervals = list(zip(times[::2], times[1::2]))
    return intervals


def extract_and_concat_segments(
    input_path: str,
    output_path: str,
    intervals: List[Tuple[float, float]],
    min_segment_duration: float = 1.0,
) -> None:
    """Extract non-contiguous segments and concatenate using FFmpeg stream copy.

    Uses -to (end position) instead of -t (duration) to ensure
    no speech is cut at segment boundaries.

    Keyframe alignment may include extra silence at segment starts (acceptable).

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        intervals: List of (start, end) tuples in seconds
        min_segment_duration: Minimum segment duration to include (default: 1s)
    """
    # Filter out very short segments
    intervals = [(s, e) for s, e in intervals if e - s >= min_segment_duration]

    if not intervals:
        shutil.copyfile(input_path, output_path)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []

        for i, (start, end) in enumerate(intervals):
            seg_file = os.path.join(temp_dir, f"segment_{i:03d}.mkv")

            # CRITICAL: Use -to (absolute end position), not -t (duration)
            # This ensures we don't cut speech at segment ends
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(end),
                "-i",
                input_path,
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                seg_file,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to extract segment {i} ({start}-{end}s): {e.stderr.decode() if e.stderr else e}"
                )
                raise
            segment_files.append(seg_file)

        # Create concat list
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        # Concatenate to final MP4
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to concatenate segments: {e.stderr.decode() if e.stderr else e}"
            )
            raise


def mp4_silence_remove(input_path: str, output_path: str):
    """Remove silence intervals from video using FFmpeg stream copy.

    Performance: 10-100x faster than MoviePy (stream copy vs re-encoding).
    Accuracy: Keyframe-aligned (includes extra silence at boundaries, never cuts speech).

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
    """
    logger.info(f"Detecting silence in: {input_path}")
    silence_times = detect_silence_intervals(input_path)

    if not silence_times:
        logger.info("No silence >= 7s detected, copying file as-is")
        shutil.copyfile(input_path, output_path)
        return

    logger.info(f"Detected {len(silence_times) // 2} silence intervals")
    video_duration = get_video_duration(input_path)
    intervals = compute_non_silence_intervals(silence_times, video_duration)

    if not intervals:
        logger.warning(
            "Video appears to be all silence, extracting first 10s as fallback"
        )
        cmd = ["ffmpeg", "-y", "-i", input_path, "-t", "10", "-c", "copy", output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to extract fallback segment: {e.stderr.decode() if e.stderr else e}"
            )
            raise
        return

    logger.info(f"Extracting {len(intervals)} non-silence segments")
    extract_and_concat_segments(input_path, output_path, intervals)
