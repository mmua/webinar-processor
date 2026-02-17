"""
Integration Workflow Tests
========================

This module contains integration tests that verify end-to-end workflows
across multiple components of the webinar processor.

Test Verification Strategy
-------------------------
- Test complete workflows from video download to transcription
- Verify interaction between multiple components
- Mock external services and resource-intensive operations
- Ensure the complete flow works as expected

These tests validate that:
- Commands can be chained together successfully
- Data flows correctly between processing steps
- The system handles the complete workflow without errors
- External dependencies are properly integrated

Testing approach:
1. Mock external services (YouTube, ASR/diarization, etc.)
2. Set up test environment with dummy files
3. Execute command sequences as a user would
4. Verify outputs at each stage
"""

from unittest.mock import patch, Mock
from click.testing import CliRunner
from webinar_processor import cli

def test_basic_workflow(mock_youtube, mock_openai, temp_dir):
    """
    Test basic webinar processing workflow: download -> transcribe.
    
    Verification strategy:
    1. Video download should complete successfully
    2. Downloaded video should be prepared for transcription
    3. Transcription should attempt to process the video
    4. The workflow should handle dependencies appropriately
    
    This test mocks:
    - YouTube download API
    - FFmpeg video processing
    - Transcription service calls
    - Diarization service calls
    
    The test avoids actual network calls and resource-intensive operations
    while still verifying the workflow functions correctly.
    """
    runner = CliRunner()
    
    # Mock YouTube download
    with patch('webinar_processor.commands.cmd_yt_download.YouTube', return_value=mock_youtube):
        with patch(
            'webinar_processor.commands.cmd_yt_download.requests.get',
            return_value=Mock(content=b"poster-bytes")
        ):
            download_result = runner.invoke(
                cli,
                ['download', 'https://youtube.com/watch?v=test', '--output-dir', str(temp_dir)]
            )
            assert download_result.exit_code == 0, "Download command failed"
    
    # Create dummy video files for transcription
    video_path = temp_dir / "test_video.mp4" 
    video_path.touch()
    stripped_path = temp_dir / "test_video.stripped.mp4"
    stripped_path.touch()
    
    # Mock ffmpeg utilities and transcription service calls.
    with patch(
        'webinar_processor.commands.cmd_transcribe.mp4_silence_remove',
        side_effect=lambda input_path, output_path: None
    ):
        with patch(
            'webinar_processor.commands.cmd_transcribe.convert_mp4_to_wav',
            side_effect=lambda input_path, output_path: None
        ):
            with patch(
                'webinar_processor.commands.cmd_transcribe.transcribe_wav',
                return_value={
                    "text": "Test transcription",
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "Test transcription"}
                    ],
                    "language": "english",
                    "model": "antony66/whisper-large-v3-russian",
                }
            ):
                with patch(
                    'webinar_processor.commands.cmd_transcribe.diarize_wav',
                    return_value=[
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "speaker": "SPEAKER_00",
                            "text": "Test transcription",
                        }
                    ]
                ):
                    transcript_path = temp_dir / "transcript.json"
                    transcribe_result = runner.invoke(
                        cli,
                        ['transcribe', str(video_path), str(transcript_path), 'en'],
                        catch_exceptions=True
                    )
                    assert transcribe_result.exit_code == 0, (
                        f"Transcribe command failed: {transcribe_result.output}"
                    )


def test_transcribe_with_audio_normalization(temp_dir):
    """Test transcribe workflow with optional normalization enabled."""
    runner = CliRunner()

    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    transcript_path = temp_dir / "transcript.json"

    with patch(
        'webinar_processor.commands.cmd_transcribe.mp4_silence_remove',
        side_effect=lambda input_path, output_path: None
    ):
        with patch(
            'webinar_processor.commands.cmd_transcribe.convert_mp4_to_wav',
            side_effect=lambda input_path, output_path: None
        ):
            with patch(
                'webinar_processor.commands.cmd_transcribe.normalize_audio_file',
                side_effect=lambda input_path, output_path: None
            ) as normalize_mock:
                with patch(
                    'webinar_processor.commands.cmd_transcribe.transcribe_wav',
                    return_value={
                        "text": "Normalized test transcription",
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": "Normalized test transcription"}
                        ],
                        "language": "russian",
                        "model": "antony66/whisper-large-v3-russian",
                    }
                ):
                    with patch(
                        'webinar_processor.commands.cmd_transcribe.diarize_wav',
                        return_value=[
                            {
                                "start": 0.0,
                                "end": 2.0,
                                "speaker": "SPEAKER_00",
                                "text": "Normalized test transcription",
                            }
                        ]
                    ):
                        transcribe_result = runner.invoke(
                            cli,
                            [
                                'transcribe',
                                '--normalize-audio',
                                str(video_path),
                                str(transcript_path),
                                'ru',
                            ],
                            catch_exceptions=True
                        )
                        assert transcribe_result.exit_code == 0, (
                            f"Transcribe command failed: {transcribe_result.output}"
                        )
                        assert normalize_mock.call_count == 1, (
                            "Normalization should run when --normalize-audio is enabled"
                        )
