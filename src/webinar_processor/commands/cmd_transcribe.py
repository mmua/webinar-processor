import os
import tempfile
import click
import json

from webinar_processor.services.transcription_service import diarize_wav, transcribe_wav
from webinar_processor.utils.ffmpeg import (
    convert_mp4_to_wav,
    mp4_silence_remove,
    get_wav_filename,
    normalize_audio_file,
)


def _resolve_transcript_path(webinar_path: str, transcript_path: str) -> str:
    if transcript_path:
        return transcript_path
    transcript_dir = os.path.dirname(webinar_path)
    return os.path.join(transcript_dir, "transcript.json")


@click.command()
@click.argument('webinar_path', nargs=1)
@click.argument('transcript_path', default='')
@click.argument('language', nargs=1, default="ru")
@click.option(
    '--normalize-audio/--no-normalize-audio',
    default=False,
    show_default=True,
    help='Normalize loudness before ASR (recommended for phone-call audio).',
)
def transcribe(
    webinar_path: str,
    transcript_path: str,
    language: str,
    normalize_audio: bool,
):
    """
    Transcribe video file with speaker detection
    """
    transcript_path = _resolve_transcript_path(webinar_path, transcript_path)

    # Trim video.
    output_file, ext = os.path.splitext(webinar_path)
    output_name = output_file + ".stripped" + ext

    # If the input is a wav file, skip silence removal (moviepy can't process wav)
    if os.path.splitext(webinar_path)[1].lower() == ".wav":
        output_name = webinar_path
    else:
        mp4_silence_remove(webinar_path, output_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_filename = get_wav_filename(output_name, tmpdir)
        convert_mp4_to_wav(output_name, wav_filename)
        asr_audio_path = wav_filename

        if normalize_audio:
            normalized_wav = os.path.join(tmpdir, "normalized.wav")
            normalize_audio_file(wav_filename, normalized_wav)
            asr_audio_path = normalized_wav

        asr_result = transcribe_wav(asr_audio_path, language=language)

        asr_path = transcript_path + ".asr"
        with open(asr_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(asr_result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        try:
            result = diarize_wav(asr_audio_path, asr_result)
        except ValueError as exc:
            click.echo(click.style(f"Error: {exc}", fg="red"))
            raise click.Abort() from exc

        with open(transcript_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        click.echo(asr_path)


@click.command()
@click.argument('webinar_path', nargs=1)
@click.argument('transcript_path', nargs=1)
def diarize(webinar_path: str, transcript_path: str):
    """
    Diarize video file with speaker detection after transcription
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_filename = get_wav_filename(webinar_path, tmpdir)
        convert_mp4_to_wav(webinar_path, wav_filename)

        with open(transcript_path + ".asr", "r", encoding="utf-8") as json_file:
            asr_result = json.load(json_file)

        try:
            result = diarize_wav(wav_filename, asr_result)
        except ValueError as exc:
            click.echo(click.style(f"Error: {exc}", fg="red"))
            raise click.Abort() from exc

        with open(transcript_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        return asr_result, result 