
import os
import tempfile
from typing import Dict, List
import click
import json

from webinar_processor.utils.ffmpeg import convert_mp4_to_wav, mp4_silence_remove
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.path import get_wav_filename


def diarize_wav(wav_filename: str, transcription_result: List[Dict]):
        from pyannote.audio import Pipeline
        from pyannote_whisper.utils import diarize_text

        # # Указываем путь до файла с конфигом, он должен быть в той же директории, как сказано на шаге 3.
        # config_path = get_config_path('diarization.yaml')
        # pipeline = Pipeline.from_pretrained(config_path)
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token is None:
            click.echo(click.style(f'Error: HuggingFace token is not set', fg='red'))
            raise click.Abort

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        )

        # Сегментация аудио-файла на реплики спикеров. Путь обязательно абсолютный.
        diarization_result = pipeline(wav_filename)

        diarization_list = []
        # print the result
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                diarization_list.append((turn.start, turn.end, f'speaker_{speaker}'))
  

        # Пересечение расшифровки и сегментаци.
        final_result = diarize_text(transcription_result, diarization_result)

        # Вывод результата.
        result = []
        for seg, spk, text in final_result:
            segment = {
                'start': seg.start,
                'end': seg.end,
                'speaker': spk,
                'text': text
            }
            result.append(segment)
        return result


def transcribe_wav(wav_filename: str, language="ru"):
    import whisper
    prompts = {
         "ru": "Добрый день, это вебинар по управлению организациями и проектами.",
         "en": "Hello, this is project managment webinar. Stay tuned."
    }
    model = whisper.load_model("large-v2")
    asr_result = model.transcribe(wav_filename, language=language, initial_prompt=prompts[language])
    return asr_result


@click.command()
@click.argument('webinar_path', nargs=1)
@click.argument('transcript_path', default='')
@click.argument('language', nargs=1, default="ru")
def transcribe(webinar_path: str, transcript_path: str, language: str):
    """
    Transcribe video file with speaker detection
    """
    if not transcript_path:
        transcript_dir = os.path.dirname(webinar_path)
        transcript_path = os.path.join(transcript_dir, "transcript.json")

    # trim video 
    output_file, ext = os.path.splitext(webinar_path)
    output_name = output_file + ".stripped" + ext
    mp4_silence_remove(webinar_path, output_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_filename = get_wav_filename(output_name, tmpdir)
        convert_mp4_to_wav(output_name, wav_filename)

        asr_result = transcribe_wav(wav_filename, language=language)
        with open(transcript_path + ".asr", "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(asr_result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        result = diarize_wav(wav_filename, asr_result)
        with open(transcript_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        return asr_result, result


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

        result = diarize_wav(wav_filename, asr_result)
        with open(transcript_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)

        return asr_result, result
