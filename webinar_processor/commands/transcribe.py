
import tempfile
from typing import Dict, List
import click
import json
from pywhispercpp.model import Model

from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text

from webinar_processor.utils.ffmpeg import convert_mp4_to_wav
from webinar_processor.utils.path import get_wav_filename
from webinar_processor.utils.config import get_config_path

def diarize_wav(wav_filename: str, transcription_result: List[Dict]):
        # Указываем путь до файла с конфигом, он должен быть в той же директории, как сказано на шаге 3.
        config_path = get_config_path('diarization.yaml')
        pipeline = Pipeline.from_pretrained(config_path)

        # Сегментация аудио-файла на реплики спикеров. Путь обязательно абсолютный.
        diarization_result = pipeline(wav_filename)

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


def transcribe_wav(model_path: str, wav_filename: str, language="ru"):
    model = Model(model_path, n_threads=8)
    # Указываем путь до аудио-файла, кторый будем расшифровывать в текст. Путь обязательно абсолютный.
    asr_result = model.transcribe(wav_filename, language=language)
    # Конвертация результата в формат, который понимает pyannote-whisper.
    result = {'segments': list()}

    for item in asr_result:
        result['segments'].append({
            'start': item.t0 / 100,
            'end': item.t1 / 100,
            'text': item.text
            }
        )
    return result


@click.command()
@click.argument('model_path', nargs=1)
@click.argument('webinar_path', nargs=1)
@click.argument('transcript_path', nargs=1)
@click.argument('language', nargs=1, default="ru")
def transcribe(model_path: str, webinar_path: str, transcript_path: str, language: str):
    """
    Transcribe video file with speaker detection
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_filename = get_wav_filename(webinar_path, tmpdir)
        convert_mp4_to_wav(webinar_path, wav_filename)

        result = transcribe_wav(model_path, wav_filename, language=language)

        result = diarize_wav(wav_filename, result)

        with open(transcript_path, "w", encoding="utf-8") as json_file:
            serialized_result = json.dumps(result, indent=4, ensure_ascii=False)
            json_file.write(serialized_result)
