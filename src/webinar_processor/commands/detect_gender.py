import pickle
import click
from scipy.io import wavfile
from webinar_processor.gender_detection.voice_features import extract_features
from webinar_processor.gender_detection.identify_gender import identify_gender
from webinar_processor.utils.package import get_model_path


def read_segment(wav_path, start_time, end_time):
    """
    Reads a segment from a WAV file.
    
    :param wav_path: Path to the WAV file.
    :param start_time: Start time of the segment in seconds.
    :param end_time: End time of the segment in seconds.
    :return: rate, segment
    """
    rate, audio = wavfile.read(wav_path)
    
    # Calculate start and end index
    start_index = int(start_time * rate)
    end_index = int(end_time * rate)
    
    # Slice the audio array to get the segment
    segment = audio[start_index:end_index]
    
    return rate, segment


@click.command()
@click.argument('wav_path', nargs=1)
@click.argument('start', nargs=1, type=float)
@click.argument('end', nargs=1, type=float)
def detect_gender(wav_path: str, start: float, end: float):
    """
    Detect gender with audio segment
    """
    rate, audio = read_segment(wav_path, start, end)
    features = extract_features(rate, audio)

    females_model_path = get_model_path("females.gmm")
    females_gmm = pickle.load(open(females_model_path, 'rb'))

    males_model_path = get_model_path("males.gmm")
    males_gmm = pickle.load(open(males_model_path, 'rb'))

    gender = identify_gender(females_gmm, males_gmm, features)
    print(f"Detected gender: {gender}")
