import tempfile
import click
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from webinar_processor.utils.ffmpeg import convert_mp4_to_wav

from webinar_processor.utils.path import get_wav_filename

@click.command()
@click.argument('webinar_path', nargs=1)
@click.argument('transcript_path', nargs=1)
@click.argument('identities_path', nargs=1)
def identify(webinar_path: str, transcript_path: str, identities_path: str):
    model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    audio = Audio(sample_rate=16000, mono="downmix")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_filename = get_wav_filename(webinar_path, tmpdir)
        convert_mp4_to_wav(webinar_path, wav_filename)

    # create dict of speakers with stacked embeddings

    # load known speakers embeddings

    # identify all known speakers in transcription
    
    # detect gender for unknown speakers
    
    # re-label speakers transcription, mark unknown speakers with funny names 
     
    # extract embedding for a speaker speaking between t=3s and t=6s
    speaker1 = Segment(3., 6.)
    waveform1, sample_rate = audio.crop("audio.wav", speaker1)
    embedding1 = model(waveform1[None])

    # extract embedding for a speaker speaking between t=7s and t=12s
    speaker2 = Segment(7., 12.)
    waveform2, sample_rate = audio.crop("audio.wav", speaker2)
    embedding2 = model(waveform2[None])

    # compare embeddings using "cosine" distance
    from scipy.spatial.distance import cdist
    distance = cdist(embedding1, embedding2, metric="cosine")