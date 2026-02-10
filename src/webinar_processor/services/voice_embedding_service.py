import numpy as np
from typing import Dict, List, Optional, Tuple
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import tempfile
import os
from webinar_processor.utils.ffmpeg import convert_mp4_to_wav, get_wav_filename


class VoiceEmbeddingService:
    """Service for extracting and managing voice embeddings."""

    DEFAULT_MODEL = 'pyannote/wespeaker-voxceleb-resnet34-LM'

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get(
            'SPEAKER_EMBEDDING_MODEL', self.DEFAULT_MODEL
        )
        self.model = PretrainedSpeakerEmbedding(self.model_name)
        self.audio = Audio(sample_rate=16000, mono="downmix")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension for the current model."""
        # WeSpeaker ResNet34-LM produces 256-dim embeddings
        if 'wespeaker' in self.model_name.lower() or 'resnet' in self.model_name.lower():
            return 256
        # ECAPA-TDNN produces 192-dim embeddings
        if 'ecapa' in self.model_name.lower():
            return 192
        # Default: try to infer from model, fallback to 256
        return 256

    def extract_embedding(self, audio_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Extract voice embedding from an audio segment."""
        try:
            segment = Segment(start_time, end_time)
            waveform, sample_rate = self.audio.crop(audio_path, segment)
            embedding = self.model(waveform[None])
            return embedding[0]
        except Exception as e:
            print(f"Error extracting voice embedding: {str(e)}")
            return None

    def extract_single_speaker_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract embedding from full audio file, treating all speech as one speaker.

        Used for enrollment — processes the entire audio as a single speaker.
        """
        try:
            waveform, sample_rate = self.audio(audio_path)
            embedding = self.model(waveform[None])
            return embedding[0]
        except Exception as e:
            print(f"Error extracting single-speaker embedding: {str(e)}")
            return None

    def get_speaker_embeddings(self,
                               audio_path: str,
                               transcript: List[Dict],
                               min_duration: float = 3.0) -> Dict[str, List[np.ndarray]]:
        """Get representative embeddings for each speaker in a transcript."""
        speaker_embeddings = {}

        for segment in transcript:
            speaker_id = segment.get('speaker')
            if not speaker_id:
                continue

            duration = segment['end'] - segment['start']
            if duration < min_duration:
                continue

            embedding = self.extract_embedding(
                audio_path,
                segment['start'],
                segment['end']
            )

            if embedding is not None:
                if speaker_id not in speaker_embeddings:
                    speaker_embeddings[speaker_id] = []
                speaker_embeddings[speaker_id].append(embedding)

        return speaker_embeddings

    def get_mean_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate mean embedding from a list of embeddings."""
        return np.mean(embeddings, axis=0)

    @staticmethod
    def update_mean_embedding(stored: np.ndarray, new: np.ndarray, n_samples: int) -> np.ndarray:
        """Compute running average when re-encountering a known speaker.

        Args:
            stored: Current stored mean embedding
            new: New embedding to incorporate
            n_samples: Number of samples that contributed to `stored`

        Returns:
            Updated mean embedding (n_samples+1 weighted average)
        """
        return (stored * n_samples + new) / (n_samples + 1)

    def process_audio_file(self,
                           audio_path: str,
                           transcript: List[Dict],
                           min_duration: float = 3.0) -> Dict[str, np.ndarray]:
        """Process an audio file and get mean embeddings for each speaker."""
        # Convert to WAV if needed
        if not audio_path.lower().endswith('.wav'):
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = get_wav_filename(audio_path, tmpdir)
                convert_mp4_to_wav(audio_path, wav_path)
                audio_path = wav_path

        # Get embeddings for each speaker
        speaker_embeddings = self.get_speaker_embeddings(
            audio_path,
            transcript,
            min_duration
        )

        # Calculate mean embeddings
        mean_embeddings = {}
        for speaker_id, embeddings in speaker_embeddings.items():
            if embeddings:
                mean_embeddings[speaker_id] = self.get_mean_embedding(embeddings)

        return mean_embeddings
