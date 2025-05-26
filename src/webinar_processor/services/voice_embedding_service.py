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
    
    def __init__(self):
        """Initialize the voice embedding service."""
        self.model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
        self.audio = Audio(sample_rate=16000, mono="downmix")
    
    def extract_embedding(self, audio_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """
        Extract voice embedding from an audio segment.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Voice embedding vector or None if extraction failed
        """
        try:
            # Extract audio segment
            segment = Segment(start_time, end_time)
            waveform, sample_rate = self.audio.crop(audio_path, segment)
            
            # Generate embedding
            embedding = self.model(waveform[None])
            return embedding[0]  # Return first (and only) embedding
            
        except Exception as e:
            print(f"Error extracting voice embedding: {str(e)}")
            return None
    
    def get_speaker_embeddings(self, 
                             audio_path: str, 
                             transcript: List[Dict],
                             min_duration: float = 3.0) -> Dict[str, List[np.ndarray]]:
        """
        Get representative embeddings for each speaker in a transcript.
        
        Args:
            audio_path: Path to the audio file
            transcript: List of transcript segments
            min_duration: Minimum duration for a segment to be considered
            
        Returns:
            Dictionary mapping speaker IDs to lists of embeddings
        """
        speaker_embeddings = {}
        
        # Group segments by speaker
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
        """
        Calculate mean embedding from a list of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Mean embedding vector
        """
        return np.mean(embeddings, axis=0)
    
    def process_audio_file(self, 
                          audio_path: str,
                          transcript: List[Dict],
                          min_duration: float = 3.0) -> Dict[str, np.ndarray]:
        """
        Process an audio file and get mean embeddings for each speaker.
        
        Args:
            audio_path: Path to the audio file
            transcript: List of transcript segments
            min_duration: Minimum duration for a segment to be considered
            
        Returns:
            Dictionary mapping speaker IDs to mean embeddings
        """
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