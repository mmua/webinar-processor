# src/webinar_processor/services/gender_detection_service.py
import os
import tempfile
from typing import Dict, List, Union, Optional
import numpy as np

class GenderDetectionService:
    """Service for detecting speaker gender from audio segments."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the gender detection service.
        
        Args:
            model_path: Optional path to a custom model, defaults to pretrained model
        """
        # Lazy-load the model only when needed
        self._model = None
        self._model_path = model_path
    
    @property
    def model(self):
        """Lazy-load the gender classification model."""
        if self._model is None:
            from voice_gender_classification import GenderClassificationPipeline
            
            if self._model_path:
                # Load custom model if provided
                self._model = GenderClassificationPipeline.from_pretrained(self._model_path)
            else:
                # Use default pretrained model
                self._model = GenderClassificationPipeline.from_pretrained(
                    "griko/gender_cls_svm_ecapa_voxceleb"
                )
        
        return self._model
    
    def detect_gender(self, audio_path: str) -> str:
        """
        Detect gender from a complete audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Gender as "male" or "female"
        """
        try:
            result = self.model(audio_path)
            return result[0]  # Returns "male" or "female"
        except Exception as e:
            print(f"Error detecting gender: {str(e)}")
            return "unknown"
    
    def detect_gender_from_segments(self, audio_path: str, 
                                   segments: List[Dict[str, Union[float, str]]]) -> Dict[str, str]:
        """
        Detect gender for each speaker in a set of segments.
        
        Args:
            audio_path: Path to the full audio file
            segments: List of segments with speaker IDs, start and end times
            
        Returns:
            Dictionary mapping speaker IDs to detected genders
        """
        import soundfile as sf
        from pydub import AudioSegment
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            speaker = segment.get("speaker", "")
            if not speaker:
                continue
                
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
                
            speaker_segments[speaker].append({
                "start": segment["start"],
                "end": segment["end"]
            })
        
        # Create temporary directory for audio segments
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract and classify each speaker
            speaker_genders = {}
            
            for speaker, segs in speaker_segments.items():
                # Only process speakers with enough audio
                total_duration = sum(s["end"] - s["start"] for s in segs)
                if total_duration < 3.0:  # Skip if less than 3 seconds
                    continue
                
                # Extract the longest segment for this speaker
                longest_segment = max(segs, key=lambda s: s["end"] - s["start"])
                
                # Load full audio
                audio = AudioSegment.from_file(audio_path)
                
                # Extract segment (convert seconds to milliseconds)
                start_ms = int(longest_segment["start"] * 1000)
                end_ms = int(longest_segment["end"] * 1000)
                segment_audio = audio[start_ms:end_ms]
                
                # Save to temporary file
                temp_path = os.path.join(tmpdir, f"speaker_{speaker}.wav")
                segment_audio.export(temp_path, format="wav")
                
                # Detect gender
                gender = self.detect_gender(temp_path)
                speaker_genders[speaker] = gender
            
            return speaker_genders
    
    def batch_detect_gender(self, audio_paths: List[str]) -> List[str]:
        """
        Batch detection of gender from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of genders in the same order as input paths
        """
        results = self.model(audio_paths)
        return results