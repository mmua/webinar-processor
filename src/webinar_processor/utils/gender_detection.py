# src/webinar_processor/utils/gender_detection.py
import os
from typing import Dict, List, Optional

class GenderDetector:
    """Utility class for gender detection."""
    
    @staticmethod
    def detect_gender_in_transcript(transcript: List[Dict], 
                                  audio_path: str) -> List[Dict]:
        """
        Add gender information to a transcript.
        
        Args:
            transcript: List of transcript segments
            audio_path: Path to the audio file
            
        Returns:
            Updated transcript with gender information
        """
        from ..services.gender_detection_service import GenderDetectionService
        
        # Initialize service
        service = GenderDetectionService()
        
        # Get gender for each speaker
        speaker_genders = service.detect_gender_from_segments(audio_path, transcript)
        
        # Update transcript with gender info
        for segment in transcript:
            speaker = segment.get("speaker", "")
            if speaker and speaker in speaker_genders:
                segment["gender"] = speaker_genders[speaker]
        
        return transcript
    
    @staticmethod
    def get_majority_gender(segments: List[Dict], speaker_id: str) -> str:
        """
        Get the majority gender for a speaker across segments.
        
        Args:
            segments: List of transcript segments
            speaker_id: Speaker ID to get gender for
            
        Returns:
            Majority gender as "male", "female", or "unknown"
        """
        genders = {}
        
        # Count gender occurrences
        for segment in segments:
            if segment.get("speaker") == speaker_id and "gender" in segment:
                gender = segment["gender"]
                genders[gender] = genders.get(gender, 0) + 1
        
        # Find majority
        if not genders:
            return "unknown"
            
        return max(genders.items(), key=lambda x: x[1])[0]
