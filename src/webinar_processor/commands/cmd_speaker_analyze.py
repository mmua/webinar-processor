import click
import json
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService


def consolidate_segments(segments: List[Dict], gap_threshold: float = 0.5,
                         min_duration: float = 5.0, max_duration: float = 10.0,
                         max_samples_per_speaker: int = 3) -> Dict[str, List[Dict]]:
    """Group consecutive segments from the same speaker into consolidated samples.
    
    Creates few high-confidence samples from strictly consecutive segments.
    Only groups segments with no gaps between them (gap_threshold only for very small pauses).
    Picks the longest consecutive sequences first.
    
    Args:
        segments: List of transcript segments with start, end, speaker, text
        gap_threshold: Maximum gap (seconds) to merge consecutive segments (should be small!)
        min_duration: Minimum sample duration
        max_duration: Maximum sample duration
        max_samples_per_speaker: Maximum number of samples to extract per speaker
        
    Returns:
        Dict mapping temp_speaker_id -> list of consolidated samples
        Each sample has: start_time, end_time, duration, text_preview
    """
    # Group segments by speaker
    speaker_segments = {}
    for seg in segments:
        speaker = seg.get('speaker')
        if not speaker:
            continue
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(seg)
    
    consolidated = {}
    
    for speaker, segs in speaker_segments.items():
        # Sort by start time
        segs.sort(key=lambda x: x['start'])
        
        # Build consecutive runs (no gaps or very small gaps)
        runs = []
        current_run = {
            'start_time': segs[0]['start'],
            'end_time': segs[0]['end'],
            'text_parts': [segs[0]['text']]
        }
        
        for seg in segs[1:]:
            gap = seg['start'] - current_run['end_time']
            potential_duration = seg['end'] - current_run['start_time']
            
            # Only merge if gap is very small AND we haven't exceeded max duration
            if gap <= gap_threshold and potential_duration <= max_duration:
                # Merge into current run
                current_run['end_time'] = seg['end']
                current_run['text_parts'].append(seg['text'])
            else:
                # End current run
                duration = current_run['end_time'] - current_run['start_time']
                if duration >= min_duration:
                    runs.append({
                        'start_time': current_run['start_time'],
                        'end_time': current_run['end_time'],
                        'duration': duration,
                        'text_preview': ' '.join(current_run['text_parts'])[:200]
                    })
                
                # Start new run
                current_run = {
                    'start_time': seg['start'],
                    'end_time': seg['end'],
                    'text_parts': [seg['text']]
                }
        
        # Don't forget the last run
        duration = current_run['end_time'] - current_run['start_time']
        if duration >= min_duration:
            runs.append({
                'start_time': current_run['start_time'],
                'end_time': current_run['end_time'],
                'duration': duration,
                'text_preview': ' '.join(current_run['text_parts'])[:200]
            })
        
        # Sort runs by duration (longest first) and take top N
        runs.sort(key=lambda x: x['duration'], reverse=True)
        consolidated[speaker] = runs[:max_samples_per_speaker]
    
    return consolidated


def find_video_file(directory: str) -> Tuple[str, str]:
    """Find video file (.stripped.mp4) and transcript in directory.
    
    Returns:
        Tuple of (video_filename, transcript_path) or raises error
    """
    dir_path = Path(directory)
    
    # Find transcript
    transcript_path = dir_path / "transcript.json"
    if not transcript_path.exists():
        raise click.ClickException(f"transcript.json not found in {directory}")
    
    # Find video file (any .stripped.mp4 or fallback to .mp4)
    video_files = list(dir_path.glob("*.stripped.mp4"))
    if not video_files:
        # Try any mp4
        video_files = list(dir_path.glob("*.mp4"))
    
    if not video_files:
        raise click.ClickException(f"No video file found in {directory}")
    
    video_filename = video_files[0].name
    
    return video_filename, str(transcript_path)


def encode_embedding(embedding: np.ndarray) -> str:
    """Encode numpy array to base64 string for JSON storage."""
    return base64.b64encode(embedding.tobytes()).decode('utf-8')


def decode_embedding(encoded: str) -> np.ndarray:
    """Decode base64 string back to numpy array."""
    from webinar_processor.services.speaker_database import EMBEDDING_DTYPE
    bytes_data = base64.b64decode(encoded)
    return np.frombuffer(bytes_data, dtype=EMBEDDING_DTYPE)


@click.command('analyze')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--gap-threshold', type=float, default=0.5, help='Max gap (seconds) to merge segments (default: 0.5)')
@click.option('--min-duration', type=float, default=5.0, help='Minimum sample duration')
@click.option('--max-duration', type=float, default=10.0, help='Maximum sample duration (default: 10s, shorter=cleaner)')
@click.option('--max-samples', type=int, default=3, help='Maximum samples per speaker (default: 3)')
def analyze(directory: str, gap_threshold: float, min_duration: float, max_duration: float, max_samples: int):
    """Analyze a webinar directory and extract speaker samples.
    
    Reads transcript.json, groups segments by speaker, creates consolidated
    samples, and extracts voice embeddings. Results saved to speaker_analysis.json.
    """
    directory = os.path.abspath(directory)
    
    click.echo(f"Analyzing: {directory}")
    
    # Find files
    try:
        video_filename, transcript_path = find_video_file(directory)
    except click.ClickException as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    
    video_path = os.path.join(directory, video_filename)
    
    click.echo(f"Video: {video_filename}")
    click.echo(f"Transcript: {transcript_path}")
    
    # Read transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        click.echo(f"Error reading transcript: {e}", err=True)
        raise click.Abort()
    
    if not isinstance(transcript, list) or not transcript:
        click.echo("Error: Transcript must be a non-empty list of segments", err=True)
        raise click.Abort()
    
    # Consolidate segments
    click.echo(f"\nConsolidating segments (gap_threshold={gap_threshold}s, min={min_duration}s, max={max_duration}s, max_samples={max_samples})...")
    consolidated = consolidate_segments(transcript, gap_threshold, min_duration, max_duration, max_samples)
    
    if not consolidated:
        click.echo("No speakers found in transcript")
        return
    
    click.echo(f"Found {len(consolidated)} speakers")
    for speaker, samples in consolidated.items():
        total_duration = sum(s['duration'] for s in samples)
        click.echo(f"  {speaker}: {len(samples)} samples, {total_duration:.1f}s total")
    
    # Initialize services
    voice_service = VoiceEmbeddingService()
    
    # Build analysis data structure
    analysis = {
        'directory': directory,
        'video_filename': video_filename,
        'transcript_path': transcript_path,
        'analyzed_at': None,  # Will fill in after processing
        'speakers': {}
    }
    
    # Process each speaker
    total_samples_created = 0
    
    with click.progressbar(consolidated.items(), label='Extracting embeddings') as items:
        for temp_speaker_id, samples in items:
            # Calculate total duration and segment count
            total_duration = sum(s['duration'] for s in samples)
            segment_count = len([s for s in transcript if s.get('speaker') == temp_speaker_id])
            
            # Build speaker entry
            speaker_entry = {
                'temp_id': temp_speaker_id,
                'total_duration': total_duration,
                'segment_count': segment_count,
                'labeled_name': None,  # Will be filled by 'label' command
                'samples': []
            }
            
            # Extract embeddings for each sample
            for i, sample in enumerate(samples):
                # Extract embedding
                embedding = voice_service.extract_embedding(
                    video_path,
                    sample['start_time'],
                    sample['end_time']
                )
                
                if embedding is not None:
                    speaker_entry['samples'].append({
                        'index': i,
                        'start_time': sample['start_time'],
                        'end_time': sample['end_time'],
                        'duration': sample['duration'],
                        'text_preview': sample['text_preview'],
                        'embedding': encode_embedding(embedding)
                    })
                    total_samples_created += 1
                else:
                    click.echo(f"\nWarning: Failed to extract embedding for {temp_speaker_id} sample {i}", err=True)
            
            analysis['speakers'][temp_speaker_id] = speaker_entry
    
    # Save analysis to JSON file
    analysis['analyzed_at'] = datetime.now().isoformat()
    analysis_path = os.path.join(directory, 'speaker_analysis.json')
    
    try:
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    except Exception as e:
        click.echo(f"Error saving analysis: {e}", err=True)
        raise click.Abort()
    
    click.echo(f"\nDone! Created {len(consolidated)} speaker entries with {total_samples_created} samples")
    click.echo(f"Saved to: {analysis_path}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Run 'webinar_processor speakers label {directory}' to label speakers")
    click.echo(f"  2. Run 'webinar_processor speakers identify {directory}' to identify speakers")
