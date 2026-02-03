import click
import json
from typing import Optional, List, Dict, Tuple
import os
from webinar_processor.services.speaker_database import SpeakerDatabase
from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService
from webinar_processor.llm import LLMClient, LLMError

import logging
logger = logging.getLogger(__name__)

@click.group()
def speakers():
    """Manage speaker profiles and identifications."""
    pass

@speakers.command()
@click.argument('speaker_id')
@click.option('--name', help='Manually confirm speaker name')
@click.option('--gender', type=click.Choice(['male', 'female', 'unknown']), help='Set speaker gender')
def update(speaker_id: str, name: Optional[str], gender: Optional[str]):
    """Update speaker information."""
    db = SpeakerDatabase()
    
    if not name and not gender:
        click.echo("Please provide at least one update (--name or --gender)")
        return
    
    success = db.update_speaker(
        speaker_id=speaker_id,
        confirmed_name=name,
        gender=gender
    )
    
    if success:
        click.echo(f"Successfully updated speaker {speaker_id}")
    else:
        click.echo(f"Failed to update speaker {speaker_id}")

@speakers.command()
@click.argument('speaker_id')
def info(speaker_id: str):
    """Show speaker information."""
    db = SpeakerDatabase()
    speaker = db.get_speaker(speaker_id)
    
    if speaker:
        # Format the output
        click.echo(f"Speaker ID: {speaker['speaker_id']}")
        click.echo(f"Canonical Name: {speaker['canonical_name'] or 'Not set'}")
        click.echo(f"Inferred Name: {speaker['inferred_name'] or 'Not set'}")
        click.echo(f"Confirmed Name: {speaker['confirmed_name'] or 'Not set'}")
        click.echo(f"Gender: {speaker['gender'] or 'Unknown'}")
        click.echo(f"First Detected: {speaker['first_detected']}")
        click.echo(f"Last Updated: {speaker['last_updated']}")
        click.echo(f"Confidence Score: {speaker['confidence_score']}")
        
        if speaker['metadata']:
            click.echo("\nMetadata:")
            click.echo(json.dumps(speaker['metadata'], indent=2))
    else:
        click.echo(f"Speaker {speaker_id} not found")

def detect_self_introductions(transcript: List[Dict], llm_client: LLMClient) -> Dict[str, str]:
    """
    Detect self-introductions in the transcript and extract speaker names.
    Returns a mapping of speaker IDs to inferred names.
    """
    # Group consecutive segments by speaker
    speaker_segments = {}

    for segment in transcript:
        speaker = segment['speaker']
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)
        
    # Process each speaker's segments
    name_mappings = {}
    for speaker_id, segments in speaker_segments.items():
        # Combine text from first few segments (usually introductions)
        text = " ".join(seg['text'] for seg in segments[:10])
        
        # Use LLM to extract speaker name
        try:
            name = llm_client.extract_speaker_name(text)
            if name:
                name_mappings[speaker_id] = name
        except LLMError as e:
            logger.warning(f"Failed to extract name for speaker {speaker_id}: {e}")
            continue  # Skip this speaker, continue with others
    
    return name_mappings

@speakers.command()
@click.argument('transcript_path')
@click.argument('audio_path')
@click.option('--threshold', type=float, default=0.7, help='Similarity threshold for speaker matching')
@click.option('--min-duration', type=float, default=3.0, help='Minimum duration for voice segments')
def relabel(transcript_path: str, audio_path: str, threshold: float, min_duration: float):
    """
    Re-label speakers in a transcript using voice embeddings and the speaker database.
    
    This command will:
    1. Extract voice embeddings from the audio file for each speaker
    2. Match embeddings against the speaker database
    3. Create new speaker entries for unmatched speakers
    4. Detect self-introductions and update speaker names
    5. Update speaker labels with confirmed/inferred names
    6. Save the updated transcript
    """
    try:
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # Initialize services
        db = SpeakerDatabase()
        voice_service = VoiceEmbeddingService()
        llm_client = LLMClient()
        
        # Get all existing speakers from database
        existing_speakers = db.get_all_speakers()
        click.echo(f"Found {len(existing_speakers)} existing speakers in database")
        
        # Get mean embeddings for each speaker in transcript
        click.echo("Extracting voice embeddings...")
        mean_embeddings = voice_service.process_audio_file(
            audio_path,
            transcript,
            min_duration
        )
        
        if not mean_embeddings:
            click.echo("No valid voice embeddings could be extracted")
            return
        
        # Track speaker mappings and new speakers
        speaker_mappings = {}  # Maps temporary IDs to database IDs
        new_speaker_count = 0
        
        # First pass: Try to match existing speakers
        click.echo("Matching speakers against database...")
        for temp_id, embedding in mean_embeddings.items():
            match = db.find_matching_speaker(embedding, threshold)
            if match:
                matched_id, similarity = match
                speaker_mappings[temp_id] = matched_id
                click.echo(f"Matched {temp_id} to existing speaker {matched_id} (similarity: {similarity:.2f})")
            else:
                # Create new speaker entry
                new_id = f"SPEAKER_{len(existing_speakers) + new_speaker_count:02d}"
                success = db.add_speaker(
                    speaker_id=new_id,
                    voice_embedding=embedding
                )
                if success:
                    speaker_mappings[temp_id] = new_id
                    new_speaker_count += 1
                    click.echo(f"Created new speaker entry: {new_id}")
                else:
                    click.echo(f"Failed to create new speaker entry for {temp_id}")
        
        # Detect self-introductions and update speaker names
        click.echo("Detecting self-introductions...")
        name_mappings = detect_self_introductions(transcript, llm_client)
        
        # Update speaker names in database
        for speaker_id, name in name_mappings.items():
            if speaker_id in speaker_mappings:
                db_id = speaker_mappings[speaker_id]
                speaker = db.get_speaker(db_id)
                if speaker and not speaker.get('confirmed_name') and not speaker.get('inferred_name'):
                    db.update_speaker(
                        speaker_id=db_id,
                        inferred_name=name
                    )
                    click.echo(f"Updated speaker {db_id} with inferred name: {name}")
        
        # Update transcript with new speaker IDs and names
        updated = False
        for segment in transcript:
            temp_id = segment.get('speaker')
            if temp_id in speaker_mappings:
                new_id = speaker_mappings[temp_id]
                if new_id != temp_id:
                    # Get speaker info from database
                    speaker = db.get_speaker(new_id)
                    if speaker:
                        # Use confirmed name if available, otherwise inferred name, fallback to ID
                        name = speaker.get('confirmed_name') or speaker.get('inferred_name') or new_id
                        segment['speaker'] = name
                        updated = True
                        if name != new_id:
                            click.echo(f"Updated {temp_id} to {name}")
                    else:
                        segment['speaker'] = new_id
                        updated = True
        
        if updated:
            # Save updated transcript
            output_path = transcript_path + '.relabeled'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=4, ensure_ascii=False)
            click.echo(f"Updated transcript saved to {output_path}")
            click.echo(f"Created {new_speaker_count} new speaker entries")
        else:
            click.echo("No speaker labels were updated")
            
    except Exception as e:
        click.echo(f"Error processing transcript: {str(e)}") 