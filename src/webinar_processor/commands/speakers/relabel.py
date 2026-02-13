"""Re-label speakers in a transcript using voice embeddings."""

import click
import json
import os
from typing import List, Dict, Optional

from webinar_processor.llm import LLMError
from webinar_processor.services.speaker_name_extractor import extract_speaker_name

import logging
logger = logging.getLogger(__name__)


def detect_self_introductions(transcript: List[Dict]) -> Dict[str, str]:
    """Detect self-introductions in the transcript and extract speaker names."""
    speaker_segments: Dict[str, list] = {}

    for segment in transcript:
        speaker = segment['speaker']
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)

    name_mappings = {}
    for speaker_id, segments in speaker_segments.items():
        text = " ".join(seg['text'] for seg in segments[:10])

        try:
            name = extract_speaker_name(text)
            if name:
                name_mappings[speaker_id] = name
        except LLMError as e:
            logger.warning(f"Failed to extract name for speaker {speaker_id}: {e}")
            continue

    return name_mappings


@click.command()
@click.argument('transcript_path')
@click.argument('audio_path')
@click.option('--output', '-o', 'output_path', help='Output file path for relabeled transcript')
@click.option('--in-place', is_flag=True, help='Overwrite the original transcript file')
@click.option('--threshold', type=float, default=0.7, help='Similarity threshold for speaker matching')
@click.option('--min-duration', type=float, default=3.0, help='Minimum duration for voice segments')
def relabel(transcript_path: str, audio_path: str, output_path: Optional[str],
            in_place: bool, threshold: float, min_duration: float):
    """Re-label speakers in a transcript using voice embeddings.

    Extracts voice embeddings, matches against the speaker database,
    creates new entries for unknown speakers with UUID-based IDs,
    and updates the transcript with identified speaker names.
    """
    from webinar_processor.services.speaker_database import SpeakerDatabase
    from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService
    from .crud import generate_speaker_id

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        db = SpeakerDatabase()
        voice_service = VoiceEmbeddingService()

        existing_speakers = db.get_all_speakers()
        click.echo(f"Found {len(existing_speakers)} existing speakers in database")

        click.echo("Extracting voice embeddings...")
        mean_embeddings = voice_service.process_audio_file(
            audio_path,
            transcript,
            min_duration
        )

        if not mean_embeddings:
            click.echo("No valid voice embeddings could be extracted")
            return

        speaker_mappings = {}  # Maps temp IDs to database IDs
        new_speaker_count = 0

        click.echo("Matching speakers against database...")
        for temp_id, embedding in mean_embeddings.items():
            match = db.find_matching_speaker(embedding, threshold)
            if match:
                matched_id, similarity = match
                speaker_mappings[temp_id] = matched_id
                click.echo(f"Matched {temp_id} to existing speaker {matched_id} (similarity: {similarity:.2f})")

                # Running-average update of stored embedding
                matched_speaker = db.get_speaker(matched_id)
                if matched_speaker and matched_speaker.get('voice_embedding') is not None:
                    n_samples = matched_speaker.get('num_samples', 1) or 1
                    updated_embedding = VoiceEmbeddingService.update_mean_embedding(
                        matched_speaker['voice_embedding'], embedding, n_samples
                    )
                    db.update_speaker(
                        speaker_id=matched_id,
                        voice_embedding=updated_embedding,
                        num_samples=n_samples + 1,
                    )
            else:
                # Create new speaker with UUID-based ID
                new_id = generate_speaker_id()
                success = db.add_speaker(
                    speaker_id=new_id,
                    voice_embedding=embedding,
                )
                if success:
                    speaker_mappings[temp_id] = new_id
                    new_speaker_count += 1
                    click.echo(f"Created new speaker entry: {new_id}")
                else:
                    click.echo(f"Failed to create new speaker entry for {temp_id}")

        # Track appearances
        transcript_abs = os.path.abspath(transcript_path)
        audio_abs = os.path.abspath(audio_path)
        for temp_id, db_id in speaker_mappings.items():
            db.add_appearance(db_id, transcript_abs, audio_abs, temp_id)

        # Detect self-introductions and update speaker names
        click.echo("Detecting self-introductions...")
        name_mappings = detect_self_introductions(transcript)

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
                    speaker = db.get_speaker(new_id)
                    if speaker:
                        name = speaker.get('confirmed_name') or speaker.get('inferred_name') or new_id
                        segment['speaker'] = name
                        updated = True
                        if name != new_id:
                            click.echo(f"Updated {temp_id} to {name}")
                    else:
                        segment['speaker'] = new_id
                        updated = True

        if updated:
            if in_place:
                dest = transcript_path
            elif output_path:
                dest = output_path
            else:
                dest = transcript_path + '.relabeled'

            with open(dest, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=4, ensure_ascii=False)
            click.echo(f"Updated transcript saved to {dest}")
            click.echo(f"Created {new_speaker_count} new speaker entries")
        else:
            click.echo("No speaker labels were updated")

    except Exception as e:
        click.echo(f"Error processing transcript: {str(e)}")
