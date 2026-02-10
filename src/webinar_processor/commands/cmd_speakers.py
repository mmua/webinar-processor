import click
import json
import uuid
from typing import Optional, List, Dict, Tuple
import os
from webinar_processor.llm import LLMClient, LLMError

import logging
logger = logging.getLogger(__name__)


def _generate_speaker_id() -> str:
    """Generate a UUID-based speaker ID."""
    return f"spk_{uuid.uuid4().hex[:8]}"


@click.group()
def speakers():
    """Manage speaker profiles and identifications."""
    pass


@speakers.command('list')
@click.option('--json-output', '--json', 'json_output', is_flag=True, help='Output as JSON')
def list_speakers(json_output: bool):
    """List all speakers in the database."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()
    all_speakers = db.get_all_speakers()

    if not all_speakers:
        click.echo("No speakers in database.")
        return

    if json_output:
        output = []
        for s in all_speakers:
            entry = {
                'speaker_id': s['speaker_id'],
                'confirmed_name': s.get('confirmed_name'),
                'inferred_name': s.get('inferred_name'),
                'gender': s.get('gender'),
                'first_detected': s.get('first_detected'),
                'num_samples': s.get('num_samples', 1),
                'appearance_count': s.get('appearance_count', 0),
                'confidence_score': s.get('confidence_score', 0.0),
                'notes': s.get('notes'),
            }
            output.append(entry)
        click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # Table output
    header = f"{'ID':<16} {'Name':<24} {'Gender':<8} {'First Seen':<12} {'Appearances':<13} {'Confidence'}"
    click.echo(header)
    click.echo("-" * len(header))

    for s in all_speakers:
        # Display name: confirmed (C) > inferred (I) > ID
        name = s.get('confirmed_name')
        name_tag = '(C)'
        if not name:
            name = s.get('inferred_name')
            name_tag = '(I)'
        if not name:
            name = s['speaker_id']
            name_tag = ''

        display_name = f"{name} {name_tag}".strip()
        if len(display_name) > 22:
            display_name = display_name[:19] + "..."

        gender = s.get('gender') or '-'
        first_seen = (s.get('first_detected') or '')[:10]
        appearances = s.get('appearance_count', 0)
        confidence = s.get('confidence_score', 0.0) or 0.0

        click.echo(f"{s['speaker_id']:<16} {display_name:<24} {gender:<8} {first_seen:<12} {appearances:<13} {confidence:.2f}")


@speakers.command()
@click.option('--name', required=True, help='Speaker name')
@click.option('--audio', required=True, type=click.Path(exists=True), help='Audio file for voice enrollment')
@click.option('--gender', type=click.Choice(['male', 'female', 'unknown']), help='Speaker gender')
@click.option('--min-duration', type=float, default=3.0, help='Minimum audio duration in seconds')
@click.option('--notes', help='Additional notes about the speaker')
def enroll(name: str, audio: str, gender: Optional[str], min_duration: float, notes: Optional[str]):
    """Enroll a new speaker from an audio file."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService

    click.echo(f"Extracting voice embedding from {audio}...")
    voice_service = VoiceEmbeddingService()
    embedding = voice_service.extract_single_speaker_embedding(audio)

    if embedding is None:
        click.echo("Error: Failed to extract voice embedding from audio.")
        return

    db = SpeakerDatabase()

    # Check for existing similar speakers
    match = db.find_matching_speaker(embedding, threshold=0.7)
    if match:
        matched_id, similarity = match
        matched_speaker = db.get_speaker(matched_id)
        matched_name = (matched_speaker.get('confirmed_name') or
                        matched_speaker.get('inferred_name') or
                        matched_id) if matched_speaker else matched_id
        click.echo(f"Warning: Similar speaker found: {matched_name} ({matched_id}, similarity: {similarity:.2f})")
        if not click.confirm("Continue enrolling as a new speaker?"):
            return

    speaker_id = _generate_speaker_id()
    success = db.add_speaker(
        speaker_id=speaker_id,
        voice_embedding=embedding,
        confirmed_name=name,
        gender=gender,
        confidence_score=1.0,
        notes=notes,
    )

    if success:
        click.echo(f"Enrolled speaker: {name} (ID: {speaker_id})")
    else:
        click.echo("Error: Failed to enroll speaker.")


@speakers.command()
@click.argument('speaker_id')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
def delete(speaker_id: str, yes: bool):
    """Delete a speaker from the database."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()

    speaker = db.get_speaker(speaker_id)
    if not speaker:
        click.echo(f"Speaker {speaker_id} not found.")
        return

    display_name = (speaker.get('confirmed_name') or
                    speaker.get('inferred_name') or
                    speaker_id)
    appearances = db.get_appearances(speaker_id)

    click.echo(f"Speaker: {display_name} ({speaker_id})")
    click.echo(f"Appearances: {len(appearances)}")
    if speaker.get('notes'):
        click.echo(f"Notes: {speaker['notes']}")

    if not yes:
        if not click.confirm("Delete this speaker and all their appearances?"):
            return

    if db.delete_speaker(speaker_id):
        click.echo(f"Deleted speaker {speaker_id}")
    else:
        click.echo(f"Error: Failed to delete speaker {speaker_id}")


@speakers.command()
@click.argument('source_id')
@click.argument('target_id')
def merge(source_id: str, target_id: str):
    """Merge source speaker into target speaker.

    Averages embeddings weighted by num_samples.
    Target inherits source's confirmed_name if target has none.
    Appearances are transferred, then source is deleted.
    """
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()

    source = db.get_speaker(source_id)
    target = db.get_speaker(target_id)

    if not source:
        click.echo(f"Source speaker {source_id} not found.")
        return
    if not target:
        click.echo(f"Target speaker {target_id} not found.")
        return

    source_name = source.get('confirmed_name') or source.get('inferred_name') or source_id
    target_name = target.get('confirmed_name') or target.get('inferred_name') or target_id

    click.echo(f"Merging: {source_name} ({source_id}) -> {target_name} ({target_id})")

    if db.merge_speakers(source_id, target_id):
        click.echo(f"Successfully merged {source_id} into {target_id}")
    else:
        click.echo("Error: Merge failed.")


@speakers.command()
@click.argument('speaker_id')
@click.option('--name', help='Manually confirm speaker name')
@click.option('--gender', type=click.Choice(['male', 'female', 'unknown']), help='Set speaker gender')
@click.option('--notes', help='Set notes for the speaker')
def update(speaker_id: str, name: Optional[str], gender: Optional[str], notes: Optional[str]):
    """Update speaker information."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()

    if not name and not gender and notes is None:
        click.echo("Please provide at least one update (--name, --gender, or --notes)")
        return

    success = db.update_speaker(
        speaker_id=speaker_id,
        confirmed_name=name,
        gender=gender,
        notes=notes,
    )

    if success:
        click.echo(f"Successfully updated speaker {speaker_id}")
    else:
        click.echo(f"Failed to update speaker {speaker_id}")


@speakers.command()
@click.argument('speaker_id')
def info(speaker_id: str):
    """Show detailed speaker information."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()
    speaker = db.get_speaker(speaker_id)

    if not speaker:
        click.echo(f"Speaker {speaker_id} not found")
        return

    click.echo(f"Speaker ID: {speaker['speaker_id']}")
    click.echo(f"Confirmed Name: {speaker.get('confirmed_name') or 'Not set'}")
    click.echo(f"Inferred Name: {speaker.get('inferred_name') or 'Not set'}")
    click.echo(f"Gender: {speaker.get('gender') or 'Unknown'}")
    click.echo(f"Samples: {speaker.get('num_samples', 1)}")
    click.echo(f"First Detected: {speaker.get('first_detected')}")
    click.echo(f"Last Updated: {speaker.get('last_updated')}")
    click.echo(f"Confidence Score: {speaker.get('confidence_score', 0.0)}")
    if speaker.get('notes'):
        click.echo(f"Notes: {speaker['notes']}")

    # Show appearances
    appearances = db.get_appearances(speaker_id)
    if appearances:
        click.echo(f"\nAppearances ({len(appearances)}):")
        for a in appearances:
            label = f" (as {a['original_label']})" if a.get('original_label') else ""
            click.echo(f"  - {a['transcript_path']}{label}  [{a['created_at']}]")
    else:
        click.echo("\nNo recorded appearances.")


def detect_self_introductions(transcript: List[Dict], llm_client: LLMClient) -> Dict[str, str]:
    """Detect self-introductions in the transcript and extract speaker names."""
    speaker_segments = {}

    for segment in transcript:
        speaker = segment['speaker']
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)

    name_mappings = {}
    for speaker_id, segments in speaker_segments.items():
        text = " ".join(seg['text'] for seg in segments[:10])

        try:
            name = llm_client.extract_speaker_name(text)
            if name:
                name_mappings[speaker_id] = name
        except LLMError as e:
            logger.warning(f"Failed to extract name for speaker {speaker_id}: {e}")
            continue

    return name_mappings


@speakers.command()
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
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        from webinar_processor.services.speaker_database import SpeakerDatabase
        from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService
        db = SpeakerDatabase()
        voice_service = VoiceEmbeddingService()
        llm_client = LLMClient()

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
                new_id = _generate_speaker_id()
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
        name_mappings = detect_self_introductions(transcript, llm_client)

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
            # Determine output path
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


# Import and register new speaker commands
from .cmd_speaker_analyze import analyze as speaker_analyze
from .cmd_speaker_label import label as speaker_label
from .cmd_speaker_identify import identify as speaker_identify
from .cmd_speaker_apply import apply as speaker_apply

speakers.add_command(speaker_analyze)
speakers.add_command(speaker_label)
speakers.add_command(speaker_identify)
speakers.add_command(speaker_apply)
