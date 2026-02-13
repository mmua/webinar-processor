"""CRUD subcommands for the speakers group: list, enroll, delete, merge, update, info."""

import click
import json
import uuid
from typing import Optional

import logging
logger = logging.getLogger(__name__)


def generate_speaker_id() -> str:
    """Generate a UUID-based speaker ID."""
    return f"spk_{uuid.uuid4().hex[:8]}"


@click.command('list')
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
    header = f"{'ID':<16} {'Name':<32} {'First Seen':<12} {'Appearances':<13} {'Confidence'}"
    click.echo(header)
    click.echo("-" * len(header))

    for s in all_speakers:
        name = s.get('confirmed_name')
        name_tag = '(C)'
        if not name:
            name = s.get('inferred_name')
            name_tag = '(I)'
        if not name:
            name = s['speaker_id']
            name_tag = ''

        display_name = f"{name} {name_tag}".strip()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        first_seen = (s.get('first_detected') or '')[:10]
        appearances = s.get('appearance_count', 0)
        confidence = s.get('confidence_score', 0.0) or 0.0

        click.echo(f"{s['speaker_id']:<16} {display_name:<32} {first_seen:<12} {appearances:<13} {confidence:.2f}")


@click.command()
@click.option('--name', required=True, help='Speaker name')
@click.option('--audio', required=True, type=click.Path(exists=True), help='Audio file for voice enrollment')
@click.option('--min-duration', type=float, default=3.0, help='Minimum audio duration in seconds')
@click.option('--notes', help='Additional notes about the speaker')
def enroll(name: str, audio: str, min_duration: float, notes: Optional[str]):
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

    speaker_id = generate_speaker_id()
    success = db.add_speaker(
        speaker_id=speaker_id,
        voice_embedding=embedding,
        confirmed_name=name,
        confidence_score=1.0,
        notes=notes,
    )

    if success:
        click.echo(f"Enrolled speaker: {name} (ID: {speaker_id})")
    else:
        click.echo("Error: Failed to enroll speaker.")


@click.command()
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


@click.command()
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


@click.command()
@click.argument('speaker_id')
@click.option('--name', help='Manually confirm speaker name')
@click.option('--notes', help='Set notes for the speaker')
def update(speaker_id: str, name: Optional[str], notes: Optional[str]):
    """Update speaker information."""
    from webinar_processor.services.speaker_database import SpeakerDatabase
    db = SpeakerDatabase()

    if not name and notes is None:
        click.echo("Please provide at least one update (--name or --notes)")
        return

    success = db.update_speaker(
        speaker_id=speaker_id,
        confirmed_name=name,
        notes=notes,
    )

    if success:
        click.echo(f"Successfully updated speaker {speaker_id}")
    else:
        click.echo(f"Failed to update speaker {speaker_id}")


@click.command()
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
    click.echo(f"Samples: {speaker.get('num_samples', 1)}")
    click.echo(f"First Detected: {speaker.get('first_detected')}")
    click.echo(f"Last Updated: {speaker.get('last_updated')}")
    click.echo(f"Confidence Score: {speaker.get('confidence_score', 0.0)}")
    if speaker.get('notes'):
        click.echo(f"Notes: {speaker['notes']}")

    appearances = db.get_appearances(speaker_id)
    if appearances:
        click.echo(f"\nAppearances ({len(appearances)}):")
        for a in appearances:
            label = f" (as {a['original_label']})" if a.get('original_label') else ""
            click.echo(f"  - {a['transcript_path']}{label}  [{a['created_at']}]")
    else:
        click.echo("\nNo recorded appearances.")
