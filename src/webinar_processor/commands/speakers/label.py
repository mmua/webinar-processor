import click
import json
import os

from webinar_processor.services.speaker_database import SpeakerDatabase
from webinar_processor.utils.embedding_codec import decode_embedding


def format_timestamp(seconds: float) -> str:
    """Convert seconds to compact H:MM:SS.ms format (no leading zeros)."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}.{ms:02d}"
    elif mins > 0:
        return f"{mins}:{secs:02d}.{ms:02d}"
    else:
        return f"{secs}.{ms:02d}"


@click.command('label')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
def label(directory):
    """Interactively label speakers in a webinar directory.
    
    Reads speaker_analysis.json, shows speakers sorted by duration,
    and allows you to label them. Saves results back to the JSON file
    and adds confirmed speakers to the database.
    """
    directory = os.path.abspath(directory)
    analysis_path = os.path.join(directory, 'speaker_analysis.json')
    
    # Check for analysis file
    if not os.path.exists(analysis_path):
        click.echo(f"Error: speaker_analysis.json not found in {directory}", err=True)
        click.echo("Run 'webinar_processor speakers analyze <directory>' first.", err=True)
        raise click.Abort()
    
    # Read analysis
    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
    except Exception as e:
        click.echo(f"Error reading analysis: {e}", err=True)
        raise click.Abort()
    
    video_filename = analysis['video_filename']
    video_path = os.path.join(directory, video_filename)
    speakers = analysis.get('speakers', {})
    
    if not speakers:
        click.echo("No speakers found in analysis")
        return
    
    # Get unlabeled speakers sorted by duration
    unlabeled = [
        (temp_id, data) for temp_id, data in speakers.items()
        if data.get('labeled_name') is None
    ]
    unlabeled.sort(key=lambda x: x[1]['total_duration'], reverse=True)
    
    if not unlabeled:
        click.echo("All speakers are already labeled!")
        return
    
    click.echo(f"Found {len(unlabeled)} unlabeled speakers in {directory}")
    click.echo("Showing speakers sorted by duration (longest first)")
    click.echo("-" * 60)
    
    db = SpeakerDatabase()
    labeled_count = 0
    skipped_count = 0
    
    for i, (temp_id, speaker_data) in enumerate(unlabeled, 1):
        duration = speaker_data['total_duration']
        segment_count = speaker_data['segment_count']
        samples = speaker_data.get('samples', [])
        
        if not samples:
            click.echo(f"\n[{i}/{len(unlabeled)}] {temp_id} - No samples, skipping")
            skipped_count += 1
            continue
        
        # Display speaker info
        click.echo(f"\n[{i}/{len(unlabeled)}] {temp_id}")
        click.echo(f"Duration: {duration:.1f}s ({segment_count} segments)")
        click.echo(f"Samples: {len(samples)}")
        
        # Show each sample with audio extraction command
        clean_samples = []
        for j, sample in enumerate(samples, 1):
            start = sample['start_time']
            end = sample['end_time']
            sample_duration = end - start
            is_clean = sample.get('is_clean', False)
            
            click.echo(f"\n  Sample {j}/{len(samples)} {'[CLEAN]' if is_clean else ''}:")
            click.echo(f"  Time: {format_timestamp(start)} - {format_timestamp(end)} ({sample_duration:.1f}s)")
            click.echo(f"  Text: {sample['text_preview'][:100]}...")
            
            # Print ffmpeg command
            click.echo(f"  # ffmpeg -ss {start:.1f} -t {sample_duration:.1f} \\")
            click.echo(f"  #        -i '{video_path}' -vn -acodec copy /tmp/sample{j}.wav")
            click.echo(f"  # ffplay /tmp/sample{j}.wav")
        
        # Ask which samples are clean
        click.echo("\n  Which samples are clean (single speaker only)?")
        click.echo("  Enter sample numbers separated by commas (e.g., '1,2'), or:")
        click.echo("    'all' = all samples are clean")
        click.echo("    'none' = no clean samples (skip this speaker)")
        click.echo("    's' = skip this speaker for now")
        click.echo("    'q' = quit")
        
        clean_input = click.prompt("  Clean samples", default="1", show_default=False).strip()
        
        if clean_input.lower() == 'q':
            click.echo(f"\nQuitting. Labeled {labeled_count}, skipped {skipped_count}")
            _save_analysis(analysis_path, analysis)
            return
        elif clean_input.lower() == 's' or clean_input.lower() == 'none':
            skipped_count += 1
            click.echo("  Skipped")
            continue
        elif clean_input.lower() == 'all':
            clean_indices = list(range(1, len(samples) + 1))
        else:
            try:
                clean_indices = [int(x.strip()) for x in clean_input.split(',')]
                clean_indices = [i for i in clean_indices if 1 <= i <= len(samples)]
            except ValueError:
                click.echo("  Invalid input, skipping", err=True)
                skipped_count += 1
                continue
        
        if not clean_indices:
            click.echo("  No valid samples selected, skipping")
            skipped_count += 1
            continue
        
        # Mark clean samples
        for idx in clean_indices:
            samples[idx - 1]['is_clean'] = True
        clean_samples = [samples[idx - 1] for idx in clean_indices]
        
        # Get speaker name
        click.echo(f"\n  Selected {len(clean_samples)} clean sample(s)")
        name = click.prompt("  Speaker name").strip()
        
        if not name:
            click.echo("  No name entered, skipping")
            skipped_count += 1
            continue
        
        # Label the speaker
        speaker_data['labeled_name'] = name
        
        # Add to database using ONLY clean samples
        speaker_id = _add_speaker_to_db(db, name, clean_samples)
        if speaker_id:
            speaker_data['speaker_id'] = speaker_id
            labeled_count += 1
            click.echo(f"  ✓ Labeled as '{name}' using {len(clean_samples)} clean sample(s) (ID: {speaker_id})")
        else:
            click.echo("  ✗ Failed to add to database", err=True)
    
    # Save analysis back to file
    _save_analysis(analysis_path, analysis)
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Done! Labeled {labeled_count}, skipped {skipped_count}")
    click.echo(f"Analysis saved to: {analysis_path}")
    
    remaining = len([s for s in speakers.values() if s.get('labeled_name') is None])
    if remaining > 0:
        click.echo(f"\n{remaining} speakers still unlabeled")
    else:
        click.echo("\nAll speakers labeled!")
        click.echo("Run 'webinar_processor speakers identify <other_directory>' to identify speakers in other webinars")


def _add_speaker_to_db(db: SpeakerDatabase, name: str, samples: list):
    """Add a labeled speaker to the database using all clean sample embeddings."""
    import numpy as np
    from .crud import generate_speaker_id

    try:
        if not samples:
            return None

        # Average all clean sample embeddings for a robust representation
        embeddings = []
        for sample in samples:
            try:
                embeddings.append(decode_embedding(sample['embedding']))
            except Exception:
                continue

        if not embeddings:
            return None

        embedding = np.mean(embeddings, axis=0).astype(np.float32)
        
        # Check if speaker with this name already exists
        all_speakers = db.get_all_speakers()
        for speaker in all_speakers:
            if speaker.get('confirmed_name') == name:
                # Update existing speaker with weighted average embedding
                existing_emb = speaker.get('voice_embedding')
                if existing_emb is not None:
                    n = speaker.get('num_samples', 1) or 1
                    new_emb = (existing_emb * n + embedding * len(embeddings)) / (n + len(embeddings))
                    db.update_speaker(
                        speaker['speaker_id'],
                        voice_embedding=new_emb.astype(np.float32),
                        num_samples=n + len(embeddings)
                    )
                return speaker['speaker_id']

        # Create new speaker
        speaker_id = generate_speaker_id()
        success = db.add_speaker(
            speaker_id=speaker_id,
            voice_embedding=embedding,
            confirmed_name=name,
            confidence_score=1.0,
            num_samples=len(embeddings),
        )
        
        if success:
            return speaker_id
        else:
            return None
            
    except Exception as e:
        click.echo(f"Error adding speaker to database: {e}", err=True)
        return None


def _save_analysis(analysis_path: str, analysis: dict):
    """Save analysis back to JSON file."""
    try:
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    except Exception as e:
        click.echo(f"Warning: Failed to save analysis: {e}", err=True)
