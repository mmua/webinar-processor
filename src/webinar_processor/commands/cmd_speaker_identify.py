import click
import json
import numpy as np
import os
from webinar_processor.services.speaker_database import SpeakerDatabase
from webinar_processor.commands.cmd_speaker_analyze import decode_embedding


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def match_speaker_voting(target_samples: list, reference_speakers: list, threshold: float = 0.7,
                         min_matches: int = 2) -> tuple:
    """Match a speaker using voting across multiple samples.
    
    Args:
        target_samples: List of sample dicts with 'embedding' key
        reference_speakers: List of speaker dicts with 'voice_embedding' key
        threshold: Minimum similarity to count as a match
        min_matches: Minimum number of matching samples required
        
    Returns:
        Tuple of (speaker_name, confidence, speaker_id) or (None, 0.0, None) if no match
    """
    if not target_samples or not reference_speakers:
        return None, 0.0, None

    # Decode target embeddings
    target_embs = []
    for sample in target_samples:
        try:
            emb = decode_embedding(sample['embedding'])
            target_embs.append(emb)
        except Exception:
            continue

    if not target_embs:
        return None, 0.0, None
    
    # For each reference speaker, count matches
    best_match_name = None
    best_match_id = None
    best_matches = 0
    best_confidence = 0.0
    
    for ref_speaker in reference_speakers:
        ref_emb = ref_speaker.get('voice_embedding')
        if ref_emb is None:
            continue
        
        matches = 0
        total_similarity = 0.0
        
        # Compare each target sample against reference
        for target_emb in target_embs:
            sim = calculate_similarity(target_emb, ref_emb)
            if sim >= threshold:
                matches += 1
                total_similarity += sim
        
        if matches >= min_matches and matches > best_matches:
            best_matches = matches
            best_match_name = ref_speaker.get('confirmed_name') or ref_speaker.get('inferred_name') or ref_speaker['speaker_id']
            best_match_id = ref_speaker['speaker_id']
            best_confidence = total_similarity / matches if matches > 0 else 0.0
    
    return best_match_name, best_confidence, best_match_id


@click.command('identify')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--threshold', type=float, default=0.7, help='Similarity threshold')
@click.option('--min-matches', type=int, default=2, help='Minimum matching samples required')
@click.option('--dry-run', is_flag=True, help='Show matches without updating')
def identify(directory: str, threshold: float, min_matches: int, dry_run: bool):
    """Identify speakers in a webinar using reference library.
    
    Matches unlabeled speakers against confirmed speakers in the database
    (those manually labeled with 'speakers label'). Uses voting: requires 
    multiple sample matches. Results saved to speaker_analysis.json.
    """
    directory = os.path.abspath(directory)
    analysis_path = os.path.join(directory, 'speaker_analysis.json')
    
    click.echo(f"Identifying speakers in: {directory}")
    
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
    
    speakers = analysis.get('speakers', {})
    if not speakers:
        click.echo("No speakers found in analysis")
        return
    
    # Get reference speakers from database
    db = SpeakerDatabase()
    reference_speakers = db.get_all_speakers()
    
    # Filter to speakers with embeddings (confirmed/labelled)
    reference_speakers = [s for s in reference_speakers if s.get('voice_embedding') is not None]
    
    if not reference_speakers:
        click.echo("Error: No reference speakers found in database.", err=True)
        click.echo("Run 'webinar_processor speakers label <directory>' first to create reference speakers.", err=True)
        raise click.Abort()
    
    click.echo(f"Using {len(reference_speakers)} reference speakers from database")
    for ref in reference_speakers:
        name = ref.get('confirmed_name') or ref.get('inferred_name') or ref['speaker_id']
        click.echo(f"  - {name}")
    
    # Get unlabeled speakers
    unlabeled = [
        (temp_id, data) for temp_id, data in speakers.items()
        if data.get('labeled_name') is None and data.get('identified_name') is None
    ]
    
    if not unlabeled:
        click.echo("\nNo unlabeled speakers to identify")
        return
    
    click.echo(f"\nFound {len(unlabeled)} speakers to identify")
    click.echo(f"Threshold: {threshold}, Min matches: {min_matches}")
    click.echo("Note: Will only use samples marked as 'clean' (single speaker)")
    click.echo("-" * 60)
    
    identified_count = 0
    uncertain_count = 0
    
    for temp_id, speaker_data in unlabeled:
        duration = speaker_data['total_duration']
        all_samples = speaker_data.get('samples', [])
        
        # Use only clean samples for matching
        samples = [s for s in all_samples if s.get('is_clean', False)]
        
        if not samples and all_samples:
            click.echo(f"\n  {temp_id}: No clean samples marked. Use 'label' command first to mark clean samples.")
            uncertain_count += 1
            continue
        
        if not samples:
            uncertain_count += 1
            continue
        
        # Match against reference speakers
        matched_name, confidence, matched_id = match_speaker_voting(
            samples, reference_speakers, threshold, min_matches
        )
        
        if matched_name and confidence > 0:
            identified_count += 1
            
            if not dry_run:
                speaker_data['identified_name'] = matched_name
                speaker_data['identified_speaker_id'] = matched_id
                speaker_data['identification_confidence'] = confidence
            
            status = "✓"
            info = f"{matched_name} (confidence: {confidence:.2f})"
        else:
            uncertain_count += 1
            status = "?"
            info = "uncertain"
        
        click.echo(f"  {status} {temp_id} ({duration:.1f}s) -> {info}")
    
    # Save results
    if not dry_run:
        try:
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            click.echo(f"\nResults saved to: {analysis_path}")
        except Exception as e:
            click.echo(f"Warning: Failed to save results: {e}", err=True)
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Identified: {identified_count}, Uncertain: {uncertain_count}")
    
    if dry_run:
        click.echo("\n(Dry run - no changes made)")
    else:
        click.echo(f"\nNext step: Run 'webinar_processor speakers apply {directory}' to update transcript")
