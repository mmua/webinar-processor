import click
import json
import os


@click.command('apply')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--output-suffix', default='.labeled', help='Suffix for output file')
def apply(directory: str, output_suffix: str):
    """Apply speaker labels to transcript and create labeled version.
    
    Reads transcript.json and speaker_analysis.json, then creates
    transcript.labeled.json with real speaker names instead of temporary IDs.
    """
    directory = os.path.abspath(directory)
    
    click.echo(f"Applying labels to: {directory}")
    
    # Check for transcript
    transcript_path = os.path.join(directory, "transcript.json")
    if not os.path.exists(transcript_path):
        click.echo(f"Error: transcript.json not found in {directory}", err=True)
        raise click.Abort()
    
    # Check for analysis
    analysis_path = os.path.join(directory, 'speaker_analysis.json')
    if not os.path.exists(analysis_path):
        click.echo(f"Error: speaker_analysis.json not found in {directory}", err=True)
        click.echo("Run 'webinar_processor speakers analyze <directory>' first.", err=True)
        raise click.Abort()
    
    # Read transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        click.echo(f"Error reading transcript: {e}", err=True)
        raise click.Abort()
    
    if not isinstance(transcript, list):
        click.echo("Error: Transcript must be a list of segments", err=True)
        raise click.Abort()
    
    # Read analysis
    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
    except Exception as e:
        click.echo(f"Error reading analysis: {e}", err=True)
        raise click.Abort()
    
    speakers = analysis.get('speakers', {})
    
    # Build mapping: temp_id -> name
    speaker_mapping = {}
    unlabeled_count = 0
    
    for temp_id, data in speakers.items():
        # Priority: labeled_name > identified_name > temp_id
        name = data.get('labeled_name') or data.get('identified_name')
        if name:
            speaker_mapping[temp_id] = name
        else:
            speaker_mapping[temp_id] = temp_id
            unlabeled_count += 1
    
    click.echo("\nSpeaker mapping:")
    for temp_id, name in sorted(speaker_mapping.items()):
        if temp_id != name:
            status = "labeled" if temp_id in speakers and speakers[temp_id].get('labeled_name') else "identified"
            click.echo(f"  {temp_id} -> {name} ({status})")
        else:
            click.echo(f"  {temp_id} -> (unlabeled)")
    
    if unlabeled_count > 0:
        click.echo(f"\nWarning: {unlabeled_count} speakers are unlabeled", err=True)
    
    # Apply mapping to transcript
    updated_count = 0
    for segment in transcript:
        temp_id = segment.get('speaker')
        if temp_id in speaker_mapping:
            new_name = speaker_mapping[temp_id]
            if new_name != temp_id:
                segment['speaker'] = new_name
                updated_count += 1
    
    # Write output
    output_path = transcript_path.replace('.json', f'{output_suffix}.json')
    if output_path == transcript_path:
        output_path = transcript_path + '.labeled'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
    except Exception as e:
        click.echo(f"Error writing output: {e}", err=True)
        raise click.Abort()
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Updated {updated_count} segments")
    click.echo(f"Created: {output_path}")
    
    if unlabeled_count > 0:
        click.echo("\nTo label remaining speakers:")
        click.echo(f"  webinar_processor speakers label {directory}")
    
    click.echo("\nTo process more webinars:")
    click.echo("  webinar_processor speakers analyze <directory>")
    click.echo("  webinar_processor speakers label <directory>  # for reference speakers")
    click.echo("  webinar_processor speakers identify <directory>")
    click.echo("  webinar_processor speakers apply <directory>")
