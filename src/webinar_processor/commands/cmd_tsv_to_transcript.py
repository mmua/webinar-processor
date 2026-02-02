import os
import json
import click


@click.command()
@click.argument('tsv_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('transcript_path', type=click.Path(exists=False), default='', nargs=1)
def tsv_to_transcript(tsv_file: click.File,  transcript_path: str):
    """
    Transforms tab separated transcript into transcript and asr file.

    """
    utterances = []
    full_text = ""
    for line in tsv_file:
        try:
            start, end, text = line.strip().split("\t")
            if not text:
                continue
            utterances.append({"speaker": "SPEAKER_00", "start": float(start), "end": float(end), "text": text})
        except ValueError:
            continue

        full_text += text + "\n"

    asr = {
        "text": full_text,
        "segments": utterances
    }

    if not transcript_path:
        transcript_path = os.path.join(os.path.dirname(tsv_file.name), "transcript.json")


    with open(transcript_path, "w", encoding="utf-8") as of:
        json.dump(utterances, of, ensure_ascii=False, indent=4)
    with open(transcript_path + ".asr", "w", encoding="utf-8") as of:
        json.dump(asr, of, ensure_ascii=False, indent=4)
