#!/usr/bin/env bash

set -euo pipefail

FORCE=false

# Usage function to display help
usage() {
    echo "Usage: $0 [--force] /path/to/webinar_directory"
    echo ""
    echo "Options:"
    echo "  --force    Regenerate all files even if they exist"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Error: Unknown option '$1'" >&2
            usage
            ;;
        *)
            WEBINAR_DIR="$1"
            shift
            ;;
    esac
done

# Check if the webinar directory is provided
if [[ -z "${WEBINAR_DIR:-}" ]]; then
    usage
fi

# Verify that the webinar directory exists
if [[ ! -d "$WEBINAR_DIR" ]]; then
    echo "Error: Directory '$WEBINAR_DIR' does not exist." >&2
    exit 1
fi

# Check if webinar_processor is installed
if ! command -v webinar_processor >/dev/null 2>&1; then
    echo "Error: 'webinar_processor' command not found in PATH." >&2
    exit 1
fi

# Define file paths
DIARIZED_FILE="$WEBINAR_DIR/transcript.json"
ASR_FILE="$WEBINAR_DIR/transcript.json.asr"
TRANSCRIPT_TXT="$WEBINAR_DIR/transcript.txt"
STORY_TXT="$WEBINAR_DIR/story.txt"
SUMMARY_TXT="$WEBINAR_DIR/summary.txt"
QUIZ_TXT="$WEBINAR_DIR/quiz.txt"
ANALYSIS_FILE="$WEBINAR_DIR/speaker_analysis.json"
LABELED_FILE="$WEBINAR_DIR/transcript.labeled.json"

# Prefer diarized transcript (speaker labels, better formatting)
if [[ -f "$DIARIZED_FILE" ]]; then
    INPUT_FILE="$DIARIZED_FILE"
elif [[ -f "$ASR_FILE" ]]; then
    INPUT_FILE="$ASR_FILE"
else
    echo "Error: No transcript found (tried '$DIARIZED_FILE' and '$ASR_FILE')." >&2
    exit 1
fi

echo "Input: $INPUT_FILE"

# Step 1: Create raw transcript text (reference)
if [[ -f "$ASR_FILE" ]]; then
    if [[ "$FORCE" == true ]] || [[ ! -f "$TRANSCRIPT_TXT" ]]; then
        echo "[1/5] Creating raw transcript text..."
        webinar_processor raw-text "$ASR_FILE" --output-file "$TRANSCRIPT_TXT"
    else
        echo "[1/5] Transcript exists, skipping..."
    fi
else
    echo "[1/5] No ASR file, skipping raw text..."
fi

# Step 2: Create story (outline + per-section, prompt-cached)
if [[ "$FORCE" == true ]] || [[ ! -f "$STORY_TXT" ]]; then
    echo "[2/5] Creating story from transcript..."
    webinar_processor storytell "$INPUT_FILE" --output-file "$STORY_TXT"
else
    echo "[2/5] Story exists, skipping..."
fi

# Step 3: Create summary (transcript cached from storytell)
if [[ "$FORCE" == true ]] || [[ ! -f "$SUMMARY_TXT" ]]; then
    echo "[3/5] Generating transcript summary..."
    webinar_processor summarize "$INPUT_FILE" --output-file "$SUMMARY_TXT"
else
    echo "[3/5] Summary exists, skipping..."
fi

# Step 4: Create quiz (transcript cached from storytell)
if [[ "$FORCE" == true ]] || [[ ! -f "$QUIZ_TXT" ]]; then
    echo "[4/5] Generating quiz..."
    webinar_processor quiz "$INPUT_FILE" --output-file "$QUIZ_TXT"
else
    echo "[4/5] Quiz exists, skipping..."
fi

# Step 5: Speaker identification (analyze + identify + apply)
# Requires diarized transcript and a video file with audio track.
# 'identify' needs reference speakers in the database — run 'speakers label'
# interactively on a few webinars first to build the reference library.
if [[ -f "$DIARIZED_FILE" ]]; then
    VIDEO_FILE=$(find "$WEBINAR_DIR" -maxdepth 1 \( -name "*.stripped.mp4" -o -name "*.mp4" \) | head -1)
    if [[ -n "${VIDEO_FILE:-}" ]]; then
        if [[ "$FORCE" == true ]] || [[ ! -f "$ANALYSIS_FILE" ]]; then
            echo "[5/5] Analyzing speakers..."
            webinar_processor speakers analyze "$WEBINAR_DIR"
        else
            echo "[5/5] Speaker analysis exists, skipping analyze..."
        fi

        # Identify only if we have reference speakers in the database
        if [[ -f "$ANALYSIS_FILE" ]]; then
            if [[ "$FORCE" == true ]] || [[ ! -f "$LABELED_FILE" ]]; then
                echo "[5/5] Identifying and applying speaker labels..."
                webinar_processor speakers identify "$WEBINAR_DIR" || echo "  (no reference speakers yet — run 'speakers label' first)"
                webinar_processor speakers apply "$WEBINAR_DIR" || true
            else
                echo "[5/5] Labeled transcript exists, skipping..."
            fi
        fi
    else
        echo "[5/5] No video file found, skipping speaker identification..."
    fi
else
    echo "[5/5] No diarized transcript, skipping speaker identification..."
fi

echo "Webinar processing completed successfully."
