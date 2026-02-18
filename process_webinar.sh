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
STORY_TXT="$WEBINAR_DIR/story.txt"
SUMMARY_TXT="$WEBINAR_DIR/summary.txt"
QUIZ_TXT="$WEBINAR_DIR/quiz.txt"
VERIFY_REPORT="$WEBINAR_DIR/verify_report.md"
FIXED_TRANSCRIPT="$WEBINAR_DIR/transcript.fixed.json"
ANALYSIS_FILE="$WEBINAR_DIR/speaker_analysis.json"
LABELED_FILE="$WEBINAR_DIR/transcript.labeled.json"

# Prefer fixed transcript, then diarized, then ASR
if [[ -f "$FIXED_TRANSCRIPT" ]]; then
    INPUT_FILE="$FIXED_TRANSCRIPT"
elif [[ -f "$DIARIZED_FILE" ]]; then
    INPUT_FILE="$DIARIZED_FILE"
elif [[ -f "$ASR_FILE" ]]; then
    INPUT_FILE="$ASR_FILE"
else
    echo "Error: No transcript found (tried '$FIXED_TRANSCRIPT', '$DIARIZED_FILE' and '$ASR_FILE')." >&2
    exit 1
fi

echo "Input: $INPUT_FILE"

# Step 0: Transcript verification (optional, requires video file)
if [[ -f "$DIARIZED_FILE" ]]; then
    VIDEO_FILE=$(find "$WEBINAR_DIR" -maxdepth 1 \( -name "*.stripped.mp4" -o -name "*.mp4" \) | head -1)
    if [[ -n "${VIDEO_FILE:-}" ]]; then
        if [[ "$FORCE" == true ]] || [[ ! -f "$VERIFY_REPORT" ]]; then
            echo "[0/6] Verifying transcript quality..."
            webinar_processor transcript-verify "$DIARIZED_FILE" --media "$VIDEO_FILE" --report "$VERIFY_REPORT" || echo "  (verification completed with warnings)"
        else
            echo "[0/6] Verification report exists, skipping..."
        fi
        
        # Step 1: Fix transcript issues found during verification
        if [[ "$FORCE" == true ]] || [[ ! -f "$FIXED_TRANSCRIPT" ]]; then
            if [[ -f "$VERIFY_REPORT" ]]; then
                echo "[1/6] Fixing transcript issues..."
                webinar_processor transcript-fix "$DIARIZED_FILE" --media "$VIDEO_FILE" --report "$VERIFY_REPORT" --out "$FIXED_TRANSCRIPT" || echo "  (fix completed with warnings)"
            else
                echo "[1/6] No verification report, skipping transcript fix..."
            fi
        else
            echo "[1/6] Fixed transcript exists, skipping..."
        fi
    else
        echo "[0/6] No video file found, skipping transcript verification..."
        echo "[1/6] No video file found, skipping transcript fix..."
    fi
else
    echo "[0/6] No diarized transcript, skipping verification..."
    echo "[1/6] No diarized transcript, skipping fix..."
fi

# Step 2: Create story (outline + per-section, prompt-cached)
if [[ "$FORCE" == true ]] || [[ ! -f "$STORY_TXT" ]]; then
    echo "[2/6] Creating story from transcript..."
    webinar_processor storytell "$INPUT_FILE" --output-file "$STORY_TXT"
else
    echo "[2/6] Story exists, skipping..."
fi

# Step 3: Create summary (transcript cached from storytell)
if [[ "$FORCE" == true ]] || [[ ! -f "$SUMMARY_TXT" ]]; then
    echo "[3/6] Generating transcript summary..."
    webinar_processor summarize "$INPUT_FILE" --output-file "$SUMMARY_TXT"
else
    echo "[3/6] Summary exists, skipping..."
fi

# Step 4: Create quiz (transcript cached from storytell)
if [[ "$FORCE" == true ]] || [[ ! -f "$QUIZ_TXT" ]]; then
    echo "[4/6] Generating quiz..."
    webinar_processor quiz "$INPUT_FILE" --output-file "$QUIZ_TXT"
else
    echo "[4/6] Quiz exists, skipping..."
fi

# Step 5: Speaker identification (analyze + identify + apply)
# Requires diarized transcript and a video file with audio track.
# 'identify' needs reference speakers in the database — run 'speakers label'
# interactively on a few webinars first to build the reference library.
if [[ -f "$DIARIZED_FILE" ]]; then
    VIDEO_FILE=$(find "$WEBINAR_DIR" -maxdepth 1 \( -name "*.stripped.mp4" -o -name "*.mp4" \) | head -1)
    if [[ -n "${VIDEO_FILE:-}" ]]; then
        if [[ "$FORCE" == true ]] || [[ ! -f "$ANALYSIS_FILE" ]]; then
            echo "[5/6] Analyzing speakers..."
            webinar_processor speakers analyze "$WEBINAR_DIR"
        else
            echo "[5/6] Speaker analysis exists, skipping analyze..."
        fi

        # Identify only if we have reference speakers in the database
        if [[ -f "$ANALYSIS_FILE" ]]; then
            if [[ "$FORCE" == true ]] || [[ ! -f "$LABELED_FILE" ]]; then
                echo "[6/6] Identifying and applying speaker labels..."
                webinar_processor speakers identify "$WEBINAR_DIR" || echo "  (no reference speakers yet — run 'speakers label' first)"
                webinar_processor speakers apply "$WEBINAR_DIR" || true
            else
                echo "[6/6] Labeled transcript exists, skipping..."
            fi
        fi
    else
        echo "[5/6] No video file found, skipping speaker identification..."
    fi
else
    echo "[6/6] No diarized transcript, skipping speaker identification..."
fi

echo "Webinar processing completed successfully."
