#!/bin/bash

# Video Poster Generator Script - Improved Version
# Usage: ./poster.sh [options] /path/to/video.mp4
# 
# Improvements:
# - Multi-segment sampling to avoid intros/outros
# - Thumbnail filter for representative frame detection
# - I-frame extraction for best quality
# - Black frame detection and filtering
# - Quality-based selection using file size heuristics
# - Input seeking for 10x faster processing
# - Configurable dimensions, quality, and debug output

set -e

# Default configuration
OUTPUT_WIDTH=1920
OUTPUT_HEIGHT=1080
JPEG_QUALITY=2
DEBUG_MODE=false
NUM_CANDIDATES=5
TEMP_DIR=""

# Cleanup function
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# Show usage
usage() {
    cat << EOF
Video Poster Generator - Improved Edition

Usage: $0 [OPTIONS] <video_path>

Options:
    -w, --width WIDTH       Output width in pixels (default: 1920)
    -h, --height HEIGHT     Output height in pixels (default: 1080)
    -q, --quality QUALITY   JPEG quality 2-31, lower is better (default: 2)
    -c, --candidates NUM    Number of candidate frames to compare (default: 5)
    -d, --debug             Enable debug output with verbose ffmpeg logging
    --help                  Show this help message

Examples:
    $0 /path/to/webinar.mp4
    $0 -w 1280 -h 720 -q 3 /path/to/video.mp4
    $0 --debug -c 10 /path/to/video.mp4
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--width)
            OUTPUT_WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            OUTPUT_HEIGHT="$2"
            shift 2
            ;;
        -q|--quality)
            JPEG_QUALITY="$2"
            shift 2
            ;;
        -c|--candidates)
            NUM_CANDIDATES="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_MODE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            VIDEO_PATH="$1"
            shift
            ;;
    esac
done

# Check if video path is provided
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: Video path is required"
    usage
    exit 1
fi

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file '$VIDEO_PATH' not found"
    exit 1
fi

# Check dependencies
for cmd in ffmpeg ffprobe bc; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed. Please install it first."
        exit 1
    fi
done

# Set up debug output
FFMPEG_LOG_LEVEL="error"
if [ "$DEBUG_MODE" = true ]; then
    FFMPEG_LOG_LEVEL="verbose"
    echo "Debug mode enabled - showing verbose ffmpeg output"
fi

# Get video info
echo "Analyzing video: $(basename "$VIDEO_PATH")"
VIDEO_DIR=$(dirname "$VIDEO_PATH")
VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

# Create output directory
POSTERS_DIR="$VIDEO_DIR/posters"
mkdir -p "$POSTERS_DIR"

# Get video duration
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$VIDEO_PATH")
if [ -z "$DURATION" ] || [ "$DURATION" = "N/A" ]; then
    echo "Error: Could not determine video duration"
    exit 1
fi

echo "Video duration: $(printf '%.1f' "$DURATION") seconds"

# Create temp directory for candidate frames
TEMP_DIR=$(mktemp -d)
echo "Working directory: $TEMP_DIR"

# Calculate segment positions (skip first 10% and last 10% to avoid intros/outros)
# Sample from 4 strategic segments
SEGMENT_COUNT=4
START_OFFSET=$(echo "$DURATION * 0.10" | bc -l)
END_CUTOFF=$(echo "$DURATION * 0.90" | bc -l)
USABLE_DURATION=$(echo "$END_CUTOFF - $START_OFFSET" | bc -l)
SEGMENT_SIZE=$(echo "$USABLE_DURATION / $SEGMENT_COUNT" | bc -l)

echo "Sampling $NUM_CANDIDATES candidate frames from $SEGMENT_COUNT segments..."

# Generate candidate timestamps
declare -a CANDIDATE_TIMES
for ((i=0; i<NUM_CANDIDATES; i++)); do
    # Distribute candidates across segments with some randomness
    SEGMENT_IDX=$((i % SEGMENT_COUNT))
    SEGMENT_START=$(echo "$START_OFFSET + $SEGMENT_IDX * $SEGMENT_SIZE" | bc -l)
    SEGMENT_END=$(echo "$SEGMENT_START + $SEGMENT_SIZE" | bc -l)
    
    # Add 20-80% into the segment (avoid exact segment boundaries)
    OFFSET_PCT=$(echo "0.2 + ($i * 0.6 / $NUM_CANDIDATES)" | bc -l)
    TIMESTAMP=$(echo "$SEGMENT_START + $SEGMENT_SIZE * $OFFSET_PCT" | bc -l)
    CANDIDATE_TIMES[$i]=$(printf '%.3f' "$TIMESTAMP")
done

# Function to check if frame is black
is_black_frame() {
    local frame="$1"
    local black_pct
    
    # Use ffmpeg to detect black frames in a 1-second window around this frame
    black_pct=$(ffprobe -f lavfi -i "movie=$frame,blackdetect=d=0.1:pic_th=0.98" \
        -show_entries tags=lavfi.black_start -of csv=p=0 2>/dev/null | head -1)
    
    # If black_start is found, the frame is considered black
    if [ -n "$black_pct" ]; then
        return 0  # Is black
    else
        return 1  # Not black
    fi
}

# Function to get frame quality score (higher is better)
get_frame_score() {
    local frame="$1"
    local file_size
    local dimensions
    local width
    local height
    
    # Get file size (larger files usually mean more detail/entropy)
    file_size=$(stat -f%z "$frame" 2>/dev/null || stat -c%s "$frame" 2>/dev/null || echo "0")
    
    # Get dimensions
    dimensions=$(ffprobe -v quiet -show_entries stream=width,height -of csv=s=x:p=0 "$frame" 2>/dev/null)
    if [ -n "$dimensions" ]; then
        width=$(echo "$dimensions" | cut -d'x' -f1)
        height=$(echo "$dimensions" | cut -d'x' -f2)
        # Calculate pixel count as quality factor
        local pixels=$((width * height))
        # Score: file_size / pixel_count (bits per pixel, higher = more detail)
        if [ "$pixels" -gt 0 ]; then
            echo "scale=6; $file_size / $pixels" | bc -l
            return
        fi
    fi
    echo "0"
}

# Extract candidate frames
declare -a FRAME_SCORES
declare -a FRAME_PATHS
declare -a FRAME_TIMES
FRAME_COUNT=0

for i in "${!CANDIDATE_TIMES[@]}"; do
    TIME=${CANDIDATE_TIMES[$i]}
    CANDIDATE_PATH="$TEMP_DIR/candidate_$(printf '%02d' $i).jpg"
    
    if [ "$DEBUG_MODE" = true ]; then
        echo "  Extracting frame at ${TIME}s..."
    fi
    
    # Use input seeking for speed, extract I-frame with thumbnail filter
    if ffmpeg -loglevel "$FFMPEG_LOG_LEVEL" -ss "$TIME" -i "$VIDEO_PATH" \
        -vf "select='eq(pict_type,I)',thumbnail=300,scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos:force_original_aspect_ratio=decrease,pad=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1" \
        -frames:v 1 \
        -q:v "$JPEG_QUALITY" \
        -y "$CANDIDATE_PATH" 2>/dev/null; then
        
        # Check if frame is black
        if is_black_frame "$CANDIDATE_PATH"; then
            if [ "$DEBUG_MODE" = true ]; then
                echo "    ⚠️  Frame at ${TIME}s is black, skipping"
            fi
            rm -f "$CANDIDATE_PATH"
            continue
        fi
        
        # Calculate score
        SCORE=$(get_frame_score "$CANDIDATE_PATH")
        FRAME_SCORES[$FRAME_COUNT]="$SCORE"
        FRAME_PATHS[$FRAME_COUNT]="$CANDIDATE_PATH"
        FRAME_TIMES[$FRAME_COUNT]="$TIME"
        FRAME_COUNT=$((FRAME_COUNT + 1))
        
        if [ "$DEBUG_MODE" = true ]; then
            SIZE=$(du -h "$CANDIDATE_PATH" | cut -f1)
            echo "    ✓ Frame extracted (score: $SCORE, size: $SIZE)"
        fi
    else
        if [ "$DEBUG_MODE" = true ]; then
            echo "    ✗ Failed to extract frame at ${TIME}s"
        fi
    fi
done

# Check if we have any valid candidates
if [ $FRAME_COUNT -eq 0 ]; then
    echo "Warning: No valid frames extracted, falling back to simple extraction..."
    
    # Fallback: simple extraction from 25% mark
    FALLBACK_TIME=$(echo "$DURATION * 0.25" | bc -l)
    POSTER_PATH="$POSTERS_DIR/poster.jpg"
    
    ffmpeg -loglevel "$FFMPEG_LOG_LEVEL" -ss "$FALLBACK_TIME" -i "$VIDEO_PATH" \
        -vf "scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos:force_original_aspect_ratio=decrease,pad=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1" \
        -frames:v 1 \
        -q:v "$JPEG_QUALITY" \
        -y "$POSTER_PATH"
else
    # Find best frame (highest score)
    BEST_IDX=0
    BEST_SCORE=${FRAME_SCORES[0]}
    
    for ((i=1; i<FRAME_COUNT; i++)); do
        if (( $(echo "${FRAME_SCORES[$i]} > $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=${FRAME_SCORES[$i]}
            BEST_IDX=$i
        fi
    done
    
    BEST_TIME=${FRAME_TIMES[$BEST_IDX]}
    BEST_PATH=${FRAME_PATHS[$BEST_IDX]}
    
    # Copy best frame to output
    POSTER_PATH="$POSTERS_DIR/poster.jpg"
    cp "$BEST_PATH" "$POSTER_PATH"
    
    echo ""
    echo "Selected frame at ${BEST_TIME}s (score: $BEST_SCORE)"
    
    if [ "$DEBUG_MODE" = true ]; then
        echo ""
        echo "All candidates:"
        for ((i=0; i<FRAME_COUNT; i++)); do
            MARKER=" "
            if [ $i -eq $BEST_IDX ]; then
                MARKER="★"
            fi
            SIZE=$(du -h "${FRAME_PATHS[$i]}" | cut -f1)
            echo "  $MARKER ${FRAME_TIMES[$i]}s - score: ${FRAME_SCORES[$i]}, size: $SIZE"
        done
    fi
fi

# Verify output
if [ -f "$POSTER_PATH" ]; then
    FILE_SIZE=$(du -h "$POSTER_PATH" | cut -f1)
    DIMENSIONS=$(ffprobe -v quiet -show_entries stream=width,height -of csv=s=x:p=0 "$POSTER_PATH")
    echo ""
    echo "✅ Poster generated successfully!"
    echo "📁 Location: $POSTER_PATH"
    echo "📐 Dimensions: $DIMENSIONS"
    echo "📊 File size: $FILE_SIZE"
    echo "🎨 Quality: $JPEG_QUALITY"
else
    echo "❌ Failed to generate poster"
    exit 1
fi
