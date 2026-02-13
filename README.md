# Webinar Processor

Turn webinar recordings into educational content: articles, summaries, and quizzes.

Webinar Processor takes a video file, transcribes it with speaker diarization, then uses LLMs to generate structured educational materials. It powers the content pipeline for [snap-study.ru](https://snap-study.ru).

## What it does

- **Transcribe** video with speaker diarization (Whisper + pyannote)
- **Generate articles** from transcripts using an outline-first strategy with prompt caching
- **Create summaries** (500-1000 word overviews)
- **Generate quizzes** with explanations for each answer
- **Identify speakers** across webinars using voice embeddings
- **Extract poster frames** from video

## Quick start

```bash
# Install into a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Set up API keys
cp .env.example .env
# Edit .env with your LLM API key

# Generate an article from a transcript
webinar_processor storytell transcript.json --output-file article.txt

# Generate a summary
webinar_processor summarize transcript.json --output-file summary.txt

# Generate a quiz
webinar_processor quiz transcript.json --output-file quiz.txt
```

## Full pipeline

```bash
# 1. Download video
webinar_processor download https://youtu.be/VIDEO_ID -o video/

# 2. Transcribe with speaker diarization
webinar_processor transcribe video/recording.mp4

# 3. Extract poster frame
./poster.sh video/recording.mp4

# 4. Generate content
webinar_processor storytell video/transcript.json --output-file video/story.txt
webinar_processor summarize video/transcript.json --output-file video/summary.txt
webinar_processor quiz video/transcript.json --output-file video/quiz.txt

# 5. Upload to platform
webinar_processor upload-webinar video/recording.mp4 --title "Lecture Title" --slug lecture-1
webinar_processor upload-quiz video/quiz.txt lecture-1
```

## Transcript formats

Two formats are accepted:

**Diarized** (from `transcribe` command): JSON array of segments with timestamps and speakers.
```json
[{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00", "text": "Hello everyone..."}]
```

**ASR** (flat text): JSON object with a `text` field.
```json
{"text": "Hello everyone, today we will discuss..."}
```

## Speaker identification

Manage a persistent speaker database to identify recurring speakers across webinars:

```bash
# Analyze a webinar for speaker voice samples
webinar_processor speakers analyze video/

# Interactively label speakers (listen to samples, assign names)
webinar_processor speakers label video/

# Identify speakers in a new webinar using the reference library
webinar_processor speakers identify new-video/

# Apply identified names to the transcript
webinar_processor speakers apply new-video/

# Manual speaker management
webinar_processor speakers list
webinar_processor speakers enroll --name "Dr. Smith" --audio sample.wav
webinar_processor speakers info spk_a1b2c3d4
webinar_processor speakers merge spk_old spk_new
```

## Configuration

LLM models are configured via environment variables:

```bash
LLM_API_KEY=your_key            # Required
LLM_BASE_URL=https://api.openai.com/v1  # Default
LLM_DEFAULT_MODEL=gpt-5-mini    # Fallback model
LLM_STORY_MODEL=gpt-5.2         # Per-task override
LLM_SUMMARIZATION_MODEL=gpt-5-mini
LLM_QUIZ_MODEL=gpt-5.2
```

Priority: task-specific env var > `LLM_DEFAULT_MODEL` > hardcoded defaults.

## How article generation works

The `storytell` command uses an outline-first strategy optimized for prompt caching:

1. **Outline call**: LLM generates a JSON outline (sections + terminology) from the full transcript
2. **Section calls**: Each section is written separately, all sharing an identical prompt prefix (transcript + outline + terms)
3. **Appendix call**: Key terms and references, same cached prefix

Because calls 2-N share a byte-identical prefix, the API caches the transcript at 1/10th cost after the first section call. A 2-hour lecture costs roughly the same as processing the transcript once.

For transcripts that exceed the model's context window, the system automatically condenses into chunks first, then runs the outline strategy on the condensed notes.

## Development

```bash
# Run tests
python -m pytest

# Run linter
ruff check src/

# Install in development mode
pip install -e .
```

## License

MIT. See [LICENSE](LICENSE).
