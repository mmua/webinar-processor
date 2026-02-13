# CLAUDE.md

## Build & Run

```bash
../venv/bin/pip install -e .
../venv/bin/webinar_processor --help
../venv/bin/webinar_processor storytell transcript.json --output-file story.txt
../venv/bin/webinar_processor summarize transcript.json --output-file summary.txt
../venv/bin/webinar_processor quiz transcript.json --output-file quiz.txt
```

## Testing

```bash
../venv/bin/python -m pytest                        # all tests
../venv/bin/python -m pytest tests/test_llm.py -v   # single file
../venv/bin/python -m pytest -k "test_get_api_key"  # by pattern
```

## Architecture

```
src/webinar_processor/
  __init__.py              CLI entry point, logging config
  commands/
    cmd_*.py               Thin Click commands (I/O + delegation)
    speakers/              Speaker management subcommands
  services/
    storytell_service.py   Article generation (3 strategies)
    transcript_service.py  Load & format transcripts
    speaker_database.py    SQLite speaker profiles
    speaker_name_extractor.py  LLM name extraction
    voice_embedding_service.py Voice embeddings (pyannote)
  llm/
    config.py              Model selection & env vars
    client.py              OpenAI wrapper + token pre-check
    constants.py           TOKEN_LIMITS, OUTPUT_LIMITS
    exceptions.py          LLMError, TokenLimitError
  utils/
    completion.py          get_completion() with retry, singleton client
    io.py                  load_prompt_template, write_output
    transcript_formatter.py  Diarized text formatting
    token.py               tiktoken wrapper
    embedding_codec.py     base64 numpy encode/decode
    ffmpeg.py              Audio/video conversion
    package.py             Resource path resolution
  resources/conf/*.txt     Prompt templates ({placeholder} syntax)
```

### Layer rules

- **Commands** depend on services and utils. They own all Click interaction (echo, Abort, options).
- **Services** use `logging`, never `click`. They raise standard exceptions (`ValueError`, `LLMError`). Commands catch and translate to user-facing output.
- **LLM layer** is a thin OpenAI wrapper. Domain logic does not belong here.
- **Utils** are stateless helpers. `completion.py` holds the singleton `LLMClient`.

### Adding a command

1. Create `commands/cmd_<name>.py` with a `@click.command()` function
2. Import in `commands/__init__.py`, add to `__all__`
3. Register in `__init__.py` with `cli.add_command(commands.<name>)`

### Adding a prompt

1. Create `.txt` in `resources/conf/`
2. Load with `load_prompt_template(get_config_path("filename.txt"))`
3. Use `{placeholder}` syntax

## Key Design Decisions

**Output window << input window.** LLMs read 128-256K tokens but output 32-128K max. Never ask the LLM to reproduce the full transcript. Instead: small, focused output per call.

**Outline + per-section with prompt caching.** A 2-hour lecture is 100-200K tokens. Single-pass truncates or fails. We generate a JSON outline first, then write one section at a time. All section calls share a byte-identical prefix (transcript + outline + terms), so the API caches it at 1/10th cost. Any prefix mutation breaks the cache.

**Terminology in the cached prefix.** Terms are extracted in the outline call and baked into the shared prefix. Every section sees the same terminology from the start, avoiding inconsistencies.

**No polish pass.** Rewriting the full article would hit the output window limit and double cost. Quality comes from: terms in prefix, prev/next section context, and focused section prompts.

**Chunked fallback.** When text exceeds context, we condense 40-min chunks first, then run outline+sections on condensed notes. If notes still don't fit, single-pass from notes.

**Shared prefix across commands.** `summarize`, `quiz`, and `storytell` share the same transcript prefix format. Running them in sequence reuses the API cache.

## Model Configuration

Priority: `LLM_<TASK>_MODEL` env var > `LLM_DEFAULT_MODEL` env var > hardcoded defaults in `llm/config.py`.

Task keys: `story`, `summarization`, `quiz`, `speaker_extraction`, `default`.

## Input Formats

- **Diarized**: JSON array of `{start, end, speaker, text}` segments
- **ASR**: JSON object with `"text"` field

Detection: `is_diarized_format()` in `transcript_formatter.py`. Loading: `transcript_service.load_and_format_transcript()`.

## Language

Prompts and output are in Russian. Code and comments are in English.

## Integration Contracts

### Quiz Format

```markdown
## Квиз: [Topic]

### Вопрос 1: [Question]
- **A) # Correct answer**
  > Объяснение: [Why correct]
- **B) Wrong answer**
  > Объяснение: [Why wrong]
```

Rules: 8-12 questions, 3-5 options each, exactly one correct (marked with `#`), every option has an explanation blockquote.

### Upload API

Both use `Authorization: Bearer {EDU_PATH_TOKEN}`.

- **upload_webinar**: POST to `EDU_PATH_API_ENDPOINT` with multipart form (title, slug, video_file, poster_file, transcript_file). Returns 201.
- **upload_quiz**: POST to `EDU_PATH_QUIZ_ENDPOINT` with slug + quiz markdown content. Returns 201.
