# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Use the project venv (not system python)
../venv/bin/pip install -e .
../venv/bin/webinar_processor --help

# Run a specific command
../venv/bin/webinar_processor storytell transcript.json --output-file story.txt
../venv/bin/webinar_processor summarize transcript.json --output-file summary.txt
../venv/bin/webinar_processor quiz transcript.json --output-file quiz.txt
```

## Testing

```bash
../venv/bin/python -m pytest                              # all tests
../venv/bin/python -m pytest tests/test_llm.py -v         # single file
../venv/bin/python -m pytest tests/test_llm.py::TestLLMConfig::test_get_api_key_from_llm_api_key  # single test
../venv/bin/python -m pytest -k "test_get_api_key"        # by pattern
```

Known pre-existing failures (not ours): `test_generate_token_limit_exceeded`, `test_ensure_dir_exists`, `test_voice_*`.

## Architecture

**CLI layer** (`src/webinar_processor/`): Click-based CLI. Entry point is `cli` in `__init__.py`. Commands are registered via `cli.add_command()`. Each command lives in `commands/cmd_*.py`, exports a `@click.command()` function, and is re-exported through `commands/__init__.py`.

**LLM layer** (`llm/`): `LLMConfig` reads env vars (LLM_API_KEY, LLM_BASE_URL, per-task model overrides like `LLM_STORY_MODEL`). `LLMClient` wraps OpenAI API with token-limit pre-check. `constants.py` has TOKEN_LIMITS and OUTPUT_LIMITS dicts.

**Completion helper** (`utils/openai.py`): `get_completion(prompt, model, max_tokens)` is the main function commands use. Has `@retry` with exponential backoff (7 attempts). Singleton `LLMClient`.

**Prompt templates** live in `src/webinar_processor/resources/conf/*.txt`. Loaded via `BaseCommand.load_prompt_template(get_config_path("filename.txt"))`. Templates use `str.format()` placeholders like `{text}`.

**BaseCommand** (`commands/base_command.py`): Static utility methods shared across commands — `load_json_file`, `load_prompt_template`, `write_output`, `handle_llm_error`.

## Storytell Pipeline (main content generation)

The `storytell` command in `cmd_summarize.py` is the most complex, with 3 strategies:

1. **Outline + per-section** (default): Generates JSON outline+terms, then builds a cached prefix (transcript + outline + terms). Each section call appends a different suffix to the same prefix, enabling prompt caching at 1/10th cost.
2. **Single-pass** (`--single-pass`): One LLM call with `storytell-prompt.txt`.
3. **Chunked** (auto for long transcripts): Splits into time/size chunks, condenses each, then feeds condensed notes into strategy 1 or 2.

Prompt caching design: all calls after the outline share an identical prefix string. The prefix must be byte-identical across calls for the API cache to hit.

## Adding a New Command

1. Create `src/webinar_processor/commands/cmd_<name>.py` with a `@click.command()` function
2. Use `BaseCommand` for file I/O and error handling
3. Add import to `commands/__init__.py` and `__all__`
4. Register in `__init__.py` with `cli.add_command(commands.<name>)`

## Adding a Prompt

1. Create `.txt` file in `src/webinar_processor/resources/conf/`
2. Load with `BaseCommand.load_prompt_template(get_config_path("filename.txt"))`
3. Use `{placeholder}` syntax for template variables

## Model Configuration

Priority: `LLM_<TASK>_MODEL` env var > `LLM_DEFAULT_MODEL` env var > hardcoded defaults in `llm/config.py`.

Task keys: `story`, `summarization`, `quiz`, `topics`, `polish`, `speaker_extraction`, `default`.

## Input Formats

Two transcript formats are accepted:
- **Diarized**: JSON array with `{start, end, speaker, text}` segments
- **ASR**: JSON object with `"text"` field

Detection via `is_diarized_format()` in `utils/transcript_formatter.py`.

## Key Constraints & Design Decisions

**Output window << input window.** LLMs can read 128-256K tokens but only output 32-128K. Never ask the LLM to reproduce the full transcript plus modifications. Instead: small, focused output per call (one section at a time).

**Transcripts are very long.** A 2-hour lecture is 100-200K tokens. This drives the entire architecture — single-pass won't fit or produces truncated output, so we split into outline + N section calls.

**Prompt caching cuts cost 10x.** The OpenAI API caches identical prompt prefixes. We exploit this: the cached prefix (transcript + outline + terms) is sent identically on every section call. Only the suffix (section task) varies. This means the transcript is charged at full price once (outline call), then at 1/10th for each section and appendix call. **Any change to the prefix string between calls breaks the cache** — it must be byte-identical.

**Terminology extracted early.** Terms are extracted in the outline call (not a separate pass) and included in the cached prefix. This ensures every section uses consistent terminology from the start, rather than fixing inconsistencies after the fact.

**No polish step.** We eliminated a "rewrite full text" polish pass. It would require reproducing the entire article (output window problem) and adds cost. Instead, quality comes from: terms in the prefix, per-section context (prev/next section info), and a good section prompt.

**Chunked fallback for extreme length.** When the transcript exceeds the model's context window, we condense chunks first (40-min time windows or 50K-token blocks with overlap), then run the normal outline+sections pipeline on the condensed notes. If even condensed notes don't fit, there's a final single-pass-from-notes fallback.

**Shared prompt prefix across commands.** The `summarize`, `quiz`, and `storytell` commands all use the same prompt prefix format (transcript header + transcript + separator). This means running them in sequence benefits from the API cache — the transcript is only fully processed on the first call.

## Language

Content prompts and output are in Russian. Code, comments, and variable names are in English.

## Integration Contracts

Contracts defining data formats exchanged with external systems (e.g., snap-study website).

### Quiz Format Contract

The `quiz` command generates quizzes in a specific Markdown format consumed by the hosting platform.

**Format Specification:**
```markdown
## Квиз: [Topic from transcript]

### Вопрос 1: [Question text based on lecture content]
- **A) # Correct answer**
  > Объяснение: [Detailed explanation why correct, referencing lecture concepts]
- **B) Incorrect option**
  > Объяснение: [Explanation of misconception or why wrong]
- **C) Incorrect option**
  > Объяснение: [Explanation...]
[3-5 options total]

### Вопрос 2: ...
[8-12 questions total]
```

**Structure Requirements:**
- Header: `## Квиз: ` followed by topic
- Questions: `### Вопрос N: ` prefix (sequential numbering)
- Options: Bullet list with `**X) ` prefix where X is A, B, C, etc.
- Correct marker: `#` immediately after `**X) ` (e.g., `**A) # Correct`)
- Explanations: Blockquote (`>`) on line following each option
- Language: Russian
- Content: Understanding-based (not memorization), using lecture examples

**Validation Rules:**
- 8-12 questions per quiz
- 3-5 answer options per question
- Exactly one correct answer per question
- Every option must have an explanation blockquote
- Explanations must reference specific lecture content

### Webinar Upload Contract

The `upload_webinar` and `upload_quiz` commands communicate with the snap-study API using multipart/form-data POST requests.

**Webinar Upload (`upload_webinar`):**
- **Endpoint**: Configurable via `EDU_PATH_API_ENDPOINT` env var
- **Method**: POST
- **Headers**: `Authorization: Bearer {EDU_PATH_TOKEN}`
- **Form Data**:
  - `title`: Webinar title
  - `slug`: Unique identifier
  - `summary`: Short summary text (optional)
  - `long_summary`: Full article text (optional)
- **Files**:
  - `video_file`: Video file (required)
  - `poster_file`: Thumbnail image (optional, defaults to `posters/poster.jpg`)
  - `transcript_file`: Transcript JSON (optional, defaults to `transcript.json`)
- **Success**: HTTP 201

**Quiz Upload (`upload_quiz`):**
- **Endpoint**: Configurable, defaults to specific snap-study URL
- **Method**: POST
- **Headers**: `Authorization: Bearer {EDU_PATH_TOKEN}`
- **Form Data**:
  - `slug`: Webinar identifier
  - `content`: Quiz markdown content (see Quiz Format Contract above)
- **Success**: HTTP 201

**Note**: Upload commands are coupled to HSE's snap-study infrastructure and will be migrated to the website project in the future.
