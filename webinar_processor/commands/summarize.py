import json
import os
from typing import List
import click
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
import openai  # for OpenAI API calls
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from webinar_processor.utils.package import get_config_path

_ = load_dotenv(find_dotenv())


@retry(wait=wait_random_exponential(min=5, max=120), stop=stop_after_attempt(6))
def get_completion(prompt, model="gpt-3.5-turbo-16k"):
  messages = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
     model=model,
     messages=messages,
     temperature=0, # this is the degree of randomness of the model's output
  )
  return response.choices[0].message["content"]


def split_text_to_sentences(text: str, language: str = "ru") -> List[str]:
    import spacy
    nlp = spacy.load("ru_core_news_md")
    doc = nlp(text)
    return [str(sent) for sent in doc.sents]


def count_tokens(model: str, text: str):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError as e:
        raise click.ClickException(f"Invalid model: {model}") from e
    tokens = encoding.encode(text)
    return len(tokens)


def generate_text_segments(model, long_text, language, token_limit):
    sentences = split_text_to_sentences(long_text, language)
    
    current_segment = []
    current_token_count = 0

    for sentence in sentences:
        token_count_for_sentence = count_tokens(model, sentence)
        
        # If adding the next sentence exceeds the token_limit
        if current_token_count + token_count_for_sentence > token_limit:
            # Yield the current segment
            yield ' '.join(current_segment)
            
            # Start a new segment
            current_segment = []
            current_token_count = 0

        current_segment.append(sentence)
        current_token_count += token_count_for_sentence

    # Yield the last segment if it's not empty
    if current_segment:
        yield ' '.join(current_segment)


def get_token_limit_summary(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 10000,
        'gpt-4': 3000,
        'gpt-4-1106-preview': 35000
    }
    return token_limit.get(model, 2500)


def get_token_limit_story(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 7500,
        'gpt-4': 3700,
        'gpt-4-1106-preview': 9000
    }
    return token_limit.get(model, 2000)


def create_summary(text: str, language: str, model: str, prompt: str) -> str:
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    click.echo(click.style(f'text length {len(text)} split into {len(segments)} segments', fg='green'))
    for segment in segments:
        try:
            request = prompt.format(resume=resume, text=segment)
            resume = get_completion(request, model)
        except openai.error.InvalidRequestError:
            click.echo(click.style(f'Error: OpenAI invalid request', fg='red'))
            click.echo(request)
            break

    return resume

def create_story(text: str, language: str, model: str, prompt_template: str) -> str:
    token_limit = get_token_limit_story(model)
    story = ""
    
    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        prompt = prompt_template.format(text=segment)
        part = get_completion(prompt, model)
        story += "\n\n" + part
    return story


@click.command()
@click.argument('asr_path', nargs=1)
@click.argument('model', nargs=1, default="gpt-3.5-turbo-16k")
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def summarize(asr_path: str, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create transcript summary
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style(f'Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    with open(asr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not prompt_file:
        prompt_file = get_config_path("short-summary.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt = pf.read()

    summary = create_summary(data["text"], language, model, prompt)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(summary)
    else:
        click.echo(summary)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default="gpt-3.5-turbo-16k")
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def storytell(asr_file: click.File, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create a story
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style(f'Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    data = json.load(asr_file)

    if not prompt_file:
        prompt_file = get_config_path("long-story.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_template = pf.read()

    story = create_story(data["text"], language, model, prompt_template)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(story)
    else:
        click.echo(story)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def raw_text(asr_file: click.File, output_file: str):
    """
    Write raw transcript text
    """

    data = json.load(asr_file)
    story = data["text"]

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(story)
    else:
        click.echo(story)
