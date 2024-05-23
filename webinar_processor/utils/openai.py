from typing import List
import openai
import tiktoken
import spacy

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from sbert_punc_case_ru import SbertPuncCase

sbert_punc_model = SbertPuncCase()

# Load models once and reuse them
spacy_models = {
    "ru": spacy.load("ru_core_news_md"),
    "en": spacy.load("en_core_web_md")
}

DEFAULT_LONG_CONTEXT_MODEL = "gpt-4o"

@retry(wait=wait_random_exponential(multiplier=1, min=30, max=120), stop=stop_after_attempt(7))
def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_token_limit_summary(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 8000,
        'gpt-4': 3000,
        'gpt-4-1106-preview': 100000,
        'gpt-4o': 100000,
        'gpt-4-turbo-preview': 100000
    }
    return token_limit.get(model, 2500)


def get_token_limit_story(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 4000,
        'gpt-4': 4000,
        'gpt-4-1106-preview': 4000,
        'gpt-4o': 4000,
        'gpt-4-turbo-preview': 4000
    }
    return token_limit.get(model, 2000)


def sber_text_punctuate(text: str) -> str:
    return sbert_punc_model.punctuate(text)


def spacy_split_sentences(text: str, language: str = "ru") -> List[str]:
    nlp = spacy_models.get(language)
    if not nlp:
        raise ValueError(f"Unsupported language: {language}")

    doc = nlp(text)
    return [str(sent) for sent in doc.sents]


def split_text_to_sentences(text: str, language: str = "ru", max_sent_len: int = 500) -> List[str]:
    sents = spacy_split_sentences(text, language)
    result = []

    for s in sents:
        if len(s) > max_sent_len:
            s = sber_text_punctuate(s)
            result.extend(spacy_split_sentences(s, language))
        else:
            result.append(s)

    return result


def count_tokens(model: str, text: str):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError as e:
        raise RuntimeError(f"Invalid model: {model}") from e
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


def text_transform(text: str, language: str, model: str, prompt_template: str) -> str:
    token_limit = get_token_limit_story(model)
    output = ""

    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        prompt = prompt_template.format(text=segment)

        part = get_completion(prompt, model)
        output += "\n" + part
    return output


def create_summary(text: str, language: str, model: str, prompt: str) -> str:
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        try:
            request = prompt.format(resume=resume, text=segment)
            resume = get_completion(request, model)
        except openai.error.InvalidRequestError as e:
            raise RuntimeError('Error: OpenAI invalid request') from e

    return resume


def create_summary_with_context(text: str, context: str, language: str, model: str, prompt: str) -> str:
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        try:
            request = prompt.format(resume=resume, text=segment, context=context)
            resume = get_completion(request, model)
        except openai.error.InvalidRequestError as e:
            raise RuntimeError('Error: OpenAI invalid request') from e

    return resume
