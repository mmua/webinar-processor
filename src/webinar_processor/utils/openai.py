from typing import List
import logging
import tiktoken
import spacy

from webinar_processor.llm import LLMClient, LLMConfig
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

spacy_models = {"ru": None, "en": None}

_llm_client = None

TOKEN_LIMITS = {
    'gpt-4.1': 1047576, 'gpt-4.1-mini': 1047576, 'gpt-4.1-nano': 1047576,
    'gpt-4o': 128000, 'gpt-4o-mini': 128000, 'gpt-4-turbo': 128000,
    'gpt-3.5-turbo-0125': 16000,
    'gpt-5.2': 128000, 'gpt-5-mini': 128000,
}

OUTPUT_LIMITS = {
    'gpt-4.1': 32768, 'gpt-4.1-mini': 32768, 'gpt-4.1-nano': 32768,
    'gpt-4o': 32768, 'gpt-4o-mini': 16384, 'gpt-4-turbo': 32768,
    'gpt-3.5-turbo-0125': 4096,
    'gpt-5.2': 64000, 'gpt-5-mini': 64000,
}

def get_output_limit(model: str) -> int:
    return OUTPUT_LIMITS.get(model, 4096)

def get_token_limit_summary(model: str) -> int:
    return min(TOKEN_LIMITS.get(model, 8000), OUTPUT_LIMITS.get(model, 4096))

def get_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

@retry(wait=wait_random_exponential(multiplier=1, min=30, max=120), stop=stop_after_attempt(7))
def get_completion(prompt, model=None, max_tokens=None):
    client = get_client()
    if model is None:
        model = LLMConfig.get_model('default')
    if max_tokens is None:
        max_tokens = get_output_limit(model)
    result = client.generate(prompt, model=model, max_tokens=max_tokens)
    if result is None:
        raise RuntimeError(f'LLM generation failed for model {model}')
    return result

def get_token_limit_story(model: str) -> int:
    return get_token_limit_summary(model)


def sbert_text_punctuate(text: str) -> str:
    from sbert_punc_case_ru import SbertPuncCase
    sbert_punc_model = SbertPuncCase()
    return sbert_punc_model.punctuate(text)


def spacy_split_sentences(text: str, language: str = "ru") -> List[str]:
    nlp = spacy_models.get(language)
    if nlp is None:
        model_name = "ru_core_news_md" if language == "ru" else "en_core_web_md"
        nlp = spacy.load(model_name)
        spacy_models[language] = nlp
    doc = nlp(text)
    return [str(sent) for sent in doc.sents]


def split_text_to_sentences(text: str, language: str = "ru", max_sent_len: int = 500) -> List[str]:
    sents = spacy_split_sentences(text, language)
    result = []
    for s in sents:
        if len(s) > max_sent_len:
            s = sbert_text_punctuate(s)
            result.extend(spacy_split_sentences(s, language))
        else:
            result.append(s)
    return result


def count_tokens(model: str, text: str):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


def generate_text_segments(model, long_text, language, token_limit):
    sentences = split_text_to_sentences(long_text, language)
    current_segment = []
    current_token_count = 0
    
    for sentence in sentences:
        token_count = count_tokens(model, sentence)
        if current_token_count + token_count > token_limit:
            if current_segment:
                yield ' '.join(current_segment)
            current_segment = []
            current_token_count = 0
        current_segment.append(sentence)
        current_token_count += token_count
    
    if current_segment:
        yield ' '.join(current_segment)


def text_transform(text: str, language: str, model: str, prompt_template: str) -> str:
    token_limit = get_token_limit_story(model)
    output = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for i, segment in enumerate(segments):
        prompt = prompt_template.format(text=segment)
        part = get_completion(prompt, model)
        if part:
            output += "\n" + part
        else:
            logger.warning(f"Segment {i+1}/{len(segments)} returned empty result, skipping")
    return output

def create_summary(text: str, language: str, model: str, prompt: str) -> str:
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for i, segment in enumerate(segments):
        request = prompt.format(resume=resume, text=segment)
        result = get_completion(request, model)
        if result:
            resume = result
        else:
            logger.warning(f"Summary segment {i+1}/{len(segments)} returned empty result")
    return resume

def create_summary_with_context(text: str, context: str, language: str, model: str, prompt: str) -> str:
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for i, segment in enumerate(segments):
        request = prompt.format(resume=resume, text=segment, context=context)
        result = get_completion(request, model)
        if result:
            resume = result
        else:
            logger.warning(f"Contextual summary segment {i+1}/{len(segments)} returned empty result")
    return resume
