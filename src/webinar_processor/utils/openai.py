from typing import List
import openai
import tiktoken
import spacy
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

class IdentityPuncCase:
    """Fallback punctuation model that returns text unchanged."""
    def punctuate(self, text: str) -> str:
        return text

try:
    from sbert_punc_case_ru import SbertPuncCase
    sbert_punc_model = SbertPuncCase()
except ImportError:
    # Use identity model for testing and when sbert_punc_case_ru is not available
    sbert_punc_model = IdentityPuncCase()

# Load models once and reuse them
spacy_models = {
    "ru": spacy.load("ru_core_news_md"),
    "en": spacy.load("en_core_web_md")
}

DEFAULT_LONG_CONTEXT_MODEL = "gpt-4o"


@retry(wait=wait_random_exponential(multiplier=1, min=30, max=120), stop=stop_after_attempt(7))
def get_completion(prompt, model="gpt-4o-mini"):
    """
    Get completion from OpenAI API with the given prompt and model.
        
    Args:
        prompt: The prompt to send to the API
        model: The model to use for completion
        
    Returns:
        The generated text from the API response
    """
    try:
        # Create a client instance with the API key, ensuring no global config interferes
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        # Create the client with minimal configuration
        client = openai.OpenAI(
            api_key=api_key,
            # No additional config to avoid issues with kwargs
            base_url="https://api.openai.com/v1",
        )
        
        # Create the request
        messages = [{"role": "user", "content": prompt}]
        
        # Make the API request
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        
        # Extract and return the generated content
        return response.choices[0].message.content
    except Exception as e:
        # Print detailed error information for debugging
        print(f"Error in get_completion: {type(e).__name__}: {str(e)}")
        raise


def get_token_limit_summary(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 8000,
        'gpt-4': 3000,
        'gpt-4-1106-preview': 100000,
        'gpt-4o': 100000,
        'gpt-4o-mini': 100000,
        'gpt-4-turbo-preview': 100000
    }
    return token_limit.get(model, 2500)


def get_token_limit_story(model: str) -> int:
    token_limit = {
        'gpt-3.5-turbo-16k': 4000,
        'gpt-4': 4000,
        'gpt-4o': 8000,
        'gpt-4o-mini': 8000,
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
        encoding = tiktoken.encoding_for_model(model)
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
    """
    Transform text using OpenAI API with the given prompt template.
    
    Args:
        text: The text to transform
        language: The language of the text
        model: The model to use for transformation
        prompt_template: The prompt template to use
        
    Returns:
        The transformed text
    """
    token_limit = get_token_limit_story(model)
    output = ""

    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        try:
            prompt = prompt_template.format(text=segment)
            part = get_completion(prompt, model)
            output += "\n" + part
        except Exception as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an error: {e}")
            raise RuntimeError('Error: OpenAI request failed') from e
            
    return output


def create_summary(text: str, language: str, model: str, prompt: str) -> str:
    """
    Create a summary of the given text using OpenAI API.
    
    Args:
        text: The text to summarize
        language: The language of the text
        model: The model to use for summarization
        prompt: The prompt template to use
        
    Returns:
        The generated summary
    """
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        try:
            request = prompt.format(resume=resume, text=segment)
            resume = get_completion(request, model)
        except Exception as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an error: {e}")
            raise RuntimeError('Error: OpenAI request failed') from e
        return resume


def create_summary_with_context(text: str, context: str, language: str, model: str, prompt: str) -> str:
    """
    Create a summary of the given text with additional context using OpenAI API.
    
    Args:
        text: The text to summarize
        context: Additional context to include in the prompt
        language: The language of the text
        model: The model to use for summarization
        prompt: The prompt template to use
        
    Returns:
        The generated summary
    """
    token_limit = get_token_limit_summary(model)
    resume = ""
    segments = list(generate_text_segments(model, text, language, token_limit))
    for segment in segments:
        try:
            request = prompt.format(resume=resume, text=segment, context=context)
            resume = get_completion(request, model)
        except Exception as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an error: {e}")
            raise RuntimeError('Error: OpenAI request failed') from e

    return resume
