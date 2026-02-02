from vertexai.preview.generative_models import GenerativeModel


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(multiplier=1, min=30, max=120), stop=stop_after_attempt(7))
def get_completion(prompt, model="gemini-1.0-pro"):
    gemini_pro_model = GenerativeModel(model)
    response = gemini_pro_model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text
