"""Encode/decode numpy embeddings for JSON-safe storage."""

import base64
import numpy as np


def encode_embedding(embedding: np.ndarray) -> str:
    """Encode numpy array to base64 string for JSON storage."""
    return base64.b64encode(embedding.tobytes()).decode('utf-8')


def decode_embedding(encoded: str) -> np.ndarray:
    """Decode base64 string back to numpy array."""
    from webinar_processor.services.speaker_database import EMBEDDING_DTYPE
    bytes_data = base64.b64decode(encoded)
    return np.frombuffer(bytes_data, dtype=EMBEDDING_DTYPE)
