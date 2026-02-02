import os
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock, patch
from pyannote.core import Segment
from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService

@pytest.fixture
def mock_audio():
    """Create a mock audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Create a simple sine wave audio file
        sample_rate = 16000
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
        f.write(audio_data.tobytes())
        return f.name

@pytest.fixture
def mock_transcript():
    """Create a mock transcript for testing."""
    return [
        {
            'speaker': 'SPEAKER_1',
            'start': 0.0,
            'end': 3.0,
            'text': 'Hello, this is speaker one.'
        },
        {
            'speaker': 'SPEAKER_2',
            'start': 3.0,
            'end': 5.0,
            'text': 'And this is speaker two.'
        }
    ]

@pytest.fixture
def voice_service():
    """Create a VoiceEmbeddingService instance for testing."""
    return VoiceEmbeddingService()

def test_extract_embedding(voice_service, mock_audio):
    """Test extracting voice embedding from an audio segment."""
    # Mock the model to return a fixed embedding
    mock_embedding = np.random.rand(192)  # ECAPA-TDNN embeddings are 192-dimensional
    voice_service.model = Mock(return_value=mock_embedding[None])
    
    # Test extraction
    embedding = voice_service.extract_embedding(mock_audio, 0.0, 3.0)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (192,)
    assert np.array_equal(embedding, mock_embedding)

def test_extract_embedding_error(voice_service, mock_audio):
    """Test error handling in embedding extraction."""
    # Mock the model to raise an exception
    voice_service.model = Mock(side_effect=Exception("Test error"))
    
    # Test extraction
    embedding = voice_service.extract_embedding(mock_audio, 0.0, 3.0)
    
    assert embedding is None

def test_get_speaker_embeddings(voice_service, mock_audio, mock_transcript):
    """Test getting embeddings for multiple speakers."""
    # Mock the extract_embedding method
    mock_embedding = np.random.rand(192)
    voice_service.extract_embedding = Mock(return_value=mock_embedding)
    
    # Test getting embeddings
    embeddings = voice_service.get_speaker_embeddings(
        mock_audio,
        mock_transcript,
        min_duration=2.0
    )
    
    assert isinstance(embeddings, dict)
    assert 'SPEAKER_1' in embeddings
    assert 'SPEAKER_2' in embeddings
    assert len(embeddings['SPEAKER_1']) == 1
    assert len(embeddings['SPEAKER_2']) == 1
    assert np.array_equal(embeddings['SPEAKER_1'][0], mock_embedding)
    assert np.array_equal(embeddings['SPEAKER_2'][0], mock_embedding)

def test_get_speaker_embeddings_min_duration(voice_service, mock_audio, mock_transcript):
    """Test minimum duration filtering in speaker embeddings."""
    # Mock the extract_embedding method
    mock_embedding = np.random.rand(192)
    voice_service.extract_embedding = Mock(return_value=mock_embedding)
    
    # Test with high minimum duration
    embeddings = voice_service.get_speaker_embeddings(
        mock_audio,
        mock_transcript,
        min_duration=4.0
    )
    
    assert isinstance(embeddings, dict)
    assert len(embeddings) == 0  # No segments should pass the duration filter

def test_get_mean_embedding(voice_service):
    """Test calculating mean embedding from multiple embeddings."""
    # Create test embeddings
    embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        np.array([3.0, 4.0, 5.0])
    ]
    
    # Calculate mean
    mean = voice_service.get_mean_embedding(embeddings)
    
    assert isinstance(mean, np.ndarray)
    assert mean.shape == (3,)
    assert np.array_equal(mean, np.array([2.0, 3.0, 4.0]))

def test_process_audio_file(voice_service, mock_audio, mock_transcript):
    """Test processing an audio file and getting mean embeddings."""
    # Mock the get_speaker_embeddings method
    mock_embeddings = {
        'SPEAKER_1': [np.random.rand(192) for _ in range(3)],
        'SPEAKER_2': [np.random.rand(192) for _ in range(2)]
    }
    voice_service.get_speaker_embeddings = Mock(return_value=mock_embeddings)
    
    # Test processing
    mean_embeddings = voice_service.process_audio_file(
        mock_audio,
        mock_transcript,
        min_duration=2.0
    )
    
    assert isinstance(mean_embeddings, dict)
    assert 'SPEAKER_1' in mean_embeddings
    assert 'SPEAKER_2' in mean_embeddings
    assert mean_embeddings['SPEAKER_1'].shape == (192,)
    assert mean_embeddings['SPEAKER_2'].shape == (192,)

@patch('webinar_processor.utils.ffmpeg.convert_mp4_to_wav')
def test_process_audio_file_mp4(convert_mock, voice_service, mock_transcript):
    """Test processing an MP4 file."""
    # Create a mock MP4 file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        mp4_path = f.name
    
    # Mock the conversion and embedding extraction
    mock_embeddings = {
        'SPEAKER_1': [np.random.rand(192) for _ in range(3)],
        'SPEAKER_2': [np.random.rand(192) for _ in range(2)]
    }
    voice_service.get_speaker_embeddings = Mock(return_value=mock_embeddings)
    
    # Test processing
    mean_embeddings = voice_service.process_audio_file(
        mp4_path,
        mock_transcript,
        min_duration=2.0
    )
    
    assert convert_mock.called
    assert isinstance(mean_embeddings, dict)
    assert len(mean_embeddings) == 2
    
    # Cleanup
    os.unlink(mp4_path)

def test_process_audio_file_no_embeddings(voice_service, mock_audio, mock_transcript):
    """Test processing when no valid embeddings can be extracted."""
    # Mock the get_speaker_embeddings method to return empty results
    voice_service.get_speaker_embeddings = Mock(return_value={})
    
    # Test processing
    mean_embeddings = voice_service.process_audio_file(
        mock_audio,
        mock_transcript,
        min_duration=2.0
    )
    
    assert isinstance(mean_embeddings, dict)
    assert len(mean_embeddings) == 0 