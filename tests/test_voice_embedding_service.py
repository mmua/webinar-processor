import os
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock, patch
from webinar_processor.services.voice_embedding_service import VoiceEmbeddingService

@pytest.fixture
def mock_audio():
    """Create a mock audio path for testing."""
    # Returns a path that looks like a WAV file but won't actually be opened
    # The actual audio processing is mocked in the test
    return "/tmp/mock_test_audio.wav"

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

def test_extract_embedding(voice_service):
    """Test extracting voice embedding from an audio segment."""
    # Mocks the audio module to return a simple waveform
    mock_embedding = np.random.rand(256).astype(np.float32)
    voice_service.model = Mock(return_value=np.array([mock_embedding]))
    voice_service.audio = Mock()
    # waveform must be a numpy array for proper indexing
    waveform = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds at 16kHz
    voice_service.audio.crop = Mock(return_value=(waveform, 16000))
    
    # Test extraction
    embedding = voice_service.extract_embedding(mock_audio, 0.0, 3.0)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (256,)
    assert np.allclose(embedding, mock_embedding)

def test_extract_embedding_error(voice_service, mock_audio):
    """Test error handling in embedding extraction."""
    # Mock the model to raise an exception
    voice_service.model = Mock(side_effect=Exception("Test error"))
    
    # Test extraction
    embedding = voice_service.extract_embedding(mock_audio, 0.0, 3.0)
    
    assert embedding is None

def test_get_speaker_embeddings(voice_service, mock_audio, mock_transcript):
    """Test getting embeddings for multiple speakers."""
    # Mocks extract_embedding method
    mock_embedding = np.random.rand(256).astype(np.float32)
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
    assert np.allclose(embeddings['SPEAKER_1'][0], mock_embedding)
    assert np.allclose(embeddings['SPEAKER_2'][0], mock_embedding)

def test_get_speaker_embeddings_min_duration(voice_service, mock_audio, mock_transcript):
    """Test minimum duration filtering in speaker embeddings."""
    # Mock the extract_embedding method
    mock_embedding = np.random.rand(256).astype(np.float32)
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
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
        np.array([3.0, 4.0, 5.0, 6.0])
    ]
    
    # Calculate mean
    mean = voice_service.get_mean_embedding(embeddings)
    
    assert isinstance(mean, np.ndarray)
    assert mean.shape == (4,)
    assert np.array_equal(mean, np.array([2.0, 3.0, 4.0, 5.0]))

def test_process_audio_file(voice_service, mock_audio, mock_transcript):
    """Test processing an audio file and getting mean embeddings."""
    # Mocks get_speaker_embeddings method
    mock_embeddings = {
        'SPEAKER_1': [np.random.rand(256).astype(np.float32) for _ in range(3)],
        'SPEAKER_2': [np.random.rand(256).astype(np.float32) for _ in range(2)]
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
    assert mean_embeddings['SPEAKER_1'].shape == (256,)
    assert mean_embeddings['SPEAKER_2'].shape == (256,)

@patch('webinar_processor.services.voice_embedding_service.convert_mp4_to_wav')
@patch.object(VoiceEmbeddingService, 'get_speaker_embeddings')
def test_process_audio_file_mp4(mock_get_embeddings, convert_mock, voice_service, mock_transcript):
    """Test processing an MP4 file."""
    # Create a mock MP4 file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        mp4_path = f.name
    
    # Mock embedding extraction
    mock_embeddings = {
        'SPEAKER_1': [np.random.rand(256).astype(np.float32) for _ in range(3)],
        'SPEAKER_2': [np.random.rand(256).astype(np.float32) for _ in range(2)]
    }
    mock_get_embeddings.return_value = mock_embeddings
    
    # Test processing
    mean_embeddings = voice_service.process_audio_file(
        mp4_path,
        mock_transcript,
        min_duration=2.0
    )
    
    assert convert_mock.called
    assert mock_get_embeddings.called
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