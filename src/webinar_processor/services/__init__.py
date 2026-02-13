"""Services for business logic.

Heavy dependencies (torch, speechbrain, pyannote) are imported lazily
at call sites — do NOT add eager imports of SpeakerDatabase,
or VoiceEmbeddingService here.
"""
