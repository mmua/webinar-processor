import sqlite3
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import os

EMBEDDING_DIM = 256
EMBEDDING_DTYPE = np.float32


class SpeakerDatabase:
    """Service for managing speaker profiles and identifications."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            home_dir = Path.home()
            db_dir = home_dir / ".webinar_processor"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "speakers.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    canonical_name TEXT,
                    inferred_name TEXT,
                    confirmed_name TEXT,
                    gender TEXT,
                    voice_embedding BLOB,
                    num_samples INTEGER DEFAULT 1,
                    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL DEFAULT 0.0,
                    notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS speaker_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker_id TEXT NOT NULL,
                    transcript_path TEXT NOT NULL,
                    audio_path TEXT,
                    original_label TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id) ON DELETE CASCADE,
                    UNIQUE(speaker_id, transcript_path)
                )
            """)

            conn.commit()

    def _validate_and_prepare_embedding(self, voice_embedding: np.ndarray) -> Optional[np.ndarray]:
        """Validate and prepare embedding for storage."""
        if not isinstance(voice_embedding, np.ndarray):
            print("Error: voice_embedding must be a numpy array.")
            return None
        if voice_embedding.ndim == 0:
            print(f"Error: voice_embedding is a 0-dim array ({voice_embedding.item()}), expected {EMBEDDING_DIM} dimensions.")
            return None
        if voice_embedding.size != EMBEDDING_DIM:
            print(f"Error: voice_embedding has {voice_embedding.size} elements, expected {EMBEDDING_DIM}.")
            return None
        return voice_embedding.astype(EMBEDDING_DTYPE).reshape((EMBEDDING_DIM,))

    def add_speaker(self,
                    speaker_id: str,
                    voice_embedding: np.ndarray,
                    inferred_name: Optional[str] = None,
                    confirmed_name: Optional[str] = None,
                    gender: Optional[str] = None,
                    confidence_score: float = 0.0,
                    num_samples: int = 1,
                    notes: Optional[str] = None) -> bool:
        """Add a new speaker to the database."""
        prepared_embedding = self._validate_and_prepare_embedding(voice_embedding)
        if prepared_embedding is None:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                embedding_bytes = prepared_embedding.tobytes()

                cursor.execute("""
                    INSERT INTO speakers (
                        speaker_id, inferred_name, confirmed_name, gender,
                        voice_embedding, confidence_score, num_samples, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (speaker_id, inferred_name, confirmed_name, gender,
                      embedding_bytes, confidence_score, num_samples, notes))

                conn.commit()
                return True

        except sqlite3.IntegrityError:
            print(f"Error: Speaker ID {speaker_id} already exists.")
            return False
        except Exception as e:
            print(f"Error adding speaker {speaker_id}: {str(e)}")
            return False

    def update_speaker(self,
                       speaker_id: str,
                       confirmed_name: Optional[str] = None,
                       inferred_name: Optional[str] = None,
                       gender: Optional[str] = None,
                       voice_embedding: Optional[np.ndarray] = None,
                       num_samples: Optional[int] = None,
                       notes: Optional[str] = None) -> bool:
        """Update an existing speaker's information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                update_fields = []
                params = []

                if confirmed_name is not None:
                    update_fields.append("confirmed_name = ?")
                    params.append(confirmed_name)

                if inferred_name is not None:
                    update_fields.append("inferred_name = ?")
                    params.append(inferred_name)

                if gender is not None:
                    update_fields.append("gender = ?")
                    params.append(gender)

                if voice_embedding is not None:
                    prepared_embedding = self._validate_and_prepare_embedding(voice_embedding)
                    if prepared_embedding is None:
                        return False
                    update_fields.append("voice_embedding = ?")
                    params.append(prepared_embedding.tobytes())

                if num_samples is not None:
                    update_fields.append("num_samples = ?")
                    params.append(num_samples)

                if notes is not None:
                    update_fields.append("notes = ?")
                    params.append(notes)

                if update_fields:
                    update_fields.append("last_updated = CURRENT_TIMESTAMP")
                    query = f"""
                        UPDATE speakers
                        SET {', '.join(update_fields)}
                        WHERE speaker_id = ?
                    """
                    params.append(speaker_id)

                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount > 0

                return False

        except Exception as e:
            print(f"Error updating speaker {speaker_id}: {str(e)}")
            return False

    def _deserialize_embedding(self, embedding_bytes: Optional[bytes]) -> Optional[np.ndarray]:
        """Deserialize embedding from bytes with correct dtype and shape."""
        if embedding_bytes is None:
            return None
        embedding = np.frombuffer(embedding_bytes, dtype=EMBEDDING_DTYPE)
        if embedding.size == EMBEDDING_DIM:
            return embedding.reshape((EMBEDDING_DIM,))
        else:
            print(f"Warning: Stored embedding has unexpected size {embedding.size}, expected {EMBEDDING_DIM}. Skipping.")
            return None

    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """Retrieve a speaker's information from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT speaker_id, canonical_name, inferred_name, confirmed_name,
                           gender, voice_embedding, num_samples, first_detected,
                           last_updated, confidence_score, notes
                    FROM speakers
                    WHERE speaker_id = ?
                """, (speaker_id,))

                row = cursor.fetchone()
                if row:
                    speaker = dict(row)
                    speaker['voice_embedding'] = self._deserialize_embedding(row['voice_embedding'])
                    return speaker
                return None

        except Exception as e:
            print(f"Error retrieving speaker {speaker_id}: {str(e)}")
            return None

    def find_matching_speaker(self, voice_embedding: np.ndarray, threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """Find a matching speaker in the database based on voice embedding."""
        target_embedding = self._validate_and_prepare_embedding(voice_embedding)
        if target_embedding is None:
            print("Error: Invalid voice embedding provided for matching.")
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT speaker_id, voice_embedding FROM speakers WHERE voice_embedding IS NOT NULL")

                best_match_id = None
                highest_similarity = -1.0

                for row_speaker_id, stored_embedding_bytes in cursor.fetchall():
                    if stored_embedding_bytes:
                        stored_embedding = self._deserialize_embedding(stored_embedding_bytes)

                        if stored_embedding is None or stored_embedding.shape != (EMBEDDING_DIM,):
                            continue

                        similarity = np.dot(target_embedding, stored_embedding) / (
                            np.linalg.norm(target_embedding) * np.linalg.norm(stored_embedding)
                        )

                        if similarity > highest_similarity and similarity >= threshold:
                            highest_similarity = similarity
                            best_match_id = row_speaker_id

                return (best_match_id, float(highest_similarity)) if best_match_id else None

        except Exception as e:
            print(f"Error finding matching speaker: {str(e)}")
            return None

    def get_all_speakers(self) -> List[Dict]:
        """Retrieve all speakers from the database with appearance counts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.speaker_id, s.canonical_name, s.inferred_name,
                           s.confirmed_name, s.gender, s.voice_embedding,
                           s.num_samples, s.first_detected, s.last_updated,
                           s.confidence_score, s.notes,
                           COUNT(a.id) as appearance_count
                    FROM speakers s
                    LEFT JOIN speaker_appearances a ON s.speaker_id = a.speaker_id
                    GROUP BY s.speaker_id
                """)

                speakers = []
                for row in cursor.fetchall():
                    speaker_data = dict(row)
                    speaker_data['voice_embedding'] = self._deserialize_embedding(row['voice_embedding'])
                    speakers.append(speaker_data)
                return speakers

        except Exception as e:
            print(f"Error retrieving all speakers: {str(e)}")
            return []

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker and their appearances (CASCADE)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting speaker {speaker_id}: {str(e)}")
            return False

    def merge_speakers(self, source_id: str, target_id: str) -> bool:
        """Merge source speaker into target. Average embeddings weighted by num_samples."""
        try:
            source = self.get_speaker(source_id)
            target = self.get_speaker(target_id)

            if not source:
                print(f"Error: Source speaker {source_id} not found.")
                return False
            if not target:
                print(f"Error: Target speaker {target_id} not found.")
                return False

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")

                # Average embeddings weighted by num_samples
                if source['voice_embedding'] is not None and target['voice_embedding'] is not None:
                    s_n = source.get('num_samples', 1) or 1
                    t_n = target.get('num_samples', 1) or 1
                    total = s_n + t_n
                    merged_embedding = (source['voice_embedding'] * s_n + target['voice_embedding'] * t_n) / total
                    merged_embedding = merged_embedding.astype(EMBEDDING_DTYPE)
                    new_num_samples = total
                elif source['voice_embedding'] is not None:
                    merged_embedding = source['voice_embedding']
                    new_num_samples = source.get('num_samples', 1) or 1
                else:
                    merged_embedding = target['voice_embedding']
                    new_num_samples = target.get('num_samples', 1) or 1

                # Inherit names: target keeps its own, fills gaps from source
                new_confirmed = target.get('confirmed_name') or source.get('confirmed_name')
                new_inferred = target.get('inferred_name') or source.get('inferred_name')

                # Higher confidence wins
                new_confidence = max(
                    target.get('confidence_score', 0.0) or 0.0,
                    source.get('confidence_score', 0.0) or 0.0
                )

                # Update target
                update_params = [new_confirmed, new_inferred, new_num_samples, new_confidence]
                embedding_clause = ""
                if merged_embedding is not None:
                    embedding_clause = ", voice_embedding = ?"
                    update_params.append(merged_embedding.tobytes())

                update_params.append(target_id)

                cursor.execute(f"""
                    UPDATE speakers
                    SET confirmed_name = ?, inferred_name = ?,
                        num_samples = ?, confidence_score = ?,
                        last_updated = CURRENT_TIMESTAMP
                        {embedding_clause}
                    WHERE speaker_id = ?
                """, update_params)

                # Transfer appearances from source to target
                # Use INSERT OR IGNORE to skip duplicates (unique constraint)
                cursor.execute("""
                    INSERT OR IGNORE INTO speaker_appearances (speaker_id, transcript_path, audio_path, original_label)
                    SELECT ?, transcript_path, audio_path, original_label
                    FROM speaker_appearances
                    WHERE speaker_id = ?
                """, (target_id, source_id))

                # Delete source speaker (CASCADE removes remaining appearances)
                cursor.execute("DELETE FROM speakers WHERE speaker_id = ?", (source_id,))

                conn.commit()
                return True

        except Exception as e:
            print(f"Error merging speakers {source_id} -> {target_id}: {str(e)}")
            return False

    def add_appearance(self, speaker_id: str, transcript_path: str,
                       audio_path: Optional[str] = None,
                       original_label: Optional[str] = None) -> bool:
        """Record a speaker appearance in a transcript."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO speaker_appearances
                        (speaker_id, transcript_path, audio_path, original_label)
                    VALUES (?, ?, ?, ?)
                """, (speaker_id, transcript_path, audio_path, original_label))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error adding appearance for {speaker_id}: {str(e)}")
            return False

    def get_appearances(self, speaker_id: str) -> List[Dict]:
        """Get all appearances for a speaker."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, speaker_id, transcript_path, audio_path,
                           original_label, created_at
                    FROM speaker_appearances
                    WHERE speaker_id = ?
                    ORDER BY created_at DESC
                """, (speaker_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting appearances for {speaker_id}: {str(e)}")
            return []
