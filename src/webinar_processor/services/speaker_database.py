import sqlite3
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import os

# Define constants for embedding dimensions and dtype
EMBEDDING_DIM = 192
EMBEDDING_DTYPE = np.float32

class SpeakerDatabase:
    """Service for managing speaker profiles and identifications."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the speaker database.
        
        Args:
            db_path: Optional path to the SQLite database file. If not provided,
                    will use a default location in the user's home directory.
        """
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
            
            # Create speakers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    canonical_name TEXT,
                    inferred_name TEXT,
                    confirmed_name TEXT,
                    gender TEXT,
                    voice_embedding BLOB,
                    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL DEFAULT 0.0
                )
            """)

            conn.commit()
    
    def _validate_and_prepare_embedding(self, voice_embedding: np.ndarray) -> Optional[np.ndarray]:
        """Validate and prepare embedding for storage."""
        if not isinstance(voice_embedding, np.ndarray):
            print("Error: voice_embedding must be a numpy array.")
            return None
        if voice_embedding.ndim == 0: # Handle 0-dim arrays that can arise from some np operations
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
                   gender: Optional[str] = None) -> bool:
        """
        Add a new speaker to the database.
        
        Args:
            speaker_id: Unique identifier for the speaker
            voice_embedding: Voice embedding vector (must be {EMBEDDING_DIM}-dimensional)
            inferred_name: Name inferred from context (optional)
            gender: Detected gender (optional)
            
        Returns:
            bool: True if speaker was added successfully
        """
        prepared_embedding = self._validate_and_prepare_embedding(voice_embedding)
        if prepared_embedding is None:
            return False
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                embedding_bytes = prepared_embedding.tobytes()
                
                cursor.execute("""
                    INSERT INTO speakers (
                        speaker_id, inferred_name, gender, 
                        voice_embedding
                    ) VALUES (?, ?, ?, ?)
                """, (speaker_id, inferred_name, gender, 
                      embedding_bytes))
                
                conn.commit()
                return True
                
        except sqlite3.IntegrityError:
            # Speaker ID already exists
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
                      voice_embedding: Optional[np.ndarray] = None) -> bool:
        """
        Update an existing speaker's information.
        
        Args:
            speaker_id: ID of the speaker to update
            confirmed_name: Manually confirmed name (optional)
            inferred_name: Updated inferred name (optional)
            gender: Updated gender (optional)
            voice_embedding: Updated voice embedding (must be {EMBEDDING_DIM}-dimensional) (optional)
            
        Returns:
            bool: True if update was successful
        """
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
                        return False # Validation failed
                    update_fields.append("voice_embedding = ?")
                    params.append(prepared_embedding.tobytes())
                
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
                
                return False # No fields to update
                
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
            # This case should ideally not happen if data is stored correctly.
            # It indicates a potential issue with previously stored data or a bug.
            print(f"Warning: Stored embedding has unexpected size {embedding.size}, expected {EMBEDDING_DIM}. Returning as is, but might cause issues.")
            # Attempt to reshape if possible, otherwise return raw buffer to avoid crashing,
            # though this will likely lead to errors downstream.
            # A more robust solution might involve trying to pad/truncate or skip the record.
            return embedding


    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """
        Retrieve a speaker's information from the database.
        
        Args:
            speaker_id: ID of the speaker to retrieve
            
        Returns:
            Dict containing speaker information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Access columns by name
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT speaker_id, canonical_name, inferred_name, confirmed_name,
                           gender, voice_embedding, first_detected, last_updated,
                           confidence_score
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
        """
        Find a matching speaker in the database based on voice embedding.
        
        Args:
            voice_embedding: Voice embedding vector to match ({EMBEDDING_DIM}-dimensional, {EMBEDDING_DTYPE})
            threshold: Similarity threshold (0-1)
            
        Returns:
            Tuple of (speaker_id, similarity_score) or None if no match found
        """
        # Ensure incoming embedding is correctly shaped and typed for comparison
        target_embedding = self._validate_and_prepare_embedding(voice_embedding)
        if target_embedding is None:
            print("Error: Invalid voice embedding provided for matching.")
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT speaker_id, voice_embedding FROM speakers WHERE voice_embedding IS NOT NULL")
                
                best_match_id = None
                highest_similarity = -1.0 # Cosine similarity ranges from -1 to 1
                
                for row_speaker_id, stored_embedding_bytes in cursor.fetchall():
                    if stored_embedding_bytes:
                        stored_embedding = self._deserialize_embedding(stored_embedding_bytes)
                        
                        if stored_embedding is None or stored_embedding.shape != (EMBEDDING_DIM,):
                            print(f"Warning: Skipping speaker {row_speaker_id} due to invalid stored embedding shape after deserialization: {stored_embedding.shape if stored_embedding is not None else 'None'}")
                            continue

                        # Calculate cosine similarity
                        # Ensure both are 1D arrays for dot product
                        similarity = np.dot(target_embedding, stored_embedding) / (
                            np.linalg.norm(target_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > highest_similarity and similarity >= threshold:
                            highest_similarity = similarity
                            best_match_id = row_speaker_id
                
                return (best_match_id, highest_similarity) if best_match_id else None
                
        except Exception as e:
            print(f"Error finding matching speaker: {str(e)}")
            return None

    def get_all_speakers(self) -> List[Dict]:
        """
        Retrieve all speakers from the database.
        
        Returns:
            List of speaker dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Access columns by name
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT speaker_id, canonical_name, inferred_name, confirmed_name,
                           gender, voice_embedding, first_detected, last_updated,
                           confidence_score
                    FROM speakers
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

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a temporary database for testing
    db = SpeakerDatabase(db_path=":memory:")

    # Test adding a speaker with a correctly shaped embedding
    correct_embedding_1 = np.random.rand(EMBEDDING_DIM).astype(EMBEDDING_DTYPE)
    print(f"Adding speaker_01 with embedding shape: {correct_embedding_1.shape}, dtype: {correct_embedding_1.dtype}")
    db.add_speaker("speaker_01", correct_embedding_1, inferred_name="Speaker One")

    # Test adding a speaker with a differently shaped embedding (should fail or be reshaped by validation)
    incorrect_embedding_shape = np.random.rand(96).astype(EMBEDDING_DTYPE)
    print(f"Attempting to add speaker_02 with embedding shape: {incorrect_embedding_shape.shape}")
    db.add_speaker("speaker_02", incorrect_embedding_shape, inferred_name="Speaker Two (Incorrect Shape)")

    # Test adding a speaker with a different dtype embedding
    correct_embedding_float64 = np.random.rand(EMBEDDING_DIM).astype(np.float64)
    print(f"Adding speaker_03 with embedding shape: {correct_embedding_float64.shape}, dtype: {correct_embedding_float64.dtype}")
    db.add_speaker("speaker_03", correct_embedding_float64, inferred_name="Speaker Three (Float64)")
    
    # Retrieve and check speaker_01
    retrieved_speaker_01 = db.get_speaker("speaker_01")
    if retrieved_speaker_01 and retrieved_speaker_01['voice_embedding'] is not None:
        print(f"Retrieved speaker_01. Embedding shape: {retrieved_speaker_01['voice_embedding'].shape}, dtype: {retrieved_speaker_01['voice_embedding'].dtype}")
        assert retrieved_speaker_01['voice_embedding'].shape == (EMBEDDING_DIM,)
        assert retrieved_speaker_01['voice_embedding'].dtype == EMBEDDING_DTYPE
    else:
        print("Failed to retrieve speaker_01 or its embedding.")

    # Retrieve and check speaker_03 (should have been converted to EMBEDDING_DTYPE)
    retrieved_speaker_03 = db.get_speaker("speaker_03")
    if retrieved_speaker_03 and retrieved_speaker_03['voice_embedding'] is not None:
        print(f"Retrieved speaker_03. Embedding shape: {retrieved_speaker_03['voice_embedding'].shape}, dtype: {retrieved_speaker_03['voice_embedding'].dtype}")
        assert retrieved_speaker_03['voice_embedding'].shape == (EMBEDDING_DIM,)
        assert retrieved_speaker_03['voice_embedding'].dtype == EMBEDDING_DTYPE
    else:
        print("Failed to retrieve speaker_03 or its embedding.")

    # Test finding a matching speaker
    test_embedding_match = correct_embedding_1 + np.random.normal(0, 0.1, EMBEDDING_DIM).astype(EMBEDDING_DTYPE) # Slightly noisy version
    print(f"Finding match for embedding with shape: {test_embedding_match.shape}")

    match_result = db.find_matching_speaker(test_embedding_match, threshold=0.5)
    if match_result:
        print(f"Found matching speaker: {match_result[0]} with score {match_result[1]}")
    else:
        print("No matching speaker found for test_embedding_match.")
        
    # Test with an embedding that should not match anything (or has wrong dimension)
    non_matching_embedding = np.random.rand(EMBEDDING_DIM).astype(EMBEDDING_DTYPE)
    print(f"Finding match for a random non-matching embedding shape: {non_matching_embedding.shape}")
    match_result_none = db.find_matching_speaker(non_matching_embedding, threshold=0.99)
    if match_result_none:
        print(f"Unexpected match: {match_result_none[0]} with score {match_result_none[1]}")
    else:
        print("Correctly found no matching speaker for random embedding.")

    # Test with an incorrectly shaped embedding for matching
    incorrect_shape_for_matching = np.random.rand(96).astype(EMBEDDING_DTYPE)
    print(f"Finding match for an incorrectly shaped embedding: {incorrect_shape_for_matching.shape}")
    match_result_bad_shape = db.find_matching_speaker(incorrect_shape_for_matching)
    if match_result_bad_shape:
         print(f"Unexpected match with bad shape: {match_result_bad_shape[0]} with score {match_result_bad_shape[1]}")
    else:
        print("Correctly found no match or errored for bad shape embedding during matching.")


    # Test get_all_speakers
    all_speakers = db.get_all_speakers()
    print(f"All speakers ({len(all_speakers)}):")
    for spk in all_speakers:
        emb_info = f"shape: {spk['voice_embedding'].shape}, dtype: {spk['voice_embedding'].dtype}" if spk['voice_embedding'] is not None else "No embedding"
        print(f"  ID: {spk['speaker_id']}, Name: {spk.get('inferred_name', 'N/A')}, Embedding: {emb_info}")

    # Test case where an embedding in DB might be malformed (e.g. wrong size)
    # Manually insert a "bad" embedding to test deserialization robustness
    conn = sqlite3.connect(":memory:") # Use the same in-memory DB for this direct manipulation if needed, or the file path
    # For this test, let's assume `db` is using ":memory:" as per the example start
    # If db = SpeakerDatabase() without path, it creates a file. For isolated test, use ":memory:"
    # We need to ensure this write happens to the SAME db instance `db` is using.
    # The current if __name__ == "__main__" creates a new :memory: db each time.
    # To properly test this part, we would need to use `db.db_path`
    
    # This specific test for malformed data in DB is harder to inject into the :memory: db
    # without direct access to the connection object `db` uses internally or by creating a file-based DB.
    # The _deserialize_embedding method now includes a warning for unexpected sizes.
    print("Test with intentionally malformed embedding (requires manual DB entry or more complex setup)")
    # Simulating a read of such data:
    malformed_bytes = np.random.rand(100).astype(EMBEDDING_DTYPE).tobytes() # e.g. 100 elements instead of 192
    deserialized_malformed = db._deserialize_embedding(malformed_bytes)
    if deserialized_malformed is not None:
        print(f"Deserialized malformed embedding: shape {deserialized_malformed.shape}, dtype {deserialized_malformed.dtype}")
    else:
        print("Deserialized malformed embedding returned None (as expected if bytes were None).")
    print("SpeakerDatabase tests finished.")
