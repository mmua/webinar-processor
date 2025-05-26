import sqlite3
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import os

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
                    confidence_score REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """)

            conn.commit()
    
    def add_speaker(self, 
                   speaker_id: str,
                   voice_embedding: np.ndarray,
                   inferred_name: Optional[str] = None,
                   gender: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Add a new speaker to the database.
        
        Args:
            speaker_id: Unique identifier for the speaker
            voice_embedding: Voice embedding vector
            inferred_name: Name inferred from context (optional)
            gender: Detected gender (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            bool: True if speaker was added successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert numpy array to bytes for storage
                embedding_bytes = voice_embedding.tobytes()
                
                # Convert metadata dict to JSON string
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute("""
                    INSERT INTO speakers (
                        speaker_id, inferred_name, gender, 
                        voice_embedding, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                """, (speaker_id, inferred_name, gender, 
                      embedding_bytes, metadata_json))
                
                conn.commit()
                return True
                
        except sqlite3.IntegrityError:
            # Speaker ID already exists
            return False
        except Exception as e:
            print(f"Error adding speaker: {str(e)}")
            return False
    
    def update_speaker(self,
                      speaker_id: str,
                      confirmed_name: Optional[str] = None,
                      inferred_name: Optional[str] = None,
                      gender: Optional[str] = None,
                      voice_embedding: Optional[np.ndarray] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing speaker's information.
        
        Args:
            speaker_id: ID of the speaker to update
            confirmed_name: Manually confirmed name (optional)
            inferred_name: Updated inferred name (optional)
            gender: Updated gender (optional)
            voice_embedding: Updated voice embedding (optional)
            metadata: Updated metadata (optional)
            
        Returns:
            bool: True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query dynamically based on provided fields
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
                    update_fields.append("voice_embedding = ?")
                    params.append(voice_embedding.tobytes())
                
                if metadata is not None:
                    update_fields.append("metadata = ?")
                    params.append(json.dumps(metadata))
                
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
            print(f"Error updating speaker: {str(e)}")
            return False
    
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
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT speaker_id, canonical_name, inferred_name, confirmed_name,
                           gender, voice_embedding, first_detected, last_updated,
                           confidence_score, metadata
                    FROM speakers
                    WHERE speaker_id = ?
                """, (speaker_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dictionary
                    speaker = {
                        'speaker_id': row[0],
                        'canonical_name': row[1],
                        'inferred_name': row[2],
                        'confirmed_name': row[3],
                        'gender': row[4],
                        'voice_embedding': np.frombuffer(row[5]) if row[5] else None,
                        'first_detected': row[6],
                        'last_updated': row[7],
                        'confidence_score': row[8],
                        'metadata': json.loads(row[9]) if row[9] else None
                    }
                    return speaker
                return None
                
        except Exception as e:
            print(f"Error retrieving speaker: {str(e)}")
            return None

    def find_matching_speaker(self, voice_embedding: np.ndarray, threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Find a matching speaker in the database based on voice embedding.
        
        Args:
            voice_embedding: Voice embedding vector to match
            threshold: Similarity threshold (0-1)
            
        Returns:
            Tuple of (speaker_id, similarity_score) or None if no match found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT speaker_id, voice_embedding FROM speakers WHERE voice_embedding IS NOT NULL")
                
                best_match = None
                best_score = 0.0
                
                for speaker_id, stored_embedding in cursor.fetchall():
                    if stored_embedding:
                        stored_embedding = np.frombuffer(stored_embedding)
                        # Calculate cosine similarity
                        similarity = np.dot(voice_embedding, stored_embedding) / (
                            np.linalg.norm(voice_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > best_score and similarity >= threshold:
                            best_score = similarity
                            best_match = speaker_id
                
                return (best_match, best_score) if best_match else None
                
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
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT speaker_id, canonical_name, inferred_name, confirmed_name,
                           gender, voice_embedding, first_detected, last_updated,
                           confidence_score, metadata
                    FROM speakers
                """)
                
                speakers = []
                for row in cursor.fetchall():
                    speaker = {
                        'speaker_id': row[0],
                        'canonical_name': row[1],
                        'inferred_name': row[2],
                        'confirmed_name': row[3],
                        'gender': row[4],
                        'voice_embedding': np.frombuffer(row[5]) if row[5] else None,
                        'first_detected': row[6],
                        'last_updated': row[7],
                        'confidence_score': row[8],
                        'metadata': json.loads(row[9]) if row[9] else None
                    }
                    speakers.append(speaker)
                return speakers
                
        except Exception as e:
            print(f"Error retrieving speakers: {str(e)}")
            return []
