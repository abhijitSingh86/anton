"""Knowledge store with vector similarity search."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modular_agents.core.models import (
    CodeChunk,
    ChunkRelation,
    IndexedRepo,
    ProjectType,
    TaskLearning,
)

if TYPE_CHECKING:
    from modular_agents.knowledge.base import EmbeddingProvider


class KnowledgeStore:
    """SQLite-based knowledge store with vector similarity search.

    Uses hnswlib for fast vector operations (pip-installable, no compilation needed).
    Falls back to numpy-based search if hnswlib is not available.
    """

    def __init__(
        self,
        db_path: Path,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize the knowledge store.

        Args:
            db_path: Path to SQLite database file
            embedding_provider: Optional embedding provider for generating vectors
        """
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self._conn: sqlite3.Connection | None = None
        self._vector_index = None  # Will be initialized on connect

    def connect(self) -> None:
        """Connect to the database and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys = ON")

        # Initialize vector index
        from modular_agents.knowledge.vector_index import (
            FallbackVectorIndex,
            VectorIndex,
        )

        dimension = self.embedding_provider.dimension if self.embedding_provider else 300

        # Try to use hnswlib, fall back to numpy if not available
        vector_index = VectorIndex(dimension=dimension)
        if vector_index.is_available():
            self._vector_index = vector_index
        else:
            import warnings
            warnings.warn(
                "hnswlib not found. Using slower numpy-based similarity search. "
                "For better performance, install with: pip install 'anton[knowledge]'",
                RuntimeWarning
            )
            self._vector_index = FallbackVectorIndex(dimension=dimension)

        # Try to load existing vector index
        index_dir = self.db_path.parent / ".vector_index"
        if index_dir.exists():
            self._vector_index.load(index_dir)

        # Create schema
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        # Indexed repositories table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_repos (
                repo_path TEXT PRIMARY KEY,
                project_type TEXT NOT NULL,
                language TEXT NOT NULL,
                total_chunks INTEGER DEFAULT 0,
                total_files INTEGER DEFAULT 0,
                indexed_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        # Code chunks table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id TEXT PRIMARY KEY,
                repo_path TEXT NOT NULL,
                file_path TEXT NOT NULL,
                module_name TEXT NOT NULL,
                language TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                summary TEXT,
                purpose TEXT,
                dependencies TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (repo_path) REFERENCES indexed_repos(repo_path)
            )
        """)

        # Indexed files table (for tracking file modification times)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                repo_path TEXT NOT NULL,
                file_path TEXT NOT NULL,
                module_name TEXT NOT NULL,
                language TEXT NOT NULL,
                file_mtime REAL NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY (repo_path, file_path),
                FOREIGN KEY (repo_path) REFERENCES indexed_repos(repo_path)
            )
        """)

        # Create indices for common queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_repo
            ON code_chunks(repo_path)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_module
            ON code_chunks(module_name)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_type
            ON code_chunks(chunk_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_name
            ON code_chunks(name)
        """)

        # Chunk relations table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_relations (
                source_chunk_id TEXT NOT NULL,
                target_chunk_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (source_chunk_id, target_chunk_id, relation_type),
                FOREIGN KEY (source_chunk_id) REFERENCES code_chunks(id),
                FOREIGN KEY (target_chunk_id) REFERENCES code_chunks(id)
            )
        """)

        # Task learnings table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_learnings (
                id TEXT PRIMARY KEY,
                task_description TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                module_names TEXT,
                patterns_learned TEXT,
                code_chunks TEXT,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                completed_at TEXT NOT NULL,
                FOREIGN KEY (repo_path) REFERENCES indexed_repos(repo_path)
            )
        """)

        # Note: Embeddings are now stored in the vector index (hnswlib or fallback)
        # We don't need an embeddings table in SQLite anymore

        self._conn.commit()

    def close(self) -> None:
        """Close database connection and save vector index."""
        # Save vector index
        if self._vector_index:
            index_dir = self.db_path.parent / ".vector_index"
            self._vector_index.save(index_dir)

        # Close database
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_repo(self, repo: IndexedRepo) -> None:
        """Add or update indexed repository metadata."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO indexed_repos
            (repo_path, project_type, language, total_chunks, total_files, indexed_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                repo.repo_path,
                repo.project_type.value,
                repo.language,
                repo.total_chunks,
                repo.total_files,
                repo.indexed_at.isoformat(),
                repo.last_updated.isoformat(),
            ),
        )
        self._conn.commit()

    def get_repo(self, repo_path: str) -> IndexedRepo | None:
        """Get indexed repository metadata."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        row = self._conn.execute(
            "SELECT * FROM indexed_repos WHERE repo_path = ?",
            (repo_path,),
        ).fetchone()

        if not row:
            return None

        return IndexedRepo(
            repo_path=row["repo_path"],
            project_type=ProjectType(row["project_type"]),
            language=row["language"],
            total_chunks=row["total_chunks"],
            total_files=row["total_files"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
            last_updated=datetime.fromisoformat(row["last_updated"]),
        )

    def list_repos(self) -> list[IndexedRepo]:
        """List all indexed repositories.

        Returns:
            List of IndexedRepo objects, sorted by last_updated (newest first)
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        rows = self._conn.execute(
            "SELECT * FROM indexed_repos ORDER BY last_updated DESC"
        ).fetchall()

        return [
            IndexedRepo(
                repo_path=row["repo_path"],
                project_type=ProjectType(row["project_type"]),
                language=row["language"],
                total_chunks=row["total_chunks"],
                total_files=row["total_files"],
                indexed_at=datetime.fromisoformat(row["indexed_at"]),
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )
            for row in rows
        ]

    async def add_chunk(self, chunk: CodeChunk, generate_embedding: bool = True) -> None:
        """Add a code chunk with optional embedding generation."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        # Generate embedding if requested and provider available
        if generate_embedding and self.embedding_provider and not chunk.embedding:
            chunk.embedding = await self.embedding_provider.embed_text(chunk.content)

        # Insert chunk
        self._conn.execute(
            """
            INSERT OR REPLACE INTO code_chunks
            (id, repo_path, file_path, module_name, language, chunk_type, name,
             content, start_line, end_line, summary, purpose, dependencies,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                chunk.repo_path,
                chunk.file_path,
                chunk.module_name,
                chunk.language,
                chunk.chunk_type,
                chunk.name,
                chunk.content,
                chunk.start_line,
                chunk.end_line,
                chunk.summary,
                chunk.purpose,
                json.dumps(chunk.dependencies),
                chunk.created_at.isoformat(),
                chunk.updated_at.isoformat(),
            ),
        )

        # Add embedding to vector index if available
        if chunk.embedding and self._vector_index:
            self._vector_index.add_vector(chunk.id, chunk.embedding)

        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """Get a code chunk by ID."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        row = self._conn.execute(
            "SELECT * FROM code_chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()

        if not row:
            return None

        # Note: Embeddings are stored in vector index, not in SQL
        return CodeChunk(
            id=row["id"],
            repo_path=row["repo_path"],
            file_path=row["file_path"],
            module_name=row["module_name"],
            language=row["language"],
            chunk_type=row["chunk_type"],
            name=row["name"],
            content=row["content"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            summary=row["summary"] or "",
            purpose=row["purpose"] or "",
            dependencies=json.loads(row["dependencies"]) if row["dependencies"] else [],
            embedding=None,  # Embeddings stored in vector index
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        repo_path: str | None = None,
        module_name: str | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Search for similar code chunks using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            repo_path: Optional filter by repository
            module_name: Optional filter by module

        Returns:
            List of (chunk, similarity_score) tuples, ordered by similarity
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        if not self.embedding_provider:
            raise RuntimeError("No embedding provider configured")

        if not self._vector_index:
            raise RuntimeError("Vector index not initialized")

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)

        # Search vector index (gets more results than limit for filtering)
        vector_results = self._vector_index.search(query_embedding, k=limit * 3)

        # Fetch chunk details from database and apply filters
        results = []
        for chunk_id, similarity in vector_results:
            # Get chunk from database
            chunk = self.get_chunk(chunk_id)
            if not chunk:
                continue

            # Apply filters
            if repo_path and chunk.repo_path != repo_path:
                continue
            if module_name and chunk.module_name != module_name:
                continue

            results.append((chunk, similarity))

            # Stop when we have enough results
            if len(results) >= limit:
                break

        return results

    def add_relation(self, relation: ChunkRelation) -> None:
        """Add a relationship between code chunks."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunk_relations
            (source_chunk_id, target_chunk_id, relation_type, weight)
            VALUES (?, ?, ?, ?)
            """,
            (
                relation.source_chunk_id,
                relation.target_chunk_id,
                relation.relation_type,
                relation.weight,
            ),
        )
        self._conn.commit()

    def add_indexed_file(
        self,
        repo_path: str,
        file_path: str,
        module_name: str,
        language: str,
        file_mtime: float,
        chunk_count: int,
    ) -> None:
        """Add or update indexed file metadata.

        Args:
            repo_path: Repository path
            file_path: Relative file path
            module_name: Module name
            language: Programming language
            file_mtime: File modification time (timestamp)
            chunk_count: Number of chunks created from this file
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO indexed_files
            (repo_path, file_path, module_name, language, file_mtime, chunk_count, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                repo_path,
                file_path,
                module_name,
                language,
                file_mtime,
                chunk_count,
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

    def get_indexed_file(self, repo_path: str, file_path: str) -> dict | None:
        """Get indexed file metadata.

        Args:
            repo_path: Repository path
            file_path: Relative file path

        Returns:
            Dictionary with file metadata or None if not found
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        row = self._conn.execute(
            "SELECT * FROM indexed_files WHERE repo_path = ? AND file_path = ?",
            (repo_path, file_path),
        ).fetchone()

        if not row:
            return None

        return {
            "repo_path": row["repo_path"],
            "file_path": row["file_path"],
            "module_name": row["module_name"],
            "language": row["language"],
            "file_mtime": row["file_mtime"],
            "chunk_count": row["chunk_count"],
            "indexed_at": datetime.fromisoformat(row["indexed_at"]),
        }

    def needs_reindex(self, repo_path: str, file_path: str, current_mtime: float) -> bool:
        """Check if a file needs to be re-indexed.

        Args:
            repo_path: Repository path
            file_path: Relative file path
            current_mtime: Current file modification time

        Returns:
            True if file needs re-indexing (new or modified)
        """
        indexed_file = self.get_indexed_file(repo_path, file_path)

        if not indexed_file:
            # File not indexed yet
            return True

        # Check if file was modified
        return current_mtime > indexed_file["file_mtime"]

    def delete_file_chunks(self, repo_path: str, file_path: str) -> int:
        """Delete all chunks for a specific file.

        Args:
            repo_path: Repository path
            file_path: Relative file path

        Returns:
            Number of chunks deleted
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        # Get chunk IDs first (to remove from vector index)
        rows = self._conn.execute(
            "SELECT id FROM code_chunks WHERE repo_path = ? AND file_path = ?",
            (repo_path, file_path),
        ).fetchall()

        chunk_ids = [row["id"] for row in rows]

        # Remove from vector index
        if self._vector_index and chunk_ids:
            for chunk_id in chunk_ids:
                self._vector_index.remove_vector(chunk_id)

        # Delete from database
        cursor = self._conn.execute(
            "DELETE FROM code_chunks WHERE repo_path = ? AND file_path = ?",
            (repo_path, file_path),
        )

        deleted = cursor.rowcount
        self._conn.commit()

        return deleted

    def list_indexed_files(self, repo_path: str) -> list[dict]:
        """List all indexed files for a repository.

        Args:
            repo_path: Repository path

        Returns:
            List of file metadata dictionaries
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        rows = self._conn.execute(
            "SELECT * FROM indexed_files WHERE repo_path = ? ORDER BY file_path",
            (repo_path,),
        ).fetchall()

        return [
            {
                "repo_path": row["repo_path"],
                "file_path": row["file_path"],
                "module_name": row["module_name"],
                "language": row["language"],
                "file_mtime": row["file_mtime"],
                "chunk_count": row["chunk_count"],
                "indexed_at": datetime.fromisoformat(row["indexed_at"]),
            }
            for row in rows
        ]

    def delete_indexed_file(self, repo_path: str, file_path: str) -> None:
        """Delete indexed file metadata.

        Args:
            repo_path: Repository path
            file_path: Relative file path
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        self._conn.execute(
            "DELETE FROM indexed_files WHERE repo_path = ? AND file_path = ?",
            (repo_path, file_path),
        )
        self._conn.commit()

    def add_task_learning(self, learning: TaskLearning) -> None:
        """Add task learning to knowledge base."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO task_learnings
            (id, task_description, repo_path, module_names, patterns_learned,
             code_chunks, success, error_message, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learning.id,
                learning.task_description,
                learning.repo_path,
                json.dumps(learning.module_names),
                json.dumps(learning.patterns_learned),
                json.dumps(learning.code_chunks),
                1 if learning.success else 0,
                learning.error_message,
                learning.completed_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        stats = {}

        # Count repositories
        row = self._conn.execute("SELECT COUNT(*) as count FROM indexed_repos").fetchone()
        stats["total_repos"] = row["count"]

        # Count chunks
        row = self._conn.execute("SELECT COUNT(*) as count FROM code_chunks").fetchone()
        stats["total_chunks"] = row["count"]

        # Count embeddings from vector index
        if self._vector_index:
            index_stats = self._vector_index.get_stats()
            stats["total_embeddings"] = index_stats.get("vectors", 0)
            stats["vector_index_type"] = "hnswlib" if index_stats.get("fallback") is None else "numpy_fallback"
        else:
            stats["total_embeddings"] = 0
            stats["vector_index_type"] = "none"

        # Count relations
        row = self._conn.execute("SELECT COUNT(*) as count FROM chunk_relations").fetchone()
        stats["total_relations"] = row["count"]

        # Count task learnings
        row = self._conn.execute("SELECT COUNT(*) as count FROM task_learnings").fetchone()
        stats["total_learnings"] = row["count"]

        return stats
