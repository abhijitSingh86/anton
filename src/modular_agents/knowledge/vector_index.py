"""Vector index using HNSW for fast similarity search."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class VectorIndex:
    """Vector index using hnswlib for fast approximate nearest neighbor search.

    This is automatically installed with the knowledge dependencies and provides
    fast vector search without requiring external SQLite extensions.
    """

    def __init__(self, dimension: int, max_elements: int = 100000):
        """Initialize the vector index.

        Args:
            dimension: Dimensionality of vectors
            max_elements: Maximum number of vectors to store
        """
        self.dimension = dimension
        self.max_elements = max_elements
        self._index = None
        self._id_to_idx = {}  # Map chunk_id to index position
        self._idx_to_id = {}  # Map index position to chunk_id
        self._next_idx = 0

        try:
            import hnswlib
            self._hnswlib = hnswlib
            self._available = True
        except ImportError:
            self._hnswlib = None
            self._available = False

    def is_available(self) -> bool:
        """Check if hnswlib is available."""
        return self._available

    def initialize(self):
        """Initialize the HNSW index."""
        if not self._available:
            raise ImportError(
                "hnswlib is required for vector search. "
                "Install with: pip install 'anton[knowledge]'"
            )

        self._index = self._hnswlib.Index(space='cosine', dim=self.dimension)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=200,  # Controls index quality
            M=16,  # Number of connections per element
        )
        self._index.set_ef(50)  # Controls search quality

    def add_vector(self, chunk_id: str, vector: list[float]):
        """Add a vector to the index.

        Args:
            chunk_id: Unique identifier for the chunk
            vector: Vector embedding
        """
        if not self._available:
            return

        if self._index is None:
            self.initialize()

        # Convert to numpy array
        vec_array = np.array(vector, dtype=np.float32).reshape(1, -1)

        # Add to index
        idx = self._next_idx
        self._index.add_items(vec_array, np.array([idx]))

        # Track mapping
        self._id_to_idx[chunk_id] = idx
        self._idx_to_id[idx] = chunk_id
        self._next_idx += 1

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            List of (chunk_id, distance) tuples, sorted by similarity
        """
        if not self._available or self._index is None or self._next_idx == 0:
            return []

        # Convert to numpy array
        query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Search
        labels, distances = self._index.knn_query(query_array, k=min(k, self._next_idx))

        # Convert to chunk IDs
        results = []
        for idx, distance in zip(labels[0], distances[0]):
            chunk_id = self._idx_to_id.get(int(idx))
            if chunk_id:
                # Convert cosine distance to similarity score (1 - distance)
                similarity = 1.0 - float(distance)
                results.append((chunk_id, similarity))

        return results

    def remove_vector(self, chunk_id: str):
        """Remove a vector from the index.

        Note: hnswlib doesn't support true deletion, so we just remove from mappings.
        For production use, periodically rebuild the index.
        """
        if chunk_id in self._id_to_idx:
            idx = self._id_to_idx[chunk_id]
            del self._id_to_idx[chunk_id]
            del self._idx_to_id[idx]

    def save(self, path: Path):
        """Save the index to disk.

        Args:
            path: Directory to save index files
        """
        if not self._available or self._index is None:
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save HNSW index
        index_file = path / "vector.hnsw"
        self._index.save_index(str(index_file))

        # Save mappings
        mappings = {
            "id_to_idx": self._id_to_idx,
            "idx_to_id": self._idx_to_id,
            "next_idx": self._next_idx,
            "dimension": self.dimension,
            "max_elements": self.max_elements,
        }
        mappings_file = path / "mappings.pkl"
        with open(mappings_file, "wb") as f:
            pickle.dump(mappings, f)

    def load(self, path: Path) -> bool:
        """Load the index from disk.

        Args:
            path: Directory containing index files

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self._available:
            return False

        path = Path(path)
        index_file = path / "vector.hnsw"
        mappings_file = path / "mappings.pkl"

        if not index_file.exists() or not mappings_file.exists():
            return False

        try:
            # Load mappings
            with open(mappings_file, "rb") as f:
                mappings = pickle.load(f)

            self.dimension = mappings["dimension"]
            self.max_elements = mappings["max_elements"]
            self._id_to_idx = mappings["id_to_idx"]
            self._idx_to_id = mappings["idx_to_id"]
            self._next_idx = mappings["next_idx"]

            # Load HNSW index
            self._index = self._hnswlib.Index(space='cosine', dim=self.dimension)
            self._index.load_index(str(index_file), max_elements=self.max_elements)
            self._index.set_ef(50)

            return True

        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "available": self._available,
            "initialized": self._index is not None,
            "vectors": self._next_idx,
            "dimension": self.dimension,
            "max_elements": self.max_elements,
        }


class FallbackVectorIndex:
    """Fallback vector index using pure NumPy for when hnswlib is not available."""

    def __init__(self, dimension: int, max_elements: int = 100000):
        """Initialize the fallback index.

        Args:
            dimension: Dimensionality of vectors
            max_elements: Maximum number of vectors (ignored for numpy)
        """
        self.dimension = dimension
        self._vectors = {}  # chunk_id -> vector

    def is_available(self) -> bool:
        """Always available (uses numpy)."""
        return True

    def initialize(self):
        """No initialization needed."""
        pass

    def add_vector(self, chunk_id: str, vector: list[float]):
        """Add a vector."""
        self._vectors[chunk_id] = np.array(vector, dtype=np.float32)

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search using brute-force cosine similarity."""
        if not self._vectors:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)

        # Compute cosine similarities
        similarities = []
        for chunk_id, vec in self._vectors.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query, vec) / (query_norm * vec_norm))
            similarities.append((chunk_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def remove_vector(self, chunk_id: str):
        """Remove a vector."""
        if chunk_id in self._vectors:
            del self._vectors[chunk_id]

    def save(self, path: Path):
        """Save vectors to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "vectors": {k: v.tolist() for k, v in self._vectors.items()},
        }

        vectors_file = path / "vectors_fallback.json"
        with open(vectors_file, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> bool:
        """Load vectors from disk."""
        path = Path(path)
        vectors_file = path / "vectors_fallback.json"

        if not vectors_file.exists():
            return False

        try:
            with open(vectors_file, "r") as f:
                data = json.load(f)

            self.dimension = data["dimension"]
            self._vectors = {
                k: np.array(v, dtype=np.float32)
                for k, v in data["vectors"].items()
            }

            return True

        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "available": True,
            "initialized": True,
            "vectors": len(self._vectors),
            "dimension": self.dimension,
            "fallback": True,
        }
