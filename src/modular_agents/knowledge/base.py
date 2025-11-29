"""Base interfaces for knowledge base components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modular_agents.core.models import CodeChunk


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Vector embedding as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of vector embeddings
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        pass


class CodeParser(ABC):
    """Abstract interface for code parsing with LLM assistance."""

    @abstractmethod
    async def parse_file(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
    ) -> list[CodeChunk]:
        """Parse a file into code chunks with LLM assistance.

        Args:
            file_path: Relative path to the file
            content: File contents
            language: Programming language
            module_name: Module this file belongs to
            repo_path: Repository root path

        Returns:
            List of code chunks extracted from the file
        """
        pass

    @abstractmethod
    async def analyze_chunk(self, chunk: CodeChunk) -> CodeChunk:
        """Analyze a code chunk to extract summary and purpose.

        Args:
            chunk: Code chunk to analyze

        Returns:
            Code chunk with summary and purpose filled in
        """
        pass
