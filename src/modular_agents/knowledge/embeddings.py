"""Embedding providers for code chunks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from modular_agents.knowledge.base import EmbeddingProvider


class GemmaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using google/embeddinggemma-300m model.

    This model generates 300-dimensional embeddings optimized for text similarity.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        device: str = "cpu",
    ):
        """Initialize the Gemma embedding provider.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache model files
            device: Device to run model on ("cpu", "cuda", "mps")
        """
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

        # Guess initial dimension based on model name
        self._dimension = self._guess_dimension(model_name)

    @staticmethod
    def _guess_dimension(model_name: str) -> int:
        """Guess embedding dimension based on model name."""
        # Common dimension mappings
        if "embeddinggemma-300m" in model_name:
            return 300
        elif "all-MiniLM-L6-v2" in model_name:
            return 384
        elif "all-mpnet-base-v2" in model_name:
            return 768
        elif "bge-small" in model_name:
            return 384
        elif "bge-base" in model_name:
            return 768
        elif "bge-large" in model_name:
            return 1024
        else:
            # Default fallback
            return 384

    async def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for GemmaEmbeddingProvider. "
                "Install with: pip install transformers torch sentencepiece"
            ) from e

        # Load in a thread to avoid blocking
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir,
            )
            model = AutoModel.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir,
            )
            model.to(self._device)
            model.eval()
            return tokenizer, model

        loop = asyncio.get_event_loop()
        self._tokenizer, self._model = await loop.run_in_executor(None, _load)

        # Detect embedding dimension from model config
        if hasattr(self._model.config, 'hidden_size'):
            self._dimension = self._model.config.hidden_size
        else:
            # Fallback: run a test embedding to determine dimension
            import torch
            with torch.no_grad():
                test_input = self._tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                test_input = {k: v.to(self._device) for k, v in test_input.items()}
                test_output = self._model(**test_input)
                test_embedding = test_output.last_hidden_state.mean(dim=1)
                self._dimension = test_embedding.shape[1]

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        import torch

        await self._ensure_loaded()

        def _embed():
            # Tokenize
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Convert to list of lists
            return embeddings.cpu().numpy().tolist()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embed)

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._model_name


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrapper that caches embeddings to disk."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_dir: Path,
    ):
        """Initialize cached embedding provider.

        Args:
            provider: Underlying embedding provider
            cache_dir: Directory to store cached embeddings
        """
        self._provider = provider
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding with caching."""
        import hashlib
        import json

        # Create cache key from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_file = self._cache_dir / f"{text_hash}.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                return cached["embedding"]

        # Generate and cache
        embedding = await self._provider.embed_text(text)
        with open(cache_file, "w") as f:
            json.dump({"text_hash": text_hash, "embedding": embedding}, f)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with caching."""
        import hashlib
        import json

        embeddings = []
        to_compute = []
        indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cache_file = self._cache_dir / f"{text_hash}.json"

            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                    embeddings.append(cached["embedding"])
            else:
                embeddings.append(None)  # Placeholder
                to_compute.append(text)
                indices.append((i, text_hash))

        # Compute missing embeddings
        if to_compute:
            computed = await self._provider.embed_batch(to_compute)

            # Fill in results and cache
            for (idx, text_hash), embedding in zip(indices, computed):
                embeddings[idx] = embedding

                cache_file = self._cache_dir / f"{text_hash}.json"
                with open(cache_file, "w") as f:
                    json.dump({"text_hash": text_hash, "embedding": embedding}, f)

        return embeddings

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._provider.dimension

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._provider.model_name
