"""
Embedding service module with configurable backends.

Supports OpenAI, vLLM (offline), and NoOp (disabled) backends.
Configure via EMBEDDING_BACKEND environment variable.
"""

from content.embeddings.embedding_service import (
    EmbeddingService,
    EmbeddingBackend,
    OpenAIEmbeddingBackend,
    VLLMEmbeddingBackend,
    NoOpEmbeddingBackend,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "VLLMEmbeddingBackend",
    "NoOpEmbeddingBackend",
]
