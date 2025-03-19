"""Chamois package."""

import importlib.metadata

from chamois.embeddings import Embedding, embed

__version__ = importlib.metadata.version("chamois")

__all__ = ["Embedding", "__version__", "embed"]
