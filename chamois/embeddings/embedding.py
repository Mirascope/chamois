"""The `Embedding` Class."""


class Embedding:
    """A class representing an embedding of a text chunk."""

    def __init__(self, text: str, embedding: list[float], model: str) -> None:
        """Initialize an Embedding instance.

        Args:
            text: The text chunk that was embedded
            embedding: The embedding vector
            model: The model used to create the embedding
        """
        self.text = text
        self.embedding = embedding
        self.model = model

    def __repr__(self) -> str:
        """Return a string representation of the embedding."""
        return f"Embedding(text={self.text[:20]}..., model={self.model}, dimensions={len(self.embedding)})"
