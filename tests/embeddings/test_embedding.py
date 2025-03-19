"""Tests the `Embedding` class."""

from chamois.embeddings import Embedding


def test_embedding_init() -> None:
    """Test that an Embedding can be initialized with the expected values."""
    text = "This is a test"
    embedding = [0.1, 0.2, 0.3]
    model = "openai:text-embedding-3-small"

    embed = Embedding(text=text, embedding=embedding, model=model)

    assert embed.text == text
    assert embed.embedding == embedding
    assert embed.model == model


def test_embedding_repr() -> None:
    """Test the string representation of an Embedding."""
    text = "This is a test with a somewhat longer text that will be truncated"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    model = "openai:text-embedding-3-small"

    embed = Embedding(text=text, embedding=embedding, model=model)
    representation = repr(embed)

    # Check that the first part of the text is included (might be shortened)
    assert "text=This is a test" in representation
    assert "model=openai:text-embedding-3-small" in representation
    assert "dimensions=5" in representation
