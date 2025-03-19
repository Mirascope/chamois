"""Tests the `embed` decorator."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from chamois.embeddings import Embedding, embed


def test_embed_invalid_model_format() -> None:
    """Test that an invalid model format raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid model identifier"):
        embed("invalid_model")


def test_embed_unsupported_provider() -> None:
    """Test that an unsupported provider raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        embed("unsupported:model")


@patch("chamois.embeddings._providers.openai.embed")
def test_embed_decorator_sync(mock_openai_embed: Mock) -> None:
    """Test the embed decorator with a synchronous function."""
    # Setup mock
    mock_embedding = Embedding(
        text="test text",
        embedding=[0.1, 0.2, 0.3],
        model="openai:text-embedding-3-small",
    )
    mock_openai_embed.return_value = [mock_embedding]

    # Define embedding function
    @embed("openai:text-embedding-3-small")
    def split_text(text: str) -> list[str]:
        return [text]

    # Call the decorated function
    result = split_text("test text")

    # Verify results
    assert len(result) == 1
    assert result[0].text == "test text"
    assert result[0].embedding == [0.1, 0.2, 0.3]
    assert result[0].model == "openai:text-embedding-3-small"

    # Verify mock was called with correct arguments
    mock_openai_embed.assert_called_once()
    _, kwargs = mock_openai_embed.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["chunks"] == ["test text"]
    assert kwargs["dims"] == 512  # Default value
    assert kwargs["custom_client"] is None


@patch("chamois.embeddings._providers.openai.embed_async")
@pytest.mark.asyncio
async def test_embed_decorator_async(mock_openai_embed_async: Mock) -> None:
    """Test the embed decorator with an asynchronous function."""
    # Setup mock
    mock_embedding = Embedding(
        text="async test",
        embedding=[0.4, 0.5, 0.6],
        model="openai:text-embedding-3-small",
    )
    mock_openai_embed_async.return_value = [mock_embedding]

    # Define async embedding function
    @embed("openai:text-embedding-3-small", dims=1024)
    async def split_text_async(text: str) -> list[str]:
        await asyncio.sleep(0.01)  # Simulate async work
        return [text]

    # Call the decorated function
    result = await split_text_async("async test")

    # Verify results
    assert len(result) == 1
    assert result[0].text == "async test"
    assert result[0].embedding == [0.4, 0.5, 0.6]
    assert result[0].model == "openai:text-embedding-3-small"

    # Verify mock was called with correct arguments
    mock_openai_embed_async.assert_called_once()
    _, kwargs = mock_openai_embed_async.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["chunks"] == ["async test"]
    assert kwargs["dims"] == 1024  # Custom value
    assert kwargs["custom_client"] is None


@patch("chamois.embeddings._providers.openai.embed")
def test_embed_with_custom_client(mock_openai_embed: Mock) -> None:
    """Test the embed decorator with a custom client."""
    # Setup mock
    mock_embedding = Embedding(
        text="custom client test",
        embedding=[0.7, 0.8, 0.9],
        model="openai:text-embedding-3-small",
    )
    mock_openai_embed.return_value = [mock_embedding]

    # Mock client
    mock_client = object()

    # Define embedding function
    @embed("openai:text-embedding-3-small", custom_client=mock_client)  # pyright: ignore [reportArgumentType]
    def split_text(text: str) -> list[str]:
        return [text]

    # Call the decorated function
    result = split_text("custom client test")

    # Verify results
    assert len(result) == 1

    # Verify mock was called with custom client
    mock_openai_embed.assert_called_once()
    _, kwargs = mock_openai_embed.call_args
    assert kwargs["custom_client"] is mock_client
