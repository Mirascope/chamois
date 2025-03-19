"""Tests for the OpenAI embedding utilities."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chamois.embeddings._providers.openai import embed, embed_async
from chamois.embeddings.embedding import Embedding


@patch("openai.OpenAI")
def test_embed_with_default_client(mock_openai_client: Mock) -> None:
    """Test embedding with default OpenAI client."""
    # Set up mocks
    mock_client = Mock()
    mock_openai_client.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    # Call the function
    chunks = ["First chunk", "Second chunk"]
    result = embed(
        model="text-embedding-3-small",
        chunks=chunks,
        dims=512,
        custom_client=None,
    )

    # Verify results
    assert len(result) == 2
    assert isinstance(result[0], Embedding)
    assert result[0].text == "First chunk"
    assert result[0].embedding == [0.1, 0.2, 0.3]
    assert result[0].model == "openai:text-embedding-3-small"

    assert result[1].text == "Second chunk"
    assert result[1].embedding == [0.4, 0.5, 0.6]
    assert result[1].model == "openai:text-embedding-3-small"

    # Verify client was created and used correctly
    mock_openai_client.assert_called_once()
    mock_client.embeddings.create.assert_called_once_with(
        input=chunks,
        model="text-embedding-3-small",
        dimensions=512,
    )


def test_embed_with_custom_client() -> None:
    """Test embedding with a custom OpenAI client."""
    # Create mock client
    mock_client = MagicMock()

    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.7, 0.8, 0.9])]
    mock_client.embeddings.create.return_value = mock_response

    # Call the function
    chunks = ["Custom client test"]
    result = embed(
        model="text-embedding-3-small",
        chunks=chunks,
        dims=1024,
        custom_client=mock_client,
    )

    # Verify results
    assert len(result) == 1
    assert result[0].text == "Custom client test"
    assert result[0].embedding == [0.7, 0.8, 0.9]

    # Verify custom client was used
    mock_client.embeddings.create.assert_called_once_with(
        input=chunks,
        model="text-embedding-3-small",
        dimensions=1024,
    )


def test_embed_with_large_batch() -> None:
    """Test embedding with a batch larger than MAX_BATCH_SIZE."""
    # Create mock client
    mock_client = MagicMock()

    # Create mock responses for two batches
    mock_response1 = Mock()
    mock_response1.data = [Mock(embedding=[0.1, 0.2]) for _ in range(20)]

    mock_response2 = Mock()
    mock_response2.data = [Mock(embedding=[0.3, 0.4]) for _ in range(5)]

    mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]

    # Create a batch of 25 chunks (should be split into 20 + 5)
    chunks = [f"Chunk {i}" for i in range(25)]

    # Call the function
    result = embed(
        model="text-embedding-3-small",
        chunks=chunks,
        dims=512,
        custom_client=mock_client,
    )

    # Verify results
    assert len(result) == 25

    # First 20 should have [0.1, 0.2] embeddings
    for i in range(20):
        assert result[i].text == f"Chunk {i}"
        assert result[i].embedding == [0.1, 0.2]

    # Last 5 should have [0.3, 0.4] embeddings
    for i in range(20, 25):
        assert result[i].text == f"Chunk {i}"
        assert result[i].embedding == [0.3, 0.4]

    # Verify client was called twice with correct batches
    assert mock_client.embeddings.create.call_count == 2

    first_call_args = mock_client.embeddings.create.call_args_list[0][1]
    assert len(first_call_args["input"]) == 20
    assert first_call_args["input"] == chunks[:20]

    second_call_args = mock_client.embeddings.create.call_args_list[1][1]
    assert len(second_call_args["input"]) == 5
    assert second_call_args["input"] == chunks[20:]


@patch("openai.AsyncOpenAI")
@pytest.mark.asyncio
async def test_embed_async_with_default_client(mock_async_openai_client: Mock) -> None:
    """Test async embedding with default AsyncOpenAI client."""
    # Set up mocks
    mock_client = AsyncMock()
    mock_async_openai_client.return_value = mock_client

    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    # Call the function
    chunks = ["First async chunk", "Second async chunk"]
    result = await embed_async(
        model="text-embedding-3-small",
        chunks=chunks,
        dims=512,
        custom_client=None,
    )

    # Verify results
    assert len(result) == 2
    assert isinstance(result[0], Embedding)
    assert result[0].text == "First async chunk"
    assert result[0].embedding == [0.1, 0.2, 0.3]

    # Verify client was created and used correctly
    mock_async_openai_client.assert_called_once()
    mock_client.embeddings.create.assert_called_once_with(
        input=chunks,
        model="text-embedding-3-small",
        dimensions=512,
    )


@pytest.mark.asyncio
async def test_embed_async_with_large_batch() -> None:
    """Test async embedding with a batch larger than MAX_BATCH_SIZE."""
    # Create mock client
    mock_client = AsyncMock()

    # For the first batch (20 items)
    mock_response1 = Mock()
    mock_response1.data = [Mock(embedding=[0.1, 0.2]) for _ in range(20)]

    # For the second batch (5 items)
    mock_response2 = Mock()
    mock_response2.data = [Mock(embedding=[0.3, 0.4]) for _ in range(5)]

    # Set up the side effect for consecutive calls
    mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]

    # Create a batch of 25 chunks (should be split into 2 batches: 20 + 5)
    chunks = [f"Async Chunk {i}" for i in range(25)]

    # Call the function
    result = await embed_async(
        model="text-embedding-3-small",
        chunks=chunks,
        dims=512,
        custom_client=mock_client,
    )

    # Verify results
    assert len(result) == 25

    # First 20 should have [0.1, 0.2] embeddings
    for i in range(20):
        assert result[i].text == f"Async Chunk {i}"
        assert result[i].embedding == [0.1, 0.2]

    # Last 5 should have [0.3, 0.4] embeddings
    for i in range(20, 25):
        assert result[i].text == f"Async Chunk {i}"
        assert result[i].embedding == [0.3, 0.4]

    # Should have made 2 batches
    assert mock_client.embeddings.create.call_count == 2

    # Check the batches
    batch_calls = mock_client.embeddings.create.call_args_list

    # First call should have 20 items
    assert len(batch_calls[0][1]["input"]) == 20
    # Second call should have 5 items
    assert len(batch_calls[1][1]["input"]) == 5
