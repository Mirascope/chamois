"""OpenAI-specific embedding functions."""

import asyncio
from typing import TYPE_CHECKING

from ..embedding import Embedding

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
else:
    AsyncOpenAI = AsyncAzureOpenAI = OpenAI = AzureOpenAI = object

MAX_BATCH_SIZE = 20  # Adjust based on provider limits


def embed(
    *,
    model: str,
    chunks: list[str],
    dims: int,
    custom_client: OpenAI | AzureOpenAI | None,
) -> list[Embedding]:
    """Embed a list of text chunks using OpenAI and the given model.

    Args:
        model: The OpenAI model to use
        chunks: List of text chunks to embed
        dimensions: The number of dimensions for the embeddings
        custom_client: Optional custom OpenAI client

    Returns:
        List of Embedding objects
    """
    client = custom_client
    if not client:
        try:
            from openai import OpenAI

            client = OpenAI()
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenAI embeddings. "
                "Install it with: pip install chamois[openai]"
            )

    results = []
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i : i + MAX_BATCH_SIZE]

        # Get embeddings for the batch
        response = client.embeddings.create(
            input=batch,
            model=model,
            dimensions=dims,
        )

        # Create Embedding objects
        for j, data in enumerate(response.data):
            results.append(
                Embedding(
                    text=batch[j],
                    embedding=data.embedding,
                    model=f"openai:{model}",
                )
            )

    return results


async def embed_async(
    *,
    model: str,
    chunks: list[str],
    dims: int,
    custom_client: AsyncOpenAI | AsyncAzureOpenAI | None,
) -> list[Embedding]:
    """Asynchronously embed a list of text chunks using OpenAI and the given model.

    Args:
        model: The OpenAI model to use
        chunks: List of text chunks to embed
        dimensions: The number of dimensions for the embeddings
        custom_client: Optional custom async OpenAI client

    Returns:
        List of Embedding objects
    """
    client = custom_client
    if not client:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenAI embeddings. "
                "Install it with: pip install chamois[openai]"
            )

    results = []
    tasks = []

    async def process_batch(batch_chunks: list[str]) -> list[Embedding]:
        response = await client.embeddings.create(
            input=batch_chunks,
            model=model,
            dimensions=dims,
        )

        batch_results = []
        for j, data in enumerate(response.data):
            batch_results.append(
                Embedding(
                    text=batch_chunks[j],
                    embedding=data.embedding,
                    model=f"openai:{model}",
                )
            )
        return batch_results

    # Create tasks for each batch
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i : i + MAX_BATCH_SIZE]
        tasks.append(process_batch(batch))

    # Run all tasks concurrently
    batch_results = await asyncio.gather(*tasks)

    # Flatten results
    for batch in batch_results:
        results.extend(batch)

    return results
