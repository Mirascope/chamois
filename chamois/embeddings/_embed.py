"""The `embed` decorator implementation."""

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import TYPE_CHECKING, TypeAlias, overload

from ._protocols import (
    AsyncEmbeddingFunction,
    EmbedDecorator,
    EmbeddingFunction,
    embed_fn_is_async,
)
from .embedding import Embedding

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

    OpenAIClient: TypeAlias = AsyncAzureOpenAI | AsyncOpenAI | AzureOpenAI | OpenAI
else:
    OpenAIClient: TypeAlias = object


def embed(
    model: str,
    dims: int = 512,
    custom_client: OpenAIClient | None = None,
) -> EmbedDecorator:
    """Decorator for embedding functions.

    Args:
        model: The model identifier to use for embeddings, in the format "provider:model_name"
        dims: The number of dimensions for the embeddings
        custom_client: Optional custom provider client

    Returns:
        A decorator function
    """
    parts = model.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model identifier: {model}. Expected format: 'provider:model_name'"
        )
    provider, model = parts[0], parts[1]

    if provider == "openai":
        from ._providers.openai import embed as openai_embed
        from ._providers.openai import embed_async as openai_embed_async

        _embed = openai_embed
        _embed_async = openai_embed_async
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    @overload
    def decorator(
        fn: AsyncEmbeddingFunction,
    ) -> Callable[[str], Coroutine[None, None, list[Embedding]]]: ...

    @overload
    def decorator(
        fn: EmbeddingFunction,
    ) -> Callable[[str], list[Embedding]]: ...

    def decorator(
        fn: EmbeddingFunction | AsyncEmbeddingFunction,
    ) -> (
        Callable[[str], list[Embedding]]
        | Callable[[str], Coroutine[None, None, list[Embedding]]]
    ):
        if embed_fn_is_async(fn):

            @wraps(fn)
            async def wrapper_async(text: str) -> list[Embedding]:
                chunks = await fn(text)
                return await _embed_async(
                    model=model,
                    chunks=chunks,
                    dims=dims,
                    custom_client=custom_client,  # pyright: ignore [reportArgumentType]
                )

            return wrapper_async
        else:

            @wraps(fn)
            def wrapper(text: str) -> list[Embedding]:
                chunks = fn(text)
                return _embed(
                    model=model,
                    chunks=chunks,
                    dims=dims,
                    custom_client=custom_client,  # pyright: ignore [reportArgumentType]
                )

            return wrapper

    return decorator
