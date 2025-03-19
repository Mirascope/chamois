"""Protocols for the `embedding` module."""

import inspect
from collections.abc import Callable, Coroutine
from typing import Protocol, TypeVar, overload

from typing_extensions import TypeIs

from .embedding import Embedding

_R = TypeVar("_R")


class EmbeddingFunction(Protocol):
    """Protocol for `embed`-decorated functions."""

    def __call__(self, text: str) -> list[str]: ...


class AsyncEmbeddingFunction(Protocol):
    """Protocol for asynchronous `embed`-decorated functions."""

    async def __call__(self, text: str) -> list[str]: ...


def embed_fn_is_async(
    fn: EmbeddingFunction | AsyncEmbeddingFunction,
) -> TypeIs[AsyncEmbeddingFunction]:
    return inspect.iscoroutinefunction(fn)


class EmbedDecorator(Protocol):
    """Protocol for the `embed` decorator."""

    @overload
    def __call__(
        self,
        fn: AsyncEmbeddingFunction,
    ) -> Callable[[str], Coroutine[None, None, list[Embedding]]]: ...

    @overload
    def __call__(
        self,
        fn: EmbeddingFunction,
    ) -> Callable[[str], list[Embedding]]: ...

    def __call__(
        self,
        fn: EmbeddingFunction | AsyncEmbeddingFunction,
    ) -> (
        Callable[[str], list[Embedding]]
        | Callable[[str], Coroutine[None, None, list[Embedding]]]
    ): ...
