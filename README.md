# Chamois

Goal: build a standardized interface for developing retrieval systems.

This project is under active development, and we expect to release additional features in the coming months as we iterate and finalize the first version of the interface.
In its current form, Chamois is a simple wrapper on embedding endpoints like OpenAI.

## Installation

```bash
pip install chamois[openai]
```

## Usage

For optimal performance, use `async` to run as many of the I/O embedding operations in parallel as possible:

```python
import asyncio

import chamois


@chamois.embed("openai:text-embedding-3-small", dims=128)
async def embed_paragraphs(text: str) -> list[str]:
    return text.strip().split("\n\n")


text = """
Once upon a time in a forest far away, there lived a family of rabbits.

They spent their days gathering food and playing games.
"""
embeddings: list[chamois.Embedding] = asyncio.run(embed_paragraphs(text))
print(embeddings)
# > [
#     Embedding(
#       text=Once upon a time in ...,
#       model=openai:text-embedding-3-small,
#       dimensions=128
#     ),
#     Embedding(
#       text=They spent their day...,
#       model=openai:text-embedding-3-small,
#       dimensions=128
#     ),
#   ]
```

You can of course also run things synchronously:

```python
import chamois


@chamois.embed("openai:text-embedding-3-small", dims=128)
def embed_paragraphs(text: str) -> list[str]:
    return text.strip().split("\n\n")


text = """
Once upon a time in a forest far away, there lived a family of rabbits.

They spent their days gathering food and playing games.
"""
embeddings: list[chamois.Embedding] = embed_paragraphs(text)
print(embeddings)
```

## Supported Providers

- `openai`: OpenAI's embedding models (requires `pip install chamois[openai]`)
  - Example: `openai:text-embedding-3-small`
  - Example: `openai:text-embedding-ada-002`

## Versioning

Chamois uses [Semantic Versioning](https://semver.org/).

## License

This project is licensed under the terms of the [MIT License](https://github.com/Mirascope/chamois/blob/main/LICENSE).
