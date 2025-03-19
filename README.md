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

# Define a function that splits text into chunks
@chamois.embed("openai:text-embedding-3-small", dims=128)
async def embed_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs and embed them."""
    return text.strip().split("\n\n")

# Process text with the decorated function
text = """
Once upon a time in a forest far away, there lived a family of rabbits.

They spent their days gathering food and playing games.
"""
embeddings: list[chamois.Embedding] = asyncio.run(embed_paragraphs(text))

# Access the embedding vectors
print(f"Number of embeddings: {len(embeddings)}")
# > 2
print(f"First embedding dimensions: {len(embeddings[0].embedding)}")
# > 128
print(f"First few values: {embeddings[0].embedding[:5]}")
# > [0.016108348965644836, 0.012391675263643265, ...]
```

You can of course also run things synchronously:

```python
import chamois

@chamois.embed("openai:text-embedding-3-small", dims=128)
def embed_paragraphs(text: str) -> list[str]:
    return text.strip().split("\n\n")

text = """Once upon a time..."""
embeddings: list[chamois.Embedding] = embed_paragraphs(text)

print(f"Number of embeddings: {len(embeddings)}")
print(f"First embedding dimensions: {len(embeddings[0].embedding)}")
print(f"First few values: {embeddings[0].embedding[:5]}")
```

## Supported Providers

- `openai`: OpenAI's embedding models (requires `pip install chamois[openai]`)
  - Example: `openai:text-embedding-3-small`
  - Example: `openai:text-embedding-ada-002`

## Versioning

Chamois uses [Semantic Versioning](https://semver.org/).

## License

This project is licensed under the terms of the [MIT License](https://github.com/Mirascope/chamois/blob/main/LICENSE).
