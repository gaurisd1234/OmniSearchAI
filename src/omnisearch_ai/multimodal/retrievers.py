"""Independent text-to-image and image-to-image retrieval interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnisearch_ai.multimodal.image_index import ChromaImageIndex
from omnisearch_ai.multimodal.images import ImageSearchResult


class TextToImageRetriever:
    """Retrieve indexed images from a natural-language CLIP text query."""

    def __init__(self, index: ChromaImageIndex) -> None:
        self.index = index

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]:
        return self.index.search_text(query, top_k=top_k, where=where)


class ImageToImageRetriever:
    """Retrieve visually similar indexed images from an uploaded image path."""

    def __init__(self, index: ChromaImageIndex) -> None:
        self.index = index

    def search(
        self,
        image_path: str | Path,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        exclude_self: bool = True,
    ) -> list[ImageSearchResult]:
        return self.index.search_image(
            image_path,
            top_k=top_k,
            where=where,
            exclude_self=exclude_self,
        )
