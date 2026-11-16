"""Run text-to-image and image-to-image searches over a small image folder.

First execution loads/downloads CLIP weights. OCR is opt-in because EasyOCR
downloads recognition weights on first use. Both searches use the same CLIP
joint embedding space; fixture images are intentionally simple smoke examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnisearch_ai.multimodal import (
    ChromaImageIndex,
    ImageToImageRetriever,
    TextToImageRetriever,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir", type=Path, default=ROOT / "tests" / "fixtures" / "images"
    )
    parser.add_argument("--query", default="a red square")
    parser.add_argument("--query-image", type=Path, default=None)
    parser.add_argument("--ocr", action="store_true", help="run EasyOCR while indexing")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "vector_store" / "images-demo")
    args = parser.parse_args()
    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not images:
        parser.error(f"no supported images found under {args.image_dir}")
    query_image = args.query_image or images[0]

    with ChromaImageIndex(path=args.index_dir) as index:
        report = index.index_images(images, extract_ocr=args.ocr)
        text_hits = TextToImageRetriever(index).search(args.query, top_k=3)
        image_hits = ImageToImageRetriever(index).search(
            query_image, top_k=3, exclude_self=True
        )
    print(
        json.dumps(
            {
                "index_report": {
                    "indexed": report.indexed_count,
                    "skipped": report.skipped_count,
                    "ocr_extracted": report.ocr_extracted_count,
                },
                "text_to_image": {
                    "query": args.query,
                    "results": [
                        {
                            "path": hit.path,
                            "image_id": hit.image_id,
                            "score": hit.score,
                            "ocr_text": hit.ocr_text,
                            "document_name": hit.document_name,
                            "page_number": hit.page_number,
                        }
                        for hit in text_hits
                    ],
                },
                "image_to_image": {
                    "query_image": str(query_image),
                    "results": [
                        {
                            "path": hit.path,
                            "image_id": hit.image_id,
                            "score": hit.score,
                            "ocr_text": hit.ocr_text,
                        }
                        for hit in image_hits
                    ],
                },
                "warning": "Tiny synthetic image fixture; not an effectiveness benchmark.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
