import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from omnisearch_ai.multimodal import (
    CLIPImageTextEmbedder,
    ChromaImageIndex,
    EasyOCRTextExtractor,
    ImageToImageRetriever,
    ImageProcessor,
    TextToImageRetriever,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "tests" / "fixtures" / "images"


class ColorEmbedder:
    """Deterministic offline joint-space fixture for Chroma tests."""

    def encode_images(self, images):
        vectors = []
        for image in images:
            pixel = image.getpixel((48, 48))
            if pixel[0] > pixel[1] and pixel[0] > pixel[2]:
                vector = [1.0, 0.0, 0.0]
            elif pixel[2] > pixel[0] and pixel[2] > pixel[1]:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(self, texts):
        mapping = {
            "red": [1.0, 0.0, 0.0],
            "blue": [0.0, 1.0, 0.0],
            "green": [0.0, 0.0, 1.0],
        }
        return np.asarray(
            [mapping[next((key for key in mapping if key in text.lower()), "red")] for text in texts],
            dtype=np.float32,
        )


class FakeProcessor:
    def __call__(self, *, text=None, images=None, **_kwargs):
        if text is not None:
            mapping = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 1.0, 0.0]}
            values = [mapping[next((key for key in mapping if key in item.lower()), "red")] for item in text]
            return {"inputs": torch.tensor(values, dtype=torch.float32)}
        values = []
        for image in images:
            pixel = image.getpixel((48, 48))
            values.append([float(pixel[0] > pixel[2]), 0.0, float(pixel[2] > pixel[0])])
        return {"inputs": torch.tensor(values, dtype=torch.float32)}


class FakeCLIPModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_text_features(self, *, inputs):
        return inputs

    def get_image_features(self, *, inputs):
        return inputs


class ImageProcessingTests(unittest.TestCase):
    def test_fixture_images_are_valid_and_have_stable_content_ids(self):
        processor = ImageProcessor()
        red_path = IMAGE_DIR / "red_square.png"
        record_a = processor.record(red_path)
        record_b = processor.record(red_path)

        self.assertEqual(record_a.image_id, record_b.image_id)
        self.assertEqual((record_a.width, record_a.height), (96, 96))
        self.assertEqual(record_a.image_format, "png")

    def test_clip_encodes_image_and_text_in_normalized_joint_space(self):
        embedder = CLIPImageTextEmbedder(
            device="cpu",
            model=FakeCLIPModel(),
            processor=FakeProcessor(),
            cuda_batch_size=1,
            cpu_batch_size=1,
        )
        image, _ = ImageProcessor().open(IMAGE_DIR / "red_square.png")

        text_vector = embedder.encode_texts(["a red square"])[0]
        image_vector = embedder.encode_images([image])[0]

        np.testing.assert_allclose(text_vector, image_vector)
        self.assertAlmostEqual(float(np.linalg.norm(image_vector)), 1.0)

    def test_easyocr_adapter_returns_joined_text_from_mock_reader(self):
        class Reader:
            def readtext(self, image, *, detail, paragraph):
                self.args = (image.shape, detail, paragraph)
                return ["Figure 1: Results", "  Revenue increased  "]

        reader = Reader()
        text = EasyOCRTextExtractor(reader=reader).extract(
            np.zeros((10, 20, 3), dtype=np.uint8)
        )

        self.assertEqual(text, "Figure 1: Results\nRevenue increased")
        self.assertEqual(reader.args, ((10, 20, 3), 0, True))

    def test_indexing_without_ocr_uses_filename_and_is_incremental(self):
        image = IMAGE_DIR / "green_triangle.png"
        with tempfile.TemporaryDirectory() as directory:
            with ChromaImageIndex(
                path=Path(directory) / "chroma", embedder=ColorEmbedder()
            ) as index:
                first = index.index_images([image])
                second = index.index_images([image])
                result = index.search_text("green", top_k=1)[0]
                self.assertEqual(first.indexed_count, 1)
                self.assertEqual(second.skipped_count, 1)
                self.assertEqual(result.ocr_text, "")
                self.assertEqual(Path(result.path).name, image.name)

    def test_collection_rejects_incompatible_embedding_model(self):
        class AlternateEmbedder(ColorEmbedder):
            pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chroma"
            with ChromaImageIndex(path=path, embedder=ColorEmbedder()):
                pass
            with self.assertRaisesRegex(ValueError, "use a new collection"):
                ChromaImageIndex(path=path, embedder=AlternateEmbedder())

    def test_chroma_supports_text_image_and_image_image_search_with_provenance(self):
        images = [
            IMAGE_DIR / "red_square.png",
            IMAGE_DIR / "blue_circle.png",
            IMAGE_DIR / "green_triangle.png",
        ]
        associations = {
            str(images[0]): {
                "document_id": "doc-17",
                "document_name": "figures.pdf",
                "document_reference": "figures.pdf#page=2&figure=1",
                "page_number": 2,
                "metadata": {"category": "diagram"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            with ChromaImageIndex(
                path=Path(directory) / "chroma",
                embedder=ColorEmbedder(),
                ocr_extractor=EasyOCRTextExtractor(reader=type("Reader", (), {
                    "readtext": lambda self, _image, **_kwargs: ["Revenue chart"]
                })()),
            ) as index:
                report = index.index_images(
                    images,
                    associations=associations,
                    extract_ocr=True,
                )
                repeated = index.index_images(images, associations=associations)
                repeated_without_association = index.index_images([images[0]])
                text_hits = TextToImageRetriever(index).search("red chart", top_k=3)
                associated_hits = TextToImageRetriever(index).search(
                    "red", top_k=1, where={"document_id": "doc-17"}
                )
                image_hits = ImageToImageRetriever(index).search(
                    images[1], top_k=2, exclude_self=True
                )

                self.assertEqual(report.indexed_count, 3)
                self.assertEqual(report.ocr_extracted_count, 3)
                self.assertEqual(repeated.indexed_count, 0)
                self.assertEqual(repeated.skipped_count, 3)
                self.assertEqual(repeated_without_association.skipped_count, 1)
                self.assertEqual(Path(text_hits[0].path).name, "red_square.png")
                self.assertEqual(len(associated_hits), 1)
                self.assertEqual(text_hits[0].document_id, "doc-17")
                self.assertEqual(text_hits[0].document_name, "figures.pdf")
                self.assertEqual(text_hits[0].page_number, 2)
                self.assertEqual(text_hits[0].document_reference, "figures.pdf#page=2&figure=1")
                self.assertEqual(text_hits[0].ocr_text, "Revenue chart")
                self.assertEqual(len(image_hits), 2)
                self.assertNotIn(Path(images[1]).name, [Path(hit.path).name for hit in image_hits])
                self.assertEqual(index.delete_document("doc-17"), 1)
                green_id = ImageProcessor().record(images[2]).image_id
                self.assertTrue(index.delete_image(green_id))


if __name__ == "__main__":
    unittest.main()
