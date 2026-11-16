"""Image records, preprocessing, OCR, and shared CLIP image/text embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Image content and its provenance for multimodal indexing."""

    image_id: str
    path: Path
    content_hash: str
    width: int
    height: int
    image_format: str
    ocr_text: str = ""
    document_id: str | None = None
    document_name: str | None = None
    document_reference: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageSearchResult:
    image_id: str
    path: str
    score: float
    ocr_text: str
    document_id: str | None
    document_name: str | None
    document_reference: str | None
    page_number: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageProcessor:
    """Open/normalize images with Pillow and assign content-addressed IDs."""

    @staticmethod
    def _pillow():
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError(
                "Image processing requires Pillow; install omnisearch-ai[multimodal]"
            ) from exc
        return Image, ImageOps

    def open(self, path: str | Path):
        Image, ImageOps = self._pillow()
        try:
            with Image.open(path) as original:
                image_format = str(original.format or "unknown").lower()
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.load()
        except Exception as exc:
            raise ValueError(f"Could not read image {path}: {exc}") from exc
        return image, image_format

    def record(
        self,
        path: str | Path,
        *,
        ocr_text: str = "",
        document_id: str | None = None,
        document_name: str | None = None,
        document_reference: str | None = None,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRecord:
        path = Path(path).expanduser().resolve()
        image, image_format = self.open(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ImageRecord(
            image_id=f"img_{digest}",
            path=path,
            content_hash=digest,
            width=image.width,
            height=image.height,
            image_format=image_format,
            ocr_text=ocr_text.strip(),
            document_id=document_id,
            document_name=document_name,
            document_reference=document_reference,
            page_number=page_number,
            metadata=metadata or {},
        )


class EasyOCRTextExtractor:
    """Lazy EasyOCR wrapper; OCR can run on CPU to preserve GPU memory."""

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        *,
        gpu: bool = False,
        reader: Any | None = None,
    ) -> None:
        if reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "OCR requires EasyOCR; install omnisearch-ai[multimodal]"
                ) from exc
            reader = easyocr.Reader(list(languages), gpu=gpu)
        self.reader = reader

    def extract(self, image: Any) -> str:
        """Return recognized lines in reading order as newline-separated text."""
        lines = self.reader.readtext(image, detail=0, paragraph=True)
        return "\n".join(str(line).strip() for line in lines if str(line).strip())


class CLIPImageTextEmbedder:
    """Encode text and images into OpenAI CLIP ViT-B/32's shared vector space.

    The model is loaded only on construction unless both model and processor
    are injected. On CUDA, small image batches are used; CUDA OOM halves the
    batch and eventually retries on CPU when fallback is enabled.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        *,
        device: str = "auto",
        cuda_batch_size: int = 2,
        cpu_batch_size: int = 4,
        fallback_to_cpu: bool = True,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if cuda_batch_size <= 0 or cpu_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CLIP embeddings require PyTorch") from exc
        cuda_available = bool(torch.cuda.is_available())
        if device == "auto":
            device = "cuda" if cuda_available else "cpu"
        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested for CLIP but unavailable")
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        if (model is None) != (processor is None):
            raise ValueError("inject both model and processor, or neither")
        if model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "CLIP requires transformers; install omnisearch-ai[multimodal]"
                ) from exc
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
        try:
            model.to(device)
        except (RuntimeError, MemoryError) as exc:
            if device != "cuda" or not fallback_to_cpu or not self._is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            model.to("cpu")
            device = "cpu"
        model.eval()
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.device = device
        self.cuda_batch_size = cuda_batch_size
        self.cpu_batch_size = cpu_batch_size
        self.fallback_to_cpu = fallback_to_cpu
        self._torch = torch

    @staticmethod
    def _features(output: Any):
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output

    def _encode(self, inputs: Any, modality: str):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            if modality == "text":
                vectors = self.model.get_text_features(**inputs)
            else:
                vectors = self.model.get_image_features(**inputs)
        vectors = self._features(vectors)
        return self._torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "out of memory" in text and ("cuda" in text or "gpu" in text)

    def _run_batches(self, values: Sequence[Any], modality: str):
        if not values:
            return self._torch.empty((0, 512), dtype=self._torch.float32)
        batch_size = self.cuda_batch_size if self.device == "cuda" else self.cpu_batch_size
        outputs = []
        cursor = 0
        while cursor < len(values):
            current_size = min(batch_size, len(values) - cursor)
            batch = values[cursor : cursor + current_size]
            try:
                inputs = (
                    self.processor(text=list(batch), return_tensors="pt", padding=True, truncation=True)
                    if modality == "text"
                    else self.processor(images=list(batch), return_tensors="pt")
                )
                vectors = self._encode(inputs, modality)
                outputs.extend(vectors.cpu().numpy())
                cursor += current_size
            except (RuntimeError, MemoryError) as exc:
                if self.device != "cuda" or not self._is_cuda_oom(exc):
                    raise
                self._torch.cuda.empty_cache()
                if current_size > 1:
                    batch_size = max(1, current_size // 2)
                elif self.fallback_to_cpu:
                    self.model.to("cpu")
                    self.device = "cpu"
                    batch_size = self.cpu_batch_size
                else:
                    raise
        import numpy as np

        return np.asarray(outputs, dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]):
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all CLIP text inputs must be strings")
        return self._run_batches(texts, "text")

    def encode_images(self, images: Sequence[Any]):
        return self._run_batches(images, "image")
