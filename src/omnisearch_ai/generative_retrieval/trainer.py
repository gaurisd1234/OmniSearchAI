"""Small-batch, gradient-accumulated fine-tuning for T5 document-ID generation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from omnisearch_ai.config import load_config
from omnisearch_ai.generative_retrieval.data import PairSplits, QueryDocumentPair
from omnisearch_ai.generative_retrieval.model import (
    T5Components,
    _is_cuda_oom,
    load_t5_components,
)
from omnisearch_ai.generative_retrieval.vocabulary import DocumentIdVocabulary


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 5e-5
    max_input_tokens: int = 256
    max_output_tokens: int = 4
    seed: int = 42
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if min(
            self.epochs,
            self.batch_size,
            self.gradient_accumulation,
            self.max_input_tokens,
            self.max_output_tokens,
        ) <= 0:
            raise ValueError("epochs, batch sizes, and token limits must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float | None


@dataclass(frozen=True, slots=True)
class FineTuneReport:
    device: str
    model_name: str
    train_pairs: int
    validation_pairs: int
    effective_batch_size: int
    gradient_checkpointing: bool
    epoch_metrics: tuple[EpochMetrics, ...]
    output_dir: str


def _move_optimizer_state(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _batch_tensors(
    tokenizer: Any,
    vocabulary: DocumentIdVocabulary,
    pairs: Sequence[QueryDocumentPair],
    config: FineTuneConfig,
    device: str,
) -> dict[str, Any]:
    inputs = tokenizer(
        [f"retrieve document: {pair.query}" for pair in pairs],
        max_length=config.max_input_tokens,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    targets = tokenizer(
        [vocabulary.encode(pair.document_id) for pair in pairs],
        max_length=config.max_output_tokens,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        **{key: value.to(device) for key, value in inputs.items()},
        "labels": labels.to(device),
    }


def _evaluate_loss(
    components: T5Components,
    pairs: Sequence[QueryDocumentPair],
    config: FineTuneConfig,
) -> float | None:
    if not pairs:
        return None
    torch = __import__("torch")
    model = components.model
    batches = [
        pairs[start : start + config.batch_size]
        for start in range(0, len(pairs), config.batch_size)
    ]
    while True:
        try:
            model.eval()
            losses = []
            with torch.inference_mode():
                for batch in batches:
                    tensors = _batch_tensors(
                        components.tokenizer,
                        components.vocabulary,
                        batch,
                        config,
                        components.device,
                    )
                    losses.append(float(model(**tensors).loss.detach().cpu()))
            return sum(losses) / len(losses)
        except (RuntimeError, MemoryError) as exc:
            if components.device != "cuda" or not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            model.to("cpu")
            components.device = "cpu"


def fine_tune_t5(
    splits: PairSplits,
    vocabulary: DocumentIdVocabulary,
    output_dir: str | Path,
    *,
    components: T5Components | None = None,
    model_name: str | None = None,
    device: str = "auto",
    config: FineTuneConfig | None = None,
    config_path: str | Path | None = None,
) -> FineTuneReport:
    """Fine-tune a T5 checkpoint and save an epoch checkpoint plus final model.

    CUDA OOM while training retries the current gradient-accumulation group on
    CPU. The default model and batch settings target a 4 GB GPU, but runtime
    requirements vary with sequence lengths and other processes.
    """
    if not splits.train:
        raise ValueError("training split must contain at least one query/document pair")
    for pair in (*splits.train, *splits.validation):
        vocabulary.encode(pair.document_id)
    config = config or _config_from_app(config_path)
    if components is None:
        components = load_t5_components(
            vocabulary,
            model_name=model_name,
            device=device,
            config_path=config_path,
        )
    torch = __import__("torch")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model = components.model
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_metrics = []
    rng = random.Random(config.seed)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_pairs = list(splits.train)
        rng.shuffle(epoch_pairs)
        batches = [
            epoch_pairs[start : start + config.batch_size]
            for start in range(0, len(epoch_pairs), config.batch_size)
        ]
        group_losses = []
        for group_start in range(0, len(batches), config.gradient_accumulation):
            group = batches[group_start : group_start + config.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            while True:
                try:
                    losses = []
                    divisor = len(group)
                    for batch in group:
                        tensors = _batch_tensors(
                            components.tokenizer,
                            vocabulary,
                            batch,
                            config,
                            components.device,
                        )
                        loss = model(**tensors).loss
                        (loss / divisor).backward()
                        losses.append(float(loss.detach().cpu()))
                    optimizer.step()
                    group_losses.extend(losses)
                    break
                except (RuntimeError, MemoryError) as exc:
                    if components.device != "cuda" or not _is_cuda_oom(exc):
                        raise
                    torch.cuda.empty_cache()
                    model.to("cpu")
                    _move_optimizer_state(optimizer, "cpu")
                    components.device = "cpu"
                    optimizer.zero_grad(set_to_none=True)
                    # Replay this whole accumulation group on CPU after an OOM.
        validation_loss = _evaluate_loss(components, splits.validation, config)
        if components.device == "cpu":
            _move_optimizer_state(optimizer, "cpu")
        epoch_metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=sum(group_losses) / max(len(group_losses), 1),
                validation_loss=validation_loss,
            )
        )
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(checkpoint_dir)
        components.tokenizer.save_pretrained(checkpoint_dir)
        vocabulary.save(checkpoint_dir / "document_id_vocabulary.json")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.save_pretrained(output_dir)
    components.tokenizer.save_pretrained(output_dir)
    vocabulary.save(output_dir / "document_id_vocabulary.json")
    report = FineTuneReport(
        device=components.device,
        model_name=components.model_name,
        train_pairs=len(splits.train),
        validation_pairs=len(splits.validation),
        effective_batch_size=config.batch_size * config.gradient_accumulation,
        gradient_checkpointing=config.gradient_checkpointing,
        epoch_metrics=tuple(epoch_metrics),
        output_dir=str(output_dir),
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _config_from_app(config_path: str | Path | None) -> FineTuneConfig:
    config = load_config(config_path)
    return FineTuneConfig(
        epochs=config.generative_epochs,
        batch_size=config.generative_batch_size,
        gradient_accumulation=config.generative_gradient_accumulation,
        learning_rate=config.generative_learning_rate,
        max_input_tokens=config.generative_max_input_tokens,
        max_output_tokens=config.generative_max_output_tokens,
        seed=config.seed,
    )
