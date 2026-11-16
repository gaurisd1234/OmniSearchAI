"""Research code for retrieval methods that generate document identifiers."""

from omnisearch_ai.generative_retrieval.data import (
    PairSplits,
    QueryDocumentPair,
    create_pairs,
    split_pairs,
    write_pairs_jsonl,
)
from omnisearch_ai.generative_retrieval.evaluation import (
    MethodEvaluation,
    compare_retrievers,
    write_evaluation_report,
)
from omnisearch_ai.generative_retrieval.model import T5Components, load_t5_components
from omnisearch_ai.generative_retrieval.retriever import (
    CandidateGenerationStats,
    GenerativeRetriever,
)
from omnisearch_ai.generative_retrieval.trainer import (
    EpochMetrics,
    FineTuneConfig,
    FineTuneReport,
    fine_tune_t5,
)
from omnisearch_ai.generative_retrieval.vocabulary import DocumentIdVocabulary

__all__ = [
    "CandidateGenerationStats",
    "DocumentIdVocabulary",
    "EpochMetrics",
    "FineTuneConfig",
    "FineTuneReport",
    "GenerativeRetriever",
    "MethodEvaluation",
    "PairSplits",
    "QueryDocumentPair",
    "T5Components",
    "compare_retrievers",
    "create_pairs",
    "fine_tune_t5",
    "load_t5_components",
    "split_pairs",
    "write_evaluation_report",
    "write_pairs_jsonl",
]
