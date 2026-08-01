from .bigquery import (
    ClassificationSource,
    iter_documents_for_classification,
    mark_documents_classified,
)
from .cloud import (
    classification_result_object_path,
    classification_source_object_path,
)
from .processing import (
    ALLOWED_SECTIONS,
    CLASSIFICATION_SCHEMA,
    ClassificationResult,
    classify_paragraph_parquet,
)
from .prompts import (
    CLASSIFICATION_CRITERIA_PLACEHOLDER,
    DEFAULT_CLASSIFICATION_CRITERIA,
    DOCUMENT_PART_CLASSIFICATION_PROMPT,
)
from .settings import (
    DEFAULT_DOCUMENT_CLASSIFICATION_SETTINGS,
    DocumentClassificationSettings,
)

__all__ = [
    "ALLOWED_SECTIONS",
    "CLASSIFICATION_CRITERIA_PLACEHOLDER",
    "CLASSIFICATION_SCHEMA",
    "ClassificationResult",
    "ClassificationSource",
    "DEFAULT_DOCUMENT_CLASSIFICATION_SETTINGS",
    "DEFAULT_CLASSIFICATION_CRITERIA",
    "DOCUMENT_PART_CLASSIFICATION_PROMPT",
    "DocumentClassificationSettings",
    "classification_result_object_path",
    "classification_source_object_path",
    "classify_paragraph_parquet",
    "iter_documents_for_classification",
    "mark_documents_classified",
    "run_document_classification_pipeline",
]


def run_document_classification_pipeline(*args, **kwargs):
    from .pipeline import run_document_classification_pipeline as _run

    return _run(*args, **kwargs)
