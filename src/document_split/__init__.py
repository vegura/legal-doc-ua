from .config import (
    BASE_SCHEMA,
    BIGQUERY_TABLE,
    DEFAULT_EXTRACTION_PROMPT,
    DESTINATION_BUCKET,
    CRIMINAL_SCHEMA,
    CRIMINAL_PROMPT,
    INFO_VERSION,
    MODEL_ID,
    PROJECT_ID,
    SOURCE_BUCKET,
    ExtractionSettings,
    StorageSettings,
)
from .pipeline import run_pipeline
from .processing import (
    Paragraph,
    ParagraphBatch,
    build_paragraph_batches,
    parse_document_to_parquet,
    rtf_bytes_to_paragraphs,
    validate_model_payload,
)
__all__ = [
    "BASE_SCHEMA",
    "BIGQUERY_TABLE",
    "DEFAULT_EXTRACTION_PROMPT",
    "DESTINATION_BUCKET",
    "CRIMINAL_SCHEMA",
    "CRIMINAL_PROMPT",
    "INFO_VERSION",
    "MODEL_ID",
    "PROJECT_ID",
    "SOURCE_BUCKET",
    "ExtractionSettings",
    "Paragraph",
    "ParagraphBatch",
    "StorageSettings",
    "build_paragraph_batches",
    "parse_document_to_parquet",
    "rtf_bytes_to_paragraphs",
    "run_pipeline",
    "run_self_checks",
    "validate_model_payload",
]


def run_self_checks() -> None:
    from .self_checks import run_self_checks as _run_self_checks

    _run_self_checks()
