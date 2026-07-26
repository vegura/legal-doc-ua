from .processing import (
    PARAGRAPH_SCHEMA,
    PARQUET_COMPRESSION,
    parse_document_to_artifacts,
    populate_paragraph_artifacts,
)
from .runtime import (
    DocumentTextParsingClients,
    StorageAuthMode,
    create_bigquery_client,
    create_google_cloud_clients,
    create_storage_client,
)
from .settings import (
    DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS,
    DOCUMENT_TEXT_PARSING_ROOT,
    DocumentTextParsingSettings,
)

__all__ = [
    "DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS",
    "DOCUMENT_TEXT_PARSING_ROOT",
    "DocumentTextParsingClients",
    "DocumentTextParsingSettings",
    "PARAGRAPH_SCHEMA",
    "PARQUET_COMPRESSION",
    "StorageAuthMode",
    "create_bigquery_client",
    "create_google_cloud_clients",
    "create_storage_client",
    "parse_document_to_artifacts",
    "populate_paragraph_artifacts",
    "run_document_text_parsing_pipeline",
]


def run_document_text_parsing_pipeline(*args, **kwargs):
    from .pipeline import run_document_text_parsing_pipeline as _run

    return _run(*args, **kwargs)
