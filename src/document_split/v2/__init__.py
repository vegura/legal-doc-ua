from .document import finalize_v2_document, process_v2_document
from .document_text_parsing import (
    DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS,
    DOCUMENT_TEXT_PARSING_ROOT,
    PARAGRAPH_SCHEMA,
    DocumentTextParsingClients,
    DocumentTextParsingSettings,
    create_bigquery_client,
    create_google_cloud_clients,
    create_storage_client,
    parse_document_to_artifacts,
)
from .handlers import (
    CaseAndParagraphClassificationHandler,
    CourtReasoningPartHandler,
    IntroductoryPartHandler,
    PlaceholderPromptHandler,
    ResultPartHandler,
    RtfToParagraphParquetHandler,
    V2Handler,
    build_default_v2_chain,
    merge_v2_values,
)
from .prompts import (
    CASE_CLASSIFICATION_PROMPT,
    COURT_REASONING_PART_PROMPT,
    INTRODUCTORY_PART_PROMPT,
    PLACEHOLDER_HANDLER_PROMPT,
    RESULT_PART_PROMPT,
)
from .settings import (
    DEFAULT_V2_CHAIN_SETTINGS,
    DEFAULT_V2_EXTRACTION_SETTINGS,
    DEFAULT_V2_STORAGE_SETTINGS,
    V2_INFO_VERSION,
    V2ChainSettings,
    validate_v2_extraction_settings,
)
from .state import V2DocumentState, V2HandlerContext

__all__ = [
    "CASE_CLASSIFICATION_PROMPT",
    "COURT_REASONING_PART_PROMPT",
    "CaseAndParagraphClassificationHandler",
    "CourtReasoningPartHandler",
    "DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS",
    "DEFAULT_V2_CHAIN_SETTINGS",
    "DEFAULT_V2_EXTRACTION_SETTINGS",
    "DEFAULT_V2_STORAGE_SETTINGS",
    "INTRODUCTORY_PART_PROMPT",
    "IntroductoryPartHandler",
    "DOCUMENT_TEXT_PARSING_ROOT",
    "DocumentTextParsingClients",
    "DocumentTextParsingSettings",
    "PARAGRAPH_SCHEMA",
    "PLACEHOLDER_HANDLER_PROMPT",
    "PlaceholderPromptHandler",
    "RESULT_PART_PROMPT",
    "ResultPartHandler",
    "RtfToParagraphParquetHandler",
    "V2ChainSettings",
    "V2DocumentState",
    "V2Handler",
    "V2HandlerContext",
    "V2_INFO_VERSION",
    "build_default_v2_chain",
    "create_bigquery_client",
    "create_google_cloud_clients",
    "create_storage_client",
    "create_v2_document_folder",
    "finalize_v2_document",
    "list_completed_v2_document_ids",
    "merge_v2_values",
    "parse_document_to_artifacts",
    "process_v2_document",
    "run_document_text_parsing_pipeline",
    "run_v2_pipeline",
    "v2_document_prefix",
    "v2_result_object_path",
    "validate_v2_extraction_settings",
]


def run_v2_pipeline(*args, **kwargs):
    from .pipeline import run_v2_pipeline as _run_v2_pipeline

    return _run_v2_pipeline(*args, **kwargs)


def run_document_text_parsing_pipeline(*args, **kwargs):
    from .document_text_parsing import (
        run_document_text_parsing_pipeline as _run,
    )

    return _run(*args, **kwargs)


def create_v2_document_folder(*args, **kwargs):
    from .cloud import create_v2_document_folder as _create

    return _create(*args, **kwargs)


def list_completed_v2_document_ids(*args, **kwargs):
    from .cloud import list_completed_v2_document_ids as _list

    return _list(*args, **kwargs)


def v2_document_prefix(*args, **kwargs):
    from .cloud import v2_document_prefix as _prefix

    return _prefix(*args, **kwargs)


def v2_result_object_path(*args, **kwargs):
    from .cloud import v2_result_object_path as _path

    return _path(*args, **kwargs)
