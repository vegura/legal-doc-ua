from .cloud import (
    create_v2_document_folder,
    list_completed_v2_document_ids,
    v2_document_prefix,
    v2_result_object_path,
)
from .document import finalize_v2_document, process_v2_document
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
    "DEFAULT_V2_CHAIN_SETTINGS",
    "DEFAULT_V2_EXTRACTION_SETTINGS",
    "DEFAULT_V2_STORAGE_SETTINGS",
    "INTRODUCTORY_PART_PROMPT",
    "IntroductoryPartHandler",
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
    "create_v2_document_folder",
    "finalize_v2_document",
    "list_completed_v2_document_ids",
    "merge_v2_values",
    "process_v2_document",
    "run_v2_pipeline",
    "v2_document_prefix",
    "v2_result_object_path",
    "validate_v2_extraction_settings",
]


def run_v2_pipeline(*args, **kwargs):
    from .pipeline import run_v2_pipeline as _run_v2_pipeline

    return _run_v2_pipeline(*args, **kwargs)
