from .document import (
    finalize_v2_document,
    process_parts_v2_document,
)
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
    PartParagraphParquetHandler,
    CourtReasoningPartHandler,
    IntroductoryPartHandler,
    PlaceholderPromptHandler,
    ResultPartHandler,
    V2Handler,
    build_v2_map_schema,
    build_parts_v2_chain,
    compose_v2_map_messages,
    compose_v2_reduce_messages,
    group_selected_paragraphs,
    merge_v2_values,
    normalize_sequential_parts,
)
from .settings import (
    DEFAULT_V2_CHAIN_SETTINGS,
    DEFAULT_V2_EXTRACTION_SETTINGS,
    DEFAULT_V2_PART_PROCESSING_PROMPTS,
    DEFAULT_V2_STORAGE_SETTINGS,
    SAMPLE_V2_PART_PROCESSING_PROMPTS,
    PART_PROCESSING_PROMPT_PLACEHOLDER,
    V2_DISTRIBUTED_INFO_VERSION,
    V2_INFO_VERSION,
    V2ChainSettings,
    V2PartProcessingMode,
    V2PartProcessingPrompts,
    validate_v2_extraction_settings,
)
from .sample_processing import (
    SampleProcessingContext,
    SmokeDocumentResult,
    build_schema_population_report,
    build_sample_processing_contexts,
    run_real_data_smoke,
)
from .state import V2DocumentState, V2HandlerContext
from .prompts import (
    INTRODUCTORY_PART_PROMPT,
    REASONING_PART_PROMPT,
    RESOLUTION_PART_PROMPT,
)

__all__ = [
    "PartParagraphParquetHandler",
    "CourtReasoningPartHandler",
    "DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS",
    "DEFAULT_V2_CHAIN_SETTINGS",
    "DEFAULT_V2_EXTRACTION_SETTINGS",
    "DEFAULT_V2_PART_PROCESSING_PROMPTS",
    "DEFAULT_V2_STORAGE_SETTINGS",
    "IntroductoryPartHandler",
    "DOCUMENT_TEXT_PARSING_ROOT",
    "DocumentTextParsingClients",
    "DocumentTextParsingSettings",
    "PARAGRAPH_SCHEMA",
    "PART_PROCESSING_PROMPT_PLACEHOLDER",
    "INTRODUCTORY_PART_PROMPT",
    "REASONING_PART_PROMPT",
    "RESOLUTION_PART_PROMPT",
    "PlaceholderPromptHandler",
    "ResultPartHandler",
    "V2ChainSettings",
    "V2PartProcessingMode",
    "V2PartProcessingPrompts",
    "V2DocumentState",
    "V2Handler",
    "V2HandlerContext",
    "V2_INFO_VERSION",
    "V2_DISTRIBUTED_INFO_VERSION",
    "SAMPLE_V2_PART_PROCESSING_PROMPTS",
    "SampleProcessingContext",
    "SmokeDocumentResult",
    "build_schema_population_report",
    "build_v2_map_schema",
    "build_parts_v2_chain",
    "build_sample_processing_contexts",
    "compose_v2_map_messages",
    "compose_v2_reduce_messages",
    "group_selected_paragraphs",
    "normalize_sequential_parts",
    "create_bigquery_client",
    "create_google_cloud_clients",
    "create_storage_client",
    "create_v2_document_folder",
    "finalize_v2_document",
    "list_completed_v2_document_ids",
    "merge_v2_values",
    "parse_document_to_artifacts",
    "process_parts_v2_document",
    "run_document_text_parsing_pipeline",
    "run_real_data_smoke",
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
