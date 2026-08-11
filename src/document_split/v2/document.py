from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ExtractionSettings
from ..processing import validate_part_payload
from .handlers import (
    V2Handler,
    build_parts_v2_chain,
)
from .settings import (
    DEFAULT_V2_CHAIN_SETTINGS,
    DEFAULT_V2_EXTRACTION_SETTINGS,
    DEFAULT_V2_PART_PROCESSING_PROMPTS,
    V2ChainSettings,
    V2PartProcessingPrompts,
    validate_v2_extraction_settings,
)
from .state import V2DocumentState, V2HandlerContext


def finalize_v2_document(
    state: V2DocumentState,
    extraction_settings: ExtractionSettings,
) -> V2DocumentState:
    normalized = validate_part_payload(
        state.extraction,
        extraction_settings.extraction_schema,
    )
    state.final_object = normalized
    state.final_row = {
        "document_id": str(state.document_id),
        "text": state.full_text,
        **{
            name: normalized[name]
            for name in extraction_settings.extraction_schema.names
        },
    }
    table = pa.Table.from_pylist(
        [state.final_row],
        schema=extraction_settings.output_schema,
    )
    pq.write_table(
        table,
        state.artifact_path("result.parquet"),
        compression=extraction_settings.parquet_compression,
        use_dictionary=True,
        write_statistics=True,
    )
    state.write_json_artifact("result.json", normalized)
    state.write_json_artifact(
        "chain_summary.json",
        {
            "document_id": state.document_id,
            "paragraph_count": len(state.paragraphs),
            "model_calls": state.model_calls,
            "warnings": state.warnings,
            "handlers": list(state.handler_outputs),
        },
    )
    return state


def process_parts_v2_document(
    *,
    document_id: str,
    justice_kind: int,
    parts_parquet_bytes: bytes,
    work_dir: Path,
    model_pipe: Any,
    tokenizer: Any,
    extraction_settings: ExtractionSettings = (
        DEFAULT_V2_EXTRACTION_SETTINGS
    ),
    chain_settings: V2ChainSettings = DEFAULT_V2_CHAIN_SETTINGS,
    part_prompts: V2PartProcessingPrompts = (
        DEFAULT_V2_PART_PROCESSING_PROMPTS
    ),
    chain: V2Handler | None = None,
    production: bool = True,
) -> V2DocumentState:
    """Route upstream paragraph parts and create one merged result row."""

    validate_v2_extraction_settings(extraction_settings)
    chain_settings.validate()
    part_prompts.validate(production=production)
    state = V2DocumentState(
        document_id=str(document_id),
        justice_kind=int(justice_kind),
        raw_rtf=b"",
        work_dir=work_dir,
        parts_parquet_bytes=parts_parquet_bytes,
    )
    context = V2HandlerContext(
        model_pipe=model_pipe,
        tokenizer=tokenizer,
        extraction_settings=extraction_settings,
        chain_settings=chain_settings,
        include_base_prompt=False,
    )
    active_chain = chain or build_parts_v2_chain(part_prompts)
    active_chain.handle(state, context)
    return finalize_v2_document(state, extraction_settings)
