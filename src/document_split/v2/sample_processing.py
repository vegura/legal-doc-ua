from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa

from ..config import CRIMINAL_SCHEMA
from ..processing import (
    build_paragraph_batches,
    read_part_paragraphs_parquet,
)
from .handlers import (
    compose_v2_handler_messages,
    group_selected_paragraphs,
    normalize_sequential_parts,
)
from .settings import (
    DEFAULT_V2_CHAIN_SETTINGS,
    DEFAULT_V2_EXTRACTION_SETTINGS,
    DEFAULT_V2_PART_PROCESSING_PROMPTS,
    SAMPLE_V2_PART_PROCESSING_PROMPTS,
    V2ChainSettings,
    V2PartProcessingPrompts,
)
from .state import V2DocumentState


@dataclass(frozen=True)
class SampleProcessingContext:
    """One ready-to-send LLM context for a filtered document part."""

    processor_name: str
    target_paragraph_ids: tuple[int, ...]
    messages: list[dict[str, Any]]
    output_schema: pa.Schema
    processing_part_index: int
    processing_part_count: int


@dataclass(frozen=True)
class SmokeDocumentResult:
    """Artifacts and population statistics for one inferred document."""

    document_id: str
    artifact_dir: Path
    model_calls: int
    warnings: tuple[str, ...]
    population: dict[str, dict[str, int | bool]]


def _value_is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return True


def build_schema_population_report(
    payload: Mapping[str, Any],
    schema: pa.Schema = CRIMINAL_SCHEMA,
) -> dict[str, dict[str, int | bool]]:
    """Summarize non-null population recursively for manual smoke review."""

    report: dict[str, dict[str, int | bool]] = {}

    def visit_field(
        field: pa.Field,
        values: Sequence[Any],
        path: str,
    ) -> None:
        populated = sum(_value_is_populated(value) for value in values)
        report[path] = {
            "populated": populated > 0,
            "populated_values": populated,
            "observed_values": len(values),
        }
        data_type = field.type
        if pa.types.is_struct(data_type):
            mappings = [
                value for value in values if isinstance(value, Mapping)
            ]
            for child in data_type:
                visit_field(
                    child,
                    [value.get(child.name) for value in mappings],
                    f"{path}.{child.name}",
                )
        elif pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
            items = [
                item
                for value in values
                if isinstance(value, list)
                for item in value
            ]
            value_field = data_type.value_field
            if pa.types.is_struct(value_field.type):
                mappings = [
                    item for item in items if isinstance(item, Mapping)
                ]
                for child in value_field.type:
                    visit_field(
                        child,
                        [value.get(child.name) for value in mappings],
                        f"{path}[].{child.name}",
                    )

    for schema_field in schema:
        visit_field(
            schema_field,
            [payload.get(schema_field.name)],
            schema_field.name,
        )
    return report


def run_real_data_smoke(
    *,
    classification_files: Sequence[Path],
    output_root: Path,
    model_pipe: Any,
    tokenizer: Any,
    justice_kind: int = 2,
    chain_settings: V2ChainSettings = DEFAULT_V2_CHAIN_SETTINGS,
    part_prompts: V2PartProcessingPrompts = (
        DEFAULT_V2_PART_PROCESSING_PROMPTS
    ),
) -> list[SmokeDocumentResult]:
    """Run the production prompts locally over classified real documents.

    This deliberately bypasses cloud uploads and skip-existing behavior. Each
    document gets an isolated artifact directory containing the raw handler
    responses, normalized JSON, final Parquet, and a population report.
    """

    from .document import process_parts_v2_document

    output_root.mkdir(parents=True, exist_ok=True)
    results: list[SmokeDocumentResult] = []
    for classification_file in classification_files:
        classification_file = Path(classification_file)
        document_id = classification_file.parent.name
        if not classification_file.is_file():
            raise FileNotFoundError(classification_file)
        artifact_dir = output_root / document_id
        state = process_parts_v2_document(
            document_id=document_id,
            justice_kind=justice_kind,
            parts_parquet_bytes=classification_file.read_bytes(),
            work_dir=artifact_dir,
            model_pipe=model_pipe,
            tokenizer=tokenizer,
            extraction_settings=DEFAULT_V2_EXTRACTION_SETTINGS,
            chain_settings=chain_settings,
            part_prompts=part_prompts,
            production=True,
        )
        if state.final_object is None:
            raise RuntimeError(f"Document {document_id} has no final object")
        population = build_schema_population_report(state.final_object)
        (artifact_dir / "population_report.json").write_text(
            json.dumps(
                population,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        results.append(
            SmokeDocumentResult(
                document_id=document_id,
                artifact_dir=artifact_dir,
                model_calls=state.model_calls,
                warnings=tuple(state.warnings),
                population=population,
            )
        )
    return results


def build_sample_processing_contexts(
    *,
    document_id: str,
    justice_kind: int,
    parts_parquet_bytes: bytes,
    tokenizer: Any,
    chain_settings: V2ChainSettings = DEFAULT_V2_CHAIN_SETTINGS,
    part_prompts: V2PartProcessingPrompts = (
        SAMPLE_V2_PART_PROCESSING_PROMPTS
    ),
) -> dict[str, list[SampleProcessingContext]]:
    """Demonstrate routing without performing any model inference.

    The returned messages are exactly the contexts that a processor would send
    to an LLM. Replace ``part_prompts`` with the ontology instructions you
    define. Sample prompts are deliberately rejected by the production runner.
    """

    chain_settings.validate()
    part_prompts.validate(production=False)
    paragraphs, part_assignments = read_part_paragraphs_parquet(
        parts_parquet_bytes,
        document_id=str(document_id),
    )
    if chain_settings.part_processing_mode == "sequential":
        part_assignments = normalize_sequential_parts(
            paragraphs,
            part_assignments,
        )
    state = V2DocumentState(
        document_id=str(document_id),
        justice_kind=int(justice_kind),
        raw_rtf=b"",
        work_dir=Path("."),
        parts_parquet_bytes=parts_parquet_bytes,
        paragraphs=paragraphs,
        part_assignments=part_assignments,
    )

    section_by_id = {
        item["paragraph_index"]: item["section"]
        for item in part_assignments
    }
    specifications = {
        "introductory_part": (
            frozenset({"introductory"}),
            part_prompts.introductory,
            pa.schema([CRIMINAL_SCHEMA.field("introductory_part")]),
        ),
        "reasoning_part": (
            frozenset({"descriptive", "reasoning"}),
            part_prompts.reasoning,
            pa.schema([CRIMINAL_SCHEMA.field("reasoning_part")]),
        ),
        "operative_part": (
            frozenset({"operative"}),
            part_prompts.operative,
            pa.schema([CRIMINAL_SCHEMA.field("operative_part")]),
        ),
    }

    result: dict[str, list[SampleProcessingContext]] = {}
    for processor_name, (sections, prompt, output_schema) in (
        specifications.items()
    ):
        selected_ids = {
            paragraph.paragraph_id
            for paragraph in paragraphs
            if section_by_id[paragraph.paragraph_id] in sections
        }
        processing_parts = group_selected_paragraphs(
            paragraphs,
            selected_ids,
            mode=chain_settings.part_processing_mode,
            group_key_by_id=section_by_id,
        )
        contexts: list[SampleProcessingContext] = []
        for part_index, processing_part in enumerate(
            processing_parts,
            start=1,
        ):
            batches = build_paragraph_batches(
                processing_part,
                tokenizer,
                target_chunk_tokens=chain_settings.target_batch_tokens,
                overlap_tokens=chain_settings.overlap_tokens,
            )
            for batch in batches:
                contexts.append(
                    SampleProcessingContext(
                        processor_name=processor_name,
                        target_paragraph_ids=tuple(
                            paragraph.paragraph_id
                            for paragraph in batch.targets
                        ),
                        messages=compose_v2_handler_messages(
                            state=state,
                            batch=batch,
                            base_prompt="",
                            handler_prompt=prompt,
                            output_schema=output_schema,
                            processing_part_index=part_index,
                            processing_part_count=len(processing_parts),
                        ),
                        output_schema=output_schema,
                        processing_part_index=part_index,
                        processing_part_count=len(processing_parts),
                    )
                )
        result[processor_name] = contexts
    return result
