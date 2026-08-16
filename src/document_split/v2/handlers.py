from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import pyarrow as pa

from ..config import CRIMINAL_SCHEMA
from ..processing import (
    Paragraph,
    ParagraphBatch,
    build_paragraph_batches,
    build_response_contract,
    count_message_tokens,
    extract_generated_text,
    normalize_arrow_value,
    normalize_known_document_layout,
    paragraph_block,
    parse_json_response,
    read_part_paragraphs_parquet,
    validate_arrow_value,
)
from .state import V2DocumentState, V2HandlerContext
from .settings import (
    PART_PROCESSING_PROMPT_PLACEHOLDER,
    V2PartProcessingMode,
    V2PartProcessingPrompts,
)


class V2Handler(ABC):
    """A chain-of-responsibility transformation over V2DocumentState."""

    def __init__(self) -> None:
        self._next: V2Handler | None = None

    def set_next(self, handler: "V2Handler") -> "V2Handler":
        self._next = handler
        return handler

    def handle(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        transformed = self.transform(state, context)
        if self._next is None:
            return transformed
        return self._next.handle(transformed, context)

    @abstractmethod
    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        raise NotImplementedError


class PartParagraphParquetHandler(V2Handler):
    """Load the authoritative paragraph text and section labels."""

    handler_name = "upstream_parts"

    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        if state.parts_parquet_bytes is None:
            raise ValueError(
                f"Document {state.document_id} has no classification.parquet"
            )
        paragraphs, source_parts = read_part_paragraphs_parquet(
            state.parts_parquet_bytes,
            document_id=state.document_id,
        )
        part_assignments = (
            normalize_sequential_parts(
                paragraphs,
                source_parts,
            )
            if context.chain_settings.part_processing_mode == "sequential"
            else source_parts
        )
        state.paragraphs = paragraphs
        state.numbered_text = "\n".join(
            paragraph_block(paragraph) for paragraph in paragraphs
        )
        state.part_assignments = part_assignments
        result = {"paragraph_parts": part_assignments}
        state.handler_outputs[self.handler_name] = result
        state.artifact_path("source_parts.parquet").write_bytes(
            state.parts_parquet_bytes
        )
        state.write_json_artifact(
            "numbered_document.json",
            {
                "document_id": state.document_id,
                "paragraph_count": len(paragraphs),
                "numbered_text": state.numbered_text,
            },
        )
        state.write_json_artifact(
            f"handlers/{self.handler_name}/result.json",
            result,
        )
        return state


class PromptWiredHandler(V2Handler, ABC):
    handler_name: str
    handler_prompt: str

    def _batches(
        self,
        paragraphs: Sequence[Paragraph],
        context: V2HandlerContext,
    ) -> list[ParagraphBatch]:
        return build_paragraph_batches(
            paragraphs,
            context.tokenizer,
            target_chunk_tokens=(
                context.chain_settings.target_batch_tokens
            ),
            overlap_tokens=context.chain_settings.overlap_tokens,
        )

    def _call_model(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
        batch: ParagraphBatch,
        output_schema: pa.Schema,
        batch_index: int,
        *,
        processing_part_index: int = 1,
        processing_part_count: int = 1,
    ) -> dict[str, Any]:
        messages = compose_v2_handler_messages(
            state=state,
            batch=batch,
            base_prompt=(
                context.extraction_settings.prompt
                if context.include_base_prompt
                else ""
            ),
            handler_prompt=self.handler_prompt,
            output_schema=output_schema,
            processing_part_index=processing_part_index,
            processing_part_count=processing_part_count,
        )
        input_tokens = count_message_tokens(
            context.tokenizer,
            messages,
        )
        if (
            input_tokens + context.extraction_settings.max_new_tokens
            > context.extraction_settings.model_context_tokens
        ):
            target_ids = [
                paragraph.paragraph_id for paragraph in batch.targets
            ]
            raise ValueError(
                f"{self.handler_name} batch {batch_index} for paragraphs "
                f"{target_ids} exceeds the model context: input_tokens="
                f"{input_tokens}, max_new_tokens="
                f"{context.extraction_settings.max_new_tokens}"
            )

        response = context.model_pipe(
            text=messages,
            return_full_text=False,
        )
        state.model_calls += 1
        response_text = extract_generated_text(response)
        state.artifact_path(
            f"handlers/{self.handler_name}/batch-{batch_index:04d}-raw.txt"
        ).write_text(response_text, encoding="utf-8")
        try:
            payload = parse_json_response(response_text)
        except Exception as exc:
            preview = response_text[:500].replace("\n", "\\n")
            raise RuntimeError(
                f"{self.handler_name} batch {batch_index} returned invalid "
                f"JSON: {type(exc).__name__}: {exc}. Preview: {preview!r}"
            ) from exc
        state.write_json_artifact(
            f"handlers/{self.handler_name}/batch-{batch_index:04d}.json",
            payload,
        )
        return payload


class SectionExtractionHandler(PromptWiredHandler):
    field_name: str
    selected_sections: frozenset[str]

    @property
    def output_schema(self) -> pa.Schema:
        return pa.schema([CRIMINAL_SCHEMA.field(self.field_name)])

    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        selected_ids = {
            assignment["paragraph_index"]
            for assignment in state.part_assignments
            if assignment["section"] in self.selected_sections
        }
        section_by_id = {
            assignment["paragraph_index"]: assignment["section"]
            for assignment in state.part_assignments
        }
        processing_parts = group_selected_paragraphs(
            state.paragraphs,
            selected_ids,
            mode=context.chain_settings.part_processing_mode,
            group_key_by_id=section_by_id,
        )
        if not processing_parts:
            result = {self.field_name: None}
            state.extraction[self.field_name] = None
            state.handler_outputs[self.handler_name] = result
            state.write_json_artifact(
                f"handlers/{self.handler_name}/result.json",
                result,
            )
            return state

        merged: Any = None
        batch_index = 0
        for part_index, processing_part in enumerate(
            processing_parts,
            start=1,
        ):
            for batch in self._batches(processing_part, context):
                batch_index += 1
                payload = self._call_model(
                    state,
                    context,
                    batch,
                    self.output_schema,
                    batch_index,
                    processing_part_index=part_index,
                    processing_part_count=len(processing_parts),
                )
                normalized = normalize_v2_payload(
                    payload,
                    self.output_schema,
                )
                _validate_nested_paragraph_indexes(
                    normalized[self.field_name],
                    {
                        paragraph.paragraph_id
                        for paragraph in batch.targets
                    },
                    path=self.field_name,
                )
                merged = merge_v2_values(
                    merged,
                    normalized[self.field_name],
                    path=self.field_name,
                    warnings=state.warnings,
                )

        schema_field = self.output_schema.field(self.field_name)
        validate_arrow_value(
            merged,
            schema_field,
            self.field_name,
        )
        result = {self.field_name: merged}
        state.extraction[self.field_name] = merged
        state.handler_outputs[self.handler_name] = result
        state.write_json_artifact(
            f"handlers/{self.handler_name}/result.json",
            result,
        )
        return state


class IntroductoryPartHandler(SectionExtractionHandler):
    handler_name = "introductory_part"
    field_name = "introductory_part"
    selected_sections = frozenset({"introductory"})

    def __init__(self, *, prompt: str) -> None:
        super().__init__()
        self.handler_prompt = prompt


class CourtReasoningPartHandler(SectionExtractionHandler):
    handler_name = "court_reasoning_part"
    field_name = "reasoning_part"
    selected_sections = frozenset({"descriptive", "reasoning"})

    def __init__(self, *, prompt: str) -> None:
        super().__init__()
        self.handler_prompt = prompt


class ResultPartHandler(SectionExtractionHandler):
    handler_name = "result_part"
    field_name = "operative_part"
    selected_sections = frozenset({"operative"})

    def __init__(self, *, prompt: str) -> None:
        super().__init__()
        self.handler_prompt = prompt


class PlaceholderPromptHandler(PromptWiredHandler):
    """Optional prompt handler whose output stays outside the final schema."""

    handler_name = "placeholder"

    def __init__(
        self,
        *,
        prompt: str = PART_PROCESSING_PROMPT_PLACEHOLDER,
        output_schema: pa.Schema | None = None,
        selected_sections: frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self.handler_prompt = prompt
        self.output_schema = output_schema or pa.schema(
            [pa.field("placeholder", pa.string())]
        )
        self.selected_sections = selected_sections

    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        paragraphs = state.paragraphs
        if self.selected_sections is not None:
            selected_ids = {
                item["paragraph_index"]
                for item in state.part_assignments
                if item["section"] in self.selected_sections
            }
            paragraphs = tuple(
                paragraph
                for paragraph in paragraphs
                if paragraph.paragraph_id in selected_ids
            )
        if not paragraphs:
            state.handler_outputs[self.handler_name] = None
            return state

        batch_results: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(
            self._batches(paragraphs, context),
            start=1,
        ):
            payload = self._call_model(
                state,
                context,
                batch,
                self.output_schema,
                batch_index,
            )
            batch_results.append(
                normalize_v2_payload(payload, self.output_schema)
            )
        state.handler_outputs[self.handler_name] = batch_results
        state.write_json_artifact(
            f"handlers/{self.handler_name}/result.json",
            batch_results,
        )
        return state


def compose_v2_handler_messages(
    *,
    state: V2DocumentState,
    batch: ParagraphBatch,
    base_prompt: str,
    handler_prompt: str,
    output_schema: pa.Schema,
    processing_part_index: int = 1,
    processing_part_count: int = 1,
) -> list[dict[str, Any]]:
    context_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.context
    )
    target_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.targets
    )
    target_ids = [
        paragraph.paragraph_id for paragraph in batch.targets
    ]
    contract = json.dumps(
        build_response_contract(output_schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    visible_ids = {
        paragraph.paragraph_id
        for paragraph in (*batch.context, *batch.targets)
    }
    part_assignments = [
        assignment
        for assignment in state.part_assignments
        if assignment["paragraph_index"] in visible_ids
    ]
    user_text = f"""The document is represented as a numbered paragraph list.
All paragraph indexes are global and must remain unchanged.

PROCESSING PART:
{processing_part_index} of {processing_part_count}

TARGET PARAGRAPH IDS:
{target_ids}

CONTEXT PARAGRAPHS (read-only; do not return batch records for these):
{context_text or "(none)"}

TARGET PARAGRAPHS:
{target_text}

UPSTREAM PARAGRAPH PARTS (routing metadata; do not return them):
{json.dumps(part_assignments, ensure_ascii=False, separators=(",", ":"))}

Fulfill every schema field supported by TARGET PARAGRAPHS. Return one compact
JSON object only, without Markdown or commentary.

JSON CONTRACT:
{contract}
"""
    base_instructions = (
        "The following canonical prompt is additional reference material:\n\n"
        f"{base_prompt}\n\n"
        if base_prompt.strip()
        else ""
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are executing one transformation in a "
                        "chain-of-responsibility.\n\n"
                        f"{base_instructions}"
                        "ACTIVE HANDLER INSTRUCTIONS:\n"
                        f"{handler_prompt}\n\n"
                        "Return only the active handler's schema fields and "
                        "follow the JSON contract in the user message."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def group_selected_paragraphs(
    paragraphs: Sequence[Paragraph],
    selected_ids: set[int],
    *,
    mode: V2PartProcessingMode,
    group_key_by_id: Mapping[int, str] | None = None,
) -> tuple[tuple[Paragraph, ...], ...]:
    """Select one combined part or one effective sequential document part."""

    if mode == "filtered":
        selected = tuple(
            paragraph
            for paragraph in paragraphs
            if paragraph.paragraph_id in selected_ids
        )
        return (selected,) if selected else ()
    if mode != "sequential":
        raise ValueError(f"Unsupported part processing mode: {mode!r}")

    groups: list[tuple[Paragraph, ...]] = []
    current: list[Paragraph] = []
    current_group_key: str | None = None
    for paragraph in paragraphs:
        if paragraph.paragraph_id in selected_ids:
            group_key = (
                group_key_by_id.get(paragraph.paragraph_id)
                if group_key_by_id is not None
                else None
            )
            if current and group_key != current_group_key:
                groups.append(tuple(current))
                current = []
            current.append(paragraph)
            current_group_key = group_key
        elif current:
            groups.append(tuple(current))
            current = []
            current_group_key = None
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def normalize_sequential_parts(
    paragraphs: Sequence[Paragraph],
    part_assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce the three non-regressing parts of a criminal decision.

    ``descriptive`` and ``reasoning`` both signal the combined reasoning part.
    Once a later part begins, a later regressing label does not move the
    effective processing state backwards.
    """

    source_section_by_id = {
        item["paragraph_index"]: item["section"]
        for item in part_assignments
    }
    stage_by_section = {
        "introductory": (0, "introductory"),
        "descriptive": (1, "reasoning"),
        "reasoning": (1, "reasoning"),
        "operative": (2, "operative"),
    }
    current_stage = -1
    current_section: str | None = None
    effective: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        paragraph_id = paragraph.paragraph_id
        source_section = source_section_by_id[paragraph_id]
        try:
            candidate_stage, candidate_section = stage_by_section[
                source_section
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported section for paragraph {paragraph_id}: "
                f"{source_section!r}"
            ) from exc
        if candidate_stage > current_stage:
            current_stage = candidate_stage
            current_section = candidate_section
        effective.append(
            {
                "paragraph_index": paragraph_id,
                "section": current_section,
            }
        )
    return effective


def normalize_v2_payload(
    payload: Mapping[str, Any],
    schema: pa.Schema,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Handler response must be a JSON object")
    payload = normalize_known_document_layout(payload, schema)
    unexpected = set(payload) - set(schema.names)
    if unexpected:
        raise ValueError(
            f"Handler response has unexpected fields: {sorted(unexpected)}"
        )
    normalized: dict[str, Any] = {}
    for schema_field in schema:
        if schema_field.name in payload:
            value = normalize_arrow_value(
                payload[schema_field.name],
                schema_field,
                schema_field.name,
            )
        elif schema_field.nullable:
            value = None
        else:
            raise ValueError(f"{schema_field.name} is required")
        validate_arrow_value(value, schema_field, schema_field.name)
        normalized[schema_field.name] = value
    return normalized


def merge_v2_values(
    current: Any,
    incoming: Any,
    *,
    path: str,
    warnings: list[str],
) -> Any:
    if current is None:
        return incoming
    if incoming is None or current == incoming:
        return current
    if isinstance(current, list) and isinstance(incoming, list):
        merged = list(current)
        seen = {
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
            for value in current
        }
        for value in incoming:
            identity = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
            if identity not in seen:
                seen.add(identity)
                merged.append(value)
        return merged
    if isinstance(current, Mapping) and isinstance(incoming, Mapping):
        keys = list(dict.fromkeys([*current.keys(), *incoming.keys()]))
        return {
            key: merge_v2_values(
                current.get(key),
                incoming.get(key),
                path=f"{path}.{key}",
                warnings=warnings,
            )
            for key in keys
        }

    warnings.append(
        f"Conflicting batch values at {path}; retained the more detailed value"
    )
    if isinstance(current, str) and isinstance(incoming, str):
        return incoming if len(incoming) > len(current) else current
    return current


def build_parts_v2_chain(
    part_prompts: V2PartProcessingPrompts,
) -> V2Handler:
    """Route upstream paragraph parts to specialized extraction handlers."""

    root = PartParagraphParquetHandler()
    introductory = IntroductoryPartHandler(prompt=part_prompts.introductory)
    reasoning = CourtReasoningPartHandler(prompt=part_prompts.reasoning)
    result = ResultPartHandler(prompt=part_prompts.operative)
    root.set_next(introductory)
    introductory.set_next(reasoning)
    reasoning.set_next(result)
    return root


def _validate_nested_paragraph_indexes(
    value: Any,
    target_ids: set[int],
    *,
    path: str,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "paragraph_index":
                if type(child) is not int or child not in target_ids:
                    raise ValueError(
                        f"{child_path} must reference a target paragraph "
                        f"from {sorted(target_ids)}, got {child!r}"
                    )
            else:
                _validate_nested_paragraph_indexes(
                    child,
                    target_ids,
                    path=child_path,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_paragraph_indexes(
                child,
                target_ids,
                path=f"{path}[{index}]",
            )
