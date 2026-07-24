from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CRIMINAL_SCHEMA
from ..processing import (
    Paragraph,
    ParagraphBatch,
    build_paragraph_batches,
    build_response_contract,
    count_message_tokens,
    extract_generated_text,
    normalize_arrow_value,
    paragraph_block,
    parse_json_response,
    rtf_bytes_to_paragraphs,
    validate_arrow_value,
)
from .prompts import (
    CASE_CLASSIFICATION_PROMPT,
    COURT_REASONING_PART_PROMPT,
    INTRODUCTORY_PART_PROMPT,
    PLACEHOLDER_HANDLER_PROMPT,
    RESULT_PART_PROMPT,
)
from .state import V2DocumentState, V2HandlerContext


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


class RtfToParagraphParquetHandler(V2Handler):
    """Normalize RTF, number paragraphs, and persist the source table."""

    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        del context
        paragraphs = tuple(rtf_bytes_to_paragraphs(state.raw_rtf))
        if not paragraphs:
            raise ValueError(
                f"Document {state.document_id} contains no paragraphs"
            )
        state.paragraphs = paragraphs
        state.numbered_text = "\n".join(
            paragraph_block(paragraph) for paragraph in paragraphs
        )
        paragraph_rows = [
            {
                "document_id": str(state.document_id),
                "paragraph_index": paragraph.paragraph_id,
                "paragraph_order": paragraph.paragraph_order,
                "numbered_text": paragraph_block(paragraph),
                "text": paragraph.text,
            }
            for paragraph in paragraphs
        ]
        paragraph_schema = pa.schema(
            [
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("paragraph_index", pa.int32(), nullable=False),
                pa.field("paragraph_order", pa.int32(), nullable=False),
                pa.field("numbered_text", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(
                paragraph_rows,
                schema=paragraph_schema,
            ),
            state.artifact_path("paragraphs.parquet"),
            compression="zstd",
        )
        state.write_json_artifact(
            "numbered_document.json",
            {
                "document_id": state.document_id,
                "paragraph_count": len(paragraphs),
                "numbered_text": state.numbered_text,
            },
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
    ) -> dict[str, Any]:
        messages = compose_v2_handler_messages(
            state=state,
            batch=batch,
            base_prompt=context.extraction_settings.prompt,
            handler_prompt=self.handler_prompt,
            output_schema=output_schema,
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


class CaseAndParagraphClassificationHandler(PromptWiredHandler):
    handler_name = "case_and_paragraph_classification"
    handler_prompt = CASE_CLASSIFICATION_PROMPT

    @property
    def output_schema(self) -> pa.Schema:
        return pa.schema(
            [
                CRIMINAL_SCHEMA.field("decision_stage"),
                CRIMINAL_SCHEMA.field("paragraph_classification"),
            ]
        )

    def transform(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
    ) -> V2DocumentState:
        classifications: list[dict[str, Any]] = []
        stages: list[str] = []
        batches = self._batches(state.paragraphs, context)

        for batch_index, batch in enumerate(batches, start=1):
            payload = self._call_model(
                state,
                context,
                batch,
                self.output_schema,
                batch_index,
            )
            normalized = normalize_v2_payload(
                payload,
                self.output_schema,
            )
            batch_classifications = normalized[
                "paragraph_classification"
            ]
            _validate_batch_classifications(
                batch_classifications,
                batch.targets,
            )
            classifications.extend(batch_classifications)
            stages.append(normalized["decision_stage"])

        expected_ids = [
            paragraph.paragraph_id for paragraph in state.paragraphs
        ]
        returned_ids = [
            value["paragraph_index"] for value in classifications
        ]
        if returned_ids != expected_ids:
            raise ValueError(
                "Classification batches lost or reordered paragraphs: "
                f"expected {expected_ids}, got {returned_ids}"
            )

        result = {
            "decision_stage": _merge_decision_stages(stages),
            "paragraph_classification": classifications,
        }
        state.extraction.update(result)
        state.handler_outputs[self.handler_name] = result
        state.write_json_artifact(
            f"handlers/{self.handler_name}/result.json",
            result,
        )
        return state


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
            classification["paragraph_index"]
            for classification in state.extraction.get(
                "paragraph_classification",
                [],
            )
            if classification["section"] in self.selected_sections
        }
        selected = tuple(
            paragraph
            for paragraph in state.paragraphs
            if paragraph.paragraph_id in selected_ids
        )
        if not selected:
            result = {self.field_name: None}
            state.extraction[self.field_name] = None
            state.handler_outputs[self.handler_name] = result
            state.write_json_artifact(
                f"handlers/{self.handler_name}/result.json",
                result,
            )
            return state

        merged: Any = None
        batches = self._batches(selected, context)
        for batch_index, batch in enumerate(batches, start=1):
            payload = self._call_model(
                state,
                context,
                batch,
                self.output_schema,
                batch_index,
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
    handler_prompt = INTRODUCTORY_PART_PROMPT
    field_name = "introductory_part"
    selected_sections = frozenset({"introductory"})


class CourtReasoningPartHandler(SectionExtractionHandler):
    handler_name = "court_reasoning_part"
    handler_prompt = COURT_REASONING_PART_PROMPT
    field_name = "reasoning_part"
    selected_sections = frozenset({"descriptive", "reasoning"})


class ResultPartHandler(SectionExtractionHandler):
    handler_name = "result_part"
    handler_prompt = RESULT_PART_PROMPT
    field_name = "operative_part"
    selected_sections = frozenset({"operative"})


class PlaceholderPromptHandler(PromptWiredHandler):
    """Optional prompt handler whose output stays outside the final schema."""

    handler_name = "placeholder"

    def __init__(
        self,
        *,
        prompt: str = PLACEHOLDER_HANDLER_PROMPT,
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
                for item in state.extraction.get(
                    "paragraph_classification",
                    [],
                )
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
    classifications = state.extraction.get(
        "paragraph_classification",
        [],
    )
    user_text = f"""The document is represented as a numbered paragraph list.
All paragraph indexes are global and must remain unchanged.

TARGET PARAGRAPH IDS:
{target_ids}

CONTEXT PARAGRAPHS (read-only; do not return batch records for these):
{context_text or "(none)"}

TARGET PARAGRAPHS:
{target_text}

KNOWN PARAGRAPH CLASSIFICATION:
{json.dumps(classifications, ensure_ascii=False, separators=(",", ":"))}

Fulfill every schema field supported by TARGET PARAGRAPHS. Return one compact
JSON object only, without Markdown or commentary.

JSON CONTRACT:
{contract}
"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are executing one transformation in a "
                        "chain-of-responsibility.\n\n"
                        "The following canonical prompt defines the legal "
                        "meaning of fields, but its full-document response "
                        "example is reference material only:\n\n"
                        f"{base_prompt}\n\n"
                        "ACTIVE HANDLER INSTRUCTIONS:\n"
                        f"{handler_prompt}\n\n"
                        "The active handler instructions and the JSON "
                        "contract in the user message override any broader "
                        "output shape shown in the canonical prompt. Return "
                        "only the active handler's schema fields."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def normalize_v2_payload(
    payload: Mapping[str, Any],
    schema: pa.Schema,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Handler response must be a JSON object")
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


def build_default_v2_chain() -> V2Handler:
    root = RtfToParagraphParquetHandler()
    classification = CaseAndParagraphClassificationHandler()
    introductory = IntroductoryPartHandler()
    reasoning = CourtReasoningPartHandler()
    result = ResultPartHandler()
    root.set_next(classification)
    classification.set_next(introductory)
    introductory.set_next(reasoning)
    reasoning.set_next(result)
    return root


def _validate_batch_classifications(
    classifications: Any,
    targets: Sequence[Paragraph],
) -> None:
    if not isinstance(classifications, list):
        raise TypeError("paragraph_classification must be a list")
    expected_ids = [
        paragraph.paragraph_id for paragraph in targets
    ]
    returned_ids: list[int] = []
    allowed = {
        "introductory",
        "descriptive",
        "reasoning",
        "operative",
    }
    for index, classification in enumerate(classifications):
        if not isinstance(classification, Mapping):
            raise TypeError(
                f"paragraph_classification[{index}] must be an object"
            )
        paragraph_id = classification["paragraph_index"]
        section = classification["section"]
        if section not in allowed:
            raise ValueError(
                f"Unsupported section for paragraph {paragraph_id}: "
                f"{section!r}"
            )
        returned_ids.append(paragraph_id)
    if returned_ids != expected_ids:
        raise ValueError(
            "Classification batch must return each target once and in order: "
            f"expected {expected_ids}, got {returned_ids}"
        )


def _merge_decision_stages(stages: Sequence[str]) -> str:
    normalized = [
        stage for stage in stages if stage != "undetermined"
    ]
    if not normalized:
        return "undetermined"
    if "final" in normalized:
        return "final"
    if "intermediate" in normalized:
        return "intermediate"
    return normalized[0]


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
