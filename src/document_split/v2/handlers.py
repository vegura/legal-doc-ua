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


V2_MAP_GROUNDING_VERSION = 3


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
        return self._invoke_model(
            state,
            context,
            messages,
            artifact_stem=f"batch-{batch_index:04d}",
            call_description=f"batch {batch_index}",
            target_ids=[
                paragraph.paragraph_id for paragraph in batch.targets
            ],
        )

    def _invoke_model(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
        messages: Sequence[Mapping[str, Any]],
        *,
        artifact_stem: str,
        call_description: str,
        target_ids: Sequence[int],
    ) -> dict[str, Any]:
        input_tokens = count_message_tokens(
            context.tokenizer,
            messages,
        )
        if (
            input_tokens + context.extraction_settings.max_new_tokens
            > context.extraction_settings.model_context_tokens
        ):
            raise ValueError(
                f"{self.handler_name} {call_description} for paragraphs "
                f"{target_ids} exceeds the model context: input_tokens="
                f"{input_tokens}, max_new_tokens="
                f"{context.extraction_settings.max_new_tokens}"
            )

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": (
                context.extraction_settings.max_new_tokens
            ),
            "do_sample": context.extraction_settings.temperature > 0,
        }
        if context.extraction_settings.temperature > 0:
            generation_kwargs["temperature"] = (
                context.extraction_settings.temperature
            )
        state.write_json_artifact(
            f"handlers/{self.handler_name}/{artifact_stem}-input.json",
            {
                "messages": list(messages),
                "generation_kwargs": generation_kwargs,
                "target_paragraph_ids": list(target_ids),
            },
        )
        response = context.model_pipe(
            text=messages,
            return_full_text=False,
            generate_kwargs=generation_kwargs,
        )
        state.model_calls += 1
        response_text = extract_generated_text(response)
        state.artifact_path(
            f"handlers/{self.handler_name}/{artifact_stem}-raw.txt"
        ).write_text(response_text, encoding="utf-8")
        try:
            payload = parse_json_response(response_text)
        except Exception as exc:
            preview = response_text[:500].replace("\n", "\\n")
            raise RuntimeError(
                f"{self.handler_name} {call_description} returned invalid "
                f"JSON: {type(exc).__name__}: {exc}. Preview: {preview!r}"
            ) from exc
        state.write_json_artifact(
            f"handlers/{self.handler_name}/{artifact_stem}.json",
            payload,
        )
        return payload


class SectionExtractionHandler(PromptWiredHandler):
    field_name: str
    selected_sections: frozenset[str]

    @property
    def output_schema(self) -> pa.Schema:
        return pa.schema([CRIMINAL_SCHEMA.field(self.field_name)])

    @property
    def map_schema(self) -> pa.Schema:
        return build_v2_map_schema(CRIMINAL_SCHEMA.field(self.field_name))

    def _call_map_model(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
        batch: ParagraphBatch,
        batch_index: int,
        *,
        processing_part_index: int,
        processing_part_count: int,
        attempt: int,
        validation_feedback: str | None,
    ) -> dict[str, Any]:
        messages = compose_v2_map_messages(
            state=state,
            batch=batch,
            base_prompt=(
                context.extraction_settings.prompt
                if context.include_base_prompt
                else ""
            ),
            handler_prompt=self.handler_prompt,
            section_field=CRIMINAL_SCHEMA.field(self.field_name),
            processing_part_index=processing_part_index,
            processing_part_count=processing_part_count,
            validation_feedback=validation_feedback,
        )
        target_ids = [
            paragraph.paragraph_id for paragraph in batch.targets
        ]
        return self._invoke_model(
            state,
            context,
            messages,
            artifact_stem=(
                f"map-batch-{batch_index:04d}-attempt-{attempt:02d}"
            ),
            call_description=(
                f"map batch {batch_index} attempt {attempt}"
            ),
            target_ids=target_ids,
        )

    def _call_reduce_model(
        self,
        state: V2DocumentState,
        context: V2HandlerContext,
        paragraphs: Sequence[Paragraph],
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        messages = compose_v2_reduce_messages(
            state=state,
            paragraphs=paragraphs,
            observations=observations,
            base_prompt=(
                context.extraction_settings.prompt
                if context.include_base_prompt
                else ""
            ),
            handler_prompt=self.handler_prompt,
            output_schema=self.output_schema,
        )
        target_ids = [paragraph.paragraph_id for paragraph in paragraphs]
        return self._invoke_model(
            state,
            context,
            messages,
            artifact_stem="reduce",
            call_description="reduce stage",
            target_ids=target_ids,
        )

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

        selected_paragraphs = tuple(
            paragraph
            for paragraph in state.paragraphs
            if paragraph.paragraph_id in selected_ids
        )
        observations: list[dict[str, Any]] = []
        batch_index = 0
        for part_index, processing_part in enumerate(
            processing_parts,
            start=1,
        ):
            for batch in self._batches(processing_part, context):
                batch_index += 1
                validation_feedback: str | None = None
                max_attempts = (
                    context.extraction_settings.json_retries + 1
                )
                for attempt in range(1, max_attempts + 1):
                    try:
                        payload = self._call_map_model(
                            state,
                            context,
                            batch,
                            batch_index,
                            processing_part_index=part_index,
                            processing_part_count=len(processing_parts),
                            attempt=attempt,
                            validation_feedback=validation_feedback,
                        )
                        normalized = normalize_v2_payload(
                            payload,
                            self.map_schema,
                        )
                        batch_observations = _validate_map_observations(
                            normalized["observations"],
                            selected_paragraphs,
                            section_path=self.field_name,
                            warnings=state.warnings,
                        )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        if (
                            isinstance(exc, RuntimeError)
                            and "returned invalid JSON" not in str(exc)
                        ) or (
                            isinstance(exc, ValueError)
                            and "exceeds the model context" in str(exc)
                        ):
                            raise
                        if attempt >= max_attempts:
                            raise
                        validation_feedback = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        state.warnings.append(
                            f"Retrying {self.handler_name} map batch "
                            f"{batch_index} after rejected attempt "
                            f"{attempt}: {validation_feedback}"
                        )
                        continue
                    break
                observations.extend(batch_observations)

        state.write_json_artifact(
            f"handlers/{self.handler_name}/map-observations.json",
            {"observations": observations},
        )
        reduced_payload = self._call_reduce_model(
            state,
            context,
            selected_paragraphs,
            observations,
        )
        normalized = normalize_v2_payload(
            reduced_payload,
            self.output_schema,
        )
        reduced = normalized[self.field_name]
        observation_ids = _collect_paragraph_indexes(observations)
        _validate_nested_paragraph_indexes(
            reduced,
            observation_ids,
            path=self.field_name,
        )

        schema_field = self.output_schema.field(self.field_name)
        validate_arrow_value(
            reduced,
            schema_field,
            self.field_name,
        )
        result = {self.field_name: reduced}
        state.extraction[self.field_name] = reduced
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


def build_v2_map_schema(section_field: pa.Field) -> pa.Schema:
    """Build the grounded observation contract used by map calls."""

    observation_type = pa.struct(
        [
            pa.field("paragraph_index", pa.int32(), nullable=False),
            pa.field("source_quote", pa.string(), nullable=False),
            pa.field(
                "extraction",
                section_field.type,
                nullable=False,
            ),
            pa.field(
                "conviction_law_articles",
                pa.list_(pa.string()),
                nullable=True,
            ),
        ]
    )
    return pa.schema(
        [
            pa.field(
                "observations",
                pa.list_(observation_type),
                nullable=False,
            )
        ]
    )


def compose_v2_map_messages(
    *,
    state: V2DocumentState,
    batch: ParagraphBatch,
    base_prompt: str,
    handler_prompt: str,
    section_field: pa.Field,
    processing_part_index: int = 1,
    processing_part_count: int = 1,
    validation_feedback: str | None = None,
) -> list[dict[str, Any]]:
    """Compose one paragraph-grounded map request."""

    context_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.context
    )
    target_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.targets
    )
    target_ids = [
        paragraph.paragraph_id for paragraph in batch.targets
    ]
    visible_ids = {
        paragraph.paragraph_id
        for paragraph in (*batch.context, *batch.targets)
    }
    part_assignments = [
        assignment
        for assignment in state.part_assignments
        if assignment["paragraph_index"] in visible_ids
    ]
    contract = json.dumps(
        build_response_contract(build_v2_map_schema(section_field)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    retry_text = (
        "\nPREVIOUS ATTEMPT REJECTED:\n"
        f"{validation_feedback}\n"
        "Regenerate the entire response and correct that validation error.\n"
        if validation_feedback
        else ""
    )
    user_text = f"""PROCESSING STAGE:
MAP

ACTIVE OUTPUT FIELD:
"{section_field.name}"

PROCESSING PART:
{processing_part_index} of {processing_part_count}

TARGET PARAGRAPH IDS:
{target_ids}

CONTEXT PARAGRAPHS (read-only; do not extract observations from these):
{context_text or "(none)"}

TARGET PARAGRAPHS:
{target_text}

UPSTREAM PARAGRAPH PARTS (routing metadata; do not return them):
{json.dumps(part_assignments, ensure_ascii=False, separators=(",", ":"))}
{retry_text}

Return paragraph-grounded candidate observations only.

For every observation:
- paragraph_index must be copied from the target paragraph_id tag;
- source_quote must be the shortest contiguous passage that directly supports
  the extracted fields, copied character-for-character from that same target;
- do not copy the full paragraph when a smaller relevant passage is sufficient;
- never paraphrase, correct, summarize, or join non-contiguous passages inside
  one source_quote; create separate observations when separate passages are
  needed;
- extraction is a partial value for {section_field.name}, without the outer
  {section_field.name} wrapper;
- include only fields directly supported by source_quote;
- prefer one observation per target paragraph;
- every nested source_quote follows the same minimal contiguous-passage rule;
- if a nested record refers to another target paragraph, it must carry that
  paragraph's ID and its own minimal character-for-character source_quote;
- if one paragraph supports several observations, reuse the same paragraph ID;
- never increment paragraph IDs for sentences inside one paragraph;
- never copy field definitions or instructions as extracted values.

If the targets support no extraction, return an empty observations list.
Return one compact JSON object only, without Markdown or commentary.

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
                        "You are the MAP stage of a grounded legal-document "
                        "extraction pipeline.\n\n"
                        f"{base_instructions}"
                        "ACTIVE HANDLER INSTRUCTIONS:\n"
                        f"{handler_prompt}\n\n"
                        "The field descriptions above are instructions, not "
                        "document facts. Every non-null candidate must be "
                        "grounded by an exact source_quote from its declared "
                        "target paragraph. Follow the JSON contract in the "
                        "user message."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def compose_v2_reduce_messages(
    *,
    state: V2DocumentState,
    paragraphs: Sequence[Paragraph],
    observations: Sequence[Mapping[str, Any]],
    base_prompt: str,
    handler_prompt: str,
    output_schema: pa.Schema,
) -> list[dict[str, Any]]:
    """Compose one document-section reduction from validated map results."""

    source_text = "\n".join(
        paragraph_block(paragraph) for paragraph in paragraphs
    )
    target_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    target_id_set = set(target_ids)
    part_assignments = [
        assignment
        for assignment in state.part_assignments
        if assignment["paragraph_index"] in target_id_set
    ]
    compact_observations = [
        compact
        for observation in observations
        if (compact := _compact_json_value(observation)) is not None
    ]
    contract = json.dumps(
        build_response_contract(output_schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user_text = f"""PROCESSING STAGE:
REDUCE

ACTIVE OUTPUT FIELD:
"{output_schema.names[0]}"

TARGET PARAGRAPH IDS:
{target_ids}

AUTHORITATIVE SOURCE PARAGRAPHS:
{source_text}

UPSTREAM PARAGRAPH PARTS (routing metadata; do not return them):
{json.dumps(part_assignments, ensure_ascii=False, separators=(",", ":"))}

VALIDATED MAP OBSERVATIONS:
{json.dumps(compact_observations, ensure_ascii=False, separators=(",", ":"))}

Create the single canonical section result from the authoritative source and
the grounded map observations. Resolve duplicates and conflicts using the
source paragraphs, never string length or batch order.

Rules:
- map observations are candidates; source paragraphs are authoritative;
- populate only facts represented by a validated map observation; use the
  source paragraphs to verify and resolve those candidates, not to invent new
  un-mapped facts;
- never introduce a date, number, legal provision, person, disposition, or
  conclusion that is absent from the source paragraphs;
- every final paragraph_index must belong to a validated map observation;
- preserve global paragraph IDs and reuse an ID for multiple records from the
  same paragraph;
- return null for unsupported fields;
- field descriptions and headings are instructions, never output values;
- source_quote is map-stage provenance only and must not appear in the final
  result because it is not part of the final schema;
- return only the active output field required by the JSON contract.

Return one compact JSON object only, without Markdown or commentary.

FINAL JSON CONTRACT:
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
                        "You are the REDUCE stage of a grounded legal-document "
                        "extraction pipeline.\n\n"
                        f"{base_instructions}"
                        "ACTIVE HANDLER INSTRUCTIONS:\n"
                        f"{handler_prompt}\n\n"
                        "Produce one source-grounded canonical result. The "
                        "source_quote instruction applies only to map "
                        "observations; do not add source_quote to the final "
                        "schema."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


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


def _validate_map_observations(
    observations: Sequence[Mapping[str, Any]],
    targets: Sequence[Paragraph],
    *,
    section_path: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Require every map candidate to be grounded in its claimed target."""

    targets_by_id = {
        paragraph.paragraph_id: paragraph for paragraph in targets
    }
    validated: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        path = f"observations[{index}]"
        paragraph_index = observation["paragraph_index"]
        extraction = observation["extraction"]
        if not _value_has_content(extraction):
            warnings.append(
                f"Dropped {section_path} {path} for paragraph "
                f"{paragraph_index}: extraction has no populated values"
            )
            continue
        paragraph = targets_by_id.get(paragraph_index)
        if paragraph is None:
            raise ValueError(
                f"{path}.paragraph_index must reference one of "
                f"{sorted(targets_by_id)}, got {paragraph_index!r}"
            )
        _validate_grounded_source_quote(
            observation["source_quote"],
            paragraph.text,
            path=f"{path}.source_quote",
            paragraph_index=paragraph_index,
        )
        _validate_nested_map_grounding(
            extraction,
            targets_by_id,
            outer_paragraph_index=paragraph_index,
            path=f"{path}.extraction.{section_path}",
        )
        validated.append(dict(observation))
    return validated


def _normalize_grounding_text(value: str) -> str:
    return " ".join(value.split())


def _validate_grounded_source_quote(
    source_quote: str,
    paragraph_text: str,
    *,
    path: str,
    paragraph_index: int,
) -> None:
    """Validate one relevant quote and expose where a long quote diverges."""

    normalized_quote = _normalize_grounding_text(source_quote)
    normalized_source = _normalize_grounding_text(paragraph_text)
    if not normalized_quote:
        raise ValueError(f"{path} cannot be empty")
    if normalized_quote in normalized_source:
        return

    matched_prefix_length = _longest_grounded_prefix_length(
        normalized_quote,
        normalized_source,
    )
    quote_start = max(0, matched_prefix_length - 40)
    quote_end = min(len(normalized_quote), matched_prefix_length + 80)
    quote_context = normalized_quote[quote_start:quote_end]

    matched_prefix = normalized_quote[:matched_prefix_length]
    source_offset = (
        normalized_source.find(matched_prefix) if matched_prefix else -1
    )
    source_context = ""
    if source_offset >= 0:
        mismatch_offset = source_offset + matched_prefix_length
        source_start = max(0, mismatch_offset - 40)
        source_end = min(len(normalized_source), mismatch_offset + 80)
        source_context = normalized_source[source_start:source_end]

    details = (
        f"matched_prefix_chars={matched_prefix_length}, "
        f"generated_near_mismatch={quote_context!r}"
    )
    if source_context:
        details += f", paragraph_near_mismatch={source_context!r}"
    raise ValueError(
        f"{path} is not an exact quote from paragraph "
        f"{paragraph_index}: {details}"
    )


def _longest_grounded_prefix_length(quote: str, source: str) -> int:
    """Return the longest prefix of quote occurring contiguously in source."""

    lower = 0
    upper = len(quote)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if quote[:candidate] in source:
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _validate_nested_map_grounding(
    value: Any,
    targets_by_id: Mapping[int, Paragraph],
    *,
    outer_paragraph_index: int,
    path: str,
) -> None:
    """Validate nested map records against their own target paragraphs."""

    if isinstance(value, Mapping):
        record_paragraph_index = value.get("paragraph_index")
        if "paragraph_index" in value:
            child_path = f"{path}.paragraph_index"
            if (
                type(record_paragraph_index) is not int
                or record_paragraph_index not in targets_by_id
            ):
                raise ValueError(
                    f"{child_path} must reference a target paragraph from "
                    f"{sorted(targets_by_id)}, got "
                    f"{record_paragraph_index!r}"
                )
            source_quote = value.get("source_quote")
            if record_paragraph_index != outer_paragraph_index and (
                not isinstance(source_quote, str) or not source_quote.strip()
            ):
                raise ValueError(
                    f"{path}.source_quote is required when nested "
                    f"paragraph_index={record_paragraph_index} differs "
                    f"from outer paragraph {outer_paragraph_index}"
                )
            if isinstance(source_quote, str) and source_quote.strip():
                _validate_grounded_source_quote(
                    source_quote,
                    targets_by_id[record_paragraph_index].text,
                    path=f"{path}.source_quote",
                    paragraph_index=record_paragraph_index,
                )
        for key, child in value.items():
            _validate_nested_map_grounding(
                child,
                targets_by_id,
                outer_paragraph_index=outer_paragraph_index,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_map_grounding(
                child,
                targets_by_id,
                outer_paragraph_index=outer_paragraph_index,
                path=f"{path}[{index}]",
            )


def _collect_paragraph_indexes(value: Any) -> set[int]:
    indexes: set[int] = set()
    if isinstance(value, Mapping):
        paragraph_index = value.get("paragraph_index")
        if type(paragraph_index) is int:
            indexes.add(paragraph_index)
        for child in value.values():
            indexes.update(_collect_paragraph_indexes(child))
    elif isinstance(value, list):
        for child in value:
            indexes.update(_collect_paragraph_indexes(child))
    return indexes


def _value_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_value_has_content(child) for child in value.values())
    if isinstance(value, list):
        return any(_value_has_content(child) for child in value)
    return True


def _compact_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        compact = {
            key: compact_child
            for key, child in value.items()
            if (compact_child := _compact_json_value(child)) is not None
        }
        return compact or None
    if isinstance(value, list):
        compact = [
            compact_child
            for child in value
            if (compact_child := _compact_json_value(child)) is not None
        ]
        return compact or None
    if isinstance(value, str) and not value.strip():
        return None
    return value


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


def _drop_out_of_batch_paragraph_records(
    value: Any,
    target_ids: set[int],
    *,
    path: str,
    warnings: list[str],
) -> Any:
    """Discard model records assigned to paragraphs outside this batch."""

    if isinstance(value, Mapping):
        return {
            key: _drop_out_of_batch_paragraph_records(
                child,
                target_ids,
                path=f"{path}.{key}",
                warnings=warnings,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        cleaned: list[Any] = []
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(child, Mapping) and "paragraph_index" in child:
                paragraph_index = child["paragraph_index"]
                if (
                    type(paragraph_index) is not int
                    or paragraph_index not in target_ids
                ):
                    warnings.append(
                        f"Dropped {child_path} with out-of-batch "
                        f"paragraph_index={paragraph_index!r}; expected one "
                        f"of {sorted(target_ids)}"
                    )
                    continue
            cleaned.append(
                _drop_out_of_batch_paragraph_records(
                    child,
                    target_ids,
                    path=child_path,
                    warnings=warnings,
                )
            )
        return cleaned
    return value
