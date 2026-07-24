from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from striprtf.striprtf import rtf_to_text

from .config import ExtractionSettings


_CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    paragraph_order: int
    text: str


@dataclass(frozen=True)
class ParagraphBatch:
    context: tuple[Paragraph, ...]
    targets: tuple[Paragraph, ...]


def normalize_plain_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _CONTROL_CHARACTERS.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_plain_text_into_paragraphs(text: str) -> list[str]:
    # striprtf renders each RTF ``\par`` as a newline. Treat every non-empty
    # rendered line as a paragraph so the prompt's paragraph identifiers match
    # the source document instead of collapsing many RTF paragraphs together.
    return [line.strip() for line in text.split("\n") if line.strip()]


def rtf_bytes_to_paragraphs(rtf_bytes: bytes) -> list[Paragraph]:
    # latin-1 preserves every byte one-to-one. striprtf then honors an explicit
    # RTF code page when the document declares one.
    rtf_source = rtf_bytes.decode("latin-1")
    plain_text = rtf_to_text(rtf_source, errors="ignore")
    normalized = normalize_plain_text(plain_text)
    paragraph_texts = split_plain_text_into_paragraphs(normalized)
    return [
        Paragraph(
            paragraph_id=index,
            paragraph_order=index,
            text=value,
        )
        for index, value in enumerate(paragraph_texts, start=1)
    ]


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def paragraph_block(paragraph: Paragraph) -> str:
    return f"[paragraph_id={paragraph.paragraph_id}] {paragraph.text}"


def select_overlap_context(
    paragraphs: Sequence[Paragraph],
    target_start: int,
    tokenizer,
    overlap_tokens: int,
) -> tuple[Paragraph, ...]:
    if target_start <= 0 or overlap_tokens <= 0:
        return ()

    selected: list[Paragraph] = []
    token_count = 0
    for paragraph in reversed(paragraphs[:target_start]):
        selected.append(paragraph)
        token_count += count_tokens(tokenizer, paragraph_block(paragraph))
        if token_count >= overlap_tokens:
            break
    selected.reverse()
    return tuple(selected)


def build_paragraph_batches(
    paragraphs: Sequence[Paragraph],
    tokenizer,
    target_chunk_tokens: int,
    overlap_tokens: int,
) -> list[ParagraphBatch]:
    if target_chunk_tokens <= 0:
        raise ValueError("target_chunk_tokens must be positive")

    batches: list[ParagraphBatch] = []
    target_start = 0
    while target_start < len(paragraphs):
        context = select_overlap_context(
            paragraphs,
            target_start,
            tokenizer,
            overlap_tokens,
        )
        used_tokens = sum(
            count_tokens(tokenizer, paragraph_block(value))
            for value in context
        )
        targets: list[Paragraph] = []
        cursor = target_start

        while cursor < len(paragraphs):
            paragraph = paragraphs[cursor]
            paragraph_tokens = count_tokens(
                tokenizer, paragraph_block(paragraph)
            )
            if (
                targets
                and used_tokens + paragraph_tokens > target_chunk_tokens
            ):
                break
            targets.append(paragraph)
            used_tokens += paragraph_tokens
            cursor += 1

            # Preserve a single oversized paragraph. Its complete chat prompt
            # is checked against the model context before inference.
            if len(targets) == 1 and used_tokens > target_chunk_tokens:
                break

        if not targets:
            raise RuntimeError("Paragraph batching made no progress")
        batches.append(
            ParagraphBatch(context=context, targets=tuple(targets))
        )
        target_start += len(targets)

    flattened_ids = [
        paragraph.paragraph_id
        for batch in batches
        for paragraph in batch.targets
    ]
    expected_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    if flattened_ids != expected_ids:
        raise AssertionError(
            "Paragraph batching lost or duplicated target paragraphs"
        )
    return batches


def arrow_type_contract(data_type: pa.DataType) -> dict[str, Any]:
    if pa.types.is_integer(data_type):
        return {"type": ["integer", "null"]}
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return {"type": ["string", "null"]}
    if pa.types.is_boolean(data_type):
        return {"type": ["boolean", "null"]}
    if pa.types.is_floating(data_type):
        return {"type": ["number", "null"]}
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return {
            "type": ["array", "null"],
            "items": arrow_field_contract(data_type.value_field),
        }
    if pa.types.is_struct(data_type):
        properties = {
            child.name: arrow_field_contract(child)
            for child in data_type
        }
        return {
            "type": ["object", "null"],
            "properties": properties,
            "required": [
                child.name for child in data_type if not child.nullable
            ],
            "additionalProperties": False,
        }
    raise TypeError(
        f"Unsupported prompt-dependent Arrow type: {data_type}"
    )


def arrow_field_contract(field: pa.Field) -> dict[str, Any]:
    contract = arrow_type_contract(field.type)
    if not field.nullable:
        allowed_types = contract.get("type")
        if isinstance(allowed_types, list):
            contract["type"] = [
                value for value in allowed_types if value != "null"
            ]
    return contract


def build_paragraph_response_contract(
    extraction_schema: pa.Schema,
) -> dict[str, Any]:
    paragraph_properties: dict[str, Any] = {
        "paragraph_id": {"type": "integer"},
        "section_id": {
            "type": "integer",
            "minimum": 0,
            "maximum": 32767,
        },
    }
    for schema_field in extraction_schema:
        paragraph_properties[schema_field.name] = arrow_type_contract(
            schema_field.type
        )
    return {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": paragraph_properties,
                    "required": list(paragraph_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    }


def build_response_contract(
    extraction_schema: pa.Schema,
) -> dict[str, Any]:
    properties = {
        field.name: arrow_field_contract(field)
        for field in extraction_schema
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [
            field.name
            for field in extraction_schema
            if not field.nullable
        ],
        "additionalProperties": False,
    }


def build_messages(
    research_prompt: str,
    extraction_schema: pa.Schema,
    batch: ParagraphBatch,
    known_section_ids: Mapping[int, int],
) -> list[dict[str, Any]]:
    target_text = "\n\n".join(
        paragraph_block(value) for value in batch.targets
    )
    target_ids = [value.paragraph_id for value in batch.targets]
    contract = json.dumps(
        build_response_contract(extraction_schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user_text = f"""The following text is the complete court document.

DOCUMENT PARAGRAPHS:
{target_text}

In paragraph_classification, return exactly one item for every paragraph_index
in {target_ids}, in this order.

Return one document-level JSON object matching this contract. Include every
configured field, using null when the prompt does not support a value. Do not
add keys, commentary, or Markdown fences. Return compact JSON without
indentation or unnecessary whitespace.
JSON CONTRACT:
{contract}
"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": research_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def count_message_tokens(
    tokenizer, messages: Sequence[Mapping[str, Any]]
) -> int:
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)
    except Exception:
        text_parts: list[str] = []
        for message in messages:
            for content in message.get("content", []):
                if (
                    isinstance(content, Mapping)
                    and content.get("type") == "text"
                ):
                    text_parts.append(str(content.get("text", "")))
        return count_tokens(tokenizer, "\n".join(text_parts))


def extract_generated_text(result: Any) -> str:
    value = result
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, Mapping) and "generated_text" in value:
        value = value["generated_text"]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in reversed(value):
            if (
                isinstance(item, Mapping)
                and item.get("role") == "assistant"
            ):
                content = item.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = [
                        str(piece.get("text", ""))
                        for piece in content
                        if isinstance(piece, Mapping)
                        and piece.get("type") == "text"
                    ]
                    if parts:
                        return "".join(parts).strip()
    raise ValueError(
        "Could not locate assistant text in the pipeline response"
    )


def parse_json_response(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if fence:
        cleaned = fence.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Never repair a response cut off by the generation limit: doing so
        # could turn an incomplete extraction into an apparently valid record.
        # For a complete object, repair common deterministic model defects such
        # as an unescaped quote or a missing comma, then apply the unchanged
        # PyArrow-backed validator to the repaired value.
        if not cleaned.rstrip().endswith("}"):
            raise
        from json_repair import repair_json

        payload = repair_json(
            cleaned,
            return_objects=True,
            skip_json_loads=True,
        )
    if not isinstance(payload, dict):
        raise ValueError("The model response must be a JSON object")
    return payload


def validate_arrow_value(
    value: Any,
    arrow_field: pa.Field,
    path: str,
) -> None:
    if value is None:
        if not arrow_field.nullable:
            raise ValueError(f"{path} cannot be null")
        return

    data_type = arrow_field.type
    if pa.types.is_integer(data_type):
        if type(value) is not int:
            raise TypeError(f"{path} must be an integer")
        bit_width = data_type.bit_width
        if pa.types.is_signed_integer(data_type):
            minimum = -(2 ** (bit_width - 1))
            maximum = 2 ** (bit_width - 1) - 1
        else:
            minimum, maximum = 0, 2**bit_width - 1
        if not minimum <= value <= maximum:
            raise ValueError(f"{path} does not fit {data_type}")
        return
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return
    if pa.types.is_boolean(data_type):
        if type(value) is not bool:
            raise TypeError(f"{path} must be a boolean")
        return
    if pa.types.is_floating(data_type):
        if type(value) not in {int, float}:
            raise TypeError(f"{path} must be numeric")
        return
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a list")
        item_field = data_type.value_field
        for index, item in enumerate(value):
            validate_arrow_value(
                item,
                item_field,
                f"{path}[{index}]",
            )
        return
    if pa.types.is_struct(data_type):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        expected = {child.name for child in data_type}
        if set(value) != expected:
            raise ValueError(
                f"{path} fields differ: expected {sorted(expected)}, "
                f"got {sorted(value)}"
            )
        for child in data_type:
            validate_arrow_value(
                value[child.name],
                child,
                f"{path}.{child.name}",
            )
        return
    raise TypeError(f"Unsupported Arrow type at {path}: {data_type}")


def normalize_arrow_value(
    value: Any,
    arrow_field: pa.Field,
    path: str,
) -> Any:
    if value is None:
        if not arrow_field.nullable:
            raise ValueError(f"{path} cannot be null")
        return None

    data_type = arrow_field.type
    if pa.types.is_struct(data_type):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        expected = {child.name for child in data_type}
        unexpected = set(value) - expected
        if unexpected:
            raise ValueError(
                f"{path} has unexpected fields: {sorted(unexpected)}"
            )
        normalized: dict[str, Any] = {}
        for child in data_type:
            child_path = f"{path}.{child.name}"
            if child.name in value:
                normalized[child.name] = normalize_arrow_value(
                    value[child.name],
                    child,
                    child_path,
                )
            elif child.nullable:
                normalized[child.name] = None
            else:
                raise ValueError(f"{child_path} is required")
        return normalized
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a list")
        return [
            normalize_arrow_value(
                item,
                data_type.value_field,
                f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value


def normalize_known_document_layout(
    payload: Mapping[str, Any],
    extraction_schema: pa.Schema,
) -> dict[str, Any]:
    normalized_payload = dict(payload)
    operative_value = normalized_payload.get("operative_part")
    if not isinstance(operative_value, Mapping):
        return normalized_payload

    if "operative_part" not in extraction_schema.names:
        return normalized_payload
    operative_field = extraction_schema.field("operative_part")
    if not pa.types.is_struct(operative_field.type):
        return normalized_payload
    operative_fields = {field.name for field in operative_field.type}

    conviction_field = operative_field.type.field("conviction_operative")
    if not pa.types.is_struct(conviction_field.type):
        return normalized_payload
    conviction_fields = {field.name for field in conviction_field.type}

    operative = dict(operative_value)
    conviction_value = operative.get("conviction_operative")
    if not isinstance(conviction_value, Mapping):
        return normalized_payload
    conviction = dict(conviction_value)

    misplaced = (set(conviction) - conviction_fields) & operative_fields
    for field_name in misplaced:
        nested_value = conviction.pop(field_name)
        if field_name in operative:
            if operative[field_name] != nested_value:
                raise ValueError(
                    f"Conflicting values for operative_part.{field_name}"
                )
        else:
            operative[field_name] = nested_value

    operative["conviction_operative"] = conviction
    normalized_payload["operative_part"] = operative
    return normalized_payload


def validate_model_payload(
    payload: Mapping[str, Any],
    targets: Sequence[Paragraph],
    extraction_schema: pa.Schema,
    previous_section_id: int | None,
) -> tuple[list[dict[str, Any]], int]:
    if set(payload) != {"paragraphs"}:
        raise ValueError(
            "The top-level response must contain only 'paragraphs'"
        )
    returned = payload["paragraphs"]
    if not isinstance(returned, list):
        raise TypeError("paragraphs must be a list")

    target_by_id = {value.paragraph_id: value for value in targets}
    expected_ids = [value.paragraph_id for value in targets]
    expected_fields = {
        "paragraph_id",
        "section_id",
        *extraction_schema.names,
    }
    result_by_id: dict[int, dict[str, Any]] = {}

    for index, row in enumerate(returned):
        if not isinstance(row, dict):
            raise TypeError(f"paragraphs[{index}] must be an object")
        if set(row) != expected_fields:
            raise ValueError(
                f"paragraphs[{index}] fields differ from the configured schema"
            )
        paragraph_id = row["paragraph_id"]
        if type(paragraph_id) is not int or paragraph_id not in target_by_id:
            raise ValueError(f"Unknown paragraph_id: {paragraph_id!r}")
        if paragraph_id in result_by_id:
            raise ValueError(f"Duplicate paragraph_id: {paragraph_id}")

        section_id = row["section_id"]
        validate_arrow_value(
            section_id,
            pa.field("section_id", pa.int16(), nullable=False),
            f"paragraphs[{index}].section_id",
        )
        if section_id < 0:
            raise ValueError("section_id cannot be negative")

        for schema_field in extraction_schema:
            validate_arrow_value(
                row[schema_field.name],
                schema_field,
                f"paragraphs[{index}].{schema_field.name}",
            )

        entities = row.get("entities")
        if entities is not None:
            paragraph_text = target_by_id[paragraph_id].text
            for entity_index, entity in enumerate(entities):
                start = entity["start_offset"]
                end = entity["end_offset"]
                if not 0 <= start <= end <= len(paragraph_text):
                    raise ValueError(
                        f"entities[{entity_index}] offsets are outside "
                        f"paragraph {paragraph_id}"
                    )
        result_by_id[paragraph_id] = dict(row)

    if set(result_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result_by_id))
        raise ValueError(
            f"Model response is missing target paragraphs: {missing}"
        )

    ordered = [
        result_by_id[paragraph_id] for paragraph_id in expected_ids
    ]
    last_section_id = previous_section_id
    for index, row in enumerate(ordered):
        section_id = row["section_id"]
        if last_section_id is None:
            if index == 0 and section_id != 0:
                raise ValueError("The first document section_id must be 0")
        elif section_id not in {last_section_id, last_section_id + 1}:
            raise ValueError(
                "section_id must stay the same or increment by exactly one"
            )
        last_section_id = section_id
    if last_section_id is None:
        raise ValueError("A target batch cannot be empty")
    return ordered, last_section_id


def validate_document_payload(
    payload: Mapping[str, Any],
    paragraphs: Sequence[Paragraph],
    extraction_schema: pa.Schema,
) -> tuple[dict[str, Any], dict[int, int]]:
    payload = normalize_known_document_layout(payload, extraction_schema)
    expected_fields = set(extraction_schema.names)
    unexpected = set(payload) - expected_fields
    if unexpected:
        raise ValueError(
            f"Document has unexpected fields: {sorted(unexpected)}"
        )
    normalized: dict[str, Any] = {}
    for schema_field in extraction_schema:
        if schema_field.name in payload:
            normalized[schema_field.name] = normalize_arrow_value(
                payload[schema_field.name],
                schema_field,
                schema_field.name,
            )
        elif schema_field.nullable:
            normalized[schema_field.name] = None
        else:
            raise ValueError(f"{schema_field.name} is required")
        validate_arrow_value(
            normalized[schema_field.name],
            schema_field,
            schema_field.name,
        )

    if "paragraph_classification" not in normalized:
        raise ValueError(
            "Document extraction schema must contain "
            "paragraph_classification"
        )
    classifications = normalized["paragraph_classification"]
    if not isinstance(classifications, list):
        raise TypeError("paragraph_classification must be a list")

    expected_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    returned_ids: list[int] = []
    section_labels: list[str] = []
    allowed_sections = {
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
        paragraph_index = classification["paragraph_index"]
        section = classification["section"]
        returned_ids.append(paragraph_index)
        if section not in allowed_sections:
            raise ValueError(
                f"Unsupported section at paragraph {paragraph_index}: "
                f"{section!r}"
            )
        section_labels.append(section)
    if returned_ids != expected_ids:
        raise ValueError(
            "paragraph_classification must contain every document paragraph "
            f"once and in order: expected {expected_ids}, got {returned_ids}"
        )

    section_ids: dict[int, int] = {}
    current_section_id = -1
    previous_label: str | None = None
    for paragraph_id, label in zip(returned_ids, section_labels):
        if label != previous_label:
            current_section_id += 1
            previous_label = label
        section_ids[paragraph_id] = current_section_id
    return normalized, section_ids


def generate_validated_batch(
    model_pipe,
    tokenizer,
    messages: list[dict[str, Any]],
    targets: Sequence[Paragraph],
    extraction_schema: pa.Schema,
    previous_section_id: int | None,
    settings: ExtractionSettings,
) -> tuple[list[dict[str, Any]], int]:
    input_tokens = count_message_tokens(tokenizer, messages)
    if input_tokens + settings.max_new_tokens > settings.model_context_tokens:
        paragraph_ids = [value.paragraph_id for value in targets]
        raise ValueError(
            f"Prompt for target paragraphs {paragraph_ids} exceeds the model "
            "context; the document is not truncated"
        )

    attempt_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(settings.json_retries + 1):
        result = model_pipe(
            text=attempt_messages,
            return_full_text=False,
        )
        response_text = extract_generated_text(result)
        try:
            payload = parse_json_response(response_text)
            return validate_model_payload(
                payload,
                targets,
                extraction_schema,
                previous_section_id,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= settings.json_retries:
                break
            attempt_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response_text}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"The response was invalid: {exc}. Return a "
                                "corrected JSON object only, following the "
                                "original contract exactly."
                            ),
                        }
                    ],
                },
            ]
    raise RuntimeError(
        "Model did not return schema-valid JSON"
    ) from last_error


def generate_validated_document(
    model_pipe,
    tokenizer,
    messages: list[dict[str, Any]],
    paragraphs: Sequence[Paragraph],
    extraction_schema: pa.Schema,
    settings: ExtractionSettings,
) -> tuple[dict[str, Any], dict[int, int]]:
    input_tokens = count_message_tokens(tokenizer, messages)
    if input_tokens + settings.max_new_tokens > settings.model_context_tokens:
        raise ValueError(
            f"Complete document prompt uses {input_tokens} input tokens; "
            f"with max_new_tokens={settings.max_new_tokens} it exceeds "
            f"model_context_tokens={settings.model_context_tokens}"
        )

    attempt_messages = list(messages)
    last_error: Exception | None = None
    last_response = ""
    for attempt in range(settings.json_retries + 1):
        result = model_pipe(
            text=attempt_messages,
            return_full_text=False,
        )
        last_response = extract_generated_text(result)
        try:
            payload = parse_json_response(last_response)
            return validate_document_payload(
                payload,
                paragraphs,
                extraction_schema,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= settings.json_retries:
                break
            attempt_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": last_response}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"The response was invalid: {exc}. Return a "
                                "corrected document-level JSON object only, "
                                "following the original contract exactly."
                            ),
                        }
                    ],
                },
            ]
    response_is_complete = last_response.rstrip().endswith("}")
    preview = last_response[:500].replace("\n", "\\n")
    truncation_hint = (
        f" Response appears truncated at max_new_tokens="
        f"{settings.max_new_tokens}; increase the generation budget."
        if not response_is_complete
        else ""
    )
    raise RuntimeError(
        "Model did not return schema-valid JSON after "
        f"{settings.json_retries + 1} attempts. Last error: "
        f"{type(last_error).__name__}: {last_error}. "
        f"Response characters: {len(last_response)}. "
        f"Ended with closing brace: {response_is_complete}."
        f"{truncation_hint} "
        f"Response preview: {preview!r}"
    ) from last_error


def extract_document_rows(
    document_id: str,
    paragraphs: Sequence[Paragraph],
    model_pipe,
    tokenizer,
    settings: ExtractionSettings,
) -> list[dict[str, Any]]:
    settings.validate(production=True)
    if not paragraphs:
        raise ValueError(f"Document {document_id} contains no paragraphs")

    # Document-level extraction requires the model to see every paragraph in a
    # single request. The context-limit check fails
    # explicitly if the complete document and response budget do not fit.
    batch = ParagraphBatch(context=(), targets=tuple(paragraphs))
    messages = build_messages(
        settings.prompt,
        settings.extraction_schema,
        batch,
        known_section_ids={},
    )
    document_extraction, _section_ids = generate_validated_document(
        model_pipe,
        tokenizer,
        messages,
        paragraphs,
        settings.extraction_schema,
        settings=settings,
    )
    # Paragraphs are an internal prompt/indexing mechanism. Persist the complete
    # document and its single document-level extraction as exactly one row.
    return [
        {
            "document_id": str(document_id),
            "text": "\n\n".join(
                paragraph.text for paragraph in paragraphs
            ),
            **{
                name: document_extraction[name]
                for name in settings.extraction_schema.names
            },
        }
    ]


def rows_to_parquet_bytes(
    rows: Sequence[Mapping[str, Any]],
    output_schema: pa.Schema,
    compression: str,
) -> bytes:
    table = pa.Table.from_pylist(list(rows), schema=output_schema)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    )
    return buffer.getvalue()


def parse_document_to_parquet(
    document_id: str,
    rtf_bytes: bytes,
    model_pipe,
    tokenizer,
    settings: ExtractionSettings,
) -> tuple[bytes, int]:
    paragraphs = rtf_bytes_to_paragraphs(rtf_bytes)
    rows = extract_document_rows(
        document_id,
        paragraphs,
        model_pipe,
        tokenizer,
        settings,
    )
    return (
        rows_to_parquet_bytes(
            rows,
            settings.output_schema,
            settings.parquet_compression,
        ),
        len(paragraphs),
    )
