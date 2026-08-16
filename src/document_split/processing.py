from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from striprtf.striprtf import rtf_to_text

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

    operative = dict(operative_value)
    for nested_field_name in (
        "conviction_operative",
        "acquittal_operative",
    ):
        nested_field = operative_field.type.field(nested_field_name)
        if not pa.types.is_struct(nested_field.type):
            continue
        nested_fields = {field.name for field in nested_field.type}
        nested_value = operative.get(nested_field_name)
        if not isinstance(nested_value, Mapping):
            continue
        nested = dict(nested_value)
        misplaced = (set(nested) - nested_fields) & operative_fields
        for field_name in misplaced:
            field_value = nested.pop(field_name)
            if field_name in operative:
                if operative[field_name] != field_value:
                    raise ValueError(
                        f"Conflicting values for operative_part.{field_name}"
                    )
            else:
                operative[field_name] = field_value
        operative[nested_field_name] = nested

    normalized_payload["operative_part"] = operative
    return normalized_payload


def validate_part_payload(
    payload: Mapping[str, Any],
    extraction_schema: pa.Schema,
) -> dict[str, Any]:
    """Normalize and validate the merged outputs of the three part handlers."""

    payload = normalize_known_document_layout(payload, extraction_schema)
    expected_fields = set(extraction_schema.names)
    unexpected = set(payload) - expected_fields
    if unexpected:
        raise ValueError(
            f"Part extraction has unexpected fields: {sorted(unexpected)}"
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
    return normalized

def read_part_paragraphs_parquet(
    parquet_bytes: bytes,
    *,
    document_id: str,
) -> tuple[tuple[Paragraph, ...], list[dict[str, Any]]]:
    """Load paragraph text and upstream part assignments for one document."""

    table = pq.read_table(pa.BufferReader(parquet_bytes), use_threads=False)
    required_columns = {
        "document_id",
        "paragraph_index",
        "paragraph_order",
        "text",
    }
    part_column = "part" if "part" in table.column_names else "section"
    required_columns.add(part_column)
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise ValueError(
            "classification.parquet is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows = table.to_pylist()
    if not rows:
        raise ValueError(
            f"Part file for document {document_id} has no paragraph rows"
        )
    expected_indexes = list(range(1, len(rows) + 1))
    indexes = [row["paragraph_index"] for row in rows]
    if indexes != expected_indexes:
        raise ValueError(
            "Part-file paragraph indexes must be consecutive and start at 1: "
            f"expected {expected_indexes}, got {indexes}"
        )
    if any(str(row["document_id"]) != str(document_id) for row in rows):
        raise ValueError(
            "classification.parquet contains a different document_id than "
            f"{document_id}"
        )

    allowed_sections = {
        "introductory",
        "descriptive",
        "reasoning",
        "operative",
    }
    part_assignments: list[dict[str, Any]] = []
    paragraphs: list[Paragraph] = []
    for row in rows:
        section = row[part_column]
        if section not in allowed_sections:
            raise ValueError(
                f"Unsupported upstream part at paragraph "
                f"{row['paragraph_index']}: {section!r}"
            )
        paragraphs.append(
            Paragraph(
                paragraph_id=row["paragraph_index"],
                paragraph_order=row["paragraph_order"],
                text=row["text"],
            )
        )
        part_assignments.append(
            {
                "paragraph_index": row["paragraph_index"],
                "section": section,
            }
        )
    return tuple(paragraphs), part_assignments
