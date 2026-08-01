from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ...processing import (
    Paragraph,
    ParagraphBatch,
    build_paragraph_batches,
    count_message_tokens,
    extract_generated_text,
    paragraph_block,
    parse_json_response,
)
from ..document_text_parsing.processing import PARAGRAPH_SCHEMA
from .settings import DocumentClassificationSettings


ALLOWED_SECTIONS = frozenset(
    {"introductory", "reasoning", "operative"}
)
CLASSIFICATION_SCHEMA = pa.schema(
    [
        *PARAGRAPH_SCHEMA,
        pa.field("section", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ClassificationResult:
    parquet_bytes: bytes
    paragraph_count: int
    model_calls: int


def read_paragraphs_parquet(
    parquet_bytes: bytes,
    *,
    document_id: str,
) -> tuple[pa.Table, tuple[Paragraph, ...]]:
    table = pq.read_table(pa.BufferReader(parquet_bytes), use_threads=False)
    if table.schema != PARAGRAPH_SCHEMA:
        raise ValueError(
            "Unexpected paragraphs.parquet schema; expected "
            f"{PARAGRAPH_SCHEMA}, got {table.schema}"
        )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"Document {document_id} has no paragraph rows")
    expected_ids = list(range(1, len(rows) + 1))
    actual_ids = [row["paragraph_index"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            "Paragraph indexes must be consecutive and start at 1: "
            f"expected {expected_ids}, got {actual_ids}"
        )
    if any(str(row["document_id"]) != str(document_id) for row in rows):
        raise ValueError(
            f"paragraphs.parquet contains a document_id other than {document_id}"
        )
    paragraphs = tuple(
        Paragraph(
            paragraph_id=row["paragraph_index"],
            paragraph_order=row["paragraph_order"],
            text=row["text"],
        )
        for row in rows
    )
    return table, paragraphs


def compose_classification_messages(
    batch: ParagraphBatch,
    settings: DocumentClassificationSettings,
) -> list[dict[str, Any]]:
    context_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.context
    )
    target_text = "\n".join(
        paragraph_block(paragraph) for paragraph in batch.targets
    )
    target_ids = [paragraph.paragraph_id for paragraph in batch.targets]
    contract = {
        "type": "object",
        "description": (
            "Keys are target paragraph IDs encoded as strings; values are "
            "document-part classes"
        ),
        "propertyNames": {
            "enum": [str(paragraph_id) for paragraph_id in target_ids]
        },
        "additionalProperties": {
            "type": "string",
            "enum": sorted(ALLOWED_SECTIONS),
        },
        "minProperties": len(target_ids),
        "maxProperties": len(target_ids),
    }
    user_text = f"""The document is a globally numbered paragraph list.

TARGET PARAGRAPH IDS:
{json.dumps(target_ids)}

PRECEDING CONTEXT (read-only; do not classify these paragraphs):
{context_text or "(none)"}

TARGET PARAGRAPHS:
{target_text}

Return every target paragraph exactly once and in the same order. Return one
compact JSON object only.

JSON CONTRACT:
{json.dumps(contract, ensure_ascii=False, separators=(",", ":"))}
"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": settings.prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def validate_batch_classifications(
    payload: Mapping[str, Any],
    targets: Sequence[Paragraph],
) -> dict[int, str]:
    expected_ids = [paragraph.paragraph_id for paragraph in targets]
    expected_keys = [str(paragraph_id) for paragraph_id in expected_ids]
    returned_keys = list(payload)
    if returned_keys != expected_keys:
        raise ValueError(
            "Classification response must return every target paragraph as "
            "an ordered string key: "
            f"expected {expected_keys}, got {returned_keys}"
        )

    normalized: dict[int, str] = {}
    for paragraph_id in expected_ids:
        section = payload[str(paragraph_id)]
        if section not in ALLOWED_SECTIONS:
            raise ValueError(
                f"Unsupported section for paragraph {paragraph_id}: {section!r}"
            )
        normalized[paragraph_id] = section
    return normalized


def classify_paragraph_parquet(
    *,
    document_id: str,
    parquet_bytes: bytes,
    model_pipe: Any,
    tokenizer: Any,
    settings: DocumentClassificationSettings,
) -> ClassificationResult:
    settings.validate(production=True)
    source_table, paragraphs = read_paragraphs_parquet(
        parquet_bytes,
        document_id=document_id,
    )
    batches = build_paragraph_batches(
        paragraphs,
        tokenizer,
        target_chunk_tokens=settings.target_batch_tokens,
        overlap_tokens=settings.overlap_tokens,
    )
    classifications: dict[int, str] = {}
    for batch_index, batch in enumerate(batches, start=1):
        messages = compose_classification_messages(batch, settings)
        input_tokens = count_message_tokens(tokenizer, messages)
        if input_tokens + settings.max_new_tokens > settings.model_context_tokens:
            raise ValueError(
                f"Classification batch {batch_index} exceeds the model context: "
                f"input_tokens={input_tokens}, max_new_tokens="
                f"{settings.max_new_tokens}"
            )
        response = model_pipe(text=messages, return_full_text=False)
        response_text = extract_generated_text(response)
        payload = parse_json_response(response_text)
        classifications.update(
            validate_batch_classifications(payload, batch.targets)
        )

    expected_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    returned_ids = list(classifications)
    if returned_ids != expected_ids:
        raise AssertionError(
            f"Batches lost or reordered paragraphs: {returned_ids}"
        )
    rows = [
        {**row, "section": classifications[row["paragraph_index"]]}
        for row in source_table.to_pylist()
    ]
    result_table = pa.Table.from_pylist(rows, schema=CLASSIFICATION_SCHEMA)
    output = io.BytesIO()
    pq.write_table(
        result_table,
        output,
        compression=settings.parquet_compression,
        use_dictionary=True,
        write_statistics=True,
    )
    return ClassificationResult(
        parquet_bytes=output.getvalue(),
        paragraph_count=len(paragraphs),
        model_calls=len(batches),
    )
