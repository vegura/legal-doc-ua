from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import BASE_SCHEMA, CRIMINAL_PROMPT, CRIMINAL_SCHEMA
from .processing import (
    Paragraph,
    ParagraphBatch,
    build_messages,
    parse_json_response,
    rtf_bytes_to_paragraphs,
    validate_document_payload,
)


OPENAI_MODEL = "gpt-5.6"


@dataclass(frozen=True)
class OpenAIExtractionSettings:
    """Configuration for the independent OpenAI extraction path."""

    prompt: str = CRIMINAL_PROMPT
    extraction_schema: pa.Schema = CRIMINAL_SCHEMA
    model: str = OPENAI_MODEL
    max_output_tokens: int = 32_768
    parquet_compression: str = "zstd"
    store_response: bool = False
    timeout_seconds: float = 900.0

    @property
    def output_schema(self) -> pa.Schema:
        return pa.schema([*BASE_SCHEMA, *self.extraction_schema])

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("OpenAI extraction prompt cannot be empty")
        if not self.model.strip():
            raise ValueError("OpenAI model cannot be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.parquet_compression not in {"zstd", "gzip", "snappy"}:
            raise ValueError("Unsupported Parquet compression")
        duplicates = set(BASE_SCHEMA.names).intersection(
            self.extraction_schema.names
        )
        if duplicates:
            raise ValueError(
                "Extraction fields conflict with base fields: "
                f"{sorted(duplicates)}"
            )


DEFAULT_OPENAI_EXTRACTION_SETTINGS = OpenAIExtractionSettings()


@dataclass(frozen=True)
class ComposedOpenAIPrompt:
    instructions: str
    input: str


@dataclass(frozen=True)
class OpenAIExtractionResult:
    document: dict[str, Any]
    response_id: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class OpenAIParquetResult:
    parquet_bytes: bytes
    paragraph_count: int
    response_id: str | None
    input_tokens: int
    output_tokens: int


def compose_openai_prompt(
    paragraphs: Sequence[Paragraph],
    settings: OpenAIExtractionSettings = DEFAULT_OPENAI_EXTRACTION_SETTINGS,
) -> ComposedOpenAIPrompt:
    """Compose the full-document prompt and canonical schema contract."""

    settings.validate()
    if not paragraphs:
        raise ValueError("Cannot compose a prompt for an empty document")

    messages = build_messages(
        settings.prompt,
        settings.extraction_schema,
        ParagraphBatch(context=(), targets=tuple(paragraphs)),
        known_section_ids={},
    )
    return ComposedOpenAIPrompt(
        instructions=_message_text(messages[0]),
        input=_message_text(messages[1]),
    )


def create_openai_client(
    api_key: str | None = None,
    *,
    timeout_seconds: float = 900.0,
):
    """Create an OpenAI client; without api_key the SDK reads OPENAI_API_KEY."""

    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=timeout_seconds)


def extract_document_with_openai(
    client,
    paragraphs: Sequence[Paragraph],
    settings: OpenAIExtractionSettings = DEFAULT_OPENAI_EXTRACTION_SETTINGS,
) -> OpenAIExtractionResult:
    """Call the Responses API once and validate the result against PyArrow."""

    prompt = compose_openai_prompt(paragraphs, settings)
    response = client.responses.create(
        model=settings.model,
        instructions=prompt.instructions,
        input=prompt.input,
        max_output_tokens=settings.max_output_tokens,
        text={"format": {"type": "json_object"}},
        store=settings.store_response,
        timeout=settings.timeout_seconds,
    )

    status = getattr(response, "status", None)
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        raise RuntimeError(
            f"OpenAI response did not complete: status={status!r}, "
            f"details={details!r}, response_id="
            f"{getattr(response, 'id', None)!r}"
        )

    response_text = getattr(response, "output_text", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError(
            "OpenAI response completed without output text; response_id="
            f"{getattr(response, 'id', None)!r}"
        )

    payload = parse_json_response(response_text)
    document, _section_ids = validate_document_payload(
        payload,
        paragraphs,
        settings.extraction_schema,
    )
    usage = getattr(response, "usage", None)
    return OpenAIExtractionResult(
        document=document,
        response_id=getattr(response, "id", None),
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
    )


def parse_document_to_parquet_openai(
    document_id: str,
    rtf_bytes: bytes,
    client,
    settings: OpenAIExtractionSettings = DEFAULT_OPENAI_EXTRACTION_SETTINGS,
) -> OpenAIParquetResult:
    """Extract one complete RTF document into one schema-valid Parquet row."""

    paragraphs = rtf_bytes_to_paragraphs(rtf_bytes)
    if not paragraphs:
        raise ValueError(f"Document {document_id} contains no paragraphs")

    extraction = extract_document_with_openai(
        client,
        paragraphs,
        settings,
    )
    row = {
        "document_id": str(document_id),
        "text": "\n\n".join(paragraph.text for paragraph in paragraphs),
        **{
            name: extraction.document[name]
            for name in settings.extraction_schema.names
        },
    }
    table = pa.Table.from_pylist([row], schema=settings.output_schema)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression=settings.parquet_compression,
        use_dictionary=True,
        write_statistics=True,
    )
    return OpenAIParquetResult(
        parquet_bytes=buffer.getvalue(),
        paragraph_count=len(paragraphs),
        response_id=extraction.response_id,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
    )


def _message_text(message: Mapping[str, Any]) -> str:
    parts = [
        str(item.get("text", ""))
        for item in message.get("content", [])
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    text = "".join(parts).strip()
    if not text:
        raise ValueError("Composed prompt message contains no text")
    return text


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, Mapping):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    return int(value or 0)
