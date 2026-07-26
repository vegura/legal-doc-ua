from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ...processing import paragraph_block, rtf_bytes_to_paragraphs
from ..state import V2DocumentState


PARAGRAPH_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("paragraph_index", pa.int32(), nullable=False),
        pa.field("paragraph_order", pa.int32(), nullable=False),
        pa.field("numbered_text", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)
PARQUET_COMPRESSION = "zstd"


def populate_paragraph_artifacts(
    state: V2DocumentState,
) -> V2DocumentState:
    paragraphs = tuple(rtf_bytes_to_paragraphs(state.raw_rtf))
    if not paragraphs:
        raise ValueError(
            f"Document {state.document_id} contains no paragraphs"
        )

    state.paragraphs = paragraphs
    state.numbered_text = "\n".join(
        paragraph_block(paragraph) for paragraph in paragraphs
    )
    rows = [
        {
            "document_id": str(state.document_id),
            "paragraph_index": paragraph.paragraph_id,
            "paragraph_order": paragraph.paragraph_order,
            "numbered_text": paragraph_block(paragraph),
            "text": paragraph.text,
        }
        for paragraph in paragraphs
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARAGRAPH_SCHEMA),
        state.artifact_path("paragraphs.parquet"),
        compression=PARQUET_COMPRESSION,
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


def parse_document_to_artifacts(
    *,
    document_id: str,
    justice_kind: int,
    raw_rtf: bytes,
    output_dir: Path,
) -> V2DocumentState:
    state = V2DocumentState(
        document_id=str(document_id),
        justice_kind=int(justice_kind),
        raw_rtf=raw_rtf,
        work_dir=Path(output_dir),
    )
    return populate_paragraph_artifacts(state)
