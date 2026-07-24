from __future__ import annotations

import io
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .cloud import (
    build_manifest_identity,
    destination_object_path,
    source_object_path,
)
from .config import (
    BASE_SCHEMA,
    CRIMINAL_SCHEMA,
    ExtractionSettings,
    StorageSettings,
)
from .processing import (
    Paragraph,
    build_paragraph_batches,
    extract_document_rows,
    parse_document_to_parquet,
    parse_json_response,
    rows_to_parquet_bytes,
    rtf_bytes_to_paragraphs,
    validate_model_payload,
)

_VALIDATION_SCHEMA = pa.schema(
    [
        pa.field(
            "legal_references",
            pa.list_(
                pa.struct(
                    [
                        pa.field("act_id", pa.int32()),
                        pa.field("article", pa.int16()),
                        pa.field("part", pa.int16()),
                        pa.field("paragraph", pa.int16()),
                    ]
                )
            ),
        ),
        pa.field(
            "entities",
            pa.list_(
                pa.struct(
                    [
                        pa.field("entity_type", pa.int8()),
                        pa.field("start_offset", pa.int32()),
                        pa.field("end_offset", pa.int32()),
                        pa.field("normalized_entity_id", pa.string()),
                    ]
                )
            ),
        ),
        pa.field("labels", pa.list_(pa.int16())),
        pa.field("split", pa.int8()),
    ]
)


class _WhitespaceTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return list(range(len(text.split())))


def run_self_checks() -> None:
    reasoning_type = CRIMINAL_SCHEMA.field("reasoning_part").type
    argument_type = reasoning_type.field("court_reasoning_arguments").type
    legal_references_type = argument_type.value_type.field(
        "legal_references"
    ).type
    assert legal_references_type == pa.list_(pa.string())

    example_settings = ExtractionSettings(
        prompt="test prompt",
        extraction_schema=_VALIDATION_SCHEMA,
    )
    example_settings.validate(production=True)
    assert (
        example_settings.output_schema.names[: len(BASE_SCHEMA.names)]
        == BASE_SCHEMA.names
    )

    reduced_schema = pa.schema(
        [pa.field("labels", pa.list_(pa.int16()))]
    )
    reduced_settings = ExtractionSettings(
        prompt="test prompt",
        extraction_schema=reduced_schema,
    )
    assert reduced_settings.output_schema.names == [
        *BASE_SCHEMA.names,
        "labels",
    ]

    sample_rtf = (
        r"{\rtf1\ansi\uc1 "
        r"\u1055?\u1077?\u1088?\u1096?\u1080?\u1081?"
        r"\par\par Second paragraph}"
    ).encode("latin-1")
    parsed = rtf_bytes_to_paragraphs(sample_rtf)
    assert len(parsed) == 2
    assert parsed[0].text == "Перший"
    assert parsed[1].paragraph_id == 2

    tokenizer = _WhitespaceTokenizer()
    paragraphs = [
        Paragraph(index, index, "word " * 6 + str(index))
        for index in range(1, 7)
    ]
    batches = build_paragraph_batches(
        paragraphs,
        tokenizer,
        target_chunk_tokens=25,
        overlap_tokens=5,
    )
    assert [
        paragraph.paragraph_id
        for batch in batches
        for paragraph in batch.targets
    ] == list(range(1, 7))
    assert all(batch.context for batch in batches[1:])

    targets = [
        Paragraph(1, 1, "Law applies to Alice."),
        Paragraph(2, 2, "No reference."),
    ]
    payload = {
        "paragraphs": [
            {
                "paragraph_id": 1,
                "section_id": 0,
                "legal_references": [
                    {
                        "act_id": 1,
                        "article": 2,
                        "part": 3,
                        "paragraph": 4,
                    }
                ],
                "entities": [
                    {
                        "entity_type": 1,
                        "start_offset": 15,
                        "end_offset": 20,
                        "normalized_entity_id": "person:alice",
                    }
                ],
                "labels": [1, 2],
                "split": 0,
            },
            {
                "paragraph_id": 2,
                "section_id": 1,
                "legal_references": [],
                "entities": [],
                "labels": [],
                "split": 1,
            },
        ]
    }
    validated, last_section = validate_model_payload(
        payload,
        targets,
        _VALIDATION_SCHEMA,
        previous_section_id=None,
    )
    assert last_section == 1

    document_rows = [
        {
            "document_id": "123",
            "text": "\n\n".join(
                paragraph.text for paragraph in targets
            ),
            **{
                name: validated[0][name]
                for name in _VALIDATION_SCHEMA.names
            },
        }
    ]
    parquet_bytes = rows_to_parquet_bytes(
        document_rows,
        example_settings.output_schema,
        "zstd",
    )
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.schema == example_settings.output_schema
    assert table.to_pylist() == document_rows

    model_calls: list[list[dict[str, Any]]] = []
    document_payload = {
        "decision_stage": "final",
        "paragraph_classification": [
            {"paragraph_index": 1, "section": "introductory"},
            {"paragraph_index": 2, "section": "reasoning"},
        ],
        "operative_part": {
            "guilt_status": "guilty",
            "conviction_operative": {
                "conviction_decision": "Found guilty",
                "final_sentence": "Fine",
                "probation": None,
            },
        },
    }

    def model_pipe(*, text, **_kwargs):
        model_calls.append(text)
        return [{"generated_text": json.dumps(document_payload)}]

    full_document_settings = ExtractionSettings(
        prompt="test prompt",
        extraction_schema=CRIMINAL_SCHEMA,
    )
    full_document_rows = extract_document_rows(
        "123",
        targets,
        model_pipe,
        tokenizer,
        full_document_settings,
    )
    assert len(model_calls) == 1
    sent_document = model_calls[0][1]["content"][0]["text"]
    assert all(paragraph.text in sent_document for paragraph in targets)
    assert len(full_document_rows) == 1
    assert full_document_rows[0]["document_id"] == "123"
    assert full_document_rows[0]["text"] == "\n\n".join(
        paragraph.text for paragraph in targets
    )
    assert full_document_rows[0]["decision_stage"] == "final"
    normalized_operative = full_document_rows[0]["operative_part"]
    assert normalized_operative["juvenile_educator_decision"] is None
    assert normalized_operative["final_sentence"] == "Fine"
    assert normalized_operative["probation"] is None
    assert (
        normalized_operative["conviction_operative"][
            "conviction_decision"
        ]
        == "Found guilty"
    )

    parsed_parquet, paragraph_count = parse_document_to_parquet(
        "123",
        sample_rtf,
        model_pipe,
        tokenizer,
        full_document_settings,
    )
    parsed_table = pq.read_table(io.BytesIO(parsed_parquet))
    assert paragraph_count == 2
    assert parsed_table.num_rows == 1
    assert parsed_table.schema == full_document_settings.output_schema
    assert parsed_table.column("document_id").to_pylist() == ["123"]
    assert parsed_table.column("text").to_pylist() == [
        "Перший\n\nSecond paragraph"
    ]

    def assert_invalid(candidate: dict[str, Any]) -> None:
        try:
            validate_model_payload(
                candidate,
                targets,
                _VALIDATION_SCHEMA,
                previous_section_id=None,
            )
        except (TypeError, ValueError):
            return
        raise AssertionError("Invalid model payload was accepted")

    bad_offsets = json.loads(json.dumps(payload))
    bad_offsets["paragraphs"][0]["entities"][0]["end_offset"] = 999
    assert_invalid(bad_offsets)

    missing_paragraph = json.loads(json.dumps(payload))
    missing_paragraph["paragraphs"].pop()
    assert_invalid(missing_paragraph)

    duplicate_paragraph = json.loads(json.dumps(payload))
    duplicate_paragraph["paragraphs"].append(
        dict(duplicate_paragraph["paragraphs"][0])
    )
    assert_invalid(duplicate_paragraph)

    unexpected_field = json.loads(json.dumps(payload))
    unexpected_field["paragraphs"][0]["unexpected"] = True
    assert_invalid(unexpected_field)

    integer_overflow = json.loads(json.dumps(payload))
    integer_overflow["paragraphs"][0]["split"] = 128
    assert_invalid(integer_overflow)

    section_jump = json.loads(json.dumps(payload))
    section_jump["paragraphs"][1]["section_id"] = 2
    assert_invalid(section_jump)

    fenced = parse_json_response(
        '```json\n{"paragraphs": []}\n```'
    )
    assert fenced == {"paragraphs": []}

    assert source_object_path(2, "121118598") == "2/121118598.rtf"
    assert (
        destination_object_path(
            "info_version_1",
            2,
            "121118598",
        )
        == "info_version_1/2/121118598.parquet"
    )

    identity_v1 = build_manifest_identity(
        reduced_settings,
        StorageSettings(),
        "commit-1",
    )
    identity_changed = build_manifest_identity(
        ExtractionSettings(
            prompt="changed",
            extraction_schema=reduced_schema,
        ),
        StorageSettings(),
        "commit-1",
    )
    assert identity_v1 != identity_changed
    print("All document_split self-checks passed.")


if __name__ == "__main__":
    run_self_checks()
