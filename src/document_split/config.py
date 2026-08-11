from __future__ import annotations

import re
from dataclasses import dataclass

import pyarrow as pa


PROJECT_ID = "lab-test-project-1-305710"
BIGQUERY_TABLE = f"{PROJECT_ID}.court_data_2024.document_data"
SOURCE_BUCKET = "court_data_2024"
DESTINATION_BUCKET = "court_data_2024_structured"
INFO_VERSION = "info_version_10"
PART_DOCUMENT_PREFIX = "document_text_parsing/info_version_9"

MODEL_ID = "lapa-llm/lapa-v0.1.2-instruct"
MODEL_REVISION: str | None = None

PROMPT_PLACEHOLDER_MARKER = "<DEFINE_EXTRACTION_INSTRUCTIONS>"
DEFAULT_EXTRACTION_PROMPT = (
    "Handler-specific part prompts define all extraction instructions."
)

BASE_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)

CRIMINAL_SCHEMA = pa.schema([
    pa.field("introductory_part", pa.struct([
        pa.field("decision_date", pa.string()),
        pa.field("decision_place", pa.string()),
        pa.field("court_composition", pa.string()),
        pa.field("criminal_proceeding_number", pa.string()),
        pa.field("accused_birth_date", pa.string()),
        pa.field("accused_birth_place", pa.string()),
        pa.field("accused_residence_place", pa.string()),
        pa.field("accused_occupation", pa.string()),
        pa.field("accused_education", pa.string()),
        pa.field("accused_marital_status", pa.string()),
        pa.field("criminal_law_article", pa.list_(
            pa.struct([
                pa.field("paragraph", pa.string()),
                pa.field("part", pa.string()),
                pa.field("article", pa.string()),
                pa.field("act_name", pa.string()),
                pa.field("adoption_date", pa.string()),
            ])
        )),
        pa.field("proceeding_parties_and_participants", pa.list_(
            pa.struct([
                pa.field("name", pa.string()),
                pa.field("participant_role", pa.string()),
                pa.field("participant_group", pa.string()),
                pa.field("credentials", pa.string()),
                pa.field("relation_to_case", pa.string()),
            ])
        ))
    ])),
    pa.field("reasoning_part", pa.struct([
        pa.field("guilt_status", pa.string()),
        pa.field("court_reasoning_arguments", pa.list_(
            pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("argument_type", pa.string()),
                pa.field("argument", pa.string()),
                pa.field("legal_references", pa.list_(pa.string())),
            ])
        )),
        pa.field("acquittal_reasoning", pa.struct([
            pa.field("unproven_charge", pa.string()),
            pa.field("acquittal_grounds", pa.string()),
            pa.field("rejected_prosecution_evidence_reasons", pa.string()),
            pa.field("other_decision_reasons", pa.string()),
        ])),
        pa.field("conviction_reasoning", pa.struct([
            pa.field("proven_charge", pa.string()),
            pa.field("offense_place", pa.string()),
            pa.field("offense_time", pa.string()),
            pa.field("offense_method", pa.string()),
            pa.field("offense_consequences", pa.string()),
            pa.field("form_of_guilt", pa.string()),
            pa.field("offense_motive", pa.string()),
            pa.field("conviction_law_articles", pa.list_(pa.struct([
                pa.field("paragraph", pa.string()),
                pa.field("part", pa.string()),
                pa.field("article", pa.string()),
                pa.field("act_name", pa.string()),
                pa.field("adoption_date", pa.string()),
            ]))),
            pa.field("supporting_evidence", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("evidence_type", pa.string()),
                pa.field("evidence", pa.string()),
                pa.field("established_circumstance", pa.string()),
                pa.field("court_assessment", pa.string()),
            ]))),
            pa.field("rejected_evidence_reasons", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("evidence", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("decision", pa.string()),
            ]))),
            pa.field("charge_change_reasons", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("original_charge", pa.string()),
                pa.field("changed_charge", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("legal_basis", pa.string()),
            ]))),
            pa.field("unsubstantiated_charge_reasons", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("charge", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("outcome", pa.string()),
            ]))),
            pa.field("mitigating_circumstances", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("circumstance", pa.string()),
                pa.field("court_conclusion", pa.string()),
                pa.field("legal_basis", pa.string()),
            ]))),
            pa.field("aggravating_circumstances", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("circumstance", pa.string()),
                pa.field("court_conclusion", pa.string()),
                pa.field("legal_basis", pa.string()),
            ]))),
            pa.field("sentencing_reasons", pa.list_(pa.struct([
                pa.field("paragraph_index", pa.int32()),
                pa.field("reason_type", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("considered_factors", pa.string()),
                pa.field("legal_basis", pa.string()),
            ]))),
            pa.field("civil_claim_reasons", pa.string()),
            pa.field("other_decision_reasons", pa.string()),
            pa.field("applied_legal_provisions", pa.string()),
        ]))
    ])),
    pa.field("operative_part", pa.struct([
        pa.field("guilt_status", pa.string()),
        pa.field("acquittal_operative", pa.struct([
            pa.field("accused_name", pa.string()),
            pa.field("acquittal_decision", pa.string()),
            pa.field("restoration_of_rights", pa.string()),
            pa.field("security_measures_decision", pa.string()),
            pa.field("physical_evidence_decision", pa.string()),
            pa.field("procedural_costs_decision", pa.string()),
            pa.field("entry_into_force_and_appeal", pa.string()),
            pa.field("copy_receipt_procedure", pa.string()),
        ])),
        pa.field("conviction_operative", pa.struct([
            pa.field("accused_name", pa.string()),
            pa.field("conviction_decision", pa.string()),
            pa.field("conviction_law_articles", pa.list_(pa.struct([
                pa.field("paragraph", pa.string()),
                pa.field("part", pa.string()),
                pa.field("article", pa.string()),
                pa.field("act_name", pa.string()),
                pa.field("adoption_date", pa.string()),
            ]))),
            pa.field("penalties_by_charge", pa.list_(pa.struct([
                pa.field("charge", pa.string()),
                pa.field("law_article", pa.struct([
                    pa.field("paragraph", pa.string()),
                    pa.field("part", pa.string()),
                    pa.field("article", pa.string()),
                    pa.field("act_name", pa.string()),
                    pa.field("adoption_date", pa.string()),
                ])),
                pa.field("punishment", pa.string())
            ]))),
        ])),
        pa.field("final_sentence", pa.string()),
        pa.field("sentence_start", pa.string()),
        pa.field("compulsory_treatment_decision", pa.string()),
        pa.field("juvenile_educator_decision", pa.string()),
        pa.field("civil_claim_decision", pa.string()),
        pa.field("property_recovery_decision", pa.string()),
        pa.field("physical_evidence_decision", pa.string()),
        pa.field("costs_reimbursement_decision", pa.string()),
        pa.field("security_measures_decision", pa.string()),
        pa.field("pretrial_detention_credit", pa.string()),
        pa.field("entry_into_force_and_appeal", pa.string()),
        pa.field("copy_receipt_procedure", pa.string()),
        pa.field("charge_outcomes", pa.list_(pa.struct([
            pa.field("charge", pa.string()),
            pa.field("outcome", pa.string()),
            pa.field("law_article", pa.struct([
                pa.field("paragraph", pa.string()),
                pa.field("part", pa.string()),
                pa.field("article", pa.string()),
                pa.field("act_name", pa.string()),
                pa.field("adoption_date", pa.string()),
            ])),
        ]))),
        pa.field("release_from_punishment", pa.string()),
        pa.field("probation", pa.struct([
            pa.field("applied", pa.bool_()),
            pa.field("probation_period", pa.string()),
            pa.field("obligations", pa.string()),
            pa.field("supervising_persons_or_authorities", pa.string()),
        ]))
    ]))
])

@dataclass(frozen=True)
class ExtractionSettings:
    prompt: str
    extraction_schema: pa.Schema
    model_id: str = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    model_context_tokens: int = 131_072
    max_new_tokens: int = 32_768
    json_retries: int = 0
    parquet_compression: str = "zstd"

    @property
    def output_schema(self) -> pa.Schema:
        return pa.schema([*BASE_SCHEMA, *self.extraction_schema])

    def validate(self, production: bool = False) -> None:
        if production and (
            not self.prompt.strip() or PROMPT_PLACEHOLDER_MARKER in self.prompt
        ):
            raise ValueError(
                "Replace the research prompt before starting a production run"
            )

        reserved = set(BASE_SCHEMA.names)
        duplicates = reserved.intersection(self.extraction_schema.names)
        if duplicates:
            raise ValueError(
                f"Extraction fields conflict with base fields: {sorted(duplicates)}"
            )
        if len(set(self.extraction_schema.names)) != len(
            self.extraction_schema.names
        ):
            raise ValueError("Extraction field names must be unique")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.json_retries < 0:
            raise ValueError("json_retries cannot be negative")
        if self.parquet_compression not in {"zstd", "gzip", "snappy"}:
            raise ValueError("Unsupported Parquet compression")


@dataclass(frozen=True)
class StorageSettings:
    project_id: str = PROJECT_ID
    bigquery_table: str = BIGQUERY_TABLE
    source_bucket: str = SOURCE_BUCKET
    destination_bucket: str = DESTINATION_BUCKET
    parts_bucket: str = DESTINATION_BUCKET
    parts_prefix: str = PART_DOCUMENT_PREFIX
    info_version: str = INFO_VERSION
    justice_kinds: tuple[int, ...] = (2,)
    bigquery_page_size: int = 1_000
    limit: int | None = None
    skip_existing: bool = True
    run_log_flush_size: int = 100

    @property
    def version_prefix(self) -> str:
        return self.info_version.strip("/")

    def validate(self) -> None:
        if not re.fullmatch(
            r"info_version_[1-9][0-9]*", self.version_prefix
        ):
            raise ValueError("info_version must look like info_version_1")
        if not self.justice_kinds:
            raise ValueError("At least one justice kind is required")
        if not self.parts_bucket.strip():
            raise ValueError("parts_bucket cannot be empty")
        if not self.parts_prefix.strip().strip("/"):
            raise ValueError("parts_prefix cannot be empty")
        if self.bigquery_page_size <= 0:
            raise ValueError("bigquery_page_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
