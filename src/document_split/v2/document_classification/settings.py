from __future__ import annotations

import re
from dataclasses import dataclass

from ...config import (
    BIGQUERY_TABLE,
    DESTINATION_BUCKET,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ID,
)
from ..document_text_parsing.settings import DOCUMENT_TEXT_PARSING_ROOT
from .prompts import (
    CLASSIFICATION_CRITERIA_PLACEHOLDER,
    DOCUMENT_PART_CLASSIFICATION_PROMPT,
)


@dataclass(frozen=True)
class DocumentClassificationSettings:
    project_id: str = PROJECT_ID
    bigquery_table: str = BIGQUERY_TABLE
    bucket: str = DESTINATION_BUCKET
    document_prefix: str = DOCUMENT_TEXT_PARSING_ROOT
    source_parquet_name: str = "paragraphs.parquet"
    result_parquet_name: str = "classification.parquet"
    progress_column: str = "is_classified"
    justice_kinds: tuple[int, ...] = (2,)
    document_ids: tuple[int, ...] | None = None
    bigquery_page_size: int = 500
    progress_update_batch_size: int = 100
    limit: int | None = None
    show_progress: bool = True
    run_log_flush_size: int = 100
    prompt: str = DOCUMENT_PART_CLASSIFICATION_PROMPT
    model_id: str = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    model_context_tokens: int = 131_072
    max_new_tokens: int = 8_192
    target_batch_tokens: int = 6_000
    overlap_tokens: int = 256
    parquet_compression: str = "zstd"
    auth_mode: str = "adc"
    colab_service_account_secret: str = "cloud_access"

    @property
    def normalized_document_prefix(self) -> str:
        return self.document_prefix.strip().strip("/")

    def validate(self, *, production: bool = True) -> None:
        if not self.project_id.strip():
            raise ValueError("project_id cannot be empty")
        if not re.fullmatch(
            r"[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+",
            self.bigquery_table,
        ):
            raise ValueError(
                "bigquery_table must be a fully-qualified "
                "project.dataset.table identifier"
            )
        if not self.bucket.strip():
            raise ValueError("bucket cannot be empty")
        if not self.normalized_document_prefix:
            raise ValueError("document_prefix cannot be empty")
        for field_name, value in (
            ("source_parquet_name", self.source_parquet_name),
            ("result_parquet_name", self.result_parquet_name),
        ):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.parquet", value):
                raise ValueError(f"{field_name} must be a Parquet filename")
        if self.source_parquet_name == self.result_parquet_name:
            raise ValueError("Source and result Parquet names must differ")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.progress_column):
            raise ValueError("progress_column must be a BigQuery column name")
        if not self.justice_kinds or any(
            type(value) is not int or value <= 0
            for value in self.justice_kinds
        ):
            raise ValueError("justice_kinds must contain positive integers")
        if self.document_ids is not None and (
            not self.document_ids
            or any(
                type(value) is not int or value <= 0
                for value in self.document_ids
            )
        ):
            raise ValueError("document_ids must contain positive integers")
        for field_name, value in (
            ("bigquery_page_size", self.bigquery_page_size),
            ("progress_update_batch_size", self.progress_update_batch_size),
            ("run_log_flush_size", self.run_log_flush_size),
            ("model_context_tokens", self.model_context_tokens),
            ("max_new_tokens", self.max_new_tokens),
            ("target_batch_tokens", self.target_batch_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.max_new_tokens >= self.model_context_tokens:
            raise ValueError(
                "max_new_tokens must be below model_context_tokens"
            )
        if self.parquet_compression not in {"zstd", "gzip", "snappy"}:
            raise ValueError("Unsupported Parquet compression")
        if self.auth_mode not in {"adc", "colab_secret", "colab_user"}:
            raise ValueError(
                "auth_mode must be 'adc', 'colab_secret', or 'colab_user'"
            )
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if production and CLASSIFICATION_CRITERIA_PLACEHOLDER in self.prompt:
            raise ValueError(
                "Fill DOCUMENT_PART_CLASSIFICATION_PROMPT classification "
                "criteria before running the production pipeline"
            )


DEFAULT_DOCUMENT_CLASSIFICATION_SETTINGS = (
    DocumentClassificationSettings()
)
