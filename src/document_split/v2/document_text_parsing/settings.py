from __future__ import annotations

import re
from dataclasses import dataclass

from ...config import (
    BIGQUERY_TABLE,
    DESTINATION_BUCKET,
    PROJECT_ID,
    SOURCE_BUCKET,
)
from ..settings import V2_INFO_VERSION


DOCUMENT_TEXT_PARSING_ROOT = (
    f"document_text_parsing/{V2_INFO_VERSION}"
)


def _normalize_prefix(value: str) -> str:
    return value.strip().strip("/")


@dataclass(frozen=True)
class DocumentTextParsingSettings:
    project_id: str = PROJECT_ID
    bigquery_table: str = BIGQUERY_TABLE
    source_bucket: str = SOURCE_BUCKET
    destination_bucket: str = DESTINATION_BUCKET
    source_prefix: str = ""
    destination_prefix: str = DOCUMENT_TEXT_PARSING_ROOT
    justice_kinds: tuple[int, ...] = (2,)
    batch_size: int = 500
    max_workers: int = 5
    limit: int | None = None
    overwrite_existing: bool = False
    show_progress: bool = True
    run_log_flush_size: int = 100

    @property
    def normalized_source_prefix(self) -> str:
        return _normalize_prefix(self.source_prefix)

    @property
    def normalized_destination_prefix(self) -> str:
        return _normalize_prefix(self.destination_prefix)

    def validate(self) -> None:
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
        if not self.source_bucket.strip():
            raise ValueError("source_bucket cannot be empty")
        if not self.destination_bucket.strip():
            raise ValueError("destination_bucket cannot be empty")
        if self.source_bucket == self.destination_bucket:
            raise ValueError(
                "source_bucket and destination_bucket must be different"
            )
        if not self.normalized_destination_prefix:
            raise ValueError("destination_prefix cannot be empty")
        if not self.justice_kinds:
            raise ValueError("At least one justice kind is required")
        if any(
            type(justice_kind) is not int or justice_kind <= 0
            for justice_kind in self.justice_kinds
        ):
            raise ValueError("justice_kinds must contain positive integers")
        if len(set(self.justice_kinds)) != len(self.justice_kinds):
            raise ValueError("justice_kinds cannot contain duplicates")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if type(self.show_progress) is not bool:
            raise ValueError("show_progress must be a boolean")
        if self.run_log_flush_size <= 0:
            raise ValueError("run_log_flush_size must be positive")


DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS = (
    DocumentTextParsingSettings()
)
