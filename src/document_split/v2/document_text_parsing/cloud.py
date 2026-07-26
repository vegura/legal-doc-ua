from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from .processing import (
    PARAGRAPH_SCHEMA,
    PARQUET_COMPRESSION,
)
from ..settings import V2_INFO_VERSION
from .settings import DocumentTextParsingSettings


@dataclass(frozen=True)
class SourceDocument:
    justice_kind: int
    document_id: str
    object_name: str


def _join_object_path(*parts: str) -> str:
    return "/".join(
        part.strip("/")
        for part in parts
        if part and part.strip("/")
    )


def source_object_path(
    source_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return _join_object_path(
        source_prefix,
        str(int(justice_kind)),
        f"{document_id}.rtf",
    )


def document_destination_prefix(
    destination_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return _join_object_path(
        destination_prefix,
        str(int(justice_kind)),
        str(document_id),
    )


def paragraph_object_path(
    destination_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    document_prefix = document_destination_prefix(
        destination_prefix,
        justice_kind,
        document_id,
    )
    return f"{document_prefix}/paragraphs.parquet"


def numbered_document_object_path(
    destination_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    document_prefix = document_destination_prefix(
        destination_prefix,
        justice_kind,
        document_id,
    )
    return f"{document_prefix}/numbered_document.json"


def document_marker_object_path(
    destination_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    document_prefix = document_destination_prefix(
        destination_prefix,
        justice_kind,
        document_id,
    )
    return f"{document_prefix}/_document.json"


def list_completed_document_ids(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
    justice_kind: int,
) -> set[str]:
    prefix = _join_object_path(
        settings.normalized_destination_prefix,
        str(int(justice_kind)),
    ) + "/"
    completed: set[str] = set()
    for blob in storage_client.list_blobs(
        settings.destination_bucket,
        prefix=prefix,
    ):
        relative = blob.name[len(prefix) :]
        parts = relative.split("/")
        if (
            len(parts) == 2
            and parts[0]
            and parts[1] == "paragraphs.parquet"
        ):
            completed.add(parts[0])
    return completed


def build_manifest_identity(
    settings: DocumentTextParsingSettings,
) -> dict[str, Any]:
    return {
        "pipeline": "document_split.v2.document_text_parsing",
        "info_version": V2_INFO_VERSION,
        "source": {
            "bigquery_table": settings.bigquery_table,
            "is_parsed_column": "is_parsed",
            "source_bucket": settings.source_bucket,
            "source_prefix": settings.normalized_source_prefix,
            "object_path": (
                "{source_prefix}/{justice_kind}/{doc_id}.rtf"
            ),
        },
        "destination_prefix": (
            settings.normalized_destination_prefix
        ),
        "arrow_schema": PARAGRAPH_SCHEMA.to_string(
            show_field_metadata=True,
            show_schema_metadata=True,
        ),
        "parquet": {
            "compression": PARQUET_COMPRESSION,
            "row_unit": "paragraph",
        },
        "normalization": {
            "rtf_byte_decode": "latin-1",
            "rtf_errors": "ignore",
            "unicode_form": "NFC",
            "normalize_newlines": True,
            "replace_nbsp": True,
            "strip_control_characters": True,
            "paragraph_unit": "non-empty rendered line",
            "paragraph_ids_start_at": 1,
        },
        "artifacts": {
            "marker": "_document.json",
            "numbered_document": "numbered_document.json",
            "completion": "paragraphs.parquet",
        },
    }


def prepare_manifest(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
) -> dict[str, Any]:
    manifest_path = (
        f"{settings.normalized_destination_prefix}/manifest.json"
    )
    blob = storage_client.bucket(
        settings.destination_bucket
    ).blob(manifest_path)
    expected_identity = build_manifest_identity(settings)

    if blob.exists(client=storage_client):
        manifest = json.loads(blob.download_as_text())
        if manifest.get("identity") != expected_identity:
            raise RuntimeError(
                f"Paragraph parsing settings differ from "
                f"gs://{settings.destination_bucket}/{manifest_path}; "
                "choose a new destination_prefix"
            )
        return manifest

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": expected_identity,
    }
    try:
        blob.upload_from_string(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            if_generation_match=0,
        )
    except PreconditionFailed:
        return prepare_manifest(storage_client, settings)
    return manifest


def download_source_document(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
    source: SourceDocument,
) -> bytes:
    blob = storage_client.bucket(settings.source_bucket).blob(
        source.object_name
    )
    return blob.download_as_bytes()


def _upload_marker(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
    source: SourceDocument,
) -> None:
    object_path = document_marker_object_path(
        settings.normalized_destination_prefix,
        source.justice_kind,
        source.document_id,
    )
    payload = {
        "document_id": source.document_id,
        "justice_kind": source.justice_kind,
        "source_bucket": settings.source_bucket,
        "source_object": source.object_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    kwargs: dict[str, Any] = {}
    if not settings.overwrite_existing:
        kwargs["if_generation_match"] = 0
    try:
        storage_client.bucket(settings.destination_bucket).blob(
            object_path
        ).upload_from_string(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            **kwargs,
        )
    except PreconditionFailed:
        pass


def upload_document_artifacts(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
    source: SourceDocument,
    work_dir: Path,
) -> bool:
    _upload_marker(storage_client, settings, source)

    numbered_path = numbered_document_object_path(
        settings.normalized_destination_prefix,
        source.justice_kind,
        source.document_id,
    )
    storage_client.bucket(settings.destination_bucket).blob(
        numbered_path
    ).upload_from_filename(
        str(work_dir / "numbered_document.json"),
        content_type="application/json; charset=utf-8",
    )

    parquet_path = paragraph_object_path(
        settings.normalized_destination_prefix,
        source.justice_kind,
        source.document_id,
    )
    kwargs: dict[str, Any] = {}
    if not settings.overwrite_existing:
        kwargs["if_generation_match"] = 0
    try:
        storage_client.bucket(settings.destination_bucket).blob(
            parquet_path
        ).upload_from_filename(
            str(work_dir / "paragraphs.parquet"),
            content_type="application/vnd.apache.parquet",
            **kwargs,
        )
    except PreconditionFailed:
        return False
    return True


@dataclass
class ParagraphRunLogger:
    storage_client: storage.Client
    bucket_name: str
    destination_prefix: str
    flush_size: int = 100
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ-"
        )
        + uuid.uuid4().hex[:8]
    )
    pending: list[dict[str, Any]] = field(default_factory=list)
    shard_index: int = 0

    def record(self, event: Mapping[str, Any]) -> None:
        self.pending.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **dict(event),
            }
        )
        if len(self.pending) >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.shard_index += 1
        object_path = (
            f"{self.destination_prefix}/_runs/{self.run_id}/"
            f"events-{self.shard_index:06d}.jsonl"
        )
        payload = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True)
            + "\n"
            for event in self.pending
        )
        self.storage_client.bucket(self.bucket_name).blob(
            object_path
        ).upload_from_string(
            payload.encode("utf-8"),
            content_type="application/x-ndjson; charset=utf-8",
            if_generation_match=0,
        )
        self.pending.clear()

    def close(self, summary: Mapping[str, Any]) -> None:
        self.flush()
        object_path = (
            f"{self.destination_prefix}/_runs/{self.run_id}/"
            "summary.json"
        )
        payload = {
            "run_id": self.run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            **dict(summary),
        }
        self.storage_client.bucket(self.bucket_name).blob(
            object_path
        ).upload_from_string(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            if_generation_match=0,
        )
