from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

import pyarrow as pa
from google.api_core.exceptions import PreconditionFailed
from google.cloud import bigquery, storage


def canonical_schema_text(schema: pa.Schema) -> str:
    return schema.to_string(
        show_field_metadata=True,
        show_schema_metadata=True,
    )


def iter_part_document_ids(
    bq_client: bigquery.Client,
    table_id: str,
    justice_kind: int,
    page_size: int,
    limit: int | None = None,
) -> Iterator[str]:
    """Yield documents whose upstream paragraph-part file is ready."""

    limit_clause = "" if limit is None else f"LIMIT {int(limit)}"
    query = f"""
        SELECT DISTINCT CAST(doc_id AS STRING) AS document_id
        FROM `{table_id}`
        WHERE doc_id IS NOT NULL
          AND justice_kind = @justice_kind
          AND is_classified = TRUE
        ORDER BY document_id
        {limit_clause}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "justice_kind",
                "INT64",
                int(justice_kind),
            )
        ]
    )
    rows = bq_client.query(query, job_config=job_config).result(
        page_size=page_size
    )
    for row in rows:
        yield str(row.document_id)


def part_document_object_path(
    parts_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return (
        f"{parts_prefix.strip('/')}/{int(justice_kind)}/"
        f"{document_id}/classification.parquet"
    )


def download_part_document_parquet(
    storage_client: storage.Client,
    bucket_name: str,
    parts_prefix: str,
    justice_kind: int,
    document_id: str,
) -> bytes:
    object_path = part_document_object_path(
        parts_prefix,
        justice_kind,
        document_id,
    )
    blob = storage_client.bucket(bucket_name).blob(object_path)
    if not blob.exists(client=storage_client):
        raise FileNotFoundError(f"gs://{bucket_name}/{object_path}")
    return blob.download_as_bytes()


def upload_parquet_atomically(
    storage_client: storage.Client,
    bucket_name: str,
    object_path: str,
    parquet_bytes: bytes,
) -> bool:
    blob = storage_client.bucket(bucket_name).blob(object_path)
    try:
        blob.upload_from_string(
            parquet_bytes,
            content_type="application/vnd.apache.parquet",
            if_generation_match=0,
        )
        return True
    except PreconditionFailed:
        return False


@dataclass
class RunLogger:
    storage_client: storage.Client
    bucket_name: str
    version_prefix: str
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
            f"{self.version_prefix}/_runs/{self.run_id}/"
            f"events-{self.shard_index:06d}.jsonl"
        )
        payload = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
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
            f"{self.version_prefix}/_runs/{self.run_id}/summary.json"
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
