from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

import pyarrow as pa
from google.api_core.exceptions import PreconditionFailed
from google.cloud import bigquery, storage
from huggingface_hub import model_info

from .config import ExtractionSettings, StorageSettings


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_schema_text(schema: pa.Schema) -> str:
    return schema.to_string(
        show_field_metadata=True,
        show_schema_metadata=True,
    )


def build_manifest_identity(
    extraction: ExtractionSettings,
    storage_settings: StorageSettings,
    model_revision: str,
) -> dict[str, Any]:
    schema_text = canonical_schema_text(extraction.output_schema)
    return {
        "info_version": storage_settings.version_prefix,
        "prompt": extraction.prompt,
        "prompt_sha256": _sha256_text(extraction.prompt),
        "arrow_schema": schema_text,
        "arrow_schema_sha256": _sha256_text(schema_text),
        "model": {
            "id": extraction.model_id,
            "revision": model_revision,
            "dtype": "bfloat16",
            "task": "image-text-to-text",
        },
        "document_processing": {
            "model_input_unit": "complete_document",
            "model_response_unit": "document",
            "parquet_row_unit": "document",
            "paragraph_ids_start_at": 1,
            "section_ids_start_at": 0,
            "model_context_tokens": extraction.model_context_tokens,
        },
        "generation": {
            "max_new_tokens": extraction.max_new_tokens,
            "do_sample": False,
            "json_retries": extraction.json_retries,
        },
        "parquet": {"compression": extraction.parquet_compression},
    }


def prepare_version_manifest(
    storage_client: storage.Client,
    extraction: ExtractionSettings,
    storage_settings: StorageSettings,
    hf_token: str | None,
) -> tuple[str, dict[str, Any]]:
    bucket = storage_client.bucket(storage_settings.destination_bucket)
    manifest_path = f"{storage_settings.version_prefix}/manifest.json"
    blob = bucket.blob(manifest_path)

    if blob.exists(client=storage_client):
        manifest = json.loads(blob.download_as_text())
        existing_identity = manifest.get("identity")
        if not isinstance(existing_identity, dict):
            raise RuntimeError(
                f"Malformed research manifest: gs://{bucket.name}/{manifest_path}"
            )
        existing_revision = existing_identity.get("model", {}).get(
            "revision"
        )
        if not existing_revision:
            raise RuntimeError(
                "Existing manifest does not contain a pinned model revision"
            )
        if (
            extraction.model_revision
            and extraction.model_revision != existing_revision
        ):
            raise RuntimeError(
                "Configured model revision differs from the immutable "
                "version manifest"
            )
        expected = build_manifest_identity(
            extraction,
            storage_settings,
            existing_revision,
        )
        if existing_identity != expected:
            raise RuntimeError(
                f"Research settings differ from {manifest_path}; "
                "choose a new info version"
            )
        return existing_revision, manifest

    resolved_revision = extraction.model_revision
    if resolved_revision is None:
        resolved_revision = model_info(
            extraction.model_id,
            token=hf_token,
        ).sha
    if not resolved_revision:
        raise RuntimeError("Could not resolve an immutable model revision")

    identity = build_manifest_identity(
        extraction,
        storage_settings,
        resolved_revision,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
    }
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    try:
        blob.upload_from_string(
            payload,
            content_type="application/json; charset=utf-8",
            if_generation_match=0,
        )
    except PreconditionFailed:
        return prepare_version_manifest(
            storage_client,
            extraction,
            storage_settings,
            hf_token,
        )
    return resolved_revision, manifest


def source_object_path(justice_kind: int, document_id: str) -> str:
    return f"{int(justice_kind)}/{document_id}.rtf"


def destination_object_path(
    version_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return (
        f"{version_prefix.strip('/')}/{int(justice_kind)}/"
        f"{document_id}.parquet"
    )


def iter_document_ids(
    bq_client: bigquery.Client,
    table_id: str,
    justice_kind: int,
    page_size: int,
    limit: int | None = None,
) -> Iterator[str]:
    limit_clause = "" if limit is None else f"LIMIT {int(limit)}"
    query = f"""
        SELECT DISTINCT CAST(doc_id AS STRING) AS document_id
        FROM `{table_id}`
        WHERE doc_id IS NOT NULL AND justice_kind = @justice_kind
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


def list_completed_document_ids(
    storage_client: storage.Client,
    bucket_name: str,
    version_prefix: str,
    justice_kind: int,
) -> set[str]:
    prefix = f"{version_prefix.strip('/')}/{int(justice_kind)}/"
    completed: set[str] = set()
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        relative = blob.name[len(prefix) :]
        if "/" not in relative and relative.endswith(".parquet"):
            completed.add(relative[: -len(".parquet")])
    return completed


def download_source_rtf(
    storage_client: storage.Client,
    bucket_name: str,
    justice_kind: int,
    document_id: str,
) -> bytes:
    object_path = source_object_path(justice_kind, document_id)
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
