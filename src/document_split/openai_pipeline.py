from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from google.api_core.exceptions import PreconditionFailed
from google.cloud import bigquery, storage
from tqdm.auto import tqdm

from .cloud import (
    RunLogger,
    canonical_schema_text,
    destination_object_path,
    download_source_rtf,
    iter_document_ids,
    list_completed_document_ids,
    source_object_path,
    upload_parquet_atomically,
)
from .config import (
    BIGQUERY_TABLE,
    DESTINATION_BUCKET,
    PROJECT_ID,
    SOURCE_BUCKET,
)
from .openai_runtime import (
    DEFAULT_OPENAI_EXTRACTION_SETTINGS,
    OpenAIExtractionSettings,
    create_openai_client,
    parse_document_to_parquet_openai,
)


OPENAI_INFO_VERSION = "openai_info_version_1"


@dataclass(frozen=True)
class OpenAIStorageSettings:
    project_id: str = PROJECT_ID
    bigquery_table: str = BIGQUERY_TABLE
    source_bucket: str = SOURCE_BUCKET
    destination_bucket: str = DESTINATION_BUCKET
    info_version: str = OPENAI_INFO_VERSION
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
            r"openai_info_version_[1-9][0-9]*",
            self.version_prefix,
        ):
            raise ValueError(
                "OpenAI info_version must look like openai_info_version_1"
            )
        if not self.justice_kinds:
            raise ValueError("At least one justice kind is required")
        if self.bigquery_page_size <= 0:
            raise ValueError("bigquery_page_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")


DEFAULT_OPENAI_STORAGE_SETTINGS = OpenAIStorageSettings()


@dataclass(frozen=True)
class OpenAIClients:
    bigquery: bigquery.Client
    storage: storage.Client
    openai: Any


def load_openai_colab_clients(
    storage_settings: OpenAIStorageSettings,
    extraction_settings: OpenAIExtractionSettings,
) -> OpenAIClients:
    """Load Colab secrets without importing Torch or Transformers."""

    from google.colab import userdata
    from google.oauth2 import service_account

    raw_cloud_secret = userdata.get("cloud_access")
    if not raw_cloud_secret:
        raise RuntimeError("Missing Colab secret: cloud_access")
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(raw_cloud_secret)
    )

    api_key = userdata.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Colab secret: OPENAI_API_KEY")

    return OpenAIClients(
        bigquery=bigquery.Client(
            project=storage_settings.project_id,
            credentials=credentials,
        ),
        storage=storage.Client(
            project=storage_settings.project_id,
            credentials=credentials,
        ),
        openai=create_openai_client(
            api_key,
            timeout_seconds=extraction_settings.timeout_seconds,
        ),
    )


def build_openai_manifest_identity(
    extraction: OpenAIExtractionSettings,
    storage_settings: OpenAIStorageSettings,
) -> dict[str, Any]:
    schema_text = canonical_schema_text(extraction.output_schema)
    return {
        "info_version": storage_settings.version_prefix,
        "prompt": extraction.prompt,
        "prompt_sha256": _sha256_text(extraction.prompt),
        "arrow_schema": schema_text,
        "arrow_schema_sha256": _sha256_text(schema_text),
        "provider": "openai",
        "model": {"id": extraction.model},
        "document_processing": {
            "model_input_unit": "complete_document",
            "model_response_unit": "document",
            "parquet_row_unit": "document",
            "paragraph_ids_start_at": 1,
        },
        "generation": {
            "api": "responses",
            "response_format": "json_object",
            "max_output_tokens": extraction.max_output_tokens,
            "store": extraction.store_response,
            "attempts": 1,
        },
        "parquet": {"compression": extraction.parquet_compression},
    }


def prepare_openai_version_manifest(
    storage_client: storage.Client,
    extraction: OpenAIExtractionSettings,
    storage_settings: OpenAIStorageSettings,
) -> dict[str, Any]:
    bucket = storage_client.bucket(storage_settings.destination_bucket)
    manifest_path = f"{storage_settings.version_prefix}/manifest.json"
    blob = bucket.blob(manifest_path)
    identity = build_openai_manifest_identity(
        extraction,
        storage_settings,
    )

    if blob.exists(client=storage_client):
        manifest = json.loads(blob.download_as_text())
        if manifest.get("identity") != identity:
            raise RuntimeError(
                f"Research settings differ from {manifest_path}; "
                "choose a new OpenAI info version"
            )
        return manifest

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
        return prepare_openai_version_manifest(
            storage_client,
            extraction,
            storage_settings,
        )
    return manifest


def run_openai_pipeline(
    extraction: OpenAIExtractionSettings = (
        DEFAULT_OPENAI_EXTRACTION_SETTINGS
    ),
    storage_settings: OpenAIStorageSettings = (
        DEFAULT_OPENAI_STORAGE_SETTINGS
    ),
) -> dict[str, int]:
    """Run the independent OpenAI Responses API extraction pipeline."""

    extraction.validate()
    storage_settings.validate()
    clients = load_openai_colab_clients(storage_settings, extraction)
    manifest = prepare_openai_version_manifest(
        clients.storage,
        extraction,
        storage_settings,
    )
    print(
        f"Using OpenAI {extraction.model} for "
        f"{manifest['identity']['info_version']}"
    )

    logger = RunLogger(
        storage_client=clients.storage,
        bucket_name=storage_settings.destination_bucket,
        version_prefix=storage_settings.version_prefix,
        flush_size=storage_settings.run_log_flush_size,
    )
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "paragraphs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    remaining = storage_settings.limit

    try:
        for justice_kind in storage_settings.justice_kinds:
            if remaining is not None and remaining <= 0:
                break

            completed = (
                list_completed_document_ids(
                    clients.storage,
                    storage_settings.destination_bucket,
                    storage_settings.version_prefix,
                    justice_kind,
                )
                if storage_settings.skip_existing
                else set()
            )
            document_ids = iter_document_ids(
                clients.bigquery,
                storage_settings.bigquery_table,
                justice_kind,
                storage_settings.bigquery_page_size,
                remaining,
            )
            for document_id in tqdm(
                document_ids,
                desc=f"openai justice_kind={justice_kind}",
                unit="doc",
            ):
                if remaining is not None and remaining <= 0:
                    break
                if document_id in completed:
                    counters["skipped_existing"] += 1
                    if remaining is not None:
                        remaining -= 1
                    continue

                output_path = destination_object_path(
                    storage_settings.version_prefix,
                    justice_kind,
                    document_id,
                )
                try:
                    raw_rtf = download_source_rtf(
                        clients.storage,
                        storage_settings.source_bucket,
                        justice_kind,
                        document_id,
                    )
                    result = parse_document_to_parquet_openai(
                        document_id,
                        raw_rtf,
                        clients.openai,
                        extraction,
                    )
                    created = upload_parquet_atomically(
                        clients.storage,
                        storage_settings.destination_bucket,
                        output_path,
                        result.parquet_bytes,
                    )
                    if created:
                        counters["processed"] += 1
                        counters["paragraphs"] += result.paragraph_count
                        counters["input_tokens"] += result.input_tokens
                        counters["output_tokens"] += result.output_tokens
                    else:
                        counters["skipped_existing"] += 1
                except Exception as exc:
                    counters["failed"] += 1
                    logger.record(
                        {
                            "status": "failed",
                            "document_id": document_id,
                            "justice_kind": justice_kind,
                            "source_object": source_object_path(
                                justice_kind,
                                document_id,
                            ),
                            "destination_object": output_path,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                finally:
                    if remaining is not None:
                        remaining -= 1
    finally:
        logger.close(counters)
    return counters


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
