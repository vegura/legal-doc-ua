from __future__ import annotations

from google.cloud import storage

from ..document_text_parsing.cloud import document_destination_prefix
from .bigquery import ClassificationSource
from .settings import DocumentClassificationSettings


def classification_source_object_path(
    settings: DocumentClassificationSettings,
    source: ClassificationSource,
) -> str:
    prefix = document_destination_prefix(
        settings.normalized_document_prefix,
        source.justice_kind,
        source.document_id,
    )
    return f"{prefix}/{settings.source_parquet_name}"


def classification_result_object_path(
    settings: DocumentClassificationSettings,
    source: ClassificationSource,
) -> str:
    prefix = document_destination_prefix(
        settings.normalized_document_prefix,
        source.justice_kind,
        source.document_id,
    )
    return f"{prefix}/{settings.result_parquet_name}"


def download_paragraph_parquet(
    storage_client: storage.Client,
    settings: DocumentClassificationSettings,
    source: ClassificationSource,
) -> bytes:
    object_path = classification_source_object_path(settings, source)
    blob = storage_client.bucket(settings.bucket).blob(object_path)
    if not blob.exists(client=storage_client):
        raise FileNotFoundError(f"gs://{settings.bucket}/{object_path}")
    return blob.download_as_bytes()


def upload_classification_parquet(
    storage_client: storage.Client,
    settings: DocumentClassificationSettings,
    source: ClassificationSource,
    parquet_bytes: bytes,
) -> None:
    object_path = classification_result_object_path(settings, source)
    storage_client.bucket(settings.bucket).blob(object_path).upload_from_string(
        parquet_bytes,
        content_type="application/vnd.apache.parquet",
    )
