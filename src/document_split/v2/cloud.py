from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from huggingface_hub import model_info

from ..cloud import canonical_schema_text
from ..config import ExtractionSettings, StorageSettings
from .settings import V2ChainSettings, V2PartProcessingPrompts


def v2_document_prefix(
    version_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return (
        f"{version_prefix.strip('/')}/{int(justice_kind)}/"
        f"{document_id}"
    )


def v2_result_object_path(
    version_prefix: str,
    justice_kind: int,
    document_id: str,
) -> str:
    return (
        f"{v2_document_prefix(version_prefix, justice_kind, document_id)}"
        "/result.parquet"
    )


def create_v2_document_folder(
    storage_client: storage.Client,
    bucket_name: str,
    version_prefix: str,
    justice_kind: int,
    document_id: str,
    source_object: str,
) -> str:
    prefix = v2_document_prefix(
        version_prefix,
        justice_kind,
        document_id,
    )
    marker_path = f"{prefix}/_document.json"
    marker = {
        "document_id": str(document_id),
        "justice_kind": int(justice_kind),
        "source_object": source_object,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    blob = storage_client.bucket(bucket_name).blob(marker_path)
    try:
        blob.upload_from_string(
            json.dumps(
                marker,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            if_generation_match=0,
        )
    except PreconditionFailed:
        pass
    return prefix


def upload_v2_artifacts(
    storage_client: storage.Client,
    bucket_name: str,
    document_prefix: str,
    work_dir: Path,
) -> None:
    for path in sorted(work_dir.rglob("*")):
        if not path.is_file() or path.name == "result.parquet":
            continue
        relative = path.relative_to(work_dir).as_posix()
        content_type = (
            "application/vnd.apache.parquet"
            if path.suffix == ".parquet"
            else "application/json; charset=utf-8"
        )
        storage_client.bucket(bucket_name).blob(
            f"{document_prefix}/{relative}"
        ).upload_from_filename(
            str(path),
            content_type=content_type,
        )


def list_completed_v2_document_ids(
    storage_client: storage.Client,
    bucket_name: str,
    version_prefix: str,
    justice_kind: int,
) -> set[str]:
    prefix = f"{version_prefix.strip('/')}/{int(justice_kind)}/"
    completed: set[str] = set()
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        relative = blob.name[len(prefix) :]
        parts = relative.split("/")
        if len(parts) == 2 and parts[1] == "result.parquet":
            completed.add(parts[0])
    return completed


def build_v2_manifest_identity(
    extraction_settings: ExtractionSettings,
    storage_settings: StorageSettings,
    chain_settings: V2ChainSettings,
    part_prompts: V2PartProcessingPrompts,
    model_revision: str,
) -> dict[str, Any]:
    schema_text = canonical_schema_text(
        extraction_settings.output_schema
    )
    prompts = {
        "introductory_part": part_prompts.introductory,
        "court_reasoning_part": part_prompts.reasoning,
        "result_part": part_prompts.operative,
    }
    prompts_text = json.dumps(
        prompts,
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "pipeline": "document_split.v2",
        "info_version": storage_settings.version_prefix,
        "base_prompt_sha256": _sha256_text(
            extraction_settings.prompt
        ),
        "handler_prompts": prompts,
        "handler_prompts_sha256": _sha256_text(prompts_text),
        "arrow_schema": schema_text,
        "arrow_schema_sha256": _sha256_text(schema_text),
        "model": {
            "id": extraction_settings.model_id,
            "revision": model_revision,
            "dtype": "bfloat16",
        },
        "batching": {
            "target_batch_tokens": chain_settings.target_batch_tokens,
            "overlap_tokens": chain_settings.overlap_tokens,
            "part_processing_mode": chain_settings.part_processing_mode,
            "processing_strategy": "grounded_map_reduce",
            "max_new_tokens": extraction_settings.max_new_tokens,
            "temperature": extraction_settings.temperature,
            "do_sample": extraction_settings.temperature > 0,
        },
        "parts_input": {
            "bucket": storage_settings.parts_bucket,
            "prefix": storage_settings.parts_prefix.strip("/"),
            "object_path": (
                "{prefix}/{justice_kind}/{document_id}/"
                "classification.parquet"
            ),
            "bigquery_filter": "is_classified = TRUE",
        },
        "artifacts": {
            "document_folder": True,
            "parts_source": "source_parts.parquet",
            "final": "result.parquet",
            "parquet_row_unit": "document",
        },
    }


def prepare_v2_manifest(
    storage_client: storage.Client,
    extraction_settings: ExtractionSettings,
    storage_settings: StorageSettings,
    chain_settings: V2ChainSettings,
    part_prompts: V2PartProcessingPrompts,
    hf_token: str | None,
) -> tuple[str, dict[str, Any]]:
    bucket = storage_client.bucket(storage_settings.destination_bucket)
    manifest_path = f"{storage_settings.version_prefix}/manifest.json"
    blob = bucket.blob(manifest_path)

    if blob.exists(client=storage_client):
        manifest = json.loads(blob.download_as_text())
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise RuntimeError(f"Malformed V2 manifest: {manifest_path}")
        revision = identity.get("model", {}).get("revision")
        if not revision:
            raise RuntimeError("V2 manifest has no model revision")
        expected = build_v2_manifest_identity(
            extraction_settings,
            storage_settings,
            chain_settings,
            part_prompts,
            revision,
        )
        if identity != expected:
            raise RuntimeError(
                f"V2 settings differ from {manifest_path}; "
                "choose a new info version"
            )
        return str(revision), manifest

    revision = extraction_settings.model_revision
    if revision is None:
        revision = model_info(
            extraction_settings.model_id,
            token=hf_token,
        ).sha
    if not revision:
        raise RuntimeError("Could not resolve the V2 model revision")
    identity = build_v2_manifest_identity(
        extraction_settings,
        storage_settings,
        chain_settings,
        part_prompts,
        revision,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
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
        return prepare_v2_manifest(
            storage_client,
            extraction_settings,
            storage_settings,
            chain_settings,
            part_prompts,
            hf_token,
        )
    return str(revision), manifest


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
