from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from ftfy import fix_text
from google.cloud import bigquery, storage
from striprtf.striprtf import rtf_to_text


MODEL_NAME = "BAAI/bge-small-en-v1.5"
CRIMINAL_JUSTICE_KIND = 2
OTHER_LEGAL_KINDS = (1, 3, 4, 5)
DEFAULT_DATASET_PREFIX = "legal-bge-pretraining/v1"

_TABLE_ID_RE = re.compile(
    r"^[A-Za-z0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

SOURCE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("justice_kind", pa.int8()),
        ("source_object", pa.string()),
    ]
)

CLEAN_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("justice_kind", pa.int8()),
        ("source_object", pa.string()),
        ("text", pa.large_string()),
        ("text_hash", pa.string()),
        ("char_count", pa.int64()),
    ]
)

ERROR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("justice_kind", pa.int8()),
        ("source_object", pa.string()),
        ("error", pa.string()),
    ]
)


@dataclass(frozen=True)
class DataPreparationConfig:
    project_id: str
    table_id: str
    source_bucket: str
    dataset_bucket: str
    checkpoint_bucket: str
    dataset_prefix: str = DEFAULT_DATASET_PREFIX
    noncriminal_fraction: float = 0.10
    validation_fraction: float = 0.01
    shard_rows: int = 1_000
    download_workers: int = 16
    min_characters: int = 200
    min_cyrillic_ratio: float = 0.20
    seed: int = 42
    limit: int | None = None

    def validate(self) -> None:
        validate_distinct_buckets(
            self.source_bucket, self.dataset_bucket, self.checkpoint_bucket
        )
        if not _TABLE_ID_RE.fullmatch(self.table_id):
            raise ValueError(
                "table_id must be a fully qualified project.dataset.table identifier"
            )
        if not 0 < self.noncriminal_fraction < 1:
            raise ValueError("noncriminal_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.shard_rows <= 0:
            raise ValueError("shard_rows must be positive")
        if self.download_workers <= 0:
            raise ValueError("download_workers must be positive")
        if self.min_characters <= 0:
            raise ValueError("min_characters must be positive")
        if not 0 <= self.min_cyrillic_ratio <= 1:
            raise ValueError("min_cyrillic_ratio must be between 0 and 1")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")

    @property
    def normalized_prefix(self) -> str:
        return self.dataset_prefix.strip("/")

    def dataset_identity(self) -> dict[str, Any]:
        """Fields that must remain fixed for every resumable dataset prefix."""
        return {
            "project_id": self.project_id,
            "table_id": self.table_id,
            "source_bucket": self.source_bucket,
            "noncriminal_fraction": self.noncriminal_fraction,
            "validation_fraction": self.validation_fraction,
            "shard_rows": self.shard_rows,
            "min_characters": self.min_characters,
            "min_cyrillic_ratio": self.min_cyrillic_ratio,
            "seed": self.seed,
            "limit": self.limit,
        }


def validate_distinct_buckets(
    source_bucket: str, dataset_bucket: str, checkpoint_bucket: str
) -> None:
    names = [name.strip() for name in (source_bucket, dataset_bucket, checkpoint_bucket)]
    if any(not name for name in names):
        raise ValueError("source, dataset, and checkpoint bucket names are required")
    if len(set(names)) != 3:
        raise ValueError(
            "source, cleaned-dataset, and model-checkpoint buckets must be different"
        )


def stable_hash_int(value: str, seed: int = 42) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def assign_split(
    doc_id: str, validation_fraction: float = 0.01, seed: int = 42
) -> str:
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    threshold = int(validation_fraction * 10_000)
    return "validation" if stable_hash_int(doc_id, seed) % 10_000 < threshold else "train"


def noncriminal_documents_per_kind(
    criminal_count: int,
    noncriminal_fraction: float = 0.10,
    other_kind_count: int = len(OTHER_LEGAL_KINDS),
) -> int:
    """Return the equal per-kind sample needed for the requested final mixture."""
    if criminal_count < 0:
        raise ValueError("criminal_count cannot be negative")
    if not 0 <= noncriminal_fraction < 1:
        raise ValueError("noncriminal_fraction must be in [0, 1)")
    if other_kind_count <= 0:
        raise ValueError("other_kind_count must be positive")
    if criminal_count == 0 or noncriminal_fraction == 0:
        return 0
    total_other = criminal_count * noncriminal_fraction / (1 - noncriminal_fraction)
    return math.ceil(total_other / other_kind_count)


def select_lowest_ranked(
    items: Iterable[tuple[str, str]], count: int, seed: int
) -> list[tuple[str, str]]:
    """Deterministically select ``count`` (doc_id, object_name) pairs with lowest hashes."""
    if count <= 0:
        return []
    heap: list[tuple[int, str, str]] = []
    for doc_id, object_name in items:
        rank = stable_hash_int(doc_id, seed)
        candidate = (-rank, doc_id, object_name)
        if len(heap) < count:
            heapq.heappush(heap, candidate)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, candidate)
    selected = [(doc_id, object_name) for _, doc_id, object_name in heap]
    return sorted(selected, key=lambda item: stable_hash_int(item[0], seed))


def normalize_legal_text(text: str) -> str:
    text = fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = _CONTROL_RE.sub("", text)
    text = "\n".join(_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum("CYRILLIC" in unicodedata.name(char, "") for char in letters)
    return cyrillic / len(letters)


def clean_rtf_bytes(
    payload: bytes,
    *,
    min_characters: int = 200,
    min_cyrillic_ratio: float = 0.20,
) -> str:
    if not payload:
        raise ValueError("empty RTF payload")
    rtf_source = payload.decode("latin-1")
    plain_text = rtf_to_text(rtf_source, encoding="cp1251", errors="replace")
    cleaned = normalize_legal_text(plain_text)
    if len(cleaned) < min_characters:
        raise ValueError(
            f"cleaned document is too short ({len(cleaned)} < {min_characters})"
        )
    ratio = cyrillic_ratio(cleaned)
    if ratio < min_cyrillic_ratio:
        raise ValueError(
            f"cleaned document has too little Cyrillic text ({ratio:.3f})"
        )
    return cleaned


def _write_rows(
    writer: pq.ParquetWriter, rows: Sequence[dict[str, Any]], schema: pa.Schema
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(list(rows), schema=schema))


def _criminal_count(config: DataPreparationConfig, client: bigquery.Client) -> int:
    query = (
        "SELECT COUNT(DISTINCT doc_id) AS document_count "
        f"FROM `{config.table_id}` "
        "WHERE justice_kind = @justice_kind AND doc_id IS NOT NULL"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "justice_kind", "INT64", CRIMINAL_JUSTICE_KIND
            )
        ]
    )
    row = next(iter(client.query(query, job_config=job_config).result()))
    count = int(row.document_count)
    return min(count, config.limit) if config.limit is not None else count


def _iter_criminal_documents(
    config: DataPreparationConfig, client: bigquery.Client
) -> Iterator[dict[str, Any]]:
    limit_clause = f" LIMIT {config.limit}" if config.limit is not None else ""
    query = (
        "SELECT DISTINCT CAST(doc_id AS STRING) AS doc_id "
        f"FROM `{config.table_id}` "
        "WHERE justice_kind = @justice_kind AND doc_id IS NOT NULL "
        f"ORDER BY doc_id{limit_clause}"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "justice_kind", "INT64", CRIMINAL_JUSTICE_KIND
            )
        ]
    )
    for row in client.query(query, job_config=job_config).result(page_size=10_000):
        doc_id = str(row.doc_id)
        yield {
            "doc_id": doc_id,
            "justice_kind": CRIMINAL_JUSTICE_KIND,
            "source_object": f"{CRIMINAL_JUSTICE_KIND}/{doc_id}.rtf",
        }


def _iter_kind_objects(
    client: storage.Client, bucket_name: str, justice_kind: int
) -> Iterator[tuple[str, str]]:
    prefix = f"{justice_kind}/"
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if not blob.name.lower().endswith(".rtf"):
            continue
        yield Path(blob.name).stem, blob.name


def _upload_file(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    local_path: Path,
    *,
    content_type: str = "application/octet-stream",
) -> None:
    blob = client.bucket(bucket_name).blob(object_name)
    blob.upload_from_filename(
        str(local_path), content_type=content_type, checksum="auto"
    )


def _upload_json(
    client: storage.Client, bucket_name: str, object_name: str, value: Any
) -> None:
    client.bucket(bucket_name).blob(object_name).upload_from_string(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        content_type="application/json",
        checksum="auto",
    )


def build_source_manifest(
    config: DataPreparationConfig,
    work_dir: Path,
    *,
    bigquery_client: bigquery.Client | None = None,
    storage_client: storage.Client | None = None,
) -> tuple[Path, dict[int, int]]:
    config.validate()
    work_dir.mkdir(parents=True, exist_ok=True)
    bq_client = bigquery_client or bigquery.Client(project=config.project_id)
    gcs_client = storage_client or storage.Client(project=config.project_id)
    manifest_path = work_dir / "source.parquet"

    criminal_count = _criminal_count(config, bq_client)
    other_per_kind = noncriminal_documents_per_kind(
        criminal_count, config.noncriminal_fraction
    )
    counts: dict[int, int] = {CRIMINAL_JUSTICE_KIND: 0}

    with pq.ParquetWriter(manifest_path, SOURCE_SCHEMA, compression="zstd") as writer:
        batch: list[dict[str, Any]] = []
        for row in _iter_criminal_documents(config, bq_client):
            batch.append(row)
            counts[CRIMINAL_JUSTICE_KIND] += 1
            if len(batch) >= 10_000:
                _write_rows(writer, batch, SOURCE_SCHEMA)
                batch.clear()
        _write_rows(writer, batch, SOURCE_SCHEMA)

        for justice_kind in OTHER_LEGAL_KINDS:
            selected = select_lowest_ranked(
                _iter_kind_objects(gcs_client, config.source_bucket, justice_kind),
                other_per_kind,
                config.seed + justice_kind,
            )
            rows = [
                {
                    "doc_id": doc_id,
                    "justice_kind": justice_kind,
                    "source_object": object_name,
                }
                for doc_id, object_name in selected
            ]
            counts[justice_kind] = len(rows)
            for start in range(0, len(rows), 10_000):
                _write_rows(writer, rows[start : start + 10_000], SOURCE_SCHEMA)

    prefix = config.normalized_prefix
    _upload_file(
        gcs_client,
        config.dataset_bucket,
        f"{prefix}/manifests/source.parquet",
        manifest_path,
        content_type="application/vnd.apache.parquet",
    )
    _upload_json(
        gcs_client,
        config.dataset_bucket,
        f"{prefix}/manifests/source.json",
        {
            "model_specification": MODEL_NAME,
            "counts_by_justice_kind": counts,
            "configuration": asdict(config),
            "dataset_identity": config.dataset_identity(),
        },
    )
    gcs_client.bucket(config.dataset_bucket).blob(
        f"{prefix}/manifests/source._SUCCESS"
    ).upload_from_string("ok\n", content_type="text/plain")
    return manifest_path, counts


def get_or_build_source_manifest(
    config: DataPreparationConfig,
    work_dir: Path,
    *,
    bigquery_client: bigquery.Client | None = None,
    storage_client: storage.Client | None = None,
) -> Path:
    config.validate()
    gcs_client = storage_client or storage.Client(project=config.project_id)
    prefix = config.normalized_prefix
    marker = gcs_client.bucket(config.dataset_bucket).blob(
        f"{prefix}/manifests/source._SUCCESS"
    )
    local_manifest = work_dir / "source.parquet"
    if marker.exists():
        metadata_text = gcs_client.bucket(config.dataset_bucket).blob(
            f"{prefix}/manifests/source.json"
        ).download_as_text(checksum="auto")
        metadata = json.loads(metadata_text)
        if metadata.get("dataset_identity") != config.dataset_identity():
            raise ValueError(
                "The existing dataset prefix was created with different options; "
                "use a new --dataset-prefix instead of mixing incompatible shards"
            )
        gcs_client.bucket(config.dataset_bucket).blob(
            f"{prefix}/manifests/source.parquet"
        ).download_to_filename(str(local_manifest), checksum="auto")
        return local_manifest
    manifest, _ = build_source_manifest(
        config,
        work_dir,
        bigquery_client=bigquery_client,
        storage_client=gcs_client,
    )
    return manifest


def _clean_source_row(
    row: dict[str, Any],
    config: DataPreparationConfig,
    client: storage.Client,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = (
            client.bucket(config.source_bucket)
            .blob(str(row["source_object"]))
            .download_as_bytes(checksum="auto")
        )
        text = clean_rtf_bytes(
            payload,
            min_characters=config.min_characters,
            min_cyrillic_ratio=config.min_cyrillic_ratio,
        )
        clean_row = {
            **row,
            "text": text,
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "char_count": len(text),
        }
        return clean_row, None
    except Exception as exc:  # document-level failures must not abort a shard
        error_row = {**row, "error": f"{type(exc).__name__}: {exc}"}
        return None, error_row


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]], schema: pa.Schema) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema),
        path,
        compression="zstd",
    )


def clean_and_shard_manifest(
    config: DataPreparationConfig,
    manifest_path: Path,
    work_dir: Path,
    *,
    storage_client: storage.Client | None = None,
) -> dict[str, int]:
    config.validate()
    gcs_client = storage_client or storage.Client(project=config.project_id)
    prefix = config.normalized_prefix
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = {"source": 0, "clean": 0, "errors": 0, "train": 0, "validation": 0}

    parquet_file = pq.ParquetFile(manifest_path)
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        for shard_index, record_batch in enumerate(
            parquet_file.iter_batches(batch_size=config.shard_rows)
        ):
            marker_name = f"{prefix}/shards/part-{shard_index:06d}._SUCCESS"
            if gcs_client.bucket(config.dataset_bucket).blob(marker_name).exists():
                continue

            source_rows = record_batch.to_pylist()
            stats["source"] += len(source_rows)
            results = list(
                executor.map(
                    lambda row: _clean_source_row(row, config, gcs_client),
                    source_rows,
                )
            )
            clean_rows = [clean for clean, _ in results if clean is not None]
            error_rows = [error for _, error in results if error is not None]
            train_rows: list[dict[str, Any]] = []
            validation_rows: list[dict[str, Any]] = []
            for row in clean_rows:
                split = assign_split(
                    str(row["doc_id"]), config.validation_fraction, config.seed
                )
                (validation_rows if split == "validation" else train_rows).append(row)

            local_train = work_dir / f"train-{shard_index:06d}.parquet"
            local_validation = work_dir / f"validation-{shard_index:06d}.parquet"
            local_errors = work_dir / f"errors-{shard_index:06d}.parquet"
            _write_parquet(local_train, train_rows, CLEAN_SCHEMA)
            _write_parquet(local_validation, validation_rows, CLEAN_SCHEMA)
            _write_parquet(local_errors, error_rows, ERROR_SCHEMA)

            _upload_file(
                gcs_client,
                config.dataset_bucket,
                f"{prefix}/clean/train/part-{shard_index:06d}.parquet",
                local_train,
                content_type="application/vnd.apache.parquet",
            )
            _upload_file(
                gcs_client,
                config.dataset_bucket,
                f"{prefix}/clean/validation/part-{shard_index:06d}.parquet",
                local_validation,
                content_type="application/vnd.apache.parquet",
            )
            _upload_file(
                gcs_client,
                config.dataset_bucket,
                f"{prefix}/errors/part-{shard_index:06d}.parquet",
                local_errors,
                content_type="application/vnd.apache.parquet",
            )
            gcs_client.bucket(config.dataset_bucket).blob(marker_name).upload_from_string(
                "ok\n", content_type="text/plain", if_generation_match=0
            )

            stats["clean"] += len(clean_rows)
            stats["errors"] += len(error_rows)
            stats["train"] += len(train_rows)
            stats["validation"] += len(validation_rows)
            local_train.unlink(missing_ok=True)
            local_validation.unlink(missing_ok=True)
            local_errors.unlink(missing_ok=True)

    _upload_json(
        gcs_client,
        config.dataset_bucket,
        f"{prefix}/dataset_manifest.json",
        {"configuration": asdict(config), "completed_run_stats": stats},
    )
    return stats


def prepare_dataset(config: DataPreparationConfig) -> dict[str, int]:
    config.validate()
    with tempfile.TemporaryDirectory(prefix="legal-pretraining-data-") as temp_dir:
        work_dir = Path(temp_dir)
        manifest_path = get_or_build_source_manifest(config, work_dir)
        return clean_and_shard_manifest(config, manifest_path, work_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and shard Ukrainian legal RTF documents for MLM pre-training."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--source-bucket", required=True)
    parser.add_argument("--dataset-bucket", required=True)
    parser.add_argument("--checkpoint-bucket", required=True)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--noncriminal-fraction", type=float, default=0.10)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--shard-rows", type=int, default=1_000)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--min-cyrillic-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = DataPreparationConfig(
        project_id=args.project_id,
        table_id=args.table_id,
        source_bucket=args.source_bucket,
        dataset_bucket=args.dataset_bucket,
        checkpoint_bucket=args.checkpoint_bucket,
        dataset_prefix=args.dataset_prefix,
        noncriminal_fraction=args.noncriminal_fraction,
        validation_fraction=args.validation_fraction,
        shard_rows=args.shard_rows,
        download_workers=args.download_workers,
        min_characters=args.min_characters,
        min_cyrillic_ratio=args.min_cyrillic_ratio,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(prepare_dataset(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
