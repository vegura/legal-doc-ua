from __future__ import annotations

import argparse
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from google.cloud import storage


def _safe_filename_from_url(url: str, fallback_stem: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if name:
        return name
    return f"{fallback_stem}.bin"


def download_document(url: str, temp_dir: Path, fallback_stem: str) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename_from_url(url, fallback_stem=fallback_stem)
    local_path = temp_dir / filename

    with urllib.request.urlopen(url) as response, local_path.open("wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    return local_path


def upload_document_to_gcs(
    local_path: Path,
    bucket_name: str,
    destination_blob: str,
    client: storage.Client | None = None,
) -> str:
    gcs_client = client or storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination_blob}"


def _iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for row in df.itertuples(index=False):
        yield row._asdict()


def upload_dataset_documents_to_gcs(
    dataset: pd.DataFrame,
    bucket_name: str,
    temp_dir: Path | None = None,
    url_column: str = "doc_url",
    id_column: str = "doc_id",
    prefix: str = "",
    limit: int | None = None,
    catalog_column: str | None = None,
    filename_id_column: str | None = None,
    on_batch_complete: Callable[[pd.DataFrame], None] | None = None,
) -> None:
    if limit is not None:
        dataset = dataset.iloc[:limit]

    prefix = prefix.strip("/")
    if prefix:
        prefix = f"{prefix}/"

    if temp_dir is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _run_pipeline_rows(
                dataset,
                Path(tmp_dir),
                bucket_name,
                url_column=url_column,
                id_column=id_column,
                prefix=prefix,
                catalog_column=catalog_column,
                filename_id_column=filename_id_column,
                limit=limit,
            )
    else:
        _run_pipeline_rows(
            dataset,
            temp_dir,
            bucket_name,
            url_column=url_column,
            id_column=id_column,
            prefix=prefix,
            catalog_column=catalog_column,
            filename_id_column=filename_id_column,
            limit=limit,
        )

    if on_batch_complete is not None:
        on_batch_complete(dataset)


def _run_pipeline_rows(
    df: pd.DataFrame,
    temp_dir: Path,
    bucket_name: str,
    url_column: str,
    id_column: str,
    prefix: str,
    catalog_column: str | None,
    filename_id_column: str | None,
    limit: int | None,
) -> None:
    gcs_client = storage.Client()
    processed = 0
    skipped = 0

    for row in _iter_rows(df):
        if limit is not None and processed >= limit:
            break

        url = row.get(url_column)
        doc_id = row.get(id_column)
        catalog_value = row.get(catalog_column) if catalog_column else None
        filename_id = row.get(filename_id_column) if filename_id_column else None

        if not url or not doc_id:
            skipped += 1
            continue

        try:
            local_path = download_document(str(url), temp_dir, fallback_stem=str(doc_id))
            catalog_prefix = ""
            if catalog_value is not None and str(catalog_value).strip():
                safe_catalog = str(catalog_value).strip().replace("/", "_").replace("\\", "_")
                catalog_prefix = f"{safe_catalog}/"
            if filename_id is not None and str(filename_id).strip():
                suffix = local_path.suffix
                blob_name = f"{str(filename_id).strip()}{suffix}"
            else:
                blob_name = local_path.name
            destination_blob = f"{prefix}{catalog_prefix}{blob_name}"
            gcs_path = upload_document_to_gcs(
                local_path=local_path,
                bucket_name=bucket_name,
                destination_blob=destination_blob,
                client=gcs_client,
            )
            processed += 1
            print(f"Uploaded {doc_id} -> {gcs_path}")
        except Exception as exc:
            skipped += 1
            print(f"Failed {doc_id} ({url}): {exc}")

    print(f"Done. Uploaded: {processed}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download documents and upload to GCS.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--url-column", default="doc_url")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--catalog-column", default=None)
    parser.add_argument("--filename-id-column", default=None)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_path, sep=args.sep)
    upload_dataset_documents_to_gcs(
        dataset=dataset,
        bucket_name=args.bucket,
        temp_dir=args.temp_dir,
        url_column=args.url_column,
        id_column=args.id_column,
        prefix=args.prefix,
        limit=args.limit,
        catalog_column=args.catalog_column,
        filename_id_column=args.filename_id_column,
    )


if __name__ == "__main__":
    main()
