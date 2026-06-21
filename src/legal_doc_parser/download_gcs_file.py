"""Download a single file from Google Cloud Storage.

Usage:
    python download_gcs_file.py gs://court_data_2024/5/117892250.rtf
    python download_gcs_file.py gs://court_data_2024/5/117892250.rtf -o ./local.rtf
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a gs://bucket/path/to/blob URI into (bucket, blob_name)."""
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Not a valid GCS URI: {uri!r} (expected gs://bucket/path)")
    return parsed.netloc, parsed.path.lstrip("/")


def download_blob(uri: str, destination: Path | None = None) -> Path:
    """Download the GCS object at `uri` to `destination` and return the local path."""
    bucket_name, blob_name = parse_gcs_uri(uri)
    if not blob_name:
        raise ValueError(f"No object path in URI: {uri!r}")

    if destination is None:
        destination = Path(Path(blob_name).name)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(destination))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a file from Google Cloud Storage.")
    parser.add_argument("uri", help="GCS URI, e.g. gs://court_data_2024/5/117892250.rtf")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Local output path (defaults to the object's file name in the current directory).",
    )
    args = parser.parse_args()

    path = download_blob(args.uri, args.output)
    print(f"Downloaded {args.uri} -> {path}")


if __name__ == "__main__":
    main()
