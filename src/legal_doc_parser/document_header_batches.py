from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import pandas as pd

try:
    from .document_header_pipeline import extract_document_headers
except ImportError:  # pragma: no cover - supports notebook-style local imports
    from document_header_pipeline import extract_document_headers


def iter_dataframe_batches(dataset: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield dataset.iloc[start:end].copy()


def extract_document_headers_in_batches(
    dataset: pd.DataFrame,
    batch_size: int,
    url_column: str | None = None,
    id_column: str = "doc_id",
    limit: int | None = None,
    max_workers: int = 1,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
    sleep_seconds: float = 0,
    on_batch_complete: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    if limit is not None:
        dataset = dataset.iloc[:limit]

    batches = list(iter_dataframe_batches(dataset, batch_size))

    def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        return extract_document_headers(
            dataset=batch,
            url_column=url_column,
            id_column=id_column,
            timeout=timeout,
            request_headers=request_headers,
            sleep_seconds=sleep_seconds,
            on_batch_complete=on_batch_complete,
        )

    if max_workers <= 1:
        processed_batches = [_process_batch(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_batches = list(executor.map(_process_batch, batches))

    if not processed_batches:
        return dataset.copy()
    return pd.concat(processed_batches, axis=0)
