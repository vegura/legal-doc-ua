from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from google.cloud import bigquery

from .settings import DocumentClassificationSettings


@dataclass(frozen=True)
class ClassificationSource:
    document_id: str
    justice_kind: int


def iter_documents_for_classification(
    bigquery_client: bigquery.Client,
    settings: DocumentClassificationSettings,
) -> Iterator[ClassificationSource]:
    document_filter = (
        ""
        if settings.document_ids is None
        else "\n          AND doc_id IN UNNEST(@document_ids)"
    )
    limit_clause = (
        "" if settings.limit is None else "\nLIMIT @document_limit"
    )
    query = f"""
        SELECT DISTINCT
            CAST(doc_id AS INT64) AS doc_id,
            CAST(justice_kind AS INT64) AS justice_kind
        FROM `{settings.bigquery_table}`
        WHERE doc_id IS NOT NULL
          AND justice_kind IN UNNEST(@justice_kinds)
          AND is_parsed = TRUE
          AND COALESCE({settings.progress_column}, FALSE) = FALSE
          {document_filter}
        ORDER BY justice_kind, doc_id
        {limit_clause}
    """
    parameters: list[
        bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter
    ] = [
        bigquery.ArrayQueryParameter(
            "justice_kinds", "INT64", list(settings.justice_kinds)
        )
    ]
    if settings.document_ids is not None:
        parameters.append(
            bigquery.ArrayQueryParameter(
                "document_ids", "INT64", list(settings.document_ids)
            )
        )
    if settings.limit is not None:
        parameters.append(
            bigquery.ScalarQueryParameter(
                "document_limit", "INT64", settings.limit
            )
        )
    rows = bigquery_client.query(
        query,
        job_config=bigquery.QueryJobConfig(query_parameters=parameters),
    ).result(page_size=settings.bigquery_page_size)
    for row in rows:
        yield ClassificationSource(
            document_id=str(int(row.doc_id)),
            justice_kind=int(row.justice_kind),
        )


def mark_documents_classified(
    bigquery_client: bigquery.Client,
    settings: DocumentClassificationSettings,
    documents: Sequence[ClassificationSource],
) -> int:
    if not documents:
        return 0
    query = f"""
        UPDATE `{settings.bigquery_table}` AS target
        SET {settings.progress_column} = TRUE
        WHERE EXISTS (
            SELECT 1
            FROM UNNEST(@documents) AS completed
            WHERE target.doc_id = completed.doc_id
              AND target.justice_kind = completed.justice_kind
        )
    """
    values = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter(
                "doc_id", "INT64", int(document.document_id)
            ),
            bigquery.ScalarQueryParameter(
                "justice_kind", "INT64", document.justice_kind
            ),
        )
        for document in documents
    ]
    job = bigquery_client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("documents", "STRUCT", values)
            ]
        ),
    )
    job.result()
    return int(getattr(job, "num_dml_affected_rows", 0) or 0)
