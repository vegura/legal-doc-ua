from __future__ import annotations

from collections.abc import Iterator, Sequence

from google.cloud import bigquery

from .cloud import SourceDocument, source_object_path
from .settings import DocumentTextParsingSettings


def iter_unparsed_documents(
    bigquery_client: bigquery.Client,
    settings: DocumentTextParsingSettings,
) -> Iterator[SourceDocument]:
    limit_clause = (
        ""
        if settings.limit is None
        else "\nLIMIT @document_limit"
    )
    query = f"""
        SELECT DISTINCT
            CAST(doc_id AS INT64) AS doc_id,
            CAST(justice_kind AS INT64) AS justice_kind
        FROM `{settings.bigquery_table}`
        WHERE doc_id IS NOT NULL
          AND justice_kind IN UNNEST(@justice_kinds)
          AND is_parsed = FALSE
        ORDER BY justice_kind, doc_id
        {limit_clause}
    """
    query_parameters: list[
        bigquery.ArrayQueryParameter
        | bigquery.ScalarQueryParameter
    ] = [
        bigquery.ArrayQueryParameter(
            "justice_kinds",
            "INT64",
            list(settings.justice_kinds),
        )
    ]
    if settings.limit is not None:
        query_parameters.append(
            bigquery.ScalarQueryParameter(
                "document_limit",
                "INT64",
                settings.limit,
            )
        )
    job_config = bigquery.QueryJobConfig(
        query_parameters=query_parameters
    )
    rows = bigquery_client.query(
        query,
        job_config=job_config,
    ).result(page_size=settings.batch_size)
    for row in rows:
        doc_id = int(row.doc_id)
        justice_kind = int(row.justice_kind)
        yield SourceDocument(
            justice_kind=justice_kind,
            document_id=str(doc_id),
            object_name=source_object_path(
                settings.normalized_source_prefix,
                justice_kind,
                str(doc_id),
            ),
        )


def mark_documents_parsed(
    bigquery_client: bigquery.Client,
    settings: DocumentTextParsingSettings,
    documents: Sequence[SourceDocument],
) -> int:
    if not documents:
        return 0

    query = f"""
        UPDATE `{settings.bigquery_table}` AS target
        SET is_parsed = TRUE
        WHERE EXISTS (
            SELECT 1
            FROM UNNEST(@documents) AS completed
            WHERE target.doc_id = completed.doc_id
              AND target.justice_kind = completed.justice_kind
        )
    """
    document_parameters = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter(
                "doc_id",
                "INT64",
                int(document.document_id),
            ),
            bigquery.ScalarQueryParameter(
                "justice_kind",
                "INT64",
                document.justice_kind,
            ),
        )
        for document in documents
    ]
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "documents",
                "STRUCT",
                document_parameters,
            )
        ]
    )
    query_job = bigquery_client.query(
        query,
        job_config=job_config,
    )
    query_job.result()
    return int(
        getattr(query_job, "num_dml_affected_rows", 0) or 0
    )
