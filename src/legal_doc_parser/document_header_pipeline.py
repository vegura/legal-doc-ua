from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Iterable

import pandas as pd

REVIEW_URL_TEMPLATE = "https://reyestr.court.gov.ua/Review/{review_id}"

DOCUMENT_HEADER_COLUMNS = (
    "document_header_case_category",
    "document_header_sent_by_court",
    "document_header_registered",
    "document_header_public_access_provided",
    "document_header_court_proceeding_number",
    "document_header_erdr_criminal_proceeding_number",
)

DOCUMENT_HEADER_METADATA_COLUMNS = (
    "document_header_info_table_text",
    "document_header_info_table_rows_json",
    "document_header_url",
    "document_header_status_code",
    "document_header_error",
)

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass
class InfoTableRow:
    text: str
    bold_values: list[str]
    cells: list[str]


@dataclass
class HeaderTable:
    table_id: str | None
    rows: list[InfoTableRow]


class _HeaderTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HeaderTable] = []
        self._table_stack: list[HeaderTable] = []
        self._in_row = False
        self._in_cell = False
        self._in_bold = False
        self._current_row_parts: list[str] = []
        self._current_cell_parts: list[str] = []
        self._current_row_cells: list[str] = []
        self._current_bold_parts: list[str] = []
        self._current_row_bold_values: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attrs_dict = {name.lower(): value for name, value in attrs}

        if tag == "table":
            self._table_stack.append(
                HeaderTable(table_id=attrs_dict.get("id"), rows=[])
            )
            return

        if not self._table_stack:
            return

        if tag == "tr":
            self._in_row = True
            self._current_row_parts = []
            self._current_row_cells = []
            self._current_row_bold_values = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell_parts = []
        elif tag == "b" and self._in_row:
            self._in_bold = True
            self._current_bold_parts = []

    def handle_endtag(self, tag: str) -> None:
        if not self._table_stack:
            return

        if tag == "b" and self._in_bold:
            value = _normalize_text("".join(self._current_bold_parts))
            if value:
                self._current_row_bold_values.append(value)
            self._in_bold = False
            self._current_bold_parts = []
        elif tag in {"td", "th"} and self._in_cell:
            value = _normalize_text("".join(self._current_cell_parts))
            if value:
                self._current_row_cells.append(value)
            self._in_cell = False
            self._current_cell_parts = []
        elif tag == "tr" and self._in_row:
            row_text = _normalize_text("".join(self._current_row_parts))
            self._table_stack[-1].rows.append(
                InfoTableRow(
                    text=row_text,
                    bold_values=list(self._current_row_bold_values),
                    cells=list(self._current_row_cells),
                )
            )
            self._in_row = False
            self._current_row_parts = []
            self._current_row_cells = []
            self._current_row_bold_values = []
        elif tag == "table":
            table = self._table_stack.pop()
            self.tables.append(table)

    def handle_data(self, data: str) -> None:
        if not self._table_stack or not self._in_row:
            return

        self._current_row_parts.append(data)
        if self._in_cell:
            self._current_cell_parts.append(data)
        if self._in_bold:
            self._current_bold_parts.append(data)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    normalized = html.unescape(str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_date(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    return normalized.rstrip(".").strip()


def _clean_value(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    return normalized or None


def _strip_case_number_from_category(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None

    case_number, separator, category = normalized.partition(":")
    if separator and "/" in case_number and any(char.isdigit() for char in case_number):
        return _clean_value(category)
    return normalized


def _extract_labeled_value(
    text: str,
    label: str,
    stop_labels: Iterable[str] = (),
) -> str | None:
    stop_pattern = "|".join(re.escape(stop_label) for stop_label in stop_labels)
    if stop_pattern:
        pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\s+(?:{stop_pattern})\s*:|$)"
    else:
        pattern = rf"{re.escape(label)}\s*:\s*(.*)$"

    match = re.search(pattern, text)
    if not match:
        return None
    return _clean_value(match.group(1))


def _extract_value_from_row(
    row: InfoTableRow,
    label: str,
    stop_labels: Iterable[str] = (),
) -> str | None:
    for index, cell in enumerate(row.cells):
        if label not in cell:
            continue

        value = _extract_labeled_value(cell, label, stop_labels)
        if value:
            return value

        remainder = re.sub(rf"^{re.escape(label)}\s*:?\s*", "", cell).strip()
        if remainder and remainder != cell:
            return _clean_value(remainder)

        for next_cell in row.cells[index + 1 :]:
            next_value = _clean_value(next_cell)
            if next_value:
                return next_value

    return _extract_labeled_value(row.text, label, stop_labels)


def _extract_case_category_from_text(text: str) -> str | None:
    match = re.search(r"Категорія\s+справи(?:\s*№)?\s*(.*)$", text)
    if not match:
        return None
    return _strip_case_number_from_category(match.group(1))


def _extract_case_category_from_row(row: InfoTableRow) -> str | None:
    for index, cell in enumerate(row.cells):
        if "Категорія справи" not in cell:
            continue

        category = _extract_case_category_from_text(cell)
        if category:
            return category

        for next_cell in row.cells[index + 1 :]:
            category = _strip_case_number_from_category(next_cell)
            if category:
                return category

    return _extract_case_category_from_text(row.text)


def _select_document_header_table(tables: list[HeaderTable]) -> HeaderTable | None:
    for table in tables:
        if table.table_id == "info":
            return table
    return None


def _serialize_table_rows(table: HeaderTable) -> str:
    rows = [
        {
            "text": row.text,
            "cells": row.cells,
            "bold_values": row.bold_values,
        }
        for row in table.rows
    ]
    return json.dumps(rows, ensure_ascii=False)


def parse_document_header_html(document_html: str) -> dict[str, str | None]:
    parser = _HeaderTableParser()
    parser.feed(document_html)
    parser.close()

    header = {column: None for column in DOCUMENT_HEADER_COLUMNS}
    header_table = _select_document_header_table(parser.tables)
    if header_table is None:
        return header

    header["document_header_info_table_text"] = "\n".join(
        row.text for row in header_table.rows
    )
    header["document_header_info_table_rows_json"] = _serialize_table_rows(
        header_table
    )

    date_labels = (
        "Надіслано судом",
        "Зареєстровано",
        "Забезпечено надання загального доступу",
    )

    for row in header_table.rows:
        text = row.text
        bold_values = row.bold_values

        if "Категорія справи" in text:
            if bold_values:
                header["document_header_case_category"] = (
                    _strip_case_number_from_category(bold_values[0])
                )
            else:
                header["document_header_case_category"] = (
                    _extract_case_category_from_row(row)
                )
        elif any(label in text for label in date_labels):
            if len(bold_values) >= 3 and all(label in text for label in date_labels):
                header["document_header_sent_by_court"] = _clean_date(bold_values[0])
                header["document_header_registered"] = _clean_date(bold_values[1])
                header["document_header_public_access_provided"] = _clean_date(
                    bold_values[2]
                )
            else:
                header["document_header_sent_by_court"] = _clean_date(
                    _extract_value_from_row(row, date_labels[0], date_labels[1:])
                )
                header["document_header_registered"] = _clean_date(
                    _extract_value_from_row(row, date_labels[1], (date_labels[2],))
                )
                header["document_header_public_access_provided"] = _clean_date(
                    _extract_value_from_row(row, date_labels[2])
                )
        elif "Номер судового провадження" in text:
            header["document_header_court_proceeding_number"] = _clean_value(
                bold_values[0]
                if bold_values
                else _extract_value_from_row(row, "Номер судового провадження")
            )
        elif "Номер кримінального провадження в ЄРДР" in text:
            header["document_header_erdr_criminal_proceeding_number"] = _clean_value(
                bold_values[0]
                if bold_values
                else _extract_value_from_row(
                    row, "Номер кримінального провадження в ЄРДР"
                )
            )

    return header


def _is_review_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return "/Review/" in parsed.path


def build_review_url(review_id: str | int) -> str:
    review_id_text = str(review_id).strip()
    if not review_id_text:
        raise ValueError("review_id must not be empty")
    if review_id_text.startswith(("http://", "https://")):
        if not _is_review_url(review_id_text):
            raise ValueError(
                "Document header extraction requires a Review page URL, "
                f"got: {review_id_text}"
            )
        return review_id_text

    quoted_review_id = urllib.parse.quote(review_id_text, safe="")
    return REVIEW_URL_TEMPLATE.format(review_id=quoted_review_id)


def download_document_header_html(
    review_id_or_url: str | int,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
) -> tuple[str, int | None, str]:
    url = build_review_url(review_id_or_url)
    headers = dict(DEFAULT_REQUEST_HEADERS)
    if request_headers:
        headers.update(request_headers)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        status_code = response.getcode()
    return body.decode(charset, errors="replace"), status_code, url


def fetch_document_header(
    review_id_or_url: str | int,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        column: None
        for column in (*DOCUMENT_HEADER_COLUMNS, *DOCUMENT_HEADER_METADATA_COLUMNS)
    }
    result["document_header_url"] = build_review_url(review_id_or_url)

    try:
        document_html, status_code, url = download_document_header_html(
            review_id_or_url=review_id_or_url,
            timeout=timeout,
            request_headers=request_headers,
        )
        parsed_header = parse_document_header_html(document_html)
        result.update(parsed_header)
        result["document_header_url"] = url
        result["document_header_status_code"] = status_code
        if not any(parsed_header.values()):
            result["document_header_error"] = "No document header values found in HTML"
    except urllib.error.HTTPError as exc:
        result["document_header_status_code"] = exc.code
        result["document_header_error"] = str(exc)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        result["document_header_error"] = str(exc)

    return result


def _iter_rows(df: pd.DataFrame) -> Iterable[tuple[object, dict]]:
    for index, row in df.iterrows():
        yield index, row.to_dict()


def _resolve_review_id_or_url(
    row: dict,
    url_column: str | None,
    id_column: str,
) -> object:
    if url_column and url_column in row and pd.notna(row[url_column]):
        url = _normalize_text(str(row[url_column]))
        if url and _is_review_url(url):
            return url

    if id_column in row and pd.notna(row[id_column]):
        review_id = _normalize_text(str(row[id_column]))
        if review_id:
            return review_id

    if url_column and url_column in row and pd.notna(row[url_column]):
        url = _normalize_text(str(row[url_column]))
        if url:
            raise ValueError(
                "Document header extraction requires doc_id or a Review page URL; "
                f"non-header URL was provided: {url}"
            )

    source_columns = f"{url_column!r} or {id_column!r}" if url_column else repr(id_column)
    raise ValueError(f"Missing document review identifier in {source_columns}")


def extract_document_headers(
    dataset: pd.DataFrame,
    url_column: str | None = None,
    id_column: str = "doc_id",
    limit: int | None = None,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
    sleep_seconds: float = 0,
    on_batch_complete: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    if limit is not None:
        dataset = dataset.iloc[:limit]

    result_dataset = dataset.copy()
    for column in (*DOCUMENT_HEADER_COLUMNS, *DOCUMENT_HEADER_METADATA_COLUMNS):
        if column not in result_dataset.columns:
            result_dataset[column] = pd.NA

    for index, row in _iter_rows(result_dataset):
        try:
            review_id_or_url = _resolve_review_id_or_url(
                row=row,
                url_column=url_column,
                id_column=id_column,
            )
            header = fetch_document_header(
                review_id_or_url=review_id_or_url,
                timeout=timeout,
                request_headers=request_headers,
            )
        except Exception as exc:
            header = {
                column: None
                for column in (*DOCUMENT_HEADER_COLUMNS, *DOCUMENT_HEADER_METADATA_COLUMNS)
            }
            header["document_header_error"] = str(exc)

        for column, value in header.items():
            result_dataset.at[index, column] = value

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if on_batch_complete is not None:
        on_batch_complete(result_dataset)

    return result_dataset
