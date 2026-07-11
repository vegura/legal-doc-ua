#!/usr/bin/env python3
"""
prepare_rtf_for_openai.py

Convert Ukrainian court documents in RTF format into:
1) Clean UTF-8 plain text
2) Paragraphized JSON (with paragraph IDs and char offsets)
3) JSONL chunks "ready to be fed to OpenAI"
4) Optional OpenAI Batch API JSONL request file for /v1/responses

Why this script exists:
- Court RTFs often contain Cyrillic text stored as CP1251 bytes (escaped as \\u or \\'xx),
  which can appear as mojibake after naive conversion.
- For downstream structure extraction (nodes/edges) it's useful to keep paragraph IDs.

Requirements:
- pandoc (CLI) must be installed and available in PATH.
  On Ubuntu: sudo apt-get install pandoc
- Python 3.10+

Usage examples:
  python prepare_rtf_for_openai.py --input /path/to/file.rtf --out_dir ./openai_ready
  python prepare_rtf_for_openai.py --input_dir ./rtfs --out_dir ./openai_ready --max_chars 8000

Outputs (in out_dir):
  - <doc_id>.json              (full text + paragraph list)
  - openai_inputs.jsonl        (doc chunks, generic)
  - batch_requests_responses.jsonl  (Batch API requests to /v1/responses)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MONTHS_UA = [
    "січня","лютого","березня","квітня","травня","червня",
    "липня","серпня","вересня","жовтня","листопада","грудня"
]

def check_pandoc_installed() -> bool:
    """Check if pandoc is installed and available in PATH."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def pandoc_rtf_to_plain(rtf_path: Path) -> str:
    """Convert RTF -> plain text via pandoc, keeping paragraph breaks."""
    cmd = ["pandoc", "-f", "rtf", "-t", "plain", "--wrap=none", str(rtf_path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise RuntimeError(f"pandoc failed for {rtf_path.name}: {p.stderr[:800]}")
        return p.stdout
    except FileNotFoundError:
        raise RuntimeError(
            "pandoc is not installed or not found in PATH.\n"
            "Please install pandoc:\n"
            "  - Ubuntu/Debian: sudo apt-get install pandoc\n"
            "  - macOS: brew install pandoc\n"
            "  - Windows: choco install pandoc or download from https://pandoc.org"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pandoc timed out while processing {rtf_path.name}")

def fix_cp1251_mojibake_charwise(s: str) -> str:
    """
    Reverse typical CP1251->Latin-1 mojibake char-by-char.
    Safe for mixed content: characters > U+00FF remain unchanged.
    """
    out_chars: list[str] = []
    for ch in s:
        o = ord(ch)
        if o <= 255:
            out_chars.append(bytes([o]).decode("cp1251"))
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_paragraphs(s: str) -> list[str]:
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]
    # Flatten any remaining newlines inside paragraphs
    return [re.sub(r"\n+", " ", p).strip() for p in raw_paras]

def guess_section_tag(p: str) -> str | None:
    up = p.upper()
    if up.strip() in {"УХВАЛА", "РІШЕННЯ", "ПОСТАНОВА", "ВИРОК"}:
        return "TITLE"
    if "ВСТАНОВИВ" in up or "ВСТАНОВИЛА" in up:
        return "FACTS_BEGIN"
    if "УХВАЛИВ" in up or "ВИРІШИВ" in up or "ПОСТАНОВИВ" in up:
        return "RULING_BEGIN"
    if "КЕРУЮЧИСЬ" in up:
        return "LEGAL_BASIS"
    return None

def guess_doc_type(paras: list[str]) -> str | None:
    for p in paras[:10]:
        up = p.strip().upper()
        if up in {"УХВАЛА", "РІШЕННЯ", "ПОСТАНОВА", "ВИРОК"}:
            return up
    return None

def extract_metadata(paras: list[str]) -> dict[str, Any]:
    head = "\n".join(paras[:25])
    meta: dict[str, Any] = {}

    # Case number
    m = re.search(r"Справа\s*№\s*([0-9/\-]+)", head, flags=re.IGNORECASE)
    if m:
        meta["case_number"] = m.group(1).strip()
    else:
        # Some docs start directly with like "154/3125/21"
        m2 = re.search(r"^(\d+/\d+/\d+)\s*$", "\n".join(paras[:5]), flags=re.MULTILINE)
        meta["case_number"] = m2.group(1).strip() if m2 else None

    # Proceeding number
    m = re.search(r"Провадження.*?№\s*([0-9A-Za-zА-Яа-яІіЇїЄєҐґ/.\-]+)", head, flags=re.IGNORECASE)
    if m:
        meta["proceeding_number"] = m.group(1).strip()
    else:
        m2 = re.search(r"\b\d+-кс/[0-9/]+\b|\b1-кс/[0-9/]+\b", head)
        meta["proceeding_number"] = m2.group(0) if m2 else None

    # Date (UA) + ISO
    months_re = "|".join(MONTHS_UA)
    m = re.search(rf"\b(\d{{1,2}})\s+({months_re})\s+(\d{{4}})\s+року\b", head, flags=re.IGNORECASE)
    if m:
        day, month, year = m.group(1), m.group(2).lower(), m.group(3)
        month_num = MONTHS_UA.index(month) + 1 if month in MONTHS_UA else None
        meta["date_ua"] = f"{int(day):02d} {month} {year}"
        meta["date_iso"] = f"{year}-{month_num:02d}-{int(day):02d}" if month_num else None
    else:
        meta["date_ua"] = None
        meta["date_iso"] = None

    # City (often "м. Одеса", "м. Володимир")
    m = re.search(r"року\s+м\.\s*([A-Za-zА-Яа-яІіЇїЄєҐґ\- ]+)", head)
    meta["city"] = m.group(1).strip() if m else None

    # A first line mentioning "суд"
    court_line = None
    for p in paras[:30]:
        if re.search(r"\bсуд\b", p, flags=re.IGNORECASE):
            court_line = p
            break
    meta["court_line"] = court_line

    return meta

def make_doc_id(path: Path, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{h}"

def paragraphize(paras: list[str]) -> tuple[str, list[dict[str, Any]]]:
    full = "\n\n".join(paras)
    offsets: list[dict[str, Any]] = []
    pos = 0
    for i, p in enumerate(paras, start=1):
        start = pos
        end = start + len(p)
        offsets.append({
            "p": i,
            "start": start,
            "end": end,
            "section_tag": guess_section_tag(p),
            "text": p
        })
        pos = end + 2  # \n\n
    return full, offsets

def chunk_paragraphs(paras: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0
    chunk_id = 1

    for para in paras:
        txt = para["text"]
        add_len = len(txt) + (2 if cur else 0)
        if cur and (cur_len + add_len) > max_chars:
            chunks.append({
                "chunk_id": chunk_id,
                "p_from": cur[0]["p"],
                "p_to": cur[-1]["p"],
                "text": "\n\n".join(p["text"] for p in cur),
            })
            chunk_id += 1
            cur = []
            cur_len = 0

        cur.append(para)
        cur_len += add_len

    if cur:
        chunks.append({
            "chunk_id": chunk_id,
            "p_from": cur[0]["p"],
            "p_to": cur[-1]["p"],
            "text": "\n\n".join(p["text"] for p in cur),
        })
    return chunks

def label_chunk_with_paragraph_ids(doc_paras: list[dict[str, Any]], p_from: int, p_to: int) -> str:
    """Return chunk text where each paragraph is prefixed by [pN]."""
    selected = [p for p in doc_paras if p_from <= p["p"] <= p_to]
    return "\n\n".join([f"[p{p['p']}] {p['text']}" for p in selected])

def prepare_document(rtf_path: Path) -> dict[str, Any]:
    raw = pandoc_rtf_to_plain(rtf_path)
    fixed = fix_cp1251_mojibake_charwise(raw)
    normalized = normalize_text(fixed)
    paras = split_paragraphs(normalized)
    text, para_objs = paragraphize(paras)
    meta = extract_metadata(paras)
    doc_type = guess_doc_type(paras)
    doc_id = make_doc_id(rtf_path, text)

    return {
        "doc_id": doc_id,
        "source_file": str(rtf_path),
        "language": "uk",
        "doc_type_guess": doc_type,
        "metadata": meta,
        "text": text,
        "paragraphs": para_objs,
    }

def iter_rtf_files(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        files = [input_path]
    elif input_dir:
        if not input_dir.exists():
            raise FileNotFoundError(str(input_dir))
        files = sorted(input_dir.glob("*.rtf"))
    else:
        raise ValueError("Provide --input or --input_dir")
    return files

def parse_docs_from_dir(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rtf_files = iter_rtf_files(None, out_dir)
    except (FileNotFoundError, ValueError) as e:
        print("ERROR: No RTF files found to process.", file=sys.stderr)

    if not rtf_files:
        print("ERROR: No RTF files found to process.", file=sys.stderr)

    print(f"Found {len(rtf_files)} RTF file(s) to process")
    
    docs: list[dict[str, Any]] = []
    for i, rtf in enumerate(rtf_files, start=1):
        try:
            print(f"Processing [{i}/{len(rtf_files)}]: {rtf.name} ...", end=" ")
            doc = prepare_document(rtf)
            docs.append(doc)
            doc_path = out_dir / f"{doc['doc_id']}.json"
            doc_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"✓ ({len(doc['paragraphs'])} paragraphs)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue
    
    if not docs:
        print("\nERROR: No documents were successfully processed.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nGenerating output files...")
    
    openai_inputs_path = out_dir / "openai_inputs.jsonl"
    total_chunks = 0
    with open(openai_inputs_path, "w", encoding="utf-8") as f:
        for d in docs:
            for ch in chunk_paragraphs(d["paragraphs"], max_chars=args.max_chars):
                rec = {
                    "doc_id": d["doc_id"],
                    "chunk_id": ch["chunk_id"],
                    "p_from": ch["p_from"],
                    "p_to": ch["p_to"],
                    "metadata": d["metadata"],
                    "text": ch["text"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    # 3) Batch API template for /v1/responses (JSON mode placeholder)
    batch_path = out_dir / "batch_requests_responses.jsonl"
    with open(batch_path, "w", encoding="utf-8") as f:
        for d in docs:
            for ch in chunk_paragraphs(d["paragraphs"], max_chars=args.max_chars):
                custom_id = f"{d['doc_id']}_c{ch['chunk_id']:03d}"
                labeled_text = label_chunk_with_paragraph_ids(d["paragraphs"], ch["p_from"], ch["p_to"])
                body = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": "Extract a knowledge-graph JSON from the Ukrainian court decision text. Output JSON only."},
                        {"role": "user", "content": labeled_text},
                    ],
                    # Replace this with Structured Outputs (json_schema) when you plug in your schema
                    "text": {"format": {"type": "json_object"}},
                }
                line = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(docs)} document(s)")
    print(f"✓ Generated {total_chunks} chunk(s)")
    print(f"✓ Output directory: {out_dir.absolute()}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  • {len(docs)} document JSON file(s)")
    print(f"  • {openai_inputs_path.name} ({total_chunks} chunks)")
    print(f"  • {batch_path.name} ({total_chunks} batch requests)")
    print()

def main():
    ap = argparse.ArgumentParser(
        description="Convert Ukrainian court documents (RTF) to OpenAI-ready formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input document.rtf --out_dir ./output
  %(prog)s --input_dir ./rtf_files --out_dir ./output --max_chars 8000
  
Output files:
  - <doc_id>.json              : Full document with paragraph structure
  - openai_inputs.jsonl        : Generic chunked inputs
  - batch_requests_responses.jsonl : OpenAI Batch API requests
  
Requirements:
  - pandoc must be installed (brew install pandoc / apt-get install pandoc)
        """
    )
    ap.add_argument("--input", type=Path, default=None, help="Single RTF file")
    ap.add_argument("--input_dir", type=Path, default=None, help="Directory of .rtf files")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--max_chars", type=int, default=8000, help="Max chars per chunk for OpenAI calls")
    ap.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model name for batch request template")
    args = ap.parse_args()

    # Check pandoc installation
    if not check_pandoc_installed():
        print("ERROR: pandoc is not installed or not found in PATH.", file=sys.stderr)
        print("Please install pandoc:", file=sys.stderr)
        print("  - Ubuntu/Debian: sudo apt-get install pandoc", file=sys.stderr)
        print("  - macOS: brew install pandoc", file=sys.stderr)
        print("  - Windows: choco install pandoc", file=sys.stderr)
        print("  - Or download from: https://pandoc.org/installing.html", file=sys.stderr)
        sys.exit(1)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rtf_files = iter_rtf_files(args.input, args.input_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not rtf_files:
        print("ERROR: No RTF files found to process.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(rtf_files)} RTF file(s) to process")

    # 1) Convert and write per-doc JSON
    docs: list[dict[str, Any]] = []
    for i, rtf in enumerate(rtf_files, start=1):
        try:
            print(f"Processing [{i}/{len(rtf_files)}]: {rtf.name} ...", end=" ")
            doc = prepare_document(rtf)
            docs.append(doc)
            doc_path = out_dir / f"{doc['doc_id']}.json"
            doc_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"✓ ({len(doc['paragraphs'])} paragraphs)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue
    
    if not docs:
        print("\nERROR: No documents were successfully processed.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nGenerating output files...")

    # 2) Generic OpenAI input JSONL (chunks of plain text)
    openai_inputs_path = out_dir / "openai_inputs.jsonl"
    total_chunks = 0
    with open(openai_inputs_path, "w", encoding="utf-8") as f:
        for d in docs:
            for ch in chunk_paragraphs(d["paragraphs"], max_chars=args.max_chars):
                rec = {
                    "doc_id": d["doc_id"],
                    "chunk_id": ch["chunk_id"],
                    "p_from": ch["p_from"],
                    "p_to": ch["p_to"],
                    "metadata": d["metadata"],
                    "text": ch["text"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    # 3) Batch API template for /v1/responses (JSON mode placeholder)
    batch_path = out_dir / "batch_requests_responses.jsonl"
    with open(batch_path, "w", encoding="utf-8") as f:
        for d in docs:
            for ch in chunk_paragraphs(d["paragraphs"], max_chars=args.max_chars):
                custom_id = f"{d['doc_id']}_c{ch['chunk_id']:03d}"
                labeled_text = label_chunk_with_paragraph_ids(d["paragraphs"], ch["p_from"], ch["p_to"])
                body = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": "Extract a knowledge-graph JSON from the Ukrainian court decision text. Output JSON only."},
                        {"role": "user", "content": labeled_text},
                    ],
                    # Replace this with Structured Outputs (json_schema) when you plug in your schema
                    "text": {"format": {"type": "json_object"}},
                }
                line = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(docs)} document(s)")
    print(f"✓ Generated {total_chunks} chunk(s)")
    print(f"✓ Output directory: {out_dir.absolute()}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  • {len(docs)} document JSON file(s)")
    print(f"  • {openai_inputs_path.name} ({total_chunks} chunks)")
    print(f"  • {batch_path.name} ({total_chunks} batch requests)")
    print()

if __name__ == "__main__":
    main()
