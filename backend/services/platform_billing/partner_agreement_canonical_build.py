"""Construction contrôlée des PDF canoniques à partir des sources Markdown.

Usage :
  python -m services.platform_billing.partner_agreement_canonical_build \\
      --version lirie-partner-terms-v1.20 --update-manifest

  Écrasement d'une version existante (explicite uniquement) :
  ... --force-overwrite --update-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.platform_billing.partner_agreement_canonical import (
    canonical_hashes_path,
    canonical_pdf_path,
    canonical_root,
    canonical_source_path,
)

LIRIE_GREEN = colors.HexColor("#00796B")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CanonTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=LIRIE_GREEN,
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "CanonH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=LIRIE_GREEN,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "h3": ParagraphStyle(
            "CanonH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9,
            spaceBefore=5,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "CanonBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_JUSTIFY,
            spaceAfter=3,
        ),
        "meta": ParagraphStyle(
            "CanonMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "CanonBullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            leftIndent=10,
            spaceAfter=1,
        ),
        "cell": ParagraphStyle(
            "CanonCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
        ),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(text: str) -> str:
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        r"<font face='Courier' size='7'>\1</font>",
        text,
    )
    return text


def markdown_to_flowables(md: str) -> list:
    styles = _styles()
    flow: list = []
    lines = md.replace("\r\n", "\n").split("\n")
    i = 0
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        data = []
        for row in table_rows:
            if all(set(cell.strip()) <= {"-", ":"} for cell in row):
                continue
            data.append([Paragraph(_inline_md(c.strip()), styles["cell"]) for c in row])
        if data:
            n = len(data[0])
            width = 16.5 * cm
            col_w = [width / n] * n
            tbl = Table(data, colWidths=col_w)
            tbl.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.96, 0.95)),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            flow.append(tbl)
            flow.append(Spacer(1, 4))
        table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            table_rows.append(cells)
            i += 1
            continue
        if table_rows:
            flush_table()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            flow.append(Paragraph(_inline_md(line[2:].strip()), styles["title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(_inline_md(line[3:].strip()), styles["h2"]))
        elif line.startswith("### "):
            flow.append(Paragraph(_inline_md(line[4:].strip()), styles["h3"]))
        elif line.startswith("---"):
            flow.append(Spacer(1, 6))
        elif line.startswith("- "):
            flow.append(
                Paragraph("• " + _inline_md(line[2:].strip()), styles["bullet"])
            )
        else:
            style = styles["meta"] if line.startswith("**") else styles["body"]
            flow.append(Paragraph(_inline_md(line), style))
        i += 1
    if table_rows:
        flush_table()
    return flow


def build_canonical_pdf_bytes(source_md: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="LIRIE — document canonique",
        author="LIRIE",
        creator="LIRIE-canonical-build",
    )
    doc.build(markdown_to_flowables(source_md))
    return buffer.getvalue()


def _update_manifest(version: str, digest: str, size: int) -> None:
    path = canonical_hashes_path()
    manifest: dict = {}
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[version] = {
        "version": version,
        "sha256": digest,
        "size_bytes": size,
        "source": f"sources/{version}.md",
        "pdf": f"pdf/{version}.pdf",
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def rebuild_version(
    version: str,
    *,
    force_overwrite: bool = False,
    update_manifest: bool = False,
) -> dict:
    source = canonical_source_path(version)
    if not source.is_file():
        raise FileNotFoundError(f"Source manquante : {source}")
    pdf_bytes = build_canonical_pdf_bytes(source.read_text(encoding="utf-8"))
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    target = canonical_pdf_path(version)
    tmp_dir = canonical_root() / ".rebuild_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{version}.pdf"
    tmp_path.write_bytes(pdf_bytes)

    if target.is_file():
        old_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if old_sha == digest:
            if update_manifest:
                _update_manifest(version, digest, len(pdf_bytes))
            return {
                "version": version,
                "status": "unchanged",
                "sha256": digest,
                "size_bytes": len(pdf_bytes),
            }
        if not force_overwrite:
            return {
                "version": version,
                "status": "blocked_overwrite",
                "old_sha256": old_sha,
                "new_sha256": digest,
                "size_bytes": len(pdf_bytes),
                "tmp_path": str(tmp_path),
                "message": (
                    "Écrasement silencieux refusé. Utilisez une nouvelle version "
                    "ou --force-overwrite --update-manifest."
                ),
            }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_bytes)
    if update_manifest:
        _update_manifest(version, digest, len(pdf_bytes))
    return {
        "version": version,
        "status": "written",
        "sha256": digest,
        "size_bytes": len(pdf_bytes),
        "tmp_path": str(tmp_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruction contrôlée des PDF canoniques LIRIE"
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args(argv)
    result = rebuild_version(
        args.version,
        force_overwrite=args.force_overwrite,
        update_manifest=args.update_manifest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result.get("status") == "blocked_overwrite" else 0


if __name__ == "__main__":
    sys.exit(main())
