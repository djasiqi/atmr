#!/usr/bin/env python3
"""Export failed/error/skipped test nodeids from parse_pytest_junit reports."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_section(text: str, title: str) -> list[str]:
    pattern = rf"=== {title} \(\d+\) ===\n(.*?)(?=\n=== |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []
    lines = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("->"):
            continue
        lines.append(line)
    return lines


def main() -> None:
    before = Path(sys.argv[1]).read_text(encoding="utf-8")
    after_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("ci-failure-lists.txt")

    sections = ["FAILED", "ERROR", "SKIPPED", "SCOPE_MISMATCH"]
    chunks = [
        "LISTES TESTS CI — commit f216189f (reproduction locale + Postgres 5433)",
        "Source: pytest backend/tests --junitxml=ci-before-fix.xml",
        "",
    ]
    resume = re.search(r"=== RESUME ===\n(.*?)\n\n", before, re.DOTALL)
    if resume:
        chunks.append(resume.group(1).strip())
        chunks.append("")

    for title in sections:
        items = extract_section(before, title)
        chunks.append(f"--- {title} ({len(items)}) ---")
        chunks.extend(items)
        chunks.append("")

    if after_path and after_path.exists():
        after = after_path.read_text(encoding="utf-8")
        chunks.append("=" * 60)
        chunks.append("APRES FIX ScopeMismatch + trace_id (ci-after-fix.xml)")
        resume2 = re.search(r"=== RESUME ===\n(.*?)\n\n", after, re.DOTALL)
        if resume2:
            chunks.append(resume2.group(1).strip())
            chunks.append("")
        for title in sections:
            items = extract_section(after, title)
            chunks.append(f"--- {title} ({len(items)}) ---")
            chunks.extend(items)
            chunks.append("")

    out.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Written {out} ({len(chunks)} lines)")


if __name__ == "__main__":
    main()
