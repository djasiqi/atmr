#!/usr/bin/env python3
"""Parse pytest JUnit XML into failed/error/skipped lists."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    failed: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    scope_mismatch: list[str] = []

    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        nodeid = f"{classname}::{name}"
        for child in case:
            msg = child.get("message", "") or child.text or ""
            if child.tag == "failure":
                failed.append((nodeid, msg))
                if "ScopeMismatch" in msg:
                    scope_mismatch.append(nodeid)
            elif child.tag == "error":
                errors.append((nodeid, msg))
                if "ScopeMismatch" in msg:
                    scope_mismatch.append(nodeid)
            elif child.tag == "skipped":
                skipped.append((nodeid, msg))

    cats = Counter()
    for _, msg in failed + errors:
        first_line = msg.split("\n")[0][:100] if msg else "unknown"
        cats[first_line] += 1

    return {
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "scope_mismatch": scope_mismatch,
        "categories": cats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml")
    parser.add_argument("--out", help="Write full lists to this file")
    args = parser.parse_args()

    data = parse_junit(Path(args.junit_xml))
    lines = [
        "=== RESUME ===",
        f"FAILED:  {len(data['failed'])}",
        f"ERROR:   {len(data['errors'])}",
        f"SKIPPED: {len(data['skipped'])}",
        f"ScopeMismatch: {len(data['scope_mismatch'])}",
        "",
        "=== TOP CAUSES (première ligne) ===",
    ]
    for cause, count in data["categories"].most_common(30):
        lines.append(f"{count:4d}  {cause}")

    sections = [
        ("FAILED", data["failed"]),
        ("ERROR", data["errors"]),
        ("SKIPPED", data["skipped"]),
        ("SCOPE_MISMATCH", [(n, "") for n in data["scope_mismatch"]]),
    ]
    for title, items in sections:
        lines.extend(["", f"=== {title} ({len(items)}) ==="])
        for nodeid, msg in items:
            lines.append(nodeid)
            if msg and title != "SCOPE_MISMATCH":
                first = msg.split("\n")[0][:120]
                lines.append(f"  -> {first}")

    output = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Full report written to {args.out} ({len(output)} chars)")
    else:
        print(output[:8000])
        if len(output) > 8000:
            print(f"\n... ({len(output)} chars total, use --out for full file)")


if __name__ == "__main__":
    main()
