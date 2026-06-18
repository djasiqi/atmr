#!/usr/bin/env python3
"""Redige les IPv4 publiques dans un fichier texte (preuves ops-readiness)."""

from __future__ import annotations

import ipaddress
import re
import sys
from pathlib import Path

IPV4_RE = re.compile(
    r"\b(?P<ip>(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3})\b"
)

REDACTED = "[REDACTED_IPV4]"


def _is_public_ipv4(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    if ip.version != 4:
        return False
    if str(ip) in ("127.0.0.1", "0.0.0.0"):
        return False
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return False
    if ip.is_reserved or ip.is_unspecified:
        return False
    return True


def _should_redact_match(line: str, m: re.Match[str]) -> bool:
    ip = m.group("ip")
    if not _is_public_ipv4(ip):
        return False
    end = m.end()
    if end < len(line) and line[end] == ".":
        rest = line[end + 1 :]
        if rest and rest[0].isdigit():
            return False
    return True


def redact_text(text: str) -> tuple[str, int]:
    count = 0
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        parts: list[str] = []
        last = 0
        for m in IPV4_RE.finditer(line):
            if not _should_redact_match(line, m):
                continue
            parts.append(line[last : m.start()])
            parts.append(REDACTED)
            last = m.end()
            count += 1
        parts.append(line[last:])
        out_lines.append("".join(parts))
    return "".join(out_lines), count


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: redact_public_ipv4_in_text.py <fichier>...", file=sys.stderr)
        return 2
    total = 0
    for arg in argv[1:]:
        path = Path(arg)
        original = path.read_text(encoding="utf-8", errors="replace")
        redacted, n = redact_text(original)
        if n:
            path.write_text(redacted, encoding="utf-8", newline="\n")
            print(f"{path}: {n} IPv4 publique(s) redigee(s)")
            total += n
        else:
            print(f"{path}: rien a rediger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
