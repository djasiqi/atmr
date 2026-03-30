#!/usr/bin/env python3
"""
Lit les lignes access Traefik (stdin ou fichier) et écrit une ligne TSV par requête :
  timestamp|method|path|status|duration_ms

La durée en fin de ligne (ex. 118ms) sert de proxy « coût total » (réseau + app + DB).
"""
from __future__ import annotations

import re
import sys

# [30/Mar/2026:20:30:00 +0000] "GET /path HTTP/2.0" 200 17086 ... 118ms
LINE_RE = re.compile(
    r"\[([^\]]+)\]\s+"
    r'"([A-Z]+)\s+(\S+)\s+HTTP/[^"]+"\s+'
    r"(\d{3})\s+"
)
DUR_RE = re.compile(r"(\d+)ms\s*$")


def parse_line(line: str) -> tuple[str, str, str, int, int] | None:
    line = line.strip()
    if "30/Mar/2026" not in line:
        return None
    m = LINE_RE.search(line)
    if not m:
        return None
    dm = DUR_RE.search(line)
    if not dm:
        return None
    ts, method, path, status = m.group(1), m.group(2), m.group(3), int(m.group(4))
    dur = int(dm.group(1))
    return ts, method, path, status, dur


def main() -> None:
    paths: list[str] = []
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    if paths:
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    r = parse_line(line)
                    if r:
                        ts, method, path, status, dur = r
                        print(f"{ts}|{method}|{path}|{status}|{dur}")
    else:
        for line in sys.stdin:
            r = parse_line(line)
            if r:
                ts, method, path, status, dur = r
                print(f"{ts}|{method}|{path}|{status}|{dur}")


if __name__ == "__main__":
    main()
