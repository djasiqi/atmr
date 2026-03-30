#!/usr/bin/env python3
"""Parse Traefik docker logs: extract [timestamp] METHOD path for access lines."""
import re
import subprocess
import sys

# Combined log: IP - - [30/Mar/2026:00:01:33 +0000] "GET /path HTTP/1.1" ...
LINE_RE = re.compile(
    r"\[([^\]]+)\]\s+\"([A-Z]+)\s+(\S+)\s+HTTP/[^\"]+\""
)


def main() -> None:
    since = sys.argv[1] if len(sys.argv) > 1 else "2026-03-30T00:00:00"
    until = sys.argv[2] if len(sys.argv) > 2 else "2026-03-31T00:00:00"
    p = subprocess.Popen(
        ["docker", "logs", "traefik", "--since", since, "--until", until],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert p.stdout is not None
    for raw in p.stdout:
        line = raw.decode("utf-8", errors="replace")
        if "30/Mar/2026" not in line:
            continue
        m = LINE_RE.search(line)
        if not m:
            continue
        ts, method, path = m.group(1), m.group(2), m.group(3)
        print(f"{ts}|{method}|{path}")
    p.wait()


if __name__ == "__main__":
    main()
