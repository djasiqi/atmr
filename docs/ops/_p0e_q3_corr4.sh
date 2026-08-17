# Sample: what do 409 location responses look like in backend access? Use a live probe of recent similar? 
# Instead query if any socket emits logged. Check redis active session keys history impossible.
# Count 409 vs 202 around rotates; list POST sessions with ±30s context of 409
python3 - <<'PY'
from datetime import datetime
# Just print structured summary from traefik grep saved via docker
import subprocess
out=subprocess.check_output(["bash","-lc",'''docker logs traefik --since "2026-08-17T13:22:00Z" --until "2026-08-17T13:32:00Z" 2>&1 | grep -E "driver/me/location|tracking/sessions" | grep -v DBG'''], text=True, errors="replace")
rows=[]
for line in out.splitlines():
    # [17/Aug/2026:13:23:38 +0000] "PUT ... location HTTP/2.0" 409
    import re
    m=re.search(r"\[(.*?)\] \"(GET|PUT|POST) ([^\"]+)\" (\d+)", line)
    if not m: continue
    ts, method, path, status = m.groups()
    path=path.split(" ")[0]
    rows.append((ts, method, path, status))
print("SUMMARY_EVENTS")
for r in rows:
    if r[3] in ("409","200") or "sessions" in r[2]:
        print(f"{r[0]} {r[1]} {r[2]} -> {r[3]}")
# Gaps: any 409 in [13:25:50, 13:26:20]?
print("\n409_IN_ROTATE_WINDOW")
for r in rows:
    if r[3]=="409" and "13:26:" in r[0]:
        print(r)
print("(none)" if not any(r[3]=="409" and "13:26:" in r[0] for r in rows) else "")
print("\nPOST_SESSIONS")
for r in rows:
    if "sessions" in r[2]:
        print(r)
PY