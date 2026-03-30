#!/usr/bin/env python3
"""
Matrice endpoints × fréquence × latence (proxy « coût » via durée Traefik).

Entrée : lignes enrichies (voir traefik_extract_enriched.py) :
  timestamp|method|path|status|duration_ms

Sorties :
  - endpoint_heatmap.csv
  - endpoint_heatmap.html (tableau heatmap + scatter priorité)
  - endpoint_priority_top.json (résumé machine-readable)

Le fichier historique sans durée (3 colonnes) n'est pas suffisant : régénérer avec
  ssh deploy@HOST "docker logs traefik --since ... --until ... 2>&1 | grep 30/Mar/2026" \\
    | python scripts/traefik_extract_enriched.py > enriched.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode


def normalize_path(path: str, strip_query: bool) -> str:
    """Regroupe les IDs numériques / UUID pour agréger les endpoints."""
    u = urlparse(path)
    p = u.path or "/"
    out: list[str] = []
    for seg in p.split("/"):
        if not seg:
            continue
        if seg.isdigit():
            out.append("{id}")
        elif len(seg) >= 32 and seg.count("-") == 4:
            out.append("{uuid}")
        elif len(seg) > 12 and all(c in "0123456789abcdef-" for c in seg.lower()):
            out.append("{uuid}")
        else:
            out.append(seg)
    key = "/" + "/".join(out) if out else "/"
    if strip_query or not u.query:
        return key
    # tri des query pour stabiliser
    q = parse_qsl(u.query, keep_blank_values=True)
    q.sort()
    return key + "?" + urlencode(q)


@dataclass
class Agg:
    method: str
    key: str
    count: int = 0
    durations: list[int] = field(default_factory=list)

    def add(self, ms: int) -> None:
        self.count += 1
        self.durations.append(ms)

    @property
    def sum_ms(self) -> int:
        return sum(self.durations)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.durations, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.durations, 95)

    @property
    def p99_ms(self) -> float:
        return _percentile(self.durations, 99)

    @property
    def priority_score(self) -> float:
        """Volume × latence p95 : pression approximative sur le backend."""
        return self.count * self.p95_ms


def _percentile(values: list[int], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f]) + (k - f) * (s[c] - s[f])


def load_enriched(path: Path) -> list[tuple[str, str, str, int, int]]:
    rows: list[tuple[str, str, str, int, int]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            ts, method, path_s, status_s, dur_s = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                status = int(status_s)
                dur = int(dur_s)
            except ValueError:
                continue
            rows.append((ts, method, path_s, status, dur))
    return rows


def heat_color(t: float) -> str:
    """t in [0,1] : vert -> jaune -> rouge."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        g = 200
        r = int(400 * t)
    else:
        r = 200
        g = int(200 * (1 - (t - 0.5) * 2))
    return f"rgb({r},{g},40)"


def build_html(
    rows_csv: list[dict[str, object]],
    title: str,
    note: str,
) -> str:
    """Table HTML avec colonnes numériques en heatmap."""
    if not rows_csv:
        return f"<html><body><p>Aucune donnée.</p><p>{note}</p></body></html>"

    keys = [
        "method_key",
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "sum_ms",
        "priority_score",
    ]
    labels = {
        "method_key": "Endpoint (méthode + chemin normalisé)",
        "count": "Fréquence",
        "mean_ms": "Latence moy. (ms)",
        "p50_ms": "p50 (ms)",
        "p95_ms": "p95 (ms)",
        "p99_ms": "p99 (ms)",
        "sum_ms": "Σ durée (ms)",
        "priority_score": "Priorité (n × p95)",
    }

    def col_max(k: str) -> float:
        return max(float(r[k]) for r in rows_csv) or 1.0

    max_pri = col_max("priority_score")
    max_cnt = col_max("count")
    max_sum = col_max("sum_ms")

    th = "".join(f"<th>{labels[k]}</th>" for k in keys)
    trs = []
    for r in rows_csv:
        cells = []
        for k in keys:
            val = r[k]
            if k == "method_key":
                cells.append(f'<td style="text-align:left;white-space:pre-wrap">{val}</td>')
            elif k == "count":
                tv = float(val) / max_cnt
                cells.append(
                    f'<td style="background:{heat_color(tv)};text-align:right">{val}</td>'
                )
            elif k == "sum_ms":
                tv = float(val) / max_sum
                cells.append(
                    f'<td style="background:{heat_color(tv)};text-align:right">{val:.0f}</td>'
                )
            elif k == "priority_score":
                tv = float(val) / max_pri
                cells.append(
                    f'<td style="background:{heat_color(tv)};text-align:right">{val:.0f}</td>'
                )
            elif k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
                tv = float(val) / max(1.0, col_max(k))
                cells.append(
                    f'<td style="background:{heat_color(tv)};text-align:right">{val:.1f}</td>'
                )
            else:
                cells.append(f"<td>{val}</td>")
        trs.append("<tr>" + "".join(cells) + "</tr>")
    body = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
    th {{ background: #f0f0f0; }}
    caption {{ text-align: left; font-weight: bold; margin-bottom: 8px; }}
    .note {{ color: #444; max-width: 900px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <p class="note">{note}</p>
  <table>
    <caption>{title}</caption>
    <thead><tr>{th}</tr></thead>
    <tbody>{"".join(trs)}</tbody>
  </table>
</body>
</html>
"""
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description="Heatmap endpoints × fréquence × latence Traefik")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Fichier enrichi (timestamp|method|path|status|duration_ms)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/traefik_heatmap"),
        help="Dossier de sortie",
    )
    ap.add_argument(
        "--api-prefix",
        default="/api/",
        help="Ne garder que les chemins commençant par ce préfixe (vide = tout)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=60,
        help="Nombre max de lignes (tri par priorité décroissante)",
    )
    ap.add_argument(
        "--strip-query",
        action="store_true",
        help="Normaliser sans query string (regroupe toutes les variantes ?date=…)",
    )
    args = ap.parse_args()

    raw = load_enriched(args.input)
    if not raw:
        print(
            "Aucune ligne enrichie (5 colonnes). "
            "Le fichier tmp_traefik_access_20260330.txt sans durée ne suffit pas — "
            "régénérer avec traefik_extract_enriched.py depuis docker logs traefik.",
            file=sys.stderr,
        )
        sys.exit(1)

    prefix = args.api_prefix
    aggs: dict[tuple[str, str], Agg] = {}
    for _ts, method, path_s, _status, dur in raw:
        if prefix and not path_s.startswith(prefix):
            continue
        key = normalize_path(path_s, strip_query=args.strip_query)
        mk = (method, key)
        if mk not in aggs:
            aggs[mk] = Agg(method=method, key=key)
        aggs[mk].add(dur)

    ranked = sorted(aggs.values(), key=lambda a: a.priority_score, reverse=True)[
        : args.top
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "endpoint_heatmap.csv"
    json_path = args.out_dir / "endpoint_priority_top.json"

    rows_out: list[dict[str, object]] = []
    for a in ranked:
        mk = f"{a.method} {a.key}"
        rows_out.append(
            {
                "method_key": mk,
                "count": a.count,
                "mean_ms": round(a.mean_ms, 2),
                "p50_ms": round(a.p50_ms, 2),
                "p95_ms": round(a.p95_ms, 2),
                "p99_ms": round(a.p99_ms, 2),
                "sum_ms": a.sum_ms,
                "priority_score": round(a.priority_score, 2),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)

    note = (
        "Latence = temps total vu par Traefik (réseau + Gunicorn + app + DB). "
        "Priorité = fréquence × p95 (ms) : combine volume et lenteur. "
        "Filtrage : préfixe « "
        + (prefix or "(tout)")
        + " »."
    )
    html = build_html(rows_out, "Endpoints — fréquence × latence (proxy coût)", note)
    html_path = args.out_dir / "endpoint_heatmap.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"CSV : {csv_path}")
    print(f"HTML : {html_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
