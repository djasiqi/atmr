#!/usr/bin/env python3
"""Script pour analyser la criticité des endpoints API.

Calcule un score de criticité pour chaque endpoint et domaine basé sur:
- Méthode HTTP (POST/PATCH/DELETE = +5)
- Path contenant des mots-clés critiques
- Utilisation par web ET mobile
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Set

# Configuration des scores
SCORES = {
    "method": {
        "POST": 5,
        "PATCH": 5,
        "DELETE": 5,
        "PUT": 4,
        "GET": 1,
    },
    "path_keywords": {
        "auth/login": 5,
        "auth/refresh": 5,
        "auth/logout": 4,
        "bookings": 4,
        "dispatch": 4,
        "assignments": 4,
        "tracking": 4,
        "gps": 4,
        "eta": 4,
        "invoices": 5,
        "payments": 5,
        "partner_invoices": 5,
        "user": 5,
        "clients": 5,
        "medical": 5,
        "partnerships": 4,
        "notifications": 3,
    },
    "used_by_both": 3,  # Si utilisé par web ET mobile
}

# Domaines de regroupement
DOMAINS = {
    "auth": ["auth", "login", "logout", "refresh", "token", "session"],
    "booking": ["booking", "bookings", "trip", "trips", "mission", "missions"],
    "dispatch": ["dispatch", "assign", "assignment", "driver", "chauffeur"],
    "tracking": ["tracking", "gps", "location", "eta", "position"],
    "invoices": ["invoice", "invoices", "payment", "payments", "billing"],
    "partnerships": ["partnership", "partner", "transfer", "subcontract"],
    "notifications": ["notification", "notify", "alert", "push"],
    "clients": ["client", "clients", "patient", "medical"],
    "users": ["user", "users", "profile", "account"],
    "companies": ["company", "companies", "enterprise"],
}

# Répertoires à exclure lors de l'indexation
EXCLUDE_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    ".expo",
    ".turbo",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "coverage",
    "audit_out",
    "docs",
    ".idea",
    ".vscode",
}


def load_openapi_spec(spec_path: Path) -> dict[str, Any]:
    """Charge la spec OpenAPI depuis le fichier JSON."""
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec OpenAPI introuvable: {spec_path}")

    with spec_path.open(encoding="utf-8") as f:
        return json.load(f)


def get_domain_from_path(path: str) -> str:
    """Détermine le domaine d'un endpoint à partir de son path."""
    path_lower = path.lower()
    for domain, keywords in DOMAINS.items():
        for keyword in keywords:
            if keyword in path_lower:
                return domain
    return "other"


def calculate_method_score(method: str) -> int:
    """Calcule le score basé sur la méthode HTTP."""
    return SCORES["method"].get(method.upper(), 0)


def calculate_path_score(path: str) -> int:
    """Calcule le score basé sur les mots-clés dans le path."""
    path_lower = path.lower()
    score = 0
    for keyword, points in SCORES["path_keywords"].items():
        if keyword in path_lower:
            score += points
    return score


def build_code_index(
    directory: Path,
    extensions: list[str],
    exclude_dirs: Set[str],
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> tuple[str, dict[str, list[str]]]:
    """Indexe tous les fichiers d'un répertoire en un seul passage.
    
    Retourne:
    - Un grand texte concaténé de tous les fichiers pertinents
    - Un dictionnaire {chemin_relatif: [lignes]} pour retrouver les fichiers
    Évite de scanner les répertoires exclus (node_modules, .git, etc.).
    """
    if not directory.exists():
        return "", {}
    
    index_content = []
    file_map: dict[str, list[str]] = {}
    file_count = 0
    
    def should_exclude(path: Path) -> bool:
        """Vérifie si un chemin doit être exclu."""
        # Vérifier si un des répertoires exclus est dans le chemin
        for part in path.parts:
            if part in exclude_dirs:
                return True
        return False
    
    def scan_directory(dir_path: Path) -> None:
        """Scan récursif qui évite les répertoires exclus."""
        nonlocal file_count
        
        if max_files and file_count >= max_files:
            return
        
        try:
            # Vérifier si ce répertoire doit être exclu
            if should_exclude(dir_path):
                return
            
            # Parcourir les éléments du répertoire
            try:
                items = list(dir_path.iterdir())
            except (PermissionError, OSError):
                return
            
            for item in items:
                if max_files and file_count >= max_files:
                    break
                
                # Ignorer les répertoires exclus
                if should_exclude(item):
                    continue
                
                if item.is_file() and item.suffix in extensions:
                    try:
                        content = item.read_text(encoding="utf-8", errors="ignore")
                        index_content.append(content)
                        # Stocker le fichier avec ses lignes pour les sample_hits
                        lines = content.split("\n")
                        # Chemin relatif depuis la racine du projet
                        try:
                            rel_path = str(item.relative_to(directory.parent.parent))
                        except ValueError:
                            rel_path = str(item.relative_to(directory))
                        file_map[rel_path] = lines
                        file_count += 1
                    except Exception:
                        continue
                elif item.is_dir():
                    # Récursion pour les sous-répertoires
                    scan_directory(item)
        except Exception as e:
            if verbose:
                print(f"⚠️  Erreur lors du scan de {dir_path}: {e}")
    
    try:
        scan_directory(directory)
    except Exception as e:
        if verbose:
            print(f"⚠️  Erreur lors de l'indexation de {directory}: {e}")
    
    if verbose:
        print(f"✅ Indexé {file_count} fichiers dans {directory.name}")
    
    if max_files and file_count >= max_files:
        if verbose:
            print(f"⚠️  Limite de {max_files} fichiers atteinte pour {directory}")
    
    return "\n".join(index_content), file_map


def normalize_operation_id(operation_id: str) -> str:
    """Normalise un operationId pour la recherche (supprime underscores, etc.)."""
    return operation_id.lower().replace("_", "").replace("-", "")


def find_sample_hits(
    pattern: re.Pattern,
    file_map: dict[str, list[str]],
    max_samples: int = 3,
) -> list[dict[str, Any]]:
    """Trouve des exemples de matches dans les fichiers."""
    samples = []
    for file_path, lines in file_map.items():
        if len(samples) >= max_samples:
            break
        for line_num, line in enumerate(lines, 1):
            if pattern.search(line):
                samples.append({
                    "file": file_path,
                    "line": line_num,
                    "snippet": line.strip()[:100],  # Limiter à 100 caractères
                })
                if len(samples) >= max_samples:
                    break
    return samples


def check_usage_in_codebase(
    endpoint_path: str,
    method: str,
    operation_id: str,
    index_web: str,
    index_mobile: str,
    file_map_web: dict[str, list[str]],
    file_map_mobile: dict[str, list[str]],
) -> dict[str, Any]:
    """Vérifie si l'endpoint est utilisé dans les index de code.
    
    Retourne un dictionnaire avec:
    - web/mobile: bool
    - matched_by: liste des types de match
    - sample_hits: exemples de fichiers/lignes
    """
    usage = {"web": False, "mobile": False}
    matched_by: set[str] = set()
    sample_hits: list[dict[str, Any]] = []
    
    # 1. Détection par path brut (raw_path)
    path_escaped = re.escape(endpoint_path)
    # Extraire le path sans les paramètres pour les template strings
    path_without_params = endpoint_path.split("{")[0].rstrip("/")
    path_parts = [p for p in path_without_params.split("/") if p]
    
    raw_path_patterns = [
        # apiClient.get("/path"), apiClient.post("/path")
        rf'apiClient\.{method.lower()}\s*\(\s*["\']([^"\']*{path_escaped}[^"\']*)["\']',
        # api.get("/path"), api.post("/path")
        rf'\bapi\.{method.lower()}\s*\(\s*["\']([^"\']*{path_escaped}[^"\']*)["\']',
        # request({url: "/path"}), fetch("/path")
        rf'(?:request|fetch)\s*\([^)]*["\']([^"\']*{path_escaped}[^"\']*)["\']',
        # Path direct dans string (plus spécifique)
        rf'["\']([^"\']*{path_escaped}[^"\']*)["\']',
    ]
    
    # 2. Détection par operationId
    operation_id_patterns = []
    if operation_id:
        # Chercher operationId dans le code (ex: post_login, get_autonomous_actions_list)
        operation_id_patterns = [
            rf'\b{re.escape(operation_id)}\b',
        ]
        # Variantes sans underscores
        op_id_no_underscore = operation_id.replace("_", "")
        if op_id_no_underscore != operation_id:
            operation_id_patterns.append(rf'\b{re.escape(op_id_no_underscore)}\b')
    
    # 3. Détection par template strings (ex: `/invoices/${id}`, `/clients/${clientId}`)
    template_patterns = []
    if path_parts:
        first_part = path_parts[0]
        # Pattern pour template string: `/part1/${var}`, `"/part1/" + var
        template_patterns = [
            # Template literal avec ${}
            rf'`[^`]*{re.escape(first_part)}[^`]*\$\{{',
            # Concaténation de strings
            rf'["\']/?{re.escape(first_part)}["\']\s*[+\s]',
            # Template avec variables
            rf'`[^`]*{re.escape(first_part)}[^`]*`',
        ]
        # Si le path a plusieurs parties, chercher aussi la combinaison
        if len(path_parts) > 1:
            second_part = path_parts[1]
            template_patterns.append(
                rf'`[^`]*{re.escape(first_part)}[^`]*{re.escape(second_part)}[^`]*`'
            )
    
    # 4. Détection par clients générés (api.methodName, new ApiClass())
    client_patterns = []
    if operation_id:
        # Convertir operationId en camelCase (ex: get_autonomous_actions -> getAutonomousActions)
        parts = operation_id.split("_")
        if len(parts) > 1:
            camel_case = parts[0] + "".join(p.capitalize() for p in parts[1:])
            # Chercher api.methodName() ou apiClient.methodName()
            client_patterns = [
                rf'\bapi(?:Client)?\.{re.escape(camel_case)}\s*\(',
            ]
        # Chercher aussi le dernier mot pour les classes (ex: new AuthApi())
        if parts:
            last_word = parts[-1]
            if len(last_word) > 2:
                client_patterns.append(
                    rf'new\s+\w*{re.escape(last_word.capitalize())}\w*Api\s*\('
                )
    
    # Combiner tous les patterns avec leurs types
    all_patterns = [
        ("raw_path", p) for p in raw_path_patterns
    ] + [
        ("operation_id", p) for p in operation_id_patterns
    ] + [
        ("template_string", p) for p in template_patterns
    ] + [
        ("wrapper_call", p) for p in client_patterns
    ]
    
    # Chercher dans l'index web - tester TOUS les patterns pour collecter tous les types
    if index_web:
        for match_type, pattern_str in all_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                if pattern.search(index_web):
                    usage["web"] = True
                    matched_by.add(match_type)
                    # Trouver des exemples seulement pour le premier match de ce type
                    if match_type not in [h.get("match_type") for h in sample_hits]:
                        hits = find_sample_hits(pattern, file_map_web, max_samples=1)
                        for hit in hits:
                            hit["match_type"] = match_type
                            hit["source"] = "web"
                        sample_hits.extend(hits)
            except re.error:
                continue
    
    # Chercher dans l'index mobile - tester TOUS les patterns
    if index_mobile:
        for match_type, pattern_str in all_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                if pattern.search(index_mobile):
                    usage["mobile"] = True
                    matched_by.add(match_type)
                    # Trouver des exemples seulement pour le premier match de ce type
                    if match_type not in [h.get("match_type") for h in sample_hits]:
                        hits = find_sample_hits(pattern, file_map_mobile, max_samples=1)
                        for hit in hits:
                            hit["match_type"] = match_type
                            hit["source"] = "mobile"
                        sample_hits.extend(hits)
            except re.error:
                continue
    
    # Limiter sample_hits à 3 exemples maximum
    sample_hits = sample_hits[:3]
    
    return {
        "web": usage["web"],
        "mobile": usage["mobile"],
        "matched_by": sorted(list(matched_by)) if matched_by else [],
        "sample_hits": sample_hits,
    }


def analyze_endpoints(
    spec: dict[str, Any],
    index_web: str,
    index_mobile: str,
    file_map_web: dict[str, list[str]],
    file_map_mobile: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Analyse tous les endpoints de la spec."""
    endpoints = []
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        domain = get_domain_from_path(path)

        for method, operation in methods.items():
            if method.lower() not in ["get", "post", "put", "patch", "delete"]:
                continue

            # Calculer les scores
            method_score = calculate_method_score(method)
            path_score = calculate_path_score(path)

            # Vérifier l'utilisation dans le code (utilise les index)
            operation_id = operation.get("operationId", "")
            usage_data = check_usage_in_codebase(
                path, method, operation_id, index_web, index_mobile,
                file_map_web, file_map_mobile
            )
            
            usage = {"web": usage_data["web"], "mobile": usage_data["mobile"]}
            usage_score = (
                SCORES["used_by_both"] if (usage["web"] and usage["mobile"]) else 0
            )

            # Score total
            total_score = method_score + path_score + usage_score
            
            # Marquer pour review manuelle si score >= 12 et usage == 0
            has_usage = usage["web"] or usage["mobile"]
            needs_manual_review = total_score >= 12 and not has_usage

            endpoint_data = {
                "path": path,
                "method": method.upper(),
                "domain": domain,
                "operation_id": operation_id,
                "summary": operation.get("summary", ""),
                "tags": operation.get("tags", []),
                "scores": {
                    "method": method_score,
                    "path": path_score,
                    "usage": usage_score,
                    "total": total_score,
                },
                "usage": usage,
                "matched_by": usage_data.get("matched_by", []),
                "sample_hits": usage_data.get("sample_hits", []),
                "needs_manual_review": needs_manual_review,
            }
            endpoints.append(endpoint_data)

    return endpoints


def group_by_domain(endpoints: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Groupe les endpoints par domaine."""
    grouped = defaultdict(list)
    for endpoint in endpoints:
        grouped[endpoint["domain"]].append(endpoint)
    return dict(grouped)


def calculate_domain_scores(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Calcule les scores par domaine."""
    domain_scores = {}
    for domain, domain_endpoints in grouped.items():
        total_score = sum(e["scores"]["total"] for e in domain_endpoints)
        avg_score = total_score / len(domain_endpoints) if domain_endpoints else 0
        max_score = max((e["scores"]["total"] for e in domain_endpoints), default=0)

        domain_scores[domain] = {
            "count": len(domain_endpoints),
            "total_score": total_score,
            "avg_score": round(avg_score, 2),
            "max_score": max_score,
            "endpoints": sorted(
                domain_endpoints, key=lambda x: x["scores"]["total"], reverse=True
            ),
        }
    return domain_scores


def generate_markdown_report(
    endpoints: list[dict[str, Any]],
    domain_scores: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Génère le rapport Markdown des features critiques."""
    # Top 20 endpoints
    top_endpoints = sorted(endpoints, key=lambda x: x["scores"]["total"], reverse=True)[
        :20
    ]

    from datetime import datetime

    content = f"""# Features Critiques - Top 20 Endpoints

Généré le: {datetime.now().isoformat()}

## Top 20 Endpoints par Criticité

| Rang | Score | Méthode | Path | Domaine | Utilisation |
|------|-------|---------|------|---------|-------------|
"""

    for idx, endpoint in enumerate(top_endpoints, 1):
        usage_str = ""
        if endpoint["usage"]["web"] and endpoint["usage"]["mobile"]:
            usage_str = "Web + Mobile"
        elif endpoint["usage"]["web"]:
            usage_str = "Web"
        elif endpoint["usage"]["mobile"]:
            usage_str = "Mobile"
        else:
            usage_str = "Aucune"

        content += f"""| {idx} | **{endpoint["scores"]["total"]}** | {endpoint["method"]} | `{endpoint["path"]}` | {endpoint["domain"]} | {usage_str} |
"""

    content += """
## Détail des Scores

"""

    for endpoint in top_endpoints:
        usage_detail = "Web + Mobile" if (endpoint["usage"]["web"] and endpoint["usage"]["mobile"]) else ("Web" if endpoint["usage"]["web"] else ("Mobile" if endpoint["usage"]["mobile"] else "Aucune"))
        
        content += f"""### {endpoint["method"]} {endpoint["path"]}

- **Score total**: {endpoint["scores"]["total"]}
  - Méthode: {endpoint["scores"]["method"]}
  - Path: {endpoint["scores"]["path"]}
  - Utilisation: {endpoint["scores"]["usage"]}
- **Domaine**: {endpoint["domain"]}
- **Utilisation**: {usage_detail}
"""
        
        # Afficher les informations de détection si disponibles
        if endpoint.get("matched_by"):
            content += f"- **Détecté par**: {", ".join(endpoint["matched_by"])}\n"
        
        if endpoint.get("sample_hits"):
            content += "- **Exemples d'utilisation**:\n"
            for hit in endpoint["sample_hits"][:3]:
                content += f"  - `{hit.get("file", "N/A")}:{hit.get("line", "N/A")}` - {hit.get("snippet", "")[:80]}...\n"
        
        if endpoint.get("needs_manual_review"):
            content += f"- ⚠️ **Nécessite une review manuelle** (score élevé mais usage non détecté)\n"
        
        content += f"- **Summary**: {endpoint.get("summary", "N/A")}\n\n"

    content += """
## Scores par Domaine

| Domaine | Endpoints | Score Total | Score Moyen | Score Max |
|---------|-----------|-------------|-------------|-----------|
"""

    sorted_domains = sorted(
        domain_scores.items(), key=lambda x: x[1]["total_score"], reverse=True
    )

    for domain, scores in sorted_domains:
        content += f"""| {domain} | {scores["count"]} | {scores["total_score"]} | {scores["avg_score"]} | {scores["max_score"]} |
"""

    content += """
## Top 5 par Domaine

"""

    for domain, scores in sorted_domains[:8]:  # Top 8 domaines
        content += f"""### {domain.capitalize()}

"""
        for endpoint in scores["endpoints"][:5]:
            content += f"""- **{endpoint["scores"]["total"]}** - {endpoint["method"]} `{endpoint["path"]}`

"""

    # Section pour les endpoints nécessitant une review manuelle
    needs_review = [e for e in endpoints if e.get("needs_manual_review", False)]
    if needs_review:
        content += """
## ⚠️ Endpoints Nécessitant une Review Manuelle

Ces endpoints ont un score élevé (>= 12) mais aucune utilisation détectée dans le code.
Une vérification manuelle est recommandée pour confirmer s'ils sont réellement utilisés.

| Score | Méthode | Path | Domaine | Operation ID |
|-------|---------|------|---------|--------------|
"""
        for endpoint in sorted(needs_review, key=lambda x: x["scores"]["total"], reverse=True):
            content += f"""| {endpoint["scores"]["total"]} | {endpoint["method"]} | `{endpoint["path"]}` | {endpoint["domain"]} | {endpoint.get("operation_id", "N/A")} |
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"✅ Rapport Markdown généré: {output_path}")


def generate_json_report(
    endpoints: list[dict[str, Any]],
    domain_scores: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Génère le rapport JSON de criticité."""
    report = {
        "metadata": {
            "total_endpoints": len(endpoints),
            "domains": len(domain_scores),
        },
        "domains": {},
        "top_endpoints": sorted(
            endpoints, key=lambda x: x["scores"]["total"], reverse=True
        )[:20],
    }

    for domain, scores in domain_scores.items():
        report["domains"][domain] = {
            "count": scores["count"],
            "total_score": scores["total_score"],
            "avg_score": scores["avg_score"],
            "max_score": scores["max_score"],
            "top_endpoints": [
                {
                    "path": e["path"],
                    "method": e["method"],
                    "score": e["scores"]["total"],
                }
                for e in scores["endpoints"][:10]
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Rapport JSON généré: {output_path}")


def analyze_logs(project_root: Path) -> dict[str, Any]:
    """Analyse les logs nginx/gunicorn si disponibles (bonus)."""
    log_stats = {"available": False, "endpoints": {}}

    # Chercher les logs
    log_paths = [
        project_root / "logs" / "nginx" / "access.log",
        project_root / "logs" / "gunicorn" / "access.log",
        project_root / "backend" / "logs" / "access.log",
    ]

    for log_path in log_paths:
        if log_path.exists():
            print(f"📊 Analyse des logs: {log_path}")
            try:
                # Parser les logs (format nginx/gunicorn standard)
                endpoint_counts = defaultdict(int)
                error_counts = defaultdict(int)

                with log_path.open(encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        # Format: IP - - [timestamp] "METHOD /path HTTP/1.1" status size
                        match = re.search(
                            r'"(\w+)\s+([^\s]+)\s+HTTP[^"]*"\s+(\d+)', line
                        )
                        if match:
                            method = match.group(1)
                            path = match.group(2)
                            status = int(match.group(3))

                            endpoint_key = f"{method} {path}"
                            endpoint_counts[endpoint_key] += 1

                            if status >= 400:
                                error_counts[endpoint_key] += status

                log_stats["available"] = True
                log_stats["endpoints"] = {
                    "by_volume": dict(
                        sorted(
                            endpoint_counts.items(), key=lambda x: x[1], reverse=True
                        )[:20]
                    ),
                    "by_errors": dict(
                        sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[
                            :20
                        ]
                    ),
                }
                break
            except Exception as e:
                print(f"⚠️  Erreur lors de l'analyse des logs: {e}")

    return log_stats


def write_progress_file(audit_out_dir: Path, message: str, verbose: bool) -> None:
    """Écrit un message de progression dans un fichier si verbose."""
    if verbose:
        progress_file = audit_out_dir / "progress.txt"
        with progress_file.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Analyse la criticité des endpoints API"
    )
    parser.add_argument(
        "--no-mobile",
        action="store_true",
        help="Ne pas analyser le code mobile",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Ne pas analyser le code frontend",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Ne pas analyser les logs",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limite maximale de fichiers à indexer (ex: 20000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les messages de progression détaillés",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Déterminer le répertoire racine du projet
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "tools":
        project_root = script_path.parent.parent
    else:
        project_root = script_path.parent
    
    # Créer audit_out/ dès le début
    audit_out_dir = project_root / "audit_out"
    audit_out_dir.mkdir(parents=True, exist_ok=True)
    
    spec_path = project_root / "backend" / "docs" / "openapi.json"
    markdown_output = project_root / "docs" / "FEATURES_CRITIQUES.md"
    json_output = audit_out_dir / "criticality.json"

    print("🔍 Analyse de la criticité des endpoints API...")
    print(f"   Spec: {spec_path}")
    write_progress_file(audit_out_dir, "Démarrage de l'analyse", args.verbose)

    # Charger la spec
    spec = load_openapi_spec(spec_path)
    print(f"✅ Spec chargée: {len(spec.get('paths', {}))} paths")
    write_progress_file(
        audit_out_dir,
        f"Spec chargée: {len(spec.get('paths', {}))} paths",
        args.verbose,
    )

    # Phase d'indexation unique
    print("📚 Indexation du code...")
    write_progress_file(audit_out_dir, "Début de l'indexation", args.verbose)
    
    extensions = [".js", ".jsx", ".ts", ".tsx"]
    index_web = ""
    index_mobile = ""
    file_map_web: dict[str, list[str]] = {}
    file_map_mobile: dict[str, list[str]] = {}
    
    if not args.no_frontend:
        frontend_dir = project_root / "frontend" / "src"
        if frontend_dir.exists():
            print("   Indexation frontend...")
            index_web, file_map_web = build_code_index(
                frontend_dir, extensions, EXCLUDE_DIRS, args.max_files, args.verbose
            )
            write_progress_file(
                audit_out_dir,
                f"Indexation frontend terminée ({len(index_web)} caractères, {len(file_map_web)} fichiers)",
                args.verbose,
            )
        else:
            if args.verbose:
                print(f"⚠️  Répertoire frontend introuvable: {frontend_dir}")
    
    if not args.no_mobile:
        mobile_dir = project_root / "mobile" / "operations-app"
        if mobile_dir.exists():
            print("   Indexation mobile...")
            index_mobile, file_map_mobile = build_code_index(
                mobile_dir, extensions, EXCLUDE_DIRS, args.max_files, args.verbose
            )
            write_progress_file(
                audit_out_dir,
                f"Indexation mobile terminée ({len(index_mobile)} caractères, {len(file_map_mobile)} fichiers)",
                args.verbose,
            )
        else:
            if args.verbose:
                print(f"⚠️  Répertoire mobile introuvable: {mobile_dir}")
    
    index_time = time.time() - start_time
    print(f"✅ Indexation terminée en {index_time:.2f}s")
    write_progress_file(
        audit_out_dir, f"Indexation terminée en {index_time:.2f}s", args.verbose
    )

    # Analyser les endpoints
    print("📊 Analyse des endpoints...")
    write_progress_file(audit_out_dir, "Début de l'analyse des endpoints", args.verbose)
    endpoints = analyze_endpoints(
        spec, index_web, index_mobile, file_map_web, file_map_mobile
    )
    print(f"✅ {len(endpoints)} endpoints analysés")
    
    # Compter les endpoints nécessitant une review manuelle
    needs_review_count = sum(1 for e in endpoints if e.get("needs_manual_review", False))
    if needs_review_count > 0:
        print(f"⚠️  {needs_review_count} endpoints nécessitent une review manuelle (score >= 12, usage = 0)")
    
    write_progress_file(
        audit_out_dir, f"{len(endpoints)} endpoints analysés", args.verbose
    )

    # Grouper par domaine
    grouped = group_by_domain(endpoints)
    print(f"✅ Groupés en {len(grouped)} domaines")

    # Calculer les scores par domaine
    domain_scores = calculate_domain_scores(grouped)

    # Analyser les logs (bonus)
    log_stats = {"available": False, "endpoints": {}}
    if not args.no_logs:
        print("📊 Analyse des logs (si disponibles)...")
        log_stats = analyze_logs(project_root)

    # Générer les rapports
    print("📝 Génération des rapports...")
    generate_markdown_report(endpoints, domain_scores, markdown_output)

    # Ajouter les stats de logs au JSON si disponibles
    json_data = {
        "endpoints": sorted(
            endpoints, key=lambda x: x["scores"]["total"], reverse=True
        ),
        "domains": domain_scores,
    }
    if log_stats["available"]:
        json_data["log_stats"] = log_stats

    with json_output.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Rapport JSON généré: {json_output}")

    if log_stats["available"]:
        print("✅ Statistiques de logs incluses dans le rapport JSON")

    total_time = time.time() - start_time
    print(f"\n✅ Analyse terminée en {total_time:.2f}s!")
    print(f"   - Rapport Markdown: {markdown_output}")
    print(f"   - Rapport JSON: {json_output}")
    write_progress_file(
        audit_out_dir, f"Analyse terminée en {total_time:.2f}s", args.verbose
    )


if __name__ == "__main__":
    main()
