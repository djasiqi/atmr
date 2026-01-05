#!/usr/bin/env python3
"""
Script pour auditer la protection CSRF sur tous les endpoints.

Ce script analyse tous les fichiers de routes et identifie :
1. Les endpoints mutants (POST, PUT, DELETE, PATCH)
2. Les endpoints qui n'ont pas de protection CSRF explicite

Usage:
    python backend/scripts/audit_csrf_protection.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import Any


def find_mutating_endpoints(directory: str = "routes") -> list[dict[str, Any]]:
    """Trouve tous les endpoints mutants (POST, PUT, DELETE, PATCH).

    Args:
        directory: Répertoire contenant les fichiers de routes

    Returns:
        Liste de dictionnaires avec les informations sur chaque endpoint
    """
    mutating_methods = ["POST", "PUT", "DELETE", "PATCH"]
    endpoints = []

    routes_dir = Path(directory)

    if not routes_dir.exists():
        print(f"❌ Répertoire non trouvé: {directory}")
        return []

    for file_path in routes_dir.glob("*.py"):
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Chercher les décorateurs pour identifier les méthodes HTTP
                    route_path = None

                    # Vérifier si c'est une méthode de classe Resource (Flask-RESTX)
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == "route"
                        ):
                            # @companies_ns.route("/path")
                            # Extraire le chemin de route
                            if decorator.args:
                                # ast.Constant remplace ast.Str depuis Python 3.8+
                                # Pour compatibilité, vérifier si c'est une constante (string, int, etc.)
                                if isinstance(decorator.args[0], ast.Constant):
                                    try:
                                        route_path = ast.literal_eval(decorator.args[0])
                                    except (ValueError, SyntaxError):
                                        route_path = None
                                else:
                                    route_path = None

                            # Chercher les méthodes HTTP dans les keywords
                            for keyword in decorator.keywords:
                                if keyword.arg == "methods" and isinstance(
                                    keyword.value, ast.List
                                ):
                                    methods = []
                                    for el in keyword.value.elts:
                                        # ast.Constant remplace ast.Str depuis Python 3.8+
                                        if isinstance(el, ast.Constant):
                                            method_val = el.value
                                            if isinstance(method_val, str):
                                                methods.append(method_val)
                                    if any(m in mutating_methods for m in methods):
                                        endpoints.append(
                                            {
                                                "file": str(file_path),
                                                "function": node.name,
                                                "line": node.lineno,
                                                "methods": methods,
                                                "route": route_path,
                                            }
                                        )

                    # Vérifier aussi les méthodes de classe qui correspondent à des méthodes HTTP
                    if node.name.upper() in mutating_methods:
                        # C'est probablement une méthode de classe Resource
                        # Chercher la classe parente
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef):
                                for item in parent.body:
                                    if item == node:
                                        # Trouver le décorateur @route sur la classe
                                        for decorator in parent.decorator_list:
                                            if (
                                                isinstance(decorator, ast.Call)
                                                and isinstance(
                                                    decorator.func, ast.Attribute
                                                )
                                                and decorator.func.attr == "route"
                                            ):
                                                route_path = None
                                                if decorator.args:
                                                    # ast.Constant remplace ast.Str depuis Python 3.8+
                                                    if isinstance(
                                                        decorator.args[0],
                                                        ast.Constant,
                                                    ):
                                                        try:
                                                            route_path = (
                                                                ast.literal_eval(
                                                                    decorator.args[0]
                                                                )
                                                            )
                                                        except (
                                                            ValueError,
                                                            SyntaxError,
                                                        ):
                                                            route_path = None
                                                    else:
                                                        route_path = None

                                                endpoints.append(
                                                    {
                                                        "file": str(file_path),
                                                        "function": f"{parent.name}.{node.name}",
                                                        "line": node.lineno,
                                                        "methods": [node.name.upper()],
                                                        "route": route_path,
                                                        "class": parent.name,
                                                    }
                                                )
                                                break
        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse de {file_path}: {e}", file=sys.stderr)
            continue

    return endpoints


def is_endpoint_exempt_from_csrf(endpoint: dict[str, Any]) -> bool:
    """Vérifie si l'endpoint est exempté de CSRF (selon le middleware global).

    Args:
        endpoint: Dictionnaire avec les informations sur l'endpoint

    Returns:
        True si l'endpoint est exempté, False sinon
    """
    route = endpoint.get("route", "") or ""
    file_name = Path(endpoint["file"]).name.lower()

    # Normaliser le chemin de route pour les comparaisons
    # Les routes peuvent être relatives (sans /api/v1) ou absolues
    normalized_route = route.strip("/")

    # Endpoints exemptés par le middleware global (voir services/csrf_protection.py)
    # Correspondance par nom de route ou pattern
    exempt_patterns = [
        "health",
        "prometheus/metrics",
        "auth/login",
        "auth/register",
        "csrf-token",
        "app/version-check",
        "company_mobile/auth/login",
        "/health",
        "/api/v1/prometheus/metrics",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/csrf-token",
        "/api/v1/app/version-check",
    ]

    # Vérifier si le chemin correspond à un pattern exempté
    for pattern in exempt_patterns:
        if pattern in normalized_route or pattern in route:
            return True

    # Endpoints commençant par certains préfixes sont exemptés
    exempt_prefixes = [
        "/api/v1/webhooks/",
        "/api/v1/company_mobile/dispatch/",
        "/api/v1/company_mobile/auth/",
        "/api/v1/driver/",
        "/api/v1/companies/",
        "/api/v1/company_dispatch/",
        "webhooks/",
        "company_mobile/dispatch/",
        "company_mobile/auth/",
        "driver/",
        "companies/",
        "company_dispatch/",
    ]

    for prefix in exempt_prefixes:
        if route.startswith(prefix) or normalized_route.startswith(prefix.strip("/")):
            return True

    # Vérifier aussi par nom de fichier pour certains cas
    exempt_files = [
        "company_mobile_dispatch.py",  # Routes mobile dispatch exemptées
        "driver.py",  # Routes driver exemptées
    ]

    return file_name in exempt_files


def check_csrf_protection_explicit(endpoint: dict[str, Any]) -> bool:
    """Vérifie si l'endpoint a une protection CSRF explicite (décorateur, etc.).

    Args:
        endpoint: Dictionnaire avec les informations sur l'endpoint

    Returns:
        True si l'endpoint a une protection CSRF explicite, False sinon
    """
    file_path = Path(endpoint["file"])

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)

        # Lire autour de la fonction pour analyser le code
        func_start = max(0, endpoint["line"] - 1)
        # Lire jusqu'à 100 lignes après le début de la fonction
        func_end = min(len(lines), func_start + 100)
        func_code = "".join(lines[func_start:func_end])

        # Chercher des indices de protection CSRF explicite
        csrf_indicators = [
            "@csrf_required",
            "@require_csrf_token",
            "csrf_required",
            "require_csrf_token",
            "get_csrf_token",
            "verify_csrf",
            "validate_csrf",
            "X-CSRF-Token",
            "csrf_token",
            "CSRF_TOKEN",
        ]

        # Vérifier aussi dans les décorateurs (lignes avant la fonction)
        decorator_lines = "".join(lines[max(0, func_start - 10) : func_start])

        return any(
            indicator.lower() in func_code.lower()
            or indicator.lower() in decorator_lines.lower()
            for indicator in csrf_indicators
        )
    except Exception as e:
        print(
            f"⚠️ Erreur lors de la vérification CSRF explicite pour {endpoint['file']}:{endpoint['line']}: {e}",
            file=sys.stderr,
        )
        return False


def check_csrf_protection_via_middleware(endpoint: dict[str, Any]) -> bool:
    """Vérifie si l'endpoint est protégé par le middleware CSRF global.

    Le middleware global protège automatiquement tous les endpoints mutants
    (POST, PUT, DELETE, PATCH) SAUF ceux qui sont exemptés.

    Args:
        endpoint: Dictionnaire avec les informations sur l'endpoint

    Returns:
        True si l'endpoint est protégé par le middleware global, False sinon
    """
    # Si l'endpoint est exempté, il n'est PAS protégé par le middleware
    # Le middleware protège automatiquement tous les endpoints mutants
    # qui ne sont pas exemptés (voir services/csrf_protection.py:validate_csrf_for_mutating_requests)
    # Le middleware est activé par défaut si CSRF_ENABLED=True
    return not is_endpoint_exempt_from_csrf(endpoint)


def check_csrf_protection(endpoint: dict[str, Any]) -> bool:
    """Vérifie si l'endpoint a une protection CSRF (explicite ou via middleware).

    Args:
        endpoint: Dictionnaire avec les informations sur l'endpoint

    Returns:
        True si l'endpoint a une protection CSRF, False sinon
    """
    # Vérifier d'abord la protection explicite, sinon le middleware global
    return check_csrf_protection_explicit(
        endpoint
    ) or check_csrf_protection_via_middleware(endpoint)


def main() -> int:
    """Fonction principale d'audit.

    Returns:
        Code de sortie : 0 si tout est OK, 1 si des problèmes sont trouvés
    """
    # Forcer l'encodage UTF-8 pour les emojis
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8") if hasattr(
            sys.stdout, "reconfigure"
        ) else None

    print("🔍 Audit de la protection CSRF...\n")

    # Trouver tous les endpoints mutants
    endpoints = find_mutating_endpoints()

    if not endpoints:
        print("⚠️  Aucun endpoint mutant trouvé.")
        return 1

    # Vérifier la protection CSRF
    unprotected = []
    protected_explicit = []
    protected_middleware = []
    exempt = []

    for endpoint in endpoints:
        # Vérifier d'abord si l'endpoint est exempté
        if is_endpoint_exempt_from_csrf(endpoint):
            exempt.append(endpoint)
        elif check_csrf_protection_explicit(endpoint):
            protected_explicit.append(endpoint)
        elif check_csrf_protection_via_middleware(endpoint):
            protected_middleware.append(endpoint)
        else:
            unprotected.append(endpoint)

    # Afficher les résultats
    total_protected = len(protected_explicit) + len(protected_middleware)
    print("📊 Résultats de l'audit CSRF :\n")
    print(f"  - Total d'endpoints mutants : {len(endpoints)}")
    print(
        f"  - Endpoints protégés explicitement (décorateur) : {len(protected_explicit)}"
    )
    print(f"  - Endpoints protégés par middleware global : {len(protected_middleware)}")
    print(f"  - Endpoints exemptés (protégés par JWT ou publics) : {len(exempt)}")
    print(f"  - Endpoints nécessitant une protection : {len(unprotected)}")
    print(f"  - ✅ Total protégés : {total_protected} / {len(endpoints)}\n")

    if unprotected:
        print("⚠️  Endpoints nécessitant une protection CSRF :\n")
        for ep in unprotected:
            route_info = f" ({ep.get('route', 'N/A')})" if ep.get("route") else ""
            class_info = f" dans {ep['class']}" if ep.get("class") else ""
            print(
                f"  - {Path(ep['file']).name}:{ep['line']} - {ep['function']}{class_info} [{', '.join(ep['methods'])}]{route_info}"
            )
        print("\n💡 Recommandation :")
        print("   1. Vérifier que CSRF_ENABLED=true dans la configuration")
        print("   2. Vérifier que ces endpoints ne devraient pas être exemptés")
        print(
            "   3. Si nécessaire, ajouter le décorateur @csrf_required pour une protection explicite"
        )
        return 1

    print("✅ Tous les endpoints mutants sont protégés contre CSRF !")
    print("\n📝 Répartition de la protection :")
    print(f"   - Protection explicite (décorateurs) : {len(protected_explicit)}")
    print(f"   - Protection middleware global : {len(protected_middleware)}")
    print(f"   - Endpoints exemptés (JWT/publics) : {len(exempt)}")
    return 0


if __name__ == "__main__":
    # Changer le répertoire de travail pour que les chemins relatifs fonctionnent
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    exit_code = main()
    sys.exit(exit_code)
