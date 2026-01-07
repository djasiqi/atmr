#!/usr/bin/env python3
"""
Script d'inventaire automatique pour l'audit de cohérence ATMR/Lirie.

Extrait:
- Toutes les routes Flask (méthode, path, endpoint, auth, rôles)
- Tous les modèles SQLAlchemy (table, colonnes, contraintes, index)
- Événements Socket.IO (namespaces, events, payloads)

Usage:
    python tools/audit_inventory.py

Sortie:
    - tools/routes.json
    - tools/models.json
    - tools/socketio_events.json
"""

import ast
import json
import re
import sys
from pathlib import Path

# Ajouter le backend au path pour les imports
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Configuration
OUTPUT_DIR = Path(__file__).parent
ROUTES_OUTPUT = OUTPUT_DIR / "routes.json"
MODELS_OUTPUT = OUTPUT_DIR / "models.json"
SOCKETIO_OUTPUT = OUTPUT_DIR / "socketio_events.json"


def extract_flask_routes():
    """Extrait toutes les routes Flask depuis les fichiers routes/ et app.py."""
    routes = []

    # Routes depuis app.py (routes directes)
    app_py = BACKEND_DIR / "app.py"
    if app_py.exists():
        with open(app_py, "r", encoding="utf-8") as f:
            content = f.read()
            # Extraire @app.route(...)
            pattern = (
                r'@app\.route\(["\']([^"\']+)["\'](?:\s*,\s*methods=\[([^\]]+)\])?\)'
            )
            for match in re.finditer(pattern, content):
                path = match.group(1)
                methods_str = match.group(2) if match.group(2) else "GET"
                methods = (
                    [m.strip().strip("\"'") for m in methods_str.split(",")]
                    if methods_str
                    else ["GET"]
                )
                routes.append(
                    {
                        "method": methods[0] if len(methods) == 1 else methods,
                        "path": path,
                        "endpoint": None,  # À déterminer depuis la fonction
                        "source_file": "app.py",
                        "blueprint": None,
                        "namespace": None,
                        "auth_required": None,
                        "roles": None,
                    }
                )

    # Routes depuis routes_api.py et les fichiers routes/
    routes_dir = BACKEND_DIR / "routes"
    if routes_dir.exists():
        for route_file in routes_dir.rglob("*.py"):
            if route_file.name.startswith("__"):
                continue

            try:
                with open(route_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(route_file))

                # Chercher les classes Resource (Flask-RESTX)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Chercher les méthodes HTTP (get, post, put, patch, delete)
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method = item.name.upper()
                                if method in [
                                    "GET",
                                    "POST",
                                    "PUT",
                                    "PATCH",
                                    "DELETE",
                                    "OPTIONS",
                                ]:
                                    # Chercher le décorateur @api.doc ou @ns.route
                                    docstring = ast.get_docstring(item)
                                    routes.append(
                                        {
                                            "method": method,
                                            "path": None,  # Sera déterminé depuis le namespace
                                            "endpoint": f"{node.name}.{item.name}",
                                            "source_file": str(
                                                route_file.relative_to(BACKEND_DIR)
                                            ),
                                            "blueprint": None,
                                            "namespace": None,  # À déterminer depuis le namespace
                                            "auth_required": None,
                                            "roles": None,
                                            "docstring": docstring,
                                        }
                                    )

                # Chercher les @route() décorateurs (blueprints)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Call):
                                if isinstance(decorator.func, ast.Attribute):
                                    if decorator.func.attr == "route":
                                        # Extraire le path depuis les arguments
                                        if decorator.args:
                                            path_arg = decorator.args[0]
                                            if isinstance(path_arg, ast.Constant):
                                                path = path_arg.value
                                                # Extraire methods si présent
                                                methods = ["GET"]
                                                for keyword in decorator.keywords:
                                                    if keyword.arg == "methods":
                                                        if isinstance(
                                                            keyword.value, ast.List
                                                        ):
                                                            methods = [
                                                                el.value
                                                                if isinstance(
                                                                    el, ast.Constant
                                                                )
                                                                else str(el)
                                                                for el in keyword.value.elts
                                                            ]
                                                routes.append(
                                                    {
                                                        "method": methods[0]
                                                        if len(methods) == 1
                                                        else methods,
                                                        "path": path,
                                                        "endpoint": node.name,
                                                        "source_file": str(
                                                            route_file.relative_to(
                                                                BACKEND_DIR
                                                            )
                                                        ),
                                                        "blueprint": None,  # À déterminer
                                                        "namespace": None,
                                                        "auth_required": None,
                                                        "roles": None,
                                                    }
                                                )

            except Exception as e:
                print(
                    f"⚠️  Erreur lors de l'analyse de {route_file}: {e}", file=sys.stderr
                )

    return routes


def extract_sqlalchemy_models():
    """Extrait tous les modèles SQLAlchemy depuis models/."""
    models = []

    models_dir = BACKEND_DIR / "models"
    if not models_dir.exists():
        return models

    for model_file in models_dir.glob("*.py"):
        if model_file.name.startswith("__"):
            continue

        try:
            with open(model_file, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(model_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Vérifier si c'est un modèle SQLAlchemy (hérite de db.Model)
                    is_model = False
                    table_name = None
                    columns = []
                    relationships = []
                    indexes = []
                    constraints = []

                    for base in node.bases:
                        if isinstance(base, ast.Attribute):
                            if base.attr == "Model":
                                is_model = True
                                break
                        elif isinstance(base, ast.Name):
                            if base.id == "Model":
                                is_model = True
                                break

                    if not is_model:
                        continue

                    # Extraire __tablename__
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if (
                                    isinstance(target, ast.Name)
                                    and target.id == "__tablename__"
                                ):
                                    if isinstance(item.value, ast.Constant):
                                        table_name = item.value.value

                    # Extraire les colonnes (mapped_column, Column)
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign):
                            # Type annotation: name: Mapped[type] = mapped_column(...)
                            if isinstance(item.target, ast.Name):
                                col_name = item.target.id
                                col_type = None
                                nullable = True
                                primary_key = False
                                foreign_key = None
                                unique = False
                                index = False

                                # Analyser l'annotation de type
                                if item.annotation:
                                    if isinstance(item.annotation, ast.Subscript):
                                        if (
                                            isinstance(item.annotation.value, ast.Name)
                                            and item.annotation.value.id == "Mapped"
                                        ):
                                            if isinstance(
                                                item.annotation.slice, ast.Name
                                            ):
                                                col_type = item.annotation.slice.id

                                # Analyser la valeur (mapped_column ou Column)
                                if item.value:
                                    if isinstance(item.value, ast.Call):
                                        func_name = None
                                        if isinstance(item.value.func, ast.Name):
                                            func_name = item.value.func.id
                                        elif isinstance(item.value.func, ast.Attribute):
                                            func_name = item.value.func.attr

                                        if func_name in ["mapped_column", "Column"]:
                                            # Analyser les arguments
                                            for arg in item.value.args:
                                                if isinstance(arg, ast.Name):
                                                    if arg.id in [
                                                        "Integer",
                                                        "String",
                                                        "Boolean",
                                                        "DateTime",
                                                        "Text",
                                                        "Float",
                                                        "Numeric",
                                                    ]:
                                                        col_type = arg.id
                                            for keyword in item.value.keywords:
                                                if (
                                                    keyword.arg == "primary_key"
                                                    and isinstance(
                                                        keyword.value, ast.Constant
                                                    )
                                                ):
                                                    primary_key = keyword.value.value
                                                elif (
                                                    keyword.arg == "nullable"
                                                    and isinstance(
                                                        keyword.value, ast.Constant
                                                    )
                                                ):
                                                    nullable = keyword.value.value
                                                elif (
                                                    keyword.arg == "unique"
                                                    and isinstance(
                                                        keyword.value, ast.Constant
                                                    )
                                                ):
                                                    unique = keyword.value.value
                                                elif (
                                                    keyword.arg == "index"
                                                    and isinstance(
                                                        keyword.value, ast.Constant
                                                    )
                                                ):
                                                    index = keyword.value.value
                                                elif keyword.arg == "ForeignKey":
                                                    if isinstance(
                                                        keyword.value, ast.Constant
                                                    ):
                                                        foreign_key = (
                                                            keyword.value.value
                                                        )

                                columns.append(
                                    {
                                        "name": col_name,
                                        "type": col_type,
                                        "nullable": nullable,
                                        "primary_key": primary_key,
                                        "foreign_key": foreign_key,
                                        "unique": unique,
                                        "index": index,
                                    }
                                )

                    # Extraire __table_args__ pour indexes et constraints
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if (
                                    isinstance(target, ast.Name)
                                    and target.id == "__table_args__"
                                ):
                                    if isinstance(item.value, ast.Tuple):
                                        for elt in item.value.elts:
                                            if isinstance(elt, ast.Call):
                                                if isinstance(elt.func, ast.Name):
                                                    if elt.func.id == "Index":
                                                        index_name = None
                                                        columns_list = []
                                                        for keyword in elt.keywords:
                                                            if (
                                                                keyword.arg == "name"
                                                                and isinstance(
                                                                    keyword.value,
                                                                    ast.Constant,
                                                                )
                                                            ):
                                                                index_name = (
                                                                    keyword.value.value
                                                                )
                                                        for arg in elt.args:
                                                            if isinstance(
                                                                arg, ast.Constant
                                                            ):
                                                                columns_list.append(
                                                                    arg.value
                                                                )
                                                        if columns_list:
                                                            indexes.append(
                                                                {
                                                                    "name": index_name,
                                                                    "columns": columns_list,
                                                                }
                                                            )

                    if table_name:
                        models.append(
                            {
                                "class_name": node.name,
                                "table_name": table_name,
                                "source_file": str(model_file.relative_to(BACKEND_DIR)),
                                "columns": columns,
                                "relationships": relationships,
                                "indexes": indexes,
                                "constraints": constraints,
                            }
                        )

        except Exception as e:
            print(f"⚠️  Erreur lors de l'analyse de {model_file}: {e}", file=sys.stderr)

    return models


def extract_socketio_events():
    """Extrait tous les événements Socket.IO depuis sockets/."""
    events = []

    sockets_dir = BACKEND_DIR / "sockets"
    if not sockets_dir.exists():
        return events

    for socket_file in sockets_dir.glob("*.py"):
        if socket_file.name.startswith("__"):
            continue

        try:
            with open(socket_file, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(socket_file))

            # Chercher @socketio.on(...) ou @namespace.on(...)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr == "on":
                                    # Extraire le nom de l'événement
                                    if decorator.args:
                                        event_name_arg = decorator.args[0]
                                        if isinstance(event_name_arg, ast.Constant):
                                            event_name = event_name_arg.value
                                            events.append(
                                                {
                                                    "event": event_name,
                                                    "handler": node.name,
                                                    "namespace": None,  # À déterminer
                                                    "source_file": str(
                                                        socket_file.relative_to(
                                                            BACKEND_DIR
                                                        )
                                                    ),
                                                    "direction": "server_to_client"
                                                    if "emit"
                                                    in content[
                                                        node.lineno
                                                        - 1 : node.end_lineno
                                                    ]
                                                    else "client_to_server",
                                                }
                                            )

        except Exception as e:
            print(f"⚠️  Erreur lors de l'analyse de {socket_file}: {e}", file=sys.stderr)

    return events


def main():
    """Point d'entrée principal."""
    import sys
    import io

    # Forcer UTF-8 pour stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("[*] Extraction des routes Flask...")
    routes = extract_flask_routes()
    print(f"   [+] {len(routes)} routes trouvees")

    print("[*] Extraction des modeles SQLAlchemy...")
    models = extract_sqlalchemy_models()
    print(f"   [+] {len(models)} modeles trouves")

    print("[*] Extraction des evenements Socket.IO...")
    events = extract_socketio_events()
    print(f"   [+] {len(events)} evenements trouves")

    # Sauvegarder en JSON
    print(f"\n[*] Sauvegarde dans {OUTPUT_DIR}...")
    with open(ROUTES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(routes, f, indent=2, ensure_ascii=False)
    print(f"   [+] Routes sauvegardees: {ROUTES_OUTPUT}")

    with open(MODELS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
    print(f"   [+] Modeles sauvegardes: {MODELS_OUTPUT}")

    with open(SOCKETIO_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"   [+] Evenements Socket.IO sauvegardes: {SOCKETIO_OUTPUT}")

    print("\n[+] Inventaire termine!")


if __name__ == "__main__":
    main()
