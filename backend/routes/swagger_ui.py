# backend/routes/swagger_ui.py
"""Route pour servir Swagger UI avec la spec OpenAPI statique.

Cette route permet de visualiser la spec OpenAPI générée via Swagger UI,
même si Flask-RESTX n'est pas configuré pour générer la doc automatiquement.
"""

from pathlib import Path

from flask import (  # pyright: ignore[reportMissingImports]
    Blueprint,
    Response,
    send_from_directory,
)

# ✅ A1: Blueprint sous /api/v1 pour cohérence avec l'API
swagger_ui_bp = Blueprint("swagger_ui", __name__, url_prefix="/api/v1")


@swagger_ui_bp.route("/docs")
def swagger_ui():
    """Affiche Swagger UI avec la spec OpenAPI.

    Accessible à:
    - Local: http://localhost:5000/api/v1/docs
    - Prod: https://www.lirie.ch/api/v1/docs
    """
    # ✅ A1: Servir Swagger UI depuis un fichier statique
    # Utiliser Swagger UI via CDN pour éviter d'ajouter des dépendances
    spec_url = "/api/v1/openapi.json"  # Chemin vers la spec JSON

    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>ATMR API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: "{spec_url}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                tryItOutEnabled: true
            }});
        }};
    </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@swagger_ui_bp.route("/openapi.json")
def openapi_json():
    """Sert le fichier openapi.json depuis backend/docs/.

    Accessible à:
    - Local: http://localhost:5000/api/v1/openapi.json
    - Prod: https://www.lirie.ch/api/v1/openapi.json
    """
    # ✅ A1: Servir le fichier openapi.json généré
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    spec_file = docs_dir / "openapi.json"

    if not spec_file.exists():
        # Fallback vers swagger.json si openapi.json n'existe pas
        spec_file = docs_dir / "swagger.json"

    if not spec_file.exists():
        from flask import jsonify  # pyright: ignore[reportMissingImports]

        return jsonify({"error": "OpenAPI spec not found"}), 404

    # ✅ A1: Servir avec le bon Content-Type
    return send_from_directory(
        str(docs_dir),
        spec_file.name,
        mimetype="application/json",
    )
