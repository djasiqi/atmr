# backend/wsgi.py
# 1) Eventlet doit patcher AVANT tout autre import
import eventlet
eventlet.monkey_patch()

import os
import logging
from dotenv import load_dotenv

# 2) Charger .env (idempotent, utile en dev/CLI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# 3) Importer l'app et le socketio APRÈS monkey_patch
from app import create_app
from ext import socketio

# 4) Construire l'app selon la config demandée
config_name = os.getenv("FLASK_CONFIG", "development")
app = create_app(config_name)

# 5) Réduire le bruit des logs werkzeug
logging.getLogger("werkzeug").setLevel(
    getattr(logging, os.getenv("WERKZEUG_LOG_LEVEL", "ERROR").upper(), logging.ERROR)
)

# 6) Lancement
if __name__ == "__main__":


    socketio.run(
        app,
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_RUN_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        use_reloader=False,  # ⚠️ évite le double-lancement avec Eventlet
    )
