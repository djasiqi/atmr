#!/usr/bin/env python3
"""ChatOps killswitch pour activation rapide du mode maintenance.

Usage:
    python -m chatops.killswitch enable --reason "OSRM down"
    python -m chatops.killswitch disable
    python -m chatops.killswitch status
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fichier de flag pour le mode maintenance
# Utiliser variable d'environnement pour le chemin (défaut: /tmp/)
_MAINTENANCE_TMP_DIR = os.getenv("MAINTENANCE_TMP_DIR", "/tmp")
MAINTENANCE_FLAG_FILE = Path(_MAINTENANCE_TMP_DIR) / "atmr_maintenance_mode.flag"
MAINTENANCE_LOG_FILE = Path(_MAINTENANCE_TMP_DIR) / "atmr_maintenance.log"


def enable_maintenance(reason: str = "Maintenance mode activated"):
    """Active le mode maintenance."""
    try:
        # Créer le flag file
        MAINTENANCE_FLAG_FILE.write_text(reason)

        # Logger l'activation
        from datetime import datetime

        with MAINTENANCE_LOG_FILE.open("a") as f:
            f.write(f"{datetime.now().isoformat()} - ENABLED - {reason}\n")

        # Définir la variable d'environnement (si dans Docker)
        # nosec B104: Script d'administration nécessitant
        # la modification de variables d'environnement
        os.environ["MAINTENANCE_MODE"] = "true"

        logger.warning("✅ Maintenance mode ENABLED: %s", reason)
        logger.info("📝 To apply changes, restart the API service:")
        logger.info("   docker-compose restart api")

        return True
    except Exception as e:
        logger.error("❌ Failed to enable maintenance mode: %s", e)
        return False


def disable_maintenance():
    """Désactive le mode maintenance."""
    try:
        # Supprimer le flag file
        if MAINTENANCE_FLAG_FILE.exists():
            MAINTENANCE_FLAG_FILE.unlink()

        # Logger la désactivation
        from datetime import datetime

        with MAINTENANCE_LOG_FILE.open("a") as f:
            f.write(f"{datetime.now().isoformat()} - DISABLED\n")

        # Supprimer la variable d'environnement
        if "MAINTENANCE_MODE" in os.environ:
            # nosec B104: Script d'administration nécessitant
            # la suppression de variables d'environnement
            del os.environ["MAINTENANCE_MODE"]

        logger.info("✅ Maintenance mode DISABLED")
        logger.info("📝 To apply changes, restart the API service:")
        logger.info("   docker-compose restart api")

        return True
    except Exception as e:
        logger.error("❌ Failed to disable maintenance mode: %s", e)
        return False


def get_status() -> dict[str, bool | str | None]:
    """Retourne le statut du mode maintenance."""
    is_active = MAINTENANCE_FLAG_FILE.exists()
    reason: str | None = None

    if is_active:
        try:
            reason = MAINTENANCE_FLAG_FILE.read_text().strip()
        except Exception:
            reason = "Unknown"

    return {"active": is_active, "reason": reason}


def main():
    parser = argparse.ArgumentParser(description="Manage maintenance mode killswitch")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Enable command
    enable_parser = subparsers.add_parser("enable", help="Enable maintenance mode")
    enable_parser.add_argument(
        "--reason",
        type=str,
        default="Maintenance mode activated",
        help="Reason for enabling maintenance mode",
    )

    # Disable command
    subparsers.add_parser("disable", help="Disable maintenance mode")

    # Status command
    subparsers.add_parser("status", help="Check maintenance mode status")

    args = parser.parse_args()

    if args.command == "enable":
        success = enable_maintenance(args.reason)
        sys.exit(0 if success else 1)
    elif args.command == "disable":
        success = disable_maintenance()
        sys.exit(0 if success else 1)
    elif args.command == "status":
        status = get_status()
        if status["active"]:
            print("🔴 Maintenance mode is ACTIVE")
            print(f"   Reason: {status['reason']}")
        else:
            print("🟢 Maintenance mode is INACTIVE")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
