"""Écriture robuste sous /app/uploads (volumes Docker / bind mounts)."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_writable_dir(directory: Path) -> None:
    """Crée le dossier et tente de le rendre inscriptible (best-effort)."""
    directory.mkdir(parents=True, exist_ok=True)
    if os.access(directory, os.W_OK):
        return
    try:
        # 0o777 : nécessaire sur certains bind mounts (Windows/NFS) où chown échoue.
        directory.chmod(0o777)
    except OSError as exc:
        logger.warning("[uploads] chmod impossible sur %s: %s", directory, exc)


def write_upload_bytes(filepath: Path, data: bytes) -> None:
    """Écrit un fichier binaire sous uploads, avec une tentative de correction des droits.

    Raises:
        PermissionError: si l'écriture reste impossible après correction best-effort.
    """
    directory = filepath.parent
    ensure_writable_dir(directory)
    try:
        with filepath.open("wb") as handle:
            handle.write(data)
        return
    except PermissionError:
        logger.error(
            "[uploads] Permission denied: path=%s dir_exists=%s dir_writable=%s uid=%s",
            filepath,
            directory.is_dir(),
            os.access(directory, os.W_OK),
            os.geteuid() if hasattr(os, "geteuid") else "n/a",
        )
        # Seconde tentative après chmod agressif (bind mounts Windows/NFS).
        with contextlib.suppress(OSError):
            directory.chmod(0o777)
        try:
            with filepath.open("wb") as handle:
                handle.write(data)
            return
        except PermissionError as perm_err:
            raise PermissionError(
                "Impossible d'écrire sous "
                f"{directory} (Permission denied). "
                "Vérifiez les droits du volume /app/uploads "
                "(chown appuser + chmod a+rwX)."
            ) from perm_err
