"""Écriture robuste sous /app/uploads (volumes Docker / bind mounts)."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from shared.logging_utils import exception_type_for_log
from shared.upload_path_resolver import confine_upload_destination

logger = logging.getLogger(__name__)


def ensure_writable_dir(directory: Path, *, uploads_base: Path | None = None) -> None:
    """Crée le dossier (confiné sous uploads) et tente de le rendre inscriptible."""
    directory = confine_upload_destination(directory, uploads_base=uploads_base)
    directory.mkdir(parents=True, exist_ok=True)
    if os.access(directory, os.W_OK):
        return
    try:
        # 0o777 : nécessaire sur certains bind mounts (Windows/NFS) où chown échoue.
        directory.chmod(0o777)
    except OSError as exc:
        logger.warning(
            "[uploads] chmod impossible error_type=%s",
            exception_type_for_log(exc),
        )


def write_upload_bytes(
    filepath: Path, data: bytes, *, uploads_base: Path | None = None
) -> None:
    """Écrit un fichier binaire sous uploads, après confinement de chemin.

    Raises:
        InvalidUploadPath: si le chemin sort de la racine uploads.
        PermissionError: si l'écriture reste impossible après correction best-effort.
    """
    filepath = confine_upload_destination(filepath, uploads_base=uploads_base)
    directory = filepath.parent
    ensure_writable_dir(directory, uploads_base=uploads_base)
    try:
        with filepath.open("wb") as handle:
            handle.write(data)
        return
    except PermissionError:
        logger.error(
            "[uploads] Permission denied error_type=PermissionError dir_writable=%s",
            os.access(directory, os.W_OK),
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
                "Impossible d'écrire sous le répertoire uploads "
                "(Permission denied). "
                "Vérifiez les droits du volume /app/uploads "
                "(chown appuser + chmod a+rwX)."
            ) from perm_err
